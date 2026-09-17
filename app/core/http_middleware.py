"""
HTTP 中间件 — 统一异常处理 + Request-ID + 请求计时与性能日志

注意:
- `/api/v1/chat` 是 SSE 流式端点, 异常处理中间件会拦截 body 流, 因此统一异常
  JSON 处理对该路由跳过, 由 EventSourceResponse 自行管理流输出。
"""
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import Message
from app.core.logging import logger
from app.core.errors import AppError
import structlog

# SSE 流式路由: 不套用统一异常 JSON 响应(异常由流内部处理)
_EXCLUDE_EXCEPTION_PATHS = {"/api/v1/chat"}
# 静态资源路径前缀: 请求体可忽略再检查(可能较大)
_STATIC_PREFIX = "/static"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求生成唯一 request_id, 写入 structlog 上下文并回填到响应头"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        # 绑定到 structlog contextvar, 使该请求内所有 logger 输出带上 request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        # P0-2: 同步到审计的 request_id contextvar，节点埋点能读到当前请求 ID，链路可回溯
        from app.memory.audit.recorder import set_request_id
        set_request_id(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response


class TimingLogMiddleware(BaseHTTPMiddleware):
    """记录每个请求的耗时与状态码, 便于定位慢接口(尤其 /chat)"""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "http_request",
            method=method,
            path=path,
            status=response.status_code,
            cost_ms=round(elapsed_ms, 1),
        )
        return response


class UnifiedExceptionMiddleware(BaseHTTPMiddleware):
    """统一异常处理 — 把未捕获异常转成统一 JSON(含 request_id)。

    跳过 SSE 流式路由, 避免拦截中断流式输出。
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXCLUDE_EXCEPTION_PATHS:
            return await call_next(request)

        try:
            response = await call_next(request)
        except AppError as e:
            logger.warning("app_error", path=request.url.path, code=e.code, error=str(e))
            request_id = request.headers.get("X-Request-ID", "")
            return JSONResponse(
                status_code=e.code,
                content={"code": e.code, "message": e.message, "request_id": request_id},
            )
        except Exception as e:
            logger.exception("unhandled_exception", path=request.url.path, method=request.method, error=str(e))
            request_id = request.headers.get("X-Request-ID", "")
            # P1 修复: app.state.debug 可能未设置(main 未挂载), getattr 兜底防异常处理器自身崩溃
            show_detail = getattr(request.app.state, "debug", False)
            return JSONResponse(
                status_code=500,
                content={
                    "code": 500,
                    "message": "服务器内部错误",
                    "request_id": request_id,
                    "detail": str(e) if show_detail else None,
                },
            )
        return response


async def _body_reader(_: Request, body_type: str) -> None:
    """占位, 便于扩展 body 大小限制等"""
    pass