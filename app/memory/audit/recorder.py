"""F4 审计与可观测 — AuditRecorder + 节点埋点装饰器

外层是手写 StateGraph，无 create_agent 的 middleware 机制，因此用
「节点埋点」实现，两条路径：
- @audit_wrap(action, stage) 装饰器：包在 add_node 的节点函数外层，首尾各记一次；
- audit_rec.record(...) 显式调用：用于流程内细分动作（工具/权限/审批/中断）。

旁路式：写入失败仅 logger.warning，绝不抛错阻塞主图。
"""
import contextvars
import time
from typing import Any, Callable, Awaitable

from app.core.config import get_settings
from app.core.logging import logger

# 敏感字段白名单 → 脱敏替换
_SENSITIVE_KEYS = ("password", "token", "secret", "key", "credential", "card", "id_card", "mobile")


def _mask(value: Any, depth: int = 0) -> Any:
    """浅层脱敏：dict/list 递归脱敏键值，scalar 截断。"""
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in _SENSITIVE_KEYS else _mask(v, depth + 1))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, depth + 1) for v in value][:20]
    if depth > 2:
        return "…"
    s = str(value)
    return s[:500] + ("…" if len(s) > 500 else "")


class _AuditRecorder:
    """收集埋点事件；采样率控制，权限/审批必审计不采样。

    每个请求一个实例（contextvar 持有），天然按请求隔离 —— 并发 SSE 不串事件。
    """

    def __init__(self, request_id: str = ""):
        self._events: list[dict] = []
        self.request_id = request_id

    def record(self, *, stage: str, action: str, actor: str = "node",
               status: str = "ok", input_summary: Any = None,
               output_summary: Any = None, cost_ms: float | None = None,
               forced: bool = False, **detail: Any) -> None:
        """记录一条审计事件。forced=True 的（权限/审批）跳过采样率。"""
        s = get_settings()
        if not forced and s.audit_sampling < 1.0 and _skip_by_sample(s.audit_sampling):
            return
        self._events.append({
            # 请求级：优先用本 recorder 固定的 request_id（进业务时设置，见 begin_request）
            "request_id": self.request_id or _request_id_var.get() or "",
            "ts": time.time(),
            "tenant_id": detail.pop("tenant_id", "default"),
            "user_id": detail.pop("user_id", ""),
            "conv_id": detail.pop("conv_id", ""),
            "stage": stage,
            "action": action,
            "actor": actor,
            "input_summary": _mask(input_summary),
            "output_summary": _mask(output_summary),
            "status": status,
            "cost_ms": int(cost_ms) if cost_ms is not None else None,
            "detail": detail,
        })

    def drain(self) -> list[dict]:
        """取出并清空当前缓冲（供批量写库）。"""
        buf = self._events
        self._events = []
        return buf


# ---- 请求级上下文（contextvar，与 llm._usage_var 同款，天然按请求隔离） ----
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("audit_request_id", default="")
# 请求级 recorder：stream 开头 begin_request 创建并绑定；不足时 lazy 回退到进程级单例
_recorder_var: contextvars.ContextVar[_AuditRecorder] = contextvars.ContextVar("audit_recorder", default=None)


class _CtxRequestID:
    """request_id 上下文（请求开始时由 RequestIDMiddleware / begin_request 设置，节点内可读）"""

    def set(self, value: str):
        _request_id_var.set(value or "")

    def get(self) -> str:
        return _request_id_var.get()

    def __repr__(self) -> str:
        return _request_id_var.get()


# 兼容旧引用：CtxRequestID.set/.get 现在走 contextvar
_CtxRequestID_holder = _CtxRequestID()
CtxRequestID = _CtxRequestID_holder


def set_request_id(request_id: str) -> None:
    """请求进入业务前设置（供节点内 record 读取）。"""
    _request_id_var.set(request_id or "")


def current_recorder() -> _AuditRecorder:
    """返回本请求绑定的 recorder；未绑定（如单请求退化/测试直调）则 lazy 建一个进程级。"""
    rec = _recorder_var.get()
    if rec is None:
        rec = _AuditRecorder(request_id=_request_id_var.get())
        _recorder_var.set(rec)
    return rec


def begin_request(request_id: str) -> _AuditRecorder:
    """请求开始：绑定请求级 recorder + request_id（并发隔离的核心）。"""
    set_request_id(request_id)
    rec = _AuditRecorder(request_id=request_id)
    _recorder_var.set(rec)
    return rec


def end_request() -> list[dict]:
    """请求结束：取出本请求缓冲，并解除绑定（防止串到下一请求）。"""
    rec = _recorder_var.get()
    _recorder_var.set(None)
    return rec.drain() if rec is not None else []


def _skip_by_sample(rate: float) -> bool:
    import random
    return random.random() >= rate


# ---------- 节点装饰器 ----------
def audit_wrap(action: str = "node_end", stage: str = ""):
    """包在 add_node 的节点函数外，记录进入/结束两个埋点。"""
    def deco(fn: Callable[..., Awaitable[Any]]):
        async def wrapped(state: dict, *a, **kw):
            ctx = _ctx_of(state)
            t0 = time.time()
            try:
                res = await fn(state, *a, **kw)
                cost = (time.time() - t0) * 1000
                _record(stage=stage or action, action=action, ctx=ctx,
                        status="ok", input_summary=None,
                        output_summary=_pick_summary(res), cost_ms=cost)
                return res
            except Exception as e:
                _record(stage=stage or action, action=action, ctx=ctx,
                        status="error", output_summary=str(e), cost_ms=(time.time() - t0) * 1000)
                raise
        return wrapped
    return deco


def _ctx_of(state: dict) -> dict:
    return {
        "user_id": state.get("user_id", ""),
        "conv_id": state.get("conversation_id", ""),
    }


def _pick_summary(res: Any) -> Any:
    if isinstance(res, dict):
        for k in ("answer", "clarifications", "tool_results", "error"):
            if k in res and res[k]:
                return res[k]
        return {}
    return res


def _record(*, stage, action, ctx, status, input_summary, output_summary, cost_ms):
    current_recorder().record(stage=stage, action=action, actor="node", status=status,
                              input_summary=input_summary, output_summary=output_summary,
                              cost_ms=cost_ms, forced=True, **ctx)  # 节点级 default 强制留痕