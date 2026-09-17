import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

# 必须先于任何 app.api / app.agent import：补齐 langgraph.runtime 缺的注解符号，
# 否则后续 import 子图（create_agent）时因 langgraph 1.0.10 + langchain 1.x 不匹配而 ImportError。
import app.core.langgraph_compat  # noqa: F401
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.database import init_db
from app.core.http_middleware import RequestIDMiddleware, TimingLogMiddleware, UnifiedExceptionMiddleware
from app.api.v1 import chat, health, travel_plan, conversation, audits

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    try:
        await init_db()
    except Exception as e:
        print(f"[WARN] DB init failed: {e}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    # P1 修复: 挂载 debug 到 app.state，供异常中间件 detail 泄漏开关与监控面板读取
    app.state.debug = settings.debug
    # CORS：未配置白名单时允许任意来源但禁用 credentials（修复"通配+credentials"危险组合），配置白名单后按精确来源 + credentials
    origins = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=bool(origins),  # 仅精确来源时才允许携带凭据
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 中间件(后添加的靠内层执行, 故顺序: 异常->计时->RequestID)
    app.add_middleware(UnifiedExceptionMiddleware)
    app.add_middleware(TimingLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(travel_plan.router, prefix="/api/v1", tags=["travel_plan"])
    app.include_router(conversation.router, prefix="/api/v1", tags=["conversation"])
    app.include_router(audits.router, prefix="/api/v1", tags=["audits"])

    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
