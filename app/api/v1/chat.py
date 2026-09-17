from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService
from app.core.database import get_db
from app.core.security import require_api_token

router = APIRouter()
chat_service = ChatService()


@router.get("/monitor", dependencies=[Depends(require_api_token)])
async def get_monitor():
    """token 使用统计（需管理令牌）"""
    return await chat_service.monitor()


@router.post("/chat")
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """统一对话入口 — 自动判断新对话 vs 续跑，服务层产出 SSE 事件流"""

    async def event_stream():
        try:
            async for event in chat_service.stream(request, db):
                yield {"event": event.event_type, "data": json.dumps(event.data, ensure_ascii=False)}
        except Exception as e:  # P1-10/N2：流中途抛错时兜底 error 事件 + 保证审计落库
            yield {
                "event": "error",
                "data": json.dumps({"message": "处理过程中发生错误", "detail": str(e), "code": 500}, ensure_ascii=False),
            }
        finally:
            # N2：无论正常结束 / 异常 / SSE 断连，都冲刷本请求已埋点的审计事件，失败路径不丢留痕
            try:
                from app.memory.audit.sink import flush_audit
                await flush_audit(db, request_id=request.conversation_id or "")
            except Exception:
                pass  # 审计是旁路，绝不阻塞流

    return EventSourceResponse(event_stream())