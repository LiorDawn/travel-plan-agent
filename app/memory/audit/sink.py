"""F4 AuditSink -- 审计事件落库（旁路式，失败不阻塞主流程）"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.models import AuditLog
from datetime import datetime


async def flush_audit(db: AsyncSession, request_id: str = "") -> None:
    """取当前请求的审计缓冲并批量写库。所有异常仅记日志，绝不抛给调用链。

    请求级：只 drain 本请求 recorder 的事件（见 recorder.end_request），
    并发 SSE 互相隔离；request_id 由 recorder 内固定，不再用 conv_id 兜底串号。

    事务粒度：一批一次 commit；失败时回滚该批，保证单批原子性。
    """
    from app.memory.audit.recorder import end_request
    events = end_request()  # 取出本请求缓冲并解除绑定
    if not events:
        return
    try:
        rows = [
            AuditLog(
                request_id=ev.get("request_id") or request_id,
                ts=datetime.fromtimestamp(ev["ts"]) if ev.get("ts") else datetime.utcnow(),
                tenant_id=ev.get("tenant_id", "default"),
                user_id=ev.get("user_id", ""),
                conv_id=ev.get("conv_id", ""),
                stage=ev.get("stage", ""),
                action=ev.get("action", ""),
                actor=ev.get("actor", ""),
                input_summary=_to_str(ev.get("input_summary")),
                output_summary=_to_str(ev.get("output_summary")),
                status=ev.get("status", "ok"),
                cost_ms=ev.get("cost_ms"),
                detail=ev.get("detail") or {},
            )
            for ev in events
        ]
        db.add_all(rows)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning("audit_flush_failed", count=len(events), error=str(e))


def _to_str(value) -> str | None:
    if value is None:
        return None
    import json
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)[:2000]
        except (TypeError, ValueError):
            return str(value)
    return str(value)[:2000]