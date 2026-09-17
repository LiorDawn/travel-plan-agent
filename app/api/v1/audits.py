"""F4 审计查询端点 — 全链路留痕可回溯（B端排障/合规）"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_api_token
from app.models.models import AuditLog

router = APIRouter()


@router.get("/audits", dependencies=[Depends(require_api_token)])
async def list_audits(
    request_id: str = Query("", description="按请求串联过滤"),
    conv_id: str = Query("", description="按会话过滤"),
    action: str = Query("", description="动作类型: model_call/tool_call/node_end/..."),
    status: str = Query("", description="状态: ok/error/denied/pending_approval"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """审计日志查询（F4）。默认倒序取最近。"""
    stmt = select(AuditLog).order_by(desc(AuditLog.ts)).limit(limit)
    if request_id:
        stmt = stmt.where(AuditLog.request_id == request_id)
    if conv_id:
        stmt = stmt.where(AuditLog.conv_id == conv_id)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if status:
        stmt = stmt.where(AuditLog.status == status)

    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "id": r.id, "request_id": r.request_id, "ts": r.ts.isoformat() if r.ts else None,
            "tenant_id": r.tenant_id, "user_id": r.user_id, "conv_id": r.conv_id,
            "stage": r.stage, "action": r.action, "actor": r.actor,
            "input_summary": r.input_summary, "output_summary": r.output_summary,
            "status": r.status, "cost_ms": r.cost_ms, "detail": r.detail,
        }
        for r in rows
    ]