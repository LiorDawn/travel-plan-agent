"""对话用例编排 — 封装会话 CRUD，供 API 层薄壳调用"""
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Conversation
from app.memory import long_term
from app.core.errors import NotFoundError


class ConversationService:
    """会话管理用例"""

    @staticmethod
    async def list_conversations(user_id: str, db: AsyncSession) -> List[dict]:
        result = await db.execute(
            select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
        )
        convs = result.scalars().all()
        return [{
            "id": c.id, "title": c.title, "status": c.status,
            "created_at": str(c.created_at), "updated_at": str(c.updated_at),
        } for c in convs]

    @staticmethod
    async def create_conversation(user_id: str, title: str, db: AsyncSession) -> dict:
        conv = Conversation(user_id=user_id, title=title)
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return {"id": conv.id, "title": conv.title, "created_at": str(conv.created_at)}

    @staticmethod
    async def update_conversation(conv_id: str, title: str, db: AsyncSession) -> dict:
        conv = await db.get(Conversation, conv_id)
        if not conv:
            raise NotFoundError("会话不存在")
        if title:
            conv.title = title
        await db.commit()
        return {"id": conv_id, "title": conv.title}

    @staticmethod
    async def delete_conversation(conv_id: str, db: AsyncSession) -> dict:
        ok = await long_term.delete_conversation_cascade(db, conv_id)
        if not ok:
            raise NotFoundError("会话不存在")
        return {"deleted": conv_id}

    @staticmethod
    async def clear_all(user_id: str, db: AsyncSession) -> dict:
        convs = (await db.execute(
            select(Conversation).where(Conversation.user_id == user_id)
        )).scalars().all()
        for c in convs:
            await long_term.delete_conversation_cascade(db, c.id)
        return {"cleared": True, "user_id": user_id}

    @staticmethod
    async def get_messages(conv_id: str, db: AsyncSession) -> List[dict]:
        msgs = await long_term.get_messages(db, conv_id)
        return [{"role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in msgs]