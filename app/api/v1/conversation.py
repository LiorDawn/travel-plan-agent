from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.conversation_service import ConversationService

router = APIRouter()


@router.get("/conversations")
async def list_conversations(user_id: str = "demo", db: AsyncSession = Depends(get_db)):
    """获取会话列表"""
    return await ConversationService.list_conversations(user_id, db)


@router.post("/conversations")
async def create_conversation(user_id: str = "demo", title: str = "新对话", db: AsyncSession = Depends(get_db)):
    """创建会话"""
    return await ConversationService.create_conversation(user_id, title, db)


@router.put("/conversations/{conv_id}")
async def update_conversation(conv_id: str, title: str = "", db: AsyncSession = Depends(get_db)):
    """更新会话标题"""
    return await ConversationService.update_conversation(conv_id, title, db)


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, db: AsyncSession = Depends(get_db)):
    """删除会话（级联删除消息、方案、文件）"""
    return await ConversationService.delete_conversation(conv_id, db)


@router.delete("/conversations")
async def clear_all_conversations(user_id: str = "demo", db: AsyncSession = Depends(get_db)):
    """一键清除所有会话（级联）"""
    return await ConversationService.clear_all(user_id, db)


@router.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, db: AsyncSession = Depends(get_db)):
    """获取会话消息"""
    return await ConversationService.get_messages(conv_id, db)