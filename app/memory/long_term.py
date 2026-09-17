"""PostgreSQL 长期记忆 — 对话/方案/画像持久化"""
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import User, Conversation, Message, TravelPlan, PlanFile


async def create_conversation(db: AsyncSession, user_id: str, title: str = "新对话") -> Conversation:
    """新建一段会话记录。"""
    conv = Conversation(user_id=user_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def save_message(db: AsyncSession, conv_id: str, role: str, content: str, tool_calls: dict = None) -> Message:
    """保存一条对话消息(user/assistant)，可附带工具调用元数据。"""
    msg = Message(conversation_id=conv_id, role=role, content=content, tool_calls=tool_calls)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def save_messages(db: AsyncSession, conv_id: str, items: list[tuple[str, str]]) -> list[Message]:
    """批量保存消息（同一事务一次 commit）：items = [(role, content), ...]。

    替代逐条 save_message 的多次 commit，降低短对话多轮频繁落库的事务开销。
    """
    rows = [Message(conversation_id=conv_id, role=role, content=content) for role, content in items]
    db.add_all(rows)
    await db.commit()
    for m in rows:
        await db.refresh(m)
    return rows


async def get_conversation_messages(db: AsyncSession, conv_id: str, limit: int = 20) -> list[Message]:
    """按时间逆序取最近 limit 条会话消息(用于上下窗口)。"""
    result = await db.execute(
        select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())[::-1]


async def save_travel_plan(db: AsyncSession, plan_data: dict) -> TravelPlan:
    """保存方案主体，预算取 total。"""
    plan = TravelPlan(
        user_id=plan_data.get("user_id", ""),
        conversation_id=plan_data.get("conversation_id", ""),
        title=plan_data.get("title", "旅游方案"),
        destination=plan_data.get("destination", ""),
        start_date=plan_data.get("start_date"),
        end_date=plan_data.get("end_date"),
        budget=plan_data.get("budget", {}).get("total", 0),
        plan_data=plan_data,
        status="draft",
    )
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    return plan


async def save_plan_file(db: AsyncSession, plan_id: str, content: str, version: int = 1) -> PlanFile:
    """保存方案文件一个版本(markdown 内容 + md5 摘要)。"""
    import hashlib
    pf = PlanFile(
        plan_id=plan_id,
        version=version,
        content=content,
        content_hash=hashlib.md5(content.encode()).hexdigest(),
        change_summary=f"v{version} - 方案生成",
    )
    db.add(pf)
    await db.commit()
    await db.refresh(pf)
    return pf


async def get_user_plans(db: AsyncSession, user_id: str) -> list[TravelPlan]:
    """取用户全部方案(按创建时间倒序)。"""
    result = await db.execute(
        select(TravelPlan).where(TravelPlan.user_id == user_id).order_by(TravelPlan.created_at.desc())
    )
    return list(result.scalars().all())


async def get_plan_files(db: AsyncSession, plan_id: str) -> list[PlanFile]:
    """取方案的文件版本列表(从新到旧)。"""
    result = await db.execute(
        select(PlanFile).where(PlanFile.plan_id == plan_id).order_by(PlanFile.version.desc())
    )
    return list(result.scalars().all())


async def delete_plan(db: AsyncSession, plan_id: str) -> bool:
    """删除方案及其文件"""
    await db.execute(delete(PlanFile).where(PlanFile.plan_id == plan_id))
    result = await db.execute(delete(TravelPlan).where(TravelPlan.id == plan_id))
    await db.commit()
    return result.rowcount > 0


async def delete_plan_file(db: AsyncSession, plan_id: str, version: int = None) -> bool:
    """删除方案文件（指定版本或全部）"""
    stmt = delete(PlanFile).where(PlanFile.plan_id == plan_id)
    if version is not None:
        stmt = stmt.where(PlanFile.version == version)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


async def get_or_create_user(db: AsyncSession, user_id: str, username: str = None) -> User:
    """获取或创建用户"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        user = User(id=user_id, username=username or user_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def save_conversation(db: AsyncSession, conv_id: str, user_id: str, title: str = "新对话") -> Conversation:
    """保存对话（幂等：已存在则跳过，不重建）"""
    conv = await db.get(Conversation, conv_id)
    if conv:
        return conv
    conv = Conversation(id=conv_id, user_id=user_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_messages(db: AsyncSession, conv_id: str) -> list[Message]:
    """获取对话消息"""
    result = await db.execute(
        select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def delete_conversation_cascade(db: AsyncSession, conv_id: str) -> bool:
    """删除会话及其消息、方案、文件（级联）"""
    conv = await db.get(Conversation, conv_id)
    if not conv:
        return False
    plans = (await db.execute(
        select(TravelPlan).where(TravelPlan.conversation_id == conv_id)
    )).scalars().all()
    for p in plans:
        await db.execute(delete(PlanFile).where(PlanFile.plan_id == p.id))
    await db.execute(delete(TravelPlan).where(TravelPlan.conversation_id == conv_id))
    await db.execute(delete(Message).where(Message.conversation_id == conv_id))
    await db.delete(conv)
    await db.commit()
    return True
