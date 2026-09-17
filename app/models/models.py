from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


def gen_uuid():
    """生成字符串形式 UUID 主键。"""
    return str(uuid.uuid4())


class User(Base):
    """用户表 — 含偏好画像(常去城市/预算/风格)"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 画像
    preferred_cities = Column(ARRAY(String), default=list)
    budget_range = Column(JSON, default=dict)
    travel_style = Column(String(50), default="balanced")

    conversations = relationship("Conversation", back_populates="user")
    travel_plans = relationship("TravelPlan", back_populates="user")


class Conversation(Base):
    """会话表 — 一段对话，关联用户/消息/方案"""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(200), default="新对话")
    status = Column(String(20), default="active")
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")
    travel_plans = relationship("TravelPlan", back_populates="conversation")


class Message(Base):
    """消息表 — 会话内的一条 user/assistant 消息"""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON)
    metadata_ = Column("metadata", JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class TravelPlan(Base):
    """旅游方案表 — 主体信息 + 完整 plan_data(JSON)"""
    __tablename__ = "travel_plans"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversations.id"))
    title = Column(String(200))
    destination = Column(String(100))
    start_date = Column(String(20))
    end_date = Column(String(20))
    budget = Column(Float, default=0)
    plan_data = Column(JSON, default=dict)
    status = Column(String(20), default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="travel_plans")
    conversation = relationship("Conversation", back_populates="travel_plans")
    files = relationship("PlanFile", back_populates="travel_plan", order_by="PlanFile.version.desc()")


class PlanFile(Base):
    """方案文件版本表 — 一个方案可有多个 markdown 版本"""
    __tablename__ = "plan_files"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    plan_id = Column(String(36), ForeignKey("travel_plans.id"), nullable=False)
    version = Column(Integer, default=1)
    content = Column(Text)
    content_hash = Column(String(64))
    change_summary = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    travel_plan = relationship("TravelPlan", back_populates="files")


class AuditLog(Base):
    """F4 审计表 — 全链路关键步骤留痕（模型/工具/任务/权限/审批/中断）"""
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    request_id = Column(String(64), index=True, nullable=False)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    tenant_id = Column(String(64), default="default")
    user_id = Column(String(64))
    conv_id = Column(String(64))
    stage = Column(String(50))
    action = Column(String(50), nullable=False)
    actor = Column(String(100))
    input_summary = Column(Text)
    output_summary = Column(Text)
    status = Column(String(20), default="ok")
    cost_ms = Column(Integer)
    detail = Column(JSON, default=dict)
