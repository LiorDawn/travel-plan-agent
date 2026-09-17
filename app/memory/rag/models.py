"""RAG 域内 ORM — KnowledgeChunk（知识块/向量；对应重构文档 §3.2）

从 app/models/models.py 移入 memory/rag 域内聚，避免通用模型层混入 pgvector 依赖。
仍挂在 app.core.database.Base 上，init_db 的 create_all 仍会建表。
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, String, Text

from app.core.database import Base


def _gen_uuid() -> str:
    """生成字符串形式 UUID 主键。"""
    return str(uuid.uuid4())


class KnowledgeChunk(Base):
    """RAG 知识块表 — 历史对话/知识与向量化内容，pgvector 语义检索"""

    __tablename__ = "knowledge_chunks"

    id = Column(String(36), primary_key=True, default=_gen_uuid)
    # source: conversation | manual, 关联来源
    source_type = Column(String(20), nullable=False, default="manual")
    source_id = Column(String(36))
    # 父子块: 小块(parent_id=其所属大块id)负责精准召回; 大块(parent_id=NULL)作为上下文喂LLM
    parent_id = Column(String(36), index=True)
    content = Column(Text, nullable=False)
    # pgvector 向量（维度与 embedding 模型一致, text-embedding-v3 默认 1024）
    embedding = Column(Vector(1024))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)