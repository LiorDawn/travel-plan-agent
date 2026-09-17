"""RAG 统一出口 — 收拢外层节点与子图中间件的检索/沉淀调用（对应重构文档 §6）。

职责：把散落在各节点的 `retrieve(source_filter=...)` / `insert_chunk(...)` 调用
统一封装为三个语义清晰的接口，消除两套检索逻辑漂移：
  - recall_conversation(query)   检索历史对话记忆（外层 rag_recall，plan 前）
  - recall_manual(query)         检索攻略知识（关思考 rag_manual 节点 / 开思考 RAGInject 中间件共用）
  - persist_conversation(content) 沉淀本次问答到知识库（外层 finalize）
"""
import difflib
from datetime import datetime

from sqlalchemy import desc, select

from app.core.config import get_settings
from app.core.logging import logger
from app.core.database import AsyncSessionLocal
from app.memory.rag.retriever import chunk_text, insert_chunk, retrieve

# 模块级单例：缓存开关/TTL 读取（此前缺失导致 settings.rag_cache_* 运行时 NameError）
settings = get_settings()

# 各来源注入预算（字符）
_CONVERSATION_BUDGET = 1500  # 历史记忆（外层）
_MANUAL_BUDGET = 2500        # 攻略知识（外层 rag_manual 与子图共用）

# 记忆单块上限：超过则切片入库，避免"一条长方案整块存、检索粒度粗、挤占注入预算"
_CONVERSATION_CHUNK_MAX = 1200  # 与攻略大块 _RAG_PARENT_MAX 对齐
_CONVERSATION_CHUNK_OVERLAP = 80

# 沉淀去重：与新内容最相近的"同 user 最近一条 conversation chunk"相似度超过该值则跳过（P1-4）
_DEDUPE_SIM_THRESHOLD = 0.9


async def _cache_get(key: str) -> str | None:
    """缓存命中取格式化文本；缓存未启用/Redis 不可用时返回 None（降级走检索）。"""
    try:
        from app.core.redis import redis_pool
        return await redis_pool.get(key)
    except Exception as e:
        return None


async def _cache_set(key: str, value: str) -> None:
    """写入检索缓存并设 TTL；失败静默降级（不影响业务）。"""
    try:
        from app.core.redis import redis_pool
        await redis_pool.set(key, value, ex=settings.rag_cache_ttl)
    except Exception as e:
        logger.warning("rag_cache_set_failed", error=str(e))


async def recall_conversation(query: str, user_id: str = "anonymous") -> str:
    """检索历史对话记忆，返回格式化文本（"" 表示无命中/失败）。命中缓存则省一次检索。

    多用户隔离：将 user_id 下发到 retriever，只召回当前用户自己的对话记忆。
    """
    key = f"rag:conv:{user_id}:{query.strip()}" if settings.rag_cache_enabled else None
    if key:
        cached = await _cache_get(key)
        if cached is not None:
            return cached
    try:
        hits = await retrieve(query, source_filter="conversation", user_id=user_id)
        text = _format_within_budget(hits, _CONVERSATION_BUDGET)
        if key and text:
            await _cache_set(key, text)
        return text
    except Exception as e:
        logger.warning("rag_recall_conversation_failed", error=str(e))
        return ""


async def recall_manual(query: str) -> str:
    """检索攻略知识，返回格式化文本（"" 表示无命中/失败）。子图中间件每次模型调用前复用。"""
    key = f"rag:manual:{query.strip()}" if settings.rag_cache_enabled else None
    if key:
        cached = await _cache_get(key)
        if cached is not None:
            return cached
    try:
        hits = await retrieve(query, source_filter="manual")
        text = _format_within_budget(hits, _MANUAL_BUDGET)
        if key and text:
            await _cache_set(key, text)
        return text
    except Exception as e:
        logger.warning("rag_recall_manual_failed", error=str(e))
        return ""


async def persist_conversation(question: str, answer: str, source_id: str = None,
                               user_id: str = "anonymous") -> bool:
    """把一问一答沉淀为历史记忆（source_type=conversation），多用户隔离按 user_id 落库。

    P1-4 去重：先取该 user 最近一条 conversation chunk，与本次问答做相似度判定，
    高度重合（超过阈值）则跳过，避免"同一话题多轮重复举手"把知识库撑爆、检索噪声累积。
    """
    try:
        chunk = f"用户问：{question}\n回复：{answer}"
        # 会话级合并：仅与该会话(source_id)内最近一条 conversation chunk 做相似度判定，
        # 同一会话内连续多轮重复/近似的沉淀只留首条，降低库冗余与检索噪声。
        if await _is_duplicate(chunk, user_id, source_id):
            logger.info("rag_persist_skipped_duplicate", user_id=user_id, conv=source_id)
            return True  # 视为沉淀成功（已存在等价记忆），不重复入库

        pieces = chunk_text(chunk, chunk_size=_CONVERSATION_CHUNK_MAX,
                            overlap=_CONVERSATION_CHUNK_OVERLAP) or [chunk]
        ok = 0
        for piece in pieces:
            written = await insert_chunk(piece, source_type="conversation",
                                         source_id=source_id, metadata={"user_id": user_id})
            ok += 1 if written else 0
        return ok > 0
    except Exception as e:
        logger.warning("rag_persist_conversation_failed", error=str(e))
        return False


async def _is_duplicate(chunk: str, user_id: str, source_id: str = None) -> bool:
    """与该会话(source_id)最近一条 conversation chunk 比较相似度；无法判定一律返回 False（不误拦）。

    会话级合并：限定在同一会话内比对（source_id 匹配），避免跨会话的同 user 最近块被误判为重复；
    source_id 缺失时退化为"同 user 最近一条"。
    """
    try:
        from app.memory.rag.models import KnowledgeChunk
        async with AsyncSessionLocal() as session:
            stmt = (
                select(KnowledgeChunk)
                .where(KnowledgeChunk.source_type == "conversation",
                       KnowledgeChunk.metadata_["user_id"].astext == user_id)
            )
            if source_id:
                stmt = stmt.where(KnowledgeChunk.source_id == source_id)
            stmt = stmt.order_by(desc(KnowledgeChunk.created_at)).limit(1)
            res = await session.execute(stmt)
            latest = res.scalars().first()
        if latest is None or not latest.content:
            return False
        sim = difflib.SequenceMatcher(None, chunk, latest.content).ratio()
        return sim >= _DEDUPE_SIM_THRESHOLD
    except Exception:
        return False


def _stamp(created_at) -> str:
    """把沉淀时间格式化为 'YYYY-MM-DD'；无/非法时间返回 ''。"""
    if not created_at:
        return ""
    try:
        if hasattr(created_at, "strftime"):
            return created_at.strftime("%Y-%m-%d")
        # 兼容字符串时间
        from datetime import datetime
        return datetime.fromisoformat(str(created_at)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _format_within_budget(hits: list[dict], budget: int) -> str:
    """按预算注入：以【整块】为单位逐个累加，预算不足以再放下块时停止，不劈块不截断（对应
    次要问题"截断方式不一致"的修复——统一为整块累加，避免上游按字符硬切切断句子语义）。"""
    if not hits:
        return ""
    blocks: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        if not h.get("content"):
            continue
        stamp = _stamp(h.get("created_at"))
        prefix = f"[{i}]" if not stamp else f"[{i}]（{stamp}）"
        block = f"{prefix} {h['content']}"
        if blocks and used + len(block) + 2 > budget:  # +2 计 \n\n 分隔
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)