"""外层 checkpointer 构建 — Redis 持久化，失败回退内存（对应重构文档 §3.5）

外层 graph 的 thread_id = conversation_id；子图独立 thread_id = 调用方拼接 "_planner"。
两套断点点的存储隔离由各自 thread_id 保证，本模块只负责产出外层 saver。
"""
from app.core.logging import logger


def _build_checkpointer():
    """外层 checkpointer：优先 Redis，不可用回退 MemorySaver。"""
    from app.core.config import get_settings
    from langgraph.checkpoint.memory import MemorySaver

    settings = get_settings()
    if getattr(settings, "checkpointer", "memory").lower() == "redis":
        try:
            from langgraph.checkpoint.redis.aio import RedisSaver
            logger.warning("checkpointer_redis_enabled", conn=settings.redis_host)
            return RedisSaver.from_conn_string(settings.redis_url)
        except Exception as exc:
            logger.warning("checkpointer_redis_unavailable_fallback_memory", error=str(exc))
    logger.info("checkpointer_memory", saver="MemorySaver")
    return MemorySaver()