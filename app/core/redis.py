import redis.asyncio as aioredis
from app.core.config import get_settings

settings = get_settings()

redis_pool = aioredis.from_url(
    settings.redis_url,
    encoding="utf-8",
    decode_responses=True,
)


async def get_redis() -> aioredis.Redis:
    """FastAPI 依赖: 直接返回全局 Redis 连接池, 供短期缓存等使用。"""
    return redis_pool


async def check_redis():
    """检查 Redis 连接"""
    try:
        await redis_pool.ping()
        return True
    except Exception:
        return False
