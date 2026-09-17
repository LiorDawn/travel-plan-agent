"""RAG 过程缓存 — 改写/embedding 双缓存（Redis 优先 + 进程内存兜底）

针对"子图每次模型调用前都检索一次"的高频路径：LLM 改写 + embedding 网络往返是最重开销。
这里用两级缓存把第二~N 次检索降到"内存命中"（近乎零成本）：
  1. 读：Redis 命中直接返回；未命中回退内存缓存；二者皆无则 miss（走真实计算）。
  2. 写：先写内存（保底），再尝试写 Redis；Redis 失败仅告警，不阻塞业务（降级语义）。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from app.core.logging import logger


class LocalLruCache:
    """线程安全、带 TTL 的有界内存缓存（简单 LRU 淘汰，够用不过度设计）。"""

    def __init__(self, max_size: int = 2048) -> None:
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value = item
            if now - ts >= _EXPIRED_ATTR_TTL:  # 防御性兜底，真实 TTL 由外层判断
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._order) > self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)

    def within_ttl(self, key: str, ttl: float) -> bool:
        """带专属 TTL 的命中判定：存储时间距今 <= ttl 且键存在。"""
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return False
            return (time.time() - item[0]) <= ttl

    def get_if_fresh(self, key: str, ttl: float) -> Any | None:
        """命中且未过期才返回；过期即清除并返回 None。"""
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts >= ttl:
                self._store.pop(key, None)
                if key in self._order:
                    self._order.remove(key)
                return None
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()


# 防御性兜底 TTL（秒）—— 真实 TTL 由调用方通过 get_if_fresh(ttl) 传入
_EXPIRED_ATTR_TTL: float = 86400.0

_local_cache = LocalLruCache()


async def _redis_get(key: str) -> Any | None:
    try:
        from app.core.redis import redis_pool
        return await redis_pool.get(key)
    except Exception:
        return None


async def _redis_set(key: str, value: str, ttl: int) -> None:
    try:
        from app.core.redis import redis_pool
        await redis_pool.set(key, value, ex=ttl)
    except Exception as e:
        logger.warning("rag_proc_cache_redis_failed", error=str(e))


def _deserialize(raw: Any, loader: Callable[[str], Any]) -> Any | None:
    """把 Redis 字符串还原为对象；失败返回 None（视作 miss，不抛）。"""
    if raw is None:
        return None
    try:
        return loader(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning("rag_proc_cache_deserialize_failed", error=str(e))
        return None


def _serialize(value: Any, dumper: Callable[[Any], str]) -> str | None:
    try:
        return dumper(value)
    except Exception:
        return None


async def cached_get(key: str, ttl: int,
                     dumper: Callable[[Any], str],
                     loader: Callable[[str], Any]) -> Any | None:
    """统一读：Redis → 内存 → miss。ttl 秒。dumper/loader 负责对象的序列化与反序列化。

    - Redis 命中直接返回（跨进程/重启共享）。
    - Redis miss/不可用 → 回退内存 get_if_fresh(ttl)。
    - 返回 None 表示 miss，调用方执行真实计算并走 cached_set。
    """
    # 1) 内存兜底：最便宜、最常命中（子图同一 query 重复检索）
    mem = _local_cache.get_if_fresh(key, ttl)
    if mem is not None:
        return mem
    # 2) Redis 优先
    raw = await _redis_get(key)
    if raw is not None:
        val = _deserialize(raw, loader)
        if val is not None:
            _local_cache.set(key, val)  # 回填内存，降温数量级
            return val
    return None


async def cached_set(key: str, value: Any, ttl: int,
                     dumper: Callable[[Any], str]) -> None:
    """统一写：先写内存（保底），再尝试写 Redis。任一失败都不抛（降级）。"""
    _local_cache.set(key, value)
    s = _serialize(value, dumper)
    if s is not None:
        await _redis_set(key, s, ttl)


import json


def _kf_json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _kf_json_loads(s: str) -> Any:
    return json.loads(s)


def _vec_dumps(v: Any) -> str:
    """embedding 向量 → json 字符串（每维度小 float，体积可控）。"""
    return json.dumps(v)


def _vec_loads(s: str) -> list[float]:
    return json.loads(s)


# ---- 面向调用方的语义封装（简化 retriever 调用点） ----
async def rewrite_cache_get(query: str, ttl: int) -> Any | None:
    """改写结果缓存读：返回 (rewritten_query, keywords) 或 None。"""
    return await cached_get(f"rag:kw:{query}", ttl, _kf_json_dumps, _kf_json_loads)


async def rewrite_cache_set(query: str, rewritten: str, keywords: list, ttl: int) -> None:
    await cached_set(f"rag:kw:{query}", {"query": rewritten, "keywords": keywords}, ttl, _kf_json_dumps)


async def embedding_cache_get(text: str, ttl: int) -> list[float] | None:
    return await cached_get(f"rag:emb:{text}", ttl, _vec_dumps, _vec_loads)


async def embedding_cache_set(text: str, vec: list[float], ttl: int) -> None:
    await cached_set(f"rag:emb:{text}", vec, ttl, _vec_dumps)