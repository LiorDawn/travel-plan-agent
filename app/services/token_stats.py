"""token 使用统计落 Redis（HINCRBY）—— 从进程内存改为可跨 worker / 重启持久化

原实现 `ChatService` 的 token_stats/agg_by_intent/agg_by_conversation 全在进程内存：
重启清零、多 worker 不一致。这里用 Redis HINCRBY 按键（意图/会话）累加输入输出 token，
key 带 TTL 自动过期；Redis 不可用（本地演示）时降级回进程内 dict 兜底，行为不退化。

监控端点通过 `snapshot()` 合并 Redis 与本地兜底，返回与旧 monitor() 同构的统计。
"""
from __future__ import annotations

import threading
import time

from app.core.config import get_settings
from app.core.logging import logger


class TokenStats:
    """token 统计：Redis HINCRBY 持久化 + 进程内存降级兜底。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local: dict[str, dict] = {}   # Redis 不可用时的兜底聚合
        self._calls_local = 0
        self.total_input = 0
        self.total_output = 0
        self.total_calls = 0
        self.current_model = self._model_name()

    def _model_name(self) -> str:
        s = get_settings()
        return s.dashscope_model if s.llm_provider == "dashscope" else s.ollama_model

    # ------------------------------------------------------ keys
    @staticmethod
    def _intent_key(intent: str) -> str:
        return f"token:intent:{intent}"

    @staticmethod
    def _conv_key(conv_id: str) -> str:
        return f"token:conv:{conv_id}"

    @staticmethod
    def _ttl() -> int:
        return getattr(get_settings(), "token_stats_ttl", 7 * 86400)

    # ------------------------------------------------------ write path
    def _incr_local(self, bucket: dict, usage: dict) -> None:
        bucket["input"] = bucket.get("input", 0) + usage.get("input_tokens", 0)
        bucket["output"] = bucket.get("output", 0) + usage.get("output_tokens", 0)
        bucket["total"] = bucket.get("total", 0) + usage.get("total_tokens", 0)
        # 与 Redis 路径一致：每次 incr 计 1 次调用（usage 未显式带 calls 时默认 1）
        bucket["calls"] = bucket.get("calls", 0) + (usage.get("calls") or 1)

    async def incr(self, conv_id: str, intent: str, usage: dict) -> None:
        """把一次真实 usage 累加到 意图维度 + 会话维度（Redis HINCRBY，失败降级内存）。"""
        intent = intent or "unknown"
        self.total_calls += 1
        self.total_input += usage.get("input_tokens", 0)
        self.total_output += usage.get("output_tokens", 0)
        with self._lock:
            self._incr_local(self._local.setdefault(intent, {}), usage)
            self._incr_local(self._local.setdefault(f"__conv__{conv_id}", {}), usage)

        if not getattr(get_settings(), "token_stats_redis_enabled", True):
            return
        try:
            from app.core.redis import redis_pool
            ttl = self._ttl()
            it = self._intent_key(intent)
            cv = self._conv_key(conv_id)
            pipe = redis_pool.pipeline()
            add = {
                f"{it}:input": usage.get("input_tokens", 0),
                f"{it}:output": usage.get("output_tokens", 0),
                f"{it}:calls": 1,
                f"{cv}:input": usage.get("input_tokens", 0),
                f"{cv}:output": usage.get("output_tokens", 0),
                f"{cv}:calls": 1,
            }
            for k, v in add.items():
                pipe.incrby(k, int(v))
                pipe.expire(k, ttl)
            await pipe.execute()
        except Exception as e:
            logger.warning("token_stats_redis_failed", error=str(e))  # 静默降级，内存已兜底

    # ------------------------------------------------------ read path
    def _read_local_bucket(self, key: str) -> dict | None:
        with self._lock:
            b = self._local.get(key)
            if not b:
                return None
            return dict(b)

    async def snapshot(self) -> dict:
        """聚合 Redis 与本地兜底，返回与旧 monitor() 同构的统计结构。"""
        intents: dict[str, dict] = {}
        convs: dict[str, dict] = {}

        if getattr(get_settings(), "token_stats_redis_enabled", True):
            try:
                from app.core.redis import redis_pool
                keys = await redis_pool.keys("token:*")
                bucketed: dict[str, dict] = {}
                for k in keys or []:
                    parts = k.split(":")
                    if len(parts) < 3:
                        continue
                    kind = parts[2]          # intent | conv
                    name = ":".join(parts[3:-1]) if parts[-1] == "calls" else \
                        ":".join(parts[3:])
                    metric = parts[-1]       # input | output | calls
                    if metric not in ("input", "output", "calls"):
                        continue
                    val = await redis_pool.get(k)
                    if val is None:
                        continue
                    c = bucketed.setdefault((kind, name), {})
                    c[metric] = int(val)
                for (kind, name), c in bucketed.items():
                    c["total"] = c.get("input", 0) + c.get("output", 0)
                    (intents if kind == "intent" else convs)[name] = c
            except Exception as e:
                logger.warning("token_stats_redis_read_failed", error=str(e))

        # 合并本地兜底（Redis 不可用或刚写入未落 Redis 时）
        with self._lock:
            for k, b in self._local.items():
                if k.startswith("__conv__"):
                    name = k[len("__conv__"):]
                    convs.setdefault(name, {"input": 0, "output": 0, "total": 0, "calls": 0})
                    for m in ("input", "output", "total", "calls"):
                        convs[name][m] = convs[name].get(m, 0) + b.get(m, 0)
                else:
                    intents.setdefault(k, {"input": 0, "output": 0, "total": 0, "calls": 0})
                    for m in ("input", "output", "total", "calls"):
                        intents[k][m] = intents[k].get(m, 0) + b.get(m, 0)

        return {
            "model": self.current_model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "by_intent": intents,
            "by_conversation": convs,
        }