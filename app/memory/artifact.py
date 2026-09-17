"""上下文外部化 — ArtifactStore（F5）

大工具结果（景点列表 / 航班表 / KPIs）不进模型上下文，仅摘要 + 引用；
长原文按需落库，由 artifact_ref 懒加载取回。小结果原样放行。

优先级与稳定性：
- 旁路式：save/load 失败仅降级（截断后进上下文），绝不抛错阻塞主图。
- 阈值按序列化字符数计；小结果（≤阈值）不写库、不付摘要成本。
- 结构化结果（json/table）用确定性压缩（非 LLM，省 token 可复现）；
  非结构化 text 可 LLM 摘要（开关默认关，回退启发式截断）。
"""
import json
import time

from app.core.config import get_settings
from app.core.logging import logger

# 低于该阈值的结果原样进上下文（单位：序列化字符数）
_DEFAULT_SUMMARY_THRESHOLD = 1200
# 命中阈值后，确定性压缩最多保留多少行
_DEFAULT_TOP_N = 8
# 摘要文本最长字符（确定性压缩与启发式截断共用上限）
_DEFAULT_SUMMARY_MAX = 500
# 外部化原文在进程内的存活秒数（P1-2：无 TTL 曾导致只增不减，内存无限增长）
_DEFAULT_TTL = 86400  # 1 天


class ArtifactStore:
    """工具结果 → 摘要 + 可选 artifact_ref 的外部化存储。

    持久化默认内存表（进程内），生产可扩展 DB/Redis 实现，接口不变。
    本项目 memory 默认在 PG；此处加载用 async 能力留空、本地先行内存兜底。
    """

    def __init__(self, threshold: int | None = None, top_n: int | None = None,
                 summary_max: int | None = None, llm_summary: bool | None = None):
        s = get_settings()
        self.threshold = threshold or _DEFAULT_SUMMARY_THRESHOLD
        self.top_n = top_n or _DEFAULT_TOP_N
        self.summary_max = summary_max or _DEFAULT_SUMMARY_MAX
        self.llm_summary = s.artifact_llm_summary if getattr(s, "artifact_llm_summary", False) else (llm_summary or False)
        self.ttl = _DEFAULT_TTL
        # 进程内存储：artifactId -> (created_at, raw, content_type)
        self._store: dict[str, tuple] = {}

    # ---------- 生命周期 ----------
    def save(self, conv_id: str, tool: str, raw, content_type: str) -> str:
        """长原文落库，返回 artifact_ref。"""
        self._gc()  # 懒清理过期项，防止只增不减
        ref = f"artifact:{conv_id}:{tool}:{int(time.time())}"
        self._store[ref] = (time.time(), raw, content_type)
        return ref

    def load(self, artifact_id: str):
        """按引用取回原文；不存在或已过期返回 None。"""
        item = self._store.get(artifact_id)
        if item is None:
            return None
        if time.time() - item[0] > self.ttl:  # 过期即移除
            self._store.pop(artifact_id, None)
            return None
        return item[1]

    def _gc(self) -> None:
        """懒清理：移除所有超过 TTL 的条目，控制内存上界。"""
        now = time.time()
        expired = [ref for ref, (ts, _, _) in self._store.items() if now - ts > self.ttl]
        for ref in expired:
            self._store.pop(ref, None)

    # ---------- 对外主入口 ----------
    def maybe_summarize(self, tool: str, raw, conv_id: str) -> dict:
        """统一返回契约 {raw|summary, artifact_ref, chart}：
        - 小结果：原样返回 raw，不写库；
        - 大结果：写库 + 生成摘要，返回 {summary, artifact_ref, chart}。
        chart 字段始终原样保留（供 summarize 抽图表，绝不随截断丢弃）。
        """
        if not self._looks_large(raw):
            return {"raw": raw, "artifact_ref": None}
        try:
            content_type = self._content_type(raw)
            chart = self._extract_chart(raw)
            ref = self.save(conv_id, tool, raw, content_type)
            summary = self._summarize(raw, content_type)
            payload = {"summary": summary, "artifact_ref": ref}
            if chart is not None:   # 关键字段 chart 保留，防 analyze_kpis 图表丢失
                payload["chart"] = chart
            return payload
        except Exception as e:
            logger.warning("artifact_summarize_failed", tool=tool, error=str(e))
            # 兜底：截断到阈值内直接进上下文，绝不抛错
            return {"raw": self._truncate(raw), "artifact_ref": None}

    # ---------- 内部：判定 / 摘要 / 工具 ----------
    def _looks_large(self, raw) -> bool:
        text = self._to_text(raw)
        return len(text) > self.threshold

    def _summarize(self, raw, content_type: str) -> str:
        """按 content_type 分派摘要（同步部分）。text+LLM 摘要走 async_maybe_summarize。"""
        if content_type in ("json", "table"):
            return self._deterministic_summary(raw)
        # text 类：同步路径不调 async LLM，回退启发式截断；要 LLM 摘要用 async_maybe_summarize
        return self._truncate(raw)

    async def async_maybe_summarize(self, tool: str, raw, conv_id: str) -> dict:
        """async 版：text 且开启 LLM 摘要时走 LLM，其余同 maybe_summarize。"""
        if not self._looks_large(raw):
            return {"raw": raw, "artifact_ref": None}
        try:
            content_type = self._content_type(raw)
            chart = self._extract_chart(raw)
            ref = self.save(conv_id, tool, raw, content_type)
            summary = await self._async_summarize(raw, content_type)
            payload = {"summary": summary, "artifact_ref": ref}
            if chart is not None:
                payload["chart"] = chart
            return payload
        except Exception as e:
            logger.warning("artifact_async_summarize_failed", tool=tool, error=str(e))
            return {"raw": self._truncate(raw), "artifact_ref": None}

    async def _async_summarize(self, raw, content_type: str) -> str:
        if content_type in ("json", "table"):
            return self._deterministic_summary(raw)
        if self.llm_summary and content_type == "text":
            try:
                from app.core.llm import llm  # 延迟导入避免循环
                text = self._to_text(raw)
                return (await llm.chat(
                    system_prompt="请用 3 条以内简洁要点概括以下工具结果，保留关键数值。",
                    user_message=text[: 2000], max_tokens=200))[: self.summary_max]
            except Exception as e:
                logger.warning("artifact_llm_summary_failed", error=str(e))
                return self._truncate(raw)
        return self._truncate(raw)

    def _deterministic_summary(self, raw) -> str:
        """结构化结果：保留 top-N 行 + 统计摘要，不调 LLM。"""
        rows, stats = self._extract_rows_and_stats(raw)
        out = []
        if stats:
            out.append("统计: " + stats)
        shown = rows[: self.top_n]
        out.append(f"前 {len(shown)}/{len(rows)} 条: " + json.dumps(shown, ensure_ascii=False))
        text = "\n".join(out)
        return text[: self.summary_max]

    def _truncate(self, raw) -> str:
        text = self._to_text(raw)
        return text[: self.threshold]

    # ---------- 内部：结构解析 ----------
    def _to_text(self, raw) -> str:
        if isinstance(raw, str):
            return raw
        try:
            return json.dumps(raw, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(raw)

    @staticmethod
    def _content_type(raw) -> str:
        if isinstance(raw, list):
            return "json"
        if isinstance(raw, dict):
            inner = raw.get("data", raw.get("results", raw.get("items")))
            return "table" if isinstance(inner, list) else "text"
        if isinstance(raw, str):
            return "text"
        return "text"

    @staticmethod
    def _extract_chart(raw):
        """从结果中取出图表数据；无则 None。结构多样的 tool 各自可取。"""
        if isinstance(raw, dict):
            if "chart" in raw:
                return raw["chart"]
            # 兼容 ERP KPI 结构：放入 items 内最后一个 chart 字段
            for key in ("data", "results", "items"):
                inner = raw.get(key)
                if isinstance(inner, list):
                    for row in inner:
                        if isinstance(row, dict) and "chart" in row:
                            return row["chart"]
        return None

    @staticmethod
    def _extract_rows_and_stats(raw) -> tuple[list, str]:
        """把结构化结果压成 行列表 + 统计摘要字符串。"""
        rows: list = []
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            inner = raw.get("data", raw.get("results", raw.get("items")))
            if isinstance(inner, list):
                rows = inner
                stats = raw.get("total")
                if stats is not None:
                    return rows, f"共 {stats} 条"
        return rows, f"共 {len(rows)} 条"


# 进程内单例（execute / summarize / planner_run 共用同一存储，便于跨节点取回）
artifact_store = ArtifactStore()