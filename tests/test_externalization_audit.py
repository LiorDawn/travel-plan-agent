"""F4/F5 零 token 测试 — 上下文外部化 + 审计埋点逻辑

不调用真实 LLM / 不依赖 DB：分别直接测 ArtifactStore 的判定/摘要/保留
与 AuditRecorder 的采集/脱敏/采样。
"""
from unittest.mock import patch

from app.memory.artifact import ArtifactStore
from app.memory.audit import recorder as _ar


def _big_json(n=5):
    return {"data": [{"name": f"景点{i:03d}", "desc": "x" * 100} for i in range(n)],
            "chart": {"type": "bar", "data": [1, 2, 3]}}


def test_small_result_passthrough_no_store():
    store = ArtifactStore(threshold=500, top_n=3)
    out = store.maybe_summarize("get_attractions", {"data": [1, 2]}, "conv_1")
    assert out["artifact_ref"] is None
    assert "raw" in out  # 小结果原样
    assert store._store == {}  # 未写库


def test_big_result_summarized_with_ref_and_chart():
    store = ArtifactStore(threshold=200, top_n=2, summary_max=200)
    raw = _big_json()
    out = store.maybe_summarize("get_attractions", raw, "conv_1")
    assert out["artifact_ref"].startswith("artifact:conv_1:get_attractions:")
    assert "summary" in out and "raw" not in out       # 只进摘要
    assert out.get("chart") == raw["chart"]            # chart 保留（防丢图表）
    # 摘要不含冗长 desc（确定性压缩只保留前 top_n 条且截断）
    assert "景点00" in out["summary"]
    assert out["summary"] not in ("x" * 100)  # 不是长原文


def test_load_returns_original():
    store = ArtifactStore(threshold=200, top_n=2)
    raw = _big_json()
    out = store.maybe_summarize("kpis", raw, "conv_1")
    loaded = store.load(out["artifact_ref"])
    assert loaded is not None
    # 原文可回取且含全部数据
    data = loaded["data"] if isinstance(loaded, dict) else None
    assert data is not None and len(data) == 5


def test_exception_fallback_truncates():
    # content_type 无法识别且 llm 摘要抛错 → 兜底截断进上下文，绝不抛
    store = ArtifactStore(threshold=100, top_n=2, llm_summary=True)
    raw = "A" * 300
    out = store.maybe_summarize("text_tool", raw, "c1")
    # 无原文长串，走兜底截断
    assert len(str(out.get("summary", out.get("raw", "")))) <= 200


def test_audit_recorder_collects_and_masks():
    re = _ar._AuditRecorder()
    re.record(stage="execute", action="tool_call", actor="tool:get_attractions",
              input_summary={"city": "三亚", "password": "secret123"},
              output_summary={"data": ["x"] * 30})
    events = re.drain()
    assert len(events) == 1
    ev = events[0]
    assert ev["action"] == "tool_call"
    # 脱敏：password 被替换，长列表截断
    assert ev["input_summary"]["password"] == "***"
    assert len(ev["output_summary"]["data"]) <= 20


def test_audit_sampling_skips_non_forced():
    # 采样率 0 时非 forced 丢弃，forced(权限/审批) 必审计
    re = _ar._AuditRecorder()
    with patch.object(_ar, "get_settings", return_value=_MockSettings(0.0)):
        re.record(stage="plan", action="model_call", forced=False)
        re.record(stage="plan", action="permission_decision", forced=True)
    events = re.drain()
    assert len(events) == 1 and events[0]["action"] == "permission_decision"


class _MockSettings:
    def __init__(self, sampling):
        self.audit_sampling = sampling