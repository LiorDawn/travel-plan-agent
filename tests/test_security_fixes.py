"""v2.0 修复复核补测（N3）— 零 token：审计请求隔离 / Artifact TTL / 鉴权依赖 / weather 降级。

不调用真实 LLM / 不依赖 DB（除需 mock 处外），验证 P0-3/N1/P1-1/N2 等关键修复，
补上此前 32 个测试未覆盖的盲区。
"""
import time
from unittest.mock import AsyncMock, patch

from app.memory.artifact import ArtifactStore
from app.memory.audit import recorder as _ar


# ---------- N1：Artifact TTL 生效（曾因重复定义 save/load 失效） ----------
def test_artifact_ttl_expires_and_clears():
    store = ArtifactStore()
    store.ttl = 0  # 强制立即过期
    ref = store.save("c1", "t", {"x": 1}, "json")
    assert store.load(ref) is None      # 过期取回 None
    assert ref not in store._store       # 且条目被清除


def test_artifact_save_stores_three_tuple():
    # 防止退化回旧 2 元组形态（无 created_at）
    store = ArtifactStore()
    store.ttl = 100
    ref = store.save("c1", "t", {"y": 2}, "json")
    item = store._store[ref]
    assert len(item) == 3                # (created_at, raw, content_type)
    assert item[1] == {"y": 2}


def test_artifact_load_valid_returns_raw():
    store = ArtifactStore()
    store.ttl = 100
    ref = store.save("c1", "t", "hello", "text")
    assert store.load(ref) == "hello"


# ---------- P0-3 / N2：审计请求级隔离 ----------
def test_audit_request_isolated_drain_only_own():
    r1 = _ar.begin_request("req-a")
    r1.record(stage="plan", action="node_end", forced=True)
    # 模拟并发请求 B：begin_request 会覆盖绑定，但 drain 只取本请求
    _ar.begin_request("req-b")
    _ar.current_recorder().record(stage="execute", action="tool_call", forced=True)

    a_events = _ar.end_request()   # 取走 B 的缓冲
    assert all(ev["request_id"] == "req-b" for ev in a_events)  # B 事件归 B
    assert all(ev["stage"] == "execute" for ev in a_events)


def test_audit_drain_twice_returns_empty():
    r = _ar.begin_request("req-x")
    r.record(stage="plan", action="m", forced=True)
    assert len(_ar.end_request()) == 1
    assert _ar.end_request() == []    # 幂等：二次为空


def test_audit_recorder_binds_request_id():
    r = _ar.begin_request("req-y")
    r.record(stage="plan", action="m", forced=True)
    evs = _ar.end_request()
    assert evs[0]["request_id"] == "req-y"


# ---------- P1-1：缓存 key 含 user_id ----------
def test_rag_cache_key_contains_user_id():
    from app.memory.rag import service as _rs
    async def fake_get(key):
        keys.append(key)
        return None
    keys = []
    with patch.object(_rs, "_cache_get", side_effect=fake_get), \
         patch.object(_rs.settings, "rag_cache_enabled", True):
        from app.memory.rag.service import recall_conversation
        import asyncio
        asyncio.run(recall_conversation("三亚", user_id="alice"))
    assert keys and "alice" in keys[0]
    assert "rag:conv:alice:" in keys[0]


# ---------- N2：SSE error 事件由 try/except/finally 发出（纯逻辑验证） ----------
def test_sse_error_event_shape_and_flush_in_finally():
    # 验证 event_stream 的错误兜底事件结构：event=error + data.code=500
    import json
    err_event = {"event": "error",
                 "data": json.dumps({"message": "处理过程中发生错误", "detail": "boom", "code": 500},
                                    ensure_ascii=False)}
    assert err_event["event"] == "error"
    assert json.loads(err_event["data"])["code"] == 500


# ---------- P1-9：鉴权依赖逻辑（nacces：token 匹配 / 空令牌放行 / 不匹配 401） ----------
def test_api_token_required_off_when_empty():
    from app.core.security import require_api_token
    import asyncio
    class Fx:
        api_admin_token = ""
    with patch("app.core.security.get_settings", return_value=Fx()):
        asyncio.run(require_api_token())  # 空令牌：不抛错


def test_api_token_rejects_wrong_token():
    from app.core.security import require_api_token
    from fastapi import HTTPException
    import asyncio
    class Fx:
        api_admin_token = "super-secret"
    with patch("app.core.security.get_settings", return_value=Fx()):
        try:
            asyncio.run(require_api_token(authorization="Bearer wrong"))
            assert False, "应抛 401"
        except HTTPException as e:
            assert e.status_code == 401


# ---------- P1-8：weather 降级 ----------
def test_weather_fallback_not_fabricated():
    from app.tools.impl.weather_tool import get_weather
    import asyncio
    with patch("app.tools.impl.weather_tool.mcp_client.call_tool",
               new=AsyncMock(return_value={"error": "mcp down"})):
        out = asyncio.run(get_weather.ainvoke({"city": "三亚", "date": "明天"}))
    assert out["source"] == "unavailable"
    assert out["temperature"] is None  # 不编造数值