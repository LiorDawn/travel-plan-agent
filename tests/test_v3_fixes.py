"""v3.0 九项优化 — 零 token 单元测试（A~F）

覆盖：F2 PlanTask 唯一 id / build_plan_tasks、RAG 改写/embedding 双缓存、
子图会话级注入复用、token 统计 Redis→内存降级、统一 ClarifyPayload、
ParamCheck 缺参自愈与降级、done.success、批量 save_messages、会话级沉淀合并。
全部断言不发起真实 LLM / Redis / DB 调用。
"""
import asyncio
import unittest.mock as mock

# 复用生产兼容层：本环境 langgraph 1.0.10 + langchain 1.x 缺 langgraph.runtime 注解符号，
# 导致 import langchain.agents(.middleware) 直接 ImportError。ensure() 补齐占位后
# 即可导入并测试真实的 ParamCheck / RAGInject 中间件（而非替换桩）。
from app.core.langgraph_compat import ensure as _lg_ensure  # noqa: E402

_lg_ensure()

from app.agent import tasks


# ---------- B/F2：plan 产出 PlanTask、唯一 id、依赖角色、fallback ----------
def test_build_plan_tasks_keeps_deps_and_role():
    pt = [
        {"id": "a", "tool": "get_attractions", "depends_on": [],
         "sub_agent_type": "attraction", "desc": "查景点"},
        {"id": "b", "tool": "get_weather", "depends_on": ["a"],
         "sub_agent_type": "weather", "desc": "查天气"},
    ]
    t = tasks.build_plan_tasks(pt, [])
    assert len(t) == 2
    assert [x["sub_agent_type"] for x in t] == ["attraction", "weather"]
    assert t[1]["depends_on"] == ["a"]
    assert t[0]["id"] == "a" and t[1]["id"] == "b"
    assert all(x["status"] == "pending" for x in t)


def test_build_plan_tasks_unique_id_on_duplicate_tool():
    pt = [
        {"id": "", "tool": "search_flights", "desc": "去程"},
        {"id": "", "tool": "search_flights", "desc": "返程"},
    ]
    t = tasks.build_plan_tasks(pt, [])
    ids = [x["id"] for x in t]
    assert len(ids) == len(set(ids))            # 不撞 id
    assert ids[0].startswith("search_flights")
    assert ids[1] != ids[0]


def test_build_plan_tasks_fallback_to_tool_plan_when_empty():
    plan = [{"name": "get_weather", "params": {"city": "三亚"}, "ready": True}]
    t = tasks.build_plan_tasks([], plan)
    assert len(t) == 1 and t[0]["tool"] == "get_weather"
    # 缺 id 时从 tool_plan 兜底派生也要唯一
    t2 = tasks.build_plan_tasks([], [{"name": "x", "params": {}, "ready": True}])
    assert t2[0]["id"]


# ---------- C：RAG 改写/embedding 双缓存（Redis→内存→miss 降级） ----------
def test_local_lru_fresh_and_expiry():
    from app.memory.rag.cache import LocalLruCache
    c = LocalLruCache()
    c.set("k", {"v": 1})
    assert c.get_if_fresh("k", ttl=100) == {"v": 1}
    assert c.within_ttl("k", ttl=100) is True


def test_rewrite_cache_set_get_roundtrip():
    import asyncio
    from app.memory.rag import cache as rag_cache
    async def run():
        await rag_cache.rewrite_cache_set("三亚 攻略", "三亚", ["三亚", "攻略"], ttl=60)
        val = await rag_cache.rewrite_cache_get("三亚 攻略", ttl=60)
        assert val is not None
        assert val["query"] == "三亚"
        assert val["keywords"] == ["三亚", "攻略"]
    asyncio.run(run())


def test_embedding_cache_set_get_roundtrip():
    import asyncio
    from app.memory.rag import cache as rag_cache
    async def run():
        vec = [0.1, 0.2, 0.3]
        await rag_cache.embedding_cache_set("三亚", vec, ttl=1000)
        got = await rag_cache.embedding_cache_get("三亚", ttl=1000)
        assert got == vec
    asyncio.run(run())


# ---------- C：子图会话级注入（首检后本会话复用，不再重复检索） ----------
def test_rag_inject_session_reuse_avoids_re_retrieve():
    import asyncio
    from app.agent.subagent.middleware.rag_inject import RAGInjectMiddleware
    mw = RAGInjectMiddleware()

    class FakeState(dict):
        pass

    state = FakeState()
    recalled = {"calls": 0}

    async def fake_recall_manual(query):
        recalled["calls"] += 1
        return "三亚攻略内容"

    with mock.patch("app.agent.subagent.middleware.rag_inject.recall_manual",
                    new=fake_recall_manual):
        class Req:
            def __init__(self):
                self.state = state
                self.messages = [{"role": "user", "content": "我要去三亚"}]
                self.system_message = None
        req = Req()
        async def handler(r):
            return "ok"
        asyncio.run(mw.awrap_model_call(req, handler))
        asyncio.run(mw.awrap_model_call(req, handler))

    assert recalled["calls"] == 1          # 第二次复用缓存，未再检索
    assert state.get("rag_manual_cached") is True


# ---------- D：token 统计 Redis→内存降级 ----------
def test_token_stats_local_fallback_and_snapshot():
    import asyncio
    from app.services.token_stats import TokenStats
    ts = TokenStats()
    with mock.patch.object(ts, "_model_name", return_value="qwen-turbo"):
        async def run():
            await ts.incr("conv-1", "travel",
                          {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            snap = await ts.snapshot()
            assert snap["by_intent"]["travel"]["input"] == 10
            assert snap["by_intent"]["travel"]["calls"] == 1
            assert snap["by_conversation"]["conv-1"]["output"] == 5
            assert snap["model"] == "qwen-turbo"
        # 强制走内存路径（Redis 不可用 → 不抛）
        with mock.patch("app.services.token_stats.get_settings") as gs:
            gs.return_value = type("S", (), {"token_stats_redis_enabled": False})()
            asyncio.run(run())


# ---------- E：ParamCheck 缺参自愈 + 达到上限降级放行 ----------
def test_param_check_self_heal_injects_hint_without_interrupt():
    from langchain_core.messages import AIMessage
    from app.agent.subagent.middleware.param_check import ParamCheckMiddleware

    tc = {"id": "call_1", "name": "get_weather", "args": {}}
    state = {"messages": [AIMessage(content="", tool_calls=[tc])], "ask_count": 0}
    with mock.patch("app.agent.subagent.middleware.param_check.tool_registry") as tr:
        tr.missing_required.return_value = ["city"]
        out = ParamCheckMiddleware().after_model(state, None)

    assert out is not None                        # 自愈而非放行
    assert out["ask_count"] == 1
    assert "city" in out["messages"][0].content   # 提示让模型补全参数


def test_param_check_degrades_passes_through_at_limit():
    from langchain_core.messages import AIMessage
    from app.agent.subagent.middleware.param_check import ParamCheckMiddleware

    tc = {"id": "call_1", "name": "get_weather", "args": {}}
    state = {"messages": [AIMessage(content="", tool_calls=[tc])], "ask_count": 3}
    with mock.patch("app.agent.subagent.middleware.param_check.tool_registry") as tr:
        tr.missing_required.return_value = ["city"]
        out = ParamCheckMiddleware().after_model(state, None)

    assert out is None                            # 达到上限不再打断，放行让工具自身报错


# ---------- E：统一 ClarifyPayload（plan 缺参 → 统一协议形状） ----------
def test_clarify_payload_unified_shape():
    from app.agent.nodes.ask_user import _build_payload

    state = {"missing_params": [
        {"tool": "get_weather", "fields": ["city"]},
        {"tool": "search_flights", "fields": ["日期", "路线"]},
    ]}
    payload = _build_payload(state)
    assert set(payload) == {"source", "round", "message", "questions"}
    assert payload["source"] == "plan"
    assert len(payload["questions"]) == 2
    q = payload["questions"][0]
    assert q["meta"] == {"tool": "get_weather", "missing_fields": ["city"]}
    assert q["type"] == "text" and q["required"] is True


# ---------- F：done.success（正常 true / 中断 false） ----------
def test_finish_done_success_flag():
    from app.agent.runner import TravelPlanAgent
    r = TravelPlanAgent.__new__(TravelPlanAgent)
    ok = r._finish("c1", {}, None)
    assert ok[0].event_type == "done" and ok[0].data["success"] is True

    interrupted = r._finish("c1", {}, {"message": "?"})
    done = [e for e in interrupted if e.event_type == "done"][0]
    assert done.data["success"] is False and done.data["need_input"] is True

    erred = r._finish("c1", {}, None, RuntimeError("x"))
    done2 = [e for e in erred if e.event_type == "done"][0]
    assert done2.data["success"] is False


# ---------- F：批量 save_messages 单事务 ----------
def test_save_messages_batch_commit_once():
    from app.memory.long_term import save_messages

    class FakeDB:
        def __init__(self):
            self.added = []
            self.commits = 0
        def add_all(self, rows):
            self.added.extend(rows)
        async def commit(self):
            self.commits += 1
        async def refresh(self, m):
            m.id = 1

    async def run():
        db = FakeDB()
        rows = await save_messages(db, "conv-1", [("user", "hi"), ("assistant", "yo")])
        assert db.commits == 1                 # 一批一次 commit
        assert len(rows) == 2
        assert all(m.conversation_id == "conv-1" for m in rows)
    asyncio.run(run())


# ---------- F：会话级沉淀合并（_is_duplicate 限定 source_id） ----------
def test_is_duplicate_no_false_skip():
    from app.memory.rag import service as rag_service

    class FakeResult:
        def scalars(self):
            return self
        def first(self):
            return None   # 同 source_id 无最近块 → 返回 False，不误拦

    class FakeSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def execute(self, stmt):
            return FakeResult()

    with mock.patch.object(rag_service, "AsyncSessionLocal", return_value=FakeSession()):
        assert asyncio.run(rag_service._is_duplicate("x", "u1", "conv-1")) is False