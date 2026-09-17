"""语义判断与分级交互 — 零 token 单元测试（v2.0 取代 clarify）。

覆盖：contract 字段 / plan 解析 / ask_user need_info payload 映射 / not_doable 事件 /
summarize impossible_reason 注入 / plan prompt 指令。全部不发起真实 LLM / DB。
"""
import asyncio
import unittest.mock as mock

from app.core.langgraph_compat import ensure as _lg_ensure  # noqa: E402

_lg_ensure()

from app.agent.nodes import ask_user as au  # noqa: E402
from app.schemas.agent_plan import PlanOutput, PlanNeedField  # noqa: E402


# ---------- 契约层 ----------
def test_plan_need_field_fields():
    n = PlanNeedField(key="budget", question="预算?", required=True)
    assert n.type == "text"
    assert n.required is True
    assert n.options == []


def test_plan_output_feasibility_defaults():
    p = PlanOutput(intent="travel", confidence=0.9)
    assert p.feasibility == "feasible"
    assert p.needs == []
    assert p.impossible_reason == ""
    assert p.nudge == ""


def test_plan_output_accepts_needs():
    p = PlanOutput(intent="travel", confidence=0.8, feasibility="need_info",
                   needs=[PlanNeedField(key="days", question="几天?", required=False)])
    assert p.needs[0].key == "days"
    assert p.needs[0].required is False


# ---------- ask_user need_info payload 映射 ----------
def test_ask_user_need_info_maps_needs_to_questions():
    state = {
        "interrupt_source": "need_info",
        "needs": [
            {"key": "budget", "question": "你的预算多少?", "type": "option",
             "options": ["5000", "8000"], "required": True},
            {"key": "days", "question": "计划几天?", "type": "number", "required": False},
        ],
        "missing_params": [],
    }
    payload = au._build_payload(state)
    assert payload["source"] == "need_info"
    assert len(payload["questions"]) == 2
    q0 = payload["questions"][0]
    assert q0["type"] == "option"
    assert q0["required"] is True
    assert q0["meta"]["key"] == "budget"
    assert q0["meta"]["required"] is True
    q1 = payload["questions"][1]
    assert q1["type"] == "number"
    assert q1["meta"]["key"] == "days"
    assert q1["meta"]["required"] is False


def test_ask_user_need_info_empty_needs():
    payload = au._build_payload({"interrupt_source": "need_info", "needs": [],
                                 "missing_params": []})
    assert payload["source"] == "need_info"
    assert payload["questions"] == []


# ---------- plan 解析 needs（走 _need_to_dict，含 required 标记） ----------
def test_plan_need_to_dict_marks_required():
    from app.agent.nodes.plan import _need_to_dict
    d = {"key": "city", "question": "去哪个城市?", "required": False}
    assert _need_to_dict(d)["required"] is False
    d2 = {"key": "city", "question": "去哪个城市?"}
    assert _need_to_dict(d2)["required"] is True


# ---------- summarize prompt 含引导指令 + impossible_reason 槽 ----------
def test_summarize_prompt_contains_nudge_and_impossible():
    from app.agent.prompts.summarize_prompt import SUMMARIZE_PROMPT
    assert "{impossible_reason}" in SUMMARIZE_PROMPT
    assert "引导下一步" in SUMMARIZE_PROMPT


# ---------- plan prompt 含 feasibility 三值与 travel 强制偏好项指令 ----------
def test_plan_prompt_has_feasibility_and_required_false_hint():
    from app.agent.prompts.plan_prompt import PLAN_PROMPT
    assert "feasibility" in PLAN_PROMPT
    assert "not_doable" in PLAN_PROMPT
    assert "required=false" in PLAN_PROMPT
    assert "预算" in PLAN_PROMPT or "destination" in PLAN_PROMPT


# ---------- summarize_node 注入 impossible_reason（mock LLM 无 token） ----------
def test_summarize_injects_impossible_reason():
    from app.agent.nodes import summarize as sm

    captured = {}

    async def fake_llm(**kw):
        captured["prompt"] = kw.get("system_prompt", "")
        return "抱歉无法完成。"

    state = {
        "messages": [{"role": "user", "content": "帮我订一条不存在的航线"}],
        "recent_context": "", "rag_memories": "", "manual_memories": "",
        "tool_results": {}, "charts": [],
        "impossible_reason": "该航线不存在，请提供机场三字码。",
    }
    with mock.patch.object(sm.llm, "chat", new=fake_llm):
        out = asyncio.run(sm.summarize_node(state))
    assert "该航线不存在" in captured["prompt"]
    assert out["answer"] == "抱歉无法完成。"


# ---------- P1: 异常中间件二次崩溃修复 ----------
def test_exception_middleware_survives_missing_state():
    """P1: app.state 未设 debug 时，未捕获异常 handler 不二次崩溃，且不透传 detail。"""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from app.core.http_middleware import UnifiedExceptionMiddleware

    def boom(request):
        raise RuntimeError("boom")

    app = Starlette(routes=[Route("/x", boom, methods=["GET"])])
    app.add_middleware(UnifiedExceptionMiddleware)
    # 故意不设 app.state.debug，验证 getattr 兜底

    client = TestClient(app)
    resp = client.get("/x")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == 500
    assert "request_id" in body
    assert body["detail"] is None  # debug=False/未设置 → 不透传异常 detail