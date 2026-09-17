"""审计修复回归测试（零 token）— 覆盖 E1/E2/E3 与语义分流（v2.0 取代 clarify）。

- E1: ask_user_node 返回时 ask_count+1（"最多追问1次"真正生效）
- E2: 语义分流 need_info / not_doable 优先，机械缺参兜底
- E3: ask_user 首次挂起收集 answers 回灌 clarifications（原 clarify 职责合并），need_info_asked 置位
"""
import asyncio
from unittest.mock import patch

from app.agent import router
from app.agent.state import AgentState


def _s(**over):
    base = {
        "intent": "travel", "tool_plan": [], "missing_params": [],
        "ask_count": 0, "thinking_mode": True, "feasibility": "feasible",
        "needs": [], "need_info_asked": False,
    }
    base.update(over)
    return base


# ---------- E2：语义分流 ----------
def test_not_doable_goes_summarize():
    """① not_doable → summarize（说明无法完成 + 询问，不弹窗不 execute）。"""
    assert router.route_after_plan(_s(feasibility="not_doable")) == "summarize"


def test_need_info_first_round_interrupts():
    """② need_info 首轮 → interrupt（ask_user 弹窗收集 needs）。"""
    s = _s(feasibility="need_info",
           needs=[{"key": "budget", "question": "预算?", "required": True}])
    assert router.route_after_plan(s) == "interrupt"


def test_need_info_already_asked_degrades():
    """③ need_info 已弹过一次（need_info_asked）→ 降级 summarize 不重复弹窗。"""
    s = _s(feasibility="need_info", need_info_asked=True, ask_count=1)
    assert router.route_after_plan(s) == "summarize"


def test_need_info_second_round_degrades():
    """③ need_info ask_count>=1（已追问过仍缺）→ 降级 summarize 不二次打断。"""
    s = _s(feasibility="need_info", ask_count=1)
    assert router.route_after_plan(s) == "summarize"


def test_missing_after_ask_degrades():
    """③ 非 need_info 但已追问过仍缺参 → summarize（不无限问）。"""
    s = _s(intent="query", ask_count=1, missing_params=[{"tool": "get_attractions", "fields": ["city"]}])
    assert router.route_after_plan(s) == "summarize"


def test_missing_first_round_interrupts():
    """④ 未问过且机械缺参 → interrupt（source=plan 兜底补漏判字段）。"""
    s = _s(missing_params=[{"tool": "search_flights", "fields": ["origin"]}])
    assert router.route_after_plan(s) == "interrupt"


def test_feasible_ready_goes_execute():
    """⑦ 可行、参数齐、关思考 → execute。"""
    s = _s(thinking_mode=False, tool_plan=[{"name": "search_flights", "ready": True}])
    assert router.route_after_plan(s) == "execute"


def test_feasible_ready_thinking_goes_react():
    """⑦ 可行、参数齐、开思考 → react 子图。"""
    s = _s(tool_plan=[{"name": "search_flights", "ready": True}])
    assert router.route_after_plan(s) == "react"


def test_general_goes_direct():
    """⑤ 闲聊 → direct。"""
    assert router.route_after_plan(_s(intent="general")) == "direct"


def test_no_ready_tools_goes_summarize():
    """⑥ 无齐参工具 → summarize。"""
    s = _s(intent="query", tool_plan=[{"name": "get_attractions", "ready": False}])
    assert router.route_after_plan(s) == "summarize"


# ---------- E1：ask_user 计数与 need_info ----------
def test_ask_user_increments_count():
    """E1: ask_user_node resume 后中断返回，ask_count+1。"""
    from app.agent.nodes import ask_user as au

    def fake_interrupt(payload):
        return "北京"

    state = {"ask_count": 0, "missing_params": [{"tool": "t", "fields": ["f"]}], "interrupt_source": "plan"}
    with patch.object(au, "interrupt", new=fake_interrupt):
        out = asyncio.run(au.ask_user_node(state))
    assert out["ask_count"] == 1
    assert out["resume_input"] == "北京"


def test_ask_user_need_info_sets_flag():
    """E3: need_info 来源 resume 后置 need_info_asked，防二次弹窗。"""
    from app.agent.nodes import ask_user as au

    def fake_interrupt(payload):
        return '{"answers": {"budget": "8000"}}'

    state = {"ask_count": 0, "interrupt_source": "need_info", "needs": [],
             "missing_params": []}
    with patch.object(au, "interrupt", new=fake_interrupt):
        out = asyncio.run(au.ask_user_node(state))
    assert out["need_info_asked"] is True
    assert out["ask_count"] == 1


def test_ask_user_replays_answers_to_clarifications():
    """P0: 弹窗收集的 answers 必须回灌 clarifications，询问闭环才能通。"""
    from app.agent.nodes import ask_user as au

    def fake_interrupt(payload):
        return '{"answers": {"budget": "8000", "days": "3"}}'

    state = {"ask_count": 0, "interrupt_source": "need_info", "needs": [],
             "missing_params": [], "clarifications": {"pre": "x"}}
    with patch.object(au, "interrupt", new=fake_interrupt):
        out = asyncio.run(au.ask_user_node(state))
    assert out["clarifications"]["budget"] == "8000"
    assert out["clarifications"]["days"] == "3"
    assert out["clarifications"]["pre"] == "x"  # 保留既有，合并


def test_ask_user_compact_json_replays():
    """P0: 紧凑 {@key:value} JSON 也回灌 clarifications。"""
    from app.agent.nodes import ask_user as au

    def fake_interrupt(payload):
        return '{"city": "北京", "days": "5"}'

    state = {"ask_count": 0, "interrupt_source": "need_info", "needs": [],
             "missing_params": []}
    with patch.object(au, "interrupt", new=fake_interrupt):
        out = asyncio.run(au.ask_user_node(state))
    assert out["clarifications"]["city"] == "北京"


def test_ask_user_plain_text_does_not_overwrite():
    """P0: 纯文本回答走 resume_input，不覆盖 clarifications。"""
    from app.agent.nodes import ask_user as au

    def fake_interrupt(payload):
        return "北京就好"

    state = {"ask_count": 0, "interrupt_source": "need_info", "needs": [],
             "missing_params": [], "clarifications": {}}
    with patch.object(au, "interrupt", new=fake_interrupt):
        out = asyncio.run(au.ask_user_node(state))
    assert out["resume_input"] == "北京就好"
    assert out.get("clarifications", {}) == {}


def test_route_after_ask_user_need_info_returns_plan():
    """E3: need_info 收集完 → route_after_ask_user 回 plan 重自评。"""
    assert router.route_after_ask_user({"interrupt_source": "need_info"}) == "plan"
    assert router.route_after_ask_user({"interrupt_source": "plan"}) == "plan"
    assert router.route_after_ask_user({"interrupt_source": "subagent"}) == "planner_run"