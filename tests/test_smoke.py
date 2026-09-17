"""冒烟测试 — 验证重构后模块可导入、Agent 图可编译、事件模型与用例层可实例化。

不调用真实 LLM / 不跑联网（仅本地对象构建与图编译），用于在 CI 或本地快速发现：
import 链断裂、图装配错误、事件协议漂移等问题。
"""


def test_app_importable():
    import app.main  # noqa: F401


def test_agent_graph_compiles():
    from app.agent.graph import _compiled_graph
    assert _compiled_graph is not None


def test_chat_event_protocol():
    from app.agent.events import ChatEvent
    assert ChatEvent(event_type="message", data={"message": "hello"})
    assert ChatEvent(event_type="done", data={}).event_type == "done"


def test_services_instantiable():
    from app.services.chat_service import ChatService
    from app.services.conversation_service import ConversationService
    from app.services.plan_service import PlanService
    assert ChatService().agent is not None
    assert ConversationService()
    assert PlanService()


def test_route_after_plan_empty_state_goes_summarize():
    from app.agent.router import route_after_plan
    # 空 state（无工具/无意图唯一）→ 走 summarize（分发兜底）
    assert route_after_plan({}) == "summarize"
    assert route_after_plan({"intent": "query", "tool_plan": [{"name": "x", "ready": True}]}) == "execute"