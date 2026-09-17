"""F2 任务清单状态机 — 零 token 测试

验证：tool_plan → tasks 派生、状态流转(pending→in_progress→completed/failed)、
execute_node 推进、事件层 task.update 映射。
"""
import asyncio
import unittest.mock as mock

from app.agent import tasks
from app.agent.events import _execute_event
from app.agent.nodes import execute


def test_build_tasks_derives_from_ready_tools():
    plan = [
        {"name": "search_flights", "params": {"city": "三亚"}, "ready": True},
        {"name": "search_hotels", "params": {"city": "三亚"}, "ready": False},  # 缺参不入
    ]
    t = tasks.build_tasks(plan)
    assert len(t) == 1
    assert t[0]["tool"] == "search_flights"
    assert t[0]["status"] == "pending"
    assert "三亚" in t[0]["desc"]


def test_mark_transitions_status():
    t = tasks.build_tasks([{"name": "search_flights", "params": {}, "ready": True}])
    tid = t[0]["id"]  # F2 唯一 id（{name}:{i}），不再裸用工具名
    t2 = tasks.mark(t, tid, "completed")
    assert t2[0]["status"] == "completed"
    assert t[0]["status"] == "pending"  # 不污染原引用


def test_execute_node_marks_completed_and_failed():
    plan = [
        {"name": "get_attractions", "params": {"city": "三亚"}, "ready": True},
        {"name": "get_weather", "params": {"city": "三亚"}, "ready": True},
    ]

    def fake_get_executor(name):
        fn = mock.AsyncMock()
        if name == "get_weather":
            fn.ainvoke.side_effect = RuntimeError("boom")
        else:
            fn.ainvoke.side_effect = lambda *a, **k: {"places": ["亚龙湾"]}
        return fn

    async def run():
        return await execute.execute_node({
            "tool_plan": plan,
            "conversation_id": "conv-1",
            "tasks": None,
        })

    with mock.patch.object(execute.tool_registry, "get_executor", new=fake_get_executor):
        out = asyncio.run(run())

    status_map = {x["tool"]: x["status"] for x in out["tasks"]}
    assert status_map["get_attractions"] == "completed"
    assert status_map["get_weather"] == "failed"
    assert out["task_failed"] == 1
    assert out["active_task_id"] == ""


def test_execute_event_emits_task_update():
    evs = _execute_event({"tool_results": {"x": {}}, "tasks": [{"id": "t1", "status": "completed"}]})
    types = [e.event_type for e in evs]
    assert "task.update" in types
    task_evt = [e for e in evs if e.event_type == "task.update"][0]
    assert task_evt.data["tasks"][0]["status"] == "completed"


def test_execute_event_omits_task_when_absent():
    evs = _execute_event({"tool_results": {"x": {}}})
    assert all(e.event_type != "task.update" for e in evs)