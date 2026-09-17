"""Agent 事件协议 — ChatEvent 与节点→事件映射（SSE 统一出口；对应重构文档 §3/§12）

节点名常量集中在此（单一来源），供 graph.py 构建图与 runner 事件映射共用。
事件类型命名规则 = {节点名}.{阶段}。
"""
from typing import Callable

from pydantic import BaseModel, Field


class ChatEvent(BaseModel):
    event_type: str = Field(..., description="事件类型: plan|tool_execution|message|ask_user|chart|done")
    data: dict = Field(default_factory=dict, description="事件数据")


# ---------- 节点名（图构建 + 事件映射共用，防漂移） ----------
NODE_RAG_RECALL = "rag_recall"
NODE_PLAN = "plan"
NODE_EXECUTE = "execute"
NODE_RAG_MANUAL = "rag_manual"
NODE_SUMMARIZE = "summarize"
NODE_PLANNER_RUN = "planner_run"
NODE_ASK_USER = "ask_user"
NODE_FINALIZE = "finalize"


# ---------- 事件类型 ----------
EVT_RAG_MEMORIES      = "rag.memories"
EVT_PLAN_PROGRESS     = "plan.progress"
EVT_PLAN_RESULT       = "plan.result"
EVT_TOOL_RESULT       = "tool_execution.result"
EVT_PLANNER_PROGRESS  = "planner.progress"
EVT_ANSWER_MESSAGE    = "message"          # summarize / finalize 统一文本
EVT_TASK_UPDATE       = "task.update"      # F2 任务清单状态变更（长任务进度）
# 系统级
EVT_ASK_USER = "ask_user"
EVT_DONE = "done"


# ---------- 节点 → 事件映射（单表驱动 start/end） ----------
def _pick_event(evt_type: str, **defaults) -> Callable[[dict], list[ChatEvent]]:
    """按字段白名单构造结果事件（未写入字段回退默认值）。"""
    def _build(output: dict) -> list[ChatEvent]:
        return [ChatEvent(event_type=evt_type,
                          data={k: output.get(k, d) for k, d in defaults.items()})]
    return _build


def _answer_event(output: dict) -> list[ChatEvent]:
    """summarize/finalize 结果：文本回答 + 可选图表。"""
    events = [ChatEvent(event_type=EVT_ANSWER_MESSAGE, data={
        "message": output.get("answer", ""),
        "rag_hits": output.get("rag_hits", 0),
    })]
    if output.get("charts"):
        events.append(ChatEvent(event_type="chart", data={"charts": output["charts"]}))
    return events


def _execute_event(output: dict) -> list[ChatEvent]:
    """execute 结束：工具结果 + F2 任务清单状态（task.update 便于前端展示长任务进度）。"""
    events = [ChatEvent(event_type=EVT_TOOL_RESULT,
                        data={"tool_results": output.get("tool_results", {})})]
    if output.get("tasks"):
        # 保留 depends_on/sub_agent_type 供前端展示任务依赖与归属角色
        tasks = [{"id": t["id"], "tool": t.get("tool", ""), "desc": t.get("desc", ""),
                  "status": t.get("status"), "depends_on": t.get("depends_on") or [],
                  "sub_agent_type": t.get("sub_agent_type")}
                 for t in output["tasks"]]
        events.append(ChatEvent(event_type=EVT_TASK_UPDATE, data={"tasks": tasks}))
    return events


# node -> (开始进度(事件类型, 文案) 或 None, 结束结果构造)
_NODE_EVENTS: dict[str, tuple[tuple[str, str] | None, Callable[[dict], list[ChatEvent]]]] = {
    NODE_RAG_RECALL: ((EVT_RAG_MEMORIES, "recalling"), lambda o: []),
    NODE_PLAN:        ((EVT_PLAN_PROGRESS, "analyzing"),
                       _pick_event(EVT_PLAN_RESULT, intent="", intent_confidence=0)),
    NODE_EXECUTE:     ((EVT_TOOL_RESULT, "running"), _execute_event),
    NODE_PLANNER_RUN: ((EVT_PLANNER_PROGRESS, "planning"), lambda o: []),
    NODE_SUMMARIZE:   (None, lambda o: []),      # 广播收敛到 finalize，避免非 react 分支重复 message
    NODE_FINALIZE:    (None, _answer_event),
}