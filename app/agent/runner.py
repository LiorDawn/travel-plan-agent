"""TravelPlanAgent — 外层编排运行器（run/resume + 流式 + 中断检测；对应重构文档 §3）

从原 graph.py 拆出：只负责"如何运行已编译图"，图结构声明（add_node/add_edge）留在 graph.py。
"""
import uuid
from typing import AsyncIterator

from langgraph.types import Command

from app.agent.events import (
    ChatEvent,
    EVT_ASK_USER,
    EVT_DONE,
    _NODE_EVENTS,
)
from app.agent.graph import _compiled_graph
from app.agent.state import AgentState
from app.core.llm import llm
from app.core.logging import logger
from app.core.tracing import start_span


def _new_state(messages: list[dict], user_id: str, conversation_id: str,
               thinking_mode: bool, recent_context: str = "") -> AgentState:
    """初始化一次 run 的状态白板。"""
    return {
        "messages": messages,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "thinking_mode": thinking_mode,
        "rag_memories": "", "rag_hits": 0, "manual_memories": "",
        "recent_context": recent_context,
        "intent": "", "intent_confidence": 0.0,
        "tool_plan": [], "missing_params": [],
        "ask_count": 0,
        "questions": [], "clarifications": {},
        "answer": "", "charts": [], "tool_results": {},
    }


class TravelPlanAgent:
    """外层 LangGraph 编排 — 五分流 + 缺参中断透传 + checkpoint 增量续跑"""

    async def _stream_events(self, input_data, config: dict) -> AsyncIterator[ChatEvent]:
        """消费 astream_events：start 推进度，end 推结果（策略表驱动）。

        同时承载 OTel 链路：每个节点是一个 span（on_chain_start 开、on_chain_end 关），
        让 traced 校验能看到 plan→execute→summarize 的完整调用链与耗时。
        """
        open_spans: dict[str, object] = {}
        async for event in _compiled_graph.astream_events(input_data, config, version="v2"):
            start = event.get("event") == "on_chain_start"
            if not start and event.get("event") != "on_chain_end":
                continue
            name = event.get("name", "")
            spec = _NODE_EVENTS.get(name)
            if spec is None:
                continue
            if start:
                open_spans[name] = start_span(f"agent.node.{name}",
                                              conversation_id=config.get("configurable", {}).get("thread_id", ""))
                progress = spec[0]
                if progress:
                    yield ChatEvent(event_type=progress[0], data={"status": progress[1]})
            else:
                span = open_spans.pop(name, None)
                if span is not None:
                    span.set_attr("status", "ok")
                    span.end()
                for chat_event in spec[1](event.get("data", {}).get("output", {})):
                    yield chat_event

    def _interrupt(self, config: dict) -> dict | None:
        """读最新 checkpoint；停在 interrupt 则返回中断信息，否则 None。"""
        try:
            graph_state = _compiled_graph.get_state(config)
            if graph_state and graph_state.interrupts:
                return graph_state.interrupts[0].value
        except Exception:
            pass
        return None

    def _finish(self, conversation_id: str, usage: dict,
                interrupt: dict | None, error: Exception | None = None) -> list[ChatEvent]:
        """统一收尾：中断/异常 → ask_user + done(need_input, success=false)；正常 → done(success=true)。"""
        success = not (interrupt or error)
        done_data: dict = {"conversation_id": conversation_id, "usage": usage,
                           "success": bool(success)}
        if interrupt:
            done_data["need_input"] = True
            return [ChatEvent(event_type=EVT_ASK_USER, data=interrupt),
                    ChatEvent(event_type=EVT_DONE, data=done_data)]
        return [ChatEvent(event_type=EVT_DONE, data=done_data)]

    async def run(self, user_message: str, user_id: str = "anonymous",
                  conversation_id: str = "", thinking_mode: bool = False,
                  recent_context: str = "") -> AsyncIterator[ChatEvent]:
        """新对话：初始化状态 → 执行全图 → 检测中断 → 收尾。"""
        conversation_id = conversation_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": conversation_id}}
        state = _new_state([{"role": "user", "content": user_message}],
                           user_id, conversation_id, thinking_mode, recent_context)

        logger.info("agent_start", user_id=user_id, conv=conversation_id)
        llm.reset_usage()

        async for event in self._stream_events(state, config):
            yield event

        usage = llm.get_usage()
        interrupt = self._interrupt(config)
        logger.info("agent_interrupted" if interrupt else "agent_done", conv=conversation_id)
        for event in self._finish(conversation_id, usage, interrupt):
            yield event

    async def resume(self, user_message: str, conversation_id: str) -> AsyncIterator[ChatEvent]:
        """续跑：从 checkpoint 断点恢复，注入用户补充，再次检测中断。"""
        config = {"configurable": {"thread_id": conversation_id}}

        logger.info("agent_resume", conv=conversation_id)
        llm.reset_usage()

        async for event in self._stream_events(Command(resume=user_message), config):
            yield event

        usage = llm.get_usage()
        interrupt = self._interrupt(config)
        logger.info("agent_interrupted_again" if interrupt else "agent_resume_done",
                    conv=conversation_id)
        for event in self._finish(conversation_id, usage, interrupt):
            yield event