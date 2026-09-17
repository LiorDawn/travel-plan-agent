"""子图 ParamCheckMiddleware — 缺参自愈（对应重构文档 §8.1 / 复核 v3 收敛）

钩子：`after_model`（模型输出 tool_call 后、工具执行前）。
逻辑：检查模型发起的每个 tool_call 的必填参数是否齐全；缺失且未到上限时，不 interrupt，
而是注入"让模型自己补全"的提示（缺参自愈、不打断用户），已自愈轮次记在 state.ask_count。
达到上限则降级放行（让工具自身报错），不无限循环。子图不再 interrupt 打断——
严重缺参由外层 plan 缺参走 ask_user 统一澄清协议。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.subagent.middleware.schema import PlannerState
from app.core.logging import logger
from app.tools.registry import tool_registry


class ParamCheckMiddleware(AgentMiddleware[PlannerState]):
    """模型输出工具调用后校验必填参数；缺失则 interrupt 追问用户补充。"""

    state_schema = PlannerState

    def __init__(self, max_ask: int = 3) -> None:
        self.max_ask = max_ask

    @hook_config()
    def after_model(
        self,
        state: PlannerState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """校验最后一条 AIMessage 里的 tool_call 参数；缺参则 interrupt() 追问。"""
        messages = state.get("messages", [])
        if not messages:
            return None
        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return None

        ask_count = state.get("ask_count", 0) or 0
        # 找到第一个缺参的工具（一次处理一个，保证交互/自愈聚焦）
        for tc in last_msg.tool_calls:
            missing = tool_registry.missing_required(tc["name"], tc.get("args") or {})
            if not missing:
                continue

            # 自愈逻辑：未到追问上限前，不 interrupt，而是注入"让模型自己补全"的提示，
            # 促使模型在下一轮用对话中的信息补全参数重试（缺参自愈，不打断用户体验）。
            if ask_count < self.max_ask:
                ask_count += 1
                logger.info("param_check_self_heal", tool=tc["name"], missing=missing, ask=ask_count)
                hint = (
                    f"你调用的工具 {tc['name']} 缺少必要参数 {missing}。"
                    f"请结合本轮对话与上下文推断/补全这些参数，然后重新调用该工具。"
                    f"若确实无法补全，请明确告知用户缺少哪些信息，不要编造参数。"
                )
                return {
                    "messages": [HumanMessage(content=hint)],
                    "ask_count": ask_count,
                }

            # 已追问达到上限：直接放行，让工具自身校验报错（subagent 缺参自愈优先，
            # 不再用 interrupt 打断子图；严重缺口由外层 plan 缺参走 ask_user 统一澄清协议）
            logger.warning("param_check_degraded", tool=tc["name"], missing=missing, ask=ask_count)
            return None

        return None

    async def aafter_model(
        self,
        state: PlannerState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)