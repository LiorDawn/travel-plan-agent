"""子图 RAGInjectMiddleware — 攻略知识注入（对应重构文档 §6 / §8.2）

钩子：`awrap_model_call`（每次模型调用前）。
逻辑：取最后一条 user 消息 → `rag_service.recall_manual()`只查攻略知识 →
注入 system_message → 调用 handler。检索失败不阻塞，直接 handler。
与关思考路径的 rag_manual 节点共用同一个 rag_service.recall_manual()，策略一致。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import SystemMessage

from app.agent.subagent.middleware.schema import PlannerState
from app.core.config import get_settings
from app.core.logging import logger
from app.memory.rag.service import recall_manual


class RAGInjectMiddleware(AgentMiddleware[PlannerState]):
    """攻略知识注入 —— 会话级：首检成功后缓存 content，后续模型调用直接复用。

    不做"每次模型调用都检索"（那是 10~15 次检索 + 屡次改写/embedding 开销的根源）。
    首个模型调用前检索一次并写入 state.rag_manual_content；后续调用直接拿缓存拼 system_message，
    直到本会话结束。检索失败不阻塞，直接 handler。
    """

    state_schema = PlannerState

    async def awrap_model_call(self, request, handler) -> ModelResponse:
        """包装模型调用：优先复用 state 已缓存攻略，否则首检并回填缓存再注入。"""
        try:
            # 会话级注入开关不在此判断；开关关闭时由 state 无缓存促使每次走 recall_manual。
            use_session = getattr(get_settings(), "rag_session_inject", True)
            state = request.state if hasattr(request, "state") else None
            cached = None
            if use_session and state is not None:
                cached = state.get("rag_manual_content") or ""

            if cached:
                inject = SystemMessage(
                    content=f"\n\n【参考攻略知识】\n{cached}\n（以上为检索到的攻略，用于组织回答，若与用户问题无关可忽略）"
                )
            else:
                user_query = _last_user_text(request.messages)
                if not user_query:
                    return await handler(request)
                context = await recall_manual(user_query)
                if not context:
                    return await handler(request)
                inject = SystemMessage(
                    content=f"\n\n【参考攻略知识】\n{context}\n（以上为检索到的攻略，用于组织回答，若与用户问题无关可忽略）"
                )
                # 会话级回填：本会话后续模型调用不再重新检索
                if use_session and state is not None:
                    state["rag_manual_content"] = context
                    state["rag_manual_cached"] = True

            if request.system_message:
                request.system_message.content += inject.content
            else:
                request.system_message = SystemMessage(content=inject.content)
        except Exception as e:
            logger.warning("rag_inject_failed", error=str(e))
        return await handler(request)


def _last_user_text(messages) -> str:
    """从消息列表取最后一条 user 消息文本。"""
    from langchain_core.messages import HumanMessage
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""