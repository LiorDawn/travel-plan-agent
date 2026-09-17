"""外层 finalize 节点 — 所有分支汇合：统一输出 + 记忆沉淀（对应重构文档 §4.7 ⑤finalize）

透传 answer/charts（预留格式化扩展点）；把本次问答沉淀到知识库（conversation）；
沉淀逻辑统一走 rag_service.persist_conversation()，异常不阻塞主流程。
"""
from app.agent.state import AgentState, last_user_text
from app.core.config import get_settings
from app.core.logging import logger
from app.memory.rag.service import persist_conversation

_MIN_PERSIST_LEN = 20  # 回答足够长才沉淀，避免噪音


async def finalize_node(state: AgentState) -> AgentState:
    """统一输出 + 记忆沉淀。"""
    answer = state.get("answer", "")
    charts = state.get("charts", [])

    if get_settings().rag_enabled and len(answer) > _MIN_PERSIST_LEN:
        try:
            user_query = last_user_text(state.get("messages", []))
            await persist_conversation(user_query, answer,
                                       source_id=state.get("conversation_id"),
                                       user_id=state.get("user_id", "anonymous"))
        except Exception as e:
            logger.warning("rag_persist_failed", error=str(e))

    return {**state, "answer": answer, "charts": charts}