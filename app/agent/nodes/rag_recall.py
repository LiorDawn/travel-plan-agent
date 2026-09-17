"""外层 rag_recall 节点 — 检索历史对话记忆（对应重构文档 §4.1 ①rag_recall）

只检索 conversation（外层所有分支前置，帮助意图分类 + 指代消解），不查攻略。
逻辑统一收敛到 rag_service.recall_conversation()，检索失败不阻塞（rag_memories=""）。
"""
from app.agent.state import AgentState, last_user_text
from app.core.logging import logger
from app.memory.rag.service import recall_conversation


async def rag_recall_node(state: AgentState) -> AgentState:
    """对外层所有分支前置：只查对话历史记忆。"""
    messages = state.get("messages", [])
    if not messages:
        return {**state, "rag_memories": "", "rag_hits": 0}
    user_query = last_user_text(messages)

    rag_memories, rag_hits = "", 0
    try:
        rag_memories = await recall_conversation(user_query, user_id=state.get("user_id", "anonymous"))
        rag_hits = rag_memories.count("\n\n[") + (1 if rag_memories else 0)  # 命中块数近似
    except Exception as e:
        logger.warning("rag_recall_failed", error=str(e))

    return {**state, "rag_memories": rag_memories, "rag_hits": rag_hits}