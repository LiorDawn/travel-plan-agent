"""外层 rag_manual 节点 — 关思考路径注入攻略知识（对应重构文档 §6 / §4.3）

execute 之后、summarize 之前调用；与子图 RAGInject 共用 rag_service.recall_manual()，
保证两处攻略检索逻辑一致。检索失败不阻塞（manual_memories=""）。
"""
from app.agent.state import AgentState, last_user_text
from app.core.logging import logger
from app.memory.rag.service import recall_manual


async def rag_manual_node(state: AgentState) -> AgentState:
    """检索攻略知识注入 manual_memories（仅供 summarize 使用）。"""
    user_query = last_user_text(state.get("messages", []))

    manual_memories = ""
    try:
        manual_memories = await recall_manual(user_query)
    except Exception as e:
        logger.warning("rag_manual_failed", error=str(e))

    return {**state, "manual_memories": manual_memories}