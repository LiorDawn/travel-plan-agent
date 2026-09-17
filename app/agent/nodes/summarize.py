"""外层 summarize 节点 — 各分支汇合后的最终回答（对应重构文档 §4.4~4.6）

统一为"对用户说话"的那一次 LLM，服务四条来源（复用同一节点）：
  - direct：general 闲聊（无工具、无攻略，只带历史记忆）
  - summarize：非闲聊无工具（带历史记忆 + 攻略）
  - execute→rag_manual→summarize：快查（带历史记忆 + 攻略 + 工具结果）
  - 缺参降级直出：问一次仍缺参时经本节点直出说明
"""
import json

from app.agent.prompts.summarize_prompt import SUMMARIZE_PROMPT
from app.agent.state import AgentState, last_user_text
from app.core.llm import llm
from app.core.logging import logger


# 工具结果 → 可读文本（F5：优先吃摘要 summary，无摘要才用原文 raw）
def _fmt_results(results: dict) -> str:
    if not results:
        return ""
    lines = []
    for name, r in results.items():
        if r.get("error"):
            lines.append(f"[{name}] 查询出错：{r['error']}")
        elif r.get("summary"):
            # 大结果已外部化：只进摘要，参考 artifact_ref 不展开
            lines.append(f"[{name}] {r['summary']}")
        else:
            lines.append(f"[{name}] {json.dumps(r.get('raw'), ensure_ascii=False)}")
    return "\n\n".join(lines)


async def summarize_node(state: AgentState) -> AgentState:
    """依据历史记忆 + 攻略 + 工具结果生成最终回答。"""
    user_query = last_user_text(state.get("messages", []))

    rag_memories = state.get("rag_memories", "")
    manual_memories = state.get("manual_memories", "")
    recent_context = state.get("recent_context", "")
    tool_results = _fmt_results(state.get("tool_results", {}))
    charts = state.get("charts", [])
    # F5：图表字段由 ArtifactStore 单独保留（不在 summary 里），从顶层 chart 抽取
    if tool_results:
        for r in state.get("tool_results", {}).values():
            chart = r.get("chart")
            if chart:
                charts.append(chart)
            elif isinstance(r.get("raw"), dict) and r["raw"].get("chart"):
                # 兼容未外部化的小结果包裹结构
                charts.append(r["raw"]["chart"])

    try:
        answer = await llm.chat(
            system_prompt=SUMMARIZE_PROMPT.format(
                recent_context=recent_context or "(无)",
                rag_memories=rag_memories or "(无)",
                manual_memories=manual_memories or "(无)",
                tool_results=tool_results or "(无)",
                impossible_reason=state.get("impossible_reason", "") or "(无)",
                user_message=user_query,
            ),
            user_message=user_query,
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("summarize_failed", error=str(e))
        answer = "抱歉，我暂时无法回答，请稍后再试。"

    return {**state, "answer": answer, "charts": charts}