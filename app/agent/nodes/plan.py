"""外层 plan 节点 — 分类 + 选工具 + 抽参数 + 缺参判定（对应重构文档 §4.2 ①plan）

一次 LLM 输出 PlanOutput：intent / tools / reason。
- 对 tools 逐项比对必填参数（取自 ToolRegistry），缺则记入 missing_params。
- 一旦有缺参且 ask_count==0 → 设 interrupt_source='plan'，让 outer interrupt 一次性追问。
- resume 重进本节点时 ask_count>0，若仍缺参则保留 missing_params 走降级 summarize。
"""
from app.agent.prompts.plan_prompt import PLAN_PROMPT
from app.agent.state import AgentState, last_user_text
from app.core.config import get_settings
from app.core.llm import llm
from app.core.logging import logger
from app.schemas.agent_plan import PlanOutput, PlanTool
from app.tools.registry import tool_registry


async def plan_node(state: AgentState) -> AgentState:
    """一次 LLM 完成分类选参抽参；产出 tool_plan 与 missing_params。"""
    user_query = last_user_text(state.get("messages", []))
    rag_memories = state.get("rag_memories", "")
    recent_context = state.get("recent_context", "")
    ask_count = state.get("ask_count", 0) or 0
    clarifications = state.get("clarifications", {}) or {}

    intent, confidence = "query", 0.5
    tools_out: list[PlanTool] = []
    try:
        result = await llm.chat_agent_structured(
            system_prompt=PLAN_PROMPT.format(
                tools_desc=tool_registry.get_tools_desc(),
                recent_context=recent_context or "(无)",
                rag_memories=rag_memories or "(无)",
                clarifications=_fmt_clarifications(clarifications),
                user_message=user_query,
            ),
            user_message=user_query,
            output_model=PlanOutput,
            max_tokens=400,
        )
        intent = result.get("intent", "query")
        confidence = result.get("confidence", 0.5)
        tools_out = result.get("tools", [])
        plan_tasks_out = result.get("tasks", [])
        delegation_out = result.get("delegation", [])
        feasibility = result.get("feasibility", "feasible")
        needs_out = result.get("needs") or []
        impossible_reason = result.get("impossible_reason", "")
        nudge_out = result.get("nudge", "")
    except Exception as e:
        logger.warning("plan_failed", error=str(e))
        plan_tasks_out, delegation_out, needs_out = [], [], []
        feasibility, impossible_reason, nudge_out = "feasible", "", ""

    # 构建 tool_plan 并检查缺参
    tool_plan: list[dict] = []
    missing_params: list[dict] = []
    for tp in tools_out:
        name = tp.get("name") if isinstance(tp, dict) else getattr(tp, "name", "")
        params = (tp.get("params") if isinstance(tp, dict) else getattr(tp, "params", {})) or {}
        missing = tool_registry.missing_required(name, params)
        tool_plan.append({
            "name": name,
            "params": dict(params),
            "ready": not missing,            # 参数齐 → execute 可执行
            "missing": missing,              # 缺参清单
        })
        if missing:
            missing_params.append({"tool": name, "fields": missing})

    # 语义自评：need_info 时收集 needs（保留 required 标记，供弹窗/回灌）
    needs: list[dict] = [_need_to_dict(n) for n in needs_out]

    # 缺参/缺信息且未追问过 → 标记需中断来源，让外层 ask_user 一次性弹窗
    # 优先级：模型自评 need_info 优先于机械 missing 兜底（可行性缺口 > 字段缺漏）
    interrupt_source = state.get("interrupt_source")
    if missing_params and ask_count == 0:
        interrupt_source = "plan"
    if feasibility == "need_info" and not state.get("need_info_asked") and ask_count == 0:
        interrupt_source = "need_info"

    # F6：仅多 Agent 开启且 travel 时才把模型输出的委派计划透传；否则空列表。
    multi_agent_enabled = getattr(get_settings(), "multi_agent_enabled", False)
    delegation_plan = [d.model_dump() if not isinstance(d, dict) else d
                       for d in delegation_out] if \
        (multi_agent_enabled and intent == "travel") else []

    # F2：plan 源头产出任务清单（仅保留有效 id+tool；无效项由 execute 用 tool_plan 兜底）。
    plan_tasks = [
        {"id": t.get("id"), "tool": t.get("tool"), "desc": t.get("desc", ""),
         "depends_on": t.get("depends_on") or [], "sub_agent_type": t.get("sub_agent_type")}
        for t in plan_tasks_out
        if (t.get("id") if isinstance(t, dict) else getattr(t, "id", None))
        and (t.get("tool") if isinstance(t, dict) else getattr(t, "tool", None))
    ]

    return {**state,
            "intent": intent,
            "intent_confidence": confidence,
            "tool_plan": tool_plan,
            "missing_params": missing_params,
            "delegation_plan": delegation_plan,
            "plan_tasks": plan_tasks,
            "feasibility": feasibility,
            "needs": needs,
            "impossible_reason": impossible_reason,
            "nudge": nudge_out,
            "ask_count": ask_count,
            "interrupt_source": interrupt_source}


def _need_to_dict(n) -> dict:
    """把 PlanNeedField / dict 统一转成 dict（带回 required 标记）。"""
    if isinstance(n, dict):
        return {
            "key": n.get("key", ""), "question": n.get("question", ""),
            "type": n.get("type", "text"), "options": n.get("options") or [],
            "required": n.get("required", True),
        }
    return {
        "key": getattr(n, "key", ""), "question": getattr(n, "question", ""),
        "type": getattr(n, "type", "text"), "options": getattr(n, "options", None) or [],
        "required": getattr(n, "required", True),
    }


def _fmt_clarifications(clarifications: dict) -> str:
    """把澄清答案 dict 格式化成 '' 或 '问题id: 回答\n...' 供 prompt 消费。"""
    if not clarifications:
        return "(无)"
    return "\n".join(f"{k}: {v}" for k, v in clarifications.items())