"""外层 planner_run 节点 — 包装 create_agent 子图（对应重构文档 §4.5）

travel 分支：调用子图执行 ReAct 多步工具。子图独立 thread_id（conversation_id + _planner）。
- 首次：ainvoke(sub_input)
- resume：Command(resume=user_msg) 从断点续跑
- 子图中断(缺参) → 透传 subagent_interrupt，外层路由 END（前端 ask_user）
- 完成 → 从子图 messages 提取最终 AI 回答
"""
from langgraph.types import Command

from app.agent.state import AgentState, last_user_text
from app.core.config import get_settings
from app.core.logging import logger
from app.core.llm import llm
from langchain_core.messages import AIMessage
import asyncio


def _sub_config(conversation_id: str, suffix: str = "planner") -> dict:
    """构造子图独立 thread_id（suffix 区分主/委派子图，避免与外层/彼此 checkpoint 冲突）。"""
    return {"configurable": {"thread_id": f"{conversation_id}_{suffix}"}}


def _extract_final_answer(sub_result) -> str:
    """从子图输出 messages 取最后一条 AI 消息作为最终回答。"""
    messages = sub_result.get("messages", []) if isinstance(sub_result, dict) else []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _check_interrupt(subgraph, config: dict) -> dict | None:
    """读取子图最新 checkpoint；停在 interrupt 则返回中断信息，否则 None。"""
    try:
        graph_state = subgraph.get_state(config)
        if graph_state and graph_state.interrupts:
            return graph_state.interrupts[0].value
    except Exception as e:
        logger.warning("subagent_interrupt_read_failed", error=str(e))
    return None


async def planner_run_node(state: AgentState) -> AgentState:
    """调用 create_agent 子图执行复杂规划；F6 开启且可分派时委派给多专业子 Agent。"""
    from app.agent.subagent.factory import subagent

    conversation_id = state.get("conversation_id", "")
    user_query = state.get("resume_input") or last_user_text(state.get("messages", []))

    # F6：开关开启、travel、delegation_plan 来自 plan 且 ≥2 角色 → 走多子 Agent 委派。
    # resume 时若已有 delegation_plan 则按快照对缺失角色续跑；无委派计划一律走单例。
    try:
        settings = get_settings()
        delegation_plan = state.get("delegation_plan") or []
        if getattr(settings, "multi_agent_enabled", False) \
                and state.get("intent") == "travel" \
                and len(delegation_plan) >= 2:
            delegated = await _delegate_and_aggregate(state, delegation_plan)
            if delegated:
                return {**state, **delegated}
    except Exception as e:
        logger.warning("multi_agent_delegate_failed", error=str(e))  # 委派失败降级走单例

    config = _sub_config(conversation_id)

    # 子图对话记忆：短期窗口(本会话最近轮次) + 长期 RAG 记忆 都注入首条输入，
    # 使"按上次预算/上次那家酒店"类约束进入行程内容生成（resume 时沿用 checkpoint 中已注入的消息）。
    memory_block = _memory_block(state)

    # 构造子图输入（只传子图需要的字段）
    sub_input = {
        "messages": [{"role": "user", "content": f"{memory_block}\n\n当前用户需求：{user_query or ''}"}],
        "user_id": state.get("user_id", "anonymous"),
        "conversation_id": conversation_id,
        "ask_count": 0,
    }

    try:
        if state.get("resume_input"):
            sub_result = await subagent.ainvoke(Command(resume=user_query), config)
        else:
            sub_result = await subagent.ainvoke(sub_input, config)
    except Exception as e:
        logger.warning("planner_run_failed", error=str(e))
        return {**state, "answer": "抱歉，行程规划执行出错了，请稍后再试。"}

    answer = _extract_final_answer(sub_result) or _final_fallback(subagent, config)
    interrupt_data = _check_interrupt(subagent, config)

    # 记录子图 LLM 用量（复用外层累计，保证 done 事件 token 可见）
    _sync_usage(sub_result)

    return {**state, "answer": answer, "subagent_interrupt": interrupt_data,
            "interrupt_source": "subagent" if interrupt_data else state.get("interrupt_source")}


async def _delegate_and_aggregate(state: AgentState, delegation_plan: list[dict]) -> dict | None:
    """F6 委派：按 plan 产出的角色计划，并行把独立业务域子任务分给专业子 Agent。

    委派角色由 plan 的 delegation_plan 提供（不再运行时按工具猜）。每个子 Agent 独立
    thread_id 存快照：resume 时只对缺失结论的角色续跑（Command(resume)），已成功的不重跑。
    主上下文只收各子 Agent 的 conclusion；聚合结论一次性落库。返回聚合后的状态片段；
    不满足或失败则返回 None（调用方降级走单例）。
    """
    from app.agent.subagent.factory import get_role_subagent

    conversation_id = state.get("conversation_id", "")
    memory_block = _memory_block(state)
    user_query = state.get("resume_input") or last_user_text(state.get("messages", []))

    roles = [d.get("role") for d in delegation_plan if d.get("role")]
    if len(roles) < 2:
        return None

    cfg_role = lambda role: _sub_config(conversation_id, suffix=f"{role}")
    results: dict[str, dict] = {}

    async def _run_role(role: str) -> tuple[str, dict]:
        sub = get_role_subagent(role)
        cfg = cfg_role(role)
        prompt_hint = f"你是【{role}】专业子 Agent。当前整体需求：{user_query}"
        sub_input = {"messages": [{"role": "user", "content": f"{memory_block}\n\n{prompt_hint}"}],
                     "user_id": state.get("user_id", "anonymous"),
                     "conversation_id": conversation_id, "ask_count": 0}
        # resume 续跑：已有该角色结论则跳过；仅对结论为空/失败的角色续跑快照
        existing = state.get("delegation") or {}
        if existing.get(role) == "success":
            prior = (state.get("delegation_conclusions") or {}).get(role, "")
            return role, {"status": "success", "conclusion": prior}
        try:
            if state.get("resume_input"):
                sub_result = await sub.ainvoke(Command(resume=user_query), cfg)
            else:
                sub_result = await sub.ainvoke(sub_input, cfg)
            sub_answer = _extract_final_answer(sub_result)
            return role, {"status": "success", "conclusion": sub_answer or f"{role} 未产出结论"}
        except Exception as e:
            logger.warning("subagent_role_failed", role=role, error=str(e))
            return role, {"status": "failed", "conclusion": "", "error": str(e)}

    # 并行委派（并发 ≤3，与单次委派域数一致）；单角色失败不阻塞其它角色
    done = await asyncio.gather(*(_run_role(r) for r in roles), return_exceptions=True)
    for item in done:
        if isinstance(item, Exception):
            logger.warning("subagent_role_gather_failed", error=str(item))
            continue
        role, res = item
        results[role] = res

    # 主 Agent 聚合各子结论
    aggregate_answer = await _aggregate(state, results)

    # 聚合结论一次性落库（失败仅告警，不阻塞回答）
    try:
        from app.core.database import AsyncSessionLocal
        from app.memory import long_term
        async with AsyncSessionLocal() as db:
            await long_term.save_travel_plan(db, {
                "user_id": state.get("user_id", "anonymous"),
                "conversation_id": conversation_id,
                "destination": user_query[:30],
                "title": "多子 Agent 委派方案",
                "budget": {"total": 0},
                "plan_data": {"roles": {r: res_.get("status") for r, res_ in results.items()},
                              "answer": aggregate_answer},
            })
    except Exception as e:
        logger.warning("delegation_persist_failed", error=str(e))

    prior_conclusions = dict(state.get("delegation_conclusions") or {})
    prior_conclusions.update({r: res.get("conclusion", "") for r, res in results.items()})
    return {
        "answer": aggregate_answer,
        "delegation": {role: r.get("status") for role, r in results.items()},
        "delegation_conclusions": prior_conclusions,
    }


async def _aggregate(state, results: dict) -> str:
    """主 Agent 一次 LLM 聚合各子结论为最终回答。失败兜底拼纯文本。"""
    if not results:
        return "抱歉，子任务均未成功，暂无法聚合方案。"
    try:
        blocks = "\n\n".join(
            f"[{role}] {r.get('conclusion', '')}" for role, r in results.items()
        )
        return await llm.chat(
            system_prompt="你是主规划 Agent：请把下列各专业子 Agent 的结论整合成一份简明、完整的赴外地行程方案，"
                          "不要臆造子 Agent 未提供的数据。",
            user_message=blocks[: 3000],
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("aggregate_failed", error=str(e))
        return "\n".join(r.get("conclusion", "") for r in results.values())


def _memory_block(state: AgentState) -> str:
    """拼子图可感知的对话记忆块（短期窗口 + 长期会话偏好 + 用户澄清答案）。"""
    recent = state.get("recent_context", "") or ""
    rag = state.get("rag_memories", "") or ""
    clarifications = state.get("clarifications", {}) or {}
    parts = []
    if recent:
        parts.append("【最近对话】\n" + recent)
    if rag:
        parts.append("【历史记忆（长期偏好）】\n" + rag)
    if clarifications:
        lines = "\n".join(f"- {q}: {a}" for q, a in clarifications.items())
        parts.append("【用户澄清（必须遵守的确定约束，如预算/天数/出发地）】\n" + lines)
    if not parts:
        return ""
    return "（背景记忆，帮助理解上下文，围绕当前需求作答）\n" + "\n\n".join(parts)


def _final_fallback(subgraph, config) -> str:
    return _check_interrupt(subgraph, config) or "行程生成中，请稍候。"


def _sync_usage(sub_result) -> None:
    """尝试把子图调用累计进外层 usage（子图内部可能未走 llm.usage 统计）。"""
    raw = sub_result.get("raw") if isinstance(sub_result, dict) else None
    if raw is not None:
        llm._accumulate_usage(raw)  # noqa: SLF001