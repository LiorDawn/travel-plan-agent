"""条件路由 — plan 分流 + planner 中断路由 + ask_user 来源路由（对应重构文档 §3.6）

- route_after_plan：plan 后分流 → interrupt / direct / summarize / execute / react
- route_after_planner：子图完成后 → ask_user（缺参中断） / finalize（完成）
- route_after_ask_user：resume 后按来源 → plan（plan 缺参 / need_info） / planner_run（子图缺参）
"""
from app.agent.state import AgentState


def route_after_plan(state: AgentState) -> str:
    """plan 后的分流（v2.0：语义自评优先，删除 clarify 分支）。优先级：
    ① not_doable → summarize（说明无法完成 + 询问，不弹窗）
    ② need_info 首轮 → interrupt（去 ask_user，source=need_info 弹窗收集 needs）
    ③ 已追问过仍缺（need_info 二次 / missing 二次）→ 降级 summarize（不二次打断）
    ④ 未问过且机械缺参 → interrupt（source=plan 兜底补漏判字段）
    ⑤ general → direct；⑥ 无工具 → summarize；⑦ 有齐参工具 → execute/react
    """
    intent = state.get("intent", "query")
    tool_plan = state.get("tool_plan", [])
    missing_params = state.get("missing_params", [])
    ask_count = state.get("ask_count", 0) or 0
    feasibility = state.get("feasibility", "feasible")

    # ① 模型判定做不了 → 直出说明 + 询问（不弹窗）
    if feasibility == "not_doable":
        return "summarize"
    # ② need_info 首轮 → 弹窗收集 needs（语义缺口 > 机械字段缺漏）
    if feasibility == "need_info" and not state.get("need_info_asked") and ask_count == 0:
        return "interrupt"
    # ③ 已追问过但仍缺（语义 need_info 二次 / 机械 missing 二次）→ 降级直出，不二次打断
    if (feasibility == "need_info" and ask_count >= 1) or (missing_params and ask_count >= 1):
        return "summarize"
    # ④ 未问过且机械缺参 → 兜底 interrupt（补漏判字段）
    if missing_params:
        return "interrupt"
    # ⑤ 闲聊
    if intent == "general":
        return "direct"
    # ⑥ 无工具
    ready_tools = [t for t in tool_plan if t.get("ready")]
    if not ready_tools:
        return "summarize"
    # ⑦ 有齐参工具
    if intent == "query":
        return "execute"
    if intent == "travel" and not state.get("thinking_mode", False):
        return "execute"
    if intent == "travel":  # 开思考
        return "react"
    # 其它（query 已覆盖 / 兜底）
    return "execute"


def route_after_planner(state: AgentState) -> str:
    """planner_run 之后：子图中断(缺参) → ask_user 挂起外层发追问；正常 → finalize。

    v2.0 备注：子图 ParamCheck 已改自愈（缺参注入提示让模型补、达上限放行），当前子图
    不会 interrupt → 本分支为防御性保留（未来 F6 委派/新中间件若 reintroduce interrupt 生效）。
    """
    if state.get("subagent_interrupt"):
        return "ask_user"
    return "finalize"


def route_after_ask_user(state: AgentState) -> str:
    """ask_user resume 后按来源路由：plan 缺参 / need_info 收集完 → 回 plan 重自评；
    子图缺参 → 回 planner_run 续跑。

    v2.0 备注："planner_run" 回跳与 route_after_planner 的 ask_user 分支对应，
    同样为防御性保留（子图当前不 interrupt）。子图中断路径已死代码化，仅预留不删。
    """
    if state.get("interrupt_source") in ("plan", "need_info"):
        return "plan"
    return "planner_run"