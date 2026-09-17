"""外层 LangGraph Agent 全局状态（对应重构文档 §5.2）

外层负责流程编排（rag 检索、plan 分流、execute/子图执行、汇总、沉淀）；
子图内部 ReAct 执行用独立的 PlannerState（见 middleware/schema.py）。
"""
from typing import Any, Dict, List, TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired


def last_user_text(messages: List[dict]) -> str:
    """取消息列表中最后一条的文本内容（LangGraph 已将 dict 转成 Message 对象）。

    多个节点（plan/summarize/execute/rag_*/finalize）共用，避免各自重复「取最后一条」逻辑。
    """
    last = messages[-1] if messages else None
    return last.content if hasattr(last, "content") else str(last)


class AgentState(TypedDict):
    """外层图状态：节点间传递的编排数据。"""

    # ---- 基础 ----
    messages: Annotated[List[dict], add_messages]
    user_id: str
    conversation_id: str
    thinking_mode: bool                     # 前端思考开关：关→execute 快查 / 开→react 子图

    # ---- 外层 RAG ----
    rag_memories: str                       # 历史对话记忆（plan 前 rag_recall 注入）
    rag_hits: int
    manual_memories: str                    # 攻略知识（rag_manual 注入，关思考→summarize）
    recent_context: NotRequired[str]        # 短期窗口：最近 N 轮原文（保指代/话题延续，不参与检索）

    # ---- plan 分流 ----
    intent: str                             # general / query / travel
    tool_plan: List[dict]                   # plan 选出工具: [{name, params, ready}]
    missing_params: List[dict]              # plan 缺参工具: [{tool, fields}]
    # ---- 语义可完成性自评（v3 need/澄清体系）----
    feasibility: NotRequired[str]           # feasible / need_info / not_doable（模型自评）
    needs: NotRequired[List[dict]]          # need_info 时需补充: [{key,question,type,options,required}]
    impossible_reason: NotRequired[str]     # not_doable 时无法完成原因 + 需补什么
    nudge: NotRequired[str]                 # 可行路径下可选引导语（最终以 summarize 产出为准）
    need_info_asked: NotRequired[bool]      # 是否已发出 need_info 弹窗（状态可观测）

    # ---- 缺参追问 ----
    ask_count: int                          # 已追问次数（最多 1 次）
    interrupt_source: NotRequired[str]      # 'plan' | 'subagent' | 'need_info'，ask_user 路由依据
    subagent_interrupt: NotRequired[dict]   # 子图中断信息（透传前端 ask_user）

    # ---- 澄清询问（needs 弹窗收集，答案回灌 plan 抽参）----
    questions: NotRequired[list]            # 本次弹窗的问题 list[{id,question,type,options}]（供前端/回灌）
    clarifications: NotRequired[dict]       # 用户回答 {key: answer}，回灌 plan 抽参 + 子图

    # ---- 分支产出 ----
    answer: str
    charts: List[dict]
    tool_results: Dict[str, Any]            # execute 并行结果

    # ---- F6 委派计划（plan 产出，替代运行时按工具猜测）----
    delegation_plan: NotRequired[List[dict]]  # [{role, tool_hints}]，multi_agent 开启且 travel 时非空

    # ---- F2 任务清单（plan 源头产出，execute 优先消费）----
    plan_tasks: NotRequired[List[Dict[str, Any]]]  # [{id, tool, desc, depends_on, sub_agent_type}]

    # ---- F2 任务清单状态机 ----
    tasks: NotRequired[List[Dict[str, Any]]]  # 多步骤任务清单: [{id, tool, desc, status, updated_at}]
    active_task_id: NotRequired[str]          # 当前正在执行的任务 id

    # ---- 续跑 ----
    resume_input: NotRequired[str]          # ask_user resume 后用户补充，回灌 plan/子图