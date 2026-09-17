"""子图 PlannerState — 内层 create_agent 的自定义 state（对应重构文档 §5.3）

create_agent 默认 state 只含 messages；自定义字段通过中间件的 `state_schema` 声明，
create_agent 会自动合并所有中间件的 state_schema，因此多个中间件共用同一个
PlannerState 不冲突（各声明各用，合并后字段齐全）。
"""
from typing import Any
from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class PlannerState(AgentState):
    """子图自定义 state：ReAct 执行 + 缺参追问统计 + RAG 命中可见性。"""

    # 透传（与业务有关，供工具/中间件使用）
    user_id: NotRequired[str]
    conversation_id: NotRequired[str]

    # ParamCheckMiddleware 用：本轮已追问次数
    ask_count: NotRequired[int]

    # RAGInjectMiddleware 用：检索命中条数（流程可见性）
    rag_hits: NotRequired[int]

    # RAGInjectMiddleware 会话级注入：首检成功后缓存攻略 content，后续模型调用直接复用，
    # 避免子图每一步都重新检索（10+ 次 → 1~2 次）。
    rag_manual_content: NotRequired[str]
    rag_manual_cached: NotRequired[bool]

    # 通用扩展位（预留，供后续中间件放数据）
    _extra: NotRequired[Any]