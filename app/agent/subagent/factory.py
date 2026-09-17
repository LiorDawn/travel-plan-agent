"""create_agent 子图工厂 — 复杂旅行规划（travel）的 ReAct 多步执行 + F6 多子 Agent 委派

默认提供单一 `subagent`（多到 travel 全量工具）；F6 开启时可由 create_subagent(role)
参数化创建专业子图（transit/attraction/budget），委派后由主 itinerary 聚合。
每个子图独立 thread_id（conversation_id + role 后缀），避免与外层 checkpoint 冲突。

缺参追问由 ParamCheckMiddleware 通过 interrupt() 实现；RAG 攻略知识由
RAGInjectMiddleware 注入；官方中间件提供工具/模型限次与工具重试。
"""
# 必须先于 langchain.agents / middleware 导入：补齐 langgraph.runtime 缺的注解符号，
# 否则 import create_agent 时因版本不匹配（langgraph 1.0.10 + langchain 1.x）直接 ImportError。
# 兼容层幂等：全项目任意入口只要先走到这里即可。
import app.core.langgraph_compat  # noqa: F401,E402

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from app.agent.subagent.middleware.param_check import ParamCheckMiddleware
from app.agent.subagent.middleware.rag_inject import RAGInjectMiddleware
from app.core.llm import llm
from app.tools.registry import tool_registry


def _build_checkpointer():
    """"子图独立 checkpointer：默认内存（多 worker 可切 Redis，与 graph.py 保持一致）。"""
    from app.core.config import get_settings
    from langgraph.checkpoint.memory import MemorySaver
    settings = get_settings()
    if getattr(settings, "checkpointer", "memory").lower() == "redis":
        try:
            from langgraph.checkpoint.redis.aio import RedisSaver
            return RedisSaver.from_conn_string(settings.redis_url)
        except Exception:
            pass
    return MemorySaver()


# ---------- F6 角色表（按业务域拆，不按工具数拆） ----------
# 每个角色：绑定工具集 + 提示词路径。跨域编排由 itinerary（全量）承担。
PAIR_ROLES = {
    "transit":    ["search_flights", "search_hotels"],          # 出行：机票/酒店
    "attraction": ["get_attractions", "get_weather"],            # 游玩：景点/天气
    "budget":     ["get_employee_info", "analyze_kpis"],         # 预算：团队/预算
}
ITINERARY_TOOLS = ["search_flights", "search_hotels", "get_attractions",
                   "get_weather", "get_employee_info", "analyze_kpis"]

_ROLE_PROMPT_HINTS = {
    "transit": "你负责【出行安排】子任务：查询并聚合航班与酒店信息，只输出结构化结论(不含步骤)，"
               "大结果给摘要与引用。",
    "attraction": "你负责【游玩安排】子任务：查询并聚合景点与天气信息，只输出结构化结论，大结果给摘要与引用。",
    "budget": "你负责【预算与团队】子任务：查询团队信息与预算分析，只输出结构化结论，大结果给摘要与引用。",
    "itinerary": "你是主 Agent，负责把各子任务的结论聚合为完整方案。",
}


def _tools_for_role(role: str) -> list:
    """按角色解析工具句柄；role 不在表内回退 itinéraire 全量。"""
    names = ITINERARY_TOOLS if role == "itinerary" else PAIR_ROLES.get(role, ITINERARY_TOOLS)
    return [tool_registry.get_executor(n) for n in names if tool_registry.get_executor(n)]


def _prompt_for_role(role: str, base_prompt: str) -> str:
    hint = _ROLE_PROMPT_HINTS.get(role)
    return (hint + "\n\n" if hint else "") + base_prompt


def create_subagent(role: str | None = None, tools=None, system_prompt: str | None = None):
    """F6 参数化子图工厂：按角色/工具/prompt 创建 create_agent。默认全量（itinerary）。"""
    role = role or "itinerary"
    if system_prompt is None:
        from app.agent.subagent.prompts.planner_prompt import PLANNER_PROMPT
        system_prompt = _prompt_for_role(role, PLANNER_PROMPT)
    return create_agent(
        model=llm.raw_model,
        tools=tools or _tools_for_role(role),
        system_prompt=system_prompt,
        middleware=[
            RAGInjectMiddleware(),              # 攻略知识注入（每次模型调用前）
            ParamCheckMiddleware(max_ask=1),    # 缺参最多追问 1 次，仍缺则由模型兜底直出
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
            ModelCallLimitMiddleware(run_limit=15, exit_behavior="end"),
            ToolRetryMiddleware(max_retries=3),
        ],
        checkpointer=_build_checkpointer(),
    )


def _build_subagent():
    """构建默认单例子图（travel 全量工具，多 Agent 关闭时的回退路径，行为不变）。"""
    return create_subagent("itinerary")


# 角色子图缓存（F6 委派复用编译产物，避免每次 create_agent 重新编译的开销）
_role_cache: dict[str, object] = {}


def get_role_subagent(role: str):
    """取某角色的子图（进程内缓存）；缺失则编译并缓存。"""
    if role not in _role_cache:
        _role_cache[role] = create_subagent(role)
    return _role_cache[role]


# 进程启动时编译一次，运行时零开销（对应文档 §10.3）
# 默认单例：multi_agent_enabled=false 时 planner_run 继续用它（不改原路径）
subagent = _build_subagent()