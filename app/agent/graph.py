"""外层 LangGraph 图声明 — 只做 add_node/add_edge（对应重构文档 §3.6）

职责：声明 五分流 的节点与边；节点实现、条件路由、checkpointer、事件映射、
运行器（run/resume）分别见 nodes/、router.py、checkpointer.py、events.py、runner.py。
"""
from langgraph.graph import END, StateGraph

from app.agent.checkpointer import _build_checkpointer
from app.agent.events import (
    NODE_ASK_USER,
    NODE_EXECUTE,
    NODE_FINALIZE,
    NODE_PLAN,
    NODE_PLANNER_RUN,
    NODE_RAG_MANUAL,
    NODE_RAG_RECALL,
    NODE_SUMMARIZE,
)
from app.agent.nodes.ask_user import ask_user_node
from app.agent.nodes.execute import execute_node
from app.agent.nodes.finalize import finalize_node
from app.agent.nodes.plan import plan_node
from app.agent.nodes.planner_run import planner_run_node
from app.agent.nodes.rag_manual import rag_manual_node
from app.agent.nodes.rag_recall import rag_recall_node
from app.agent.nodes.summarize import summarize_node
from app.agent.router import route_after_ask_user, route_after_plan, route_after_planner
from app.agent.state import AgentState
from app.memory.audit.recorder import audit_wrap


def _build_graph():
    """声明并编译外层图：澄清+五分流节点 + 4 条条件路由边，挂外层 checkpoint。"""
    builder = StateGraph(AgentState)
    builder.add_node(NODE_RAG_RECALL, rag_recall_node)
    builder.add_node(NODE_PLAN, audit_wrap("node_end", "plan")(plan_node))
    builder.add_node(NODE_EXECUTE, audit_wrap("node_end", "execute")(execute_node))
    builder.add_node(NODE_RAG_MANUAL, rag_manual_node)
    builder.add_node(NODE_SUMMARIZE, audit_wrap("node_end", "summarize")(summarize_node))
    builder.add_node(NODE_PLANNER_RUN, audit_wrap("node_end", "planner_run")(planner_run_node))
    builder.add_node(NODE_ASK_USER, ask_user_node)      # 缺参（plan/子图）唯一挂起点
    builder.add_node(NODE_FINALIZE, finalize_node)

    builder.set_entry_point(NODE_RAG_RECALL)
    builder.add_edge(NODE_RAG_RECALL, NODE_PLAN)

    # plan 后分流（interrupt→ask_user；direct/summarize；execute；react→planner_run）
    builder.add_conditional_edges(
        NODE_PLAN, route_after_plan,
        {
            "interrupt": NODE_ASK_USER,
            "direct": NODE_SUMMARIZE,
            "summarize": NODE_SUMMARIZE,
            "execute": NODE_EXECUTE,
            "react": NODE_PLANNER_RUN,
        },
    )
    # 关思考快查：execute → rag_manual → summarize
    builder.add_edge(NODE_EXECUTE, NODE_RAG_MANUAL)
    builder.add_edge(NODE_RAG_MANUAL, NODE_SUMMARIZE)
    builder.add_edge(NODE_SUMMARIZE, NODE_FINALIZE)

    # 子图：中断 → ask_user；完成 → finalize
    builder.add_conditional_edges(
        NODE_PLANNER_RUN, route_after_planner,
        {"ask_user": NODE_ASK_USER, "finalize": NODE_FINALIZE},
    )
    # ask_user resume 后按来源回跳：plan 缺参 → plan 重判；子图缺参 → planner_run 续跑
    builder.add_conditional_edges(
        NODE_ASK_USER, route_after_ask_user,
        {"plan": NODE_PLAN, "planner_run": NODE_PLANNER_RUN},
    )
    builder.add_edge(NODE_FINALIZE, END)

    return builder.compile(checkpointer=_build_checkpointer())


# 进程启动编译一次，运行期零开销（runner.py 从本模块取图）
_compiled_graph = _build_graph()