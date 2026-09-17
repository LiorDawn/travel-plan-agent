"""外层 execute 节点 — 齐参工具并行执行（0 次 LLM）（对应重构文档 §4.3 ④execute）

消费 plan 产出的 tool_plan（ready=True 的工具），asyncio.gather 并行执行，
return_exceptions 防单工具失败阻塞。缺参工具不在此执行（走 interrupt/summarize）。
工具按 name 从 tool_registry.get_executor(name) 分发到 @tool async 函数，用 .ainvoke 正确触发。
"""
import asyncio
import time

from app.agent import tasks
from app.agent.state import AgentState
from app.core.logging import logger
from app.memory.artifact import artifact_store
from app.tools.registry import tool_registry


async def _run_one(name: str, params: dict, conv_id: str) -> tuple[str, dict]:
    """执行单个工具，异常不抛出；大结果走 F5 外部化（摘要+引用）。"""
    fn = tool_registry.get_executor(name)
    if fn is None:
        return name, {"error": f"未知工具 {name}", "raw": None}
    try:
        raw = await fn.ainvoke(params or {})
        # F5：结果经 ArtifactStore 判定——大结果只回摘要+引用，原文落库（async 版支持 LLM 摘要）
        return name, {"error": None, **await artifact_store.async_maybe_summarize(name, raw, conv_id)}
    except Exception as e:
        logger.warning("execute_tool_failed", tool=name, error=str(e))
        return name, {"error": str(e), "raw": None}


async def execute_node(state: AgentState) -> AgentState:
    """并行执行 plan 选出的齐参工具，结果写入 tool_results。

    F2：任务清单由 plan 源头产出（plan_tasks），execute 优先消费；plan 未产出则用
    tool_plan 兜底派生。执行仍 asyncio.gather 并行，成/败映射任务状态，供事件层发 task.update。
    """
    tool_plan = state.get("tool_plan", [])
    ready = [t for t in tool_plan if t.get("ready")]
    conv_id = state.get("conversation_id", "")

    # F2：优先用 plan_tasks 建任务清单（保留 depends_on/sub_agent_type），否则 tool_plan 兜底
    task_list = state.get("tasks") or tasks.build_plan_tasks(
        state.get("plan_tasks") or [], tool_plan)
    # 标记各待执行工具为进行中（按 tool 名匹配任务 id 前缀/相等）
    for t in ready:
        for task in task_list:
            if task.get("tool") == t.get("name") and task.get("status") == tasks.TS_PENDING:
                task_list = tasks.mark(task_list, task["id"], tasks.TS_IN_PROGRESS)
                break
    state = {**state, "tasks": task_list,
             "active_task_id": (ready[0]["name"] if ready else "")}

    n_failed = 0
    results: dict = {}
    if ready:
        done = await asyncio.gather(*(_run_one(t["name"], t.get("params") or {}, conv_id) for t in ready))
        for name, res in done:
            results[name] = res
            status = tasks.TS_FAILED if res.get("error") else tasks.TS_COMPLETED
            for task in task_list:
                if task.get("tool") == name and task.get("status") in (
                        tasks.TS_PENDING, tasks.TS_IN_PROGRESS):
                    task_list = tasks.mark(task_list, task["id"], status)
                    break
            if res.get("error"):
                n_failed += 1

    return {**state, "tool_results": results, "tasks": task_list,
            "active_task_id": "", "task_updated_at": time.time(), "task_failed": n_failed}