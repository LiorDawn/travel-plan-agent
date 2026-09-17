"""F2 任务清单状态机 — 外层轻量版

plan 产出 mult步任务最自然形态是 tool_plan（齐参工具）。本模块在 execute 入口处
由 tool_plan 派生任务清单（task per selected tool），随执行推进状态：
    pending(待执行) → in_progress(执行中) → completed(成功) / failed(失败)

不触碰子图 ReAct；事件层在 execute 结束时统一发 task.update 携带最新清单，供前端
展示 long-run 进度。
"""
from __future__ import annotations

import time
from typing import List, Dict, Any

# ---- 任务状态常量（对外契约，前端/事件复用） ----
TS_PENDING = "pending"        # 待执行
TS_IN_PROGRESS = "in_progress"  # 执行中
TS_COMPLETED = "completed"    # 成功
TS_FAILED = "failed"          # 失败


def build_tasks(tool_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 plan 的 tool_plan（ready 且 params 齐）派生任务清单。

    每个 ready 工具对应一个任务；未 ready 的不入清单（缺参由 need_info/ask_user 先补）。
    """
    tasks: List[Dict[str, Any]] = []
    for i, t in enumerate(tool_plan or []):
        if not t.get("ready"):
            continue
        tasks.append({
            "id": f"{t.get('name', 'task')}:{i}",   # 唯一化，避免同工具多实例冲突
            "tool": t.get("name", ""),
            "desc": _describe(t),
            "status": TS_PENDING,
            "updated_at": time.time(),
        })
    return tasks


def build_plan_tasks(plan_tasks, tool_plan: List[Dict[str, Any]],
                     fallback: bool = True) -> List[Dict[str, Any]]:
    """F2 轻量：按 plan 源头产出的 PlanTask 建任务清单（带 depends_on/sub_agent_type）。

    优先从 plan_tasks 消费：id 唯一化（缺 id 用 {tool}:{i} 补，防重）、保留依赖/角色字段用于展示；
    一个工具多实例由 plan 给唯一 id。plan_tasks 为空/无效项时回退到 build_tasks(tool_plan) 兜底派生。
    对齐：id=工具名(旧) → plan 唯一 id（新），依赖字段仅作前端展示，不约束执行顺序。
    """
    tasks: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for t in plan_tasks or []:
        tool = t.get("tool") if isinstance(t, dict) else getattr(t, "tool", None)
        if not tool:
            continue
        raw_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        tid = str(raw_id) if (raw_id is not None and raw_id != "") else tool
        n = 2
        while tid in used_ids:
            tid = f"{tid}#{n}"
            n += 1
        used_ids.add(tid)
        tasks.append({
            "id": tid,
            "tool": tool,
            "desc": t.get("desc") if isinstance(t, dict) else getattr(t, "desc", ""),
            "depends_on": (t.get("depends_on") if isinstance(t, dict) else getattr(t, "depends_on", [])) or [],
            "sub_agent_type": t.get("sub_agent_type") if isinstance(t, dict) else getattr(t, "sub_agent_type", None),
            "status": TS_PENDING,
            "updated_at": time.time(),
        })
    if not tasks and fallback:
        return build_tasks(tool_plan)
    return tasks


def mark(tasks: List[Dict[str, Any]], task_id: str, status: str) -> List[Dict[str, Any]]:
    """更新单个任务状态（in-place 浅拷贝返回新 list，避免污染 state 引用）。"""
    out = list(tasks or [])
    for i, task in enumerate(out):
        if task.get("id") == task_id:
            task = {**task, "status": status, "updated_at": time.time()}
            out[i] = task
    return out


def _describe(t: Dict[str, Any]) -> str:
    """人类可读任务描述，供前端展示。"""
    params = t.get("params") or {}
    parts = [f"{k}={v}" for k, v in params.items() if v not in (None, "")]
    suffix = ("(" + ", ".join(parts) + ")") if parts else ""
    return f"{t.get('name', '')}{suffix}"