"""plan 节点契约 — 结构化输出 schema（契约层，独立于 prompt）

plan 节点一次 LLM 强约束输出的结构，供 agent/nodes/plan.py 与前端/测试复用。
"""
from typing import Literal

from pydantic import BaseModel, Field


class PlanTool(BaseModel):
    """plan 选出的工具及其已确认参数"""
    name: str = Field(..., description="工具名")
    params: dict = Field(default_factory=dict, description="从用户消息中能确认的参数，如 {\"city\":\"三亚\"}")


class PlanTask(BaseModel):
    """plan 产出的任务清单项（F2 轻量：供展示/溯源/依赖关系，不约束执行顺序）"""
    id: str = Field(..., description="任务唯一标识（可用 {tool}:{i} 保证同工具多实例不冲突）")
    tool: str = Field(..., description="绑定工具名")
    desc: str = Field(default="", description="人类可读任务描述")
    depends_on: list[str] = Field(default_factory=list, description="前置依赖任务 id 列表（展示用）")
    sub_agent_type: str | None = Field(default=None, description="归属子 Agent 角色（如 transit/attraction/budget），未委派为 None")


class DelegationRole(BaseModel):
    """F6 委派计划中的单个角色（plan 产出，替代运行时按工具猜测）"""
    role: Literal["transit", "attraction", "budget"] = Field(..., description="专业子 Agent 角色")
    tool_hints: list[str] = Field(default_factory=list, description="该角色应负责的工具线索（仅提示，最终由 role 派发）")


class PlanNeedField(BaseModel):
    """需补充的信息项（feasibility=need_info 时携带，驱动弹窗收集）"""
    key: str = Field(..., description="字段键，如 budget/people/days，用于回灌")
    question: str = Field(..., description="向用户提问的文本")
    type: Literal["text", "number", "date", "option"] = Field(default="text")
    options: list[str] = Field(default_factory=list, description="type=option 时的建议选项")
    required: bool = Field(default=True,
        description="True=必须补才能完成；False=可选偏好（优化方案质量，如预算/天数/出发地）")


class PlanOutput(BaseModel):
    """plan 节点强制输出 schema"""
    intent: Literal["general", "query", "travel"] = Field(..., description="意图类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")
    tools: list[PlanTool] = Field(default_factory=list, description="需要调用的工具及已确认参数")
    tasks: list[PlanTask] = Field(default_factory=list, description="F2 任务清单（plan 源头产出）")
    delegation: list[DelegationRole] = Field(default_factory=list, description="F6 委派计划（仅多 Agent 开启且 travel 时非空）")
    feasibility: Literal["feasible", "need_info", "not_doable"] = Field(
        default="feasible", description="模型自评能否完成任务")
    needs: list[PlanNeedField] = Field(default_factory=list,
        description="需要用户补充的信息；required=True 缺一项都无法执行，False 为偏好优化项")
    impossible_reason: str = Field(default="",
        description="feasibility=not_doable 时给出无法完成的原因 + 需要什么才能完成")
    nudge: str = Field(default="", description="可行路径下，可选的下一步引导语（最终以 summarize 产出为准）")
    reason: str = Field(default="", description="判断理由")