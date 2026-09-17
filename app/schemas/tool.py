from pydantic import BaseModel, Field
from typing import List, Optional, Any


class ToolParamDef(BaseModel, strict=True):
    """工具参数定义"""
    name: str = Field(..., description="参数名")
    type: str = Field(default="str", description="参数类型: str|int|float|date|enum")
    required: bool = Field(default=True, description="是否必填")
    description: str = Field(default="", description="参数说明")
    default: Optional[Any] = Field(default=None, description="默认值")
    enum_values: Optional[List[str]] = Field(default=None, description="枚举可选值")


class ToolDef(BaseModel, strict=True):
    """工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: List[ToolParamDef] = Field(default_factory=list, description="参数列表")
    source: str = Field(default="builtin", description="来源: builtin|mcp|erp")


class ToolPlan(BaseModel, strict=True):
    """工具调用计划 — 含参数校验状态"""
    tool: ToolDef
    params: dict = Field(default_factory=dict, description="已填充的参数值")
    missing_params: List[str] = Field(default_factory=list, description="缺失的必填参数")
    ready: bool = Field(default=False, description="参数是否齐全")
