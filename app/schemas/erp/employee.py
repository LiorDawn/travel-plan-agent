"""员工 Schema — 对齐 ERP 员工信息（/api/employees/*）"""
from pydantic import BaseModel, Field


class EmployeeSchema(BaseModel):
    """员工数据模型"""
    id: str = Field(..., description="员工 ID")
    name: str = Field(default="", description="姓名")
    department: str = Field(default="", description="部门")
    position: str = Field(default="", description="职位")
    annual_budget: float = Field(default=0.0, description="年度预算")
    remaining_budget: float = Field(default=0.0, description="剩余预算")