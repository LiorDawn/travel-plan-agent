from langchain_core.tools import tool
from app.integrations.erp.client import erp_client


@tool
async def get_employee_info(employee_id: str) -> dict:
    """获取员工的基本信息、差旅预算和出行政策。

    Args:
        employee_id: 员工工号，如 "E001"
    """
    return await erp_client.get_employee(employee_id)