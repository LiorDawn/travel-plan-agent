import httpx
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class ERPClient:
    """Java ERP HTTP 客户端

    - ERP_USE_MOCK=true ：返回本地静态 mock 数据，供无 ERP 环境开发。
    - ERP_USE_MOCK=false：请求真实 ERP 地址（ERP_BASE_URL），带超时与异常捕获。
    """

    def __init__(self):
        self.use_mock = settings.erp_use_mock
        self.base_url = settings.erp_base_url.rstrip("/")
        self.api_key = ""
        self._timeout = 10.0

    async def get_employee(self, employee_id: str) -> dict:
        if self.use_mock:
            return self._mock_employee(employee_id)
        url = f"{self.base_url}/api/employees/{employee_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("erp_get_employee_failed", employee_id=employee_id, error=str(e))
            return {"error": f"ERP 员工查询失败: {str(e)}"}

    async def get_employee_budget(self, employee_id: str) -> dict:
        if self.use_mock:
            return {"annual_budget": 50000, "remaining_budget": 32000, "currency": "CNY"}
        url = f"{self.base_url}/api/employees/{employee_id}/budget"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("erp_get_budget_failed", employee_id=employee_id, error=str(e))
            return {"error": f"预算查询失败: {str(e)}"}

    async def get_employee_policy(self, employee_id: str) -> dict:
        if self.use_mock:
            return {"max_hotel_price": 500, "max_flight_cabin": "economy", "daily_allowance": 200, "require_approval": True}
        url = f"{self.base_url}/api/employees/{employee_id}/policy"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("erp_get_policy_failed", employee_id=employee_id, error=str(e))
            return {"error": f"差旅政策查询失败: {str(e)}"}

    async def submit_approval(self, plan_id: str, employee_id: str, amount: float) -> dict:
        if self.use_mock:
            return {"approval_id": "APR-001", "status": "pending", "plan_id": plan_id}
        url = f"{self.base_url}/api/approvals"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json={
                    "plan_id": plan_id, "employee_id": employee_id, "amount": amount,
                }, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("erp_submit_approval_failed", plan_id=plan_id, error=str(e))
            return {"error": f"审批提交失败: {str(e)}"}

    async def get_approval_status(self, approval_id: str) -> dict:
        if self.use_mock:
            return {"approval_id": approval_id, "status": "approved", "approver": "部门经理"}
        url = f"{self.base_url}/api/approvals/{approval_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("erp_get_approval_failed", approval_id=approval_id, error=str(e))
            return {"error": f"审批状态查询失败: {str(e)}"}

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _mock_employee(employee_id: str) -> dict:
        return {
            "id": employee_id, "employee_no": employee_id, "name": "张三",
            "email": "zhangsan@example.com", "department": "技术部", "position": "高级工程师",
            "annual_budget": 50000, "remaining_budget": 32000,
            "travel_policy": {"max_hotel_price": 500, "max_flight_cabin": "economy", "daily_allowance": 200},
        }


erp_client = ERPClient()