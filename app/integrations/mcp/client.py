"""
MCP 客户端 — 连接远程 MCP 服务，发现和调用工具

支持的 MCP 服务:
- weather: 天气查询 + 景点推荐 (端口 8100)
"""
import httpx
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


class MCPClient:
    """MCP 远程服务客户端 — 负责发现与调用外部 MCP 工具。"""

    def __init__(self):
        # URL 来自配置（.env 的 MCP_WEATHER_BASE_URL），Docker 环境可指向 mcp_weather:8100
        self.servers = {
            "weather": settings.mcp_weather_base_url,
        }

    async def list_tools(self, server_name: str) -> list[dict]:
        """获取 MCP 服务的工具列表"""
        url = f"{self.servers.get(server_name, '')}/tools/list"
        if not url.startswith("http"):
            return []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json().get("tools", [])
        except Exception as e:
            logger.warning("mcp_list_tools_failed", server=server_name, error=str(e))
            return []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """调用 MCP 服务的工具"""
        url = f"{self.servers.get(server_name, '')}/tools/call"
        if not url.startswith("http"):
            return {"error": f"Unknown server: {server_name}"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={"name": tool_name, "arguments": arguments})
                resp.raise_for_status()
                result = resp.json()
                logger.info("mcp_call", server=server_name, tool=tool_name, status="success")
                return result
        except Exception as e:
            logger.error("mcp_call_failed", server=server_name, tool=tool_name, error=str(e))
            return {"error": str(e)}


# 全局单例
mcp_client = MCPClient()