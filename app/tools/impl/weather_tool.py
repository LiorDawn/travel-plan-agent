"""天气查询工具 — 优先接入 MCP weather server 取真实数据；不可用则明确降级，绝不编造数值。"""
from langchain_core.tools import tool

from app.integrations.mcp.client import mcp_client
from app.core.logging import logger


@tool
async def get_weather(city: str, date: str = "") -> dict:
    """查询城市天气。

    Args:
        city: 城市，如"三亚"
        date: 查询日期，可空（默认今天）
    """
    # 优先走 MCP weather server（真实数据源）
    try:
        result = await mcp_client.call_tool("weather", "get_weather", {
            "city": city, "date": date or "今天",
        })
        if isinstance(result, dict) and not result.get("error"):
            payload = result.get("result", result)
            if isinstance(payload, dict) and payload.get("condition"):
                payload.setdefault("source", "mcp")
                payload.setdefault("city", city)
                payload.setdefault("date", date or "今天")
                return payload
    except Exception as e:
        logger.warning("weather_mcp_failed", city=city, error=str(e))

    # 降级：数据不可用，明确告知，不提供伪造的具体数值
    logger.warning("weather_unavailable", city=city, date=date)
    return {
        "city": city,
        "date": date or "今天",
        "condition": None,
        "temperature": None,
        "source": "unavailable",
        "advice": "天气数据暂不可用，行程安排请以实时查询为准",
    }