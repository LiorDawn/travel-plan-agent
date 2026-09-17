"""景点推荐工具 — 数据从 Java ERP /api/scenic/search 获取，失败回退 mock"""
from langchain_core.tools import tool
from app.tools.impl.erp_client_tool import ERPClientTool


@tool
async def get_attractions(city: str, category: str = "") -> list[dict]:
    """查询城市的推荐景点与游玩信息。

    Args:
        city: 城市，如"三亚"
        category: 景点分类，如 nature/history/food，不传表示全部
    """
    source, items = await ERPClientTool.search_scenics(city, category)
    scenics = []
    for s in items:
        scenics.append({
            "name": s["name"],
            "city": s["city"],
            "category": s["category"],
            "level": s["level"],
            "ticket_price": round(s["ticketPrice"], 2),
            "open_time": s["openTime"],
            "rating": s["rating"],
            "duration_hours": s["durationHours"],
            "description": s["description"],
        })
    return scenics