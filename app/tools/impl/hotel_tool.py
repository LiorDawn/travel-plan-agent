"""酒店搜索工具 — 数据从 Java ERP /api/hotel/search 获取，失败回退 mock"""
from langchain_core.tools import tool
from app.tools.impl.erp_client_tool import ERPClientTool


@tool
async def search_hotels(city: str, check_in: str, check_out: str,
                        guests: int = 1, max_star: int = 5) -> list[dict]:
    """搜索城市指定入住日期内的酒店。

    Args:
        city: 城市，如"三亚"
        check_in: 入住日期，格式 YYYY-MM-DD
        check_out: 退房日期，格式 YYYY-MM-DD
        guests: 入住人数，默认 1
        max_star: 最高星级，默认 5
    """
    source, items = await ERPClientTool.search_hotels(city, 1, max_star)
    hotels = []
    for h in items:
        hotels.append({
            "name": h["name"],
            "address": h["address"],
            "check_in": check_in,
            "check_out": check_out,
            "price_per_night": round(h["pricePerNight"], 2),
            "rating": h["rating"],
            "amenities": h["facilities"],
        })
    return hotels