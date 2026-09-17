"""航班(交通)搜索工具 — 数据从 Java ERP /api/traffic/search 获取，失败回退 mock"""
from langchain_core.tools import tool
from app.tools.impl.erp_client_tool import ERPClientTool


@tool
async def search_flights(origin: str, destination: str, date: str,
                         passengers: int = 1, mode: str = "FLIGHT") -> list[dict]:
    """查询航班或火车等交通方案。

    Args:
        origin: 出发城市，如"北京"
        destination: 到达城市，如"三亚"
        date: 出发日期，格式 YYYY-MM-DD
        passengers: 乘客人数，默认 1
        mode: 交通方式，FLIGHT=飞机/TRAIN=火车，默认 FLIGHT
    """
    source, items = await ERPClientTool.search_traffics(origin, destination, mode)
    flights = []
    for t in items:
        if not _is_flight(t):
            continue
        flights.append({
            "flight_no": t["id"],
            "airline": "ERP" if source == "erp" else "模拟航司",
            "origin": t["origin"],
            "destination": t["destination"],
            "departure_time": f"{date} {t['departTime']}",
            "arrival_time": f"{date} {t['arriveTime']}",
            "cabin_class": "economy",
            "price": round(t["price"], 2),
            "currency": "CNY",
        })
    return flights


def _is_flight(t: dict) -> bool:
    return t.get("mode", "").upper() in ("FLIGHT", "FLY")