"""统一 ERP 客户端工具 — 从 Java 后端取数，Java 不可达时回退本地 mock。

Java 后端 (travel_agent/java_erp) 提供：
  - GET /api/scenic/search?keyword&city&category  景点
  - GET /api/hotel/search?city&min_star&max_star   酒店
  - GET /api/traffic/search?origin&destination&mode 交通
"""
import httpx
from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.erp.hotel import HotelSchema
from app.schemas.erp.scenic import ScenicSchema
from app.schemas.erp.traffic import TrafficSchema

settings = get_settings()


class ERPClientTool:
    """Java ERP 数据客户端 — 每个方法返回「来源 + 数据列表」"""

    BASE = settings.erp_base_url.rstrip("/")

    @classmethod
    async def _get(cls, path: str, params: dict) -> dict | None:
        """请求 Java ERP；失败或不可达返回 None（由调用方决定回退）"""
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{cls.BASE}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("erp_tool_req_failed", path=path, error=str(e))
            return None

    @classmethod
    async def search_scenics(cls, city: str = "", category: str = "") -> tuple[str, list[dict]]:
        """查景点：优先 Java，回退本地 mock。返回 (来源, 景点列表)"""
        data = await cls._get("/api/scenic/search", {"city": city, "category": category})
        if data and data.get("items"):
            items = [ScenicSchema.model_validate(i).model_dump() for i in data["items"]]
            return "erp", items
        return "mock", _mock_scenics(city, category)

    @classmethod
    async def search_hotels(cls, city: str, min_star: int = 1, max_star: int = 5) -> tuple[str, list[dict]]:
        data = await cls._get("/api/hotel/search", {"city": city, "min_star": min_star, "max_star": max_star})
        if data and data.get("items"):
            items = [HotelSchema.model_validate(i).model_dump() for i in data["items"]]
            return "erp", items
        return "mock", _mock_hotels(city)

    @classmethod
    async def search_traffics(cls, origin: str, destination: str, mode: str = "") -> tuple[str, list[dict]]:
        data = await cls._get("/api/traffic/search", {"origin": origin, "destination": destination, "mode": mode})
        if data and data.get("items"):
            items = [TrafficSchema.model_validate(i).model_dump() for i in data["items"]]
            return "erp", items
        return "mock", _mock_traffics(origin, destination)


# ---------- 本地 mock（Java 不可达时回退；dest 为 snake_case 保持与 Java 对齐） ----------
def _mock_scenics(city: str, category: str = "") -> list[dict]:
    base = [
        {"id": "S001", "name": f"{city}热带天堂森林公园", "city": city, "category": "nature", "level": "4A",
         "ticketPrice": 158, "openTime": "08:00-17:30", "rating": 4.6, "bestSeason": "全年", "durationHours": 4,
         "description": "雨林生态景区"},
        {"id": "S002", "name": f"{city}古城历史文化街", "city": city, "category": "history", "level": "4A",
         "ticketPrice": 0, "openTime": "全天", "rating": 4.4, "bestSeason": "全年", "durationHours": 3,
         "description": "古建筑街区"},
        {"id": "S003", "name": f"{city}美食街", "city": city, "category": "food", "level": "",
         "ticketPrice": 0, "openTime": "10:00-22:00", "rating": 4.3, "bestSeason": "全年", "durationHours": 2,
         "description": "特色小吃汇聚地"},
    ]
    if category:
        base = [s for s in base if s["category"] == category]
    return base


def _mock_hotels(city: str) -> list[dict]:
    return [
        {"id": "H001", "name": f"{city}商务酒店", "city": city, "star": 4, "pricePerNight": 450,
         "rating": 4.5, "address": f"{city}市中心", "nearbyScenicIds": [], "facilities": ["WiFi", "早餐"]},
        {"id": "H002", "name": f"{city}度假酒店", "city": city, "star": 5, "pricePerNight": 880,
         "rating": 4.8, "address": f"{city}海滨", "nearbyScenicIds": [], "facilities": ["WiFi", "泳池", "SPA"]},
    ]


def _mock_traffics(origin: str, destination: str) -> list[dict]:
    return [
        {"id": "T001", "origin": origin, "destination": destination, "mode": "FLIGHT",
         "durationHours": 3.5, "price": 1280, "departTime": "08:00", "arriveTime": "11:30",
         "description": "直飞航班", "seats": 120},
        {"id": "T002", "origin": origin, "destination": destination, "mode": "TRAIN",
         "durationHours": 12, "price": 680, "departTime": "19:00", "arriveTime": "次日07:00",
         "description": "高铁", "seats": 300},
    ]