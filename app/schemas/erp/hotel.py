"""酒店 Schema — 对应 Java Hotel record（/api/hotel/search）"""
from pydantic import BaseModel, Field


class HotelSchema(BaseModel):
    """酒店数据模型"""
    id: str = Field(..., description="酒店编号")
    name: str = Field(..., description="酒店名称")
    city: str = Field(..., description="城市")
    star: int = Field(default=3, description="星级")
    pricePerNight: float = Field(default=0.0, description="每晚价格")
    rating: float = Field(default=0.0, description="评分")
    address: str = Field(default="", description="地址")
    nearbyScenicIds: list[str] = Field(default_factory=list, description="周边景点 ID")
    facilities: list[str] = Field(default_factory=list, description="设施")