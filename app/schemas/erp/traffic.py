"""交通 Schema — 对应 Java Traffic record（/api/traffic/search）"""
from pydantic import BaseModel, Field


class TrafficSchema(BaseModel):
    """交通方案数据模型"""
    id: str = Field(..., description="方案编号")
    origin: str = Field(..., description="出发地")
    destination: str = Field(..., description="目的地")
    mode: str = Field(..., description="交通方式")
    durationHours: float = Field(default=0.0, description="时长(小时)")
    price: float = Field(default=0.0, description="价格")
    departTime: str = Field(default="", description="出发时间")
    arriveTime: str = Field(default="", description="到达时间")
    description: str = Field(default="", description="描述")
    seats: int = Field(default=0, description="余票")