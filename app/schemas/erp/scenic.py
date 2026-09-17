"""景点 Schema — 对应 Java Scenic record（/api/scenic/search）"""
from pydantic import BaseModel, Field
from typing import Optional


class ScenicSchema(BaseModel):
    """景点数据模型"""
    id: str = Field(..., description="景点编号")
    name: str = Field(..., description="景点名称")
    city: str = Field(..., description="所在城市")
    category: str = Field(..., description="类别")
    level: Optional[str] = Field(default="", description="等级，如 5A")
    ticketPrice: float = Field(default=0.0, description="门票价格")
    openTime: str = Field(default="", description="开放时间")
    rating: float = Field(default=0.0, description="评分")
    bestSeason: Optional[str] = Field(default="", description="最佳季节")
    durationHours: float = Field(default=0.0, description="建议游玩时长(小时)")
    description: str = Field(default="", description="简介")