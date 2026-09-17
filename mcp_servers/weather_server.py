"""
MCP 天气查询服务 (独立进程，端口 8100)

模拟 MCP 协议: 提供工具发现 + 工具调用接口
Agent 通过 HTTP 调用本服务获取天气数据
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import random

app = FastAPI(title="MCP Weather Server", version="1.0.0")

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气信息",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "date": {"type": "string", "description": "查询日期 YYYY-MM-DD，可选"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_attractions",
        "description": "查询城市的推荐景点",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "category": {"type": "string", "description": "景点类别: nature|history|food|shopping"},
            },
            "required": ["city"],
        },
    },
]


class ToolCallRequest(BaseModel):
    name: str = Field(..., description="工具名称")
    arguments: dict = Field(default_factory=dict, description="调用参数")


@app.get("/tools/list")
async def list_tools():
    return {"tools": TOOLS}


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    if request.name == "get_weather":
        return await _get_weather(request.arguments)
    elif request.name == "get_attractions":
        return await _get_attractions(request.arguments)
    return {"error": f"Unknown tool: {request.name}"}


async def _get_weather(args: dict) -> dict:
    city = args.get("city", "未知")
    conditions = ["晴", "多云", "阴", "小雨", "阵雨"]
    temps = list(range(15, 36))
    return {
        "city": city,
        "date": args.get("date", "今天"),
        "temperature": f"{random.choice(temps)}C",
        "condition": random.choice(conditions),
        "humidity": f"{random.randint(40, 90)}%",
        "wind": f"{random.choice(['东北风', '西南风', '北风', '南风'])} {random.randint(1, 5)}级",
        "tips": "适宜出行" if random.random() > 0.3 else "建议带伞",
    }


async def _get_attractions(args: dict) -> dict:
    city = args.get("city", "未知")
    db = {
        "三亚": [
            {"name": "亚龙湾", "category": "nature", "rating": 4.8, "price": "免费"},
            {"name": "天涯海角", "category": "nature", "rating": 4.5, "price": "81元"},
            {"name": "南山寺", "category": "history", "rating": 4.7, "price": "129元"},
            {"name": "蜈支洲岛", "category": "nature", "rating": 4.6, "price": "144元"},
        ],
        "北京": [
            {"name": "故宫", "category": "history", "rating": 4.8, "price": "60元"},
            {"name": "长城", "category": "history", "rating": 4.7, "price": "40元"},
        ],
    }
    return {"city": city, "attractions": db.get(city, [
        {"name": f"{city}中心公园", "category": "nature", "rating": 4.3, "price": "免费"},
        {"name": f"{city}博物馆", "category": "history", "rating": 4.5, "price": "免费"},
    ])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
