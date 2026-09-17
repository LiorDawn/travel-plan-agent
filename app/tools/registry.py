from app.schemas.tool import ToolDef, ToolParamDef
from app.tools.impl.analysis_tool import analyze_kpis
from app.tools.impl.attraction_tool import get_attractions
from app.tools.impl.employee import get_employee_info
from app.tools.impl.flight_tool import search_flights
from app.tools.impl.hotel_tool import search_hotels
from app.tools.impl.weather_tool import get_weather

# 工具名 → @tool 可执行函数（execute 节点按 name 分发执行；impl 为纯实现，不反向依赖本模块）
_EXECUTOR_MAP: dict[str, object] = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "get_attractions": get_attractions,
    "get_weather": get_weather,
    "get_employee_info": get_employee_info,
    "analyze_kpis": analyze_kpis,
}


class ToolRegistry:
    """工具注册表 — 集中定义所有工具与其参数 Schema，供意图识别与前端展示。"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """注册内置 6 个工具的 Schema 定义（含 required/类型/默认值）。"""
        self._tools["search_flights"] = ToolDef(
            name="search_flights",
            description="搜索航班信息，需要出发城市、目的地城市、出发日期",
            source="builtin",
            parameters=[
                ToolParamDef(name="origin", type="str", required=True, description="出发城市，如 北京"),
                ToolParamDef(name="destination", type="str", required=True, description="目的城市，如 三亚"),
                ToolParamDef(name="date", type="date", required=True, description="出发日期，如 2026-09-10"),
                ToolParamDef(name="passengers", type="int", required=False, description="乘客人数", default=1),
            ],
        )
        self._tools["search_hotels"] = ToolDef(
            name="search_hotels",
            description="搜索酒店信息，需要城市、入住日期、退房日期",
            source="builtin",
            parameters=[
                ToolParamDef(name="city", type="str", required=True, description="城市名称"),
                ToolParamDef(name="check_in", type="date", required=True, description="入住日期"),
                ToolParamDef(name="check_out", type="date", required=True, description="退房日期"),
                ToolParamDef(name="guests", type="int", required=False, description="入住人数", default=1),
            ],
        )
        self._tools["get_employee_info"] = ToolDef(
            name="get_employee_info",
            description="查询员工信息（从ERP获取），需要员工编号",
            source="erp",
            parameters=[
                ToolParamDef(name="employee_id", type="str", required=True, description="员工编号"),
            ],
        )
        # 天气查询（走工具实现）
        self._tools["get_weather"] = ToolDef(
            name="get_weather",
            description="查询指定城市的天气信息",
            source="builtin",
            parameters=[
                ToolParamDef(name="city", type="str", required=True, description="城市名称"),
                ToolParamDef(name="date", type="date", required=False, description="查询日期"),
            ],
        )
        # 景点推荐（走工具实现）
        self._tools["get_attractions"] = ToolDef(
            name="get_attractions",
            description="查询城市推荐景点",
            source="builtin",
            parameters=[
                ToolParamDef(name="city", type="str", required=True, description="城市名称"),
                ToolParamDef(name="category", type="str", required=False, description="景点类别: nature|history|food|shopping"),
            ],
        )
        # 数据分析/报表（走工具实现）
        self._tools["analyze_kpis"] = ToolDef(
            name="analyze_kpis",
            description="对企业/业务数据做统计、报表、可视化，如员工成本、差旅成本、酒店入住率、部门分布",
            source="erp",
            parameters=[
                ToolParamDef(name="metric", type="str", required=False, description="统计指标（中文语义），如 员工成本/差旅成本/酒店入住率/部门分布；留空为业务概览"),
                ToolParamDef(name="period", type="str", required=False, description="统计周期，如 本月/本季度/近一年；可留空"),
            ],
        )

    def get_tool_def(self, name: str) -> dict:
        """按工具名取工具定义(含参数 Schema)；不存在返回空 dict。"""
        tool = self._tools.get(name)
        return tool.model_dump() if tool else {}

    def get_tools_desc(self) -> str:
        """生成给 LLM 看的工具清单文本(供意图识别选择工具)。"""
        lines = []
        for name, tool in self._tools.items():
            params = ", ".join(
                f"{p.name}({'必填' if p.required else '可选'})" for p in tool.parameters
            )
            lines.append(f"- {name} [{tool.source}]: {tool.description} | 参数: {params}")
        return "\n".join(lines)

    def list_tools(self) -> list[dict]:
        """返回全部注册工具的完整定义供接口/前端展示。"""
        return [t.model_dump() for t in self._tools.values()]

    def get_executor(self, name: str):
        """按工具名返回可执行的 @tool 函数；不存在返回 None。"""
        return _EXECUTOR_MAP.get(name)

    def missing_required(self, name: str, params: dict) -> list[str]:
        """返回指定工具当前缺失的必填参数名列表（依据注册表 required 定义）。

        plan 缺参判定与子图 ParamCheckMiddleware 共用，保持两处校验口径一致。
        """
        defn = self.get_tool_def(name)
        required = [p["name"] for p in (defn.get("parameters") or []) if p.get("required")]
        return [p for p in required if not params.get(p)]


tool_registry = ToolRegistry()