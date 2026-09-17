"""数据分析工具 — 对 ERP/业务数据做统计与报表，供 query/general 意图输出图表。

来源对齐 settings.erp_use_mock：默认为本地 mock 演示数据；后续接入真实 ERP 时
把各 _mock 数据集替换为 Java 后端统计接口即可，返回结构保持不变。
"""
from typing import List
from langchain_core.tools import tool
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


# ---------- 指标词 -> 分析函数 的映射 ----------
_METRIC_HINTS = {
    "employee": ["员工", "人头", "人员", "人效"],
    "cost": ["成本", "费用", "支出", "花费", "预算", "差旅"],
    "hotel": ["酒店", "入住率", "客房", "住宿"],
    "department": ["部门", "组织", "科室"],
}


def _match_metric(metric: str) -> str:
    """把用户语义(中文指标)归一到内部指标 key；unknown 表示概览。"""
    m = (metric or "").strip()
    for key, hints in _METRIC_HINTS.items():
        if any(h in m for h in hints):
            return key
    return "overview"


def _mock_employees() -> List[dict]:
    """员工 & 部门 mock（未来由 Java ERP 统计接口下发）"""
    return [
        {"name": "张三", "department": "市场部", "base_salary": 12000, "travel_budget": 18000},
        {"name": "李四", "department": "销售部", "base_salary": 11000, "travel_budget": 24000},
        {"name": "王五", "department": "研发部", "base_salary": 18000, "travel_budget": 12000},
        {"name": "赵六", "department": "市场部", "base_salary": 9500, "travel_budget": 15000},
        {"name": "孙七", "department": "销售部", "base_salary": 10500, "travel_budget": 26000},
        {"name": "周八", "department": "研发部", "base_salary": 20000, "travel_budget": 14000},
    ]


def _mock_occupancy() -> List[dict]:
    """酒店入住率 mock（月度）"""
    return [
        {"month": "1月", "rate": 0.62}, {"month": "2月", "rate": 0.75}, {"month": "3月", "rate": 0.68},
        {"month": "4月", "rate": 0.72}, {"month": "5月", "rate": 0.81}, {"month": "6月", "rate": 0.77},
        {"month": "7月", "rate": 0.92}, {"month": "8月", "rate": 0.95}, {"month": "9月", "rate": 0.85},
    ]


def _monthly_cost(employees: List[dict]) -> List[dict]:
    """按月差旅成本占比 mock"""
    return [
        {"month": "1月", "cost": 32000}, {"month": "2月", "cost": 28000}, {"month": "3月", "cost": 35000},
        {"month": "4月", "cost": 30000}, {"month": "5月", "cost": 38000}, {"month": "6月", "cost": 34000},
        {"month": "7月", "cost": 42000}, {"month": "8月", "cost": 45000},
    ]


@tool
async def analyze_kpis(metric: str = "", period: str = "") -> dict:
    """统计 ERP/业务数据（员工成本、酒店入住率等），返回文本摘要和图表数据。

    Args:
        metric: 统计指标，支持中文语义，如 员工成本/酒店入住率/部门分布；空为业务概览
        period: 统计周期（预留，可忽略）
    """
    key = _match_metric(metric)
    employees = _mock_employees()

    if key == "employee":
        total = sum(e["base_salary"] for e in employees)
        by_dept: dict = {}
        for e in employees:
            by_dept.setdefault(e["department"], []).append(e)
        dept_names = list(by_dept.keys())
        dept_avg = [int(sum(e["base_salary"] for e in em) / len(em)) for em in by_dept.values()]
        summary = (
            f"共 {len(employees)} 名员工，月薪合计 {total:,} 元。"
            f"研发部平均月薪最高（{max(dept_avg):,} 元）。"
        )
        chart = {"type": "bar", "title": "各部门平均月薪（元）", "labels": dept_names, "values": dept_avg}
        return {"metric": "员工", "summary": summary, "chart": chart}

    if key == "cost":
        rows = _monthly_cost(employees)
        labels = [r["month"] for r in rows]
        values = [r["cost"] for r in rows]
        total = sum(values)
        avg = int(total / len(rows))
        summary = f"近 8 个月差旅及业务成本累计 {total:,} 元，月均 {avg:,} 元，8 月为峰值 {max(values):,} 元。"
        chart = {"type": "line", "title": "月度差旅及业务成本（元）", "labels": labels, "values": values}
        return {"metric": "成本", "summary": summary, "chart": chart}

    if key == "hotel":
        rows = _mock_occupancy()
        labels = [r["month"] for r in rows]
        values = [int(r["rate"] * 100) for r in rows]
        avg = int(sum(r["rate"] for r in rows) / len(rows) * 100)
        peak = max(rows, key=lambda r: r["rate"])
        summary = f"近 9 个月平均入住率约 {avg}%，暑期(7-8月)最高达 {int(peak['rate']*100)}%（{peak['month']}）。"
        chart = {"type": "bar", "title": "月度酒店入住率（%）", "labels": labels, "values": values}
        return {"metric": "酒店入住率", "summary": summary, "chart": chart}

    # overview：多指标概览
    total_salary = sum(e["base_salary"] for e in employees)
    dept_count = len({e["department"] for e in employees})
    avg_occ = int(sum(r["rate"] for r in _mock_occupancy()) / 9 * 100)
    summary = (
        f"业务概览：共 {len(employees)} 名员工，{dept_count} 个部门，月薪合计 {total_salary:,} 元；"
        f"近 9 个月酒店平均入住率约 {avg_occ}%。输入具体指标（如'员工成本'、'酒店入住率'）可查看明细图表。"
    )
    # 概览用一个综合柱状图（各部门差旅预算）
    by_dept: dict = {}
    for e in employees:
        by_dept.setdefault(e["department"], 0)
        by_dept[e["department"]] += e["travel_budget"]
    chart = {"type": "bar", "title": "各部门差旅预算（元）", "labels": list(by_dept.keys()), "values": [v for v in by_dept.values()]}
    return {"metric": "概览", "summary": summary, "chart": chart}