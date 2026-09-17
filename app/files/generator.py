"""文件管理 — Markdown 生成、版本管理、预览"""
import hashlib
from datetime import datetime
from jinja2 import Template


PLAN_TEMPLATE = Template("""\
# {{ plan.get('title', '旅游方案') }}
> 状态: {{ plan.get('status', 'draft') }} | 生成时间: {{ now }}

## 基本信息
- **目的地**: {{ plan.get('destination', '待定') }}
- **行程天数**: {{ plan.get('duration', '待定') }}
- **方案概述**: {{ plan.get('summary', '暂无') }}

## 每日行程
{% for day in plan.get('days', []) %}
### 第 {{ day.day }} 天{% if day.date %} ({{ day.date }}){% endif %}
{% for act in day.get('activities', []) %}
- **{{ act.time }}** {{ act.activity }} @ {{ act.location }}
{% if act.note %}  > {{ act.note }}{% endif %}
{% endfor %}
{% endfor %}

## 交通信息
{% for t in plan.get('transportations', []) %}
- **{{ t.get('type', '交通') }}**: {{ t.get('from', '') }} → {{ t.get('to', '') }} | {{ t.get('time', '') }} | ¥{{ t.get('cost', 0) }}
{% endfor %}

## 住宿信息
{% for h in plan.get('hotels', []) %}
- **{{ h.get('name', '酒店') }}** | {{ h.get('address', '') }} | ¥{{ h.get('cost_per_night', 0) }}/晚
  > 入住: {{ h.get('check_in', '') }} | 退房: {{ h.get('check_out', '') }}
{% endfor %}

## 预算明细
{% set b = plan.get('budget', {}) %}
| 类别 | 金额 |
|------|------|
| 交通 | ¥{{ b.get('transport', 0) }} |
| 住宿 | ¥{{ b.get('hotel', 0) }} |
| 餐饮 | ¥{{ b.get('food', 0) }} |
| 其他 | ¥{{ b.get('other', 0) }} |
| **总计** | **¥{{ b.get('total', 0) }}** |

## 注意事项
{% for tip in plan.get('tips', []) %}
- {{ tip }}
{% endfor %}

---
*方案由 AI 旅游规划助手生成*
""")


def generate_markdown(plan: dict) -> str:
    """渲染方案为 Markdown"""
    return PLAN_TEMPLATE.render(plan=plan, now=datetime.now().strftime("%Y-%m-%d %H:%M"))


def compute_hash(content: str) -> str:
    """计算内容 MD5 摘要, 用于文件内容去重/变更检测。"""
    return hashlib.md5(content.encode()).hexdigest()


def markdown_to_html(markdown_content: str) -> str:
    """Markdown → HTML 预览（安全净化版）

    未安装 mistune/bleach 时，先用 html.escape 转义源码中所有特殊字符，
    再插入受控的结构化标签，保证 LLM/API 注入的 <script> 等不生效。
    生产环境建议替换为 mistune + nh3 白名单净化。
    """
    import re
    import html

    # 先整体转义，阻断任何 HTML/脚本注入
    safe = html.escape(markdown_content, quote=True)

    # 标题（按 # 数从大到小，避免子串误伤）
    for i in range(6, 0, -1):
        prefix = "#" * i
        pattern = re.compile(rf"(?m)^{prefix} (.+?)$")
        safe = pattern.sub(f"<h{i}>\\1</h{i}>", safe)

    # 粗体 / 斜体
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<em>\1</em>", safe)

    # 列表（简化为 ul/li，行首 "- " 或 "* "）
    def _list_repl(m):
        items = re.sub(r"^(?:-|\*|\d+\.)\s+", "<li>", m.group(0).strip(), flags=re.M)
        items = items.replace("\n", "</li><li>").removesuffix("<li>")
        return "<ul>" + items + "</ul>"

    safe = re.sub(r"(?m)((?:^\s*(?:-|\*|\d+\.)\s+.+\n)+)", _list_repl, safe)

    # 引用块
    def _quote_repl(m):
        lines = "<br>".join(re.sub(r"^\s*>\s?", "", ln) for ln in m.group(0).strip().splitlines())
        return f"<blockquote>{lines}</blockquote>"

    safe = re.sub(r"(?m)((?:^\s*>\s?.+\n?)+)", _quote_repl, safe)

    # 分割线
    safe = re.sub(r"(?m)^---+$", "<hr>", safe)

    # 行内代码
    safe = re.sub(r"`(.+?)`", r"<code>\1</code>", safe)

    body = safe.replace("\n", "<br>")
    return f"<html><body style='font-family:sans-serif;max-width:800px;margin:20px auto;line-height:1.8'>{body}</body></html>"
