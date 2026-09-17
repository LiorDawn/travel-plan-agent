"""langgraph / langchain 版本兼容层（零依赖变更）

背景：项目 pyproject 锁 `langchain ^0.3` / `langgraph ^0.2.50`，但本环境实际安装的是
langchain 1.1.x + langgraph 1.0.10。langchain 1.1.x 在 `langgraph.prebuilt.tool_node`
里执行 `from langgraph.runtime import ExecutionInfo, ServerInfo`，而 langgraph 1.0.10 的
`langgraph.runtime` 未导出这两个名字，导致 `import langchain.agents` —— 进而任何子图
（create_agent / create_react_agent）入口 —— 直接 ImportError。

已核实：`ExecutionInfo / ServerInfo` 在 tool_node 中仅用于**类型注解**
（`x: ExecutionInfo | None = None`），运行时不被实例化或调用，因此可以在
`langgraph.runtime` 模块上补占位符号即可化解崩溃，且不改变任何运行语义。

用法：在任何 `import langchain.agents` / `langchain.agents.middleware` 之前调用
`ensure()`。幂等：重复调用安全。若环境 langgraph 已具备这两个符号，ensure() 自动跳过。
"""
from __future__ import annotations

_INSECTED = False

# 待补齐的模块级符号（均为工具内部的类型注解占位）
_NEEDED = ("ExecutionInfo", "ServerInfo")


def ensure() -> None:
    """确保 langgraph.runtime 具备 langchain 1.x 需要的注解符号；幂等。"""
    global _INSECTED
    if _INSECTED:
        return
    try:
        import langgraph.runtime as rt  # noqa: PLC0415
        for name in _NEEDED:
            if not hasattr(rt, name):
                # 仅作为类型注解占位，无需真实实现
                setattr(rt, name, type(name, (), {}))
        _INSECTED = True
    except Exception:  # 无 langgraph 或已具备时忽略，不阻塞启动
        _INSECTED = True


# 导入即生效，保证即使是"首次 import 本模块"的入口也能自动完成注入。
ensure()