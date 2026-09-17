"""OpenTelemetry 链路追踪 — 官方 SDK + OTLP，未安装 SDK 时自动降级为轻量日志。

设计：
- OTel SDK 为可选依赖（懒加载）。SDK 缺装时降级为 stdlib logging 记录 span 起止，不影响业务。
- 统一对外接口 start_span(name, **attrs) → span 对象（.set_attr / .end），业务无需感知底层，
  支持手动管理生命周期（跨异步 start/end，如 runner 的记录点）。
- trace_span() 提供 contextmanager 便捷封装（同步场景用）。
- 启用开关/OTLP 端点来自 settings.otel_*。

典型用法（runner 节点链路）：
    spans[name] = start_span(f"agent.node.{name}")
    ...
    spans[name].set_attr("status", "ok"); spans[name].end()
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.core.config import get_settings

logger = logging.getLogger("otel")

settings = get_settings()

# ------------------------- 懒加载 OTel（未安装 SDK 则为 None） -------------------------
_tracer = None
_USE_OTEL = False


def _try_init_otel():
    """首次初始化 OTel tracer；失败/未启用则维持降级。"""
    global _tracer, _USE_OTEL
    if _tracer is not None:
        return _USE_OTEL

    if not settings.otel_enabled:
        _tracer = "noop"
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        provider = TracerProvider(
            resource=Resource.create(attributes={
                "service.name": settings.otel_service_name,
            })
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_otlp_endpoint))
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.otel_service_name)
        _USE_OTEL = True
        logger.info("otel_enabled", extra={"endpoint": settings.otel_otlp_endpoint})
    except Exception as e:  # SDK 未装 / 导入失败 → 降级日志
        logger.warning("otel_disabled_fallback", extra={"error": str(e)})
        _tracer = "noop"
    return _USE_OTEL


class BaseSpan:
    """span 通用接口：set_attr 记录属性，end 结束（降级实现退出后无操作）。"""

    def set_attr(self, key: str, value):  # noqa: D401
        raise NotImplementedError

    def end(self):  # noqa: D401
        raise NotImplementedError


class _LogSpan(BaseSpan):
    """降级 span：start 记录一条日志，end 时补耗时。"""

    def __init__(self, name: str, attrs: dict):
        self._name = name
        self._attrs = dict(attrs or {})
        self._start = time.time()
        self._ended = False
        logger.info("span.start", extra={"name": name, **self._attrs})

    def set_attr(self, key: str, value):
        self._attrs[key] = str(value)

    def end(self):
        if self._ended:
            return
        self._ended = True
        logger.info("span.end", extra={
            "name": self._name, "cost_ms": round((time.time() - self._start) * 1000, 1), **self._attrs,
        })


class _OtelSpan(BaseSpan):
    """OTel 真实 span。"""

    def __init__(self, span):
        self._span = span

    def set_attr(self, key: str, value):
        if self._span is not None:
            self._span.set_attribute(key, str(value))

    def end(self):
        if self._span is not None:
            self._span.end()


def start_span(name: str, **attrs) -> BaseSpan:
    """开启一个 span 并返回句柄：使用方负责在结束时调用 .end()。

    支持跨异步生命周期：可在 on_chain_start 打开、on_chain_end 关闭。
    """
    _try_init_otel()
    if _USE_OTEL:
        span = _tracer.start_span(name)
        if span is not None:
            span.set_attribute("span.kind", "internal")
            for k, v in attrs.items():
                span.set_attribute(k, str(v))
        return _OtelSpan(span)
    return _LogSpan(name, attrs)


@contextmanager
def trace_span(name: str, **attrs) -> Iterator[BaseSpan]:
    """便捷 contextmanager：with 块结束自动 .end()（同步场景）。"""
    span = start_span(name, **attrs)
    try:
        yield span
    finally:
        span.end()