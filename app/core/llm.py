import contextvars
from langchain.chat_models import init_chat_model
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()

# 单次请求内累计 LLM usage（contextvar 按任务/协程隔离，避免并发请求串数据）
_usage_var: contextvars.ContextVar = contextvars.ContextVar("llm_usage", default=None)


class LLMClient:
    """LLM 客户端 — 基于 LangChain init_chat_model，统一 Ollama / DashScope 多 provider。

    用 init_chat_model 一行按 provider 初始化模型，替代手写 provider 分支；底层
    走 langchain-openai / langchain-ollama 的真实实现（含调用、usage、重试）。
    """

    def __init__(self):
        self.provider = settings.llm_provider
        self._model = self._build_model()
        logger.info("llm_init", provider=self.provider,
                    model=getattr(self._model, "model_name", str(self._model)))

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def _build_model(self):
        """按 provider 构建模型。DashScope 走 OpenAI 兼容端，Ollama 走本地端。"""
        if self.provider == "dashscope":
            model = init_chat_model(
                settings.dashscope_model,
                model_provider="openai",
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )
            self.model_name = settings.dashscope_model
        else:
            model = init_chat_model(
                settings.ollama_model,
                model_provider="ollama",
                base_model=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
            self.model_name = settings.ollama_model
        return model

    @property
    def raw_model(self):
        """暴露底层 ChatModel 实例，供 create_agent 子图复用（子图需要原生 LangChain 模型）。"""
        return self._model

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    async def chat(self, system_prompt: str, user_message: str, max_tokens: int | None = None) -> str:
        """纯文本调用。max_tokens 通过 bind 绑定到模型（直接塞 invoke kwargs 不生效）。"""
        msgs = [("system", system_prompt), ("user", user_message)]
        model = self._model.bind(max_tokens=max_tokens) if max_tokens else self._model
        resp = await model.ainvoke(msgs)
        self._accumulate_usage(resp)
        return resp.content

    # ------------------------------------------------------------------
    # 统一结构化入口（官方 with_structured_output）
    # ------------------------------------------------------------------
    async def chat_agent_structured(self, system_prompt: str, user_message: str, output_model,
                                    max_tokens: int | None = None, max_retries: int = 2) -> dict:
        """全项目统一结构化入口 — 2026 主流分层链路。

        with_structured_output（按 provider 自动走工具绑定 / JSON mode）→ Pydantic 校验
        → 带错误反馈重试 → 原始文本解析降级。
        纯 schema 提取用 with_structured_output 即可，无需 create_agent 起整套工具循环。
        """
        return await self._with_structured_output(system_prompt, user_message, output_model, max_tokens, max_retries)

    async def _with_structured_output(self, system_prompt: str, user_message: str,
                                      output_model, max_tokens: int | None = None, max_retries: int = 2) -> dict:
        """官方 with_structured_output（主力）+ 校验重试 + 原始文本降级兜底。

        分层策略（对应 2026 企业主流）：
        1. 主力：with_structured_output —— LangChain 按 provider 自动选工具绑定或 JSON mode。
        2. 校验：model_dump 强类型约束。
        3. 重试：解析失败时把校验错误反馈回模型，限制 max_retries 次。
        4. 兜底：仍失败则从原始文本抽取 JSON 解析，保证流程不断（最后一层）。
        """
        model = self._model.bind(max_tokens=max_tokens) if max_tokens else self._model
        # DashScope(openai 兼容端) 对 qwen 走 function_calling 解析不稳(易返回字符串常量),
        # 改用 json_mode 更可靠; Ollama 保持默认 function_calling。
        method = "json_mode" if self.provider == "dashscope" else None
        structured = model.with_structured_output(output_model, method=method, include_raw=True)
        msgs: list = [("system", system_prompt), ("user", user_message)]
        last_raw, last_err = None, ""

        for _ in range(max_retries + 1):
            result = await structured.ainvoke(msgs)
            self._accumulate_usage(result.get("raw"))
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed.model_dump()
            # 校验失败：记下原始输出与错误，构造带错误反馈的新一轮重试
            last_raw = result.get("raw")
            last_err = str(result.get("parsing_error") or "解析失败")
            msgs.append(("assistant", getattr(last_raw, "content", "") or ""))
            msgs.append((
                "user",
                f"你上次的输出不符合要求，校验错误：{last_err}\n"
                f"请严格按以下 JSON Schema 重新输出：\n{output_model.model_json_schema()}",
            ))

        # 兜底层：从原始文本抽取 JSON 解析（最后退路，不再抛异常硬断流程）
        if last_raw is not None:
            text = last_raw.content if hasattr(last_raw, "content") else str(last_raw)
            try:
                return output_model.model_validate_json(text).model_dump()
            except Exception:
                pass
        raise ValueError(f"结构化输出多次失败: {last_err}")

    # ------------------------------------------------------------------
    # usage 累计（真实 token 数）
    # ------------------------------------------------------------------
    def _accumulate_usage(self, response) -> None:
        """把单次响应的 usage 累加进当前任务上下文"""
        usage = {}
        if isinstance(response, dict):
            usage = response.get("usage_metadata", {}) or {}
        else:
            usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            return
        cur = _usage_var.get()
        if cur is None:
            cur = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
        cur["input_tokens"] += usage.get("input_tokens", 0)
        cur["output_tokens"] += usage.get("output_tokens", 0)
        cur["total_tokens"] += usage.get("total_tokens", 0)
        cur["calls"] += 1
        _usage_var.set(cur)

    def reset_usage(self) -> None:
        """开始一次 run/resume 前清零累计"""
        _usage_var.set(None)

    def get_usage(self) -> dict:
        """取当前任务累计的 usage"""
        cur = _usage_var.get()
        if cur is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
        return dict(cur)


llm = LLMClient()