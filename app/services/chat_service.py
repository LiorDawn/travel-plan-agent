"""聊天用例编排 — 协调 Agent 运行、持久化、token 统计，向 API 层暴露事件流"""
from typing import AsyncIterator, List
import json, time, uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.events import ChatEvent
from app.agent.runner import TravelPlanAgent
from app.memory import long_term
from app.memory.audit.recorder import begin_request, CtxRequestID
from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.chat import ChatRequest
from app.services.token_stats import TokenStats

settings = get_settings()

# 短期记忆窗口：注入最近 ROUNDS 轮对话原文，预算字符数（保证不撑爆外层 plan/summarize prompt）
_RECENT_ROUNDS = 3
_RECENT_BUDGET_CHARS = 2000


def _fmt_recent_context(messages) -> str:
    """把最近几轮对话原文格式化为"user/assistant: ..."的窗口文本，保存话题延续/指代。"""
    lines = []
    for m in messages:
        role = getattr(m, "role", "")
        content = getattr(m, "content", "") or ""
        line = f"{role}: {content}"
        if role == "assistant":
            line = line.rstrip()[:500]  # 助手回复作窗口时截断，避免占满预算
        lines.append(line)
        if sum(len(l) for l in lines) > _RECENT_BUDGET_CHARS:
            lines = lines[-2:]  # 超预算时只保留最近一对，保证窗口不撑爆
            break
    return "\n".join(lines)


def _answers_natural(request_message: str) -> str:
    """把澄清 resume 载荷（clarity 前端提交的 JSON）转成自然语言文本，供落库与短期窗口展示。

    仅当 request_message 是 {"answers":{...}} 才转换；其余情况原样返回。
    """
    if not request_message or not request_message.strip():
        return request_message
    try:
        parsed = json.loads(request_message)
        answers = parsed.get("answers")
        if isinstance(parsed, dict) and isinstance(answers, dict) and answers:
            return "；".join(f"{k}: {v}" for k, v in answers.items())
    except (json.JSONDecodeError, TypeError):
        return request_message
    return request_message


class ChatService:
    """对话用例编排：run/resume 事件流 + 持久化 + token 统计"""

    def __init__(self):
        self.agent = TravelPlanAgent()
        # token 统计落 Redis（跨 worker/重启持久化），进程内存仅作 Redis 不可用兜底
        self.token_stats = TokenStats()

    def _model_name(self) -> str:
        """按 provider 返回当前使用的模型名。"""
        return settings.dashscope_model if settings.llm_provider == "dashscope" else settings.ollama_model

    @staticmethod
    def _append_content(assistant_content: List[str], data: dict) -> None:
        """从非 done 事件中提取文本内容，追加到 assistant_content"""
        if data.get("plan"):
            assistant_content.append(json.dumps(data["plan"], ensure_ascii=False))
        elif data.get("message"):
            assistant_content.append(data["message"])
        elif data.get("content"):
            assistant_content.append(data["content"])

    @staticmethod
    def _extract_intent(event: ChatEvent) -> str:
        """从 plan 结果事件里取 intent（用作 usage 聚合维度）。"""
        if event.event_type == "plan.result":
            return event.data.get("intent", "unknown")
        return ""

    async def _finalize_done(self, event: ChatEvent, start_time: float, db: AsyncSession,
                             conv_id: str, assistant_content: List[str], intent: str,
                             input_est: int = 0, include_tokens: bool = False):
        """done 事件统一收官：补耗时/模型信息、累计 token（真实 usage 优先）、持久化助手回复"""
        event.data["elapsed"] = round(time.time() - start_time, 2)
        event.data["model"] = self._model_name()

        # 真实 usage 从 done 事件透传（runner 已把 llm.get_usage() 塞进 data.usage）
        real_usage = event.data.get("usage") or {}
        if real_usage.get("total_tokens"):
            await self.token_stats.incr(conv_id, intent, real_usage)
            event.data["tokens"] = {
                "input": real_usage.get("input_tokens"),
                "output": real_usage.get("output_tokens"),
                "total": real_usage.get("total_tokens"),
                "intent": intent,
            }
        else:  # 降级：无真实 usage 时按估算值计
            est = {"input_tokens": input_est, "output_tokens": 500,
                   "total_tokens": (input_est or 0) + 500, "calls": 1}
            await self.token_stats.incr(conv_id, intent, est)
            if include_tokens:
                event.data["tokens"] = {"input": input_est, "output": 500, "intent": intent}
        if assistant_content:
            await long_term.save_message(db, conv_id, "assistant", "\n\n".join(assistant_content))

    async def stream(self, request: ChatRequest, db: AsyncSession) -> AsyncIterator[ChatEvent]:
        """统一对话入口 — 自动判断新对话 vs 续跑，产出 SSE 事件流"""
        start_time = time.time()
        assistant_content: List[str] = []

        # F4：绑定本请求级审计 recorder（并发 SSE 隔离）。request_id 来自中间件；
        #     中间件未置（如直连/测试）时回退到会话 id。
        begin_request(CtxRequestID.get() or request.conversation_id or "")

        # 新对话：持久化用户和对话
        conv_id = request.conversation_id
        intent = "unknown"
        if not request.resume:
            conv_id = conv_id or str(uuid.uuid4())
            await long_term.get_or_create_user(db, request.user_id, request.user_id)
            await long_term.save_conversation(db, conv_id, request.user_id, request.message[:50])
            await long_term.save_message(db, conv_id, "user", request.message)

        # 短期窗口：取本会话最近 ROUNDS 轮原文（去掉刚写入的当前句），保"那家/上次"指代
        recent_context = ""
        try:
            recent_msgs = await long_term.get_conversation_messages(
                db, conv_id, limit=_RECENT_ROUNDS * 2 + 1)
            recent_msgs = recent_msgs[:-1] if recent_msgs else []  # 去掉最新一条(=当前 user 句)
            recent_context = _fmt_recent_context(recent_msgs[-_RECENT_ROUNDS * 2:])
        except Exception as e:
            logger.warning("recent_context_load_failed", error=str(e))

        try:
            if request.conversation_id and request.resume:
                logger.info("chat_resume", conv=request.conversation_id)
                # 落库/短期窗口用自然语言（防原始 JSON 污染历史与指代理解）；
                # 传给 agent.resume 的仍是 JSON 原文（clarify 靠它 json.loads 取 answers）
                await long_term.save_message(db, request.conversation_id, "user",
                                             _answers_natural(request.message))
                async for event in self.agent.resume(request.message, request.conversation_id):
                    if event.event_type == "done":
                        await self._finalize_done(event, start_time, db, request.conversation_id,
                                                  assistant_content, intent)
                    else:
                        intent = self._extract_intent(event) or intent
                        self._append_content(assistant_content, event.data)
                    yield event
            else:
                async for event in self.agent.run(request.message, request.user_id, conv_id,
                                                  thinking_mode=request.thinking_mode,
                                                  recent_context=recent_context):
                    if event.event_type == "done":
                        await self._finalize_done(
                            event, start_time, db, conv_id, assistant_content, intent,
                            input_est=len(request.message) // 2, include_tokens=True,
                        )
                    else:
                        intent = self._extract_intent(event) or intent
                        self._append_content(assistant_content, event.data)
                    yield event
        finally:
            # F4/N2：无论流正常结束、中途抛错还是客户端断连（GeneratorExit），
            # 都冲刷本请求审计；旁路失败不阻塞主流程。
            await self._audit_flush(db, conv_id)

    async def _audit_flush(self, db: AsyncSession, conv_id: str) -> None:
        """把本请求攒下的审计事件批量落库；无论成败都不抛错（旁路）。"""
        try:
            from app.memory.audit.sink import flush_audit
            await flush_audit(db, request_id=conv_id)
        except Exception as e:
            logger.warning("audit_flush_final_failed", error=str(e))

    async def monitor(self) -> dict:
        """token 使用统计（落 Redis / Redis 不可用时内存兜底），含按意图/会话的真实 token 聚合。"""
        return await self.token_stats.snapshot()