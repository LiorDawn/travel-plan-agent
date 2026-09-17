"""回归测试（补充）— 与 tests/test_security_fixes.py 分工，只保留其未覆盖的两项：

- 记忆沉淀去重（P1-4）：重复沉淀不再调用 insert_chunk 入库
- chat_service.stream 异常路径仍冲刷审计（N2 try/finally）

其余（artifact TTL / 审计隔离 / 缓存 user_id / weather 降级 / 鉴权）已在
tests/test_security_fixes.py 覆盖，此处不重复。

零 token：mock DB / mock Redis / mock Agent，不联网。
"""
import asyncio
from unittest.mock import patch


# ---------- P1-4 记忆沉淀去重 ----------
def test_persist_conversation_skips_duplicate():
    async def case():
        calls = []

        async def fake_insert(piece, source_type=None, source_id=None, metadata=None):
            calls.append(piece)
            return True

        with patch("app.memory.rag.service.insert_chunk", fake_insert), \
             patch("app.memory.rag.service._is_duplicate", return_value=True):
            from app.memory.rag.service import persist_conversation
            ok = await persist_conversation("问题", "回答", source_id="c1", user_id="u1")
        return ok, calls
    ok, calls = asyncio.run(case())
    assert ok is True   # 视为已存在等价记忆
    assert calls == []  # 不再重复入库


# ---------- N2 stream 异常路径仍 flush 审计 ----------
def test_stream_flushes_audit_even_on_error():
    async def case():
        from app.services.chat_service import ChatService
        from app.schemas.chat import ChatRequest

        svc = ChatService()

        class BoomAgent:
            async def run(self, *a, **k):
                if False:
                    yield None  # 保持 async generator 语义
                raise RuntimeError("boom")

        svc.agent = BoomAgent()
        flushed = []

        async def fake_flush(db, conv_id):
            flushed.append(conv_id)

        svc._audit_flush = fake_flush

        async def fake_user(db, uid, name):
            return None

        async def fake_conv(db, cid, uid, title):
            return None

        async def fake_msg(db, cid, role, content):
            return None

        async def fake_msgs(db, cid, limit=None):
            return []

        with patch("app.services.chat_service.long_term.get_or_create_user", fake_user), \
             patch("app.services.chat_service.long_term.save_conversation", fake_conv), \
             patch("app.services.chat_service.long_term.save_message", fake_msg), \
             patch("app.services.chat_service.long_term.get_conversation_messages", fake_msgs):
            req = ChatRequest(user_id="u1", message="hi")
            try:
                async for _ev in svc.stream(req, db=None):
                    pass
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError from agent.run")
        return flushed
    flushed = asyncio.run(case())
    assert flushed, "异常路径也必须冲刷审计（try/finally）"
