"""RAG 检索增强 — 完整加工流水线

升级点（对标 docs/亮点功能清单.md 第一项）：
1. 文本切片：长文本按 chunk_size/overlap 切块入库，检索粒度更细、召回更准。
2. 查询改写：LLM 把模糊问题改写成可检索的关键词（降级为原始词兜底）。
3. 关键词增强：改写出的关键词做关键字召回（PG 的 ILIKE 全文匹配）。
4. 多路召回融合：向量（PgVector）+ 关键字（ILIKE）双路召回，RRF 重排融合。
5. Rerank + 阈值 + 来源标注：融合分数排序，低相关阈值过滤，结果带 source 字段。

依赖 DashScope text-embedding-v3（OpenAI 兼容 /embeddings 端点）。
"""
from sqlalchemy import select
from pgvector.sqlalchemy import Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from app.core.database import AsyncSessionLocal
from app.core.config import get_settings
from app.core.logging import logger
from app.memory.rag.models import KnowledgeChunk  # 用于 insert_chunk/insert_text

settings = get_settings()

# 懒加载的单例 openai 客户端（BASE_URL=兼容网关；初始化失败会触发设置异常）
_OPENAI_CLIENT = None


def _get_openai_client():
    """【作用】返回缓存的 openai 异步客户端（懒加载，进程内复用连接）。"""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import AsyncOpenAI  # 延迟 import, 避免非向量化路径无谓依赖

        _OPENAI_CLIENT = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=30,
            max_retries=1,
        )
    return _OPENAI_CLIENT

# 余弦相似度阈值(0~1), 低于此认为不够相关, 返回空避免污染上下文
_SIMILARITY_THRESHOLD = 0.3

# RRF 融合常数（越小越偏向高排位结果）
_RRF_K = 60

# 时间衰减参数：半衰期=30天，下界不低于0.2（保长期偏好不被压死）
_RAG_DECAY_HALFLIFE_DAYS = 30
_RAG_DECAY_MIN_WEIGHT = 0.2

# 父子块大小参数（攻略类）: 小块(350字)进向量库召回; 大块(800~1200字)喂给LLM
_RAG_CHILD_SIZE = 350  # 小块字符数
_RAG_CHILD_OVERLAP = 35  # 小块重叠
_RAG_PARENT_MAX = 1200  # 大块最大字符数

# 上下文注入预算：检索结果回溯成大块(parent)后可能很长(top_k 大块会撑爆 Prompt),
# 拼进 LLM Prompt 的累计字符上限, 命中内容逐个累加, 达到即停止注入。
_RAG_CONTEXT_BUDGET_CHARS = 2500


# ---------------------------------------------------------------- 向量化
async def get_embedding(text: str) -> list[float]:
    """【作用】把一段文本转成稠密向量，作为 RAG 向量化的兼容单条入口。

    【缓存】embedding 结果按 text 缓存（Redis+内存），同查询/同块复用直接命中，
    跳过网络往返——这是子图高频检索降到近零成本的收益核心。
    """
    if getattr(settings, "rag_embed_cache_enabled", True):
        from app.memory.rag.cache import embedding_cache_get
        hit = await embedding_cache_get(text, settings.rag_embed_cache_ttl)
        if hit is not None:
            return hit
    batch = await _embed_many([text])
    vec = batch[0] if batch else []
    if vec and getattr(settings, "rag_embed_cache_enabled", True):
        from app.memory.rag.cache import embedding_cache_set
        await embedding_cache_set(text, vec, settings.rag_embed_cache_ttl)
    return vec


# openai SDK 单请求支持的最大输入条数（超出自动分批）
_OAI_MAX_INPUTS = 25


async def _embed_many(texts: list[str]) -> list[list[float]]:
    """【作用】批量 embedding：多段文本一次网络往返（P1，收益最大）。

    【逻辑】openai SDK 支持单请求多输入，把 N 段文本按 _OAI_MAX_INPUTS 分批，每批一次
    /embeddings 调用，把各条结果按输入顺序对齐返回；任一批失败则该批置空向量(长度0)，
    不影响其它批，最终长度与输入对齐保证调用方可 zip。

    【实现】对 texts 每 _OAI_MAX_INPUTS 条切一批，逐批 await embeddings.create(input=batch)；
    每批按返回顺序码成 list，追加进 results。维度不符时统一告警一次。
    """
    _client = _get_openai_client()
    results: list[list[float]] = []
    for i in range(0, len(texts), _OAI_MAX_INPUTS):
        batch = texts[i:i + _OAI_MAX_INPUTS]
        try:
            resp = await _client.embeddings.create(
                model=settings.dashscope_embedding_model,
                input=batch,
            )
            # 服务端按输入顺序返回 data
            for item in resp.data:
                vec = item.embedding
                if vec and len(vec) != settings.rag_dim:
                    logger.warning("rag_embedding_dim_mismatch", got=len(vec), expect=settings.rag_dim)
                results.append(list(vec))
            # 若返回条数少于批次(异常), 补齐为 0 向量占位, 保证 zip 对齐
            for _ in range(len(batch) - len(resp.data)):
                results.append([])
        except Exception as e:
            logger.warning("rag_embedding_batch_failed", start=i, size=len(batch), error=str(e))
            results.extend([] for _ in batch)
    return results


# ---------------------------------------------------------------- 1. 文本切片
def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """【作用】把长文本按 chunk_size/overlap 切成若干块，用于入库前切分，保证检索粒度。

    【逻辑】用 langchain-text-splitters 的 RecursiveCharacterTextSplitter 递归切分：
    按「段落 → 换行 → 句号 → 分隔符」优先级逐级回退分割，块尽量落在语义完整处，
    并保留 chunk_size/overlap 重叠，避免语义被切在中间。比手写游标更健壮、少维护边界情况。

    【实现】复用库内置切分器（默认分隔符覆盖中英文本），返回文本块列表；空文本返回 []。
    """
    if chunk_size is None:
        chunk_size = settings.rag_chunk_size
    if overlap is None:
        overlap = settings.rag_chunk_overlap

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return [d.strip() for d in splitter.split_text(text) if d and d.strip()]


# ---------------------------------------------------------------- 结构化切分（攻略类）
def _split_long_para(text: str, max_chars: int, min_size: int) -> list[str]:
    """【作用】对单段超长文本，按句子切分，切在句子标点处、保证句子完整。

    【逻辑】是"单段超长 → 父块固定截断"的兜底：先按句子标点(。！？；！?;)把文本切成
    句列表，再贪心拼接句子凑成 [min_size~max_chars] 之间的整块。这样每块不入句子、不劈语义，
    也不做 1200 硬截断。若"无任何句子断点"(如连续无标点串), 则退化逐个按 max_chars 硬切
    (分成约 N 块)——这是所有切分器都到不了的极限情况, 至少保证块长有上限、可入库。

    【实现】re.split 按中文/英文句子标点拆句(保留标点); 贪心拼句凑块达到 max_chars 收拢;
    单句超上限则独立成块; 最后归并过小块到 min_size 以上。无句子断点时 range 步进 max_chars
    硬切。返回切好的文本块列表。
    """
    import re

    parts = re.split(r'(?<=[。！？；；!?;])\s*', (text or "").strip())
    sentences = [s.strip() for s in parts if s and s.strip()]
    if len(sentences) <= 1:                  # 无句子断点: 硬切兜底, 确保块长有上限
        raw = (text or "").strip()
        if not raw:
            return []
        if len(raw) <= max_chars:
            return [raw]
        return [raw[i:i + max_chars] for i in range(0, len(raw), max_chars)]
    blocks: list[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:                 # 达到上限且积累超过 min_size 就收拢
                blocks.append(cur)
                cur = s
            else:                   # 单句本身就超上限: 直接读成独立块(交给上层处理)
                blocks.append(s)
    if cur:
        blocks.append(cur)
    # 归并过小块: 确保每块不短于 min_size(不足则并入前一块), 贴合 400~1200 预期
    merged: list[str] = []
    for b in blocks:
        if merged and len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + b
        elif merged and len(merged[-1]) + len(b) <= max_chars:
            merged[-1] = merged[-1] + b
        else:
            merged.append(b)
    return [b for b in merged if b]


def _preclean(text: str, min_break_len: int = 60) -> str:
    """【作用】把"无任何断点"的超长硬文本清洗成可切文本(最治本)。

    【逻辑】所有切分器(langchain 递归/句子切)都靠分隔符断块；若某段是连续几千字的"无空格、
    无标点串"(如 base64、URL、连续中文无标点), 任何分句/递归切都会在块边界"劈字/劈词"。
    预防思路是在入库前主动清洗: 把"连续可写字符(random isalnum 含中文)长度超 min_break_len"
    处插入一个 \n 作切分提示, 让后面的切分器总能落到整齐断点, 不再依赖凑巧的标点。

    【实现】单遍扫描: 遇断点(空格/标点/符号)保留并重置连续计数 run; 遇可写字符累积 run,
    达到 min_break_len 就在其后补 \n 并重置——使任何超长连续段都被周期性打断成可切片段。
    短段(< min_break_len)原样返回。不删除、不改动原文内容, 仅追加 \n 提示。
    """
    if len(text) <= min_break_len:
        return text
    out: list[str] = []
    run = 0
    for ch in text:
        out.append(ch)
        if ch.isspace() or not ch.isalnum():   # 空格/标点/符号 = 自然断点
            run = 0
        else:
            run += 1
            if run >= min_break_len:            # 连续可写字符超长 → 插 \n 打断
                out.append("\n")
                run = 0
    return "".join(out)


def _aggregate_plain_sections(text: str, max_chars: int) -> list[dict]:
    """【作用】对无标题的纯文本，按段落聚合出多个 ≤ max_chars 的节。

    【逻辑】修"无标题时整篇降级成一节、下游被迫固定截断"的源头：把文本按空行拆成段落，
    依次合并进当前节，直到满 max_chars 才另起新节——因此每节都 ≤ max_chars，且尽量留在
    段落边界，下游 parent 入库就不再触发 `[:max]` 固定截断。

    【实现】re.split 按 `\n\n`(空行)拆段落；贪心累积到超过 max_chars 即收拢成 {"title":"","content":cur}
    并另起新节；结尾残留若存在同样收拢。返回 [{title:"",content}, ...]。
    """
    import re

    paras = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    if not paras:
        return []
    sections: list[dict] = []
    cur = ""
    for p in paras:
        # 单段超长: 先用 _split_long_para 按句子切(句子完整, 块落在 min_size~max_chars),
        # 保证每节 ≤ max_chars 且不劈语义; 无句子断点则整段返回, 交下游 _preclean+硬切兜底。
        if len(p) > max_chars:
            for piece in _split_long_para(p, max_chars, min_size=350):
                if cur:
                    sections.append({"title": "", "content": cur})
                    cur = ""
                sections.append({"title": "", "content": piece})
            continue
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur += "\n" + p
        else:
            sections.append({"title": "", "content": cur})
            cur = p
    if cur:
        sections.append({"title": "", "content": cur})
    return sections


def split_markdown_by_headings(text: str) -> list[dict]:
    """【作用】把带标题的 Markdown 攻略按标题层级 (#/##/###) 切分成"大块"。

    【逻辑】攻略天然按标题分节（如 "## 厦门景点"），按标题切能保证每块语义完整，
    且把标题提取出来作为 title，供检索标注/过滤。大块超长(_RAG_PARENT_MAX)时用
    RecursiveCharacterTextSplitter 递归切分，避免喂给 LLM 时超上下文。

    【实现】用 langchain 的 MarkdownHeaderTextSplitter 按 #/##/### 拆分（返回 Document，
    metadata 含各级标题，page_content 为正文）；取最细层级标题作 title，超长大节再过
    RecursiveCharacterTextSplitter 二次切分。比手写逐行扫描更健壮。返回 {"title","content"}。
    """
    if not text:
        return []
    splitter = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")])
    try:
        docs = splitter.split_text(text)
    except Exception as e:  # 无标题/解析异常时回退为整体一节
        logger.warning("rag_md_split_failed", error=str(e))
        docs = []
    if not docs:
            # 无标题/解析异常: 按段落聚合成多个 ≤ PARENT_MAX 的节(修源头固定截断)
            return _aggregate_plain_sections(text or "", _RAG_PARENT_MAX)

    out: list[dict] = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        title = meta.get("h3") or meta.get("h2") or meta.get("h1") or ""
        body = (getattr(doc, "page_content", "") or "").strip()
        if not body:
            continue
        if len(body) <= _RAG_PARENT_MAX:
            out.append({"title": title, "content": body})
            continue
        # 超长大节: 递归切分, 不丢段义
        chunk_spl = RecursiveCharacterTextSplitter(
            chunk_size=_RAG_PARENT_MAX, chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        )
        for part in chunk_spl.split_text(body):
            if part and part.strip():
                out.append({"title": title, "content": part.strip()})
    return out


# ---------------------------------------------------------------- 2. 查询改写 + 关键词增强
async def _extract_keywords(query: str, max_keywords: int = 5) -> list[str]:
    """【作用】查询改写＋关键词增强：把用户的模糊自然语言问题，转成更适合检索的书面查询词和一个关键词列表。

    【逻辑】先让 LLM 做改写和抽词；若 LLM 不可用或返回为空，则"降级"为直接用原问题按
    常见分隔符切出词语。两者合并后再统一清洗、去重、限长，保证关键字召回稳定可用。

    【实现】用 llm.chat_agent_structured 强制输出 {query, keywords} 的 JSON schema；
    LLM 结果为空时用正则 re.split 按中文顿号/逗号、英文逗号/空格切原问句。归并关键词后
    逐词 strip 引号标点、过滤长度<2 的词、去重，最后截断到 max_keywords 个。
    """
    import re
    from pydantic import BaseModel, Field
    from app.core.llm import llm

    # 改写结果缓存：同 query 命中（Redis/内存）省一次 LLM 改写网络往返
    if getattr(settings, "rag_rewrite_cache_enabled", True):
        from app.memory.rag.cache import rewrite_cache_get
        cached = await rewrite_cache_get(query, settings.rag_rewrite_cache_ttl)
        if cached:
            cleaned = _clean_keywords(cached.get("keywords"), cached.get("query", query),
                                      max_keywords)
            return cleaned

    class _KeywordOut(BaseModel):
        query: str = Field(..., description="改写后更适合检索的查询文本")
        keywords: list[str] = Field(..., description="提取的关键词列表，3-6 个")

    prompt = (
        "你是一个检索查询改写助手。把下面这段用户问题改写成更适合向量检索的书面查询，"
        "并提取 3-6 个核心关键词（用逗号分隔）。只返回 JSON。\n\n用户问题：{query}"
    ).format(query=query or "")
    try:
        result = await llm.chat_agent_structured(
            system_prompt="只返回 JSON，不要多余内容。",
            user_message=prompt,
            output_model=_KeywordOut,
            max_tokens=150,
        )
        kw = list(result.get("keywords", [])) if isinstance(result, dict) else []
        rewritten = result.get("query", query) if isinstance(result, dict) else query
    except Exception as e:
        logger.warning("rag_keyword_llm_failed", error=str(e))
        rewritten, kw = query, []

    cleaned = _clean_keywords(kw, rewritten, max_keywords)
    # 回填缓存（内存保底 + Redis），后续同问句直接命中
    if getattr(settings, "rag_rewrite_cache_enabled", True):
        from app.memory.rag.cache import rewrite_cache_set
        await rewrite_cache_set(query, rewritten, cleaned, settings.rag_rewrite_cache_ttl)
    return cleaned


def _clean_keywords(kw: list, rewritten: str, max_keywords: int) -> list:
    """关键词兜底+清洗+去重+限长（改写与缓存命中路径共用）。"""
    import re
    if not kw:
        kw = [w for w in re.split(r"[,，、\s]+", rewritten) if w]
    kw.append(rewritten)
    cleaned: list[str] = []
    for w in kw:
        w = (w or "").strip().strip('"').strip("'").strip("，。；!?！？ ")
        if w and len(w) >= 2 and w not in cleaned:
            cleaned.append(w)
        if len(cleaned) >= max_keywords:
            break
    return cleaned


# ---------------------------------------------------------------- 3. 单路召回
async def _search_vector(query: str, top_k: int, source_filter: str | None = None,
                         user_id: str | None = None) -> list[dict]:
    """【作用】向量召回：把查询向量化后在知识库中做语义相似度检索，找语义相近的内容。

    【逻辑】对查询文本调 get_embedding 得到向量，用 pgvector 的余弦距离算子 (<=>)
    对 knowledge_chunks 全库排序，取距离最近/相似度最高的 top_k 条；相似度低于阈值的丢弃。
    仅对 conversation（对话记忆）类型做时间衰减加权（近新远旧，下界兜底防偏好被压死），
    manual（攻略）不做时间衰减。

    【实现】SQL 取每行的相似度与 created_at；对 conversation 类型的行按
    age_days 计算权重 = max(exp(-age/半衰期), 下界)，用相似度 × 权重 作为排序用分数
    （保留原始相似度到 meta）；依旧以相似度>=阈值过滤。异常降级返回空列表。
    【参数】source_filter 非空时追加 WHERE source_type = :src，只在该来源内召回；
    user_id 非空且 source_filter="conversation" 时追加 metadata_->>'user_id'=:uid，做多用户隔离
    （对话记忆只召回当前用户自己的话题，避开跨用户串记忆）。
    """
    import math
    from datetime import datetime, timezone

    embedding = await get_embedding(query)
    if not embedding:
        return []
    async with AsyncSessionLocal() as db:
        try:
            # ORM: 相似度 = 1 - 余弦距离(KnowledgeChunk.embedding.cosine_distance), 阈值下沉 WHERE。
            # ORDER BY 余弦距离升序 → 相邻即最近邻, 配合 HNSW 索引走 ANN。类型安全且可读。
            dist_expr = KnowledgeChunk.embedding.cosine_distance(embedding)
            sim_expr = 1 - dist_expr
            where_cond = [
                KnowledgeChunk.embedding.isnot(None),
                sim_expr >= _SIMILARITY_THRESHOLD,  # 阈值下沉 SQL
            ]
            if source_filter:
                where_cond.append(KnowledgeChunk.source_type == source_filter)
            if source_filter == "conversation" and user_id:
                # 多用户隔离：仅召回 metadata.user_id == 当前用户 的对话记忆（metadata_->>'user_id'）
                where_cond.append(KnowledgeChunk.metadata_.op("->>")("user_id") == user_id)
            stmt = (
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.content,
                    KnowledgeChunk.source_type,
                    KnowledgeChunk.source_id,
                    KnowledgeChunk.parent_id,
                    KnowledgeChunk.created_at,
                    sim_expr.label("similarity"),
                )
                .where(*where_cond)
                .order_by(dist_expr)
                .limit(top_k * 3)  # 多取些供衰减后重排
            )
            rows_meta = await db.execute(stmt)
            rows = []
            now = datetime.now(timezone.utc)
            for r in rows_meta.all():
                sim = float(r[6])
                # 阈值已在 SQL 过滤, 此处仅防御性保留
                if sim < _SIMILARITY_THRESHOLD:
                    continue
                item = {
                    "content": r[1],
                    "score": round(sim, 4),
                    "source_type": r[2],
                    "source_id": r[3],
                    "parent_id": r[4],
                    "created_at": r[5],
                }
                # 仅对话记忆做时间衰减（近新远旧），下界保长期偏好不被压死
                created_at = r[5]
                if r[2] == "conversation" and created_at is not None:
                    try:
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
                    except Exception:
                        age_days = 0.0
                    weight = math.exp(-age_days / _RAG_DECAY_HALFLIFE_DAYS)
                    weight = max(weight, _RAG_DECAY_MIN_WEIGHT)
                    item["score"] = round(sim * weight, 4)
                    item["recency_weight"] = round(weight, 4)
                rows.append(item)
            # 已按衰减后分数重排（仅保留候选内降序）
            rows.sort(key=lambda x: x["score"], reverse=True)
            return rows[:top_k]
        except Exception as e:
            logger.warning("rag_vector_search_failed", error=str(e))
            return []


async def _search_keyword(keywords: list[str], top_k: int, source_filter: str | None = None,
                          user_id: str | None = None) -> list[dict]:
    """【作用】关键字召回：用查询改写出的关键词做全文匹配，弥补纯向量召回的语义偏差。

    【逻辑】对每一条知识块统计其 content 命中了几个关键词，命中越多的排序越靠前，
    代表与查询词面重叠度越高。这是与向量"语义相似"互补的"词面相似"通道。

    【实现】用 SQL 把每个关键词拼成一条 content ILIKE '%关键词%' 的布尔表达式，多个
    关键词用 + 号连接成命中计数（布尔真为1，可累加）；WHERE 命中数>0，按命中数降序取
    LIMIT。由于 ILIKE 值是动态参数，关键词用命名占位符 :k0..:kn 传入防注入。score 直接
    取"命中关键词个数"(不伪装成 0~1 相似度)。
    【时间衰减修复】修复"时间衰减只作用于向量路"的盲区：关键词路对 conversation 类型的
    结果同样采集 created_at 计算 recency_weight，score = 命中数 × 权重，使"近新远旧"在
    关键词路同样生效（下界保长期偏好不被压死）。
    【参数】source_filter 非空时追加 AND source_type = :src；source_filter="conversation"
    且 user_id 非空时追加 AND metadata_->>'user_id' = :uid，做多用户隔离。异常降级返回空列表。
    """
    import math
    from datetime import datetime, timezone
    from sqlalchemy import text as _text

    if not keywords:
        return []
    async with AsyncSessionLocal() as db:
        try:
            # 对每个关键词做 ILIKE 计数, 命中数量作为关键字分数
            conditions = " + ".join(
                f"(content ILIKE '%' || :k{i} || '%')" for i, _ in enumerate(keywords)
            )
            params = {f"k{i}": k for i, k in enumerate(keywords)}
            parts = [f"{conditions} > 0"]
            if source_filter:
                parts.append("source_type = :src")
                params["src"] = source_filter
            if source_filter == "conversation" and user_id:
                parts.append("metadata_->>'user_id' = :uid")
                params["uid"] = user_id
            sql = _text(
                f"""
                SELECT id, content, source_type, source_id,
                       ({conditions}) AS hit_cnt, created_at
                FROM knowledge_chunks
                WHERE {' AND '.join(parts)}
                ORDER BY hit_cnt DESC
                LIMIT :limit
                """
            )
            rows_meta = await db.execute(sql, {**params, "limit": top_k})
            now = datetime.now(timezone.utc)
            rows = []
            for r in rows_meta.fetchall():
                item = {
                    "content": r[1],
                    "score": float(r[4]),  # 命中词个数(≤关键词数), 不伪装成0~1相似度
                    "source_type": r[2],
                    "source_id": r[3],
                    "created_at": r[5],
                }
                # 时间衰减修复：对话记忆按新旧加权，避免"衰减只压向量路"
                created_at = r[5]
                if r[2] == "conversation" and created_at is not None:
                    try:
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
                    except Exception:
                        age_days = 0.0
                    weight = math.exp(-age_days / _RAG_DECAY_HALFLIFE_DAYS)
                    weight = max(weight, _RAG_DECAY_MIN_WEIGHT)
                    item["score"] = round(float(r[4]) * weight, 4)
                    item["recency_weight"] = round(weight, 4)
                rows.append(item)
            return rows
        except Exception as e:
            logger.warning("rag_keyword_search_failed", error=str(e))
            return []


# ---------------------------------------------------------------- 4. 多路召回融合 + Rerank
def _rrf_fuse(irs: list[list[dict]], top_k: int) -> list[dict]:
    """【作用】多路召回融合（Rerank）：把向量召回 + 关键字召回两组结果合并成一列, 并重排。

    【逻辑】采用 Reciprocal Rank Fusion(RRF)：不看绝对相似度，而看"在某一路里排第几"。
    一条内容在多路中出现且排名越靠前，融合分越高。这样两路量纲不同的分数无需对齐就能
    公平比较, 同时靠"多路都命中"天然去噪。最终只保留 top_k 条、按内容去重。

    【实现】遍历每一路的每条结果，用内容 content 作唯一键合并；每命中一路就把该条贡献
    的 1/(_RRF_K + rank + 1) 累加到融合分，并记录它来自哪几路(sources)。全部累加完
    按融合分降序取前 top_k，删掉临时字段 _rrf 并把 _src 改名 sources 输出。
    """
    merged: dict[str, dict] = {}
    for i, results in enumerate(irs):
        for rank, item in enumerate(results):
            key = item["content"]
            if key not in merged:
                merged[key] = {**item, "_src": [], "_rrf": 0.0}
            merged[key]["_rrf"] += 1.0 / (_RRF_K + rank + 1)
            merged[key]["_src"].append(i)
    ranked = sorted(merged.values(), key=lambda x: x["_rrf"], reverse=True)[:top_k]
    for item in ranked:
        item.pop("_rrf", None)
        item["sources"] = item.pop("_src")
    return ranked


async def retrieve(query: str, top_k: int = None, source_filter: str | None = None,
                   user_id: str | None = None) -> list[dict]:
    """【作用】RAG 加工流水线的主入口：对用户问题执行"切换写→双路召回→融合重排"的检索。

    【逻辑】一条查询最终要输出一组与它最相关、混排了语义与词面信号的记忆。典型链路：
    先从问题抽出关键词 → 同时做向量召回与关键字召回 → 把两路结果 RRF 融合成一份唯一
    结果按相关度降序返回。
    【实现】1) 调 _extract_keywords(query) 得到关键词；2) 调 _search_vector(query, top_k)
    得语义命中；3) 调 _search_keyword(关键词, top_k) 得词面命中；4) 打印 rag_hybrid 日志
    （记录关键词数/两路命中数）；5) 用 _rrf_fuse([vector_hits, keyword_hits], top_k) 融合
    输出。路径中的任一步失败都由内部函数降级为空列表，不抛出、不阻塞。

    【参数】source_filter 非空时只在指定来源内召回（如 "conversation"=历史记忆，
    "manual"=攻略知识），实现 RAG 两层拆分：外层只查对话记忆、子图只查攻略。
    user_id 在 source_filter="conversation" 时下发到两路做多用户隔离（只召回当前用户记忆）。
    """
    if top_k is None:
        top_k = settings.rag_top_k

    keywords = await _extract_keywords(query)
    vector_hits = await _search_vector(query, top_k, source_filter, user_id)
    keyword_hits = await _search_keyword(keywords, top_k, source_filter, user_id)
    logger.info(
        "rag_hybrid",
        query=query[:60], kw=len(keywords), src=source_filter,
        vec=len(vector_hits), kw_hits=len(keyword_hits),
    )
    fused = _rrf_fuse([vector_hits, keyword_hits], top_k)
    # 击败父块: 命中小块时回溯其父块内容(若无父块则用本块)
    fused = await _resolve_parents(fused)
    return fused


async def _resolve_parents(results: list[dict]) -> list[dict]:
    """【作用】命中子块时，把小块内容替换成其所属父块(大块)完整内容。

    【逻辑】父子块设计下，小块负责精准召回，但喂给 LLM 时应是大块上下文。对每条命中，
    若 parent_id 非空，则查询父块并以其 content 覆盖本块 content（保留原 score 标注来源）。

    【实现】收集有 parent_id 的 id；一次 SQL 查出这些父块 {id: content}；遍历替换。
    最后按替换后的内容再按 content 去重：因为多 child 会命中同一 parent，若不二次去重，
    rag_context 会把同一个 parent 重复注入 Prompt(撑爆上下文)，修正为只留首个、保留相关序。
    """
    parent_ids = {item.get("parent_id") for item in results if item.get("parent_id")}
    if not parent_ids:
        return results
    async with AsyncSessionLocal() as db:
        try:
            rows_meta = await db.execute(
                text("""
                    SELECT id, content FROM knowledge_chunks
                    WHERE id = ANY(:ids)
                """),
                {"ids": list(parent_ids)},
            )
            parent_map = {r[0]: r[1] for r in rows_meta.fetchall()}
        except Exception as e:
            logger.warning("rag_parent_resolve_failed", error=str(e))
            return results
    for item in results:
        pid = item.get("parent_id")
        if pid and pid in parent_map:
            item["content"] = parent_map[pid]
            item["from_child"] = True
    # 二次去重: 相同 parent 的多 child 替换后只保留首个, 避免重复注入 Prompt
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in results:
        key = item.get("content", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


# ---------------------------------------------------------------- 入库（含切片）
async def insert_chunk(content: str, source_type: str = "manual", source_id: str = None, metadata: dict = None) -> bool:
    """【作用】把一段内容向量化后写入知识库（knowledge_chunks 表）的原子操作。

    【逻辑】先给这一整段内容生成 embedding；向量生成成功才落库，失败则跳过并返回 False，
    保证"库里只存能检索的内容"。入库时附带来源信息与自定义元数据。

    【实现】调 get_embedding(content)；拿到向量后新开一个数据库会话，构造 KnowledgeChunk
    对象（含 content/embedding/source_type/source_id/metadata_）add 后 commit。整段写入
    （不做切片），异常捕获打 rag_insert_failed 日志并返回 False。
    """
    embedding = await get_embedding(content)
    if not embedding:
        return False
    async with AsyncSessionLocal() as db:
        try:
            chunk = KnowledgeChunk(
                source_type=source_type,
                source_id=source_id,
                content=content,
                embedding=embedding,
                metadata_=metadata or {},
            )
            db.add(chunk)
            await db.commit()
            return True
        except Exception as e:
            logger.error("rag_insert_failed", error=str(e))
            return False


async def insert_text(content: str, source_type: str = "manual", source_id: str = None, metadata: dict = None) -> int:
    """【作用】含切片入库：把一段可能很长的文本切成若干块后批量向量化写入知识库。

    【逻辑】长文本直接整段向量化会丢失检索粒度，先按 chunk_size/overlap 切块，每一块
    才能被更精地召回。返回成功写入的块数，供调用方度量沉淀是否落库。

    【实现】调 chunk_text(content) 得到切片列表；把全部块一次 _embed_many 批量向量化，
    再单会话批量 add + commit（避免逐块 embed 的网络往返）。返回成功写入的块数。
    """
    chunks = chunk_text(_preclean(content))
    if not chunks:
        return 0
    ok = 0
    async with AsyncSessionLocal() as db:
        try:
            items = [{
                "content": c, "parent_id": None,
                "source_type": source_type, "source_id": source_id, "metadata": metadata,
            } for c in chunks]
            ok = await _insert_chunks_without_session(items, db)
            await db.commit()
        except Exception as e:
            logger.error("rag_insert_text_failed", error=str(e))
            await db.rollback()
    return ok


async def insert_markdown_knowledge(content: str, source_type: str = "manual", source_id: str = None, metadata: dict = None) -> int:
    """【作用】攻略类父块入库：按标题切大块(parent)，大块内切小块(child)以父块 id 关联。

    【逻辑】实现【父母块】策略：小块进向量库负责精准召回；命中小块后拿到 parent_id
    → 回溯切片的大块作完整上下文喂给 LLM。入库顺序：先切标题得到大块，每个大块：
    ① 收集该大块为 parent；② 大块内容再按小块尺寸切片，小块以父块 id 关联。

    【实现】对 source_type=manual 且内容带标题时，用 split_markdown_by_headings 切大块。
    大块若较短(< _RAG_CHILD_SIZE)则作为单块 parent；否则建 parent 并把小块收集为 child。
    全部块一次 _embed_many 批量向量化、单会话批量 add + commit（P1：N 次网络往返 → N/25）。
    """
    sections = split_markdown_by_headings(_preclean(content))
    if not sections:
        return await insert_text(content, source_type, source_id, metadata)

    items: list[dict] = []  # 统一收集待批量入库的块
    async with AsyncSessionLocal() as db:
        try:
            for sec in sections:
                title = sec.get("title", "")
                body = sec["content"]
                meta = dict(metadata or {})
                meta["title"] = title
                if len(body) <= _RAG_CHILD_SIZE:
                    # 单块较短: 直接作为 parent 落库
                    items.append({
                        "content": body, "parent_id": None,
                        "source_type": source_type, "source_id": source_id, "metadata": meta,
                    })
                    continue
                # 父块先记录(固定 id 供子块回填 parent_id), 子块以该 id 关联
                parent_id = _new_uuid()
                parent_meta = dict(meta)
                parent_meta["is_parent"] = True
                items.append({
                    "content": body[:_RAG_PARENT_MAX], "parent_id": None, "fixed_id": parent_id,
                    "source_type": source_type, "source_id": source_id, "metadata": parent_meta,
                })
                child_meta = dict(meta)
                child_meta["parent_id"] = parent_id
                for child in chunk_text(body, _RAG_CHILD_SIZE, _RAG_CHILD_OVERLAP):
                    items.append({
                        "content": child, "parent_id": parent_id,
                        "source_type": source_type, "source_id": source_id, "metadata": child_meta,
                    })
            # 批量 embedding + 批量 add, 一次 commit
            ok = await _insert_chunks_without_session(items, db)
            await db.commit()
        except Exception as e:
            logger.error("rag_markdown_commit_failed", error=str(e))
            await db.rollback()
            ok = 0
    return ok


from uuid import uuid4 as _uuid4

def _new_uuid() -> str:
    """生成字符串 UUID（供父块 id）。"""
    return str(_uuid4())


async def _insert_chunks_without_session(items: list[dict], db) -> int:
    """【作用】在已开会话内一次性批量写入多块（P1 批量核心）。

    【逻辑】把「已收集的待入库块」全部集中，先一次批量 embedding（_embed_many, N 次
    网络往返降到 N/25），再把"向量非空"的块批量 add 进会话。内部不 commit，由外层统一。

    【实现】items 每项形如 {content, parent_id, source_type, source_id, metadata, fixed_id?}；
    先 _embed_many([i["content"] for i in items]) 得到对齐向量，zip 逐项构造 KnowledgeChunk、
    跳过向量为空的块，db.add 累积。返回成功 add 的块数。
    """
    if not items:
        return 0
    embeddings = await _embed_many([it["content"] for it in items])
    added = 0
    for it, emb in zip(items, embeddings):
        if not emb:
            continue
        chunk = KnowledgeChunk(
            id=it.get("fixed_id"),
            source_type=it["source_type"], source_id=it.get("source_id"),
            content=it["content"], embedding=emb,
            parent_id=it.get("parent_id"), metadata_=it.get("metadata") or {},
        )
        db.add(chunk)
        added += 1
    return added


# ---------------------------------------------------------------- 兼容旧接口
async def search_similar(query: str, top_k: int = None) -> list[dict]:
    """【作用】兼容旧接口：对外暴露单路向量检索，等价于调用 _search_vector。

    【逻辑】保留给历史上按语义检索的调用方直接使用；内部不涉及关键字改写，也不做融合。

    【实现】top_k 缺省取 settings.rag_top_k，然后透传给 _search_vector(query, top_k)
    返回其结果。
    """
    if top_k is None:
        top_k = settings.rag_top_k
    return await _search_vector(query, top_k)


def _within_budget(results: list[dict], max_chars: int = None) -> list[dict]:
    """【作用】按字符预算累加截断命中结果，避免大块回溯后撑爆 LLM Prompt。

    【逻辑】上下文注入是"逐条累加"的：每条回溯成大块(parent)后可能上千字，top_k 条全拼
    会挤占模型上下文。这里按累计字符上限截取前若干条（至少保留第 1 条），超预算即停止。

    【实现】遍历结果，content 长度累加，超过 max_chars 就 break；返回截断后的子集。
    """
    budget = max_chars or _RAG_CONTEXT_BUDGET_CHARS
    total = 0
    kept: list[dict] = []
    for it in (results or []):
        if kept and total + len(it.get("content", "")) > budget:
            break
        kept.append(it)
        total += len(it.get("content", ""))
    return kept


async def rag_context(query: str, top_k: int = None, max_chars: int = None) -> str:
    """【作用】把检索结果拼成一段可直接注入 LLM Prompt 的文本；无相关结果时返回空串。

    【逻辑】作为"检索 → 生成(RAG)"的桥接。先走完整加工流水线 retrieve(query)，若一条
    记忆都没有返回空字符串（让 Prompt 里该占位符为空，不影响生成）；否则把每条记忆连同
    相似度与来源格式化成一串参考段落。

    【实现】调 retrieve(query, top_k)；为空直接返回 ""；非空时用 _within_budget 按
    max_chars(默认 _RAG_CONTEXT_BUDGET_CHARS)累加截断，避免大块回溯后把 Prompt 撑爆；
    对命中项拼成 "[参考 N] (相似度:x, 来源:y)\n内容"，空行连接返回。
    """
    results = await retrieve(query, top_k)
    if not results:
        return ""
    results = _within_budget(results, max_chars or _RAG_CONTEXT_BUDGET_CHARS)
    parts = [
        f"[参考 {i+1}] (相似度: {r['score']:.2f}, 来源:{r.get('source_type','')})\n{r['content']}"
        for i, r in enumerate(results)
    ]
    return "\n\n".join(parts)