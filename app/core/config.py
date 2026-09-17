from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import model_validator


def _pg_url(host, port, user, pwd, db) -> str:
    cred = f"{user}:{pwd}" if pwd else user
    return f"postgresql+asyncpg://{cred}@{host}:{port}/{db}"


def _redis_url(host, port, pwd, db) -> str:
    if pwd:
        return f"redis://:{pwd}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 应用
    app_name: str = "travel_plan_agent"
    app_env: str = "development"
    debug: bool = False

    # Agent checkpointer：memory | redis（redis 需安装 langgraph-checkpoint-redis，持久化可跨 worker/重启恢复）
    checkpointer: str = "memory"

    # LLM
    llm_provider: str = "ollama"
    llm_temperature: float = 0.7

    # Ollama（本地模型）
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen2.5vl:7b"

    # DashScope（阿里云百炼）
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-turbo"
    # RAG embedding（DashScope 文本向量模型, text-embedding-v3 默认输出 1024 维, 不传 dimensions 参数）
    dashscope_embedding_model: str = "text-embedding-v3"
    dashscope_embedding_dim: int = 1024
    # RAG rerank（真模型重排: 对 RRF 融合后的候选按与问题相关性再排序, gte-rerank-v2）
    rag_rerank_enabled: bool = False
    dashscope_rerank_model: str = "gte-rerank-v2"

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = ""
    pg_database: str = "travel_agent"
    database_url: str = ""  # 空则自动用组件字段拼装（默认口令为空，安全）
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 5
    redis_password: str = ""
    redis_url: str = ""  # 空则自动用组件字段拼装（默认口令为空，安全）
    redis_use_llm_cache: bool = False

    # RAG（向量维度与 DashScope embedding 模型保持一致, text-embedding-v3 默认 1024 维）
    rag_enabled: bool = True
    rag_dim: int = 1024
    rag_top_k: int = 4
    rag_chunk_size: int = 480
    rag_chunk_overlap: int = 48

    # RAG 检索缓存（同问句命中缓存省 embedding 网络往返）
    rag_cache_enabled: bool = True
    rag_cache_ttl: int = 300            # 缓存秒数

    # RAG 过程缓存（改写/embedding，Redis 优先 + 内存兜底）与子图会话级注入
    rag_rewrite_cache_enabled: bool = True
    rag_rewrite_cache_ttl: int = 600     # LLM 改写结果缓存秒数
    rag_embed_cache_enabled: bool = True
    rag_embed_cache_ttl: int = 86400     # embedding 向量缓存秒数（向量复用度高）
    rag_session_inject: bool = True      # 子图内首检后会话级复用，避免每步重复检索

    # OpenTelemetry 链路追踪（官方 SDK + OTLP；SDK 未安装时自动降级为轻量日志）
    otel_enabled: bool = False
    otel_service_name: str = "travel_plan_agent"
    otel_otlp_endpoint: str = "http://192.168.100.128:4318/v1/traces"

    # token 统计（落 Redis HINCRBY，跨 worker/重启持久化；Redis 不可用降级进程内存）
    token_stats_redis_enabled: bool = True
    token_stats_ttl: int = 604800        # 统计 key TTL（7 天）

    # ERP
    erp_base_url: str = "http://localhost:8080"
    erp_use_mock: bool = True

    # MCP 外部服务
    mcp_weather_base_url: str = "http://localhost:8100"

    # --- F5 上下文外部化 ---
    artifact_threshold: int = 1200          # 大结果阈值（序列化字符数），超过则外部化
    artifact_llm_summary: bool = False      # text 类是否 LLM 摘要（默认启发式截断，省 token）

    # --- F4 审计采样（权限/审批类动作必审计，不参与采样）---
    audit_sampling: float = 1.0

    # --- F6 多子 Agent 委派 ---
    multi_agent_enabled: bool = False       # 默认关闭，退化为单子图

    # --- 安全：API 管理令牌（最小防线，方案 A） ---
    # /audits 与 /monitor 等敏感端点需 Bearer token 或 X-API-Key；
    # 空串 = 关闭鉴权（本地演示零障碍），生产用 .env 注入非空值。
    api_admin_token: str = ""

    # CORS 白名单（逗号分隔）。空 = 不做来源限制（本地演示），但会关掉 allow_credentials 以避免"通配+凭据"危险组合
    cors_origins: str = ""

    @model_validator(mode="after")
    def _fill_urls(self):
        if not self.database_url:
            self.database_url = _pg_url(self.pg_host, self.pg_port, self.pg_user,
                                        self.pg_password, self.pg_database)
        if not self.redis_url:
            self.redis_url = _redis_url(self.redis_host, self.redis_port,
                                        self.redis_password, self.redis_db)
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()
