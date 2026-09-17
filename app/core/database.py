from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


async def get_db() -> AsyncSession:
    """FastAPI 依赖: 每个请求一个独立数据库会话, 结束/异常时自动关闭归还连接池。"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """创建所有表（开发环境用，生产用 Alembic）"""
    async with engine.begin() as conn:
        # pgvector 扩展: 建 vector 类型相关表(knowledge_chunks)前必须先启用
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass  # 数据库已装扩展或缺此能力时忽略, 不影响其余建表
        await conn.run_sync(Base.metadata.create_all)
        # P0: HNSW 索引 —— embedding 一旦有量级, 没有它全表扫描会非常慢, 必须尽早建好
        #     (幂等, 余弦距离算子 vector_cosine_ops)
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
                "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            ))
        except Exception:
            pass  # 表/扩展未就绪或库版本不支持 hnsw 时忽略, 建表/降级不阻断启动
