# 数据库迁移

当前项目**未使用 alembic 迁移**，采用轻量建表策略：

- 应用启动时 `app/core/database.py::init_db()` 自动执行：
  1. `CREATE EXTENSION IF NOT EXISTS vector`（启用 pgvector，供 RAG `knowledge_chunks` 使用）；
  2. SQLAlchemy `metadata.create_all()` 依据 `app/models/models.py` 的 ORM 模型建表。

适用场景：开发/演示环境，表结构变更直接改模型并重启即可生效。

> 备注：本目录为未来平滑引入 Alembic 迁移（`alembic revision --autogenerate`）预留的位置。若需切换为版本化迁移，建议：
> 1. 安装 `alembic`；
> 2. `alembic init migrations`；
> 3. 将 `init_db()` 改为仅应用迁移、不再 `create_all`。