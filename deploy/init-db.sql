-- 初始化数据库和用户（仅限本地开发默认值）
-- 警告：以下 agent123 / erp123 为本地开发样例口令，生产环境务必替换。
-- 本脚本由 postgres 容器 entrypoint 执行，无法引用 compose 的 ${VAR}；
-- 如需接入用需修改此处口令并同步 docker-compose.yml 中的 -override 变量。
CREATE USER agent WITH PASSWORD 'agent123';
CREATE DATABASE agent_db OWNER agent;
GRANT ALL PRIVILEGES ON DATABASE agent_db TO agent;

CREATE USER erp WITH PASSWORD 'erp123';
CREATE DATABASE erp_db OWNER erp;
GRANT ALL PRIVILEGES ON DATABASE erp_db TO erp;

-- agent_db 启用 pgvector
\c agent_db
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- erp_db 建表
\c erp_db
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_no VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    department VARCHAR(200) NOT NULL,
    position VARCHAR(100),
    annual_budget DECIMAL(12,2) DEFAULT 0,
    remaining_budget DECIMAL(12,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_no VARCHAR(50) UNIQUE NOT NULL,
    plan_id VARCHAR(100) NOT NULL,
    applicant_id VARCHAR(50) NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    approver VARCHAR(100),
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP
);

INSERT INTO employees (employee_no, name, email, department, position, annual_budget, remaining_budget)
VALUES ('EMP001', '张三', 'zhangsan@example.com', '技术部', '高级工程师', 50000, 32000)
ON CONFLICT (employee_no) DO NOTHING;
