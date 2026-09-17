FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# 复制依赖文件
COPY pyproject.toml ./

# 安装 Python 依赖（不使用虚拟环境）
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# 复制应用代码
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
