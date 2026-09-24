# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# 前端构建
# ---------------------------------------------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /build

# 先只拷依赖清单，让这层能被缓存 —— 改前端代码不会重装 node_modules
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# 运行时
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/backend

# 依赖单独一层（只依赖 pyproject），改业务代码不会触发重装
COPY backend/pyproject.toml backend/README.md ./
RUN pip install .

COPY backend/app ./app

# ffmpeg 用**项目自带**的那份（6.1.1），不 apt 装 ——
# 版本可控，而且和本地开发、非 Docker 部署的行为完全一致。
#
# ⚠️ chmod 是必需的：Windows 工作区里这个文件的可执行位会丢（644），
# Docker 从工作区构建时按实际权限拷进去，不加这一步 ffmpeg 调不起来。
COPY backend/vendor/ffmpeg/ffmpeg ./vendor/ffmpeg/ffmpeg
RUN chmod 0755 ./vendor/ffmpeg/ffmpeg

# 前端产物。后端启动时会挂载 frontend/dist，所以不用单独跑前端服务。
COPY --from=frontend /build/dist /app/frontend/dist

# 数据目录：上传、抽帧、SQLite 都写这里，用卷挂出去
RUN mkdir -p /app/data

# 非 root 运行。挂载宿主目录时要保证 uid 1000 可写（见 README）。
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 用 /api/health 做健康检查 —— 它会验证 ffmpeg 是否就位
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# ⚠️ workers 必须是 1：任务状态存在进程内存里，多 worker 会让「提交任务的进程」
# 和「查询进度的进程」不是同一个，进度永远查不到。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
