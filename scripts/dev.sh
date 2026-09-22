#!/usr/bin/env bash
# 一键启动开发环境：后端 FastAPI（热重载）+ 前端 Vite
# 用法：./scripts/dev.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f backend/.env ]; then
  echo "[!] 未找到 backend/.env，正在从 .env.example 复制..."
  cp backend/.env.example backend/.env
  echo "[!] 请编辑 backend/.env 填入 VLM_API_KEY 后重新运行。"
  exit 1
fi

if [ ! -d frontend/node_modules ]; then
  echo "[*] 首次运行，安装前端依赖..."
  (cd frontend && npm install)
fi

cleanup() {
  echo
  echo "[*] 停止服务..."
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[*] 启动后端 http://127.0.0.1:8000"
(cd backend && fastapi dev) &

echo "[*] 启动前端 http://127.0.0.1:5173"
(cd frontend && npm run dev) &

echo
echo "  后端接口文档  http://127.0.0.1:8000/docs"
echo "  前端页面      http://127.0.0.1:5173"
echo "  按 Ctrl+C 停止"
wait
