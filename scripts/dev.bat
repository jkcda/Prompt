@echo off
REM 一键启动开发环境：后端 FastAPI（热重载）+ 前端 Vite
REM 用法：scripts\dev.bat
REM
REM 会打开两个独立窗口，方便分别看日志。关闭窗口即停止对应服务。

setlocal
cd /d "%~dp0.."

if not exist "backend\.env" (
  echo [!] 未找到 backend\.env，正在从 .env.example 复制...
  copy /Y "backend\.env.example" "backend\.env" >nul
  echo [!] 请编辑 backend\.env 填入 VLM_API_KEY 后重新运行本脚本。
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [*] 首次运行，安装前端依赖...
  pushd frontend
  call npm install
  popd
)

echo [*] 启动后端 http://127.0.0.1:8000
start "prompt-reverse-backend" /D "%CD%\backend" fastapi dev

echo [*] 启动前端 http://127.0.0.1:5173
start "prompt-reverse-frontend" /D "%CD%\frontend" npm run dev

echo.
echo   后端接口文档  http://127.0.0.1:8000/docs
echo   前端页面      http://127.0.0.1:5173
echo.
echo 关闭弹出的两个窗口即可停止服务。
endlocal
