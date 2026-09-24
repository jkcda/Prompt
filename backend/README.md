# 后端（FastAPI）

「视频提示词反推」的后端服务。完整说明见仓库根目录的 `README.md`。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env      # 至少填 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ **`--workers` 必须是 1。** 任务状态存在进程内存里，多 worker 会让
> 「提交任务的进程」和「查询进度的进程」不是同一个，进度永远查不到。
>
> 不用 `fastapi dev` —— 那需要额外装 `fastapi[standard]`，而 uvicorn 本来就是依赖。

## 分层

```
app/
  core/       基础设施（config / logging / db / safe_delete），无业务逻辑
  routers/    APIRouter，只做校验与返回
  schemas/    Pydantic 模型，与 routers 一一对应
  services/   业务实现；routers 只调这里
  models/     SQLModel 表定义
```

新增功能按这个分层放，**不要把业务写进 router**。

## 目录里还有什么

| 路径 | 说明 |
|---|---|
| `vendor/ffmpeg/` | 打包进仓库的 ffmpeg（Windows + Linux 各一份），部署不用另装 |
| `vendor/whisper/` | 本地语音转写模型（**可选**，不进 git，见根 README） |
| `tests/` | pytest；模型调用一律走 `tests/mock_vlm.py`，不调真实模型 |
| `app/selfcheck.py` | 不调模型的核心链路自检：`python -m app.selfcheck 视频.mp4` |
| `app/dump_prompts.py` | 导出提示词模板，检查实际发给模型的内容 |

## 测试

```bash
pytest -q
ruff check app tests
```

跑真实 ffmpeg 的测试用 `sample_video` fixture（现场生成 10 秒测试视频）。
