# 视频提示词反推

上传视频，或用 B站 / 抖音 链接，自动反推出**可直接使用的视频生成提示词**。

后端 FastAPI + Python，前端 Vue 3 + Vite，抽帧与镜头分割全部由 ffmpeg 完成，
视觉理解走任意 OpenAI 兼容的多模态模型。

支持四种输出格式：MiniMax H3（T2VA 三字段 / Ref2VA 六段式）、Seedance 2.0、通用分镜表。

---

## 目录

- [它是怎么工作的](#它是怎么工作的)
- [为什么不是「1 秒抽 10 帧」](#为什么不是1-秒抽-10-帧)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [接口一览](#接口一览)
- [B站 / 抖音抓取](#b站--抖音抓取)
- [测试](#测试)
- [已知限制](#已知限制)

---

## 它是怎么工作的

反推分两阶段。这不是为了好看，是因为一次性让模型「看图直接写提示词」会大面积丢细节——
模型的注意力会花在措辞上，次要镜头、背景细节、运镜方向基本丢失。

```
上传 / 抓取
     │
     ├─ ffprobe（缺失时用 ffmpeg -i 解析）──→ 时长 / 分辨率 / 帧率 / 有无音轨
     │
     ├─ ffmpeg select='gt(scene,T)' ──────→ 镜头切换点 → 镜头列表
     │
     ├─ ffmpeg volumedetect + silencedetect → 音量曲线 / 静音段 / 疑似卡点
     │
     ├─ ASR（可选）─────────────────────→ 带时间戳的台词 / 歌词
     │
     ▼
  帧预算分配（按镜头时长加权，每镜取 首/中/尾）
     │
     ▼
  长视频在**镜头边界**分块（不物理切割）
     │
     ├─→ Pass 1「只看不写」：每块并行送模型
     │        └→ 结构化 JSON：景别 / 运镜 / 主体 / 动作 / 光线 / 色调 /
     │           台词 / 音效 / 转场 / 置信度
     │
     ▼
  Pass 2「只写不看」：全部观察 JSON + 音频报告 + 全局统计
     │        └→ 目标格式提示词（H3 / H3-Ref / Seedance / 通用）
     ▼
  落库（SQLite）→ 前端左视频右提示词对照展示
```

**同一份观察结果可以出多种格式**，这是两阶段的额外收益：Pass 1 花钱花时间，
Pass 2 很便宜，想换格式不用重新看一遍视频。

---

## 为什么不是「1 秒抽 10 帧」

常见的做法是 `ffmpeg -vf fps=N` 均匀抽帧。这个方案在反推场景下有两个硬伤。

### 一、固定 fps 会直接爆上下文

| 方案 | 60 秒视频的帧数 | 估算 tokens |
|---|---|---|
| `fps=10`（1 秒 10 帧） | 600 帧 | ≈ **66 万** |
| `fps=2` | 120 帧 | ≈ 13 万 |
| **本项目：按镜头分配预算** | **48 帧** | ≈ **5.3 万** |

按 1024×1024 一帧约 1100 tokens 估算。主流多模态模型的上下文在 12.8 万～20 万之间，
`fps=10` 对任何超过 15 秒的视频都会失败。所以固定 fps 只适合 ≤3 秒的片段。

### 二、均匀抽帧会浪费掉 90% 的额度，还漏掉切镜

`-vf fps=N` 是**等间隔采样**，既不是关键帧（I 帧）也不是镜头切换帧。
一个 30 秒、6 个镜头的视频：

- 均匀抽帧：300 帧全铺在镜头内部，每个镜头 50 帧高度冗余，**却可能刚好跨过切点**；
- 镜头感知：每镜头取首/中/尾共 15～18 帧，**每个镜头都被看到，切点一个不漏**。

本项目用 ffmpeg 原生的场景检测拿切点，零额外依赖：

```bash
ffmpeg -i in.mp4 -vf "select='gt(scene,0.30)',showinfo" -an -f null -
# 解析 stderr 里的 pts_time 就是切点时间
```

然后按**镜头时长加权**分配帧预算，长镜头多给、短镜头少给，单镜头内部取首/中/尾
（带 8% 内缩，避开转场混合帧）。实测 10 秒 3 镜头视频：2 个切点全部命中，
9 帧覆盖 3 个镜头，约 9.9k tokens。

### 三、长视频不按固定秒数硬切

按固定秒数切会把一个镜头劈成两半，两边都推不准。本项目**只在镜头边界分块**，
而且不做物理切割——直接按绝对时间从源文件抽帧，省掉一次重编码。
块之间通过上一块的 `global_notes` 传递上下文。

> `segment_video()` 仍然保留，用于「给用户分片预览」或「喂给原生视频模型」的场景。

### 四、音频必须进管线

纯画面帧**永远推不出** `overall_soundscape`、`non_diegetic_music`、台词、口型同步——
而这些是 H3 / Seedance 格式里的硬字段。所以：

- ASR 带时间戳转写 → 台词/歌词，并按时间戳对齐到镜头；
- `volumedetect` + `silencedetect` → 判断有没有 BGM、哪里是卡点。

没配 ASR 也能跑，但输出会缺一整个维度，字段只能写 `N/A`。

---

## 项目结构

```
提示词反推/
├── backend/                        FastAPI 后端
│   ├── app/
│   │   ├── main.py                 应用装配、lifespan、前端托管、SPA 兜底
│   │   ├── dependencies.py         共享依赖注入
│   │   ├── core/                   基础设施（无业务逻辑）
│   │   │   ├── config.py           pydantic-settings + ffmpeg 路径发现
│   │   │   ├── logging.py          统一日志格式
│   │   │   └── db.py               SQLite 引擎与会话
│   │   ├── routers/                按域拆分的 APIRouter
│   │   │   ├── system.py           /health /settings /formats /stats
│   │   │   ├── upload.py           /upload
│   │   │   ├── analyze.py          /analyze /fetch /fetch/probe
│   │   │   ├── jobs.py             /jobs + SSE + /prompts
│   │   │   └── media.py            /media（含 HTTP Range）
│   │   ├── schemas/                Pydantic 模型
│   │   │   ├── media.py            媒体 / 镜头 / 帧 / 音频报告
│   │   │   ├── observation.py      Pass1 结构化观察
│   │   │   ├── job.py              任务 / 进度 / 结果 / 选项
│   │   │   └── api.py              HTTP 请求响应体
│   │   ├── models/job.py           SQLModel 表定义
│   │   └── services/               业务实现
│   │       ├── ffmpeg.py           探测 / 切块 / 抽帧 / 镜头检测
│   │       ├── selection.py        自适应选帧（token 预算 + 镜头加权）
│   │       ├── asr.py              语音转写 + 音量分析
│   │       ├── vlm.py              多模态客户端（并发 / 重试 / 减帧降级）
│   │       ├── templates.py        Pass1 观察 + Pass2 成文模板
│   │       ├── pipeline.py         两阶段编排
│   │       ├── downloader.py       B站 / 抖音抓取
│   │       ├── storage.py          SQLite 读写
│   │       ├── jobs.py             任务状态 + 事件总线
│   │       └── runner.py           任务启动器
│   ├── tests/                      80 个测试
│   │   ├── mock_vlm.py             OpenAI 兼容 mock 模型（E2E 用）
│   │   ├── test_selection.py       选帧策略（13）
│   │   ├── test_ffmpeg.py          ffmpeg 层（22，跑真实视频）
│   │   ├── test_templates.py       模板与 JSON 解析（14）
│   │   ├── test_api.py             接口集成（23）
│   │   └── test_pipeline_e2e.py    端到端管线（8）
│   ├── pyproject.toml              依赖 + [tool.fastapi] + ruff + pytest
│   └── .env.example
├── frontend/                       Vue 3 + Vite + TS + Pinia
│   └── src/
│       ├── api/index.ts            axios 封装 + SSE 订阅
│       ├── stores/analyze.ts       反推流程状态机
│       ├── components/             选源 / 播放器+帧带 / 提示词 / 进度 / 历史
│       └── views/                  HomeView / ResultView
├── data/                           运行时数据（git 忽略）
└── .gitignore
```

---

## 快速开始

### 环境要求

| 依赖 | 说明 |
|---|---|
| Python | ≥ 3.10 |
| Node.js | ≥ 18（只在构建前端时需要） |
| ffmpeg | **必需**。见下方安装说明 |
| 多模态模型 | 任意 OpenAI 兼容服务（ModelScope / 百炼 / 方舟 / OpenRouter / 本地 vLLM） |

### ffmpeg 安装

代码会按顺序自动查找，命中任一个即可：

1. `backend/.env` 里的 `FFMPEG_PATH`
2. `backend/runtime/ffmpeg.exe`
3. 系统 `PATH`
4. 常见安装目录

**不需要 ffprobe**——缺失时会自动改用 `ffmpeg -i` 解析媒体信息。

### 后端

```bash
cd backend

# 1. 装依赖（推荐用 venv）
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e ".[dev]"

# 2. 配置
cp .env.example .env
# 编辑 .env，至少填 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL

# 3. 启动
fastapi dev                     # 开发（热重载）
# 或
fastapi run                     # 生产
```

启动后：

- 接口文档 <http://127.0.0.1:8000/docs>
- 健康检查 <http://127.0.0.1:8000/api/health>
- 模型连通性自检 <http://127.0.0.1:8000/api/health/vlm>

不装 ffmpeg 也能启动，但抽帧不可用，前端会给出提示。

### 前端

**开发模式**（前后端分离，Vite 代理 `/api`）：

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

**生产模式**（后端直接托管构建产物）：

```bash
cd frontend
npm run build        # 产出 frontend/dist
```

之后访问 <http://127.0.0.1:8000/> 就是完整页面，不需要单独跑 Node。

### 自检

不调用模型，只验证 ffmpeg / 镜头检测 / 选帧 / 音频分析：

```bash
cd backend
python -m app.selfcheck path/to/video.mp4
```

输出示例：

```
[1] 媒体探测  (0.46s)
    时长   : 10.00s
    分辨率 : 640x360 @ 24.00fps
    音轨   : 有

[2] 镜头检测  (0.27s)
    切换点 : 2 个 -> [4.0, 7.0]
    镜头数 : 3

[3] 选帧规划  (0.00s)
    镜头 3 个 / 抽帧 9 张 / 覆盖 10.0s / 每镜最多 3 张
    / 角色分布 {'head': 3, 'mid': 3, 'tail': 3} / 预估 9.9k tokens

[4] 抽帧  (1.07s)
    成功   : 9/9

[5] 音频分析（跳过 ASR）
    平均音量: -21.1 dB
```

---

## 配置说明

全部配置在 `backend/.env`。完整列表见 `backend/.env.example`，以下是关键项。

### 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_API_KEY` | — | **必填** |
| `VLM_BASE_URL` | `https://api-inference.modelscope.cn/v1` | 兼容 OpenAI 格式的服务地址 |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` | 必须支持视觉输入 |
| `VLM_CONCURRENCY` | `3` | Pass1 分块并行数 |
| `VLM_TIMEOUT` | `180` | 单次请求超时（秒） |

### 抽帧预算（质量与成本的主要旋钮）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_TOTAL_FRAMES` | `48` | 单次请求总帧数上限。48 帧 ≈ 5.3 万 tokens |
| `MAX_FRAMES_PER_SHOT` | `3` | 每个镜头最多取几帧 |
| `LONG_SHOT_SECONDS` | `5.0` | 超过此时长的镜头优先补帧 |
| `FRAME_LONG_EDGE` | `896` | 送入模型的图片长边像素 |
| `FRAME_JPEG_QUALITY` | `82` | JPEG 质量 |

> 把 `FRAME_LONG_EDGE` 提到 1600 以上对反推几乎没有收益，纯粹烧 token。

### 镜头分割

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SCENE_THRESHOLD` | `0.30` | 场景切换阈值，越小越敏感 |
| `MIN_SHOT_SECONDS` | `0.8` | 短于此值的镜头并入上一个，过滤快闪噪声 |

调参经验：

- 口播 / 访谈（画面变化小）→ `0.35`
- 快剪 / 混剪（变化剧烈）→ `0.25`
- 如果检测出的镜头数明显偏多（比如 10 秒出 20 个镜头），调高阈值或调高 `MIN_SHOT_SECONDS`

### 长视频分块

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_THRESHOLD_SECONDS` | `90` | 超过此时长触发分块 |
| `CHUNK_SECONDS` | `60` | 每块目标时长 |
| `MAX_DURATION_SECONDS` | `1800` | 超长视频截断，`0` 表示不限制 |

### 语音转写（可选）

| 变量 | 说明 |
|---|---|
| `ASR_BASE_URL` / `ASR_API_KEY` / `ASR_MODEL` | 任意 OpenAI 兼容的 `/audio/transcriptions` |

留空则跳过音频维度。也可以装 `pip install -e ".[local-asr]"` 用本地 faster-whisper，
代码会优先走本地。

---

## 接口一览

20 个接口，全部在 `/docs` 里可交互调试。

### 系统

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（ffmpeg / 模型 / ASR / yt-dlp 状态） |
| GET | `/api/health/vlm` | **真实调用一次模型**，验证 key 与图片输入 |
| GET | `/api/settings` | 读取配置（key 已脱敏） |
| POST | `/api/settings` | 更新配置（写回 `.env`，有键白名单） |
| GET | `/api/formats` | 四种输出格式及说明 |
| GET | `/api/stats` | 任务统计 |

### 反推

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/upload` | 上传视频（流式落盘，边写边校验大小） |
| POST | `/api/analyze` | 对已上传文件发起反推 |
| POST | `/api/fetch/probe` | 只解析链接信息，不下载 |
| POST | `/api/fetch` | 抓取链接并反推 |

### 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/jobs` | 历史列表（走数据库，含重启前的记录） |
| GET | `/api/jobs/{id}` | 任务详情（含提示词与全部观察结果） |
| GET | `/api/jobs/{id}/events` | **SSE 实时进度** |
| POST | `/api/jobs/{id}/cancel` | 取消任务 |
| DELETE | `/api/jobs/{id}` | 删除记录 |

### 媒体

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/media/upload/{name}` | 视频流，**支持 HTTP Range** |
| GET | `/api/media/frame/{job}/{chunk}/{file}` | 抽帧图片 |
| GET | `/api/media/thumb/{job}` | 缩略图 |
| GET | `/api/media/frames/{job}` | 该任务全部抽帧的 URL 与镜头表 |

### 提示词库

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/prompts` | 收藏一条提示词 |
| GET | `/api/prompts` | 列表，支持关键词搜索 |
| DELETE | `/api/prompts/{id}` | 删除 |

### 完整调用示例

```bash
# 上传
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@demo.mp4"

# 发起反推（用上一步返回的 file_id）
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "file_id": "1790052609_4aedb865.mp4",
    "name": "demo.mp4",
    "options": {
      "format": "h3",
      "language": "en",
      "enable_asr": true,
      "max_total_frames": 48
    }
  }'

# 订阅进度（SSE）
curl -N http://127.0.0.1:8000/api/jobs/<job_id>/events

# 取结果
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

---

## B站 / 抖音抓取

### 策略分层

| 平台 | 通道 1 | 通道 2 | 失败时 |
|---|---|---|---|
| B站 | yt-dlp（官方适配完善，稳定） | — | 提示手动下载 |
| 抖音 | yt-dlp（支持不稳定） | 原生解析：短链 → `item_id` → 分享页 `_ROUTER_DATA` → 取 `play_addr`（`playwm` → `play` 去水印） | 提示手动下载 |

**抖音的签名校验变动频繁，自动抓取不是总能成功。** 这是平台特性，不是实现问题。
失败时前端会明确提示手动下载后上传——**上传通道的分析效果完全一致**，
抓取只是为了省一步操作。

### 需要登录的内容

B站大会员内容、抖音部分视频需要 cookie。在 `.env` 里二选一：

```bash
# 从本机浏览器读取（推荐）
COOKIES_FROM_BROWSER=chrome      # chrome / edge / firefox

# 或直接给 Netscape 格式的 cookies.txt
COOKIES_FILE=D:/cookies.txt
```

### 合规说明

只处理公开可访问的内容；下载件仅存本地用于分析，不做二次分发。

---

## 测试

```bash
cd backend
pytest                    # 全部 80 个
pytest -q tests/test_selection.py     # 只跑选帧策略
pytest -q tests/test_pipeline_e2e.py  # 只跑端到端
ruff check app tests                  # 静态检查
```

### 测试分布

| 文件 | 数量 | 覆盖内容 |
|---|---|---|
| `test_selection.py` | 13 | 预算不超、每镜保底、超预算时均匀降采样、长镜头优先、时间有序 |
| `test_ffmpeg.py` | 22 | 真实视频跑探测 / 场景检测 / 抽帧 / 缩放 / 音频 / 切分 |
| `test_templates.py` | 14 | Pass1 JSON 宽容解析（围栏 / 前后缀 / 尾随逗号 / 垃圾输入）、多块镜号重排 |
| `test_api.py` | 23 | 接口契约、上传校验、Range 流、SQLite 往返、提示词库 |
| `test_pipeline_e2e.py` | 8 | **完整管线**：帧数对齐、预算生效、四种格式、进度事件、分块、无 key 报错 |

### 关于 mock 模型

`tests/mock_vlm.py` 是一个 OpenAI 兼容的最小 mock 服务。端到端测试用它跑完整链路——
真实模型调用有成本、有延迟、依赖外部可用性，但管线里最容易出错的恰恰是**调用之外**的部分：
抽帧、时间戳对齐、Pass1 JSON 解析、多块合并、Pass2 组装。

它按请求里实际出现的时间戳生成对应的镜头列表，所以测试能验证「帧与时间戳是否一一对应」，
而不只是「有没有返回东西」。

也可以单独起它来手动联调，不花一分钱：

```bash
cd backend
python -m tests.mock_vlm 8899

# 另一个终端，指向 mock
VLM_API_KEY=mock-key \
VLM_BASE_URL=http://127.0.0.1:8899/v1 \
VLM_MODEL=mock-vlm \
fastapi dev
```

---

## 已知限制

| 限制 | 说明 |
|---|---|
| 纯视觉帧推不出音频维度 | 台词、口型、BGM、音效必须靠 ASR。不配 ASR 时这些字段只能是 `N/A` |
| 抖音抓取不稳定 | 平台签名频繁变动。失败请手动下载后上传 |
| 抽帧丢失帧间运动 | 抽帧是静态采样，快速运动、光流、变速等靠模型的时序推断，不如原生视频模型准。需要更高精度时可用 `segment_video()` 把分片喂给原生视频接口 |
| 超长视频被截断 | 默认 1800 秒上限，避免单任务跑太久 |
| 任务状态在内存 | 活跃任务的进度事件不持久化，重启后只能看到已落库的终态结果 |
| 无鉴权 | 当前是单机自用定位。对外部署前需要加认证与限流 |

---

## 设计取舍备忘

写代码时做的一些不显然的决定，以及原因：

1. **不依赖 ffprobe。** 系统里常常只有 `ffmpeg.exe`。媒体信息用 `ffmpeg -i` 的 stderr 解析，
   有 ffprobe 时优先用它。少一个依赖，少一类「装不上」的问题。

2. **长视频逻辑分块而非物理切割。** 抽帧用绝对时间定位即可，不需要真的切出文件。
   省掉一次重编码，也天然避免了「把镜头劈成两半」。

3. **结果类字段存 JSON 列。** Pass1 的观察字段会随提示词迭代增删，
   每次都改表结构的迁移成本远高于收益。只把需要筛选排序的字段（state / created_at /
   format / title）建成真实列。

4. **落盘文件名保持纯 ASCII。** 中文文件名虽然好认，但会出现在 URL 里，
   只要有一环没做百分号编码（curl、部分代理、Content-Disposition）就会 400。
   原始中文名放在 `name` 字段里给前端展示。

5. **任务表刻意不加 asyncio.Lock。** 所有字典操作都是同步的（中间没有 await 点），
   在单线程事件循环里本身就是原子的。加了反而会在跨事件循环使用时抛
   `bound to a different event loop`——CLI 和测试需要在新循环里跑同一份 store。

6. **Pass1 提示词里明确禁止静态动词。** `hold` / `remain` / `stay` / `final frame`
   这类词会让生成视频冻结，包括嘴部动作。同理禁止「画面唯一运动是 X」这类排他声明——
   写了之后其他一切都不动了。
