# 视频提示词反推

上传视频，或用 B站 / 抖音 链接，自动反推出**可直接使用的视频生成提示词**。

后端 FastAPI + Python，前端 Vue 3 + Vite，抽帧与镜头分割全部由 ffmpeg 完成，
视觉理解走任意 OpenAI 兼容的多模态模型。

支持**两种反推模式**：

| 模式 | 变体 | 产物 |
| --- | --- | --- |
| **H3 模式** | T2VA 三字段（默认） | 英文，字段名裸名加冒号，`[Shot N]` + 切点时间戳，无任何参考标签 |
| | Ref2VA 六段式 | 英文，六段结构。**「参考」指格式，不是让你参考原视频** —— 从视频里提取主体定义成 `<Subject N>`，供你自己挂参考图；不出现 `<Video 1>` / `<Audio 1>` |
| **Seedance 模式** | 六要素中文段 | 中文连贯段落，按 主体→动作→环境→风格→镜头→声音 顺序，末尾附负面指令 |

另有 `generic`（工具无关分镜表）作为兜底，界面上折进「其他格式」，不占主选择位。

两种 H3 变体的正文都**限 700 词以内**。

---

## 目录

- [它是怎么工作的](#它是怎么工作的)
- [两种模式的区别](#两种模式的区别)
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
     │        ├→ 逐镜头结构化 JSON：景别 / 运镜 / 主体 / 动作 / 光线 / 色调 /
     │        │   台词 / 音效 / 转场 / 置信度
     │        └→ 跨镜头主体登记表：每个主体在哪些镜头出现、哪些特征不能漂
     │
     ▼
  Pass 2「只写不看」：全部观察 JSON + 主体登记表 + 音频报告 + 全局统计
     │        └→ 目标提示词（H3 模式 / Seedance 模式）
     ▼
  落库（SQLite）→ 前端左视频右提示词对照展示
```

**同一份观察结果可以出多种格式**，这是两阶段的额外收益：Pass 1 花钱花时间，
Pass 2 很便宜，想换格式不用重新看一遍视频。

### 主体登记表是为什么加的

`ShotObservation` 是逐镜头视角——同一个角色在第 1 镜和第 5 镜会被描述成两段互不相关的
文字，模型判断不出「这是同一个人」。而 Ref2VA 的 `subject_definitions` 和
`retention_analysis` 恰恰需要跨镜头的主体身份：哪个主体、在第几镜出现、要保留到什么程度。

没有登记表，Pass 2 只能自己编标签，`retention_analysis` 的 `appears in [Shot N]` 就靠猜，
参考标签会漂。所以 Pass 1 除了逐镜头观察，还额外交一份登记表：

```json
"subjects": [
  { "label": "performer", "kind": "person",
    "description": "a performer in a dark quilted jacket, dark hair tied back",
    "shots": ["1", "2", "3", "5"],
    "notes": "the jacket hardware and the tied-back hair must not change" }
]
```

分块会让同一主体在多块里各登记一次，`merge_subjects()` 按标签归一化合并
（`performer` / `The Performer` / `a performer` 归成一条），并把块内镜号映射成全局镜号。

---

## 两种模式的区别

不是换措辞，是换目标模型，写法和硬约束都不一样。

| | H3 模式 | Seedance 模式 |
| --- | --- | --- |
| 语言 | 英文为主（台词保留原文） | 中文 |
| 结构 | 裸字段名 + 冒号，多段 | 连贯段落，无字段名 |
| 镜头标记 | `[Shot N] At MM:SS.mmm` | 末尾一句节拍表 |
| 台词 | `<d>[Language] 原文</d>` + 说话人 `(Sx)` | 引号内联，保留原文 |
| 参考标签 | Ref2VA 有 `<Subject N>`（**没有 `<Video 1>`**） | **禁止出现**，会被当字面文本 |
| 正文长度 | ≤ 700 词 | 紧凑连贯 |
| 兜底 | 不出现参考标签 | 末尾附「不要出现字幕、文字、水印」 |

### H3 模式为什么还要分 T2VA / Ref2VA

两个变体的差别是**输出格式**，不是「有没有参考素材」：

- **T2VA**：纯文字从零生成。主体外观第一次出现就要写全，且**不能出现 `<Subject N>`** ——
  没有参考图可挂，模型会把标签当字面文本画出来。
- **Ref2VA**：输出六段式，把视频里的主体提取成 `<Subject 1..N>`。
  你之后生成时给这些标签挂**自己的**参考图（比如你自己的角色），
  所以定义必须足够具体，光看文字就能认出是什么。

⚠️ **这里的「参考」指格式参考，不是让你参考那段视频。** 原视频只用来提取内容，
不会出现 `<Video 1>` / `<Audio 1>` —— 否则生成结果就被绑死在原片上了，
而你要的是「用我自己的参考图生成」。

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
│   │   │   ├── observation.py      Pass1 结构化观察 + 主体登记表
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

### 换模型不用改文件、不用重启

界面上点右上角的**模型徽标**就能换。面板里可以：

- 选服务商（ModelScope / 百炼 / 火山 / OpenRouter / 本地，一键填地址）
- 改 API Key（留空表示不改，不会把脱敏值写回去覆盖真 key）
- 拉取该服务商的模型列表，**逐个点「测」**看哪个真能用
- 保存后**下一次反推立即生效**，不用重启服务

**「测」是真的发一张纯红图**，回答里出现 `red` 才算「能读图」。
为什么不只测连通性：纯文本模型也能回一句「OK」，你会以为配好了，
真跑反推时才发现它收到图片就报错。自检必须覆盖「图片输入」这个前提。

三种测试结果：

| 结果 | 含义 |
|---|---|
| **能读图** | 可以放心用 |
| **通但读不到图** | 接口通了，但没识别出图片内容 —— 多半是纯文本模型 |
| **无法判定** | 有输出但被 max_tokens 截断。推理模型（思考过程很长）会这样，不代表不能用 |
| **不可用** | 请求没被处理 —— 模型不支持该模态，或账号没开通推理服务 |

> ⚠️ **列表里有 ≠ 你能用。** 服务商常常只返回精选列表，里面有些模型没部署推理服务，
> 调用会报 `has no provider supported`。所以必须逐个测，绿灯的才算数。

### 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_API_KEY` | — | **必填** |
| `VLM_BASE_URL` | `https://api-inference.modelscope.cn/v1` | 兼容 OpenAI 格式的服务地址 |
| `VLM_MODEL` | `Qwen/Qwen3.5-27B` | 必须支持视觉输入，见上面的测试方法 |
| `VLM_CONCURRENCY` | `3` | Pass1 分块并行数 |
| `VLM_AUDIO_INPUT` | `false` | 把音频直接附给模型。**能收音频的模型很少**，见 [音频维度](#音频维度三条路能力不同) |
| `VLM_MAX_TOKENS` | `16384` | 单次回复上限。**推理模型必须给大**，见下 |
| `VLM_DISABLE_THINKING` | `true` | 关掉推理模型的思考过程，见下 |
| `VLM_TIMEOUT` | `180` | 单次请求超时（秒） |

#### 推理模型：关掉思考，并把上限给足

反推是「照结构填内容」，不是解题 —— 推理模型的思考过程对这类任务基本是浪费，
而且会**把 token 预算吃光导致正文为空**。

实测 `deepseek-ai/DeepSeek-V4.1-Flash` 跑 6 帧 Pass1：

| 设置 | 耗时 | 思考过程 | 正文 | JSON 解析 |
|---|---|---|---|---|
| 开思考 + 6144 | 88s | 24871 字 | **0 字** | ❌ |
| 开思考 + 16384 | 236s | 53169 字 | 9053 字 | ✅ |
| **关思考 + 16384** | **15s** | 0 字 | 6576 字 | ✅ |

端到端（13.3s 视频）：**205.7s → 64.3s**。

**参数名各家不一样，必须实测。** 四种里只有一种有效：

| 参数 | 结果 |
|---|---|
| **`enable_thinking: false`（顶层）** | ✅ 有效 |
| `chat_template_kwargs.enable_thinking=false` | ✅ 也有效 |
| `thinking: false` | ❌ 返回空正文 |
| `chat_template_kwargs.thinking=false` | ❌ 仍在思考 |

默认开着是安全的 —— 服务商不认这个字段会报 400，代码会自动摘掉重试。

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

#### 渐变转场会自动重检

上面的阈值对**硬切**很准，但会漏**渐变转场**。特效类内容
（能量爆发、闪白、溶解、粒子过渡）大量使用软转场。
实测一段 13.3 秒的特效视频：

| 阈值 | 切点 | 镜头 | 抽帧 |
|---|---|---|---|
| 0.30（默认） | 1 | 2 | **6** |
| 0.20 | 6 | 3 | — |
| **0.15** | **10** | **4** | **12** |

**镜头少 → 抽帧少 → 模型看到的信息少**，反推质量直接掉。

所以镜头检测是自适应的：如果**平均镜头长度超过 6 秒**且视频不短于 8 秒
（正常剪辑不会这样），会自动降一半阈值重检一次，同时把最短镜头时长提到
1.0 秒滤掉噪声切点；只有结果明显更多（≥1.5 倍）才采用。

排查时可以看 `/api/jobs/{id}` 的 `stats`：

- `scene_cuts` —— 实际检出的切点数
- `scene_threshold` —— 最终用的阈值（和配置不同就说明触发了自适应）
- `scene_adaptive` —— 自适应说明，没触发时为空

另一个线索：**模型报的镜头数明显多于检出的镜头数，就是漏检了**。

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

留空则跳过语音识别。也可以装 `pip install -e ".[local-asr]"` 用本地 faster-whisper，
代码会优先走本地。

**但要反推有台词/唱歌的视频，ASR 基本是必需的** —— 画面帧推不出逐字台词和口型，
而 H3 的 `<d>[Language] ...</d>` 要求原文逐字。没有 ASR 时这些字段只能是 `N/A`。

### 音频维度：三条路，能力不同

很多人以为「多模态模型能传音频」，这个说法对了一半。实际上音频有三条路，
它们解决的问题**不一样**：

| 路径 | 给你什么 | 能填的字段 |
|---|---|---|
| **ASR**（`ASR_*`） | 逐句时间戳 + 逐字原文 | 台词、口型对齐、`<d>` 段落 |
| **模型听音频**（`VLM_AUDIO_INPUT`） | 一段听觉描述 | `overall_soundscape`、`non_diegetic_music` |
| **频谱实测**（永远可用，无需配置） | 频段能量分布 | 倾向性描述，如「疑似有节奏性音乐编排」 |

**三条不能互相替代。** ASR 给文字但不告诉你有没有鼓；模型听能给描述但给不出逐字原文。

#### 频谱实测（不配任何东西就有）

没配 ASR 时音频维度**不会只剩 N/A**。ffmpeg 会测两个频段的相对能量：

- 语音频段 300–3400Hz 相对能量 → 能量是否集中在人声频段
- 低频 <200Hz 相对能量 → 是否疑似有节奏性音乐编排

实测校准：

| 素材 | 语音频段 | 低频 | 解读 |
|---|---|---|---|
| 流行 MV（人声+配器） | -5.6dB | -4.3dB | 混有配器 + 疑似有节奏性音乐编排 |
| 纯 440Hz 正弦 | -1.7dB | -27.7dB | 窄带音调，低频很弱 |

两个坑记在这：

1. **滤波链要串两遍。** ffmpeg 的 highpass/lowpass 是单极点 6dB/oct，太缓 ——
   单极点 `lowpass=f=200` 挡不住 440Hz，纯正弦的低频相对能量实测 -13.8dB
   （明显是泄漏，会误判成「有中等低频」）；串两遍 12dB/oct 后降到 -27.7dB。
2. **频谱分不出人声和窄带音调。** 纯 440Hz 正弦的语音频段能量（-1.7dB）和说话一样高。
   所以措辞只能停在「能量高度集中在该频段」，**不能写「疑似以人声为主」**。

#### 模型听音频（`VLM_AUDIO_INPUT`，默认关）

协议上走 OpenAI 的 `input_audio` 内容块，音频以 base64 附带（单声道 16kHz，60 秒约 1.8MB）。

⚠️ **能收音频的模型很少。** 很多「多模态」模型只支持图片。2026-09 实测：

| 模型 | 结果 |
|---|---|
| `Qwen/Qwen3.5-27B` | ❌ `choices` 空 —— 静默忽略音频 |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | ❌ `has no provider supported` |
| `Qwen/Qwen2.5-Omni-7B` | ❌ 同上 |
| `Qwen/Qwen2-Audio-7B-Instruct` | ❌ 同上 |

`has no provider supported` = 模型在目录里有，但该账号下没部署推理服务。
**目录里有 ≠ 你能用。** 换到确实能听的模型（Gemini / GPT-4o-audio / 其他平台的
Qwen-Omni）再把这个开关打开。

开了但模型不支持会返回 `choices: null`，代码会明确报出来并提示关掉这个开关。
音频只是增强：模型明确拒绝时会先丢音频重试，不会因为它一个人把整个任务打死。

### 抓取（可选）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `COOKIES_FROM_BROWSER` | 空 | 从本机浏览器读 cookie，如 `chrome` |
| `COOKIES_FILE` | 空 | 或给 Netscape 格式的 cookies.txt 路径 |
| `YTDLP_FORMAT` | 720p 封顶 | 透传给 yt-dlp 的 `-f`，见 [清晰度封顶](#清晰度封顶) |

---

## 接口一览

23 个接口，全部在 `/docs` 里可交互调试。

### 系统

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（ffmpeg / 模型 / ASR / yt-dlp 状态） |
| GET | `/api/health/vlm` | **真实调用一次模型并附一张图**，验证 key 与图片输入 |
| POST | `/api/health/vlm` | 试一组未保存的模型配置（只改模型名时不用重填 key） |
| GET | `/api/models` | 拉服务商声明的模型列表（仅作候选，需逐个测） |
| GET | `/api/settings` | 读取配置（key 已脱敏） |
| POST | `/api/settings` | 更新配置（写回 `.env` 并同步进程环境变量，有键白名单） |
| GET | `/api/formats` | 反推模式与变体（按模式分组，`primary` 标出两种主模式） |
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

### 清晰度封顶

反推会把帧缩到长边 `FRAME_LONG_EDGE`（默认 896）再送模型，
所以下载 1080p 是纯浪费带宽和等待时间。默认封顶 720p：

```bash
YTDLP_FORMAT=bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/bv*+ba/b
```

想强制原画质就写 `bv*+ba/b`。实测 212 秒的 B站 1080p 视频，封顶后下载到 1280x720，
整条链路（下载 → 46 镜头 → 5 块 → 反推）66 秒跑完。

### 一个容易误判的坑

`bv*+ba` 这类格式**合并音视频必须调用 ffmpeg**。本机 ffmpeg 常常不在 PATH 里
（只有 `ffmpeg-static` 那种单文件），所以代码会显式把路径传给 yt-dlp 的
`ffmpeg_location`。少了这一步，现象是「链接能解析成功、但下载失败」——
很容易误判成平台反爬，然后往 cookie 方向排查，方向全错。

失败提示里现在会带上 yt-dlp 的原始报错，就是为了避免这种误判。

### 合规说明

只处理公开可访问的内容；下载件仅存本地用于分析，不做二次分发。

---

## 测试

```bash
cd backend
pytest                    # 全部 227 个
pytest -q tests/test_selection.py     # 只跑选帧策略
pytest -q tests/test_pipeline_e2e.py  # 只跑端到端
ruff check app tests                  # 静态检查
```

### 测试分布

| 文件 | 数量 | 覆盖内容 |
|---|---|---|
| `test_selection.py` | 13 | 预算不超、每镜保底、超预算时均匀降采样、长镜头优先、时间有序 |
| `test_ffmpeg.py` | 33 | 真实视频跑探测 / 场景检测 / 抽帧 / 缩放 / 音频 / 切分 / **频段能量** / **音频片段抽取** / **自适应镜头检测** |
| `test_templates.py` | 41 | Pass1 JSON 宽容解析（围栏 / 前后缀 / 尾随逗号 / 垃圾输入）、多块镜号重排、主体登记表解析与跨块合并、两种模式的模板硬约束、占位符替换干净、**音频未知时的编造禁令**、**频谱描述的措辞边界** |
| `test_downloader.py` | 24 | 中文分享文案取链接、平台识别、aweme_id、yt-dlp 选项（**含 `ffmpeg_location` 回归**）、清晰度封顶、失败原因上传 |
| `test_vlm.py` | 28 | 请求体构造（data URI / Anthropic 块 / **`input_audio` 音频块**）、响应解析（含 `choices: null`）、**空响应原因诊断**、音频格式白名单与体积上限、base_url 带不带 `/v1` 都能用 |
| `test_api.py` | 36 | 接口契约、模式分组、上传校验、Range 流、SQLite 往返、缺列自愈、提示词库、**模型热切换**、**测试不污染真实 .env** |
| `test_pipeline_e2e.py` | 15 | **完整管线**：帧数对齐、预算生效、四种格式、主体登记表贯通到 Pass2、模式不串味、进度事件、分块、无 key 报错 |

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

## 验证记录

不是「写完就交」的清单，是实际跑过的路径。

| 路径 | 数据 | 结果 |
|---|---|---|
| 上传 → H3 T2VA | 10s 合成片（无场景切换） | 6 帧 / 6 镜头 / 1 块 |
| 上传 → H3 Ref2VA | 同上 | `<Subject 2>` 存在且无 `<Subject 3>`，`retention_analysis` 行数 == 主体数 |
| 上传 → Seedance | 同上 | 无 H3 字段名、无参考标签、有节拍表、纯中文 |
| 中文名上传 | `演示片段.mp4` | `name` 保留中文，落盘 `file_id` 纯 ASCII，Range 返回 206 |
| **B站链接 → 长视频分块** | BV1GJ411x7h7，**212.3s** | 下载 1280x720 → **46 镜头 → 5 块** → 46 帧 / 50.6k tokens，全程 **66 秒** |
| **真实模型端到端** | `Qwen/Qwen3.5-27B`，10s 测试图案 | 112 秒出完整六段式 Ref2VA，结构、时间戳格式、retention 标记全对 |
| **真实模型 + 真实素材** | `DeepSeek-V4.1-Flash`，13.3s / 1080p 特效视频 | 88 秒，切点 10 / 镜头 4 / 抽帧 12 / 主体 8，正文 572 词 |
| 前端构建 | `vue-tsc + vite build` | 通过，112 模块 |
| 静态检查 | `ruff check app tests` | 通过 |
| 测试 | — | **227 passed** |

**模型真的在看图**：喂 SMPTE 彩条帧，它正确识别出彩条布局，并读出了画面里的
实际数字（6.0s 那帧是 `'6'`，9.5s 是 `'9'`）。

### 实测可用 / 不可用的模型

| 模型 | 结果 |
|---|---|
| **`Qwen/Qwen3.5-27B`** | ✅ 可用。红蓝对照测试通过，Pass1 JSON 稳定可解析 |
| `Shanghai_AI_Laboratory/Intern-S2-Preview` | ⚠️ 能读图，但会把思考过程写进 content，不守「只输出 JSON」 |
| `PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT` | ❌ `choices: null` + 0 tokens |
| `OpenGVLab/InternVL3_5-241B-A28B` | ❌ 同上 |
| `Shanghai_AI_Laboratory/Intern-S1-mini` | ❌ 同上 |

`choices: null` + `usage` 全 0 表示**请求根本没被处理**——通常是模型不支持图片输入，
或该账号没开通推理服务。代码会明确报出这个原因，不会只说「返回空内容」。

> ⚠️ 服务商的 `/v1/models` 列表**不代表账号真实可用范围**。列表里名字带 `VL` 的模型
> 也可能调不通，必须实测。判断「是否真能读图」要做**对照测试**
> （喂纯红/纯蓝图看回答是否区分得开），只问一次答对了不算证据——可能是猜的。

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

7. **模式只是分组，`format` 是唯一真源。** 没有单独的 `mode` 请求字段——
   否则会出现 `mode=h3` 配 `format=seedance` 这种自相矛盾的组合。
   `mode` 由 `templates.MODE_OF_FORMAT` 从 `format` 推导。

8. **`_COMMON_RULES` 要单独 format 一次。** `str.format` 不递归替换被代入的值：
   `template.format(common=_COMMON_RULES)` 之后，`_COMMON_RULES` 自己带的
   `{language_instruction}` 会原样漏给模型。所以 `build_pass2_system` 先对公共规则
   单独 format 一次，再代入外层模板。有测试守着这条。

9. **`init_db` 会补缺失的列。** `create_all` 只建新表、不改老表，而这个项目的结果字段
   还会继续长。补列只做加法（新增列一律可空），不会丢数据，避免老库一升级就报
   `no such column`。

10. **主体标签跨块判重要先剥冠词。** 模型会在不同块里把同一个主体写成
    `performer` / `The Performer` / `a performer`，不归一化就会写出三条 `<Subject N>`。
    但冠词只在后面紧跟空格时才剥，否则 `anime style` 会被吃成 `imestyle`。

11. **「未知」要显式标注为未知，并禁止推测。** 只写中性陈述（「未获得文本内容」）
    等于留白，模型一定会去填。实测 `enable_asr=False` 时，模型照着画面编出了
    「电子提示音与数字跳变同步」并标成 `fully_copy`——听起来完全合理，但全是假的。
    已实测出**三条**编造路径，每条都要单独堵：① 未知维度被留白；② 测量数据被越权解读
    （「能量集中语音频段」→「有人在唱歌」）；③ **视觉事件被转成声音事件**
    （看到走路写 `footsteps`、看到衣料写 `cloth rustle`）。
    第三条最隐蔽——它不是凭空编，而是「合理地推」，推的正是未知的那个维度，
    所以规则里必须点出具体反例，抽象禁令拦不住。

12. **排他性运动禁令两阶段都要写。** Pass2 早就禁止「the only motion is X」，
    但 Pass1 没有——而 Pass1 的 `global_notes` 会原样喂进 Pass2。
    只堵一处等于没堵。

13. **音频三条路不能互相替代。** ASR 给逐句时间戳 + 逐字原文；
    模型听给一段描述；频谱实测给能量分布。目标格式要求台词逐字原文，描述做不到；
    ASR 给文字但不会告诉你有没有鼓。所以三个都要留，各自填不同的字段。

14. **模型自检必须发图，不能只测连通性。** 纯文本模型也能回「OK」，
    用户会以为配好了，真跑反推时才发现收到图片就报错。
    现在发一张纯红图，回答里出现 `red` 才算通过 —— 这是对照测试的思路。
    另外自检的 `max_tokens` 不能太小：推理模型会先输出一大段思考，
    16 个 token 被吃光后报 `finish_reason=length`，看起来像「不可用」，
    实际只是额度不够。这种情形判为「无法判定」。

15. **`POST /api/settings` 要同时写 `.env` 和 `os.environ`。**
    pydantic-settings 的优先级是「环境变量 > .env」。如果服务是用
    `VLM_MODEL=x uvicorn ...` 起的，只写 `.env` 会被旧的环境变量盖住 ——
    界面上点保存毫无反应，而且不报错，是最难查的那类问题。

16. **配置写入目标要能被测试改向。** 否则跑一次 pytest 就把用户的
    `backend/.env` 改成测试里的假模型名（实测被写成 `Vendor/X`），
    测试全绿但服务起不来。现在由 `ENV_FILE` 环境变量指定，测试指向临时文件，
    并有一条测试断言真实 `.env` 的字节完全没变。
