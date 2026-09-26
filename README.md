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

**可以只推你要的那一段**：上传或抓取完成后先预览，拖动时间轴框出区间再反推。
无关的片头片尾不会被写进提示词，时间戳也是相对这个片段的。

---

## 它是怎么工作的

**一次模型调用**：帧 + 时间戳 + 用户说明 → 目标格式提示词。

```
上传 / 抓取
     │
     ├─ ffprobe（缺失时用 ffmpeg -i 解析）──→ 时长 / 分辨率 / 帧率 / 有无音轨
     │
     ├─ ffmpeg select='gt(scene,T)' ──────→ 场景变化点 → 决定在哪些位置抽帧
     │
     ├─ ffmpeg volumedetect + silencedetect → 音量曲线 / 静音段 / 疑似卡点
     │
     ├─ ASR（可选）─────────────────────→ 带时间戳的台词 / 歌词
     │
     ▼
  抽帧（按场景变化点对齐，镜头内按 0.5s 间隔）
     │
     ▼
  一次调用：帧图 + 时间戳清单 + 音频报告 + 用户说明
     │        └→ 目标提示词（H3 模式 / Seedance 模式）
     ▼
  落库（SQLite）→ 前端左视频右提示词对照展示
```

### 为什么不拆成两阶段

曾经是 Pass1「结构化观察」+ Pass2「成文」两阶段，理由是「一次性看图直接写会丢细节」。
但那是**每镜只有 1~3 帧**的旧配置下的结论。抽帧密度提到 2fps（每镜多帧）之后，
两阶段的主要作用变成了「教模型怎么观察」—— 而实测**每收紧一次这类规则、输出就退化一次**：

| 加过的规则 | 实际后果 |
|---|---|
| 「只报告帧里真实存在的」 | 压掉了合理推断，动作描述只剩 19%，读起来像照片说明 |
| 把「相机静止」当结论喂进去 | 被外推成「人物也静止」，写出一堆 `stands` |
| 「动作要占大头」 | 又得再写一段「不许编动作」去中和它 |
| 「20 秒 MV 大约 5-20 个镜头」 | 快切素材被压到 20 以内 |

**约束输出格式 ≠ 约束思考。** 现在只给三样：身份、任务、目标格式规则，
外加两条关于**产物**的硬约束（静止动词会让生成的视频冻住；音频未知时编造比留空更糟）。

同一段素材的实测对比：

| | 两阶段 | 现在（一次调用） |
|---|---|---|
| 系统提示词 | 13450 字符 | **2928 字符** |
| 端到端耗时 | 112~136s | **26s** |
| 运镜 | 全 `static` | `camera slowly pushing in` / `tracks low and sideways, then tilts up` / `glides forward and down` |

**H3 与 Seedance 的格式规则完全保留** —— 那是输出契约，错了下游解析不了。
删掉的是「教模型怎么观察」的部分。


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
│   │   │   ├── observation.py      镜头 / 主体数据结构（历史任务兼容）
│   │   │   ├── job.py              任务 / 进度 / 结果 / 选项
│   │   │   └── api.py              HTTP 请求响应体
│   │   ├── models/job.py           SQLModel 表定义
│   │   └── services/               业务实现
│   │       ├── ffmpeg.py           探测 / 抽帧 / 场景检测 / 片段截取
│   │       ├── selection.py        自适应选帧（token 预算 + 镜头加权）
│   │       ├── asr.py              语音转写 + 音量分析 + 音乐画像
│   │       ├── audio_features.py   纯 Python 音频特征（BPM / 音头 / 频段平衡）
│   │       ├── vlm.py              多模态客户端（并发 / 重试 / 减帧降级）
│   │       ├── templates.py        系统提示词（身份 + 任务 + 目标格式）
│   │       ├── pipeline.py         管线编排
│   │       ├── downloader.py       B站 / 抖音抓取
│   │       ├── storage.py          SQLite 读写
│   │       ├── jobs.py             任务状态 + 事件总线
│   │       └── runner.py           任务启动器
│   ├── tests/                      272 个测试
│   │   ├── mock_vlm.py             OpenAI 兼容 mock 模型（E2E 用）
│   │   ├── test_templates.py       模板与格式规则（52）
│   │   ├── test_api.py             接口集成（41）
│   │   ├── test_ffmpeg.py          ffmpeg 层（38，跑真实视频）
│   │   ├── test_vlm.py             模型客户端（37）
│   │   ├── test_selection.py       选帧策略（33）
│   │   ├── test_downloader.py      抓取（24）
│   │   ├── test_pipeline_e2e.py    端到端管线（18）
│   │   ├── test_asr.py             音频分析（13）
│   │   └── test_safe_delete.py     删除保护（9）
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

### ffmpeg

**已经打包在仓库里了**（Windows 79 MB / Linux 76 MB 各一份，ffmpeg 6.1.1）。
部署时不需要在服务器上另装。

查找顺序：

1. `backend/.env` 里的 `FFMPEG_PATH`（留空即跳过）
2. **`backend/vendor/ffmpeg/`** ← 自带的那份
3. `backend/runtime/`
4. 系统 `PATH`
5. 常见安装目录（`/usr/bin`、`C:/ffmpeg/bin` 等）

自带二进制排在 `PATH` 之前，是为了让行为可复现 —— 服务器上装了什么版本
不该影响这个服务的输出。

**不需要 ffprobe** —— 缺失时自动改用 `ffmpeg -i` 解析媒体信息。

**两个平台的二进制都在仓库里**，部署到哪边就用哪边：

| 平台 | 文件 | 大小 |
|---|---|---|
| Windows | `backend/vendor/ffmpeg/ffmpeg.exe` | 79 MB |
| Linux x64 | `backend/vendor/ffmpeg/ffmpeg` | 76 MB |

查找逻辑会自动按平台加不加 `.exe` 后缀，两份各自会被找到。

> Linux 那份是 6.1.1，和 Windows 同版本（取自 ffmpeg-static b6.1.1 的 linux-x64），
> 避免两个平台产出不一致。可执行位（100755）已经在 git 里标记好了。
>
> 换别的架构（arm64 等）时，把对应平台的 ffmpeg 放进同一目录、或用
> `FFMPEG_PATH` 指过去即可。

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
uvicorn app.main:app --reload              # 开发（热重载）
uvicorn app.main:app --host 0.0.0.0 --port 8000   # 生产
```

> 用 `uvicorn` 而不是 `fastapi dev` / `fastapi run` —— 后者需要额外装
> `fastapi[standard]`，而 `uvicorn` 已经是本项目声明的依赖，装上就能用。
> 想用 `fastapi` 命令的话，`pip install "fastapi[standard]"` 即可，
> 入口已经配好在 `pyproject.toml` 的 `[tool.fastapi]` 里。

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

完整配置项见 `backend/.env.example`（每一项都有注释）。这里只列常用的。

### 模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLM_API_KEY` | — | **必填** |
| `VLM_BASE_URL` | `https://api-inference.modelscope.cn/v1` | 任意 OpenAI 兼容地址 |
| `VLM_MODEL` | `Qwen/Qwen3.5-27B` | **必须支持视觉输入** |
| `VLM_CONCURRENCY` | `3` | 并发请求数（长视频分块时用） |
| `VLM_MAX_TOKENS` | `16384` | 单次回复上限，**推理模型必须给大** |
| `VLM_DISABLE_THINKING` | `true` | 关掉推理模型的思考过程（能快近 10 倍） |
| `VLM_TIMEOUT` | `180` | 单次请求超时（秒） |
| `PROMPT_WORD_LIMIT` | `0`（不压缩） | 词数上限，超了自动压缩。**默认关** —— 限制长度会逼模型合并镜头，长度就是还原度。视频模型吃不下时再设成正数 |

> ⚠️ **模型列表里有 ≠ 你能用。** 服务商常只返回精选列表，里面有些没部署推理服务。
> 界面上点模型徽标可以逐个实测 —— 会真的发一张纯红图，答出 `red` 才算通过。
> 只测连通性没用：纯文本模型也会回「OK」。

### 抽帧预算（质量与成本的主要旋钮）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FRAME_INTERVAL_SECONDS` | `0.5` | 镜头内平均多久取一帧（0.5 = 每秒两帧） |
| `MAX_FRAMES_PER_SHOT` | `24` | 单个镜头的帧数**硬上限** |
| `MAX_TOTAL_FRAMES` | `96` | 总帧数安全网，只在长视频里起作用 |
| `FRAME_LONG_EDGE` | `896` | 送入模型的图片长边像素 |
| `LONG_SHOT_SECONDS` | `5.0` | 超过此时长的镜头优先补帧 |

**帧数由镜头时长决定，不是由总预算决定**：

```
每个镜头的帧数 = clamp(镜头时长 / FRAME_INTERVAL_SECONDS, 1, MAX_FRAMES_PER_SHOT)
```

一帧约 284 tokens，所以帧数几乎不可能是瓶颈（96 帧 ≈ 2.7 万 tokens）。
真正的限制是延迟和收益递减。按「每秒约一帧」抽，15 秒视频 15 帧就够。

**一镜到底的视频**：所有内容在一个镜头里，帧数完全由 `MAX_FRAMES_PER_SHOT` 决定。
默认 24 意味着 15 秒的单镜能拿到 15 帧（一秒一帧）；想让 60 秒的单镜也一秒一帧，
把上限调到 60。

### 语音转写

**推荐接线上 API**，免费额度就够用，不用下模型、不占内存：

| 服务 | 免费额度 | 配置 |
|---|---|---|
| **硅基流动**（国内直连） | `FunAudioLLM/SenseVoiceSmall` 等标为免费 | `ASR_BASE_URL=https://api.siliconflow.cn/v1` |
| **Groq** | `whisper-large-v3-turbo`，2000 次/日 | `ASR_BASE_URL=https://api.groq.com/openai/v1` |

⚠️ 硅基流动的域名是 **`.cn`**（旧的 `.com` 会把正确的 Key 打成 401）。
只要 `ASR_BASE_URL` + `ASR_API_KEY` 都填了就走线上，不需要装任何本地包。

不配也能跑，但音频维度会缺一整个 —— 有台词/唱歌的视频，歌词字段只能写 `N/A`。

### 音频

| 变量 | 说明 |
|---|---|
| `VOCAL_ISOLATION` | 转写前做人声频段分离，提高歌词准确率 |

### 其他

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SCENE_THRESHOLD` | `0.30` | 场景切换阈值，越小越敏感 |
| `MIN_SHOT_SECONDS` | `0.8` | 最短镜头时长，抑制快闪噪声 |
| `CHUNK_THRESHOLD_SECONDS` | `90` | 超过多少秒触发分块 |
| `MAX_UPLOAD_MB` | `500` | 上传大小上限 |
| `HISTORY_KEEP_DAYS` | `30` | 任务历史保留天数（0 = 永久） |

## 接口一览

22 个接口，全部在 `/docs` 里可交互调试。

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
| POST | `/api/fetch/download` | **只下载不反推** —— 返回和上传一样的信息，供前端预览并框选片段 |
| POST | `/api/fetch` | 抓取链接并反推（一步到位，不给用户框选的机会） |

#### `/api/analyze` 的请求体

```jsonc
{
  "file_id": "1790075576_6512cc45.mp4",
  "name": "参考.mp4",
  "options": {
    "format": "h3-ref",              // h3 | h3-ref | seedance | generic
    "enable_asr": false,
    "content_hint": "一镜到底的跟拍运镜；主角是白发少女，赛博朋克冷色调",
    "trim_start": 5.5,               // 只推这一段（秒，相对原片）
    "trim_end": 11.2,                // 两个必须同时给；不给就是整片
    "max_total_frames": null,        // 留空用服务端默认
    "frame_interval_seconds": null,
    "prompt_word_limit": null
  }
}
```

**关于 `content_hint`**：用户自己写的画面说明，随帧一起送给模型。

静态帧判断不出三件事，而它们直接影响产出质量：

- **这段是一镜到底还是多镜头切换** —— 静态帧里看不出来
- **主体是谁**（角色 / 作品 / 产品 / 地点）—— 模型不认识冷门 IP
- **动作的前因后果** —— 只看到中间一段会误判

补一句话比让模型瞎猜强得多 —— 这是**外部知识**，模型再怎么推理也推不出来。
提示词里同时声明「画面与说明冲突时**相信画面**」，防止它把说明当观察结果照抄。

**关于 `trim_*`**：给了就只分析这段区间，返回的时间戳也是**相对这个片段**的
（从 0 开始），而不是原片的绝对时间 —— 因为你要拿它去生成一段新视频。

实现上是**先把片段切出来再跑整条管线**，所以探测、镜头检测、抽帧、音频、
时间戳全部天然是相对片段的。另一种做法（把偏移量传遍管线、各处记得减）
要在七八个地方各自维护，每漏一处就是一个隐蔽 bug。

截取是**帧精确**的（重新编码，不是 `-c copy`）—— 用 `-c copy` 会把切点
吸附到最近的关键帧，你选了 5.5s 却从 4.8s 开始，推出来的提示词就对不上了。

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

## 查看实际用的提示词

提示词是这个项目的核心资产 —— 改一个字都影响产出质量。所以有个导出命令，
不用在 Python 字符串里翻：

```bash
cd backend
python -m app.dump_prompts prompts-dump
```

导出 6 个文件：

| 文件 | 内容 |
|---|---|
| `system_h3.txt` | H3 T2VA 的系统提示词（身份 + 任务 + 三字段格式） |
| `system_h3-ref.txt` | H3 Ref2VA 六段式 |
| `system_seedance.txt` | Seedance 六要素中文段 |
| `system_generic.txt` | 通用格式 |
| `user.txt` | 用户消息样例 —— 能看到帧时间戳清单长什么样 |
| `payload.json` | 实际发出去的 HTTP body 骨架 |

导出目录默认 `prompts-dump/`，可以传参数改。已在 `.gitignore` 里 ——
它是生成物，随时能重新导出。

### 一次请求实际提交了什么

```jsonc
{
  "model": "...",
  "messages": [
    { "role": "system", "content": "<约 2900 字符的系统提示词>" },
    { "role": "user", "content": [
        { "type": "text", "text": "<帧时间戳列表 + 音频 + 用户画面说明>" },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
    ]}
  ],
  "max_tokens": 16384,
  "enable_thinking": false
}
```

**一次调用**出稿 —— 没有第二个「成文」阶段，模型看图的同时就在写提示词。

一次请求的体量：2 帧约 194 KB，**其中 100% 是 base64 图片**
（每帧约 75KB → base64 后约 100KB → 约 284 tokens）。

## 测试

```bash
cd backend
pytest                    # 全部 307 个
pytest -q tests/test_selection.py     # 只跑选帧策略
pytest -q tests/test_pipeline_e2e.py  # 只跑端到端
ruff check app tests                  # 静态检查
```

### 测试分布

| 文件 | 数量 | 覆盖内容 |
|---|---|---|
| `test_selection.py` | 33 | 预算不超、每镜保底、**按镜头时长定帧数**、**每镜全覆盖**、`max_per_shot` 大于 3 生效、帧间隔调密度 |
| `test_ffmpeg.py` | 38 | 真实视频跑探测 / 场景检测 / 抽帧 / 缩放 / 音频 / 切分 / 频段能量 / 音频片段抽取 / 自适应镜头检测 / **片段截取（帧精确）** |
| `test_templates.py` | 52 | **四种格式的输出契约**（H3 三字段 / Ref2VA 六段 / Seedance 六要素）、自由发挥提示词只带身份任务格式、帧时间戳清单、镜头边界标注、**逐段长度预算**、**压缩指令与结构校验** |
| `test_downloader.py` | 24 | 中文分享文案取链接、平台识别、aweme_id、yt-dlp 选项（**含 `ffmpeg_location` 回归**）、清晰度封顶、失败原因上传 |
| `test_ffmpeg_vendor.py` | 7 | ffmpeg 打包位置、跨平台查找优先级、**不硬编码开发机路径** |
| `test_vlm.py` | 37 | 请求体构造（data URI / Anthropic 块 / **`input_audio` 音频块**）、响应解析（含 `choices: null`）、**空响应原因诊断**、音频格式白名单与体积上限、**关思考与按次覆盖**、base_url 带不带 `/v1` 都能用 |
| `test_api.py` | 41 | 接口契约、模式分组、上传校验、Range 流、SQLite 往返、缺列自愈、提示词库、**模型热切换**、**测试不污染真实 .env** |
| `test_pipeline_e2e.py` | 18 | **完整管线**：帧数对齐、预算生效、四种格式、默认只调一次模型、进度事件、无 key 报错、**片段截取只推选中区间**、预览地址指向片段 |

### 关于 mock 模型

`tests/mock_vlm.py` 是一个 OpenAI 兼容的最小 mock 服务。端到端测试用它跑完整链路——
真实模型调用有成本、有延迟、依赖外部可用性，但管线里最容易出错的恰恰是**调用之外**的部分：
抽帧、时间戳对齐、帧清单构造、格式规则、提示词压缩与结构校验。

它按请求里实际出现的时间戳生成对应的镜头列表，所以测试能验证「帧与时间戳是否一一对应」，
而不只是「有没有返回东西」。它也会按系统提示词判断这次带图调用是要结构化 JSON
还是直接要提示词。

也可以单独起它来手动联调，不花一分钱：

```bash
cd backend
python -m tests.mock_vlm 8899

# 另一个终端，指向 mock
VLM_API_KEY=mock-key \
VLM_BASE_URL=http://127.0.0.1:8899/v1 \
VLM_MODEL=mock-vlm \
uvicorn app.main:app --reload
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

## 部署到服务器

**默认没有鉴权。对外暴露前必须在 `backend/.env` 里设 `AUTH_PASSWORD`**，
否则谁能连上就能提交任务 —— 攻击者不需要偷你的 VLM 密钥，
直接拿服务器当免费代理，每次调用都烧你的额度。

```bash
AUTH_USERNAME=admin
AUTH_PASSWORD=换成你的强密码
```

设上之后浏览器会弹登录框（HTTP Basic）。**不设 = 不启用**，本地开发不受影响。
健康检查 `/api/health` 是无条件放行的 —— Docker 的 HEALTHCHECK 靠它。

> ⚠️ **「换端口」「加 Nginx 反代」都不算防护。** 反代只是转发请求，不是权限；
> 不配 `auth_basic` 的 Nginx 和直接暴露 8000 端口安全性完全一样。
> 真正起作用的是应用里这两行，或者 Nginx 的 `auth_basic`（二选一即可）。
>
> ⚠️ 纯 HTTP 下 Basic Auth 的密码是明文传输的，务必同时上 HTTPS
> （Nginx + certbot），或者干脆只绑 `127.0.0.1` 走 SSH 隧道。

两种部署方式。**推荐 Docker** —— 它把「装 Python 依赖、装 ffmpeg、构建前端」三步
都封在镜像里，而且**不需要把仓库那 155MB 的二进制拉下来**
（`.dockerignore` 排掉了 Windows 那份，构建上下文只有几十 MB）。

### 方式一：Docker（推荐）

它把「装 Python 依赖、装 ffmpeg、构建前端」三步都封在镜像里，
而且**不需要把仓库那 155MB 的二进制拉下来**
（`.dockerignore` 排掉了 Windows 那份，构建上下文只有几十 MB）。

#### 先把代码弄到服务器 —— 三条路，任选

| 方式 | 适合 | 做法 |
|---|---|---|
| **A. 推 git 仓库再 clone** | 有 GitHub / Gitee / 自建 Git | `git push` → 服务器 `git clone --depth 1` |
| **B. 直接传源码** | 不想建仓库 | `rsync -av --exclude-from=.dockerignore --exclude=.git ./ user@server:/opt/app/` |
| **C. 本地构建镜像再传** | 服务器配置低、或构建要联网 | `docker build -t vpr .` → `docker save vpr \| gzip > vpr.tgz` → 传上去 `docker load` |

方式 A 最省事，也方便以后更新（`git pull && docker compose up -d --build`）。
方式 C 的服务器**完全不需要构建**，也不依赖 pip/npm 网络，代价是要传一个几百 MB 的
tar 包。

> `git clone` 加 `--depth 1` 只要最新版本，快很多。

#### 起服务

```bash
cp backend/.env.example backend/.env      # 填 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL
docker compose up -d --build
docker compose logs -f                    # 看启动日志
```

访问 `http://<服务器IP>:8000`。

#### 国内网络

依赖源默认已经走国内镜像（pip 用清华、npm 用 npmmirror），不用额外配置。
但**基础镜像**（`node:20-alpine`、`python:3.12-slim`）是从 Docker Hub 拉的，
国内可能很慢 —— 给宿主配个加速器：

```bash
# /etc/docker/daemon.json
{ "registry-mirrors": ["https://docker.m.daocloud.io"] }
# 然后 sudo systemctl restart docker
```

#### 两个坑

**数据目录权限**：容器以 uid 1000 运行，宿主目录属主不对的话上传会失败：

```bash
sudo chown -R 1000:1000 ./data
```

**重建容器不丢数据**：`data/` 挂了卷，任务历史、上传的视频、抽帧都在宿主上。

#### 资源限制

`docker-compose.yml` 里默认 `cpus: 3.0` / `mem_limit: 3g`（按 4 核机器留一个核
给系统）。按自己机器改。

### 方式二：直接跑 Python

#### 1. 拉代码

```bash
git clone <repo> && cd <repo>
```

ffmpeg 跟着仓库来，不用另装。注意仓库里带了**两个平台**的二进制（共 155 MB），
克隆会比一般项目慢一些 —— 或者用 `git clone --depth 1` 只要最新版本。

#### 2. 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

编辑 `.env`，**至少填这两项**：

```ini
VLM_API_KEY=你的密钥
VLM_BASE_URL=https://api-inference.modelscope.cn/v1
VLM_MODEL=Qwen/Qwen3.5-27B
```

**强烈建议再配上语音识别**（免费的，服务器上不占资源）：

```ini
ASR_BASE_URL=https://api.siliconflow.cn/v1
ASR_API_KEY=你的密钥
ASR_MODEL=FunAudioLLM/SenseVoiceSmall
```

不配的话音频维度会缺一整个 —— 有台词/唱歌的视频，歌词字段只能写 `N/A`。
详见「语音转写」一节。

其余留空即可（ffmpeg 会自动用自带的那份）。

> **服务器上不需要任何模型权重。** 语音识别走线上 API，
> ffmpeg 是仓库自带的静态二进制 —— `git clone` + `pip install -e .` 就够了。

#### 3. 前端

```bash
cd frontend
npm ci
npm run build        # 产物在 frontend/dist，后端会自动托管
```

后端启动时会挂载 `frontend/dist`，所以**不用单独跑前端服务**。

#### 4. 起服务

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ **workers 必须是 1。** 任务状态存在进程内存里，
> 多 worker 会导致「提交任务的进程」和「查询进度的进程」不是同一个，
> 进度会查不到。

#### 5. 反向代理（可选）

用 nginx / caddy 套一层 TLS。SSE 进度推送需要关掉缓冲：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;          # 不关的话进度不会实时推送
    proxy_read_timeout 3600s;     # 长视频反推可能跑很久
    client_max_body_size 600m;    # 与 MAX_UPLOAD_MB 对齐
}
```

### 上线前检查

| 项 | 怎么确认 |
|---|---|
| ffmpeg 就位 | `curl localhost:8000/api/health` 里 `ffmpeg` 为 `true` |
| 模型可用 | `POST /api/health/vlm` —— 会真的发一张图实测，不是只 ping 一下 |
| 前端已构建 | 浏览器打开首页有界面，不是 404 |
| 数据目录可写 | `data/` 下能创建文件（上传与抽帧都写这里） |
| 磁盘够用 | 抽帧产物按 `jobs × 帧数 × 30KB` 估；`data/tmp/` 会自动清理 |
| 语音识别（可选） | 反推一段有歌词的视频，看「音频」页签有没有转写文本；没有就看 `note` 里的原因 |

**服务器上不需要下载任何模型权重**：语音识别走线上 API，ffmpeg 是仓库自带的
静态二进制。`pip install -e .` 装完依赖就能跑，没有几百 MB 的额外下载。

> **别把 `.env` 提交进仓库**（已在 `.gitignore` 里）。
> 部署机上如果用了 CI/CD，密钥走环境变量或密钥管理，不要写进代码。

---

