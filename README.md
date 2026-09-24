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
- [部署到服务器](#部署到服务器)

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
| `fps=10`（1 秒 10 帧） | 600 帧 | ≈ **17 万** |
| `fps=2` | 120 帧 | ≈ 3.4 万 |
| **本项目：按镜头分配预算** | **48 帧** | ≈ **1.4 万** |

> 一帧的 token 成本是**实测值**：长边 896px 时约 **284 tokens**
> （448/672px 都是 198，1344px 是 602）。公式 ≈ `max(200, 像素数 / 1600)`。
> 估算见 `selection.tokens_per_frame()`。

即便如此，`fps=10` 在 1M 上下文下也吃得下 —— 但它是**最差的方案**：
均匀采样既不是关键帧也不是镜头切换帧，会把预算浪费在重复画面上，
还漏掉切镜。**帧数不是越多越好，放对位置才是。**

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

## 四个实测质量缺陷与修补（v2 管线）

拿一段 3 人偶像演唱会 MV 实测，输出 38 个镜头，暴露了四个缺陷。四个都不是
「提示词没写好」，而是**管线里缺东西**——模型没有依据，只能编或者保守。

### 一、38 个镜头全部标 static

**根因**：单帧图像里没有运动信息。模型拿不准运镜时一律选最保守的答案。

**修法**：加 `motion.py` —— ffmpeg 抽 160×90 灰度小图，纯 Python 算全局位移。

- 四角分区取中位数，而不是整幅一起估。整幅估计有个致命弱点：**主体在画面里
  移动会被当成相机运动**。实测一段相机固定在三脚架上的素材，整幅估计报出
  9.5 像素的假位移，分四角后降到 0.2 以内。
- 迭代 warp 而不是图像金字塔。金字塔在**周期性纹理**上会混叠到错误的峰
  （实测一张正弦条纹图里 8 像素的位移被解成 28.9）；迭代 warp 是连续逼近，
  不会跳到别的周期上。
- 判「静止」只看**位移**，不看画面变化量：主体在跳舞而相机架在三脚架上，
  画面变化很大但位移接近 0，运镜仍然是 static。

实测判定（合成素材，已知真值）：

| 素材 | 累计位移 | 判定 |
|---|---|---|
| 固定取景（画面里有主体在动） | 0.0 | `static` ✅ |
| 固定取景 + 噪声 | 0.2 | `static` ✅ |
| 相机右摇 | 100.8 | `pan-right` ✅ |

**只有累计位移低于阈值才允许写 `static`** —— 这条规则写进了 Pass1 提示词。

### 二、特写镜头里列出了画面外的属性

`[Shot 1] Extreme close-up` 的描述里出现了 `white socks, black ankle boots`。
特写根本看不到袜子。

**根因**：每一帧都在重新识别角色，然后把整套外观标签一次性倒出来。

**修法**（两层）：

1. Pass1 提示词加硬约束：**只描述本帧可见内容**，外观由主体登记表统一负责，
   特写不许提袜子/鞋/裤。
2. 代码兜底 `strip_offscreen_attributes()`：`close-up` / `extreme close-up` /
   `medium close-up` 的描述里出现下肢词就按**短语**删掉（先按逗号切，再按 `and`
   切，所以 "a black jacket and white socks" 只丢后半截）。中景不清理——
   中景可能拍到腰以下。

### 三、23 个镜头交替重复

`[Shot 10]`–`[Shot 32]` 内容只有 `Three performers dance.` 和
`Three performers continue.` 两句。

**根因**：抽帧按固定间隔，同一个镜头被抽成十几帧、每帧独立成一个「镜头」。

**修法**：抽帧本来就已经挂在镜头切分上（见上一章），真正缺的是**合并兜底**。
提示词里写明了「条目数 = 机位数量，不是帧数」并给了自检，实测照样输出 23 条。
所以加 `merge_adjacent_shots()`：

判据是「景别同类 + 运镜同类 + 动作是**复读内容**」。用复读而不是相似度当主判据，
是因为 `dance` 和 `continue` 互相的相似度并不高，但各自在整段里重复了十几次——
**一条描述在几分钟素材里以完全相同的话出现三次以上，它就不可能是对独立镜头的
描述**。真实的不同镜头各有各的说法，不会误合并。

### 四、音频输出是频谱术语

实际输出：

```
Sound present, content unanalysed
measured energy mostly voice band with notable low-frequency component
inferred from energy distribution alone
```

**根因**：只做了频谱能量分析，没做语义分析；而且**我们自己的 system prompt 里
就写着 `inferred from energy distribution`**，模型照抄得很自然。

**修法**：

1. **加音乐画像** `audio_features.py`：ffmpeg 抽 PCM → 纯 Python 算能量包络 →
   自相关估 BPM + 音头密度 + 瞬态簇 + 频段平衡 → 生成一句人类可读的描述：

   > The soundtrack reads as a fast pulse of roughly 128 BPM with strong,
   > clearly accented beats; with a strong low end, the kind a kick drum and
   > bass produce, bright high-frequency content.

2. **加人声分离再转写**：带通滤掉 180Hz 以下和 4kHz 以上（首选 Demucs，装了
   才用，因为它要 torch 2GB+）。MV 里鼓和贝斯能量很强，whisper 会被伴奏带偏。

3. **输出侧强制清理** `sanitize_audio_jargon()`。⚠️ **这一条必须用代码**：
   试过在提示词里列出禁用词，结果模型把这份清单本身抄进了输出——
   「不要写 content unanalysed」反而让它记住了这个词。所以清单只留在服务端，
   输出时按句删除命中的内容。

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

#### 提示词长度上限

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PROMPT_WORD_LIMIT` | `700` | 最终提示词的**整篇**词数上限 |

限的是**整篇**，不是正文 —— 六段式里正文只占 43%，`subject_definitions` 和
`retention_analysis` 加起来接近一半。不限长时实测会写到 **1795 词 / 11314 字符**，
视频生成模型吃不下。

两道措施：

1. **模板逐段给预算**（定义最多 6 条 × 15 词、正文 420 词、保留分析每条 15 词…），
   并规定超预算时先砍定义和分析、**绝不砍正文** → 降到 774 词
2. **仍超限时自动压缩一次**：把超长文本交给模型改短，再校验结构
   （六段齐全、主体标签不丢、镜头标记不丢、`N/A` 不丢），残缺就保留原稿
   → 降到 **556 词 / 3903 字符**

> ⚠️ 压缩这一步要**开思考**，和全流程相反。实测关思考时模型原样返回或只砍 74 词，
> 开思考能砍 252 词。编辑任务需要先想清楚哪些能砍；生成任务才需要关思考。

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
| `MAX_FRAMES_PER_SHOT` | `24` | 每个镜头的帧数上限（**硬上限**）。见下方「一镜到底」 |
| `FRAME_INTERVAL_SECONDS` | `1.0` | 镜头内平均多久取一帧 |
| `MAX_TOTAL_FRAMES` | `96` | 单次请求总帧数上限（**安全网**，只在长视频里起作用） |
| `LONG_SHOT_SECONDS` | `5.0` | 超过此时长的镜头优先补帧 |
| `FRAME_LONG_EDGE` | `896` | 送入模型的图片长边像素 |
| `FRAME_JPEG_QUALITY` | `82` | JPEG 质量 |

#### 帧数由镜头时长决定，不是由总预算决定

这是最容易搞错的地方。常见做法是「给一个总帧数上限，然后在镜头间分配」——
但真正该问的是「这个镜头需要几帧才能被看懂」。

```
每个镜头的帧数 = clamp(镜头时长 / FRAME_INTERVAL_SECONDS, 1, MAX_FRAMES_PER_SHOT)
```

实测对比（同一个 15 秒视频）：

| | 之前 | 现在 |
|---|---|---|
| 15s / 4 镜 | 12 帧（每镜固定 3） | **15 帧**（5.4s 镜拿 5 帧、2.1s 镜拿 3 帧） |
| 15s / 1 镜（一镜到底） | 3 帧 | **15 帧**（一秒一帧） |
| 212s / 46 镜 | 48 帧，**14 个镜头一帧都没有** | **96 帧，全覆盖** |

总预算降级成安全网：15 秒的短片只会用到 15 帧左右，把 `MAX_TOTAL_FRAMES`
调大不影响短片，只保证长视频不失控。

#### 一镜到底的视频：注意单镜上限

「一镜到底」时所有内容都在一个镜头里，帧数完全由 `MAX_FRAMES_PER_SHOT` 决定：

| 视频 | 旧默认（上限 8） | 新默认（上限 24） |
|---|---|---|
| 15s / 1 镜 | 8 帧（一帧管 1.9s） | **15 帧**（一秒一帧） |
| 60s / 1 镜 | 8 帧（一帧管 7.5s） | 24 帧 |

原来默认 8 会让连续镜头严重欠采样，模型看不出中间发生了什么，
而预算还剩一大半没用。**想让更长的单镜也一秒一帧，就把上限调到对应秒数**
（比如 60 秒的单镜设 `MAX_FRAMES_PER_SHOT=60`）。

> 试过「镜头少时自动抬高上限」，但那会让这个配置在常见场景下失效
> （4 镜时上限被抬到 24，设 3 还是 8 效果一样）。**配置项失去意义比不够灵活更糟**，
> 所以保持硬上限、只改默认值。要限制总帧数请用 `MAX_TOTAL_FRAMES`。

**按上下文选值**：一帧约 284 tokens（长边 896px，实测）。

| 模型上下文 | 建议 `MAX_TOTAL_FRAMES` | 实际占用 |
|---|---|---|
| 128k | `96`（默认） | ≈ 2.7 万 tokens |
| 256k | `300` | ≈ 8.5 万 |
| 1M | `1000` | ≈ 28 万 |

一帧才 284 tokens，所以**帧数几乎不可能是瓶颈** —— 真正的限制是延迟
（图越多推理越慢）和收益递减（同一个镜头相邻帧长得差不多）。
按「每秒约一帧」抽，15 秒视频 15 帧就够了，再多是浪费。

#### 拼图模式（contact sheet）

把帧拼成网格图再送给模型，默认**关闭**（`FRAME_SHEET_CELLS=0`）。
开启后可以给模型几倍的时间覆盖度，代价是小字会糊。

**为什么成立：模型的图片 token 有上限，成本是平坦的。**

| 网格 | 像素 | token |
|---|---|---|
| 1804×764（12 格） | 1.38 Mpx | 854 |
| 3602×1524（12 格） | 5.49 Mpx | 978 |
| 5406×2034（24 格） | 11.0 Mpx | **1063** |

**像素翻 8 倍，token 只涨 24%。** 一格里放 6 帧还是 20 帧，成本几乎一样。

实测对比（13.3s 视频，Pass1 全流程）：

| 输入 | 耗时 | 画面要素 | 镜头数 |
|---|---|---|---|
| 1fps / 12 张单帧 | 46s | 7/7 | 12（每帧一个，过碎） |
| 2fps / 3 张 3×3 | **26s** | 7/7 | **5**（更接近实际的 4） |

##### ⚠️ 代价一：小字会崩

扫描不同格数的崩点：

| 格数 | token | T恤文字 |
|---|---|---|
| **6** | 1156 | ✅ `FUTURE HERO` |
| 9 / 12 / 16 / 20 | 1128~1176 | ❌ 全部读成 `ULTRA HERO` |

6 格可靠，**9 格以上开始编**（偶尔又能读对 —— 在临界点上随机翻）。
画面描述不受影响（外套 5 次全对）。

这很危险：Pass1 有 `on_screen_text` 字段，**编出来的印字会原样进提示词**。

##### ⚠️ 代价二：时间密度换来的镜头更多，Pass2 会变慢

端到端实测：

| | 单帧模式 | 拼图模式 |
|---|---|---|
| 总耗时 | **88s** | 156s |
| 镜头数 | 13 | **24** |
| 提示词 | 556 词 | 1180 词 → 压缩到 563 |

Pass1 本身更快（26s vs 46s），但多出来的镜头让 Pass2 输出变长，
还触发了提示词压缩。**不是免费的**。

##### 怎么选

| 你的素材 | `FRAME_SHEET_CELLS` |
|---|---|
| 有重要字幕/标题，文字不能错 | `0`（单帧） |
| 只关心画面，想要更密的时间覆盖 | `9` ~ `12` |
| 要文字但想省点 token | `4` ~ `6` |

配套 `FRAME_SAMPLE_FPS`（默认 `2.0`）：拼图模式下的抽帧频率。
开拼图时自动从 1fps 提到 2fps —— 单帧模式受 token 约束只能 1fps。

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

### 语音转写

**推荐接线上 API**，免费额度就够用，不用下模型、不占内存、部署轻。

| 服务 | 免费额度 | 配置 |
|---|---|---|
| **硅基流动**（国内可直连） | `FunAudioLLM/SenseVoiceSmall`、`TeleAI/TeleSpeechASR` 标为免费 | `ASR_BASE_URL=https://api.siliconflow.cn/v1`<br>`ASR_MODEL=FunAudioLLM/SenseVoiceSmall` |
| **Groq** | `whisper-large-v3-turbo`，Free Plan 2000 次/日 | `ASR_BASE_URL=https://api.groq.com/openai/v1`<br>`ASR_MODEL=whisper-large-v3-turbo` |

⚠️ 硅基流动的域名是 **`.cn`**，旧的英文文档里写的 `.com` 会把正确的 Key 打成
`401 Token is invalid`。注册需要先实名。

只要 `ASR_BASE_URL` + `ASR_API_KEY` 都填了就走线上，**不需要装任何本地包**。

| 其他变量 | 说明 |
|---|---|
| `ASR_MODEL_PATH` | 本地模型目录（仅在用本地兜底时相关） |
| `ASR_WHISPER_MODEL` | 本地模型规格：`tiny` / `base` / `small` / `medium` |
| `ASR_CPU_THREADS` | 本地 whisper 线程数，0 = 自动 |
| `VOCAL_ISOLATION` | 转写前先做人声频段分离（默认开） |

#### 不同服务商的 `response_format` 支持不一样

代码先要 `verbose_json`（带分句时间戳，歌词能对齐到镜头），
**拿不到就自动降级成 `json`**（只有整段文本，交给 Pass1 按语义对齐），
并把实际用的格式写进备注 —— 否则「歌词对不上镜头」会变成一个查不到原因的现象。

实测：OpenAI / Groq 支持 `verbose_json`；硅基流动的 SenseVoice 系列只保证 `json`。

#### 为什么转写前要滤掉低频和高频

MV / 现场录音里鼓和贝斯能量很强，whisper 会被伴奏带偏，把歌词听成别的东西。
`isolate_vocals()` 先做带通（滤掉 180Hz 以下和 4kHz 以上）再送转写，
人声清晰度明显提升，而且零额外依赖（纯 ffmpeg 滤镜）。

装了 Demucs 会自动优先用它（真正的音源分离，效果更好），但它要 torch（2GB+），
所以只在已经装好时才用。

#### 模型不指定语言

让它自己检测 —— 指定成 `zh` 会把日语歌词硬翻成中文，而我们要的是**保留原语言**。
实测日语素材自动检出 `ja`，歌词原样保留。

#### 本地模型（可选兜底）

不配 `ASR_*` 又想离线跑的话，装本地 faster-whisper：

```bash
pip install -e ".[local-asr]"
python scripts/fetch-whisper-model.py     # 下载到 backend/vendor/whisper/
```

优先级是 **线上 API → 本地模型 → 跳过**。

模型权重单个 460MB，**不进 git**（会被远端仓库的大文件限制拒掉），但放在
`backend/vendor/whisper/` 里能跟着打包走：

| 部署方式 | 做法 |
|---|---|
| rsync / tar / docker COPY | 把 `backend/vendor/` 一起带上，服务器不用联网 |
| 纯 git | 服务器上跑一次 `python scripts/fetch-whisper-model.py` |

规格按服务器内存选（int8 推理峰值，含模型常驻）：

| 规格 | 磁盘 | 内存 | 适合 |
|---|---|---|---|
| `tiny` | 75 MB | ~250 MB | 只求「有没有人在唱」 |
| `base` | 145 MB | ~350 MB | **4 核 4G 求稳** |
| `small` | 480 MB | ~1.2 GB | 准确率与内存的平衡点 |
| `medium` | 1.5 GB | ~3 GB | 4G 机器不要碰 |

4 核 4G 跑 `small`：15 秒音频转写约 10~20 秒，加载一次性 3~8 秒（之后常驻）。
**但要串行**：`ASR_CPU_THREADS=2` 留核给 ffmpeg，别同时接多个反推任务。

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

**关于 `content_hint`**：用户自己写的画面说明，**同时喂给观察阶段和成文阶段**。

静态帧判断不出三件事，而它们直接影响产出质量：

- **这段是一镜到底还是多镜头切换** —— 静态帧里看不出来
- **主体是谁**（角色 / 作品 / 产品 / 地点）—— 模型不认识冷门 IP
- **动作的前因后果** —— 只看到中间一段会误判

补一句话比让模型瞎猜强得多。措辞上做了两层约束防止模型拿它当观察结果照抄：
Pass1 声明「画面与说明冲突时**相信画面**」，Pass2 声明「观察结果才是权威」。

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

提示词是这个项目的核心资产 —— 改一个字都影响产出质量。但 Pass1 的系统提示词
有 9000 字符，在 Python 字符串里翻没法读。所以有个导出命令：

```bash
cd backend
python -m app.dump_prompts prompts-dump
```

导出 8 个文件：

| 文件 | 内容 |
|---|---|
| `pass1_system.txt` | 观察阶段的系统提示词（含镜头结构判断规则） |
| `pass1_user.txt` | 用户消息样例 —— 能看到帧时间戳列表长什么样 |
| `pass2_system_h3.txt` | T2VA 成文提示词 |
| `pass2_system_h3-ref.txt` | Ref2VA 成文提示词 |
| `pass2_system_seedance.txt` | Seedance 成文提示词 |
| `pass2_system_generic.txt` | 通用格式 |
| `pass2_user.txt` | 观察结果是怎么整理给成文阶段的 |
| `payload.json` | 实际发出去的 HTTP body 骨架 |

导出目录默认 `prompts-dump/`，可以传参数改。已在 `.gitignore` 里 ——
它是生成物，随时能重新导出。

### 一次请求实际提交了什么

**Pass1（带图）**

```jsonc
{
  "model": "...",
  "messages": [
    { "role": "system", "content": "<9007 字符的系统提示词>" },
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

**Pass2（无图）** —— 只提交系统提示词和观察结果文本，**不重新看图**。
这样成文阶段不会因为再看到画面而改写观察结论。

一次请求的体量：2 帧约 194 KB，**其中 100% 是 base64 图片**
（每帧约 75KB → base64 后约 100KB → 约 284 tokens）。

## 查看实际用的提示词

提示词是这个项目的核心资产 —— 改一个字都影响产出质量。但 Pass1 的系统提示词
有 9000 字符，在 Python 字符串里翻没法读。所以有个导出命令：

```bash
cd backend
python -m app.dump_prompts prompts-dump
```

导出 8 个文件：

| 文件 | 内容 |
|---|---|
| `pass1_system.txt` | 观察阶段的系统提示词（含镜头结构判断规则） |
| `pass1_user.txt` | 用户消息样例 —— 能看到帧时间戳列表长什么样 |
| `pass2_system_h3.txt` | T2VA 成文提示词 |
| `pass2_system_h3-ref.txt` | Ref2VA 成文提示词 |
| `pass2_system_seedance.txt` | Seedance 成文提示词 |
| `pass2_system_generic.txt` | 通用格式 |
| `pass2_user.txt` | 观察结果是怎么整理给成文阶段的 |
| `payload.json` | 实际发出去的 HTTP body 骨架 |

导出目录默认 `prompts-dump/`，可以传参数改。已在 `.gitignore` 里 ——
它是生成物，随时能重新导出。

### 一次请求实际提交了什么

**Pass1（带图）**

```jsonc
{
  "model": "...",
  "messages": [
    { "role": "system", "content": "<9007 字符的系统提示词>" },
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

**Pass2（无图）** —— 只提交系统提示词和观察结果文本，**不重新看图**。
这样成文阶段不会因为再看到画面而改写观察结论。

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
| `test_selection.py` | 29 | 预算不超、每镜保底、**按镜头时长定帧数**、**每镜全覆盖**、`max_per_shot` 大于 3 生效、帧间隔调密度 |
| `test_ffmpeg.py` | 45 | 真实视频跑探测 / 场景检测 / 抽帧 / 缩放 / 音频 / 切分 / 频段能量 / 音频片段抽取 / 自适应镜头检测 / 拼图与布局说明 / **片段截取（帧精确）** |
| `test_templates.py` | 65 | Pass1 JSON 宽容解析、多块镜号重排、主体登记表合并、两种模式的模板硬约束、占位符替换、音频编造禁令、频谱措辞边界、**逐段长度预算**、**压缩指令与结构校验** |
| `test_downloader.py` | 24 | 中文分享文案取链接、平台识别、aweme_id、yt-dlp 选项（**含 `ffmpeg_location` 回归**）、清晰度封顶、失败原因上传 |
| `test_ffmpeg_vendor.py` | 7 | ffmpeg 打包位置、跨平台查找优先级、**不硬编码开发机路径** |
| `test_vlm.py` | 36 | 请求体构造（data URI / Anthropic 块 / **`input_audio` 音频块**）、响应解析（含 `choices: null`）、**空响应原因诊断**、音频格式白名单与体积上限、**关思考与按次覆盖**、base_url 带不带 `/v1` 都能用 |
| `test_api.py` | 36 | 接口契约、模式分组、上传校验、Range 流、SQLite 往返、缺列自愈、提示词库、**模型热切换**、**测试不污染真实 .env** |
| `test_pipeline_e2e.py` | 19 | **完整管线**：帧数对齐、预算生效、四种格式、主体登记表贯通到 Pass2、模式不串味、进度事件、分块、无 key 报错、**片段截取只推选中区间** |

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

单机自用定位，没有鉴权。**对外暴露前先加认证**（见「已知限制」）。

两种方式。**推荐 Docker** —— 它把「装 Python 依赖、装 ffmpeg、构建前端」三步
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
| 测试 | — | **307 passed** |

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
