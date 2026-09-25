"""全局配置：pydantic-settings 读取 .env，附带路径与 ffmpeg 解析。"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 目录层级：app/core/config.py -> app/core -> app -> backend -> 项目根
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRAME_DIR = DATA_DIR / "frames"
TMP_DIR = DATA_DIR / "tmp"
RUNTIME_DIR = BACKEND_DIR / "runtime"
# 打包进仓库的第三方二进制（ffmpeg 等）。部署时不用在服务器上另装，
# 也避免服务器上的版本差异影响行为。
VENDOR_DIR = BACKEND_DIR / "vendor"
VENDOR_FFMPEG_DIR = VENDOR_DIR / "ffmpeg"
# 本地语音转写的模型权重（faster-whisper）。**不进 git**（单文件 460MB，
# 会被远端仓库的大文件限制拒掉），但放在这里能跟着打包/rsync/docker 一起走。
VENDOR_WHISPER_DIR = VENDOR_DIR / "whisper"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for _d in (DATA_DIR, UPLOAD_DIR, FRAME_DIR, TMP_DIR, RUNTIME_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"
    debug: bool = False

    # ---- 持久化 ----
    # SQLite 单文件。相对路径按 data/ 解析，也可写绝对路径。
    database_url: str = "sqlite:///app.db"
    # 保留多少天的任务历史（0 = 永久保留）
    history_keep_days: int = 30

    # ---- 多模态模型 ----
    vlm_api_key: str = ""
    vlm_base_url: str = "https://api-inference.modelscope.cn/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    vlm_timeout: int = 180
    vlm_concurrency: int = 3
    # 单次回复的 token 上限。**推理模型必须给大**。
    # 踩过：DeepSeek-V4.1-Flash 思考过程 5 万多字符，6144 会被思考吃光，
    # 正文一个字都写不出来（finish_reason=length、content 为空），
    # 表现为「模型返回空内容」或「未能返回可解析的观察结果」，
    # 看起来像不支持图片，其实只是预算不够。
    # 这只是上限，非推理模型提前结束不会多花钱。
    vlm_max_tokens: int = 16384
    # 关掉推理模型的思考过程（DeepSeek-V4.x / Qwen3 这类）。
    # 实测 DeepSeek-V4.1-Flash：开着思考 88s / 思考 24871 字 / 正文 0 字；
    # 关掉后 9.9s / 思考 0 字 / 正文 490 字 —— 快近 10 倍，而且正文不再被挤掉。
    # 反推要的是「照结构填内容」，不是解数学题，思考过程基本是浪费。
    # 参数名用 OpenAI 生态里最常见的 `enable_thinking`；不支持的模型会报 400，
    # 代码会自动摘掉这个参数重试，所以默认开着是安全的。
    vlm_disable_thinking: bool = True
    # 最终提示词的词数上限。**默认关（0）—— 不做任何压缩。**
    #
    # ⚠️ 这个值以前是 700，而 H3 格式块里还写着「正文 420 词以内」。实测那段限制
    # 在**逼模型合并镜头**：同一段每 0.3 秒一刀的 PV（30 帧）——
    #     正文上限 420 词 → 6 个镜头 / 378 词
    #     上限 900 词      → 15 个镜头 / 528 词
    #     完全去掉         → **21~29 个镜头 / 1237~1549 词**（这才是这份素材真实的样子）
    # 21 个镜头需要约 1200 词，420 词的预算下模型只能把多个镜头并成一个，
    # 用户直接反馈「效果变差了」。**长度就是还原度，压了就是丢细节。**
    #
    # 而且压缩兜底也不可靠：实测模型自然写出 1549 词（29 个镜头），
    # 压缩指令让它「压到 1275 词以内」，它一口气砍到 **492 词**
    # —— 镜头标记留住了，但每个镜头的描述只剩十几个词。
    #
    # 真要收口（视频模型吃不下超长提示词）再设成正数，比如 1200。
    prompt_word_limit: int = 0
    # 把音频片段直接附给模型（OpenAI 的 input_audio 内容块）。
    # 默认关：能收音频的模型很少，很多「多模态」模型只支持图片，
    # 开了但模型不支持会返回 choices:null（代码会明确报出来）。
    # 只在没有 ASR 转写时才附——已经有转写文本就没必要再送音频。
    vlm_audio_input: bool = False

    # ---- 语音转写 ----
    asr_api_key: str = ""
    asr_base_url: str = ""
    asr_model: str = "whisper-1"
    # 本地 whisper 用哪个规格。可选 tiny / base / small / medium。
    #
    # **4 核 4G 的服务器建议 small 或 base**：small 的 int8 权重约 460MB，
    # 推理峰值内存约 1.2GB；base 只有 74MB / 约 300MB，但歌词准确率明显下降
    # （尤其是唱词）。内存吃紧就换 base。
    asr_whisper_model: str = "small"
    # 显式指定模型目录。留空则自动找 `backend/vendor/whisper/<规格>/`，
    # 再找不到才按规格名去 HuggingFace 下载（首次约 460MB）。
    asr_model_path: str = ""
    # whisper 用几个线程。0 = 自动，取 min(4, CPU 核数)。
    #
    # ⚠️ 别占满。管线里镜头检测和抽帧也要 CPU，而它们和音频分析是**并行**跑的
    # （`asyncio.create_task` 两条线），抢起来两边都慢。
    # 4 核机器留 1~2 个核给 ffmpeg 更划算。
    asr_cpu_threads: int = 0
    # 本地 faster-whisper 的模型权重从 HuggingFace 下载，国内直连很慢。
    # 留空则用 https://hf-mirror.com（国内镜像）；海外部署可以显式写
    # https://huggingface.co。也可以自己在环境变量里设 HF_ENDPOINT。
    asr_hf_endpoint: str = ""

    # ---- ffmpeg ----
    ffmpeg_path: str = ""
    ffmpeg_dir: str = ""

    # ---- 抽帧预算 ----
    # 抽帧总预算。这只是**安全网**，不是目标值 —— 真正决定抽多少帧的是
    # 「每镜每秒约一帧」+ max_frames_per_shot。15 秒的视频只会用到 15 帧左右，
    # 预算给大不影响短片的帧数，只保证长视频不会失控。
    # 按 1100 tokens/帧 估算：96 帧 ≈ 10.6 万 tokens。128k 上下文的模型够用，
    # 1M 上下文的可以放心再调大。
    max_total_frames: int = 96
    # 单个镜头的帧数上限。**硬上限**。
    #
    # 默认 24（原来 8）：踩过的坑是一镜到底的视频被这个上限卡死 ——
    # 60 秒的连续镜头只抽 8 帧，等于一帧管 7.5 秒，模型看不出内容，
    # 而预算还剩 88 帧没用。调到 24 后 15 秒的连续镜头给满 15 帧
    # （符合「一秒一帧」的直觉），4 镜头的常规视频行为完全不变。
    # 想要更长的单镜也一秒一帧，就把它调到 60 / 120。
    # 想限制**总**帧数请用 MAX_TOTAL_FRAMES，别指望这个。
    max_frames_per_shot: int = 24
    # 镜头内平均多久取一帧。**0.5 = 每秒两帧**（用户明确要求）。
    # 一秒一帧对快速动作/特效内容偏疏 —— 半秒内发生的变化会被整段漏掉，
    # 而模型看不到就只能编。一帧约 284 tokens，密度翻倍的成本可以接受。
    frame_interval_seconds: float = 0.5
    # 单个镜头超过多少秒时额外补帧（保留给长镜头加权用）
    long_shot_seconds: float = 5.0
    # 把帧拼成网格图（contact sheet）再送给模型。0 = 关闭（每帧单独一张图）。
    #
    # **默认 6（2×3）** —— 用户 09-23 提的方案，我 09-25 大删减时误删，现已恢复并默认开。
    #
    # 为什么拼：
    #   1) 模型的图片 token 成本有上限 —— 网格从 2.7 Mpx 做到 9.2 Mpx（3.4 倍），
    #      token 只从 1156 涨到 1176。**一格里放 6 帧还是 20 帧，成本几乎一样。**
    #   2) **把相邻帧放进同一张图，模型能直接并排比较** —— 判断「这是切镜还是同一个
    #      镜头里的运动」比跨图比较容易得多。这对分镜判断是关键。
    #   3) 同预算下时间覆盖度翻几倍。
    #
    # ⚠️ 代价是小字：6 格时文字可靠，9 格以上开始编
    # （实测 `FUTURE HERO` 被读成 `ULTRA HERO`）。画面描述不受影响。
    # 需要读画面上的小字就压到 4，只要画面就用 6~9。
    frame_sheet_cells: int = 6
    frame_long_edge: int = 896
    frame_jpeg_quality: int = 82

    # ---- 镜头分割 ----
    scene_threshold: float = 0.30
    min_shot_seconds: float = 0.8

    # ---- 音频 ----
    # 转写前做人声频段分离。MV / 现场录音里鼓和贝斯能量很强，whisper 会被
    # 伴奏带偏；带通滤掉 180Hz 以下和 4kHz 以上之后歌词准确率明显更高。
    vocal_isolation: bool = True

    # ---- 长视频分块 ----
    chunk_threshold_seconds: float = 90.0
    chunk_seconds: float = 60.0
    chunk_overlap_seconds: float = 2.0
    max_duration_seconds: float = 1800.0

    # ---- 上传 ----
    max_upload_mb: int = 500

    # ---- 抓取 ----
    cookies_from_browser: str = ""
    cookies_file: str = ""
    # 下载清晰度上限。反推会把帧缩到 FRAME_LONG_EDGE 再送模型，
    # 下载 1080p 纯属浪费带宽和等待时间，720p 完全够用。
    # 值直接透传给 yt-dlp 的 -f，所以可以写完整的 fallback 链。
    ytdlp_format: str = (
        "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
        "bv*[height<=720]+ba/"
        "b[height<=720]/bv*+ba/b"
    )

    # ---------- 派生属性 ----------
    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [x.strip() for x in raw.split(",") if x.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def asr_enabled(self) -> bool:
        return bool(self.asr_base_url and self.asr_api_key)

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_api_key)


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe 自动发现
# ---------------------------------------------------------------------------

def _candidates(names: tuple[str, ...]) -> list[Path]:
    """按优先级列出可能的可执行文件路径。

    优先级：显式配置 > **项目自带**（vendor / runtime）> 系统 PATH > 常见安装位置。

    自带二进制排在 PATH 之前，是为了让部署可复现 —— 服务器上装了什么版本的
    ffmpeg 不该影响这个服务的行为。仓库里带了一份 `backend/vendor/ffmpeg/ffmpeg.exe`。
    """
    exe = ".exe" if os.name == "nt" else ""
    out: list[Path] = []

    s = get_settings()

    # 1. 显式配置
    if s.ffmpeg_path and "ffmpeg" in names[0]:
        out.append(Path(s.ffmpeg_path))
    if s.ffmpeg_dir:
        for n in names:
            out.append(Path(s.ffmpeg_dir) / f"{n}{exe}")

    # 2. 项目自带（打包进仓库，部署时不用在服务器上另装）
    for d in (VENDOR_FFMPEG_DIR, RUNTIME_DIR):
        for n in names:
            out.append(d / f"{n}{exe}")

    # 3. 系统 PATH
    for n in names:
        found = shutil.which(n)
        if found:
            out.append(Path(found))

    # 4. 常见安装位置
    #
    # 注意：这里不再硬编码开发机上别的项目的路径（原来有
    # D:/nexus/.../ffmpeg-static）。那种路径在服务器上必然不存在，
    # 留着只会让「本地能跑、上线就找不到 ffmpeg」这种问题更难查。
    common_dirs = [
        PROJECT_ROOT / "bin",
        Path("C:/ffmpeg/bin"),
        Path("D:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
        Path("D:/Program Files/ffmpeg/bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ]
    for d in common_dirs:
        for n in names:
            out.append(d / f"{n}{exe}")

    return out


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str | None:
    for p in _candidates(("ffmpeg",)):
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def resolve_ffprobe() -> str | None:
    """ffprobe 通常缺失；找不到时上层用 `ffmpeg -i` 解析替代。"""
    for p in _candidates(("ffprobe",)):
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def refresh_settings() -> Settings:
    """清缓存后重新读取 .env（供 /api/settings 热更新使用）。"""
    get_settings.cache_clear()
    resolve_ffmpeg.cache_clear()
    resolve_ffprobe.cache_clear()
    return get_settings()


def env_file_path() -> Path:
    """写回配置的目标文件（`POST /api/settings` 用）。

    为什么单独抽出来：如果测试直接写真实的 `backend/.env`，跑一次测试就会把
    用户的模型配置改成测试里的假值 —— 踩过：测试把 VLM_MODEL 写成 `Vendor/X`，
    测试全绿但服务起不来了。所以允许用 ENV_FILE 环境变量把写入目标指到别处。

    刻意不走 pydantic-settings：它是用来定位配置文件本身的，从配置文件里读会自相矛盾。
    """
    custom = os.environ.get("ENV_FILE", "").strip()
    return Path(custom) if custom else (BACKEND_DIR / ".env")


def ffmpeg_required() -> str:
    path = resolve_ffmpeg()
    if not path:
        raise RuntimeError(
            "未找到 ffmpeg。请任选其一：\n"
            "  1) 把 ffmpeg 加入系统 PATH；\n"
            "  2) 在 backend/.env 里设置 FFMPEG_PATH=D:/path/to/ffmpeg.exe；\n"
            "  3) 把 ffmpeg.exe 放到 backend/runtime/ 目录下。"
        )
    return path


def python_exe() -> str:
    return sys.executable
