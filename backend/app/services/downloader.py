"""B站 / 抖音视频抓取。

策略分层（越靠前越稳）：

  B站
    1. yt-dlp —— 官方适配完善，稳定。支持 cookie 处理大会员内容。
    2. 降级：提示用户手动下载后上传。

  抖音
    1. yt-dlp —— 支持不稳定（签名常改），能成功就用。
    2. 原生解析：分享短链 → 302 拿到 item_id → 请求分享页 → 解析
       `_ROUTER_DATA` 里的 play_addr（去掉 playwm 拿无水印地址）。
    3. 降级：提示用户手动下载后上传。

合规：只处理公开可访问的内容，下载件仅存本地用于分析，不做二次分发。
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..core.config import TMP_DIR, get_settings, resolve_ffmpeg
from ..core.safe_delete import safe_delete
from ..schemas import ProbeResult

log = logging.getLogger("downloader")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

BILIBILI_HOSTS = ("bilibili.com", "b23.tv", "biligame.com")
DOUYIN_HOSTS = ("douyin.com", "iesdouyin.com", "v.douyin.com")


# ---------------------------------------------------------------------------
# 平台识别
# ---------------------------------------------------------------------------

def detect_platform(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return "unknown"
    if any(h in host for h in BILIBILI_HOSTS):
        return "bilibili"
    if any(h in host for h in DOUYIN_HOSTS):
        return "douyin"
    if host:
        return "url"
    return "unknown"


def extract_url(text: str) -> str | None:
    """从分享文案里抠出链接（抖音分享文本里混着中文和表情）。"""
    m = re.search(r"https?://[^\s\u4e00-\u9fff，。！、）)】\]]+", text or "")
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# yt-dlp 通道
# ---------------------------------------------------------------------------

def _yt_dlp_options() -> dict:
    s = get_settings()
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 30,
        "http_headers": {"User-Agent": UA},
    }
    # 关键：`bv*+ba` 这类格式合并音视频必须调用 ffmpeg，而本机 ffmpeg 常常不在 PATH 里
    # （只有 ffmpeg-static 那种单文件）。不显式告诉 yt-dlp 位置，合并步骤必然失败，
    # 表现为「解析成功但下载失败」，很容易误判成平台反爬。
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        opts["ffmpeg_location"] = ffmpeg
    if s.cookies_file:
        opts["cookiefile"] = s.cookies_file
    elif s.cookies_from_browser:
        opts["cookiesfrombrowser"] = (s.cookies_from_browser.strip().lower(),)
    return opts


def yt_dlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def probe_with_ytdlp(url: str) -> ProbeResult | None:
    if not yt_dlp_available():
        return None
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(_yt_dlp_options()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        log.info("yt-dlp 探测失败 %s: %s", url, exc)
        return None

    if not info:
        return None
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            return None
        info = entries[0]

    return ProbeResult(
        platform=detect_platform(url),
        title=str(info.get("title") or "")[:200],
        duration=float(info.get("duration") or 0.0),
        thumbnail=str(info.get("thumbnail") or ""),
        uploader=str(info.get("uploader") or info.get("channel") or ""),
        direct_url=str(info.get("url") or ""),
        note="yt-dlp",
    )


def download_with_ytdlp(url: str, out_dir: Path) -> tuple[Path | None, str]:
    """下载并返回 `(文件路径, 失败原因)`。

    失败原因要往上传，不能只写日志——否则界面只能给「会员内容/需要登录」这种
    通用猜测，把「ffmpeg 找不到导致合并失败」误报成平台反爬，排查方向完全跑偏。
    """
    if not yt_dlp_available():
        return None, "未安装 yt-dlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _yt_dlp_options()
    opts.update({
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        # 走配置的清晰度上限：反推会把帧缩到长边 896 再送模型，下 1080p 是纯浪费
        "format": get_settings().ytdlp_format,
        "merge_output_format": "mp4",
        "overwrites": True,
        # ⚠️ 必须保留合并前的原始流。
        #
        # 默认行为是合并完成后 os.remove 掉分离的音视频流，而某些运行环境在
        # os.remove 上装了删除保护（累计删除超过阈值就抛 SystemExit）——
        # 一删就把下载线程带走，任务报「SystemExit: 1」，
        # 而实际上视频早就下好、ffmpeg 也合并完了。纯属冤死。
        # 置 True 后 yt-dlp 不删，多占几 MB 临时空间，换下载稳定。
        "keepvideo": True,
    })
    # 注意这里捕获的是 BaseException 而不是 Exception：
    # 上面那个删除保护抛的 SystemExit 继承自 BaseException，
    # `except Exception` 拦不住，会让异常穿到线程池里。
    exc_note = ""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except BaseException as exc:  # noqa: BLE001
        # 下载/合并**可能已经成功**，死在后续清理上。别急着报失败，
        # 先去磁盘上找找 —— 用文件系统的事实判断，而不是相信异常。
        exc_note = " ".join(f"{type(exc).__name__}: {exc}".split())[:200]
        log.warning("yt-dlp 抛出异常（将回查磁盘确认是否已产出）%s: %s", url, exc_note)
        path = out_dir / ""

    found = _find_downloaded_video(out_dir, path)
    if found is not None:
        if exc_note:
            log.info("yt-dlp 虽抛异常但文件已产出，继续处理：%s（原因：%s）", found.name, exc_note)
        return found, ""
    if exc_note:
        return None, exc_note
    return None, f"yt-dlp 下载完成但未产出可识别的视频文件（期望 {path.name}）"


_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".flv", ".mov")


def _find_downloaded_video(out_dir: Path, expected: Path) -> Path | None:
    """在输出目录里找下载好的视频。合并后扩展名可能变（webm/flv 源合成 mp4）。"""
    if expected.name and expected.is_file() and expected.stat().st_size > 0:
        return expected
    stem = expected.stem if expected.name else ""
    if stem:
        for cand in out_dir.glob(f"{stem}.*"):
            if cand.suffix.lower() in _VIDEO_SUFFIXES and cand.stat().st_size > 0:
                return cand
    # 兜底：按体积挑最大的视频文件（keepvideo 会留下分离的音视频流，
    # 音频流通常小得多，最大的那个就是成片）
    best: Path | None = None
    for cand in out_dir.iterdir():
        if cand.suffix.lower() not in _VIDEO_SUFFIXES or cand.stat().st_size <= 0:
            continue
        if best is None or cand.stat().st_size > best.stat().st_size:
            best = cand
    return best


# ---------------------------------------------------------------------------
# 抖音原生通道
# ---------------------------------------------------------------------------

_AWEME_ID_PATTERNS = (
    re.compile(r"/video/(\d{6,})"),
    re.compile(r"modal_id=(\d{6,})"),
    re.compile(r"/note/(\d{6,})"),
    re.compile(r"/share/video/(\d{6,})"),
)


def _resolve_short_link(url: str) -> str:
    """跟随 302 拿到最终地址（v.douyin.com/xxx 短链必须走这一步）。"""
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers={"User-Agent": UA}) as c:
            resp = c.get(url)
            return str(resp.url)
    except Exception as exc:  # noqa: BLE001
        log.info("短链解析失败: %s", exc)
        return url


def _extract_aweme_id(url: str) -> str | None:
    for pat in _AWEME_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def _find_play_addr(node, depth: int = 0):
    """在 _ROUTER_DATA 里递归找 play_addr.url_list。"""
    if depth > 8:
        return None
    if isinstance(node, dict):
        pa = node.get("play_addr")
        if isinstance(pa, dict):
            urls = pa.get("url_list") or []
            if urls:
                return urls[0]
        for v in node.values():
            got = _find_play_addr(v, depth + 1)
            if got:
                return got
    elif isinstance(node, list):
        for v in node:
            got = _find_play_addr(v, depth + 1)
            if got:
                return got
    return None


def resolve_douyin(url: str) -> ProbeResult | None:
    """抖音分享页解析：拿 item_id → 请求分享页 → 从 _ROUTER_DATA 取无水印地址。"""
    final = _resolve_short_link(url)
    aweme_id = _extract_aweme_id(final) or _extract_aweme_id(url)
    if not aweme_id:
        log.info("无法从链接提取抖音 aweme_id: %s", url)
        return None

    share_url = f"https://www.iesdouyin.com/share/video/{aweme_id}/"
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True, headers={"User-Agent": UA}) as c:
            resp = c.get(share_url)
            html = resp.text
    except Exception as exc:  # noqa: BLE001
        log.info("抖音分享页请求失败: %s", exc)
        return None

    m = re.search(r"_ROUTER_DATA\s*=\s*(\{.*?\})\s*;?\s*</script>", html, re.S)
    if not m:
        log.info("抖音分享页未找到 _ROUTER_DATA（可能被风控或页面改版）")
        return None

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        log.info("抖音 _ROUTER_DATA 解析失败")
        return None

    loader = data.get("loaderData") or {}
    item = None
    for value in loader.values():
        if isinstance(value, dict) and value.get("video"):
            item = value
            break
    if item is None:
        item = loader

    video = (item or {}).get("video") or {}
    play = _find_play_addr(video) or _find_play_addr(item)
    if not play:
        return None

    # playwm 是带水印版本，换成 play 拿无水印
    clean = play.replace("playwm", "play").replace("&watermark=1", "")

    duration = 0.0
    with contextlib.suppress(TypeError, ValueError):
        duration = float(video.get("duration") or 0) / 1000.0

    desc = str((item or {}).get("desc") or "")
    author = ((item or {}).get("author") or {}).get("nickname") or ""

    cover = ""
    cover_obj = video.get("cover") or {}
    if isinstance(cover_obj, dict):
        cl = cover_obj.get("url_list") or []
        if cl:
            cover = cl[0]

    return ProbeResult(
        platform="douyin",
        title=desc[:200] or f"douyin_{aweme_id}",
        duration=duration,
        thumbnail=cover,
        uploader=str(author),
        direct_url=clean,
        note="douyin-native",
    )


def download_direct(url: str, out_dir: Path, name: str = "douyin") -> Path | None:
    """直链下载（带 Referer，抖音 CDN 会校验）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{name}_{int(time.time())}.mp4"
    headers = {"User-Agent": UA, "Referer": "https://www.douyin.com/"}
    try:
        with (
            httpx.Client(timeout=300.0, follow_redirects=True, headers=headers) as c,
            c.stream("GET", url) as resp,
        ):
            if resp.status_code >= 400:
                log.warning("直链下载失败 HTTP %s", resp.status_code)
                return None
            with dst.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 16):
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        log.warning("直链下载异常: %s", exc)
        return None

    if dst.is_file() and dst.stat().st_size > 10240:
        return dst
    safe_delete(dst)
    return None


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------

def probe(url: str) -> ProbeResult:
    """只探测信息，不下载。用于前端「先确认再分析」。"""
    url = extract_url(url) or url
    platform = detect_platform(url)

    if platform in ("bilibili", "douyin", "url"):
        got = probe_with_ytdlp(url)
        if got:
            got.platform = platform
            return got

    if platform == "douyin":
        got = resolve_douyin(url)
        if got:
            return got

    return ProbeResult(
        platform=platform,
        note="无法解析该链接，请手动下载视频后上传",
    )


def download(url: str, job_id: str) -> tuple[Path | None, ProbeResult, str]:
    """下载视频。返回 (文件路径, 探测信息, 错误说明)。

    依次尝试 yt-dlp → 平台原生通道。全部失败时返回 None 并给出可操作提示。
    """
    url = extract_url(url) or url
    platform = detect_platform(url)
    out_dir = TMP_DIR / f"fetch-{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = ProbeResult(platform=platform)
    ytdlp_reason = ""

    # --- 通道 1：yt-dlp ---
    if yt_dlp_available():
        got = probe_with_ytdlp(url)
        if got:
            got.platform = platform
            info = got
        path, ytdlp_reason = download_with_ytdlp(url, out_dir)
        if path:
            if not info.duration:
                info = probe_with_ytdlp(url) or info
            return path, info, ""
        log.info("yt-dlp 通道未成功（%s），尝试平台原生通道", ytdlp_reason)
    else:
        ytdlp_reason = "未安装 yt-dlp"
        log.info("未安装 yt-dlp，跳过该通道")

    # --- 通道 2：抖音原生 ---
    if platform == "douyin":
        got = resolve_douyin(url)
        if got:
            info = got
            path = download_direct(got.direct_url, out_dir, name=f"douyin_{job_id[:8]}")
            if path:
                return path, info, ""

    # --- 通道 3：失败，给可操作提示 ---
    if platform == "bilibili":
        hint = (
            "B站视频解析失败。可能原因：视频为会员/付费内容、已被删除、或需要登录。\n"
            "可在 backend/.env 里设置 COOKIES_FROM_BROWSER=chrome（从本机浏览器读取 cookie）后重试，"
            "或手动下载视频后上传。"
        )
    elif platform == "douyin":
        hint = (
            "抖音视频解析失败。抖音的签名校验变动频繁，自动抓取不是总能成功。\n"
            "请手动下载视频后上传——上传通道的分析效果完全一致。"
        )
    else:
        hint = "该链接无法解析。请确认是公开可访问的视频页面链接，或手动下载后上传。"

    if ytdlp_reason:
        hint += f"\n\nyt-dlp 原始报错：{ytdlp_reason}"

    return None, info, hint
