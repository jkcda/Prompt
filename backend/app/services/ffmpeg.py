"""ffmpeg 工具层。

设计要点：
1. **不依赖 ffprobe**。系统里常常只有 ffmpeg.exe，所以媒体信息用 `ffmpeg -i` 的
   stderr 解析。若恰好有 ffprobe 则优先用它（更精确）。
2. **逻辑分块优先于物理切分**。长视频不需要真的切出文件——只要按绝对时间
   从源文件抽帧即可，省掉一次重编码，也避免把镜头劈断。`segment_video()`
   仍然保留，用于需要给用户提供分片预览或喂给原生视频模型的场景。
3. 所有 ffmpeg 调用统一走 `run()`，Windows 下屏蔽控制台窗口。
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core.config import (
    FRAME_DIR,
    TMP_DIR,
    ffmpeg_required,
    get_settings,
    resolve_ffmpeg,
    resolve_ffprobe,
)
from ..schemas import MediaInfo

log = logging.getLogger("ffmpeg")

_NO_WINDOW = 0
if os.name == "nt":  # pragma: no cover - 平台相关
    _NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


# ---------------------------------------------------------------------------
# 底层调用
# ---------------------------------------------------------------------------

def run(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    """执行 ffmpeg/ffprobe，返回完成结果（不抛非零退出）。"""
    return subprocess.run(
        args,
        capture_output=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def ffmpeg_available() -> bool:
    return resolve_ffmpeg() is not None


# ---------------------------------------------------------------------------
# 媒体探测
# ---------------------------------------------------------------------------

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:\s*([A-Za-z0-9_\-]+).*?,\s*(\d{2,5})x(\d{2,5})",
    re.S,
)
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:\s*([A-Za-z0-9_\-]+)", re.S)


def probe(path: str | Path) -> MediaInfo:
    """探测媒体信息。优先 ffprobe，缺失时解析 `ffmpeg -i`。"""
    p = Path(path)
    info = MediaInfo(path=str(p))
    if p.is_file():
        info.size_bytes = p.stat().st_size

    ffprobe = resolve_ffprobe()
    if ffprobe:
        got = _probe_with_ffprobe(ffprobe, p)
        if got is not None:
            return got
    return _probe_with_ffmpeg(p)


def _probe_with_ffprobe(ffprobe: str, p: Path) -> MediaInfo | None:
    try:
        cp = run([
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(p),
        ], timeout=60)
        data = json.loads(_decode(cp.stdout) or "{}")
    except Exception:  # noqa: BLE001
        return None

    fmt = data.get("format") or {}
    info = MediaInfo(path=str(p))
    try:
        info.duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0
    with contextlib.suppress(TypeError, ValueError):
        info.size_bytes = int(fmt.get("size") or info.size_bytes)

    for st in data.get("streams") or []:
        if st.get("codec_type") == "video" and not info.has_video:
            info.has_video = True
            info.video_codec = st.get("codec_name") or ""
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.fps = _parse_fraction(st.get("avg_frame_rate") or st.get("r_frame_rate"))
        elif st.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = st.get("codec_name") or ""

    return info


def _parse_fraction(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _probe_with_ffmpeg(p: Path) -> MediaInfo:
    """`ffmpeg -i` 会把信息打到 stderr，且无输出文件时退出码为 1，属正常。"""
    info = MediaInfo(path=str(p))
    if p.is_file():
        with contextlib.suppress(OSError):
            info.size_bytes = p.stat().st_size
    try:
        cp = run([ffmpeg_required(), "-hide_banner", "-i", str(p)], timeout=60)
    except Exception as exc:  # noqa: BLE001
        log.warning("探测失败 %s: %s", p, exc)
        return info

    text = _decode(cp.stderr)

    m = _DUR_RE.search(text)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info.duration = h * 3600 + mi * 60 + s

    m = _VIDEO_RE.search(text)
    if m:
        info.has_video = True
        info.video_codec = m.group(1)
        info.width = int(m.group(2))
        info.height = int(m.group(3))
    elif "Video:" in text:
        info.has_video = True
        m2 = re.search(r"Video:\s*([A-Za-z0-9_\-]+)", text)
        if m2:
            info.video_codec = m2.group(1)

    m = _FPS_RE.search(text)
    if m:
        info.fps = float(m.group(1))

    m = _AUDIO_RE.search(text)
    if m:
        info.has_audio = True
        info.audio_codec = m.group(1)

    return info


# ---------------------------------------------------------------------------
# 镜头分割（场景切换检测）
# ---------------------------------------------------------------------------

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def detect_scene_cuts(
    path: str | Path,
    threshold: float | None = None,
    max_duration: float | None = None,
) -> list[float]:
    """返回场景切换时间点（秒，升序，不含 0）。

    用 ffmpeg 原生 `select=gt(scene,T)` + `showinfo` 完成，零额外依赖。
    这比均匀抽帧准得多——它找的是画面真正发生结构性变化的时刻。
    """
    s = get_settings()
    thr = s.scene_threshold if threshold is None else threshold

    args = [ffmpeg_required(), "-hide_banner", "-nostats"]
    if max_duration and max_duration > 0:
        args += ["-t", f"{max_duration:.3f}"]
    args += [
        "-i", str(path),
        "-vf", f"select='gt(scene,{thr:.4f})',showinfo",
        "-an", "-f", "null", "-",
    ]

    try:
        cp = run(args, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning("场景检测超时: %s", path)
        return []

    text = _decode(cp.stderr)
    cuts = sorted({float(x) for x in _PTS_RE.findall(text) if float(x) > 0.05})
    log.info("场景检测完成：%d 个切换点（阈值 %.2f）", len(cuts), thr)
    return cuts


def cuts_to_shots(
    cuts: list[float],
    duration: float,
    min_shot_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """把切换点整理成镜头区间，并合并过短的镜头（快闪噪声）。"""
    s = get_settings()
    min_len = s.min_shot_seconds if min_shot_seconds is None else min_shot_seconds
    duration = max(duration, 0.01)

    bounds = [0.0]
    for c in sorted(cuts):
        if 0.0 < c < duration:
            bounds.append(c)
    bounds.append(duration)

    shots: list[tuple[float, float]] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        if b - a <= 0.001:
            continue
        if shots and (b - a) < min_len:
            # 太短，并入上一个镜头
            prev_a, _ = shots[-1]
            shots[-1] = (prev_a, b)
        else:
            shots.append((a, b))
    return shots


def uniform_shots(duration: float, target_count: int) -> list[tuple[float, float]]:
    """兜底：无场景切换时按均匀时长切分成若干「逻辑镜头」。"""
    duration = max(duration, 0.01)
    count = max(1, min(target_count, 60))
    step = duration / count
    return [(i * step, min((i + 1) * step, duration)) for i in range(count)]


# 平均镜头长度超过这个值，就怀疑是漏检了
_COARSE_AVG_SHOT = 6.0
# 降阈值重试时用的倍率（相对配置阈值）
_ADAPTIVE_FACTORS = (0.5,)
# 降阈值时最短镜头时长的下限，用来抑制噪声切点
_ADAPTIVE_MIN_SHOT = 1.0


def detect_shots_adaptive(
    src: str | Path,
    duration: float,
    threshold: float | None = None,
    min_shot_seconds: float | None = None,
) -> tuple[list[tuple[float, float]], int, float, str]:
    """自适应镜头检测。返回 (镜头列表, 切点数, 实际用的阈值, 说明)。

    为什么不能只用固定阈值：高阈值对**硬切**很准，但会漏掉**渐变转场**。
    VFX / 特效类内容（能量爆发、闪白、溶解、粒子过渡）大量使用软转场。
    实测一段 13.3 秒的特效视频：

        阈值 0.30（默认）→ 只检出 1 个切点，2 个镜头 → 只抽到 6 帧
        阈值 0.20       → 6 个切点，3 个镜头
        阈值 0.15       → 10 个切点，4 个镜头 → 抽到 12 帧

    镜头少 → 抽帧少 → 模型看到的信息就少，反推质量直接受影响。
    而漏检的特征很好认：**平均镜头长度异常长**。

    策略：先按配置阈值检一次；如果平均镜头长度超过 6 秒且视频不短于 8 秒
    （正常剪辑不会这样），就降一半阈值重试一次，同时把最短镜头时长提到
    1.0 秒 —— 降阈值会引入噪声切点，收紧最短时长把它们滤掉。
    只有当重试结果**明显更多**（至少 1.5 倍）时才采用，避免为噪声抖动买单。

    返回空镜头列表是正常的（纯色/测试图案类视频本来就没有切换），
    由调用方决定是否退化为均匀分镜。
    """
    s = get_settings()
    threshold = s.scene_threshold if threshold is None else threshold
    min_shot_seconds = s.min_shot_seconds if min_shot_seconds is None else min_shot_seconds

    cuts = detect_scene_cuts(src, threshold=threshold, max_duration=duration)
    shots = cuts_to_shots(cuts, duration, min_shot_seconds) if cuts else []
    best = (shots, len(cuts), threshold, "")

    avg = (duration / len(shots)) if shots else duration
    suspicious = duration >= 8.0 and (not shots or avg > _COARSE_AVG_SHOT)
    if not suspicious:
        return best

    for factor in _ADAPTIVE_FACTORS:
        alt_th = round(threshold * factor, 4)
        alt_min = max(min_shot_seconds, _ADAPTIVE_MIN_SHOT)
        alt_cuts = detect_scene_cuts(src, threshold=alt_th, max_duration=duration)
        alt = cuts_to_shots(alt_cuts, duration, alt_min) if alt_cuts else []

        # 必须明显更多才换，否则宁可用原结果（降阈值容易引入噪声切点）
        if len(alt) >= max(2, int(len(shots) * 1.5)):
            log.info(
                "镜头检测偏粗（平均 %.1fs / %d 个镜头），阈值 %.2f → %.2f 重检出 %d 个镜头",
                avg, len(shots), threshold, alt_th, len(alt),
            )
            return alt, len(alt_cuts), alt_th, f"自适应：阈值 {threshold:.2f} → {alt_th:.2f}"

    return best


# ---------------------------------------------------------------------------
# 抽帧
# ---------------------------------------------------------------------------

def _scale_filter(long_edge: int) -> str:
    """长边缩放到 long_edge，另一边按比例 -2（保证偶数）。横竖屏都适用。"""
    return (
        f"scale=w='if(gt(iw,ih),{long_edge},-2)':"
        f"h='if(gt(iw,ih),-2,{long_edge})':flags=lanczos"
    )


def extract_frame(
    src: str | Path,
    time_sec: float,
    out_path: Path,
    long_edge: int | None = None,
    quality: int | None = None,
) -> Path | None:
    """在指定时间点抽一张帧。`-ss` 放在 `-i` 前用快速定位。"""
    s = get_settings()
    le = long_edge or s.frame_long_edge
    q = quality if quality is not None else s.frame_jpeg_quality

    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
        "-ss", f"{max(0.0, time_sec):.3f}",
        "-i", str(src),
        "-frames:v", "1",
        "-vf", _scale_filter(le),
        "-q:v", str(max(2, min(31, int((100 - q) / 3.3) + 2))),
        "-y", str(out_path),
    ]
    try:
        cp = run(args, timeout=120)
    except subprocess.TimeoutExpired:
        log.warning("抽帧超时 t=%.2f %s", time_sec, src)
        return None

    if out_path.is_file() and out_path.stat().st_size > 512:
        return out_path
    log.debug("抽帧无输出 t=%.2f: %s", time_sec, _decode(cp.stderr)[-300:])
    return None


def build_contact_sheets(
    frames: list[tuple[float, Path]],
    cells_per_sheet: int,
    out_dir: Path | None = None,
    cols: int | None = None,
    quality: int = 3,
) -> list[tuple[list[float], Path]]:
    """把帧拼成若干张 contact sheet（网格图）。返回 [(该图包含的时间点列表, 图路径)]。

    **为什么拼图**：实测模型的图片 token 成本有上限 ——
    网格从 2.7 Mpx 做到 9.2 Mpx（3.4 倍），token 只从 1156 涨到 1176。
    也就是说**一格里放 6 帧还是 20 帧，成本几乎一样**。
    所以同样预算下可以给模型多几倍的时间覆盖度，速度还更快。

    实测（13.3s 视频，Pass1 全流程）：
        1fps / 12 张单帧  -> 46s，画面要素 7/7，T恤文字正确
        2fps / 3 张 3x3   -> 26s，画面要素 7/7，T恤文字正确，镜头数更准

    ⚠️ **代价：小字**。6 格时文字可靠，9 格以上开始编
    （实测 9/12/16/20 格都把 `FUTURE HERO` 读成 `ULTRA HERO`，
    偶尔又能读对 —— 在临界点上随机翻）。画面描述不受影响。
    所以只关心画面时用大网格，需要读小字时把 cells_per_sheet 压到 6 以下。
    """
    if not frames or cells_per_sheet < 2:
        return []

    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="sheets-", dir=str(TMP_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    cols = cols or max(2, int(math.ceil(math.sqrt(cells_per_sheet))))
    rows = max(1, int(math.ceil(cells_per_sheet / cols)))

    # 单元格尺寸：统一缩到 896x504 再加 3px 黑边，这样 hstack/vstack 能对齐
    cw, ch, border = 896, 504, 3
    tw, th = cw + border * 2, ch + border * 2

    sheets: list[tuple[list[float], Path]] = []
    for si in range(0, n, cells_per_sheet):
        group = frames[si:si + cells_per_sheet]
        if len(group) < 2:
            # 最后只剩一帧就没必要拼图了，交给调用方当单帧处理
            sheets.append(([group[0][0]], group[0][1]))
            continue

        dst = out_dir / f"sheet{si // cells_per_sheet + 1:02d}.jpg"
        total = cols * rows
        parts: list[str] = []
        cells: list[str] = []
        for i in range(len(group)):
            parts.append(
                f"[{i}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
                f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"pad={tw}:{th}:{border}:{border}:color=black,setsar=1[v{i}]"
            )
            cells.append(f"[v{i}]")
        # 不足一格的黑块补齐，保证网格对齐
        for j in range(len(group), total):
            parts.append(f"color=c=black:s={tw}x{th}:d=1,setsar=1[p{j}]")
            cells.append(f"[p{j}]")

        # 逐行 hstack，再把各行 vstack
        row_labels: list[str] = []
        for r in range(rows):
            row = cells[r * cols:(r + 1) * cols]
            parts.append(f"{''.join(row)}hstack=inputs={len(row)}[r{r}]")
            row_labels.append(f"[r{r}]")
        parts.append(f"{''.join(row_labels)}vstack=inputs={rows}[out]")

        cmd = (
            [resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
            + [arg for _, path in group for arg in ("-i", str(path))]
            + ["-filter_complex", ";".join(parts), "-map", "[out]",
               "-frames:v", "1", "-q:v", str(quality), str(dst)]
        )
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
            if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
                sheets.append(([t for t, _ in group], dst))
            else:
                log.warning(
                    "拼图失败，退回单帧：%s",
                    proc.stderr.decode("utf-8", "replace")[-300:],
                )
                sheets.extend(([t], p) for t, p in group)
        except Exception as exc:  # noqa: BLE001
            log.warning("拼图异常，退回单帧：%s", exc)
            sheets.extend(([t], p) for t, p in group)

    return sheets


def describe_sheet_layout(
    sheets: list[tuple[list[float], Path]], cells_per_sheet: int
) -> str:
    """给模型看的网格布局说明。不说明的话模型不知道格子和时间怎么对应。"""
    if not sheets:
        return ""
    cols = max(2, int(math.ceil(math.sqrt(cells_per_sheet))))
    rows = max(1, int(math.ceil(cells_per_sheet / cols)))
    lines = [
        f"IMPORTANT — the {len(sheets)} attached image(s) are CONTACT SHEETS, not single frames.",
        f"Each sheet is a {cols}x{rows} grid holding up to {cells_per_sheet} frames in reading "
        f"order (left-to-right, then top-to-bottom), in chronological sequence.",
        "Treat every grid cell as one separate frame. Cells filled with black are padding — ignore them.",
    ]
    for i, (times, _) in enumerate(sheets, 1):
        if len(times) > 1:
            rng = ", ".join(f"{t:.2f}s" for t in times)
            lines.append(f"  sheet {i}: {rng}")
        else:
            lines.append(f"  sheet {i}: single frame at {times[0]:.2f}s")
    return "\n".join(lines)


def extract_frames_at(
    src: str | Path,
    times: list[float],
    out_dir: Path | None = None,
    workers: int = 5,
    long_edge: int | None = None,
) -> list[tuple[float, Path]]:
    """批量抽帧（并行）。返回 (时间点, 文件路径) 列表，按时间升序。"""
    if not times:
        return []

    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="frames-", dir=str(TMP_DIR)))
    out_dir.mkdir(parents=True, exist_ok=True)

    uniq: list[float] = []
    seen: set[int] = set()
    for t in sorted(times):
        key = int(round(t * 1000))
        if key not in seen:
            seen.add(key)
            uniq.append(t)

    results: dict[float, Path] = {}

    def _one(t: float) -> None:
        dst = out_dir / f"t{int(round(t * 1000)):09d}.jpg"
        got = extract_frame(src, t, dst, long_edge=long_edge)
        if got:
            results[t] = got

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_one, uniq))

    return sorted(results.items(), key=lambda kv: kv[0])


# ---------------------------------------------------------------------------
# 音频
# ---------------------------------------------------------------------------

def extract_audio(
    src: str | Path,
    out_path: Path | None = None,
    sample_rate: int = 16000,
) -> Path | None:
    """抽成 16kHz 单声道 PCM WAV（ASR 通用输入格式）。"""
    out_path = out_path or (Path(tempfile.mkdtemp(prefix="audio-", dir=str(TMP_DIR))) / "audio.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-y", str(out_path),
    ]
    try:
        run(args, timeout=300)
    except subprocess.TimeoutExpired:
        return None

    if out_path.is_file() and out_path.stat().st_size > 1024:
        return out_path
    return None


def extract_audio_segment(
    src: str | Path,
    start: float,
    end: float,
    out_path: Path | None = None,
    sample_rate: int = 16000,
) -> Path | None:
    """抽某个时间段的音频（给 Pass1 附音频用，只需该分块的那一段）。

    单声道 16kHz PCM 是为了控制体积：60 秒约 1.9MB，base64 后约 2.6MB，
    离 OpenAI 的 25MB 上限还很远。用 44.1kHz 立体声会直接放大 5 倍以上。
    """
    span = max(0.1, end - start)
    out_path = out_path or (
        Path(tempfile.mkdtemp(prefix="aseg-", dir=str(TMP_DIR))) / f"seg_{start:.2f}.wav"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
        # -ss 放在 -i 前面走快速定位，长视频上差别很明显
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(src),
        "-t", f"{span:.3f}",
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-y", str(out_path),
    ]
    try:
        run(args, timeout=300)
    except subprocess.TimeoutExpired:
        return None

    if out_path.is_file() and out_path.stat().st_size > 1024:
        return out_path
    return None


_VOL_RE = re.compile(r"(mean_volume|max_volume):\s*(-?[0-9.]+)\s*dB")
_SIL_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def _band_mean_volume(src: str | Path, filters: str) -> float | None:
    """测某个频段的平均音量（dB）。滤镜链由调用方给，例如 `lowpass=f=200`。"""
    try:
        cp = run([
            ffmpeg_required(), "-hide_banner", "-nostats",
            "-i", str(src),
            "-af", f"{filters},volumedetect",
            "-f", "null", "-",
        ], timeout=300)
    except Exception:  # noqa: BLE001
        return None
    for name, val in _VOL_RE.findall(_decode(cp.stderr)):
        if name == "mean_volume":
            return float(val)
    return None


def analyze_audio_levels(src: str | Path) -> dict:
    """音量、静音、以及频段能量分布。

    为什么加频段能量：没配 ASR 时音频内容完全未知，但如果只报一句「未转写」，
    音频维度就彻底废了。频谱能量是 ffmpeg 能直接测出来的**客观量**，
    它不能告诉你「是什么声音」，但能告诉你「能量集中在哪」——
    足够让模型写出「疑似以人声为主、低频成分弱」这种**有依据的谨慎描述**，
    而不是编造，也不是空白。
    """
    result: dict = {
        "mean_volume_db": None,
        "peak_volume_db": None,
        "silence_ratio": None,
        "loudness_points": [],
        "speech_band_db": None,
        "low_band_db": None,
    }

    try:
        cp = run([
            ffmpeg_required(), "-hide_banner", "-nostats",
            "-i", str(src),
            "-af", "volumedetect",
            "-f", "null", "-",
        ], timeout=300)
        text = _decode(cp.stderr)
        for name, val in _VOL_RE.findall(text):
            key = "mean_volume_db" if name == "mean_volume" else "peak_volume_db"
            result[key] = float(val)
    except Exception:  # noqa: BLE001
        pass

    try:
        cp = run([
            ffmpeg_required(), "-hide_banner", "-nostats",
            "-i", str(src),
            "-af", "silencedetect=noise=-35dB:d=0.4",
            "-f", "null", "-",
        ], timeout=300)
        text = _decode(cp.stderr)
        starts = [float(x) for x in _SIL_START.findall(text)]
        ends = [float(x) for x in _SIL_END.findall(text)]
        sil_total = sum(max(0.0, b - a) for a, b in zip(starts, ends, strict=False))
        duration = probe(src).duration
        if duration > 0:
            result["silence_ratio"] = round(min(1.0, sil_total / duration), 3)
        # 静音段结束点 = 声音重新进来的地方，往往是卡点
        result["loudness_points"] = [round(x, 2) for x in ends[:40]]
    except Exception:  # noqa: BLE001
        pass

    # 频段能量：语音频段（人声/对白主要落在这里）与低频段（鼓、贝斯、音乐铺底）
    #
    # 滤波链都串两遍：ffmpeg 的 highpass/lowpass 是单极点（6dB/oct），太缓。
    # 单极点 lowpass=f=200 挡不住 440Hz —— 实测纯 440Hz 正弦的低频相对能量
    # 是 -13.8dB（明显是泄漏），串成 12dB/oct 后降到 -27.7dB，区分度才够。
    full = result["mean_volume_db"]
    if full is not None:
        speech = _band_mean_volume(src, "highpass=f=300,highpass=f=300,"
                                        "lowpass=f=3400,lowpass=f=3400")
        if speech is not None:
            result["speech_band_db"] = round(speech - full, 1)
        low = _band_mean_volume(src, "lowpass=f=200,lowpass=f=200")
        if low is not None:
            result["low_band_db"] = round(low - full, 1)

    return result


# ---------------------------------------------------------------------------
# 物理切分（按需使用）
# ---------------------------------------------------------------------------

def segment_video(
    src: str | Path,
    out_dir: Path | None = None,
    chunk_seconds: float = 60.0,
    total_duration: float | None = None,
    reencode: bool = False,
) -> list[tuple[float, float, Path]]:
    """把长视频物理切成多个片段文件。

    注意：`reencode=False` 走 `-c copy`，秒级完成但切点会吸附到关键帧。
    抽帧管线并不需要它——抽帧用绝对时间定位即可（见 extract_frames_at）。
    这个方法用于「给用户提供分片预览」或「喂给原生视频模型」。
    """
    s = get_settings()
    # 注意用 is None 判断：传 0.0 表示「时长未知，别切」而不是「回退到探测」
    dur = probe(src).duration if total_duration is None else total_duration
    if dur <= 0:
        return []
    chunk_seconds = chunk_seconds or s.chunk_seconds

    out_dir = out_dir or (Path(tempfile.mkdtemp(prefix="seg-", dir=str(TMP_DIR))))
    out_dir.mkdir(parents=True, exist_ok=True)

    segs: list[tuple[float, float, Path]] = []
    start = 0.0
    idx = 0
    while start < dur - 0.05:
        length = min(chunk_seconds, dur - start)
        dst = out_dir / f"chunk{idx:03d}.mp4"
        args = [
            ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{length:.3f}",
        ]
        if not reencode:
            args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"]
        args += ["-y", str(dst)]

        try:
            run(args, timeout=900)
        except subprocess.TimeoutExpired:
            log.warning("切分超时 start=%.2f", start)

        if dst.is_file() and dst.stat().st_size > 1024:
            segs.append((start, start + length, dst))
        start += length
        idx += 1

    return segs


# ---------------------------------------------------------------------------
# 缩略图 / 封面
# ---------------------------------------------------------------------------

def make_thumbnail(src: str | Path, out_path: Path, time_sec: float = 1.0) -> Path | None:
    return extract_frame(src, time_sec, out_path, long_edge=640, quality=80)


def make_contact_sheet(
    frames: list[Path],
    out_path: Path,
    cols: int = 6,
    cell: int = 320,
) -> Path | None:
    """把抽出来的帧拼成一张联络表，便于人工核对抽帧是否合理。"""
    if not frames:
        return None
    try:
        from PIL import Image  # 可选依赖
    except ImportError:
        return None

    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (16, 16, 20))
    for i, fp in enumerate(frames):
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        im.thumbnail((cell, cell))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out_path, quality=85)
    return out_path


def cleanup(path: str | Path) -> None:
    """尽力删除临时文件。**任何情况下都不许抛异常。**

    这里刻意捕获 BaseException 而不是 OSError：清理是收尾动作，
    失败了最多留几个临时文件，绝不该把整个服务带走。

    踩过的坑：某些运行环境会在 `shutil.rmtree` 里塞进删除保护
    （拦截大量文件的递归删除），抛的是 `SystemExit`——它继承 `BaseException`
    而不是 `Exception`，所以 `except OSError` 和调用方的 `except Exception`
    都拦不住，一次清理就把 uvicorn 进程干掉了。
    """
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)
    except BaseException as exc:  # noqa: BLE001
        log.warning("清理 %s 失败（已忽略）：%s", p, exc)


def frame_dir_for(job_id: str) -> Path:
    d = FRAME_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d
