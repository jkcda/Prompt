"""音频维度分析：语音转写 + 音量/静音检测。

为什么必须有这一步：
    画面帧永远推不出 `overall_soundscape`、台词、口型同步、BGM 卡点。
    而目标格式（H3 六段式 / Seedance）里这些字段是硬要求。
    所以 ASR 时间戳 + 音量曲线是反推的必需输入，不是可选增强。

优先级：本地 faster-whisper（若已安装）→ OpenAI 兼容 ASR API → 跳过并说明。
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import httpx

from ..core.config import TMP_DIR, get_settings
from ..schemas import AudioReport, TranscriptSegment
from . import ffmpeg as ff

log = logging.getLogger("asr")

# 单个音频分片的目标时长（秒）。ASR API 通常限制 25MB / 10 分钟
SLICE_SECONDS = 300.0


# ---------------------------------------------------------------------------
# 本地 whisper（可选）
# ---------------------------------------------------------------------------

def _local_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _transcribe_local(audio_path: Path) -> tuple[str, list[TranscriptSegment]]:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), vad_filter=True)
    out: list[TranscriptSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
    return " ".join(s.text for s in out), out


# ---------------------------------------------------------------------------
# API 转写
# ---------------------------------------------------------------------------

def _slice_audio(audio_path: Path, duration: float) -> list[tuple[float, Path]]:
    """把长音频切成若干片，返回 (时间偏移, 分片路径)。"""
    if duration <= SLICE_SECONDS + 1:
        return [(0.0, audio_path)]

    out: list[tuple[float, Path]] = []
    start = 0.0
    idx = 0
    while start < duration:
        length = min(SLICE_SECONDS, duration - start)
        dst = TMP_DIR / f"asr-slice-{audio_path.stem}-{idx:03d}.wav"
        args = [
            ff.ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(audio_path),
            "-t", f"{length:.3f}",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", str(dst),
        ]
        with contextlib.suppress(Exception):
            ff.run(args, timeout=300)
        if dst.is_file() and dst.stat().st_size > 1024:
            out.append((start, dst))
        start += length
        idx += 1
    return out


def _transcribe_api(audio_path: Path, duration: float) -> tuple[str, list[TranscriptSegment]]:
    s = get_settings()
    base = s.asr_base_url.rstrip("/")
    url = f"{base}/audio/transcriptions" if not base.endswith("/audio/transcriptions") else base
    key = s.asr_api_key or s.vlm_api_key

    all_segments: list[TranscriptSegment] = []
    pieces = _slice_audio(audio_path, duration)

    with httpx.Client(timeout=300.0) as client:
        for offset, piece in pieces:
            try:
                with piece.open("rb") as fh:
                    resp = client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}"},
                        files={"file": (piece.name, fh, "audio/wav")},
                        data={
                            "model": s.asr_model,
                            "response_format": "verbose_json",
                            "temperature": "0",
                        },
                    )
                if resp.status_code >= 400:
                    log.warning("ASR 分片失败 %s: %s", resp.status_code, resp.text[:200])
                    continue
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                log.warning("ASR 请求异常: %s", exc)
                continue

            for seg in payload.get("segments") or []:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                all_segments.append(TranscriptSegment(
                    start=float(seg.get("start") or 0.0) + offset,
                    end=float(seg.get("end") or 0.0) + offset,
                    text=text,
                ))
            if not payload.get("segments") and payload.get("text"):
                all_segments.append(TranscriptSegment(
                    start=offset, end=offset + SLICE_SECONDS,
                    text=str(payload["text"]).strip(),
                ))

    # 清理分片临时文件
    for _off, piece in pieces:
        if piece != audio_path:
            ff.cleanup(piece)

    full = " ".join(s.text for s in all_segments)
    return full, all_segments


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def analyze_audio(video_path: str | Path, enable_asr: bool = True) -> AudioReport:
    """完整音频分析：转写 + 音量 + 静音。"""
    report = AudioReport()
    info = ff.probe(video_path)
    if not info.has_audio:
        report.note = "该视频没有音轨，音频维度无法反推"
        return report

    report.has_audio = True

    # --- 音量 / 静音 ---
    try:
        levels = ff.analyze_audio_levels(video_path)
        report.mean_volume_db = levels.get("mean_volume_db")
        report.peak_volume_db = levels.get("peak_volume_db")
        report.silence_ratio = levels.get("silence_ratio")
        report.loudness_points = levels.get("loudness_points") or []
    except Exception as exc:  # noqa: BLE001
        log.warning("音量分析失败: %s", exc)

    if not enable_asr:
        report.note = "已跳过语音转写（未启用）"
        return report

    # --- 语音转写 ---
    s = get_settings()
    audio_path = ff.extract_audio(video_path)
    if not audio_path:
        report.note = "音轨抽取失败"
        return report

    try:
        if _local_whisper_available():
            try:
                report.transcript, report.segments = _transcribe_local(audio_path)
                report.note = "本地 faster-whisper 转写"
                return report
            except Exception as exc:  # noqa: BLE001
                log.warning("本地 whisper 失败，回退 API: %s", exc)

        if s.asr_enabled or s.vlm_api_key:
            report.transcript, report.segments = _transcribe_api(audio_path, info.duration)
            report.note = "API 转写" if report.transcript else "API 转写返回空结果"
        else:
            report.note = "未配置 ASR（ASR_API_KEY / ASR_BASE_URL 为空），已跳过语音转写"
    except Exception as exc:  # noqa: BLE001
        report.note = f"语音转写失败：{exc}"
        log.warning(report.note)
    finally:
        ff.cleanup(audio_path.parent)

    return report


def format_transcript_for_prompt(report: AudioReport) -> str:
    """把转写整理成带时间戳的文本，供模型对齐到镜头。

    没做转写时**必须显式禁止编造声音内容**。只写一句「未获得文本内容」，
    模型会自己补出「电子提示音与数字跳变同步」「一段缓慢的合成器铺底」这类
    听起来很合理的声音描述——它根本听不到音频。反推出来的 BGM / 台词是编的，
    整条提示词就废了。
    """
    if not report.has_audio:
        return "【音频】该视频无音轨。所有音频字段请写 N/A。"

    lines: list[str] = []
    transcribed = bool(report.segments or report.transcript)

    if report.segments:
        lines.append("【语音转写（时间戳 → 文本）】")
        for seg in report.segments[:200]:
            lines.append(f"  [{seg.start:6.2f}s - {seg.end:6.2f}s] {seg.text}")
    elif report.transcript:
        lines.append("【语音转写（无时间戳）】")
        lines.append("  " + report.transcript)
    else:
        lines.append("【语音转写】未做语音识别，识别结果为空。")
        lines.append(
            "  ⚠ 音轨存在，但音频内容对你完全未知。禁止描述任何具体的声音：\n"
            "    - 不要写台词或歌词（哪怕只是「似乎在说」）\n"
            "    - 不要写 BGM 的乐器、节奏、情绪\n"
            "    - 不要写具体的音效类型（提示音、脚步、风声、衣料声等）\n"
            "    音频相关字段只能写「存在音轨但内容未分析」，或直接写 N/A。\n"
            "    编造声音比留空更糟——生成出来会和原片完全对不上。"
        )

    meta: list[str] = []
    if report.mean_volume_db is not None:
        meta.append(f"平均音量 {report.mean_volume_db:.1f}dB")
    if report.peak_volume_db is not None:
        meta.append(f"峰值 {report.peak_volume_db:.1f}dB")
    if report.silence_ratio is not None:
        meta.append(f"静音占比 {report.silence_ratio * 100:.0f}%")
    if report.loudness_points:
        pts = ", ".join(f"{p:.1f}s" for p in report.loudness_points[:12])
        meta.append(f"声音重新进入的时间点（疑似卡点）：{pts}")
    if meta:
        lines.append("【音频能量】" + "；".join(meta))

    if report.note:
        lines.append(f"【备注】{report.note}")

    if not transcribed:
        lines.append(
            "【重申】以上只有音量信息，没有任何声音内容信息。"
            "音频字段留空或写 N/A，不要凭画面猜声音。"
        )

    return "\n".join(lines)
