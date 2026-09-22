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

    # --- 音量 / 静音 / 频段能量 ---
    try:
        levels = ff.analyze_audio_levels(video_path)
        report.mean_volume_db = levels.get("mean_volume_db")
        report.peak_volume_db = levels.get("peak_volume_db")
        report.silence_ratio = levels.get("silence_ratio")
        report.loudness_points = levels.get("loudness_points") or []
        report.speech_band_db = levels.get("speech_band_db")
        report.low_band_db = levels.get("low_band_db")
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

        # 只有 ASR 真的配了才去调。原来这里写的是 `or s.vlm_api_key`，
        # 结果只要配了视觉模型的 key 就会拿它去调 ASR —— 而 ASR_BASE_URL
        # 是空的，请求直接报 "URL is missing an 'http://' protocol"。
        # 视觉模型的 key 跟语音转写是两回事，不能互相顶替。
        if s.asr_enabled:
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


def describe_spectrum(report: AudioReport) -> list[str]:
    """把频段能量翻译成**谨慎的**倾向性描述。

    措辞的边界很重要。频谱能量**分不出**「人声」和「独奏乐器/窄带音调」——
    实测一个纯 440Hz 正弦波的能量几乎全落在语音频段里，读起来和人声一样。
    所以只能说「能量集中在语音频段」，不能升级成「有人说话」。
    同理低频强只能说「有低频成分」，不能说「有鼓点」。
    """
    lines: list[str] = []
    speech, low = report.speech_band_db, report.low_band_db

    if speech is not None:
        if speech >= -2.0:
            lines.append(
                f"  语音频段(300-3400Hz)相对能量 {speech:+.1f}dB"
                "  → 能量高度集中在这一频段（人声、独奏乐器、窄带音调都可能落在这里，"
                "频谱无法区分）"
            )
        elif speech >= -8.0:
            lines.append(
                f"  语音频段(300-3400Hz)相对能量 {speech:+.1f}dB"
                "  → 该频段占比较高，但混有其他频段成分（典型情况：人声 + 配器）"
            )
        else:
            lines.append(
                f"  语音频段(300-3400Hz)相对能量 {speech:+.1f}dB"
                "  → 能量大部分在该频段之外，**不太像以人声为主**"
                "（更可能是纯音乐、环境声或宽频噪声）"
            )

    if low is not None:
        if low >= -10.0:
            lines.append(
                f"  低频(<200Hz)相对能量 {low:+.1f}dB"
                "  → 低频成分显著，**疑似有节奏性的音乐编排**（鼓/贝斯类），"
                "但也可能是低频环境噪声"
            )
        elif low >= -20.0:
            lines.append(f"  低频(<200Hz)相对能量 {low:+.1f}dB  → 有中等低频成分")
        else:
            lines.append(
                f"  低频(<200Hz)相对能量 {low:+.1f}dB"
                "  → 低频很弱，不太像有节奏性的音乐编排"
            )

    return lines


def format_transcript_for_prompt(report: AudioReport) -> str:
    """把转写整理成带时间戳的文本，供模型对齐到镜头。

    没做转写时**必须显式禁止编造声音内容**。只写一句「未获得文本内容」，
    模型会自己补出「电子提示音与数字跳变同步」「一段缓慢的合成器铺底」这类
    听起来很合理的声音描述——它根本听不到音频。反推出来的 BGM / 台词是编的，
    整条提示词就废了。

    频谱特征是可以给的：那是实测数字，不是内容。但必须标明它的边界。
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

    spectrum = describe_spectrum(report)
    if spectrum:
        lines.append("【音频频谱特征（ffmpeg 实测，不是内容识别）】")
        lines.extend(spectrum)
        lines.append(
            "  ↑ 这些是能量分布数据，只能用来做**倾向性**描述"
            "（例如「音轨疑似以人声为主，低频成分弱」）。\n"
            "    不得据此断言具体内容（不能说「有人在唱歌」「有一段合成器铺底」）。"
        )

    if report.note:
        lines.append(f"【备注】{report.note}")

    if not transcribed:
        lines.append(
            "【重申】以上只有音量与频谱数据，没有任何声音内容信息。"
            "音频字段留空、写 N/A、或只做倾向性描述，不要凭画面猜声音。"
        )

    return "\n".join(lines)
