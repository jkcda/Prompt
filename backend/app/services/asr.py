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
from . import audio_features
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


def _transcribe_local(audio_path: Path) -> tuple[str, list[TranscriptSegment], str]:
    """本地 faster-whisper 转写。返回 (全文, 分句, 语言)。

    **不指定 language**，让模型自己检测 —— 指定成 zh/en 会把日语歌词
    硬翻成中文，而我们要的是**保留原语言**。

    模型权重从 HuggingFace 下载，国内直连很慢或直接失败，所以默认把
    HF_ENDPOINT 指向镜像站（可用 ASR_HF_ENDPOINT 覆盖，或自己先设好环境变量）。
    """
    import os

    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = get_settings().asr_hf_endpoint or "https://hf-mirror.com"

    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), vad_filter=True)
    out: list[TranscriptSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
    return " ".join(s.text for s in out), out, (info.language or "")


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


def _transcribe_api(audio_path: Path, duration: float) -> tuple[str, list[TranscriptSegment], str]:
    s = get_settings()
    base = s.asr_base_url.rstrip("/")
    url = f"{base}/audio/transcriptions" if not base.endswith("/audio/transcriptions") else base
    key = s.asr_api_key or s.vlm_api_key

    all_segments: list[TranscriptSegment] = []
    language = ""
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

            # verbose_json 会带检测到的语言；拿第一个分片的就够
            language = language or str(payload.get("language") or "")
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
    return full, all_segments, language


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def analyze_audio(video_path: str | Path, enable_asr: bool = True) -> AudioReport:
    """完整音频分析：音乐画像 + 语音转写 + 音量/静音/频段能量。

    顺序有讲究：**音乐画像不需要模型**，先做。这样即使 ASR 完全不可用
    （没装 faster-whisper、也没配 API），音频维度仍然能输出一句人类可读的
    「这是一首什么歌」，而不是只剩「未识别」。
    """
    report = AudioReport()
    info = ff.probe(video_path)
    if not info.has_audio:
        report.note = "该视频没有音轨，音频维度无法反推"
        return report

    report.has_audio = True

    # --- 音量 / 静音 / 频段能量 ---
    levels: dict = {}
    try:
        levels = ff.analyze_audio_levels(video_path)
        report.mean_volume_db = levels.get("mean_volume_db")
        report.peak_volume_db = levels.get("peak_volume_db")
        report.silence_ratio = levels.get("silence_ratio")
        report.loudness_points = levels.get("loudness_points") or []
        report.speech_band_db = levels.get("speech_band_db")
        report.low_band_db = levels.get("low_band_db")
        report.high_band_db = levels.get("high_band_db")
    except Exception as exc:  # noqa: BLE001
        log.warning("音量分析失败: %s", exc)

    # --- 音乐画像：节奏 / 编制 / 动态。不需要任何模型。 ---
    try:
        profile = audio_features.analyze_music(str(video_path), levels)
        report.bpm = profile.bpm
        report.has_beat = profile.has_beat
        report.onset_rate = profile.onset_rate
        report.transient_bursts = profile.transient_bursts
        report.dynamic_range_db = profile.dynamic_range_db
        report.music_profile = profile.describe()
        report.note = profile.note
    except Exception as exc:  # noqa: BLE001
        log.warning("音乐画像分析失败: %s", exc)
        report.note = f"音乐画像分析失败：{exc}"

    if not enable_asr:
        report.note = f"{report.note}；已跳过语音转写（未启用）".strip("；")
        return report

    # --- 语音转写 ---
    s = get_settings()
    audio_path = ff.extract_audio(video_path)
    if not audio_path:
        report.note = f"{report.note}；音轨抽取失败".strip("；")
        return report

    try:
        # 先做人声频段分离再送去转写。MV / 现场录音里鼓和贝斯能量很强，
        # whisper 会被伴奏带偏；滤掉两头之后歌词的准确率明显更高。
        asr_input = audio_path
        if s.vocal_isolation:
            vocal_path, iso_note = audio_features.isolate_vocals(
                str(audio_path), str(audio_path.parent / "vocals.wav")
            )
            report.vocal_isolation = iso_note
            if vocal_path:
                asr_input = Path(vocal_path)
        else:
            report.vocal_isolation = "未做人声分离（VOCAL_ISOLATION 关闭）"

        if _local_whisper_available():
            try:
                report.transcript, report.segments, report.language = _transcribe_local(asr_input)
                report.note = f"{report.note}；本地 faster-whisper 转写".strip("；")
                return report
            except Exception as exc:  # noqa: BLE001
                log.warning("本地 whisper 失败，回退 API: %s", exc)

        # 只有 ASR 真的配了才去调。原来这里写的是 `or s.vlm_api_key`，
        # 结果只要配了视觉模型的 key 就会拿它去调 ASR —— 而 ASR_BASE_URL
        # 是空的，请求直接报 "URL is missing an 'http://' protocol"。
        # 视觉模型的 key 跟语音转写是两回事，不能互相顶替。
        if s.asr_enabled:
            report.transcript, report.segments, report.language = _transcribe_api(
                asr_input, info.duration
            )
            report.note = f"{report.note}；{'API 转写' if report.transcript else 'API 转写返回空结果'}"
        else:
            report.note = (
                f"{report.note}；未配置 ASR（ASR_API_KEY / ASR_BASE_URL 为空），已跳过语音转写"
            )
    except Exception as exc:  # noqa: BLE001
        report.note = f"语音转写失败：{exc}"
        log.warning(report.note)
    finally:
        ff.cleanup(audio_path.parent)

    return report


def describe_spectrum(report: AudioReport) -> list[str]:
    """把频段能量翻译成**谨慎的**倾向性描述。

    ⚠️ 这个函数的结果**不再直接进提示词**。实测输出里漏出过
    `voice band` / `energy distribution` 这类工具术语 —— 模型会把给它的
    措辞照抄进最终提示词，而视频模型拿到这种句子写不出任何东西。
    现在频段数据只用来支撑 `MusicProfile.describe()` 里那句人类可读的话。

    保留这个函数是因为它仍然能解释「为什么只能说到这个程度」：
    频谱能量**分不出**「人声」和「独奏乐器/窄带音调」——
    实测一个纯 440Hz 正弦波的能量几乎全落在语音频段里，读起来和人声一样。
    """
    lines: list[str] = []
    speech, low = report.speech_band_db, report.low_band_db

    if speech is not None:
        if speech >= -2.0:
            lines.append("人声/主奏频段能量占比高")
        elif speech >= -8.0:
            lines.append("人声/主奏频段占比较高，混有其他频段")
        else:
            lines.append("能量大部分落在人声频段之外（更接近纯音乐或环境声）")

    if low is not None:
        if low >= -10.0:
            lines.append("低频成分显著（鼓、贝斯这类）")
        elif low >= -20.0:
            lines.append("有中等低频成分")
        else:
            lines.append("低频很弱")

    return lines


def format_transcript_for_prompt(report: AudioReport) -> str:
    """把音频报告整理成成文阶段能直接用的文本。

    ⚠️ 这里写出来的措辞会**原样进入最终提示词**（模型习惯照抄），所以：

    1. **不给工具术语。** 实测最终输出里出现过
       `Sound present, content unanalysed` /
       `measured energy mostly voice band` / `inferred from energy distribution`
       —— 那是我们自己的 system prompt 里的措辞被抄走了。视频模型拿到
       这种句子什么也写不出来。所以这里只给人类可读的音乐描述。
    2. **不给 dB / Hz 数字。** 数字对生成视频没有意义，而且会诱导模型
       写「低频 -12dB 的成分」这种句子。数字只在服务端支撑那句描述。
    3. **「不知道」必须显式说成不知道。** 只写「未获得文本内容」这种中性
       陈述等于留白，模型一定会去填 —— 实测它编出了「电子提示音与数字
       跳变同步」并标成 `fully_copy`。所以要把「未知」写死，并逐条列出禁令。
    """
    if not report.has_audio:
        return "【音频】该视频没有音轨。所有音频字段写 N/A。"

    lines: list[str] = []

    # ---- 音乐与声音：人类可读的描述，可以直接用 ----
    lines.append("【音乐与声音（这是实测得出的描述，可以直接引用或改写措辞）】")
    if report.music_profile:
        lines.append("  " + report.music_profile)
        if report.bpm and report.has_beat:
            lines.append(f"  （节奏约 {report.bpm:.0f} BPM）")
    else:
        lines.append("  未能得出音乐描述（音轨过短或抽取失败）。")
    lines.append("")

    # ---- 歌词 / 台词 ----
    if report.segments:
        lines.append("【歌词 / 台词（时间戳 → 原文）】")
        for seg in report.segments[:200]:
            lines.append(f"  [{seg.start:6.2f}s - {seg.end:6.2f}s] {seg.text}")
        if report.language:
            lines.append(
                f"【语言】{report.language} —— 歌词/台词必须**保持原语言**，不要翻译、不要改写。"
            )
    elif report.transcript:
        lines.append("【歌词 / 台词（整段，无时间戳）】")
        lines.append("  " + report.transcript)
        if report.language:
            lines.append(f"【语言】{report.language} —— 保持原语言。")
    else:
        lines.append("【歌词 / 台词】没有做语音识别，**你对歌词和台词一无所知**。")
        lines.append(
            "  ⚠ 这是「未知」，不是「没有」。禁止编造：\n"
            "    - 不要写任何台词或歌词（哪怕加「似乎在说」「隐约听到」也不行）\n"
            "    - 不要写 BGM 的乐器、节奏、情绪（上面的音乐描述除外）\n"
            "    - 不要写具体的音效类型（提示音、脚步、风声、衣料声等）\n"
            "    没有把握的音频字段写 N/A，比编一个更像真的。"
        )
    lines.append("")

    if report.note:
        lines.append(f"【音频分析备注】{report.note}")
    lines.append("")

    # ---- 写作要求：只描述，不分析 ----
    lines.append("【声音字段的写作要求】")
    lines.append(
        "  用音乐人和场务能听懂的话描述声音：什么乐器在响、节奏多快、"
        "有没有人声、有没有观众。"
    )
    lines.append(
        "  ⛔ 不要使用音频**分析**用语（频段名称、能量/频谱的说法、分贝或赫兹"
        "这类测量单位、以及「未分析」「未识别」这种状态说明）。那些是内部工作笔记，"
        "不是提示词内容。要么给出上面那种可听可感的描述，要么留空写 N/A。"
    )
    return "\n".join(lines)
