"""音频的客观特征：节拍、瞬态、动态、频段平衡。

为什么要有这一步
----------------
反推的音频字段以前只有「频谱能量」和「未做识别」两句话，结果实测输出的是：

    Sound present, content unanalysed
    measured energy mostly voice band with notable low-frequency component
    inferred from energy distribution alone

——这是**工具的分析术语**，不是提示词该有的内容。视频模型拿到这种句子
写不出任何东西。而缺的其实只有两件事：歌词（靠 ASR）和音乐长什么样
（靠节拍 + 频段，就是这里做的）。

为什么不引入 librosa
--------------------
librosa 会带来 numpy + scipy + numba + scikit-learn（上百 MB），
而我们只需要「有没有稳定节拍、大概多快、低频有没有冲击」这几个量。
ffmpeg 抽 PCM + 纯 Python 算能量包络自相关就够了，而且零额外依赖。
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import struct
import subprocess
from dataclasses import dataclass

from ..core.config import ffmpeg_required

log = logging.getLogger("audio")

# 分析用的采样率。节拍只需要低频包络，8kHz 足够，而且数据量只有 44.1kHz 的 1/5。
_SAMPLE_RATE = 8000
# 只分析开头这么多秒。BPM 是全曲稳定的，不需要听完。
_MAX_SECONDS = 90.0
# 能量包络的帧长与跳距（样本数）。@8kHz：帧 64ms、跳 32ms。
_FRAME = 512
_HOP = 256
_TIMEOUT = 180.0

# 人声带通的两个拐点（Hz）。鼓和贝斯基本在 180Hz 以下，齿音和镲片在 4kHz 以上，
# 去掉两头能让 whisper 少受伴奏干扰。
_VOCAL_LOW = 180
_VOCAL_HIGH = 4000


@dataclass
class MusicProfile:
    """音乐/声音的客观画像。全部来自实测，没有一项是猜的。"""

    analysed: bool = False            # 是否真的跑过分析
    bpm: float | None = None          # 估计的每分钟拍数
    beat_strength: float = 0.0        # 0-1，自相关峰的显著程度
    has_beat: bool = False            # 有没有稳定节拍
    onset_rate: float = 0.0           # 每秒的音头数，反映节奏密度
    dynamic_range_db: float | None = None   # 峰值 - 平均，反映动态
    low_energy_db: float | None = None      # <200Hz 相对能量
    mid_energy_db: float | None = None      # 300-3400Hz（人声/主奏）
    high_energy_db: float | None = None     # >6kHz（镲片/齿音/空气感）
    transient_bursts: int = 0         # 高频瞬态簇的个数（掌声/欢呼的线索）
    note: str = ""

    def describe(self) -> str:
        """写成**人类可读的**句子，供成文阶段直接使用。

        措辞原则：客观数字照实说，推断必须带 hedge 并且说清推断依据。
        频谱能说明「能量在哪」，不能说明「是什么乐器在响」——
        「低频强」不等于「有鼓」，只能说「像是鼓/贝斯这类低频乐器」。
        """
        if not self.analysed:
            return ""

        parts: list[str] = []

        # ---- 节拍 ----
        if self.has_beat and self.bpm:
            tempo = "slow" if self.bpm < 90 else "mid-tempo" if self.bpm < 130 else "fast"
            parts.append(
                f"a {tempo} pulse of roughly {self.bpm:.0f} BPM"
                f"{' with strong, clearly accented beats' if self.beat_strength > 0.45 else ''}"
            )
        elif self.onset_rate >= 1.5:
            parts.append(
                f"rhythmic attacks at roughly {self.onset_rate:.0f} per second, "
                "without a single stable tempo"
            )

        # ---- 编制（靠频段平衡推断，不是识别）----
        instr: list[str] = []
        if self.low_energy_db is not None and self.low_energy_db > -10.0:
            instr.append("a strong low end, the kind a kick drum and bass produce")
        if self.high_energy_db is not None and self.high_energy_db > -12.0:
            instr.append("bright high-frequency content (cymbals, hats or air)")
        if self.mid_energy_db is not None and self.mid_energy_db > -3.0:
            instr.append("a dense mid-range carrying the main melodic and vocal line")
        if instr:
            parts.append("with " + ", ".join(instr))

        # ---- 动态 ----
        if self.dynamic_range_db is not None:
            if self.dynamic_range_db > 22:
                parts.append("wide dynamics")
            elif self.dynamic_range_db < 10:
                parts.append("heavily compressed, near-constant loudness")

        if not parts:
            return ""

        text = "The soundtrack reads as " + "; ".join(parts) + "."
        if self.transient_bursts >= 3:
            text += (
                " The dense bursts of high-frequency transients are consistent with "
                "applause or crowd noise rather than instruments."
            )
        return text


# ---------------------------------------------------------------------------
# PCM 抽取
# ---------------------------------------------------------------------------

def extract_pcm(src: str, max_seconds: float = _MAX_SECONDS) -> bytes:
    """抽单声道 16-bit PCM。失败返回空 bytes。"""
    args = [
        ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{max_seconds:.1f}",
        "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    try:
        cp = subprocess.run(args, capture_output=True, timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        log.warning("PCM 抽取失败: %s", exc)
        return b""
    return cp.stdout or b""


def _envelope(pcm: bytes) -> list[float]:
    """把 PCM 变成能量包络（每跳一格的 RMS）。"""
    n = len(pcm) // 2
    if n < _FRAME:
        return []
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    out: list[float] = []
    for start in range(0, n - _FRAME, _HOP):
        chunk = samples[start:start + _FRAME]
        total = 0
        for v in chunk:
            total += v * v
        out.append(math.sqrt(total / _FRAME))
    return out


def _onset_strength(env: list[float]) -> list[float]:
    """音头强度：能量包络的正向差分（只取上升沿）。

    为什么要取正部分：能量的**下降**不代表新的击打，只有上升沿才是音头。
    这是最朴素的 onset detector，但对「有没有稳定节拍」这个问题足够。
    """
    out: list[float] = []
    for i in range(1, len(env)):
        d = env[i] - env[i - 1]
        out.append(d if d > 0 else 0.0)
    return out


def _estimate_tempo(onset: list[float], hop_sec: float) -> tuple[float | None, float]:
    """自相关估 BPM，返回 (bpm, 强度 0-1)。

    强度用「最佳 lag 的自相关值 / 所有 lag 的平均值」衡量：
    有稳定节拍时某个 lag 会明显突出，没有节拍时各 lag 差不多平。
    """
    if len(onset) < 32:
        return None, 0.0

    # 去掉直流，让自相关只反映节奏的起伏
    mean = sum(onset) / len(onset)
    sig = [v - mean for v in onset]
    norm = sum(v * v for v in sig)
    if norm <= 1e-9:
        return None, 0.0

    min_lag = max(2, int(round(60.0 / 200.0 / hop_sec)))   # 200 BPM
    max_lag = max(min_lag + 1, int(round(60.0 / 60.0 / hop_sec)))  # 60 BPM
    max_lag = min(max_lag, len(sig) // 2)
    if max_lag <= min_lag:
        return None, 0.0

    best_lag, best_val = 0, 0.0
    vals: list[float] = []
    for lag in range(min_lag, max_lag + 1):
        acc = 0.0
        for i in range(len(sig) - lag):
            acc += sig[i] * sig[i + lag]
        acc /= norm
        vals.append(acc)
        if acc > best_val:
            best_val, best_lag = acc, lag

    if best_lag == 0:
        return None, 0.0

    avg = sum(vals) / len(vals)
    strength = 0.0 if avg <= 1e-9 else max(0.0, min(1.0, (best_val - avg) / max(1e-9, best_val)))
    bpm = 60.0 / (best_lag * hop_sec)
    return round(bpm, 1), round(strength, 3)


def _onset_threshold(onset: list[float]) -> float:
    """音头判定的门槛：明显高于平均水平的那些才算。

    ⚠️ 不能用「> 0」当门槛 —— 任何有声内容几乎每个跳格都有微小上升，
    实测会数出 13 个/秒（比最快的鼓点还密一倍），这个数字没意义，
    而且会被写进提示词里（"about 13 per second"）。
    """
    if not onset:
        return 0.0
    return statistics.fmean(onset) + 1.5 * statistics.pstdev(onset)


def _onset_rate(onset: list[float], hop_sec: float, frames: int) -> float:
    """显著音头的密度（个/秒）。"""
    if not onset or frames <= 0:
        return 0.0
    thresh = _onset_threshold(onset)
    hits = sum(1 for v in onset if v > thresh)
    return round(hits / max(0.1, frames * hop_sec), 2)


def _count_transient_bursts(onset: list[float], hop_sec: float) -> int:
    """数「密集高频瞬态簇」的个数 —— 掌声/欢呼的典型特征。

    判据是**密集**：单次击打不算，短时间窗里连续出现多个音头才算。
    所以先按 0.5 秒分窗，统计每个窗里的音头数。
    """
    if not onset:
        return 0
    per_window = max(1, int(round(0.5 / hop_sec)))
    threshold = _onset_threshold(onset)
    if threshold <= 0:
        return 0

    bursts = 0
    for start in range(0, len(onset), per_window):
        hits = sum(1 for v in onset[start:start + per_window] if v > threshold)
        if hits >= 5:
            bursts += 1
    return bursts


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def analyze_music(src: str, band_levels: dict | None = None) -> MusicProfile:
    """分析音轨的节奏与频段特征。任何失败都只体现在 `note` 里，不抛异常。

    `band_levels` 是 ffmpeg 已经算好的频段能量（见 ffmpeg.analyze_audio_levels），
    传进来避免重复解码。
    """
    profile = MusicProfile()
    pcm = extract_pcm(src)
    if not pcm:
        profile.note = "音轨抽取失败，无法分析节奏"
        return profile

    env = _envelope(pcm)
    if len(env) < 8:
        profile.note = "音频过短，无法分析节奏"
        return profile

    hop_sec = _HOP / _SAMPLE_RATE
    onset = _onset_strength(env)
    bpm, strength = _estimate_tempo(onset, hop_sec)

    profile.analysed = True
    profile.bpm = bpm
    profile.beat_strength = strength
    # 强度门槛 0.25：实测稳定的舞曲在 0.4 以上，纯人声/环境声在 0.1 以下
    profile.has_beat = bool(bpm and strength >= 0.25)
    profile.onset_rate = _onset_rate(onset, hop_sec, len(env))
    profile.transient_bursts = _count_transient_bursts(onset, hop_sec)

    if band_levels:
        profile.low_energy_db = band_levels.get("low_band_db")
        profile.mid_energy_db = band_levels.get("speech_band_db")
        profile.high_energy_db = band_levels.get("high_band_db")
        mean = band_levels.get("mean_volume_db")
        peak = band_levels.get("peak_volume_db")
        if mean is not None and peak is not None:
            profile.dynamic_range_db = round(peak - mean, 1)

    profile.note = (
        f"BPM≈{bpm:.0f}（强度 {strength:.2f}）" if bpm else "未检出稳定节拍"
    )
    return profile


def isolate_vocals(src: str, dst: str) -> tuple[str | None, str]:
    """把人声频段单独抽出来，给 ASR 用。返回 (路径, 说明)。

    为什么需要：MV / 现场录音里鼓和贝斯能量很强，whisper 会被伴奏带偏，
    把歌词听成别的东西。带通滤掉 180Hz 以下和 4kHz 以上，人声的清晰度
    明显提升，而且零依赖。

    首选 Demucs（真正的音源分离，效果更好），但它要装 torch（2GB+），
    所以只在已经装好时才用；没装就走 ffmpeg 带通。

    滤波链串两遍：ffmpeg 的 highpass/lowpass 是单极点 6dB/oct，太缓，
    单极点 lowpass=f=4000 挡不住 8kHz 的镲片。
    """
    try:
        import demucs  # noqa: F401
        has_demucs = True
    except ImportError:
        has_demucs = False

    if has_demucs:
        try:
            cp = subprocess.run(
                [__import__("sys").executable, "-m", "demucs", "--two-stems", "vocals",
                 "-o", str(dst) + "_demucs", str(src)],
                capture_output=True, timeout=900,
            )
            if cp.returncode == 0:
                from pathlib import Path
                hits = list(Path(str(dst) + "_demucs").rglob("vocals.*"))
                if hits:
                    return str(hits[0]), "Demucs 人声分离"
        except Exception as exc:  # noqa: BLE001
            log.warning("Demucs 失败，回退 ffmpeg 带通: %s", exc)

    filt = (
        f"highpass=f={_VOCAL_LOW},highpass=f={_VOCAL_LOW},"
        f"lowpass=f={_VOCAL_HIGH},lowpass=f={_VOCAL_HIGH}"
    )
    try:
        cp = subprocess.run(
            [
                ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
                "-i", str(src), "-af", filt,
                "-ac", "1", "-ar", "16000", "-y", str(dst),
            ],
            capture_output=True, timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"人声频段抽取异常：{exc}"

    from pathlib import Path
    if cp.returncode == 0 and Path(dst).is_file() and Path(dst).stat().st_size > 1024:
        return str(dst), "ffmpeg 带通提取人声频段"
    return None, "人声频段抽取失败"


# ---------------------------------------------------------------------------
# 音频分析术语的清理
# ---------------------------------------------------------------------------
#
# ⚠️ 这份清单**只留在服务端**，绝不写进提示词。
#
# 踩过的坑：一开始把禁用词列在提示词里（「禁止出现 content unanalysed、
# energy distribution、voice band」），结果模型把这份清单本身抄进了输出 ——
# 「不要写 X」反而让它记住了 X。提示词治不住这类泄漏，只能输出侧强制清理。
#
# 另一个来源是我们自己的 system prompt：它里面原本就写着
# "inferred from energy distribution"，模型照抄得很自然。

_AUDIO_JARGON: tuple[str, ...] = (
    "content unanalysed",
    "unanalysed",
    "unanalyzed",
    "energy distribution",
    "voice band",
    "speech band",
    "frequency band",
    "not identified",
    "spectral",
    "low-frequency component",
    "high-frequency component",
    "band energy",
    "inferred from",
    "measured energy",
)
# 单位单独用正则匹配：dB / Hz / kHz 出现就说明在写测量值
_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:dB|Hz|kHz|dBFS)\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def sanitize_audio_jargon(text: str) -> tuple[str, list[str]]:
    """删掉泄漏到输出里的音频分析术语所在的句子，返回 (清理后文本, 命中词)。

    为什么按**句**删而不是替换词：被命中的句子整句都是分析口吻
    （"Sound present, content unanalysed"），把词挖掉只剩残句更难看。
    整句删掉，段落空了就写 N/A —— 那才是正确的表达（我们确实不知道）。
    """
    if not text:
        return text, []

    hits: list[str] = []
    kept: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        # 保留段落名行（`xxx:` 独占一行）和空行
        if not stripped or re.fullmatch(r"[a-z][a-z0-9_]*:", stripped):
            kept.append(line)
            continue

        # 逐句过滤。缩进要保留，否则会把分镜/字段的层级结构弄乱。
        indent = line[: len(line) - len(line.lstrip())]
        pieces: list[str] = []
        for sent in _SENTENCE_SPLIT.split(line.strip()):
            low = sent.lower()
            bad = [j for j in _AUDIO_JARGON if j in low]
            if not bad and _UNIT_RE.search(sent):
                bad = ["dB/Hz"]
            if bad:
                hits.extend(bad)
                continue
            pieces.append(sent)

        if pieces:
            kept.append(indent + " ".join(pieces))
        elif hits:
            # 整行都是术语，整行删掉
            continue
        else:
            kept.append(line)

    out = "\n".join(kept)
    # 段落名后面直接空了（比如 overall_soundscape: 下一行被删光）→ 补 N/A
    out = re.sub(r"(^[a-z][a-z0-9_]*:\s*)\n(?=\s*$)", r"\1 N/A\n", out, flags=re.M)
    return out, sorted(set(hits))
