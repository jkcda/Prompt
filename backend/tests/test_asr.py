"""音频分析的单元测试。

重点在「什么时候**不该**去调 ASR」——踩过一个 bug：只要配了视觉模型的 key
就会拿它去调语音转写，而 ASR 地址是空的，请求直接报 URL 缺协议。
视觉模型的 key 和语音转写是两回事。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import config as cfg
from app.schemas import AudioReport
from app.services import asr


@pytest.fixture
def no_asr(monkeypatch):
    """ASR 完全没配，但视觉模型配了 —— 正是出问题的那个组合。"""
    monkeypatch.setenv("ASR_API_KEY", "")
    monkeypatch.setenv("ASR_BASE_URL", "")
    monkeypatch.setenv("VLM_API_KEY", "vlm-key-exists")
    cfg.refresh_settings()
    yield
    cfg.refresh_settings()


def test_no_asr_config_skips_transcription(sample_video: Path, no_asr, monkeypatch):
    """没配 ASR 就不该调转写，哪怕有 VLM key。"""
    called = {"n": 0}

    def spy(*a, **kw):
        called["n"] += 1
        return "", [], "", "API 转写"

    monkeypatch.setattr(asr, "_transcribe_api", spy)
    monkeypatch.setattr(asr, "_local_whisper_available", lambda: False)

    report = asr.analyze_audio(sample_video, enable_asr=True)

    assert called["n"] == 0, "没配 ASR 却去调了转写接口"
    assert report.transcript == ""
    assert "未配置 ASR" in report.note
    # 但音量/频谱这些不需要模型的东西仍然要测出来
    assert report.has_audio is True
    assert report.mean_volume_db is not None
    assert report.speech_band_db is not None


def test_asr_configured_does_call_transcription(sample_video: Path, monkeypatch):
    monkeypatch.setenv("ASR_API_KEY", "asr-key")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example.com/v1")
    cfg.refresh_settings()
    try:
        called = {"n": 0}

        def spy(path, duration):
            called["n"] += 1
            return "hello", [], "en", "API 转写（带时间戳）"

        monkeypatch.setattr(asr, "_transcribe_api", spy)
        monkeypatch.setattr(asr, "_local_whisper_available", lambda: False)

        report = asr.analyze_audio(sample_video, enable_asr=True)
        assert called["n"] == 1
        assert report.transcript == "hello"
        assert report.language == "en"
        # note 现在会带上音乐画像的结论，所以用 in 而不是 ==
        assert "API 转写" in report.note
    finally:
        monkeypatch.delenv("ASR_API_KEY", raising=False)
        monkeypatch.delenv("ASR_BASE_URL", raising=False)
        cfg.refresh_settings()


def test_enable_asr_false_skips_everything(sample_video: Path, monkeypatch):
    monkeypatch.setenv("ASR_API_KEY", "asr-key")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example.com/v1")
    cfg.refresh_settings()
    try:
        called = {"n": 0}
        monkeypatch.setattr(
            asr, "_transcribe_api",
            lambda *a, **kw: (called.__setitem__("n", called["n"] + 1), ("", [], "", ""))[1],
        )

        report = asr.analyze_audio(sample_video, enable_asr=False)
        assert called["n"] == 0
        assert "跳过" in report.note
        assert report.speech_band_db is not None, "关掉 ASR 也要测频谱"
    finally:
        monkeypatch.delenv("ASR_API_KEY", raising=False)
        monkeypatch.delenv("ASR_BASE_URL", raising=False)
        cfg.refresh_settings()


def test_video_without_audio_track(sample_video: Path, monkeypatch, tmp_path: Path):
    """无音轨时直接返回，不该往下走。"""
    import subprocess

    from app.core.config import resolve_ffmpeg

    ff = resolve_ffmpeg()
    if not ff:
        pytest.skip("无 ffmpeg")

    silent = tmp_path / "silent.mp4"
    subprocess.run([
        ff, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2",
        "-pix_fmt", "yuv420p", "-y", str(silent),
    ], check=True, capture_output=True, timeout=60)

    report = asr.analyze_audio(silent, enable_asr=True)
    assert report.has_audio is False
    assert "没有音轨" in report.note


def test_audio_report_spectrum_fields_are_filled(sample_video: Path):
    report = asr.analyze_audio(sample_video, enable_asr=False)
    assert isinstance(report, AudioReport)
    assert report.speech_band_db is not None
    assert report.low_band_db is not None
    assert report.high_band_db is not None


# ---------------------------------------------------------------------------
# 音乐画像与输出措辞
# ---------------------------------------------------------------------------

def test_music_profile_detects_tempo(ffmpeg_bin: str, tmp_path: Path):
    """有稳定节拍的音轨要能测出接近真实的 BPM。

    用 tremolo 把 100Hz 正弦调制成 128 BPM（2.133Hz）的脉冲串，
    这是「能量包络有明显周期」的最简模型。
    """
    import subprocess

    from app.services import audio_features

    clip = tmp_path / "beat.wav"
    subprocess.run([
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=100:duration=12",
        "-af", "tremolo=f=2.133:d=0.9",
        "-ac", "1", "-ar", "8000", "-y", str(clip),
    ], check=True, capture_output=True, timeout=120)

    profile = audio_features.analyze_music(str(clip))
    assert profile.analysed is True
    assert profile.bpm is not None, "有稳定节拍的音轨应该能估出 BPM"
    assert 110 <= profile.bpm <= 145, f"BPM 偏差过大：{profile.bpm}"
    assert profile.has_beat is True


def test_music_profile_silence_has_no_beat(ffmpeg_bin: str, tmp_path: Path):
    """纯静音不该被判成有节拍。"""
    import subprocess

    from app.services import audio_features

    clip = tmp_path / "silence.wav"
    subprocess.run([
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "6",
        "-y", str(clip),
    ], check=True, capture_output=True, timeout=60)

    profile = audio_features.analyze_music(str(clip))
    assert profile.has_beat is False


def test_audio_prompt_never_leaks_tool_terms(sample_video: Path):
    """音频段落的措辞会**原样进最终提示词**，绝不能出现分析术语。

    实测漏出过 `content unanalysed` / `voice band` / `energy distribution` ——
    那是我们自己的 system prompt 里的词被模型抄走了。
    """
    report = asr.analyze_audio(sample_video, enable_asr=False)
    text = asr.format_transcript_for_prompt(report).lower()

    for term in (
        "content unanalysed", "energy distribution", "voice band",
        "speech band", "not identified", "frequency band", "spectral",
        "low-frequency component",
    ):
        assert term not in text, f"音频段落里漏出了工具术语：{term}"


def test_audio_prompt_gives_human_readable_music(sample_video: Path):
    """有音轨时必须给一句人类可读的音乐描述，而不是只报「未识别」。"""
    report = asr.analyze_audio(sample_video, enable_asr=False)
    text = asr.format_transcript_for_prompt(report)

    assert "【音乐与声音" in text
    # 440Hz 正弦：能量集中在中频，描述里应该能说出这一点
    assert report.music_profile, "有音轨却没有生成音乐描述"
    assert "The soundtrack reads as" in report.music_profile


def test_audio_prompt_marks_unknown_as_unknown(sample_video: Path):
    """没做转写时必须把「不知道」写死。

    只写「未获得文本内容」这种中性陈述等于留白，模型一定会去填 ——
    实测它编出了「电子提示音与数字跳变同步」并标成 fully_copy。
    """
    report = asr.analyze_audio(sample_video, enable_asr=False)
    text = asr.format_transcript_for_prompt(report)

    assert "一无所知" in text
    assert "禁止编造" in text
    assert "不要写任何台词或歌词" in text


# ---------------------------------------------------------------------------
# 线上 API 优先于本地模型
# ---------------------------------------------------------------------------

def test_api_takes_priority_over_local_whisper(sample_video: Path, monkeypatch):
    """配了线上 ASR 就走线上，**即使本地模型可用**。

    本地模型要 460MB 权重 + 约 1.2GB 内存，而免费的线上 ASR 额度足够，
    部署轻得多。所以顺序是「线上 → 本地兜底 → 跳过」。
    """
    monkeypatch.setenv("ASR_API_KEY", "asr-key")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.example.com/v1")
    cfg.refresh_settings()
    try:
        called = {"api": 0, "local": 0}

        def api_spy(path, duration):
            called["api"] += 1
            return "from-api", [], "ja", "API 转写（带时间戳）"

        def local_spy(path):
            called["local"] += 1
            return "from-local", [], "ja"

        monkeypatch.setattr(asr, "_transcribe_api", api_spy)
        # 故意让本地「可用」—— 它仍然不该被调用
        monkeypatch.setattr(asr, "_local_whisper_available", lambda: True)
        monkeypatch.setattr(asr, "_transcribe_local", local_spy)

        report = asr.analyze_audio(sample_video, enable_asr=True)

        assert called["api"] == 1
        assert called["local"] == 0, "配了线上 API 却仍然走了本地模型"
        assert report.transcript == "from-api"
        assert report.language == "ja"
    finally:
        monkeypatch.delenv("ASR_API_KEY", raising=False)
        monkeypatch.delenv("ASR_BASE_URL", raising=False)
        cfg.refresh_settings()


def test_api_falls_back_to_json_when_verbose_json_unsupported(tmp_path: Path, monkeypatch):
    """服务商不支持 `verbose_json` 时要降级成 `json`，不能整个失败。

    实测的坑：硅基流动的 SenseVoiceSmall / TeleSpeechASR 只保证 `json`，
    要 `verbose_json` 会直接报错。降级之后没有分句时间戳（歌词只能按语义对齐），
    所以备注里必须说清楚，否则「歌词对不上镜头」会变成查不到原因的现象。
    """
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 4096)
    seen: list[str] = []

    class _Resp:
        def __init__(self, status: int, payload: dict):
            self.status_code = status
            self._payload = payload
            self.text = str(payload)

        def json(self) -> dict:
            return self._payload

    class _Client:
        def __init__(self, *a, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, url, headers=None, files=None, data=None):
            fmt = (data or {}).get("response_format", "")
            seen.append(fmt)
            if fmt == "verbose_json":
                return _Resp(400, {"error": "unsupported response_format"})
            return _Resp(200, {"text": "こんにちは"})

    monkeypatch.setattr(asr.httpx, "Client", _Client)

    text, segs, _lang, note = asr._transcribe_api(wav, 10.0)

    assert seen == ["verbose_json", "json"], f"降级顺序不对：{seen}"
    assert text == "こんにちは"
    assert segs == [], "纯文本模式本来就没有分句"
    assert "无时间戳" in note
