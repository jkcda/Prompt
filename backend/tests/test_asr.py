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
        return "", []

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
            return "hello", []

        monkeypatch.setattr(asr, "_transcribe_api", spy)
        monkeypatch.setattr(asr, "_local_whisper_available", lambda: False)

        report = asr.analyze_audio(sample_video, enable_asr=True)
        assert called["n"] == 1
        assert report.transcript == "hello"
        assert report.note == "API 转写"
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
        monkeypatch.setattr(asr, "_transcribe_api",
                            lambda *a, **kw: (called.__setitem__("n", called["n"] + 1), ("", []))[1])

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
