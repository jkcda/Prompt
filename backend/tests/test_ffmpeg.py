"""ffmpeg 服务层测试（依赖真实 ffmpeg 二进制）。

这些测试用真实视频跑，因为 ffmpeg 的参数细节（-ss 位置、scale 表达式、
showinfo 输出格式）靠 mock 是测不出来的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import ffmpeg as ff


def test_probe_reads_metadata(sample_video: Path):
    info = ff.probe(sample_video)
    assert info.has_video
    assert info.has_audio
    assert info.duration == pytest.approx(10.0, abs=0.5)
    assert info.width == 320
    assert info.height == 180
    assert info.fps == pytest.approx(24.0, abs=0.5)
    assert info.video_codec == "h264"
    assert info.size_bytes > 0


def test_probe_missing_file_does_not_raise(tmp_path: Path):
    info = ff.probe(tmp_path / "nope.mp4")
    assert info.has_video is False
    assert info.duration == 0.0


def test_detect_scene_cuts_finds_the_two_edits(sample_video: Path):
    """测试视频在第 4s 和第 7s 处有硬切。"""
    cuts = ff.detect_scene_cuts(sample_video, threshold=0.3)
    assert len(cuts) == 2
    assert cuts[0] == pytest.approx(4.0, abs=0.5)
    assert cuts[1] == pytest.approx(7.0, abs=0.5)


def test_detect_scene_cuts_returns_sorted_unique(sample_video: Path):
    cuts = ff.detect_scene_cuts(sample_video, threshold=0.1)
    assert cuts == sorted(cuts)
    assert len(cuts) == len(set(cuts))


def test_cuts_to_shots_covers_full_duration():
    shots = ff.cuts_to_shots([4.0, 7.0], duration=10.0)
    assert len(shots) == 3
    assert shots[0][0] == 0.0
    assert shots[-1][1] == pytest.approx(10.0)
    # 相邻镜头首尾相接，无缝隙无重叠
    for a, b in zip(shots, shots[1:], strict=False):
        assert a[1] == pytest.approx(b[0])


def test_cuts_to_shots_merges_too_short_shots():
    """0.2s 的快闪应被并入前一个镜头，而不是单独成镜。"""
    shots = ff.cuts_to_shots([3.0, 3.2, 8.0], duration=10.0, min_shot_seconds=0.8)
    assert len(shots) == 3
    assert all(b - a >= 0.8 for a, b in shots)


def test_cuts_to_shots_ignores_out_of_range_cuts():
    shots = ff.cuts_to_shots([-1.0, 5.0, 99.0], duration=10.0)
    assert shots[0][0] == 0.0
    assert shots[-1][1] == pytest.approx(10.0)


def test_uniform_shots_splits_evenly():
    shots = ff.uniform_shots(10.0, 5)
    assert len(shots) == 5
    assert all(b - a == pytest.approx(2.0) for a, b in shots)


def test_extract_frame_at_specific_time(sample_video: Path, tmp_path: Path):
    out = tmp_path / "f.jpg"
    got = ff.extract_frame(sample_video, 2.0, out, long_edge=320)
    assert got is not None
    assert got.is_file()
    assert got.stat().st_size > 512


def test_extract_frame_scales_long_edge(sample_video: Path, tmp_path: Path):
    """横屏视频长边应被缩到指定值。"""
    import struct

    out = tmp_path / "scaled.jpg"
    ff.extract_frame(sample_video, 1.0, out, long_edge=160)
    assert out.is_file()

    # 读 JPEG 的 SOF 段拿真实宽高
    data = out.read_bytes()
    i = 2
    w = h = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            break
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seg_len
    assert w == 160, f"长边应为 160，实际 {w}x{h}"
    assert h < w


def test_extract_frames_at_batch(sample_video: Path, tmp_path: Path):
    times = [0.5, 2.0, 4.5, 6.0, 9.0]
    pairs = ff.extract_frames_at(sample_video, times, tmp_path, workers=3)
    assert len(pairs) == 5
    assert [t for t, _ in pairs] == sorted(times)
    assert all(p.is_file() for _, p in pairs)


def test_extract_frames_at_dedupes_near_identical_times(sample_video: Path, tmp_path: Path):
    pairs = ff.extract_frames_at(sample_video, [1.0, 1.0, 1.0004, 2.0], tmp_path, workers=2)
    assert len(pairs) == 2


def test_extract_frames_at_empty_list(sample_video: Path, tmp_path: Path):
    assert ff.extract_frames_at(sample_video, [], tmp_path) == []


def test_extract_audio_produces_wav(sample_video: Path, tmp_path: Path):
    out = ff.extract_audio(sample_video, tmp_path / "a.wav")
    assert out is not None
    assert out.is_file()
    # 16kHz 单声道 16bit，10 秒 ≈ 320KB
    assert out.stat().st_size > 100_000
    assert out.read_bytes()[:4] == b"RIFF"


def test_analyze_audio_levels(sample_video: Path):
    levels = ff.analyze_audio_levels(sample_video)
    assert levels["mean_volume_db"] is not None
    assert levels["peak_volume_db"] is not None
    assert levels["silence_ratio"] is not None


def test_analyze_audio_levels_includes_band_energy(sample_video: Path):
    """频段能量是没配 ASR 时唯一能拿到的音频信息，必须测出来。

    sample_video 的音轨是 440Hz 正弦波，落在语音频段(300-3400Hz)内，
    所以 speech_band_db 应该接近 0（能量几乎全在这一频段）。
    """
    levels = ff.analyze_audio_levels(sample_video)
    assert levels["speech_band_db"] is not None
    assert levels["low_band_db"] is not None
    assert levels["speech_band_db"] > -6, f"440Hz 正弦应集中在语音频段，实测 {levels['speech_band_db']}"
    assert levels["low_band_db"] < levels["speech_band_db"], "440Hz 在 200Hz 以下应该没有能量"


def test_extract_audio_segment_isolates_time_range(sample_video: Path, tmp_path: Path):
    """Pass1 附音频时只送该分块那一段，不该送整片。"""
    seg = ff.extract_audio_segment(sample_video, 2.0, 5.0, tmp_path / "s.wav")
    assert seg is not None and seg.is_file()
    assert seg.read_bytes()[:4] == b"RIFF"
    info = ff.probe(seg)
    assert 2.5 < info.duration < 3.5, f"期望约 3 秒，实测 {info.duration}"


def test_extract_audio_segment_is_small_enough_for_api(sample_video: Path, tmp_path: Path):
    """16kHz 单声道是为了控制体积：60 秒约 1.8MB，远低于接口上限。"""
    seg = ff.extract_audio_segment(sample_video, 0.0, 9.5, tmp_path / "long.wav")
    assert seg is not None
    per_second = seg.stat().st_size / 9.5
    assert per_second * 60 < 5 * 1024 * 1024, f"60 秒会到 {per_second * 60 / 1048576:.1f}MB，太大"


def test_extract_audio_segment_clamps_negative_start(sample_video: Path, tmp_path: Path):
    seg = ff.extract_audio_segment(sample_video, -3.0, 2.0, tmp_path / "neg.wav")
    assert seg is not None and seg.is_file()


def test_segment_video_produces_chunks(sample_video: Path, tmp_path: Path):
    segs = ff.segment_video(sample_video, tmp_path, chunk_seconds=4.0, total_duration=10.0)
    assert len(segs) == 3
    for start, end, path in segs:
        assert path.is_file()
        assert end > start


def test_segment_video_empty_when_no_duration(sample_video: Path, tmp_path: Path):
    assert ff.segment_video(sample_video, tmp_path, total_duration=0.0) == []


# ---------------------------------------------------------------------------
# cleanup 必须绝对安全
# ---------------------------------------------------------------------------

def test_cleanup_survives_base_exception(monkeypatch, tmp_path: Path):
    """清理失败绝不能把服务带走。

    踩过：某些运行环境会在 shutil.rmtree 里塞删除保护，
    抛的是 SystemExit —— 它继承 BaseException 而不是 Exception，
    所以 `except OSError` 和调用方的 `except Exception` 都拦不住，
    一次清理就把 uvicorn 进程干掉了。
    """
    import shutil as _shutil

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(_shutil, "rmtree", boom)
    d = tmp_path / "junk"
    d.mkdir()
    ff.cleanup(d)  # 不抛异常即通过


def test_cleanup_survives_keyboard_interrupt(monkeypatch, tmp_path: Path):
    import shutil as _shutil

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(_shutil, "rmtree", boom)
    d = tmp_path / "junk"
    d.mkdir()
    ff.cleanup(d)


def test_cleanup_survives_permission_error(monkeypatch, tmp_path: Path):
    import shutil as _shutil

    def boom(*a, **kw):
        raise PermissionError("denied")

    monkeypatch.setattr(_shutil, "rmtree", boom)
    d = tmp_path / "junk"
    d.mkdir()
    ff.cleanup(d)


def test_ffmpeg_available():
    assert ff.ffmpeg_available() is True


def test_cleanup_removes_dir(tmp_path: Path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x")
    ff.cleanup(d)
    assert not d.exists()


def test_frame_dir_for_creates_dir():
    d = ff.frame_dir_for("testjob")
    assert d.is_dir()
    ff.cleanup(d)


# ---------------------------------------------------------------------------
# 自适应镜头检测
# ---------------------------------------------------------------------------

def test_adaptive_no_retry_for_normal_cutting(sample_video: Path):
    """正常剪辑（平均镜头 3 秒左右）不该触发重试。"""
    info = ff.probe(sample_video)
    shots, cuts, th, note = ff.detect_shots_adaptive(sample_video, info.duration)
    assert len(shots) >= 2
    assert note == "", "正常视频不该触发自适应"
    assert th == ff.get_settings().scene_threshold
    assert cuts >= 1


def test_adaptive_skips_short_videos(sample_video: Path):
    """短于 8 秒不重试 —— 短视频本来就可能只有一个镜头。"""
    shots, _, _, note = ff.detect_shots_adaptive(sample_video, 5.0)
    assert note == ""


def test_adaptive_retries_when_detection_is_coarse(monkeypatch, sample_video: Path):
    """平均镜头长度异常长 = 疑似漏检，要降阈值重试。

    实测：13.3s 特效视频在阈值 0.30 下只检出 1 个切点（2 镜头，平均 6.7s），
    降到 0.15 检出 10 个切点（4 镜头）—— 抽帧从 6 张变成 12 张。
    """
    calls: list[float] = []

    def fake_detect(src, threshold=None, max_duration=None):
        calls.append(threshold)
        if threshold == 0.30:
            return [8.27]                      # 只检出 1 个切点
        return [4.60, 5.43, 7.93, 8.27, 11.20, 12.73]   # 降阈值后检出更多

    monkeypatch.setattr(ff, "detect_scene_cuts", fake_detect)
    shots, cuts, th, note = ff.detect_shots_adaptive(
        sample_video, 13.33, threshold=0.30, min_shot_seconds=0.8
    )

    assert calls == [0.30, 0.15], "应该用配置阈值检一次，再降一半重试"
    assert len(shots) > 2, "应该采用重试结果"
    assert th == 0.15
    assert "自适应" in note


def test_adaptive_keeps_original_when_retry_is_not_better(monkeypatch, sample_video: Path):
    """重试结果没有明显更多时，保留原结果 —— 降阈值容易引入噪声切点。"""
    def fake_detect(src, threshold=None, max_duration=None):
        if threshold == 0.30:
            return [8.27]
        return [8.27, 8.30]   # 多了一个但没到 1.5 倍

    monkeypatch.setattr(ff, "detect_scene_cuts", fake_detect)
    shots, cuts, th, note = ff.detect_shots_adaptive(
        sample_video, 13.33, threshold=0.30, min_shot_seconds=0.8
    )
    assert note == "", "不该为噪声抖动换结果"
    assert th == 0.30


def test_adaptive_handles_no_cuts_at_all(monkeypatch, sample_video: Path):
    """纯色/测试图案类视频没有切点，要返回空而不是崩。"""
    monkeypatch.setattr(ff, "detect_scene_cuts", lambda *a, **kw: [])
    shots, cuts, th, note = ff.detect_shots_adaptive(sample_video, 10.0)
    assert shots == []
    assert cuts == 0
    assert note == ""


def test_adaptive_min_shot_raised_on_retry(monkeypatch, sample_video: Path):
    """降阈值重试时最短镜头时长要提到 1.0s，用来滤掉噪声切点。"""
    seen: list[float] = []
    real = ff.cuts_to_shots

    def spy(cuts, duration, min_shot_seconds=None):
        seen.append(min_shot_seconds)
        return real(cuts, duration, min_shot_seconds)

    monkeypatch.setattr(ff, "cuts_to_shots", spy)
    monkeypatch.setattr(ff, "detect_scene_cuts",
                        lambda *a, **kw: [8.27] if kw.get("threshold") == 0.30
                        else [4.6, 5.4, 7.9, 8.3, 11.2, 12.7])
    ff.detect_shots_adaptive(sample_video, 13.33, threshold=0.30, min_shot_seconds=0.5)
    assert seen[0] == 0.5
    assert seen[1] >= 1.0, "重试时应该收紧最短镜头时长"


def test_selfcheck_uses_the_same_shot_detection_as_the_pipeline():
    """自检脚本必须和真实管线走同一条路径。

    踩过：自检用固定阈值检出 2 个镜头，管线因为自适应检出 4 个 ——
    用户照自检结果调参会调错方向。诊断工具和实际行为不一致比没有更糟。
    """
    import inspect

    from app import selfcheck

    src = inspect.getsource(selfcheck)
    assert "detect_shots_adaptive" in src, "自检没走自适应检测，结果会和管线不一致"
    assert "detect_scene_cuts" not in src, "自检还在直接用固定阈值"
    # 选帧也要传 frame_interval，否则帧数和管线对不上
    assert "frame_interval=s.frame_interval_seconds" in src
