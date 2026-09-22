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


def test_segment_video_produces_chunks(sample_video: Path, tmp_path: Path):
    segs = ff.segment_video(sample_video, tmp_path, chunk_seconds=4.0, total_duration=10.0)
    assert len(segs) == 3
    for start, end, path in segs:
        assert path.is_file()
        assert end > start


def test_segment_video_empty_when_no_duration(sample_video: Path, tmp_path: Path):
    assert ff.segment_video(sample_video, tmp_path, total_duration=0.0) == []


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
