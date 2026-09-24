"""客观运动分析的测试。

这一层存在的理由：单帧图像里没有运动信息，模型拿不准就一律写 static。
实测一支有运镜的 MV 38 个镜头全是 static。所以这里要守住两件事：
  * 相机真的动了 → 必须测出来（否则模型没依据）
  * 相机没动、只是画面里有人在动 → 不能误报成运镜（否则会写出不存在的运镜）
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.services import motion


def _make(ffmpeg_bin: str, out: Path, vf: str, src: str = "testsrc2=size=800x450:rate=24:duration=3") -> Path:
    subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            "-vf", vf, "-t", "3", "-pix_fmt", "yuv420p", "-y", str(out),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def pan_right(ffmpeg_bin: str, tmp_path_factory) -> Path:
    """相机右摇：取景框从左往右移，画面内容因此**向左**移。"""
    d = tmp_path_factory.mktemp("motion")
    return _make(ffmpeg_bin, d / "pan_right.mp4", "crop=400:225:x='(iw-ow)*t/3':y=0")


@pytest.fixture(scope="module")
def locked_off(ffmpeg_bin: str, tmp_path_factory) -> Path:
    """三脚架固定：取景框不动，但 testsrc2 自带动态元素（相当于主体在动）。"""
    d = tmp_path_factory.mktemp("motion")
    return _make(ffmpeg_bin, d / "locked.mp4", "crop=400:225:x=100:y=50")


def test_pan_is_detected(pan_right: Path):
    m = motion.analyze_shot_motion(str(pan_right), 0.2, 2.8)

    assert m.error == "", m.error
    assert m.samples > 0
    assert not m.camera_static, "相机明显在摇，却判成了静止"
    assert m.total_shift > 10, f"累计位移太小：{m.total_shift}"
    # 取景框右移 = 内容左移，所以 u 为负
    assert m.u < 0, f"方向错了：u={m.u}"
    assert motion.suggest_camera(m).startswith("pan")


def test_locked_off_camera_is_static(locked_off: Path):
    """相机不动、画面里有东西在动 —— 运镜仍然是 static。

    这是最容易误报的一类：全局位移估计会把主体运动当成相机运动。
    实测整幅估计会报出 9.5 像素的假位移，改用四角中位数后降到 0.2 以内。
    """
    m = motion.analyze_shot_motion(str(locked_off), 0.2, 2.8)

    assert m.error == "", m.error
    assert m.camera_static, f"相机没动却判成了有运镜（total={m.total_shift}）"
    assert motion.suggest_camera(m) == "static"


def test_missing_file_returns_error_not_exception(tmp_path: Path):
    """分析失败不能抛异常 —— 它只是增强项，不该拖垮整条管线。"""
    m = motion.analyze_shot_motion(str(tmp_path / "not-here.mp4"), 0.0, 1.0)

    assert m.error
    assert m.samples == 0
    assert m.camera_static is False, "拿不到数据时不能声称静止"


def test_batch_keeps_one_result_per_shot(pan_right: Path):
    shots = [(0.0, 1.0), (1.0, 2.0), (2.0, 2.9)]
    out = motion.analyze_shots_motion(str(pan_right), shots, workers=2)

    assert len(out) == len(shots)
    assert [m.shot_index for m in out] == [0, 1, 2]
    assert all(m.samples > 0 for m in out)


def test_describe_motion_mentions_numbers_and_verdict(pan_right: Path):
    """给模型的说明要同时有「判定」和「数字」。

    只给判定（pan-left）模型会照抄成固定标签；给了数字它才能判断幅度是
    缓慢还是快速。
    """
    m = motion.analyze_shot_motion(str(pan_right), 0.2, 2.8)
    text = motion.describe_motion(m, 0)

    assert "shot 1" in text
    assert "px" in text
    assert "total movement" in text


def test_summarize_covers_every_shot(pan_right: Path):
    shots = [(0.0, 1.4), (1.4, 2.9)]
    motions = motion.analyze_shots_motion(str(pan_right), shots, workers=2)
    text = motion.summarize_for_prompt(motions)

    assert "shot 1" in text
    assert "shot 2" in text
