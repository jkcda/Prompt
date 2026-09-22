"""核心链路自检（不调用模型，只验证 ffmpeg / 镜头检测 / 选帧 / 音频分析）。

用法：
    cd backend
    python -m app.selfcheck path/to/video.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .core.config import get_settings, resolve_ffmpeg, resolve_ffprobe
from .services import asr as asr_mod
from .services import ffmpeg as ff
from .services import selection


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    video = Path(argv[1])
    if not video.is_file():
        print(f"找不到文件：{video}")
        return 1

    s = get_settings()
    print("=" * 66)
    print(f"ffmpeg  : {resolve_ffmpeg()}")
    print(f"ffprobe : {resolve_ffprobe() or '(缺失，用 ffmpeg -i 解析)'}")
    print("=" * 66)

    t0 = time.time()
    info = ff.probe(video)
    print(f"\n[1] 媒体探测  ({time.time() - t0:.2f}s)")
    print(f"    时长   : {info.duration:.2f}s")
    print(f"    分辨率 : {info.width}x{info.height} @ {info.fps:.2f}fps")
    print(f"    编码   : video={info.video_codec or '-'} audio={info.audio_codec or '-'}")
    print(f"    音轨   : {'有' if info.has_audio else '无'}")

    t0 = time.time()
    # 必须和真实管线走同一条路径，否则自检结果会误导 ——
    # 踩过：自检用固定阈值检出 2 个镜头，管线因为自适应检出 4 个，
    # 用户照着自检结果调参就调错了。
    shots, cut_count, used_th, note = ff.detect_shots_adaptive(
        video, info.duration, threshold=s.scene_threshold, min_shot_seconds=s.min_shot_seconds
    )
    print(f"\n[2] 镜头检测  ({time.time() - t0:.2f}s)")
    print(f"    切换点 : {cut_count} 个")
    print(f"    阈值   : {used_th:.2f}" + (f"（{note}）" if note else ""))

    if len(shots) < 2:
        shots = ff.uniform_shots(info.duration, max(2, min(12, int(info.duration / 4) or 2)))
        print("    (切换不足，退化为均匀逻辑镜头)")
    print(f"    镜头数 : {len(shots)}")
    for i, (a, b) in enumerate(shots[:10]):
        print(f"      #{i + 1}: {a:7.2f}s - {b:7.2f}s  ({b - a:.2f}s)")

    t0 = time.time()
    plan = selection.plan_frames(
        shots,
        s.max_total_frames,
        max_per_shot=s.max_frames_per_shot,
        long_shot_seconds=s.long_shot_seconds,
        frame_interval=s.frame_interval_seconds,
    )
    print(f"\n[3] 选帧规划  ({time.time() - t0:.2f}s)")
    print(f"    {selection.describe_plan(plan, shots)}")
    print(f"    前 8 帧: {[(round(p.time, 2), p.role) for p in plan[:8]]}")

    t0 = time.time()
    pairs = ff.extract_frames_at(video, [p.time for p in plan], workers=5)
    print(f"\n[4] 抽帧  ({time.time() - t0:.2f}s)")
    print(f"    成功   : {len(pairs)}/{len(plan)}")
    if pairs:
        sizes = [p.stat().st_size for _, p in pairs]
        print(f"    单帧   : 平均 {sum(sizes) / len(sizes) / 1024:.1f}KB")
        print(f"    输出   : {pairs[0][1].parent}")

    t0 = time.time()
    audio = asr_mod.analyze_audio(video, enable_asr=False)
    print(f"\n[5] 音频分析（跳过 ASR）  ({time.time() - t0:.2f}s)")
    print(f"    有音轨 : {audio.has_audio}")
    print(f"    平均音量: {audio.mean_volume_db} dB")
    print(f"    静音占比: {audio.silence_ratio}")
    print(f"    备注   : {audio.note}")

    print("\n" + "=" * 66)
    print("核心链路自检通过。模型调用需在 .env 里配置 VLM_API_KEY 后测试。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
