"""诊断：这条视频的「像素变化」能不能用来定切点。

用法：
    python scripts/check_cut_signal.py [视频路径]

**为什么需要这个诊断**：不是所有视频都能靠像素差分定切点。
正常实拍片是「镜头内 0.01~0.03，切点处 0.4+」，有明确对比度。
但动效密集的 PV / MV 覆盖层一直在动，像素差**恒定偏高、没有基线** ——
这种情况下调阈值是白费功夫，得换思路（让模型看图判断，或用 content_hint）。

实测一支二次元 PV：全片帧间差在 0.06~0.22 之间，平均 0.154，没有低谷 →
像素法在这条素材上不可用。

原用途：验证「跨帧窗口比较能不能抓到溶解/擦除转场」——结论是不能，
因为问题不在窗口大小，在于根本没有对比度。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

FFMPEG = r"D:\提示词反推\backend\vendor\ffmpeg\ffmpeg.exe"
CLIP = (
    Path(sys.argv[1]) if len(sys.argv) > 1
    else Path(r"D:\提示词反推\data\uploads"
              r"\1790262531_3b0a1918_1790255897_3bcca161_1790172707_5.mp4")
)
FPS = 10.0
W = 160


def load_gray(path: Path, fps: float, w: int) -> np.ndarray:
    """按 fps 抽帧，缩到 w 宽，转灰度，返回 (N, H, W) uint8。"""
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-vf", f"fps={fps},scale={w}:-2,format=gray",
        "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    # 先拿一帧的尺寸
    probe = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:-2",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"],
        capture_output=True, check=True,
    ).stdout
    # PPM 头：P6\nW H\n255\n
    parts = probe.split(b"\n", 3)
    h = int(parts[1].split()[1])
    n = len(raw) // (h * w)
    return np.frombuffer(raw[: n * h * w], dtype=np.uint8).reshape(n, h, w)


def main() -> int:
    g = load_gray(CLIP, FPS, W)
    n = g.shape[0]
    print(f"抽了 {n} 帧（{FPS:.0f}fps, {W}px 宽灰度）")

    f = g.astype(np.float32) / 255.0
    d1 = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2))       # 相邻帧
    lag = int(0.3 * FPS)                                 # 跨 0.3 秒
    d3 = np.abs(f[lag:] - f[:-lag]).mean(axis=(1, 2))    # 跨 lag 帧

    def ts(i: int) -> float:
        return i / FPS

    def report(name: str, d: np.ndarray, offset: int) -> None:
        print(f"\n{name}：")
        for lo, hi in ((0.0, 4.38), (4.38, 10.80), (10.80, 15.15)):
            seg = d[int(lo * FPS) - offset: int(hi * FPS) - offset]
            if seg.size == 0:
                continue
            top = np.argsort(seg)[::-1][:5]
            times = sorted({round(ts(int(lo * FPS) + int(i)), 2) for i in top})
            print(f"  {lo:5.2f}-{hi:5.2f}s  最大 {seg.max():.4f}  中位 {np.median(seg):.4f}"
                  f"  峰在 {times}")

    report("d1 相邻帧（= ffmpeg 的做法）", d1, 1)
    report(f"d3 跨 {lag} 帧（0.3s）", d3, lag)

    # 逐秒的帧间变化 —— **这是判断像素法是否可用的关键**
    print("\n逐秒的帧间像素变化：")
    print("  时间     变化量   柱状")
    total = int(n / FPS)
    for sec in range(total):
        seg = d1[int(sec * FPS): int((sec + 1) * FPS)]
        if seg.size == 0:
            continue
        v = float(seg.mean())
        print(f"  {sec:2d}-{sec + 1:2d}s   {v:.4f}  {'#' * int(v * 120)}")
    print(f"\n  全片平均 {d1.mean():.4f}   最大 {d1.max():.4f}")
    lo, hi = float(np.percentile(d1, 5)), float(np.percentile(d1, 95))
    print(f"  5% 分位 {lo:.4f}   95% 分位 {hi:.4f}   对比度 {hi / max(lo, 1e-6):.1f}x")
    print("\n  判读：对比度 > 8x 说明「镜头内安静、切点处突跳」，像素法可用；")
    print("        对比度 < 3x 说明画面一直在动，**像素法在这条素材上定不了切点**。")

    # 用 d3 找峰：超过中位数 4 倍且是局部极大
    med = np.median(d3)
    peaks = []
    for i in range(1, len(d3) - 1):
        if d3[i] > med * 4 and d3[i] >= d3[i - 1] and d3[i] >= d3[i + 1]:
            peaks.append(round(ts(i + lag), 2))
    # 合并 0.3s 内的重复
    merged: list[float] = []
    for p in peaks:
        if not merged or p - merged[-1] > 0.3:
            merged.append(p)
    print(f"\nd3 找出的切点（中位数 × 4 以上）：{len(merged)} 个")
    if merged:
        print(f"  {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
