"""自适应选帧策略。

这是整个反推质量的关键。核心思路：

**不要按固定 fps 均匀抽帧**，而是
  1. 先把视频切成镜头（由 ffmpeg 场景检测给出）；
  2. 把「帧预算」按镜头时长加权分配 —— 长镜头多给几帧，短镜头少给；
  3. 每个镜头内部取 首 / 中 / 尾（带内缩），因为镜头内的信息变化主要发生在
     起幅、落幅和中段动作上，均匀铺开反而浪费。

对比「1 秒抽 10 帧」：
  - 60s 视频按 fps=10 → 600 帧 ≈ 66 万 tokens，必然超上下文；
  - 本策略 60s 视频 → 默认 48 帧 ≈ 5.3 万 tokens，安全且覆盖每个镜头。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 单帧图片（长边 ~900px）在主流多模态模型里的经验 token 成本
TOKENS_PER_FRAME = 1100


@dataclass
class PlannedFrame:
    shot_index: int
    time: float
    role: str  # head | mid | tail | uniform


def _positions_in_shot(start: float, end: float, count: int) -> list[tuple[float, str]]:
    """在镜头内部挑选 count 个时间点，返回 (时间, 角色)。

    内缩 8% 或 0.12 秒，避免取到转场混合帧/重复帧。
    """
    dur = max(0.0, end - start)
    if dur <= 0.0:
        return [(start, "mid")]

    inset = min(dur * 0.08, 0.35)
    if dur < 0.5:
        inset = 0.0

    lo, hi = start + inset, end - inset
    if hi <= lo:
        lo = hi = start + dur / 2

    if count <= 1:
        return [((lo + hi) / 2, "mid")]
    if count == 2:
        return [(lo, "head"), (hi, "tail")]
    if count == 3:
        return [(lo, "head"), ((lo + hi) / 2, "mid"), (hi, "tail")]

    out: list[tuple[float, str]] = []
    span = hi - lo
    for i in range(count):
        t = lo + span * i / (count - 1)
        if i == 0:
            role = "head"
        elif i == count - 1:
            role = "tail"
        else:
            role = "mid"
        out.append((t, role))
    return out


def plan_frames(
    shots: list[tuple[float, float]],
    budget: int,
    max_per_shot: int = 3,
    long_shot_seconds: float = 5.0,
) -> list[PlannedFrame]:
    """把帧预算分配到各个镜头上。

    分配规则（按优先级）：
      1. 每个镜头保底 1 帧 —— 保证每个镜头都被模型看到；
      2. 剩余预算按镜头时长加权分配，单个镜头不超过 max_per_shot；
      3. 时长超过 long_shot_seconds 的镜头，在预算允许时优先补到 3 帧；
      4. 若镜头数本身已超过预算（超快剪视频），改为按时间均匀降采样镜头。
    """
    if not shots:
        return []
    budget = max(1, int(budget))
    max_per_shot = max(1, int(max_per_shot))

    # --- 情况 A：镜头数 > 预算，按时间均匀挑镜头 ---
    if len(shots) > budget:
        picked = _evenly_pick(len(shots), budget)
        out: list[PlannedFrame] = []
        for idx in picked:
            a, b = shots[idx]
            t, role = _positions_in_shot(a, b, 1)[0]
            out.append(PlannedFrame(idx, t, role))
        return out

    # --- 情况 B：常规加权分配 ---
    alloc = [1] * len(shots)
    remaining = budget - len(shots)

    if remaining > 0:
        durations = [max(0.05, b - a) for a, b in shots]

        # 第一轮：给超过 long_shot_seconds 的镜头补到 3 帧（长镜头信息量大）
        order = sorted(range(len(shots)), key=lambda i: durations[i], reverse=True)
        for i in order:
            while remaining > 0 and alloc[i] < min(3, max_per_shot) and durations[i] >= long_shot_seconds:
                alloc[i] += 1
                remaining -= 1

        # 第二轮：按剩余容量按时长比例分配
        while remaining > 0:
            cap_total = sum(max(0, min(max_per_shot, 3) - alloc[i]) for i in range(len(shots)))
            if cap_total <= 0:
                # 还有余量就放宽上限
                cap_total = sum(max(0, max_per_shot - alloc[i]) for i in range(len(shots)))
                if cap_total <= 0:
                    break
                weights = [max(0, max_per_shot - alloc[i]) for i in range(len(shots))]
            else:
                weights = [max(0, min(max_per_shot, 3) - alloc[i]) for i in range(len(shots))]

            total_w = sum(weights)
            if total_w <= 0:
                break

            added = 0
            for i in range(len(shots)):
                if remaining <= 0:
                    break
                share = weights[i] / total_w * remaining
                give = min(int(math.floor(share)), weights[i], remaining)
                if give > 0:
                    alloc[i] += give
                    remaining -= give
                    added += give
            if added == 0:
                # 逐个补齐，避免浮点导致分配停滞
                for i in range(len(shots)):
                    if remaining <= 0:
                        break
                    room = max_per_shot - alloc[i]
                    if room > 0:
                        alloc[i] += 1
                        remaining -= 1

    out = []
    for i, (a, b) in enumerate(shots):
        for t, role in _positions_in_shot(a, b, alloc[i]):
            out.append(PlannedFrame(i, t, role))

    out.sort(key=lambda f: f.time)
    return out


def _evenly_pick(total: int, want: int) -> list[int]:
    if want >= total:
        return list(range(total))
    return sorted({int(i * total / want) for i in range(want)})


def estimate_tokens(frame_count: int, text_chars: int = 0) -> int:
    """粗略估算这次请求的 token 消耗，用于预算保护。"""
    return frame_count * TOKENS_PER_FRAME + int(text_chars / 1.6)


def fit_budget(frame_count: int, budget: int) -> int:
    return max(1, min(frame_count, budget))


def describe_plan(plan: list[PlannedFrame], shots: list[tuple[float, float]]) -> str:
    """生成可读的选帧说明，用于日志和前端展示。"""
    from collections import Counter

    per_shot: Counter[int] = Counter(f.shot_index for f in plan)
    roles = Counter(f.role for f in plan)
    span = 0.0
    if shots:
        span = shots[-1][1] - shots[0][0]
    return (
        f"镜头 {len(shots)} 个 / 抽帧 {len(plan)} 张 / 覆盖 {span:.1f}s"
        f" / 每镜最多 {max(per_shot.values()) if per_shot else 0} 张"
        f" / 角色分布 {dict(roles)}"
        f" / 预估 {estimate_tokens(len(plan)) / 1000:.1f}k tokens"
    )
