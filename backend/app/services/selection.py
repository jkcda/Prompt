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

from dataclasses import dataclass

# 单帧图片的 token 成本。**这是实测值，不要凭感觉写。**
#
# 实测 DeepSeek-V4.1-Flash（单帧，16:9，不同长边）：
#     448px -> 198 tokens     672px -> 198 tokens
#     896px -> 284 tokens    1344px -> 602 tokens
#
# 规律：约等于 max(下限, 像素数 / 1600)。448 和 672 相同是因为模型按 patch
# 对齐，小图有个约 200 token 的下限。
#
# ⚠️ 这里原来写的是 1100 —— 高了近 4 倍，导致 README 里「96 帧 ≈ 10.6 万 tokens」
# 这种数字虚高到离谱，用户照着它估预算会严重误判（真实约 2.7 万）。
IMAGE_TOKEN_FLOOR = 200
PIXELS_PER_TOKEN = 1600
# 默认按 16:9 估。竖屏 9:16 的像素数一样，不影响结果。
DEFAULT_ASPECT = 16 / 9


def tokens_per_frame(long_edge: int = 896, aspect: float = DEFAULT_ASPECT) -> int:
    """单帧的 token 成本估算（按长边和画幅比例）。"""
    long_edge = max(64, int(long_edge))
    aspect = aspect if aspect > 0.1 else DEFAULT_ASPECT
    # 长边固定，短边由画幅决定
    w, h = (long_edge, long_edge / aspect) if aspect >= 1 else (long_edge * aspect, long_edge)
    return max(IMAGE_TOKEN_FLOOR, int(w * h / PIXELS_PER_TOKEN))


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
    max_per_shot: int = 8,
    long_shot_seconds: float = 5.0,
    frame_interval: float = 1.0,
    min_per_shot: int = 3,
) -> list[PlannedFrame]:
    """把帧预算分配到各个镜头上。

    **每个镜头要几帧，由它的时长决定，不是由总预算决定。**

    这一点踩过坑：原来每个镜头固定最多 3 帧，导致一段 13 秒的视频
    （4 个镜头）只抽到 12 帧 —— 而预算有 48 帧，四倍没用上。
    反过来长视频里 3 帧又不够覆盖一个 8 秒的镜头。

    现在的规则是「**每秒约一帧**」：
        target[i] = clamp(round(时长 / frame_interval), min_per_shot, max_per_shot)

    15 秒 / 4 个镜头 → 每镜 3~5 帧，合计约 14 帧（原来 12 帧，且分布更合理）。
    212 秒 / 46 个镜头 → 每镜目标 5 帧，但总量受 budget 约束，自动压缩到约 2 帧。

    分配规则（按优先级）：
      1. 按「每秒一帧」算每个镜头的目标帧数；
      2. 总量没超 budget → 直接用目标值；
      3. 超了 → 每个镜头先保底 min_per_shot，剩余按镜头时长加权分配，
         且不超过各自的目标值（长镜头优先补满）；
      4. 若镜头数 × min_per_shot 都超预算（超快剪），改为按时间均匀挑镜头。
    """
    if not shots:
        return []
    budget = max(1, int(budget))
    max_per_shot = max(1, int(max_per_shot))
    min_per_shot = max(1, min(int(min_per_shot), max_per_shot))
    frame_interval = max(0.1, float(frame_interval))

    durations = [max(0.05, b - a) for a, b in shots]

    # --- 每个镜头的目标帧数：每秒约一帧，夹在 [min, max] 之间 ---
    target = [
        max(min_per_shot, min(max_per_shot, int(round(d / frame_interval)) or min_per_shot))
        for d in durations
    ]

    if sum(target) <= budget:
        alloc = target
    elif len(shots) <= budget:
        # --- 情况 B：总量超预算，但**每个镜头至少能分到 1 帧** ---
        #
        # 顺序很关键：先保证「每个镜头都被模型看到」，再加密度。
        # 反过来（先给少数镜头 3 帧、其余 0 帧）会让一部分镜头完全不可见 ——
        # 46 个镜头 / 96 帧预算时，实测旧策略有 14 个镜头拿到 0 帧。
        #
        # 加密度时按「剩余容量 × 镜头时长」加权 —— 长镜头信息量大，优先补满。
        # 用轮转（每轮每人加一帧）会让长短镜头拿到一样多，等于没加权。
        alloc = [1] * len(shots)
        remaining = budget - len(shots)

        while remaining > 0:
            weights = [
                max(0, target[i] - alloc[i]) * durations[i] for i in range(len(shots))
            ]
            total_w = sum(weights)
            if total_w <= 0:
                break

            given = 0
            shares = [
                (weights[i] / total_w * remaining) if weights[i] > 0 else 0.0
                for i in range(len(shots))
            ]
            # 先给整数部分
            for i in range(len(shots)):
                give = min(int(shares[i]), target[i] - alloc[i], remaining)
                if give > 0:
                    alloc[i] += give
                    remaining -= give
                    given += give
            # 整数部分全是 0（余量太小）时，按权重从大到小逐个补 1
            if given == 0:
                for i in sorted(range(len(shots)), key=lambda k: weights[k], reverse=True):
                    if remaining <= 0:
                        break
                    if alloc[i] < target[i]:
                        alloc[i] += 1
                        remaining -= 1
                        given += 1
            if given == 0:
                break
    else:
        # --- 情况 C：镜头数比预算还多（超快剪），只能均匀挑镜头，每镜 1 帧 ---
        picked = _evenly_pick(len(shots), budget)
        out: list[PlannedFrame] = []
        for idx in picked:
            a, b = shots[idx]
            t, role = _positions_in_shot(a, b, 1)[0]
            out.append(PlannedFrame(idx, t, role))
        out.sort(key=lambda f: f.time)
        return out

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


def estimate_tokens(
    frame_count: int, text_chars: int = 0, long_edge: int = 896
) -> int:
    """粗略估算这次请求的 token 消耗，用于预算保护与展示。

    ⚠️ 用实测公式而不是拍脑袋的常数 —— 之前那个常数高了近 4 倍。
    """
    return frame_count * tokens_per_frame(long_edge) + int(text_chars / 1.6)


def fit_budget(frame_count: int, budget: int) -> int:
    return max(1, min(frame_count, budget))


def describe_plan(
    plan: list[PlannedFrame],
    shots: list[tuple[float, float]],
    long_edge: int = 896,
) -> str:
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
        f" / 预估 {estimate_tokens(len(plan), long_edge=long_edge) / 1000:.1f}k tokens"
    )
