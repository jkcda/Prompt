"""自适应选帧的单元测试。

这是整个反推质量的关键模块，逻辑分支多，必须锁住行为：
  - 每帧预算不能超
  - 每个镜头至少要有一帧（否则那个镜头的信息全丢）
  - 镜头数超过预算时要均匀降采样镜头，而不是丢掉后半段
  - 单个镜头的帧数不能超过 max_per_shot
"""

from __future__ import annotations

from app.services.selection import (
    describe_plan,
    estimate_tokens,
    plan_frames,
    tokens_per_frame,
)


def test_empty_shots_returns_empty():
    assert plan_frames([], 48) == []


def test_budget_never_exceeded():
    shots = [(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
    plan = plan_frames(shots, budget=48, max_per_shot=3)
    assert len(plan) <= 48


def test_every_shot_gets_at_least_one_frame():
    """镜头数 ≤ 预算时，每个镜头都必须被看到。"""
    shots = [(i * 3.0, i * 3.0 + 3.0) for i in range(20)]
    plan = plan_frames(shots, budget=48, max_per_shot=3)
    covered = {f.shot_index for f in plan}
    assert covered == set(range(20))


def test_per_shot_cap_respected():
    shots = [(0.0, 60.0)]  # 一个超长镜头
    plan = plan_frames(shots, budget=48, max_per_shot=3)
    assert len(plan) == 3


def test_more_shots_than_budget_downsamples_evenly():
    """100 个镜头、预算 10 → 只能覆盖 10 个镜头，但要均匀分布而不是只取前 10 个。"""
    shots = [(i * 1.0, i * 1.0 + 1.0) for i in range(100)]
    plan = plan_frames(shots, budget=10, max_per_shot=3)
    assert len(plan) == 10
    picked = sorted(f.shot_index for f in plan)
    assert picked[0] == 0
    assert picked[-1] >= 80, "应覆盖到后段镜头，而不是只取开头"


def test_frames_are_time_sorted():
    shots = [(0.0, 4.0), (4.0, 7.0), (7.0, 10.0)]
    plan = plan_frames(shots, budget=9, max_per_shot=3)
    times = [f.time for f in plan]
    assert times == sorted(times)


def test_frames_stay_inside_their_shot():
    shots = [(0.0, 4.0), (4.0, 7.0), (7.0, 10.0)]
    plan = plan_frames(shots, budget=9, max_per_shot=3)
    for f in plan:
        start, end = shots[f.shot_index]
        assert start - 1e-6 <= f.time <= end + 1e-6


def test_roles_use_head_mid_tail():
    shots = [(0.0, 5.0)]
    plan = plan_frames(shots, budget=3, max_per_shot=3)
    roles = {f.role for f in plan}
    assert roles == {"head", "mid", "tail"}


def test_single_frame_per_shot_uses_midpoint():
    shots = [(0.0, 10.0)]
    plan = plan_frames(shots, budget=1, max_per_shot=3)
    assert len(plan) == 1
    assert 4.0 < plan[0].time < 6.0


def test_long_shots_get_priority():
    """长镜头信息量大，预算有限时应优先拿到更多帧。"""
    shots = [(0.0, 1.0), (1.0, 21.0), (21.0, 22.0)]
    plan = plan_frames(shots, budget=6, max_per_shot=3, long_shot_seconds=5.0)
    counts: dict[int, int] = {}
    for f in plan:
        counts[f.shot_index] = counts.get(f.shot_index, 0) + 1
    assert counts.get(1, 0) >= counts.get(0, 0)
    assert counts.get(1, 0) == 3


def test_degenerate_zero_length_shot():
    plan = plan_frames([(5.0, 5.0)], budget=3, max_per_shot=3)
    assert len(plan) >= 1


def test_estimate_tokens_scales_with_frames():
    per = tokens_per_frame(896)
    assert estimate_tokens(10, long_edge=896) == 10 * per


def test_tokens_per_frame_matches_measurements():
    """单帧 token 成本要用实测公式，不能拍脑袋。

    实测 DeepSeek-V4.1-Flash（16:9，单帧）：
        448px -> 198      672px -> 198      896px -> 284      1344px -> 602

    这里原来硬编码 1100，高了近 4 倍 —— README 里「96 帧 ≈ 10.6 万 tokens」
    因此虚高，用户照它估预算会严重误判（真实约 2.7 万）。
    """
    for edge, measured in ((448, 198), (672, 198), (896, 284), (1344, 602)):
        got = tokens_per_frame(edge)
        # 允许 ±25% 误差：模型按 patch 对齐，不同实现会有出入
        assert abs(got - measured) / measured < 0.25, f"{edge}px: 估 {got}，实测 {measured}"


def test_tokens_per_frame_has_a_floor():
    """小图有个约 200 token 的下限（模型按 patch 对齐）。"""
    assert tokens_per_frame(200) >= 190
    assert tokens_per_frame(448) == tokens_per_frame(672)


def test_estimate_is_not_wildly_high():
    """回归：估算不能虚高到离谱 —— 96 帧实际约 2.7 万 tokens，不是 10 万。"""
    est = estimate_tokens(96, long_edge=896)
    assert 20_000 < est < 35_000, f"96 帧估成了 {est} tokens"
    assert estimate_tokens(10, 1600) > estimate_tokens(10, 0)


def test_describe_plan_is_readable():
    shots = [(0.0, 4.0), (4.0, 7.0)]
    plan = plan_frames(shots, budget=6, max_per_shot=3)
    text = describe_plan(plan, shots)
    assert "镜头 2 个" in text
    assert "抽帧" in text


# ---------------------------------------------------------------------------
# 按镜头时长定帧数（短片真正的瓶颈）
# ---------------------------------------------------------------------------

def _counts(plan, n_shots):
    c = {i: 0 for i in range(n_shots)}
    for f in plan:
        c[f.shot_index] = c.get(f.shot_index, 0) + 1
    return c


def test_short_video_uses_duration_not_fixed_cap():
    """15 秒 / 4 个镜头应该出 15 帧左右，不是固定的 12 帧。

    踩过：原来每个镜头固定最多 3 帧，15 秒视频（4 镜）只有 12 帧，
    而预算有 48 帧 —— 四倍没用上。用户直接问「你设置了 48 帧最终结果比这少得多啊」。
    """
    shots = [(0, 5.43), (5.43, 8.37), (8.37, 11.20), (11.20, 15.0)]
    plan = plan_frames(shots, budget=96, max_per_shot=8, frame_interval=1.0)
    assert 13 <= len(plan) <= 18, f"15 秒 4 镜应该 13~18 帧，实际 {len(plan)}"
    counts = _counts(plan, 4)
    assert all(v >= 3 for v in counts.values()), f"每个镜头至少 3 帧：{counts}"
    # 5.43s 的镜头应该比 2.83s 的多
    assert counts[0] > counts[2]


def test_max_frames_per_shot_above_three_is_honoured():
    """回归：max_per_shot 大于 3 时必须真的生效。

    原来 plan_frames 和 _split_budget 里都硬编码了 min(max_per_shot, 3)，
    把配置调到 8 也没用。
    """
    shots = [(0, 10.0)]
    plan = plan_frames(shots, budget=96, max_per_shot=8, frame_interval=1.0)
    assert len(plan) == 8, f"10 秒镜头 + 上限 8 应该出 8 帧，实际 {len(plan)}"

    plan3 = plan_frames(shots, budget=96, max_per_shot=3, frame_interval=1.0)
    assert len(plan3) == 3


def test_frame_interval_controls_density():
    shots = [(0, 12.0)]
    assert len(plan_frames(shots, 96, max_per_shot=20, frame_interval=1.0)) == 12
    assert len(plan_frames(shots, 96, max_per_shot=20, frame_interval=2.0)) == 6
    assert len(plan_frames(shots, 96, max_per_shot=20, frame_interval=0.5)) == 20  # 夹在上限


def test_every_shot_covered_when_budget_allows():
    """预算够时要保证每个镜头都被看到。

    踩过：46 个镜头 / 96 帧预算时，旧策略给 32 个镜头各 3 帧，
    剩下 14 个镜头拿到 0 帧 —— 模型根本看不到它们。
    """
    shots = [(i * 4.6, (i + 1) * 4.6) for i in range(46)]
    plan = plan_frames(shots, budget=96, max_per_shot=8, frame_interval=1.0)
    counts = _counts(plan, 46)
    uncovered = [i + 1 for i, v in counts.items() if v == 0]
    assert not uncovered, f"这些镜头一帧都没分到：{uncovered}"
    assert len(plan) <= 96


def test_budget_still_bounds_long_videos():
    """长视频里预算仍然是硬上限。"""
    shots = [(i * 4.6, (i + 1) * 4.6) for i in range(46)]
    plan = plan_frames(shots, budget=48, max_per_shot=8, frame_interval=1.0)
    assert len(plan) <= 48
    counts = _counts(plan, 46)
    assert all(v >= 1 for v in counts.values()), "48 帧也够覆盖 46 个镜头"


def test_extreme_shot_count_falls_back_to_even_picking():
    """镜头数比预算还多时无解，只能均匀挑 —— 但要能跑不崩。"""
    shots = [(i * 1.5, (i + 1) * 1.5) for i in range(200)]
    plan = plan_frames(shots, budget=96, max_per_shot=8, frame_interval=1.0)
    assert len(plan) <= 96
    assert len({f.shot_index for f in plan}) == 96


def test_single_long_shot_gets_dense_frames():
    """单镜头视频（长镜头访谈）以前只能拿 3 帧，现在按秒数给。"""
    plan = plan_frames([(0, 15.0)], budget=96, max_per_shot=8, frame_interval=1.0)
    assert len(plan) == 8
    times = sorted(f.time for f in plan)
    assert times[0] < 2.0 and times[-1] > 13.0, "应该铺满整个镜头"
