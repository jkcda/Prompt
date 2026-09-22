"""自适应选帧的单元测试。

这是整个反推质量的关键模块，逻辑分支多，必须锁住行为：
  - 每帧预算不能超
  - 每个镜头至少要有一帧（否则那个镜头的信息全丢）
  - 镜头数超过预算时要均匀降采样镜头，而不是丢掉后半段
  - 单个镜头的帧数不能超过 max_per_shot
"""

from __future__ import annotations

from app.services.selection import (
    TOKENS_PER_FRAME,
    describe_plan,
    estimate_tokens,
    plan_frames,
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
    assert estimate_tokens(10) == 10 * TOKENS_PER_FRAME
    assert estimate_tokens(10, 1600) > estimate_tokens(10, 0)


def test_describe_plan_is_readable():
    shots = [(0.0, 4.0), (4.0, 7.0)]
    plan = plan_frames(shots, budget=6, max_per_shot=3)
    text = describe_plan(plan, shots)
    assert "镜头 2 个" in text
    assert "抽帧" in text
