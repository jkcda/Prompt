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


def test_per_shot_cap_binds_when_shots_are_many():
    """镜头多的时候，单镜上限要生效（防止少数镜头吃光预算）。

    ⚠️ 单镜场景下这个上限**故意不生效** —— 见
    test_single_shot_is_not_capped_at_max_per_shot：一镜到底时按预算放开。
    """
    shots = [(i * 4.0, (i + 1) * 4.0) for i in range(12)]
    plan = plan_frames(shots, budget=96, max_per_shot=3, frame_interval=1.0)
    counts = _counts(plan, 12)
    assert max(counts.values()) == 3, f"单镜应该正好 3 帧：{counts}"
    assert all(v >= 1 for v in counts.values()), "每个镜头都要被覆盖"


def test_single_shot_uses_the_whole_budget_when_cap_allows():
    """上限足够时，一镜到底要把预算用满（而不是固定给几帧）。"""
    plan = plan_frames([(0.0, 60.0)], budget=48, max_per_shot=48, frame_interval=1.0)
    assert len(plan) == 48, f"应该用满预算 48 帧，实际 {len(plan)}"


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
    # 用长镜头，让「每秒一帧」的期望值超过上限，上限才成为约束
    shots = [(i * 20.0, (i + 1) * 20.0) for i in range(4)]

    def max_per(plan):
        return max(_counts(plan, 4).values())

    assert max_per(plan_frames(shots, 96, max_per_shot=8, frame_interval=1.0)) == 8
    assert max_per(plan_frames(shots, 96, max_per_shot=3, frame_interval=1.0)) == 3


def test_frame_interval_controls_density():
    shots = [(0, 12.0)]
    assert len(plan_frames(shots, 96, max_per_shot=24, frame_interval=1.0)) == 12
    assert len(plan_frames(shots, 96, max_per_shot=24, frame_interval=2.0)) == 6
    # 间隔 0.5 想要 24 帧，正好等于上限
    assert len(plan_frames(shots, 96, max_per_shot=24, frame_interval=0.5)) == 24
    # 上限更小时被夹住
    assert len(plan_frames(shots, 96, max_per_shot=8, frame_interval=0.5)) == 8


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
    plan = plan_frames([(0, 15.0)], budget=96, max_per_shot=24, frame_interval=1.0)
    assert len(plan) == 15, f"15 秒单镜应该 15 帧（一秒一帧），实际 {len(plan)}"
    times = sorted(f.time for f in plan)
    assert times[0] < 2.0 and times[-1] > 13.0, "应该铺满整个镜头"


def test_estimate_tokens_counts_sheets_not_frames():
    """拼图模式下按张数算，不是按帧数 —— 一张网格装 6 帧还是 20 帧都一样。

    实测：网格 2.7Mpx -> 9.2Mpx（3.4 倍），token 只从 1156 涨到 1176。
    """
    from app.services.selection import TOKENS_PER_SHEET

    one_sheet = estimate_tokens(24, sheet_count=1)
    assert one_sheet == TOKENS_PER_SHEET, "一张网格就该按一张算"
    assert estimate_tokens(24, sheet_count=3) == 3 * TOKENS_PER_SHEET
    # 同样 24 帧，单帧模式要贵得多
    assert estimate_tokens(24) > 5 * TOKENS_PER_SHEET


def test_describe_plan_mentions_sheets():
    shots = [(0.0, 5.0), (5.0, 10.0)]
    plan = plan_frames(shots, 96, max_per_shot=8, frame_interval=1.0)
    text = describe_plan(plan, shots, 896, sheet_count=2, sheet_cells=9)
    assert "2 张网格" in text
    assert "每张 9 格" in text


def test_default_cap_gives_one_frame_per_second_for_single_shot():
    """一镜到底时按「一秒一帧」给，别被上限卡住。

    踩过：默认每镜上限是 8，于是 15 秒的连续镜头只有 8 帧（一帧管 1.9 秒），
    60 秒的只有 8 帧（一帧管 7.5 秒）—— 模型看不出中间发生了什么，
    而预算还剩一大半没用。用户反馈「单分镜图不够」。

    修法是**把默认上限调到 24**，而不是偷偷放开上限（那会让配置项失效）。
    """
    from app.core.config import get_settings

    cap = get_settings().max_frames_per_shot
    plan = plan_frames([(0.0, 15.0)], budget=96, max_per_shot=cap, frame_interval=1.0)
    assert len(plan) == 15, f"15 秒单镜应该 15 帧，实际 {len(plan)}（上限 {cap}）"


def test_cap_is_still_a_hard_limit():
    """上限是硬的 —— 设小就真的少给。别为了照顾单镜把它变成空配置。

    试过「镜头少时按 budget/镜头数 抬高上限」，结果 4 个镜头时上限被抬到 24，
    设 3 还是设 8 效果完全一样，配置项失去意义。
    """
    for cap in (3, 8, 12):
        plan = plan_frames([(0.0, 60.0)], budget=96, max_per_shot=cap, frame_interval=1.0)
        assert len(plan) == cap, f"上限 {cap} 时应该只给 {cap} 帧，实际 {len(plan)}"


def test_four_shots_behaviour_unchanged():
    """4 个镜头的常规视频行为和原来一致 —— 别为了修单镜把常规场景改坏。"""
    shots = [(0, 5.43), (5.43, 8.37), (8.37, 11.20), (11.20, 15.0)]
    plan = plan_frames(shots, budget=96, max_per_shot=24, frame_interval=1.0)
    assert len(plan) == 15
    counts = _counts(plan, 4)
    assert counts[0] == 5 and counts[3] == 4


def test_describe_plan_reports_actual_frames_not_planned():
    """说明里要报**实际**抽到的张数。

    抽帧按毫秒去重，实际可能比规划少一两张。实测出现过「界面 41 张、实际 40 张」，
    用户会按错的数字调参。
    """
    shots = [(0.0, 10.0)]
    plan = plan_frames(shots, 96, max_per_shot=24, frame_interval=0.5)
    planned = len(plan)

    same = describe_plan(plan, shots, 896, actual_frames=planned)
    assert f"抽帧 {planned} 张" in same

    fewer = describe_plan(plan, shots, 896, actual_frames=planned - 1)
    assert f"抽帧 {planned - 1} 张" in fewer

    # 不传就按规划数（自检脚本等场景）
    assert f"抽帧 {planned} 张" in describe_plan(plan, shots, 896)


def test_first_frame_of_first_shot_is_at_zero():
    """全片第一个镜头的首帧必须落在 0.0s。

    踩过：内缩 `min(dur*8%, 0.35)` 让 8 秒镜头的首帧跑到 0.35s，
    用户反馈「首帧你不拿」。第一个镜头前面没有别的镜头，不该内缩。
    """
    plan = plan_frames([(0.0, 8.0), (8.0, 12.0)], 96, max_per_shot=24, frame_interval=0.5)
    assert plan[0].time == 0.0, f"首帧应该在 0.0s，实际 {plan[0].time}"
    assert plan[0].role == "head"


def test_later_shots_still_inset_a_little():
    """后续镜头的首帧要留一点点余量，避开切点本身那一帧。"""
    plan = plan_frames([(0.0, 4.0), (4.0, 8.0)], 96, max_per_shot=24, frame_interval=0.5)
    second = [p for p in plan if p.shot_index == 1]
    assert second, "第二个镜头应该有帧"
    head = second[0]
    assert 4.0 < head.time < 4.2, f"第二镜首帧应略大于 4.0s，实际 {head.time}"


def test_two_frames_per_second_by_default():
    """默认 2fps（用户明确要求）。

    一秒一帧对快速动作/特效内容偏疏，半秒内发生的变化会整段漏掉。
    """
    from app.core.config import get_settings

    assert get_settings().frame_interval_seconds == 0.5

    plan = plan_frames([(0.0, 10.0)], 96, max_per_shot=24, frame_interval=0.5)
    assert len(plan) == 20, f"10 秒应该 20 帧（2fps），实际 {len(plan)}"
