"""镜头条目合并与特写可见性清理的测试。

这两件事都是「提示词治不住、只能用代码兜底」的：
  * 提示词里写明了「条目数 = 机位数量」并给了自检，实测照样输出 23 条交替的
    「三人跳舞」/「三人继续」；
  * 提示词里写明了「只描述本帧可见内容」，实测 `Extreme close-up` 里照样出现
    `white socks, black ankle boots`。
"""

from __future__ import annotations

from app.schemas import ShotObservation
from app.services import templates


def _shot(size: str, camera: str, action: str, start: int = 0, end: int = 1000) -> ShotObservation:
    return ShotObservation(
        shot="?", timecode="00:00.000", start_ms=start, end_ms=end,
        shot_size=size, camera=camera, action=action,
        subject="three performers", confidence=0.9,
    )


# ---------------------------------------------------------------------------
# 相邻镜头合并
# ---------------------------------------------------------------------------

def test_repeated_placeholder_entries_are_merged():
    """实测的坏输出：23 条交替的「三人跳舞」/「三人继续」。

    注意这两句互相的相似度并不高（dance vs continue），所以不能靠相似度判，
    要靠「同一句话在整段里重复了十几次」这个事实。
    """
    shots = []
    for i in range(12):
        shots.append(_shot("medium", "static", "Three performers dance.", i * 1000, (i + 1) * 1000))
        shots.append(_shot("medium", "static", "Three performers continue.", i * 1000 + 500, i * 1000 + 1500))

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 1, f"复读条目没被合并，还剩 {len(merged)} 条"
    assert merged[0].is_continuous is True
    # 动作细节要保留，不能只剩一句
    assert "dance" in merged[0].action
    assert "continue" in merged[0].action
    assert merged[0].shot == "1"


def test_distinct_actions_are_not_merged():
    """真实的不同镜头各有各的描述，不能误合并。"""
    shots = [
        _shot("medium", "static", "The lead singer raises the microphone to her lips.", 0, 1500),
        _shot("medium", "static", "A guitarist steps forward and turns toward the crowd.", 1500, 3000),
        _shot("medium", "static", "The drummer lifts both arms above the kit.", 3000, 4500),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 3
    assert all(not s.is_continuous for s in merged)


def test_different_shot_sizes_are_not_merged():
    shots = [
        _shot("wide", "static", "Three performers dance.", 0, 1000),
        _shot("close-up", "static", "Three performers continue.", 1000, 2000),
        _shot("wide", "static", "Three performers dance.", 2000, 3000),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 3, "景别不同不该合并"


def test_different_camera_moves_are_not_merged():
    shots = [
        _shot("medium", "static", "Three performers dance.", 0, 1000),
        _shot("medium", "slow dolly-in", "Three performers dance.", 1000, 2000),
        _shot("medium", "static", "Three performers dance.", 2000, 3000),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 3, "运镜不同不该合并"


def test_missing_shot_size_is_left_alone():
    """景别缺失时保守处理 —— 宁可多留一条，也不要合并掉真实的切镜。"""
    shots = [
        _shot("", "static", "Three performers dance.", 0, 1000),
        _shot("", "static", "Three performers dance.", 1000, 2000),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 2


def test_merge_takes_later_end_and_keeps_action():
    shots = [
        _shot("medium", "static", "Three performers dance.", 0, 1000),
        _shot("medium", "static", "Three performers continue.", 1000, 2600),
        _shot("medium", "static", "Three performers dance.", 2600, 3200),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 1
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 3200, "合并后的结束时间要取最后一条"


def test_extreme_closeup_and_closeup_are_different_classes():
    """极特写和特写是明显不同的构图，不能因为都含 'close' 就归成一类。"""
    shots = [
        _shot("extreme close-up", "static", "Her eyes.", 0, 1000),
        _shot("close-up", "static", "Her eyes.", 1000, 2000),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert len(merged) == 2


def test_shot_indices_are_renumbered():
    shots = [
        _shot("medium", "static", "Three performers dance.", 0, 1000),
        _shot("medium", "static", "Three performers continue.", 1000, 2000),
        _shot("wide", "pan-left", "The camera sweeps across the stage.", 2000, 3000),
    ]

    merged = templates.merge_adjacent_shots(shots)

    assert [s.shot for s in merged] == ["1", "2"]


# ---------------------------------------------------------------------------
# 特写镜头的画面外属性清理
# ---------------------------------------------------------------------------

def test_closeup_drops_off_screen_lower_body():
    """实测的坏例子：`Extreme close-up` 里写了 `white socks, black ankle boots`。"""
    shots = [ShotObservation(
        shot="1", shot_size="extreme close-up",
        subject="the lead performer's face, white socks, black ankle boots",
        action="her lips part",
    )]

    cleaned, hits = templates.strip_offscreen_attributes(shots)

    assert hits, "应该记录命中"
    assert "sock" not in cleaned[0].subject.lower()
    assert "boot" not in cleaned[0].subject.lower()
    # 画面内的内容必须留下
    assert "face" in cleaned[0].subject


def test_closeup_keeps_the_visible_half_of_a_clause():
    """"a black jacket and white socks" 只该丢后半截。"""
    shots = [ShotObservation(
        shot="1", shot_size="close-up",
        subject="a performer in a black quilted jacket and white socks",
    )]

    cleaned, _ = templates.strip_offscreen_attributes(shots)

    assert "jacket" in cleaned[0].subject
    assert "sock" not in cleaned[0].subject.lower()


def test_medium_shot_is_not_touched():
    """中景可能拍到腰以下，不该被清理。"""
    shots = [ShotObservation(
        shot="1", shot_size="medium",
        subject="a performer in a black jacket, dark trousers and boots",
    )]

    cleaned, hits = templates.strip_offscreen_attributes(shots)

    assert not hits
    assert "trousers" in cleaned[0].subject
    assert "boots" in cleaned[0].subject


def test_all_offscreen_description_is_replaced():
    """整句都是画面外属性时也不能留 —— 留一条错的比留空更糟。

    这里 subject 全是下肢词，但 action 里有上半身证据（lips），
    所以判定「主体在上半身、下肢是画面外」成立，subject 会被整句替换。
    """
    shots = [ShotObservation(
        shot="1", shot_size="extreme close-up",
        subject="white socks and black ankle boots",
        action="her lips part",
    )]

    cleaned, hits = templates.strip_offscreen_attributes(shots)

    assert hits
    assert "sock" not in cleaned[0].subject.lower()
    assert cleaned[0].subject.strip(), "字段不能变空"


def test_leg_closeup_is_not_stripped():
    """低机位拍腿的特写**不该**被清理 —— 主体本来就是下肢。

    实测踩过：一个 shot_size 是 close-up 的镜头，描述是
    「低机位拍腿和裙子」。如果只看景别就删下肢词，整个镜头描述会被删空。
    判据必须落在「主体在哪」，所以要同时有上半身证据才动手。
    """
    shots = [ShotObservation(
        shot="1", shot_size="close-up",
        subject="a low framing of the performers' legs and feet: "
                "white flared skirts, bare legs, dark plum ankle-strap shoes",
        action="the legs shift weight as they step, the white hems swinging",
    )]

    cleaned, hits = templates.strip_offscreen_attributes(shots)

    assert not hits, "主体是下肢的镜头不该被清理"
    assert "skirts" in cleaned[0].subject
    assert "hems" in cleaned[0].action


def test_registry_reference_is_not_flagged():
    """正常描述不该被误伤。"""
    shots = [ShotObservation(
        shot="1", shot_size="close-up",
        subject="the lead performer, dark eyes, long white hair falling past her shoulders",
    )]

    cleaned, hits = templates.strip_offscreen_attributes(shots)

    assert not hits
    assert cleaned[0].subject == shots[0].subject
