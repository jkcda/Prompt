"""提示词模板与解析的单元测试。

Pass1 的 JSON 解析必须足够宽容——模型经常包 markdown 围栏、加前后缀、
留尾随逗号，解析失败就等于整个任务失败。
"""

from __future__ import annotations

from app.schemas import (
    AudioReport,
    ChunkObservation,
    MediaInfo,
    ShotObservation,
    SubjectEntry,
)
from app.services.templates import (
    FORMAT_LABELS,
    MODE_LABELS,
    MODE_VARIANTS,
    build_pass1_user,
    build_pass2_system,
    build_pass2_user,
    format_display,
    merge_observations,
    merge_subjects,
    mode_of,
    parse_pass1_json,
)

GOOD_JSON = """
{
  "shots": [
    {
      "shot": "1",
      "timecode": "00:00.000",
      "shot_size": "medium",
      "camera": "static",
      "subject": "a woman in a red coat",
      "action": "she turns toward the window, her coat swinging with the motion",
      "setting": "a cafe interior",
      "lighting": "warm window light from the left",
      "color": "warm, slightly desaturated",
      "motion_energy": "low",
      "on_screen_text": "none",
      "dialogue": "你好",
      "sfx": "cup clink",
      "transition": "cut",
      "confidence": 0.9
    }
  ],
  "global_notes": "handheld throughout"
}
"""


def test_parse_plain_json():
    shots, subjects, notes = parse_pass1_json(GOOD_JSON)
    assert len(shots) == 1
    assert shots[0].shot_size == "medium"
    assert shots[0].dialogue == "你好"
    assert subjects == []
    assert notes == "handheld throughout"


def test_parse_json_with_markdown_fence():
    shots, _, _ = parse_pass1_json(f"```json\n{GOOD_JSON}\n```")
    assert len(shots) == 1


def test_parse_json_with_preamble_and_trailing_text():
    raw = f"Sure, here is the analysis:\n{GOOD_JSON}\nLet me know if you need more."
    shots, _, _ = parse_pass1_json(raw)
    assert len(shots) == 1


def test_parse_json_with_trailing_comma():
    broken = GOOD_JSON.replace('"handheld throughout"', '"handheld throughout",')
    shots, _, _ = parse_pass1_json(broken)
    assert len(shots) == 1


def test_parse_garbage_returns_empty():
    shots, subjects, notes = parse_pass1_json("I cannot analyze this video.")
    assert shots == []
    assert subjects == []
    assert notes == ""


def test_parse_empty_string():
    shots, _, _ = parse_pass1_json("")
    assert shots == []


def test_parse_ignores_unknown_fields():
    raw = '{"shots":[{"shot":"1","unknown_field":"x","action":"walks"}],"global_notes":""}'
    shots, _, _ = parse_pass1_json(raw)
    assert len(shots) == 1
    assert shots[0].action == "walks"


def test_parse_skips_non_dict_entries():
    raw = '{"shots":["nope",{"shot":"1","action":"walks"}],"global_notes":""}'
    shots, _, _ = parse_pass1_json(raw)
    assert len(shots) == 1


# ---------------------------------------------------------------------------
# 主体登记表
# ---------------------------------------------------------------------------

SUBJECTS_JSON = """
{
  "shots": [{"shot": "1"}, {"shot": "2"}],
  "subjects": [
    {
      "label": "performer",
      "kind": "person",
      "description": "a performer in a dark jacket, hair tied back",
      "shots": ["1", "2"],
      "notes": "the jacket and the tied-back hair must not change"
    }
  ],
  "global_notes": ""
}
"""


def test_parse_subjects():
    _, subjects, _ = parse_pass1_json(SUBJECTS_JSON)
    assert len(subjects) == 1
    assert subjects[0].label == "performer"
    assert subjects[0].kind == "person"
    assert subjects[0].shots == ["1", "2"]
    assert "jacket" in subjects[0].notes


def test_parse_subjects_accepts_comma_string_shots():
    """模型有时把 shots 写成逗号串而不是数组，两种都要收。"""
    raw = '{"shots":[],"subjects":[{"label":"x","shots":"1, 2,3"}]}'
    _, subjects, _ = parse_pass1_json(raw)
    assert subjects[0].shots == ["1", "2", "3"]


def test_parse_subjects_accepts_chinese_comma():
    raw = '{"shots":[],"subjects":[{"label":"x","shots":"1，2"}]}'
    _, subjects, _ = parse_pass1_json(raw)
    assert subjects[0].shots == ["1", "2"]


def test_parse_subjects_drops_empty_entries():
    raw = '{"shots":[],"subjects":[{"label":"","description":"","notes":"","shots":[]}]}'
    _, subjects, _ = parse_pass1_json(raw)
    assert subjects == []


def test_parse_subjects_missing_key_is_empty():
    _, subjects, _ = parse_pass1_json(GOOD_JSON)
    assert subjects == []


def test_merge_subjects_dedupes_and_renumbers_shots():
    """跨块同一个主体只能出现一次，镜号要映射到重排后的全局镜号。"""
    c1 = ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1", subject="a performer in a dark jacket"),
               ShotObservation(shot="2", subject="a performer in a dark jacket")],
        subjects=[SubjectEntry(label="performer", kind="person",
                               description="dark jacket", shots=["1", "2"])],
    )
    c2 = ChunkObservation(
        chunk_index=1, start=10.0, end=20.0,
        shots=[ShotObservation(shot="1", subject="a performer in a dark jacket"),
               ShotObservation(shot="2", subject="rooftop vents")],
        subjects=[SubjectEntry(label="The Performer", kind="person",
                               description="dark jacket, hair tied back", shots=["1"])],
    )
    merged_shots = merge_observations([c1, c2])
    subjects = merge_subjects([c1, c2], merged_shots)

    assert len(subjects) == 1, "同一个主体被拆成了多条"
    assert subjects[0].shots == ["1", "2", "3"]
    # 更长的描述应该胜出
    assert "hair tied back" in subjects[0].description


def test_merge_subjects_falls_back_to_shot_lookup():
    """模型没给 shots 时，用主体描述回查镜号，否则写不出 appears in [Shot N]。"""
    c1 = ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1", subject="a performer in a dark jacket"),
               ShotObservation(shot="2", subject="a performer in a dark jacket")],
        subjects=[SubjectEntry(label="performer",
                               description="a performer in a dark jacket", shots=[])],
    )
    subjects = merge_subjects([c1], merge_observations([c1]))
    assert subjects[0].shots == ["1", "2"]


def test_merge_subjects_keeps_distinct_subjects_apart():
    c1 = ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1")],
        subjects=[
            SubjectEntry(label="performer", kind="person", shots=["1"]),
            SubjectEntry(label="rooftop", kind="environment", shots=["1"]),
        ],
    )
    subjects = merge_subjects([c1], merge_observations([c1]))
    assert {s.label for s in subjects} == {"performer", "rooftop"}


def test_pass1_user_lists_images_in_order():
    text = build_pass1_user(
        chunk_start=0.0,
        chunk_end=10.0,
        frame_marks=[(0.5, "head"), (5.0, "mid"), (9.5, "tail")],
        audio_text="【音频】无",
        media=MediaInfo(path="x", width=1920, height=1080, fps=30.0, duration=10.0),
        chunk_index=0,
        chunk_total=1,
    )
    assert "Image 1 -> timestamp 0.500s [role: head]" in text
    assert "Image 3 -> timestamp 9.500s [role: tail]" in text
    assert "1920x1080" in text
    assert "segment 1 of 1" in text


def test_merge_observations_renumbers_shots():
    c1 = ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1"), ShotObservation(shot="2")],
    )
    c2 = ChunkObservation(
        chunk_index=1, start=10.0, end=20.0,
        shots=[ShotObservation(shot="1"), ShotObservation(shot="2")],
    )
    merged = merge_observations([c2, c1])  # 故意乱序传入
    assert [s.shot for s in merged] == ["1", "2", "3", "4"]


def test_pass2_system_contains_hard_rules_for_every_format():
    for fmt in FORMAT_LABELS:
        system = build_pass2_system(fmt, "en")
        assert "NEVER use static or terminal verbs" in system
        assert "mouth movement" in system


def test_pass2_system_h3_uses_bare_field_names():
    """H3 的字段名必须是裸名 + 冒号，不能用尖括号标签包裹。"""
    system = build_pass2_system("h3", "en")
    assert "integrated_multimodal_description:" in system
    assert "<integrated_multimodal_description>" not in system
    assert "overall_soundscape:" in system
    assert "non_diegetic_music:" in system


def test_pass2_system_h3_ref_has_six_sections():
    system = build_pass2_system("h3-ref", "en")
    for section in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
    ):
        assert section in system


def test_pass2_system_seedance_is_chinese():
    system = build_pass2_system("seedance", "en")
    assert "Seedance" in system
    assert "主体" in system
    assert "无对白" in system


def test_pass2_user_includes_audio_and_shots():
    obs = [ChunkObservation(
        chunk_index=0, start=0.0, end=5.0,
        shots=[ShotObservation(shot="1", timecode="00:00.000", action="she sings")],
        global_notes="warm tone",
    )]
    audio = AudioReport(
        has_audio=True,
        segments=[],
        transcript="la la la",
        mean_volume_db=-18.0,
        silence_ratio=0.1,
    )
    text = build_pass2_user(
        observations=obs,
        audio=audio,
        media=MediaInfo(path="x", duration=5.0, width=1280, height=720, fps=24.0, has_audio=True),
        shots_summary="1 个镜头",
    )
    assert "she sings" in text
    assert "la la la" in text
    assert "ALL 1 observed shots" in text
    assert "warm tone" in text


def test_pass2_user_includes_extra_instruction():
    obs = [ChunkObservation(chunk_index=0, start=0.0, end=5.0, shots=[ShotObservation(shot="1")])]
    text = build_pass2_user(
        observations=obs,
        audio=AudioReport(),
        media=None,
        shots_summary="1 个镜头",
        extra_instruction="重点描述运镜",
    )
    assert "重点描述运镜" in text


# ---------------------------------------------------------------------------
# 两种模式：H3 / Seedance
# ---------------------------------------------------------------------------

def test_mode_of_groups_variants():
    assert mode_of("h3") == "h3"
    assert mode_of("h3-ref") == "h3"
    assert mode_of("seedance") == "seedance"
    assert mode_of("generic") == "generic"
    assert mode_of("nonsense") == "generic", "未知格式不该抛异常"


def test_only_two_primary_modes():
    """对外只暴露 H3 与 Seedance 两种模式，generic 是兜底。"""
    from app.services.templates import MODE_PRIMARY
    assert [m for m, p in MODE_PRIMARY.items() if p] == ["h3", "seedance"]
    assert MODE_PRIMARY["generic"] is False


def test_every_mode_has_exactly_one_default_variant():
    for mode, variants in MODE_VARIANTS.items():
        defaults = [v for v, is_default in variants if is_default]
        assert len(defaults) == 1, f"{mode} 的默认变体不是唯一一个"
        assert all(v in FORMAT_LABELS for v, _ in variants)


def test_format_display_reads_naturally():
    assert format_display("h3-ref") == "H3 模式 · Ref2VA 六段式（带参考素材）"
    assert format_display("seedance") == "Seedance 模式 · 六要素中文段"
    # 单变体模式下不重复前缀
    assert format_display("generic") == MODE_LABELS["generic"]


def test_no_unsubstituted_placeholders_in_any_system_prompt():
    """占位符漏给模型会让它照着字面理解，必须每个格式都替换干净。"""
    for fmt in FORMAT_LABELS:
        for lang in ("en", "zh"):
            system = build_pass2_system(fmt, lang)
            assert "{language_instruction}" not in system
            assert "{common}" not in system


def test_h3_t2va_forbids_reference_labels():
    """T2VA 没有参考素材，出现 <Subject N> 会被当成字面文本写进画面。"""
    system = build_pass2_system("h3", "en")
    assert "no reference assets" in system.lower()
    assert "Never write" in system
    assert "<Subject 1>" in system  # 出现在禁令里


def test_h3_ref_uses_subject_registry_as_source_of_labels():
    system = build_pass2_system("h3-ref", "en")
    assert "SUBJECT REGISTRY" in system
    assert "do not invent" in system


def test_seedance_mode_has_no_h3_markup():
    """Seedance 是连贯中文段落，混进字段名或 <Subject N> 就是错的。"""
    system = build_pass2_system("seedance", "en")
    assert "Seedance" in system
    assert "Do NOT use field labels" in system
    assert "Chinese only" in system
    assert "节拍" in system
    # 三字段名只能出现在禁令语境里，不能作为输出结构
    assert "integrated_multimodal_description:" not in system
    assert "overall_soundscape:" not in system


def test_seedance_beat_map_must_cover_every_shot():
    system = build_pass2_system("seedance", "en")
    assert "EVERY shot" in system


def test_pass2_user_renders_subject_registry():
    obs = [ChunkObservation(chunk_index=0, start=0.0, end=5.0,
                            shots=[ShotObservation(shot="1")])]
    subjects = [
        SubjectEntry(label="performer", kind="person",
                     description="dark jacket", shots=["1", "3"],
                     notes="hair stays tied back"),
        SubjectEntry(label="rooftop", kind="environment",
                     description="industrial roof at dusk", shots=["1"]),
    ]
    text = build_pass2_user(
        observations=obs,
        audio=AudioReport(),
        media=None,
        shots_summary="3 个镜头",
        subjects=subjects,
        fmt="h3-ref",
    )
    assert "=== SUBJECT REGISTRY" in text
    assert "1. performer (person) - [Shot 1], [Shot 3]" in text
    assert "2. rooftop (environment) - [Shot 1]" in text
    assert "hair stays tied back" in text
    assert "H3 模式" in text, "应该标明目标模式"


def test_pass2_user_omits_registry_when_empty():
    obs = [ChunkObservation(chunk_index=0, start=0.0, end=5.0,
                            shots=[ShotObservation(shot="1")])]
    text = build_pass2_user(
        observations=obs, audio=AudioReport(), media=None,
        shots_summary="1 个镜头", subjects=[],
    )
    assert "SUBJECT REGISTRY" not in text


def test_common_rules_forbid_wording_drift():
    """同一个主体在不同镜头换措辞，模型会当成两个人。"""
    system = build_pass2_system("h3", "en")
    assert "IDENTICAL across shots" in system
