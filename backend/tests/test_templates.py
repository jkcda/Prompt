"""提示词模板与解析的单元测试。

Pass1 的 JSON 解析必须足够宽容——模型经常包 markdown 围栏、加前后缀、
留尾随逗号，解析失败就等于整个任务失败。
"""

from __future__ import annotations

from app.schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation
from app.services.templates import (
    FORMAT_LABELS,
    build_pass1_user,
    build_pass2_system,
    build_pass2_user,
    merge_observations,
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
    shots, notes = parse_pass1_json(GOOD_JSON)
    assert len(shots) == 1
    assert shots[0].shot_size == "medium"
    assert shots[0].dialogue == "你好"
    assert notes == "handheld throughout"


def test_parse_json_with_markdown_fence():
    shots, _ = parse_pass1_json(f"```json\n{GOOD_JSON}\n```")
    assert len(shots) == 1


def test_parse_json_with_preamble_and_trailing_text():
    raw = f"Sure, here is the analysis:\n{GOOD_JSON}\nLet me know if you need more."
    shots, _ = parse_pass1_json(raw)
    assert len(shots) == 1


def test_parse_json_with_trailing_comma():
    broken = GOOD_JSON.replace('"handheld throughout"', '"handheld throughout",')
    shots, _ = parse_pass1_json(broken)
    assert len(shots) == 1


def test_parse_garbage_returns_empty():
    shots, notes = parse_pass1_json("I cannot analyze this video.")
    assert shots == []
    assert notes == ""


def test_parse_empty_string():
    shots, _ = parse_pass1_json("")
    assert shots == []


def test_parse_ignores_unknown_fields():
    raw = '{"shots":[{"shot":"1","unknown_field":"x","action":"walks"}],"global_notes":""}'
    shots, _ = parse_pass1_json(raw)
    assert len(shots) == 1
    assert shots[0].action == "walks"


def test_parse_skips_non_dict_entries():
    raw = '{"shots":["nope",{"shot":"1","action":"walks"}],"global_notes":""}'
    shots, _ = parse_pass1_json(raw)
    assert len(shots) == 1


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
