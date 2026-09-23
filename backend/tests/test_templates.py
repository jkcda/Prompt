"""提示词模板与解析的单元测试。

Pass1 的 JSON 解析必须足够宽容——模型经常包 markdown 围栏、加前后缀、
留尾随逗号，解析失败就等于整个任务失败。
"""

from __future__ import annotations

import json

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
    PASS1_SYSTEM,
    build_compress_system,
    build_compress_user,
    build_pass1_user,
    build_pass2_system,
    build_pass2_user,
    check_prompt_integrity,
    format_display,
    merge_observations,
    merge_subjects,
    mode_of,
    parse_pass1_json,
    prompt_sections,
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



def _unpack(raw: str):
    """把 Pass1Parse 拆成 (shots, subjects, notes) —— 沿用原来的断言写法。"""
    r = parse_pass1_json(raw)
    return r.shots, r.subjects, r.global_notes

def test_parse_plain_json():
    shots, subjects, notes = _unpack(GOOD_JSON)
    assert len(shots) == 1
    assert shots[0].shot_size == "medium"
    assert shots[0].dialogue == "你好"
    assert subjects == []
    assert notes == "handheld throughout"


def test_parse_json_with_markdown_fence():
    shots, _, _ = _unpack(f"```json\n{GOOD_JSON}\n```")
    assert len(shots) == 1


def test_parse_json_with_preamble_and_trailing_text():
    raw = f"Sure, here is the analysis:\n{GOOD_JSON}\nLet me know if you need more."
    shots, _, _ = _unpack(raw)
    assert len(shots) == 1


def test_parse_json_with_trailing_comma():
    broken = GOOD_JSON.replace('"handheld throughout"', '"handheld throughout",')
    shots, _, _ = _unpack(broken)
    assert len(shots) == 1


def test_parse_garbage_returns_empty():
    shots, subjects, notes = _unpack("I cannot analyze this video.")
    assert shots == []
    assert subjects == []
    assert notes == ""


def test_parse_empty_string():
    shots, _, _ = _unpack("")
    assert shots == []


def test_parse_ignores_unknown_fields():
    raw = '{"shots":[{"shot":"1","unknown_field":"x","action":"walks"}],"global_notes":""}'
    shots, _, _ = _unpack(raw)
    assert len(shots) == 1
    assert shots[0].action == "walks"


def test_parse_skips_non_dict_entries():
    raw = '{"shots":["nope",{"shot":"1","action":"walks"}],"global_notes":""}'
    shots, _, _ = _unpack(raw)
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
    _, subjects, _ = _unpack(SUBJECTS_JSON)
    assert len(subjects) == 1
    assert subjects[0].label == "performer"
    assert subjects[0].kind == "person"
    assert subjects[0].shots == ["1", "2"]
    assert "jacket" in subjects[0].notes


def test_parse_subjects_accepts_comma_string_shots():
    """模型有时把 shots 写成逗号串而不是数组，两种都要收。"""
    raw = '{"shots":[],"subjects":[{"label":"x","shots":"1, 2,3"}]}'
    _, subjects, _ = _unpack(raw)
    assert subjects[0].shots == ["1", "2", "3"]


def test_parse_subjects_accepts_chinese_comma():
    raw = '{"shots":[],"subjects":[{"label":"x","shots":"1，2"}]}'
    _, subjects, _ = _unpack(raw)
    assert subjects[0].shots == ["1", "2"]


def test_parse_subjects_drops_empty_entries():
    raw = '{"shots":[],"subjects":[{"label":"","description":"","notes":"","shots":[]}]}'
    _, subjects, _ = _unpack(raw)
    assert subjects == []


def test_parse_subjects_missing_key_is_empty():
    _, subjects, _ = _unpack(GOOD_JSON)
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
    assert format_display("h3-ref") == "H3 模式 · Ref2VA 六段式（格式参考）"
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


def test_h3_ref_does_not_reference_the_source_video():
    """Ref2VA 的「参考」是**格式**参考，不是让你参考原视频。

    原视频只用来提取主体 —— 定义成 <Subject N> 供用户自己挂参考图。
    出现 <Video 1> / <Audio 1> 是错的：那会把生成结果绑死在原片上，
    而用户要的是「用我自己的参考图生成」。
    """
    system = build_pass2_system("h3-ref", "en")
    assert "FORMAT, not the source video" in system
    assert "Do NOT define or mention `<Video 1>`" in system
    assert "Do not add a `<Video 1>` or `<Audio 1>` line" in system
    assert "no video reference and no audio reference" in system
    # 只允许在禁令语境里出现这两个标签名
    assert "supply their OWN reference images" in system


def test_h3_ref_retention_analysis_excludes_video_and_audio_lines():
    system = build_pass2_system("h3-ref", "en")
    assert "one line per `<Subject N>` label ONLY" in system
    assert "no video or audio line" in system


def test_word_limit_is_per_section_not_just_a_global_number():
    """长度必须**逐段**给预算，不能只给一个全局上限。

    踩过：原来只写「keep the description under 700 words」，
    实测模型写出 779 词正文、整篇 1795 词 / 11314 字符 —— 视频模型吃不下。
    模型不会自己把全局上限分配到各段，得逐段给数字。
    """
    system = build_pass2_system("h3", "en")
    assert "420 words or fewer" in system
    assert "LENGTH BUDGET" in system

    ref = build_pass2_system("h3-ref", "en")
    assert "under 700 words" in ref, "整篇上限要写清楚"
    assert "420 words" in ref
    assert "at most 6 entries" in ref, "subject_definitions 要限条数"
    assert "15 words each" in ref
    assert "40 words" in ref
    assert "25 words" in ref
    # 旧的全局写法不该再出现
    for sysp in (system, ref):
        assert "350-500 words" not in sysp


def test_h3_ref_tells_which_subjects_to_drop_when_too_many():
    """登记表超过 6 条时要给出取舍优先级，否则模型会平均用力写得又臭又长。"""
    system = build_pass2_system("h3-ref", "en")
    assert "people > wardrobe/props > environment > style/grade" in system
    assert "Drop the least important ones entirely" in system


def test_h3_ref_says_where_to_cut_when_over_budget():
    """超预算时先砍定义和分析，绝不砍正文 —— 正文才是生成要用的。"""
    system = build_pass2_system("h3-ref", "en")
    assert "never from `detailed_description`" in system


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


def test_pass1_also_forbids_exclusive_motion_claims():
    """Pass1 的 global_notes 会原样进 Pass2，禁忌措辞不能只堵一处。

    真实模型在 Pass1 写下过 "The only motion is the shifting rainbow gradient bar"，
    而那句会经 build_pass2_user 的 `segment notes` 直接注入 Pass2 的输入。
    """
    from app.services.templates import PASS1_SYSTEM

    assert "ONLY motion" in PASS1_SYSTEM
    assert "global_notes" in PASS1_SYSTEM and "verbatim" in PASS1_SYSTEM


def test_pass1_forbids_static_verbs_too():
    from app.services.templates import PASS1_SYSTEM

    for word in ("holds", "remains still", "hands rest"):
        assert word in PASS1_SYSTEM, f"Pass1 里缺少对 {word} 的禁令"


def test_pass1_asks_for_subject_registry():
    from app.services.templates import PASS1_SYSTEM

    assert '"subjects"' in PASS1_SYSTEM
    assert "SUBJECT REGISTRY" in PASS1_SYSTEM
    assert "must be ONE entry" in PASS1_SYSTEM


def test_pass1_puts_subjects_before_shots_in_the_shape():
    """JSON 形状里 subjects 要排在 shots 前面。

    模型是按顺序填的，放最后容易在输出预算用尽时被省掉 ——
    实测 DeepSeek-V4.1-Flash 就是这样漏掉了整个 subjects 数组。
    """
    from app.services.templates import PASS1_SYSTEM

    assert PASS1_SYSTEM.index('"subjects": [') < PASS1_SYSTEM.index('"shots": [')
    assert "INCOMPLETE" in PASS1_SYSTEM, "要明确说缺了会被拒"
    assert "BEFORE" in PASS1_SYSTEM and "run out of output budget" in PASS1_SYSTEM


def test_pass1_excludes_watermarks_from_subject_registry():
    """水印/台标/UI 不该进登记表。

    实测 DeepSeek 把 bilibili-watermark 登记成了主体。虽然 Pass2 自己
    过滤掉了没写进提示词，但登记表本身不该收 —— 万一模型照抄，
    生成的视频里就会出现别人的水印。
    """
    from app.services.templates import PASS1_SYSTEM

    assert "Do NOT register watermarks" in PASS1_SYSTEM
    assert "platform logos" in PASS1_SYSTEM
    assert "never in `subjects`" in PASS1_SYSTEM


# ---------------------------------------------------------------------------
# 音频未知时禁止编造
# ---------------------------------------------------------------------------

def test_audio_report_forbids_fabrication_when_not_transcribed():
    """真实模型在 enable_asr=False 时编出过「电子提示音与数字跳变同步」。

    只写一句中性的「未获得文本内容」，模型会照着画面把声音补齐。
    必须显式禁止，否则反推出来的 BGM / 台词全是假的。
    """
    from app.services.asr import format_transcript_for_prompt

    report = AudioReport(has_audio=True, mean_volume_db=-21.0, silence_ratio=0.0)
    text = format_transcript_for_prompt(report)

    assert "未知" in text
    assert "禁止描述任何具体的声音" in text
    assert "不要写台词或歌词" in text
    assert "不要写 BGM" in text
    assert "N/A" in text
    # 音量信息仍然要保留，这是唯一已知的
    assert "-21.0dB" in text


def test_audio_report_with_transcript_does_not_warn():
    from app.services.asr import format_transcript_for_prompt

    report = AudioReport(has_audio=True, transcript="la la la")
    text = format_transcript_for_prompt(report)
    assert "la la la" in text
    assert "禁止描述任何具体的声音" not in text
    assert "【重申】" not in text


def test_audio_report_no_track_says_na():
    from app.services.asr import format_transcript_for_prompt

    text = format_transcript_for_prompt(AudioReport(has_audio=False))
    assert "无音轨" in text
    assert "N/A" in text


def test_pass2_rules_forbid_fabricated_sound():
    system = build_pass2_system("h3", "en")
    assert "not transcribed" in system
    assert "Fabricated sound is worse than an empty field" in system
    assert "not even hedged" in system, "要堵住「似乎/仿佛」这种模糊编造"


def test_pass2_rules_forbid_inferring_sound_from_visuals():
    """看到人走路就写 footsteps，本质还是编声音——真实模型这么写过。"""
    system = build_pass2_system("h3", "en")
    assert "Do NOT convert visual events into sound events" in system
    assert "footsteps" in system and "cloth rustle" in system
    assert "leave the sound unspecified" in system


def test_pipeline_audio_slice_warns_when_no_transcript():
    from app.services.pipeline import _audio_slice

    text = _audio_slice(AudioReport(has_audio=True), 0.0, 10.0)
    assert "音频内容未知" in text
    assert "禁止写台词" in text


def test_pipeline_audio_slice_no_track():
    from app.services.pipeline import _audio_slice

    text = _audio_slice(AudioReport(has_audio=False), 0.0, 10.0)
    assert "没有音轨" in text
    assert "N/A" in text


# ---------------------------------------------------------------------------
# 频谱特征（无 ASR 时唯一可用的音频信息）
# ---------------------------------------------------------------------------

def test_spectrum_speech_dominant():
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-1.0, low_band_db=-25.0))
    joined = "\n".join(lines)
    assert "能量高度集中在这一频段" in joined
    assert "频谱无法区分" in joined, "纯音调也会落在这个频段，必须说明分不出来"
    assert "低频很弱" in joined
    # 不能升级成内容断言
    assert "有人说话" not in joined
    assert "对白为主" not in joined


def test_spectrum_music_like():
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-12.0, low_band_db=-4.0))
    joined = "\n".join(lines)
    assert "不太像以人声为主" in joined
    assert "疑似有节奏性的音乐编排" in joined
    assert "但也可能是低频环境噪声" in joined, "低频强不等于一定有鼓点"


def test_spectrum_mixed_case():
    """典型情况：人声 + 配器，两个频段都有能量。"""
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-4.5, low_band_db=-4.3))
    joined = "\n".join(lines)
    assert "混有其他频段成分" in joined
    assert "人声 + 配器" in joined


def test_spectrum_missing_values_are_skipped():
    from app.services.asr import describe_spectrum

    assert describe_spectrum(AudioReport(has_audio=True)) == []


def test_spectrum_is_included_in_prompt_but_bounded():
    """频谱数据可以给，但必须标明边界——它是能量分布，不是内容识别。"""
    from app.services.asr import format_transcript_for_prompt

    report = AudioReport(has_audio=True, speech_band_db=-1.2, low_band_db=-19.0)
    text = format_transcript_for_prompt(report)
    assert "音频频谱特征" in text
    assert "不是内容识别" in text
    assert "不得据此断言具体内容" in text
    # 禁令仍然在
    assert "禁止描述任何具体的声音" in text


def test_pass2_rule_distinguishes_content_from_spectrum():
    """规则要允许频谱倾向描述，否则和频谱数据自相矛盾。"""
    system = build_pass2_system("h3", "en")
    assert "distinguish CONTENT from SPECTRUM" in system
    assert "voice-dominant" in system
    assert "does not license" in system


# ---------------------------------------------------------------------------
# 超长压缩
# ---------------------------------------------------------------------------

LONG_PROMPT = """subject_definitions:
<Subject 1> - a very detailed long winded description of the boy with many redundant adjectives.
<Subject 2> - another extremely verbose definition full of filler words and padding.

summary:
[reference generation] A long summary that goes on and on about the source clip.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - lots of redundant justification here.
<Subject 2> (appears in [Shot 2]): fully_preserved - more redundant justification here.

detailed_description:
[Shot 1] A very long description with many unnecessary adjectives and adverbial padding.
[Shot 2] At 00:03.400, another long description that repeats the style once again.

overall_soundscape:
A long winded ambience description.

non_diegetic_music:
N/A"""


def test_compress_instruction_is_short_and_direct():
    """压缩指令要短而直接 —— 长篇指令实测反而让模型原样返回。

    踩过：写了一大段「MUST preserve / Cut aggressively」的指令，
    模型直接原样返回（833 词 → 833 词，字节相同）。
    换成一句「Shorten the prompt the user sends to under N words」+ 两条 keep/cut 就管用。
    """
    system = build_compress_system(700)
    assert "under" in system and "words" in system
    assert "<Subject N>" in system
    assert "[Shot N]" in system
    assert "verbatim" in system, "台词必须逐字保留"
    assert len(system.split()) < 100, f"指令太长（{len(system.split())} 词），模型会偷懒"


def test_compress_targets_below_the_limit_to_leave_headroom():
    """目标值要比真实上限更紧。

    实测说「压到 700 以内」，模型压到 803 就收手（2297 → 803，砍了 65% 但还是超）。
    让它瞄 0.85 倍，落点才在上限之内。
    """
    system = build_compress_system(700)
    assert "under 595 words" in system, "700 × 0.85 = 595"
    assert "below 700" in system, "真实上限也要写清楚"

    user = build_compress_user("a " * 2000, 700)
    assert "2000 words" in user
    assert "Cut at least 1405 words" in user, "2000 - 595 = 1405"
    assert "595 words or fewer" in user
    assert "hard ceiling 700" in user


def test_compress_does_not_mention_a_fixed_six_section_list():
    """压缩指令不能写死 Ref2VA 的六段名 —— T2VA / Seedance 会被带偏。"""
    system = build_compress_system(700)
    assert "subject_definitions" not in system
    assert "the section names present in the original" in system


def test_compress_uses_thinking():
    """压缩要开思考 —— 这是与其他环节相反的例外。

    实测 833 词的六段式：
      关思考 -> 759 词（只砍 74）
      开思考 -> 581 词（砍 252，达标）
    编辑需要先想清楚哪些能砍。有测试断言调用处传了 disable_thinking=False。
    """
    import inspect

    from app.services import pipeline

    src = inspect.getsource(pipeline.run_pipeline)
    assert "disable_thinking=False" in src
    # 而且要在压缩那一段里，不能是别的地方
    idx = src.index("build_compress_system")
    assert "disable_thinking=False" in src[idx:idx + 900]


def test_compress_user_includes_word_count():
    text = build_compress_user("a b c d e", 100)
    assert "5 words" in text
    assert "85 words or fewer" in text, "100 × 0.85 = 85"
    assert "hard ceiling 100" in text
    assert "--- BEGIN PROMPT ---" in text and "--- END PROMPT ---" in text


def test_integrity_accepts_good_compression():
    ok, why = check_prompt_integrity(LONG_PROMPT, LONG_PROMPT)
    assert ok, why


def test_integrity_rejects_missing_section():
    broken = LONG_PROMPT.replace("overall_soundscape:", "audio_notes:")
    ok, why = check_prompt_integrity(LONG_PROMPT, broken)
    assert not ok
    assert "overall_soundscape" in why


def test_integrity_rejects_dropped_subject():
    """压缩时把主体合并/删掉比超长更糟 —— 用户没法挂参考图了。"""
    broken = LONG_PROMPT.replace("<Subject 2>", "<Subject 1>")
    ok, why = check_prompt_integrity(LONG_PROMPT, broken)
    assert not ok
    assert "主体标签丢失" in why


def test_integrity_rejects_dropped_shot():
    broken = LONG_PROMPT.replace("[Shot 2]", "[Shot 1]")
    ok, why = check_prompt_integrity(LONG_PROMPT, broken)
    assert not ok
    assert "镜头标记丢失" in why


def test_integrity_rejects_dropped_na():
    broken = LONG_PROMPT.replace("N/A", "none")
    ok, why = check_prompt_integrity(LONG_PROMPT, broken)
    assert not ok
    assert "N/A" in why


def test_integrity_rejects_empty():
    ok, why = check_prompt_integrity(LONG_PROMPT, "   ")
    assert not ok
    assert "为空" in why


# T2VA 三字段格式（和 Ref2VA 六段完全不同）
T2VA_PROMPT = """integrated_multimodal_description:
A neon-soaked anime sequence, static camera. [Shot 1] Close-up of a girl's eye,
iridescent iris, black choker. [Shot 2] At 00:03.400, the frame tears into
glitch bands, saturated halftone dots flooding the left half.

overall_soundscape:
A dense synth pad with a rising sweep and no silent gap.

non_diegetic_music:
N/A"""


def test_integrity_uses_sections_from_the_original_not_a_fixed_list():
    """段落清单必须从原文里取，不能写死成 Ref2VA 的六段。

    踩过：校验里硬编码了 subject_definitions / summary / retention_analysis /
    detailed_description / overall_soundscape / non_diegetic_music，
    于是 **T2VA 三字段格式的压缩永远被判为「缺少段落」**——
    实测 T2VA 从 1175 词压到 487 词，成果全被丢掉，压缩功能对非 Ref2VA
    格式等于不存在。而 T2VA 恰恰是默认格式。
    """
    ok, why = check_prompt_integrity(T2VA_PROMPT, T2VA_PROMPT)
    assert ok, f"T2VA 自己和自己比都不通过：{why}"


def test_t2va_compression_is_accepted():
    """T2VA 压短后应当被接受（只要三字段都在）。"""
    shorter = """integrated_multimodal_description:
Neon anime sequence, static camera. [Shot 1] Girl's eye, iridescent iris.
[Shot 2] At 00:03.400, glitch bands and halftone dots flood the frame.

overall_soundscape:
Dense synth pad, rising sweep, no silence.

non_diegetic_music:
N/A"""
    assert len(shorter.split()) < len(T2VA_PROMPT.split())
    ok, why = check_prompt_integrity(T2VA_PROMPT, shorter)
    assert ok, why


def test_t2va_compression_rejected_if_a_field_is_dropped():
    broken = T2VA_PROMPT.replace("non_diegetic_music:", "music_notes:")
    ok, why = check_prompt_integrity(T2VA_PROMPT, broken)
    assert not ok
    assert "non_diegetic_music" in why


def test_prompt_sections_extracts_bare_names():
    assert prompt_sections(T2VA_PROMPT) == [
        "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"
    ]
    assert prompt_sections("no sections here at all") == []
    # 行内有内容的冒号不算段落名
    assert prompt_sections("note: something inline") == []


def test_integrity_ignores_subject_and_shot_checks_for_t2va():
    """T2VA 没有 <Subject N>，不该因为「主体标签变少」被拒（本来就是 0 个）。"""
    shorter = (
        "integrated_multimodal_description:\n"
        "[Shot 1] Girl's eye. [Shot 2] At 00:03.400, glitch bands.\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )
    ok, why = check_prompt_integrity(T2VA_PROMPT, shorter)
    assert ok, f"T2VA 没有主体标签，不该做主体校验：{why}"


# ---------------------------------------------------------------------------
# 用户画面说明（content_hint）
# ---------------------------------------------------------------------------

def _pass1(hint: str = "") -> str:
    return build_pass1_user(
        chunk_start=0.0, chunk_end=10.0,
        frame_marks=[(0.5, "head"), (5.0, "mid")],
        audio_text="【音频】无转写。", media=None,
        chunk_index=0, chunk_total=1, content_hint=hint,
    )


def test_content_hint_is_injected_into_pass1():
    """用户的画面说明要进观察阶段。

    静态帧判断不出「一镜到底还是多镜头切换」「这是什么作品/角色」，
    而这些直接影响产出质量。用户反馈「ai无法认出视频是切镜头还是一镜到底，
    所以还是需要人工提示词辅助」。
    """
    hint = "一镜到底的跟拍运镜，全程没有切镜；主角是白发少女，穿黑色风衣"
    text = _pass1(hint)
    assert hint in text
    assert "CONTEXT FROM THE PERSON WHO SUBMITTED" in text
    assert "one continuous take or has cuts" in text, "要点明它能补充镜头结构这类信息"


def test_content_hint_absent_when_empty():
    text = _pass1("")
    assert "CONTEXT FROM THE PERSON WHO SUBMITTED" not in text


def test_content_hint_is_whitespace_tolerant():
    assert "CONTEXT FROM" not in _pass1("   \n  ")


def test_content_hint_does_not_override_observation():
    """说明是辅助，不能取代实际观察 —— 否则模型会照抄用户的描述当观察结果。"""
    text = _pass1("主角是白发少女")
    assert "Do NOT copy it verbatim" in text
    assert "trust the frames" in text, "画面和说明冲突时要相信画面"


def test_content_hint_reaches_pass2():
    """成文阶段也要拿到说明 —— 观察结果可能漏掉或误判的东西要用得上。"""
    from app.schemas import AudioReport, MediaInfo

    user = build_pass2_user(
        observations=[], audio=AudioReport(), media=MediaInfo(path="x", duration=10.0),
        shots_summary="1 shot", content_hint="一镜到底，赛博朋克冷色调",
    )
    assert "CONTEXT FROM THE USER" in user
    assert "一镜到底，赛博朋克冷色调" in user
    assert "observation report above is authoritative" in user, "观察结果优先"


def test_pass2_omits_hint_when_empty():
    from app.schemas import AudioReport, MediaInfo

    user = build_pass2_user(
        observations=[], audio=AudioReport(), media=MediaInfo(path="x", duration=10.0),
        shots_summary="1 shot", content_hint="",
    )
    assert "CONTEXT FROM THE USER" not in user


# ---------------------------------------------------------------------------
# 剪辑结构：让模型自己判断切镜，而不是照搬我们的场景检测
# ---------------------------------------------------------------------------

EDIT_JSON = """{
  "subjects": [{"label": "boy", "kind": "person", "description": "a boy", "shots": ["1"]}],
  "shots": [{"shot": "1", "timecode": "00:00.000", "shot_size": "medium",
             "camera": "slow dolly-in", "subject": "boy", "action": "walks",
             "setting": "park", "lighting": "daylight", "color": "warm",
             "motion_energy": "medium", "on_screen_text": "none",
             "dialogue": "", "sfx": "", "transition": "continues", "confidence": 0.9}],
  "edit_structure": "continuous",
  "cut_points": [],
  "continuity_notes": "one unbroken camera move throughout",
  "global_notes": "single take"
}"""


def test_parse_reads_edit_structure():
    r = parse_pass1_json(EDIT_JSON)
    assert r.edit_structure == "continuous"
    assert r.cut_points == []
    assert "unbroken" in r.continuity_notes
    assert len(r.shots) == 1


def test_parse_normalizes_structure_wording():
    """模型写法五花八门，要归一到 continuous / multi_shot / unknown。"""
    for raw, want in (
        ("continuous", "continuous"),
        ("CONTINUOUS SHOT", "continuous"),
        ("one take", "continuous"),
        ("single-take", "continuous"),
        ("no cuts", "continuous"),
        ("multi_shot", "multi_shot"),
        ("multi-shot", "multi_shot"),
        ("has cuts", "multi_shot"),
        ("montage", "multi_shot"),
        ("", "unknown"),
        ("hard to tell", "unknown"),
    ):
        got = parse_pass1_json(json.dumps({"edit_structure": raw})).edit_structure
        assert got == want, f"{raw!r} 应归一为 {want}，实际 {got}"


def test_parse_cut_points_accepts_list_and_string():
    a = parse_pass1_json('{"cut_points": ["00:03.400", "00:07.900"]}')
    assert a.cut_points == ["00:03.400", "00:07.900"]
    b = parse_pass1_json('{"cut_points": "00:03.400, 00:07.900"}')
    assert b.cut_points == ["00:03.400", "00:07.900"]


def test_parse_cut_points_drops_non_timecode():
    """模型有时会塞进说明文字，只要时间码。"""
    r = parse_pass1_json('{"cut_points": ["00:03.400", "at the flash", "", "12:00"]}')
    assert r.cut_points == ["00:03.400", "12:00"]


def test_missing_edit_structure_defaults_to_unknown():
    r = parse_pass1_json('{"shots": []}')
    assert r.edit_structure == "unknown"
    assert r.cut_points == []


def test_pass1_system_explains_the_detector_is_not_ground_truth():
    """必须说清「时间戳来自检测器，不等于剪辑结构」。

    踩过：原来只说「按时间戳分组」，模型就把检测器切的每一段当成一个镜头。
    用户反馈「很多镜头其实是一镜到底的但是被切镜头了」。
    """
    s = PASS1_SYSTEM
    assert "sampling heuristic, not ground truth" in s
    assert "split a single continuous take" in s
    assert "emit **ONE** entry in `shots`" in s
    assert "tells the video model to cut there" in s, "要说清过度切分的后果"


def test_pass1_user_says_timestamps_are_a_sampling_aid():
    from app.services.templates import build_pass1_user

    text = build_pass1_user(
        chunk_start=0, chunk_end=10, frame_marks=[(0.0, "head"), (0.5, "mid")],
        audio_text="", media=None, chunk_index=0, chunk_total=1,
    )
    assert "sampling aid, not the edit structure" in text
    assert "edit_structure" in text


def test_pass2_tells_writer_not_to_cut_a_continuous_take():
    """成文阶段要被告知「这是一镜到底」——否则会写出一堆 [Shot N]。

    一镜到底的片子被写成一堆 [Shot N]，生成的视频就会在原本连续的地方硬切。
    """
    from app.schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation

    obs = [ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1", timecode="00:00.000", shot_size="medium",
                               subject="boy", action="walks")],
        edit_structure="continuous",
        continuity_notes="one unbroken move",
    )]
    user = build_pass2_user(
        observations=obs, audio=AudioReport(), media=MediaInfo(path="x", duration=10.0),
        shots_summary="1 shot",
    )
    assert "EDIT STRUCTURE" in user
    assert "CONTINUOUS" in user
    assert "must NOT contain multiple `[Shot N]` markers" in user


def test_pass2_lists_real_cut_points_for_multi_shot():
    from app.schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation

    obs = [ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1", timecode="00:00.000", shot_size="medium")],
        edit_structure="multi_shot", cut_points=["00:03.400"],
    )]
    user = build_pass2_user(
        observations=obs, audio=AudioReport(), media=MediaInfo(path="x", duration=10.0),
        shots_summary="2 shots",
    )
    assert "MULTI_SHOT" in user
    assert "00:03.400" in user
    assert "follow THESE cuts" in user, "要用模型判断的切点，不是采样时间戳"


def test_pass2_is_conservative_when_structure_unknown():
    from app.schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation

    obs = [ChunkObservation(
        chunk_index=0, start=0.0, end=10.0,
        shots=[ShotObservation(shot="1", timecode="00:00.000")],
        edit_structure="unknown",
    )]
    user = build_pass2_user(
        observations=obs, audio=AudioReport(), media=MediaInfo(path="x", duration=10.0),
        shots_summary="1 shot",
    )
    assert "UNKNOWN" in user
    assert "conservative" in user


def test_collapse_continuous_shots_merges_frame_entries():
    """一镜到底时把逐帧条目合并成一个。

    ⚠️ 为什么要代码强制：模型**判断对了却写不对**。实测一个连续运镜的 10 秒素材，
    模型正确报了 edit_structure=continuous、cut_points=[]，
    但 shots 数组仍然给了 20 个条目（一帧一个）。它自己前后矛盾。
    提示词里已经明确写了「一镜到底只输出一个条目」也没用 ——
    所以判断归模型、后果归代码。
    """
    from app.services.templates import collapse_continuous_shots

    shots = [
        ShotObservation(shot=str(i + 1), timecode=f"00:0{i}.000", shot_size="medium",
                        subject="singer", action=f"action {i}", confidence=0.8)
        for i in range(20)
    ]
    merged = collapse_continuous_shots(shots, "continuous")
    assert len(merged) == 1, f"应该合并成 1 个，实际 {len(merged)}"
    assert "action 0" in merged[0].action and "action 19" in merged[0].action, "动作细节要保留"
    assert merged[0].transition == "continues without a cut"
    assert merged[0].timecode == shots[0].timecode, "时间码取第一个"


def test_collapse_keeps_shots_for_multi_shot():
    from app.services.templates import collapse_continuous_shots

    shots = [
        ShotObservation(shot="1", timecode="00:00.000"),
        ShotObservation(shot="2", timecode="00:03.000"),
    ]
    assert len(collapse_continuous_shots(shots, "multi_shot")) == 2
    assert len(collapse_continuous_shots(shots, "unknown")) == 2


def test_collapse_dedups_repeated_actions():
    from app.services.templates import collapse_continuous_shots

    shots = [
        ShotObservation(shot="1", action="he sings"),
        ShotObservation(shot="2", action="he sings"),
        ShotObservation(shot="3", action="he turns"),
    ]
    merged = collapse_continuous_shots(shots, "continuous")
    assert merged[0].action == "he sings; he turns"


def test_collapse_noop_for_single_shot():
    from app.services.templates import collapse_continuous_shots

    one = [ShotObservation(shot="1", action="a")]
    assert collapse_continuous_shots(one, "continuous") == one


def test_dump_prompts_exports_everything(tmp_path):
    """提示词导出工具要能跑通，且覆盖全部格式。

    提示词是这个项目的核心资产，改一个字都影响产出。
    导出成文本对照着读，比在 9000 字的 Python 字符串里翻快得多。
    """
    from app import dump_prompts

    files = dump_prompts.dump(tmp_path)
    names = {f.name for f in files}
    assert "pass1_system.txt" in names
    assert "pass1_user.txt" in names
    assert "pass2_user.txt" in names
    assert "payload.json" in names
    for fmt in ("h3", "h3-ref", "seedance", "generic"):
        assert f"pass2_system_{fmt}.txt" in names, f"缺 {fmt}"

    for f in files:
        assert f.stat().st_size > 0, f"{f.name} 是空的"

    # 导出的 Pass1 用户消息里要能看到帧时间戳列表和说明注入
    p1u = (tmp_path / "pass1_user.txt").read_text(encoding="utf-8")
    assert "Image 1 -> timestamp" in p1u
    assert "sampling aid, not the edit structure" in p1u
    assert "CONTEXT FROM THE PERSON WHO SUBMITTED" in p1u

    # payload 骨架要能当 JSON 读回来，且图片是占位符
    payload = json.loads((tmp_path / "payload.json").read_text(encoding="utf-8"))
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert any(p.get("type") == "image_url" for p in payload["messages"][1]["content"])
