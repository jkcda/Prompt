"""提示词模板的单元测试。

重点测两件事：
  1. **目标格式规则**（H3 三字段 / Ref2VA 六段 / Seedance 六要素）必须精确 ——
     那是输出契约，错了下游解析不了。
  2. 自由发挥模式的系统提示词**只带身份、任务和目标格式**，不再有「怎么观察」
     的规则（那些规则实测会让输出退化）。
"""

from __future__ import annotations

import json

from app.schemas import (
    AudioReport,
    MediaInfo,
)
from app.services import templates
from app.services.templates import (
    FORMAT_LABELS,
    MODE_LABELS,
    MODE_VARIANTS,
    build_compress_system,
    build_compress_user,
    build_system,
    build_user,
    check_prompt_integrity,
    format_display,
    mode_of,
    prompt_sections,
)


def test_pass1_user_lists_images_in_order():
    text = build_user(
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
    # 开头只给「时长 + 分辨率」，不再有 segment 编号那套说法
    assert "10.00 seconds long" in text
def test_pass2_system_h3_uses_bare_field_names():
    """H3 的字段名必须是裸名 + 冒号，不能用尖括号标签包裹。"""
    system = build_system("h3", "en")
    assert "integrated_multimodal_description:" in system
    assert "<integrated_multimodal_description>" not in system
    assert "overall_soundscape:" in system
    assert "non_diegetic_music:" in system


def test_pass2_system_h3_ref_has_six_sections():
    system = build_system("h3-ref", "en")
    for section in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
    ):
        assert section in system


def test_pass2_system_seedance_is_chinese():
    system = build_system("seedance", "en")
    assert "Seedance" in system
    assert "主体" in system
    assert "无对白" in system
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
            system = build_system(fmt, lang)
            assert "{language_instruction}" not in system
            assert "{common}" not in system


def test_h3_t2va_forbids_reference_labels():
    """T2VA 没有参考素材，出现 <Subject N> 会被当成字面文本写进画面。"""
    system = build_system("h3", "en")
    assert "no reference assets" in system.lower()
    assert "Never write" in system
    assert "<Subject 1>" in system  # 出现在禁令里


def test_h3_ref_uses_subject_registry_as_source_of_labels():
    system = build_system("h3-ref", "en")
    assert "SUBJECT REGISTRY" in system
    assert "do not invent" in system


def test_h3_ref_does_not_reference_the_source_video():
    """Ref2VA 的「参考」是**格式**参考，不是让你参考原视频。

    原视频只用来提取主体 —— 定义成 <Subject N> 供用户自己挂参考图。
    出现 <Video 1> / <Audio 1> 是错的：那会把生成结果绑死在原片上，
    而用户要的是「用我自己的参考图生成」。
    """
    system = build_system("h3-ref", "en")
    assert "FORMAT, not the source video" in system
    assert "Do NOT define or mention `<Video 1>`" in system
    assert "Do not add a `<Video 1>` or `<Audio 1>` line" in system
    assert "no video reference and no audio reference" in system
    # 只允许在禁令语境里出现这两个标签名
    assert "supply their OWN reference images" in system


def test_h3_ref_retention_analysis_excludes_video_and_audio_lines():
    system = build_system("h3-ref", "en")
    assert "one line per `<Subject N>` label ONLY" in system
    assert "no video or audio line" in system


def test_format_blocks_carry_no_word_budget():
    """格式块里**不能有词数上限** —— 它会让模型合并镜头。

    实测（同一段每 0.3 秒一刀的 PV，30 帧）：

    | 正文上限 | 镜头数 | 词数 |
    |---|---|---|
    | 420 词（原来的写法） | 6 | 378 |
    | 900 词 | 15 | 528 |
    | 完全去掉 | **21** | 1237 |

    21 个镜头需要约 1200 词，420 词的预算下模型只能把多个镜头并成一个 ——
    用户直接反馈「效果变差了」。**长度只能兜底（`PROMPT_WORD_LIMIT`），
    不能写进生成时的指令。**
    """
    for fmt in ("h3", "h3-ref", "seedance", "generic"):
        system = build_system(fmt, "en")
        for banned in ("LENGTH BUDGET", "words or fewer", "under 700 words"):
            assert banned not in system, f"{fmt} 里还有词数上限：{banned}"

    # 但**结构性**约束要留：Ref2VA 最多 6 个主体，是生成端的硬限制
    ref = build_system("h3-ref", "en")
    assert "at most 6 entries" in ref, "subject_definitions 的条数上限是结构约束，要留"
    # 也要明确说「不许为了省篇幅合并镜头」
    assert "never drop or merge shots" in ref.lower() or "Do not drop or merge shots" in ref


def test_h3_ref_tells_which_subjects_to_drop_when_too_many():
    """登记表超过 6 条时要给出取舍优先级，否则模型会平均用力写得又臭又长。"""
    system = build_system("h3-ref", "en")
    assert "people > wardrobe/props > environment > style/grade" in system
    assert "Drop the least important ones entirely" in system




def test_seedance_mode_has_no_h3_markup():
    """Seedance 是连贯中文段落，混进字段名或 <Subject N> 就是错的。"""
    system = build_system("seedance", "en")
    assert "Seedance" in system
    assert "Do NOT use field labels" in system
    assert "Chinese only" in system
    assert "节拍" in system
    # 三字段名只能出现在禁令语境里，不能作为输出结构
    assert "integrated_multimodal_description:" not in system
    assert "overall_soundscape:" not in system


def test_seedance_beat_map_must_cover_every_shot():
    """Seedance 是中文提示词，断言用中文短语 —— 别在中文提示词里断言英文大写词。"""
    system = build_system("seedance", "en")
    assert "每一个镜头" in system
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

    assert "一无所知" in text
    assert "禁止编造" in text
    assert "不要写任何台词或歌词" in text
    assert "不要写 BGM 的乐器" in text
    assert "N/A" in text
    # ⚠️ 音量数字**不再**进提示词。数字对生成视频没有意义，而且会诱导模型
    # 写「低频 -12dB 的成分」这种句子 —— 实测就是这么漏出去的。
    assert "-21.0dB" not in text


def test_audio_report_with_transcript_does_not_warn():
    from app.services.asr import format_transcript_for_prompt

    report = AudioReport(has_audio=True, transcript="la la la")
    text = format_transcript_for_prompt(report)
    assert "la la la" in text
    assert "禁止编造" not in text


def test_audio_report_no_track_says_na():
    from app.services.asr import format_transcript_for_prompt

    text = format_transcript_for_prompt(AudioReport(has_audio=False))
    assert "没有音轨" in text
    assert "N/A" in text
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
# 频谱特征的措辞边界
# ---------------------------------------------------------------------------

def test_spectrum_speech_dominant():
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-1.0, low_band_db=-25.0))
    joined = "\n".join(lines)
    assert "人声/主奏频段能量占比高" in joined
    assert "低频很弱" in joined
    # 不能升级成内容断言
    assert "有人说话" not in joined
    assert "对白为主" not in joined


def test_spectrum_music_like():
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-12.0, low_band_db=-4.0))
    joined = "\n".join(lines)
    assert "能量大部分落在人声频段之外" in joined
    assert "低频成分显著" in joined
    # 低频强不等于一定有鼓点
    assert "鼓" in joined and "这类" in joined


def test_spectrum_mixed_case():
    """典型情况：人声 + 配器，两个频段都有能量。"""
    from app.services.asr import describe_spectrum

    lines = describe_spectrum(AudioReport(has_audio=True, speech_band_db=-4.5, low_band_db=-4.3))
    joined = "\n".join(lines)
    assert "混有其他频段" in joined


def test_spectrum_missing_values_are_skipped():
    from app.services.asr import describe_spectrum

    assert describe_spectrum(AudioReport(has_audio=True)) == []


def test_spectrum_numbers_stay_out_of_the_prompt():
    """频谱数据**不再**以原始形式进提示词。

    实测：给了 dB 数字和频段名之后，最终输出里出现了
    `measured energy mostly voice band` / `inferred from energy distribution`
    —— 模型把工具术语照抄走了。现在这些数字只在服务端支撑
    `MusicProfile.describe()` 生成的那句人类可读描述。
    """
    from app.services.asr import format_transcript_for_prompt

    report = AudioReport(has_audio=True, speech_band_db=-1.2, low_band_db=-19.0)
    text = format_transcript_for_prompt(report)

    # 只查**真正有害**的形式：测量数字和英文分析术语。
    # 中文的「频段 / 分贝」会出现在禁令句里，那是合理的，不算泄漏。
    assert "dB" not in text
    assert "Hz" not in text
    for term in ("content unanalysed", "energy distribution", "voice band", "spectral"):
        assert term not in text.lower(), f"音频段落里漏出了工具术语：{term}"
    assert "【音乐与声音" in text
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
    assert "words" in system and "700" in system, "要写清词数"
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
    assert "about 595 words" in system, "700 × 0.85 = 595"
    assert "never more than 700" in system, "真实上限也要写清楚"

    user = build_compress_user("a " * 2000, 700)
    assert "2000 words" in user
    assert "about 595 words" in user, "目标词数要写进用户消息"
    assert "hard ceiling 700" in user, "上限也要写清楚"
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
    assert "about 85 words" in text, "100 × 0.85 = 85"
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
    return build_user(
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
def test_user_message_says_the_timestamps_are_a_sampling_aid():
    """时间戳来自检测器，不是剪辑事实 —— 必须说清，否则模型会把采样分组当镜头。"""
    from app.services.templates import build_user

    text = build_user(
        chunk_start=0, chunk_end=10, frame_marks=[(0.0, "head"), (0.5, "mid")],
        audio_text="", media=None, chunk_index=0, chunk_total=1,
    )
    assert "the marks as hints rather than the answer" in text


def test_dump_prompts_exports_everything(tmp_path):
    """提示词导出工具要能跑通，且覆盖全部格式。

    提示词是这个项目的核心资产，改一个字都影响产出。
    导出成文本对照着读，比在 Python 字符串里翻快得多。
    """
    from app import dump_prompts

    files = dump_prompts.dump(tmp_path)
    names = {f.name for f in files}
    assert "user.txt" in names
    assert "payload.json" in names
    for fmt in ("h3", "h3-ref", "seedance", "generic"):
        assert f"system_{fmt}.txt" in names, f"缺 {fmt}"

    for f in files:
        assert f.stat().st_size > 0, f"{f.name} 是空的"

    # 导出的用户消息里要能看到帧时间戳列表和用户说明注入
    u = (tmp_path / "user.txt").read_text(encoding="utf-8")
    assert "Image 1 -> timestamp" in u
    assert "the marks as hints rather than the answer" in u
    assert "CONTEXT FROM THE PERSON WHO SUBMITTED" in u

    # payload 骨架要能当 JSON 读回来，且图片是占位符
    payload = json.loads((tmp_path / "payload.json").read_text(encoding="utf-8"))
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert any(p.get("type") == "image_url" for p in payload["messages"][1]["content"])
# ---------------------------------------------------------------------------
# 帧间隔里的镜头边界信号
# ---------------------------------------------------------------------------

def _pass1_text(marks):
    return build_user(
        chunk_start=0.0,
        chunk_end=marks[-1][0] + 0.5,
        frame_marks=marks,
        audio_text="",
        media=None,
        chunk_index=0,
        chunk_total=1,
    )


def _frame_lines(txt: str) -> list[str]:
    """只取帧列表那几行 —— 说明段落里也会提到 `sampling boundary`，
    断言必须限定在帧行上，否则永远为真。"""
    return [ln for ln in txt.split("\n") if ln.strip().startswith("Image ")]


def test_shot_boundary_is_marked_in_frame_list():
    """帧间隔突变处要标出来。

    实测模型会**识别对切点却把内容归错组**：它报了 4.380 是切点，
    却把 4.380 之后那段（花园场景）描述成前一个镜头的拖鞋。
    时间戳它看得见，但注意力没落在「间隔突变」上 —— 标出来才知道往哪看。

    抽帧是按镜头切点对齐的，所以镜头末帧（tail）和下一镜首帧（head）
    会挨得特别近，那个位置就是切点。
    """
    marks = [(0.0, "head"), (0.5, "mid"), (1.0, "mid"), (1.5, "tail"), (1.6, "head"), (2.1, "mid")]
    txt = _pass1_text(marks)

    marked = [ln for ln in _frame_lines(txt) if "sampling boundary" in ln]
    assert len(marked) == 1, f"应该只标一处，实际 {len(marked)} 处"
    assert "0.10s after the previous frame" in marked[0]
    # 只说「检测器在这里看到了变化」，不能说成已确认的切点 ——
    # 我们的检测可能把一镜到底切碎，说死了反而误导。
    assert "the marks as hints rather than the answer" in txt


def test_uniform_gaps_are_not_marked():
    """间隔均匀时不该乱标，否则等于把每一帧都说成切点。"""
    marks = [(i * 0.5, "mid") for i in range(6)]
    txt = _pass1_text(marks)

    assert not [ln for ln in _frame_lines(txt) if "sampling boundary" in ln]
    # 但「看间隔」这条提示始终在
    assert "Read the gaps, not just the timestamps" in txt


# ---------------------------------------------------------------------------
# Seedance 节拍图自洽校验
# ---------------------------------------------------------------------------
#
# 实测踩过：一次真实输出把 4 个场景（阳台 / 台球桌 / 沙发 / 台球桌）压成**一整段**，
# 用「随后画面切到」「接着」「最后」串起来，**末尾连节拍图都没有** ——
# 下游根本看不出切了几次镜。用户直接反馈「提示词漏镜头，格式也不对」。
#
# 根因是 Seedance 格式块自相矛盾：开头写「Write ONE coherent prompt」，
# 镜头规则又要求「每个镜头单独成段」，模型按 ONE 走 → 写成一整段。
# 已修（矛盾去掉 + 每镜成段提到开头 + 末尾节拍图提到开头）。


def test_beat_map_passes_when_declared_matches_body():
    txt = "开场画面。画面切到A。画面切到B。全片三个镜头，节拍为 0–1 秒、1–2 秒、2–3 秒。"
    assert templates.check_beat_map_consistency(txt) == (True, "")


def test_beat_map_accepts_chinese_numerals():
    """模型两种写法都会用：「七个镜头」和「7个镜头」。"""
    txt = "开场。画面切到A。全片两个镜头，节拍为 0–1 秒、1–2 秒。"
    assert templates.check_beat_map_consistency(txt)[0] is True
    txt2 = "开场。画面切到A。全片 2 个镜头，节拍为 0–1 秒、1–2 秒。"
    assert templates.check_beat_map_consistency(txt2)[0] is True


def test_beat_map_flags_missing_beat_map_when_there_are_cuts():
    """**有切镜却没写节拍图 = 漏镜头。** 这是用户实际报的那个 case。

    正文分了段，但末尾没有节拍图 —— 下游看不出切了几次镜。
    """
    txt = "开场画面。画面切到室内。画面切到台球桌。画面切回沙发。不要出现字幕。"
    ok, why = templates.check_beat_map_consistency(txt)
    assert ok is False
    assert "没写节拍图" in why
    assert "4 段" in why, f"要报出实际段数，实际：{why}"


def test_beat_map_counts_both_qiedao_and_qiehui():
    """⚠️ 「画面切**回**」也要数 —— 只匹配「切到」会把 4 段数成 2 次切镜。"""
    txt = "开场。画面切到A。画面切回B。全片三个镜头，节拍为 0–1 秒、1–2 秒、2–3 秒。"
    assert templates.check_beat_map_consistency(txt)[0] is True


def test_beat_map_flags_count_mismatch():
    """声明 3 个镜头却只列 2 个时间区间。"""
    txt = "开场。画面切到A。全片三个镜头，节拍为 0–1 秒、1–2 秒。"
    ok, why = templates.check_beat_map_consistency(txt)
    assert ok is False
    assert "3 个镜头" in why and "2 个时间区间" in why


def test_beat_map_allows_single_shot_without_beat_map():
    """单镜素材（不切镜）不写节拍图是正常的，不能判错。"""
    assert templates.check_beat_map_consistency("一个人走来。全片不切镜。无对白。") == (True, "")
    assert templates.check_beat_map_consistency("一个人走来。无对白。") == (True, "")


def test_beat_map_rejects_empty_output():
    ok, why = templates.check_beat_map_consistency("   ")
    assert ok is False
    assert "空" in why


def test_seedance_block_requires_one_paragraph_per_shot():
    """Seedance 格式块必须写明「每个镜头单独成段」。

    缺这条时实测模型会把 4 个场景压成一整段（因为开头写着「ONE coherent prompt」）。
    H3 一直有对应的规则（one paragraph per shot），Seedance 原来漏了。
    """
    block = templates.build_system("seedance", "zh")
    assert "每个镜头单独成段" in block
    assert "不要为了写短而删减或合并镜头" in block
    # 开头那两行要出现（而不是只埋在约束列表里）
    head = block.split("Hard constraints:")[0]
    assert "每一镜单独成段" in head or "每个镜头单独成段" in head
    assert "节拍图" in head, "节拍图要求要放在开头，埋在约束里会被忽略"


def test_seedance_block_has_no_stale_observation_report_reference():
    """⚠️ 「观察报告」是两阶段管线的产物，早就删了。

    留着这个死引用会让模型去找一份不存在的报告，从而不知道该覆盖哪些镜头。
    """
    block = templates.build_system("seedance", "zh")
    assert "观察报告" not in block
