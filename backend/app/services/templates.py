"""提示词引擎：自由发挥模式的 Prompt 模板。

一次调用：帧 + 时间戳 + 用户说明 → 目标格式提示词。

⚠️ 这里**只有两类内容**：
  1. `build_system()` —— 身份 + 任务 + 目标格式块。**刻意不写「怎么观察」的规则。**
  2. `build_user()` —— 帧时间戳清单 + 音频报告 + 用户说明。

曾经有过 Pass1（结构化观察）+ Pass2（成文）两阶段，Pass1 的系统提示词长到
14000 字符、大半在教模型「怎么观察」和「别信我们的场景检测」。实测每收紧一次
这类规则、输出就退化一次：防编造规则压掉合理推断（动作描述只剩 19%，读起来像
照片说明）、把「相机静止」当结论喂进去会被外推成「人物也静止」、写死「20 秒 MV
大约 5-20 个镜头」会把快切压到 20 以内。

**约束输出格式 ≠ 约束思考。** 现在只留目标格式（输出契约）和两条关于**产物**的
硬约束（静止动词会让视频冻住；音频未知时编造比留空更糟）。
"""

from __future__ import annotations

import re
import statistics

from ..schemas import MediaInfo


def build_user(
    chunk_start: float,
    chunk_end: float,
    frame_marks: list[tuple[float, str]],
    audio_text: str,
    media: MediaInfo | None,
    chunk_index: int,
    chunk_total: int,
    content_hint: str = "",
    closing: str = "",
    sheet_note: str = "",
) -> str:
    """组织用户消息文本（图片由调用方按顺序附在后面）。

    `content_hint` 是用户自己写的画面说明。**这东西很有用**：静态帧看不出
    「这段是一镜到底还是多镜头切换」「这是什么作品/角色」「动作的前因后果」，
    而这些直接影响产出质量。让用户补一句话，比让模型瞎猜强得多。

    """
    lines: list[str] = []
    span = chunk_end - chunk_start
    if media:
        res = f"{media.width}x{media.height}" if media.width else "unknown"
        lines.append(
            f"This video is {span:.2f} seconds long — {res} at {media.fps:.2f} fps."
        )
    else:
        lines.append(f"This video is {span:.2f} seconds long.")

    if sheet_note:
        # 拼图模式：**只给时长 + 图 + 每格里是哪些时间点**，剩下交给模型判断。
        #
        # ⚠️ 这里刻意不列逐帧清单、不给 role 标签、不加「怎么读帧间隔」的说明。
        # 用户原话：「你就把这段视频到底一共几秒传过去，然后图传上去，剩下 AI
        # 自行判断，很难吗？」—— 我原来塞了 9 段说明进去（30 行帧清单 + 每帧
        # 角色 + 切点标注 + 两段「怎么读间隔」+ 布局说明），全是在教它做事。
        lines.append("")
        lines.append(sheet_note)
        lines.append("")
        lines.append(audio_text)
        if content_hint.strip():
            lines.append("")
            lines.append("--- CONTEXT FROM THE PERSON WHO SUBMITTED THIS VIDEO ---")
            lines.append(content_hint.strip())
        lines.append("")
        lines.append(closing or build_closing())
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"The {len(frame_marks)} images attached AFTER this text are in this exact order:"
    )

    # 帧间隔本身携带信息：抽帧是按镜头切点对齐的，所以镜头末帧（tail）和
    # 下一镜首帧（head）之间会挨得特别近 —— 那个位置就是切点。
    #
    # 为什么要显式标出来：实测模型会**识别对切点却把内容归错组**
    # （说 4.380 是切点，却把 4.380 之后那段花园场景描述成拖鞋）。
    # 时间戳它看得见，但注意力没落在「间隔突变」上，标出来才知道该往哪看。
    #
    # 措辞用「sampling boundary」而不是「shot boundary」：我们的切点可能
    # 是把一镜到底切碎的误切，不能让它当成铁定的剪辑事实。
    gaps = [frame_marks[i][0] - frame_marks[i - 1][0] for i in range(1, len(frame_marks))]
    normal_gap = statistics.median(gaps) if gaps else 0.0

    prev_t: float | None = None
    prev_role = ""
    for i, (t, role) in enumerate(frame_marks, start=1):
        note = ""
        if prev_t is not None and role == "head" and prev_role == "tail":
            gap = t - prev_t
            if normal_gap > 0 and gap < normal_gap * 0.6:
                note = (
                    f"   ← sampling boundary: only {gap:.2f}s after the previous frame, "
                    f"while the usual gap is ~{normal_gap:.2f}s. Sampling is aligned to "
                    f"cuts, so this is where the detector saw a change — look carefully "
                    f"here and decide from the images whether it is a real cut."
                )
        lines.append(f"  Image {i} -> timestamp {t:.3f}s [role: {role}]{note}")
        prev_t, prev_role = t, role
    lines.append("")
    lines.append(
        "**Read the gaps, not just the timestamps.** The places marked `sampling boundary` are "
        "where the automated detector saw a change — a cut is likely there.\n"
        "\n"
        "⚠️ But the detector only catches **hard pixel jumps**. Dissolves, graphic wipes, "
        "paint-splash transitions and panel slides are everywhere in footage like this and it "
        "**misses all of them** — measured on one clip: a 6.4-second span holding 13 frames of "
        "clearly different compositions (hand close-up → eye extreme close-up → medium shot → "
        "split panel → reaching for the lens) produced **zero** detected cuts.\n"
        "\n"
        "So decide the structure from the images, using the marks as hints rather than the "
        "answer. Frames that continue one setup — same framing and subject, the action simply "
        "progressed — are ONE shot: describe the motion that plays out across them. A new "
        "setup is a new shot."
    )
    lines.append("")


    lines.append(audio_text)

    # 拼图布局：告诉模型每张网格图里装了哪几个时间点。
    # 不说明的话它不知道格子和时间怎么对应 —— 拼图反而变成噪声。
    if sheet_note:
        lines.append("")
        lines.append(sheet_note)

    # 用户补充的画面说明。放在音频之后、正式指令之前 ——
    # 位置太靠前容易被后面的长指令冲淡，太靠后又会被当成输出要求。
    if content_hint.strip():
        lines.append("")
        lines.append("--- CONTEXT FROM THE PERSON WHO SUBMITTED THIS VIDEO ---")
        lines.append("They wrote the following about this footage. Treat it as reliable "
                     "background, and use it especially for things still frames cannot show:")
        lines.append("  * whether the edit is one continuous take or has cuts between shots")
        lines.append("  * what the action is, and what happens before/after this segment")
        lines.append("  * who or what the subjects are (character, work, product, place)")
        lines.append("  * the intended style or genre")
        lines.append("")
        lines.append(content_hint.strip())
        lines.append("")
        lines.append("Fold this into your own description. Do NOT copy it verbatim, and do not "
                     "let it replace what you actually see — if the frames contradict it, "
                     "trust the frames and note the discrepancy.")

    lines.append("")
    lines.append(closing or build_closing())
    return "\n".join(lines)
_PASS2_H3 = """{common}

TARGET MODE — MiniMax H3, text-to-video with audio (T2VA). There are NO reference assets, so the \
prompt must be fully self-contained.

Output exactly three fields, in this order, each starting at the beginning of a line as a bare \
name followed by a colon. Do NOT wrap the names in angle brackets.

integrated_multimodal_description: <the main body>
overall_soundscape: <ambience and physical sounds>
non_diegetic_music: <audience-only score, or N/A>

`integrated_multimodal_description`:
- Open with one or two sentences of overall style, format and grade, BEFORE the first shot marker.
- Then one paragraph per shot: `[Shot 1]` with no timestamp, then `[Shot N] At MM:SS.mmm, ...` for \
later shots, using the cut times from the report.
- Inside each shot lead with what HAPPENS, then framing and camera movement, then just enough \
appearance / environment / lighting to support the action, then the current sound.
- With no reference assets, state each subject's appearance in full at first appearance and reuse \
the same wording afterwards. Never write `<Subject 1>`, `<Video 1>` or `<Audio 1>` — those labels \
are meaningless in T2VA and will be read as literal text.
- Speakers get stable IDs `(S1)`, `(S2)` in order of first vocal event. Dialogue and lyrics are \
written `<d>[Language] the exact words</d>`. When a speaker is on camera, state their mouth \
movement in the same shot.
- **Do not drop or merge shots to keep the prompt short.** If the footage has many cuts, the \
prompt needs many `[Shot N]` blocks — write every one of them.
- No closing or resolution marker: stop mid-flow on the last described action.

`overall_soundscape`: one or two sentences of ambience and physical action sound across the whole \
video. Do not repeat dialogue or lyrics.

`non_diegetic_music`: audience-only score — instrumentation, tempo, dynamics. `N/A` if there is \
no score.
"""


_PASS2_H3_REF = """{common}

TARGET MODE — MiniMax H3 full-reference (Ref2VA) FORMAT. "Reference" here means the prompt FORMAT, \
not the source video. The source video is only being MINED for content — it is NOT a reference \
asset and must NOT be cited as one.

- Do NOT define or mention `<Video 1>` or `<Audio 1>` — there is no video reference and no audio reference.
- `<Subject N>` labels are the reusable content extracted from the footage. The user will supply \
their OWN reference images for them, so each definition must identify the thing from text alone.

The SUBJECT REGISTRY in the user message is authoritative for labels: do not invent subjects \
outside it, and do not split one entry into several.

Output exactly six sections, in this order, each starting at the beginning of a line as a bare \
name followed by a colon (no angle brackets):

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

`subject_definitions` takes **at most 6 entries** — the generator accepts no more.

Keep the writing tight: no filler, no restating the style, no repeating an appearance after its \
first mention. **But never drop or merge shots to save space** — a cut in the footage is a \
`[Shot N]` in the output.

**Subject selection**: the registry may exceed 6 items. Pick the 6 that most need a reference \
image, in this priority: people > wardrobe/props > environment > style/grade. Drop the least important ones \
entirely rather than giving everyone a half-line.

Sections:
- `subject_definitions`: one line per selected entry, numbered in registry order `<Subject 1>`, \
`<Subject 2>`, ... — what the label denotes plus identifying features. A `style` \
entry is defined as the look and grade to carry across, not as an object. Do not add a `<Video 1>` or \
`<Audio 1>` line.
- `summary`: one short paragraph opening with a bracketed task-type prefix, e.g. \
`[reference generation]` or `[reference generation + style transfer]`. No new labels; do not call \
the source clip a reference.
- `retention_analysis`: one line per `<Subject N>` label ONLY — no video or audio line. Use the fixed \
markers `fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`, \
formatted `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`, shot list verbatim \
from the registry, with a short reason. Default is `fully_preserved` unless the report \
says the appearance changes.
- `detailed_description`: the main body. One or two sentences of style before `[Shot 1]`. Then \
`[Shot 1]` with no timestamp and `[Shot N] At MM:SS.mmm, ...` afterwards. Each shot leads with the \
action, then short appearance / environment / lighting. Insert labels at first appearance and \
wherever their role applies. Speakers use `(Sx)`; dialogue uses `<d>[Language] ...</d>`.
- `overall_soundscape` / `non_diegetic_music`: ambience and physical sound vs. audience-only score. \
`N/A` when a category is absent. Never repeat dialogue.
"""


_PASS2_SEEDANCE = """{common}

TARGET MODE — Seedance 2.0 / 即梦. Write ONE coherent natural-language prompt in CHINESE. Element \
order: 主体 → 动作 → 环境 → 风格 → 镜头 → 声音.

Hard constraints:
- Chinese only. No English except proper nouns and on-screen text.
- Do NOT use field labels, `[Shot N]` markers, H3's colon structure, or any `<Subject N>` / `<d>` markup. \
Plain prose, not a structured document.
- Flowing prose, not bullets. 主体 + 动作 take about half the length; then one sentence each for \
环境, 风格, 镜头, 声音.
- 主体 must be pinned down by colour, material, garment cut and identifying features — there are \
no reference images in this mode. 漂亮、高级、电影感十足 carry no information and are forbidden.
- 动作 must carry its physical consequence：重心转移、头发摆动、衣料起伏、配饰晃动、与地面的接触。\
没有动作描写的主体在生成视频里会变成冻住的假人。
  **动作写在最前面** —— 画面是时间的流动不是一张照片：先写这个镜头里发生了什么、怎么发展，\
再补环境与外观，且只补动作需要的那点。样本帧之间的动作要自己推断出来写进去。
- 镜头：把景别与运镜写成连续推进，例如「以全景开场，随后缓慢推轨至中近景，浅景深」。全片不切镜就写\
「全片不切镜」；有切镜则写明镜头数，并在末尾给节拍图，覆盖观察报告里的每一个镜头、不留空隙不重叠，\
例如「全片四个镜头，节拍为 0–2.5 秒、2.5–5 秒、5–7.5 秒、7.5–10 秒」。
- 声音：环境声、动作声、音乐、台词，用分号分隔。无台词则以「无对白」结尾。配乐描述放在最后。
- 报告里出现字幕、文字、logo、水印时，在末尾单独加负面指令：「不要出现字幕、文字、水印」。
- 不写时长与画幅 —— 那是独立参数。
- 台词与歌词保留原语言，就地引用。说话人出镜时，在 动作 里写清嘴部运动。
- 不以收尾或结束性的句子结尾。
"""


_PASS2_GENERIC = """{common}

TARGET MODE — a neutral, tool-agnostic storyboard prompt sheet. Write in {language_instruction} \
using this structure:

【整体风格】one paragraph: medium, genre, grade, palette, lighting logic, aspect feel, pacing.

【镜头分镜】one block per shot:
镜头 N｜MM:SS.mmm–MM:SS.mmm
  景别 / 角度：
  运镜：
  画面内容：**先写动作**（这个镜头里发生了什么、怎么发展），再补主体、环境、光线
  台词 / 人声：exact words in original language, or 无
  音效：
  转场：

【声音设计】ambience, action sound, and audience-only score, described separately.

【负面提示词】a comma-separated list of what must not appear, derived from the observation \
report (text overlays, watermarks, unwanted artefacts, identity drift).

Do not add a closing or summary section after the negative prompt.
"""


def build_compress_system(limit: int) -> str:
    """超长时的压缩指令。

    为什么不靠「生成时就守住预算」：实测模型对字数指令的服从度很差 ——
    给了逐段预算（定义 15 词/条、正文 420 词）之后仍然写出
    定义 19 词/条、正文 562 词，整篇 918 词（上限 700）。
    但「把现成的文本改短」是模型很擅长的编辑任务，比「按预算生成」可靠得多。

    ⚠️ 这个调用要**开思考**，和其他环节相反。实测（833 词的六段式）：
        长指令 + 关思考 -> 833 词（原样返回，模型根本没改）
        短指令 + 关思考 -> 759 词（只砍 74 词，不达标）
        短指令 + 开思考 -> 581 词（砍 252 词，达标）
    编辑需要先想清楚「哪些能砍」，思考过程在这里是有用的。
    生成环节才需要关思考（那是照结构填内容，思考纯属浪费）。

    ⚠️ 目标值要比真实上限**更紧**：实测说「压到 700 以内」，模型压到 803 就收手了
    （2297 → 803，确实砍了 65%，但还是超）。让它瞄 0.85 倍，落点才在上限之内。
    """
    target = max(1, int(limit * 0.85))
    return f"""You are an editor tightening a prompt. Shorten it to about {target} words \
— never more than {limit}. Cut only redundancy: adjectives, filler, repeated descriptions. \
Do NOT drop or merge shots, and do not go below {target} — every cut costs fidelity.

Keep: the section names present in the original, in order, each on its own line \
as a bare `name:`; every `<Subject N>` label; every `[Shot N]` marker; all retention \
markers; all dialogue verbatim; and `N/A` where present.

Output only the shortened prompt."""


def build_compress_user(prompt: str, limit: int) -> str:
    words = len(prompt.split())
    target = max(1, int(limit * 0.85))
    return (
        f"The following prompt is {words} words. It needs to come down to about "
        f"{target} words (hard ceiling {limit}) — no further.\n\n"
        f"--- BEGIN PROMPT ---\n{prompt}\n--- END PROMPT ---"
    )


def prompt_sections(text: str) -> list[str]:
    """取出提示词里的段落名（形如 `xxx:` 独占一行的裸名）。

    不写死段落清单 —— 三种格式的段落完全不同（T2VA 三字段 / Ref2VA 六段 /
    Seedance 无字段），写死会让校验对非 Ref2VA 格式永远失败。
    """
    return re.findall(r"^([a-z][a-z0-9_]*):\s*$", text, re.M)


def check_prompt_integrity(original: str, compressed: str) -> tuple[bool, str]:
    """压缩后校验关键结构没丢。返回 (是否可用, 原因)。

    压缩是「编辑」任务，模型偶尔会顺手删掉整节或合并主体 —— 那比超长更糟，
    所以宁可保留原文也不能接受残缺的结构。

    ⚠️ 段落清单从**原文**里取，不写死。踩过：校验里硬编码了 Ref2VA 的六段名，
    于是 T2VA / Seedance 的压缩**永远被判为残缺**（实测 T2VA 压到 487 词仍被拒），
    压缩功能对非 Ref2VA 格式等于不存在。
    """
    if not compressed.strip():
        return False, "压缩结果为空"

    want = prompt_sections(original)
    got = set(prompt_sections(compressed))
    missing = [s for s in want if s not in got]
    if missing:
        return False, f"压缩后缺少段落：{', '.join(missing)}"

    n_before = set(re.findall(r"<Subject (\d+)>", original))
    n_after = set(re.findall(r"<Subject (\d+)>", compressed))
    if not n_after.issuperset(n_before):
        lost = sorted(n_before - n_after, key=int)
        return False, f"主体标签丢失：<Subject {', <Subject '.join(lost)}>"

    shots_before = set(re.findall(r"\[Shot (\d+)\]", original))
    shots_after = set(re.findall(r"\[Shot (\d+)\]", compressed))
    if not shots_after.issuperset(shots_before):
        lost = sorted(shots_before - shots_after, key=int)
        return False, f"镜头标记丢失：[Shot {', [Shot '.join(lost)}]"

    if "N/A" in original and "N/A" not in compressed:
        return False, "N/A 段被删掉了"

    return True, ""


# ---------------------------------------------------------------------------
# 模式与变体
# ---------------------------------------------------------------------------
#
# 对外只暴露两种模式：
#   H3 模式       —— 给 MiniMax H3 用，英文，字段名裸名 + 冒号
#   Seedance 模式 —— 给 Seedance 2.0 / 即梦 用，中文连贯段落，六要素顺序
#
# 每个模式下的变体（variant）是真正的 format 取值。H3 有两个变体是因为
# T2VA 与 Ref2VA 不只是措辞不同：T2VA 没有任何参考素材，提示词必须自洽；
# Ref2VA 要输出 <Subject N> 参考标签和逐主体的 retention 等级。
# 把两者混为一谈会写出既带参考标签、又没有参考素材的四不像。

MODE_OF_FORMAT: dict[str, str] = {
    "h3": "h3",
    "h3-ref": "h3",
    "seedance": "seedance",
    "generic": "generic",
}

MODE_LABELS = {
    "h3": "H3 模式",
    "seedance": "Seedance 模式",
    "generic": "通用分镜表",
}

MODE_NOTES = {
    "h3": "输出给 MiniMax H3 的视频提示词，英文为主，字段名裸名加冒号，"
          "镜头用 [Shot N] 标记并带切点时间戳。",
    "seedance": "输出给 Seedance 2.0 / 即梦的中文提示词，"
                "按 主体→动作→环境→风格→镜头→声音 六要素写成连贯段落，不带字段名。",
    "generic": "工具无关的分镜脚本，适合人工二次加工或投喂给其他工具。",
}

# 模式 → 变体。顺序即前端展示顺序，`default=True` 的变体是该模式的默认选择。
MODE_VARIANTS: dict[str, list[tuple[str, bool]]] = {
    "h3": [("h3", True), ("h3-ref", False)],
    "seedance": [("seedance", True)],
    "generic": [("generic", True)],
}

MODE_PRIMARY = {"h3": True, "seedance": True, "generic": False}


_PASS2_BY_FORMAT = {
    "h3": _PASS2_H3,
    "h3-ref": _PASS2_H3_REF,
    "seedance": _PASS2_SEEDANCE,
    "generic": _PASS2_GENERIC,
}

FORMAT_LABELS = {
    "h3": "T2VA 三字段（从零生成）",
    "h3-ref": "Ref2VA 六段式（格式参考）",
    "seedance": "六要素中文段",
    "generic": "通用分镜表",
}

FORMAT_NOTES = {
    "h3": "三字段：integrated_multimodal_description / overall_soundscape / "
          "non_diegetic_music，带 [Shot N] 切点时间戳，不出现任何参考标签。",
    "h3-ref": "六段式：subject_definitions / summary / retention_analysis / "
              "detailed_description / overall_soundscape / non_diegetic_music。"
              "这里的「参考」指的是**格式**，不是让你参考原视频 —— "
              "原视频只用来提取主体，会定义成 <Subject N> 供你自己挂参考图；"
              "不会出现 <Video 1> / <Audio 1>。",
    "seedance": "Seedance 2.0 / 即梦。中文连贯段落，按 主体→动作→环境→风格→镜头→声音 顺序，"
                "不用字段名，镜头节拍覆盖全部镜头，末尾附负面指令。",
    "generic": "工具无关的分镜脚本，含整体风格、逐镜分镜表、声音设计、负面提示词。",
}


def mode_of(fmt: str) -> str:
    """把 format 归到所属模式。未知值按 generic 处理，不抛异常。"""
    return MODE_OF_FORMAT.get(fmt, "generic")


def format_display(fmt: str) -> str:
    """给日志和界面用的完整称呼，如「H3 模式 · Ref2VA 六段式（带参考素材）」。"""
    mode = mode_of(fmt)
    variant = FORMAT_LABELS.get(fmt, fmt)
    label = MODE_LABELS.get(mode, mode)
    # 单变体模式下前缀是冗余的（「Seedance 模式 · 六要素中文段」读起来还行，
    # 但「通用分镜表 · 通用分镜表」就重复了）
    return label if variant == label else f"{label} · {variant}"


def _language_instruction(fmt: str, language: str) -> str:
    if fmt == "seedance":
        return "Chinese (简体中文)"
    if language == "zh":
        return "Chinese (简体中文), except that dialogue and lyrics stay in their original language"
    return "English, except that dialogue and lyrics stay in their original language"


def build_format_block(fmt: str, language: str) -> str:
    """目标格式规则（H3 三字段 / Ref2VA 六段 / Seedance 六要素 / generic），不含公共规则。

    自由发挥模式只带这一段 —— 格式是**输出契约**，必须精确；其余都是建议。
    """
    lang_instr = _language_instruction(fmt, language)
    template = _PASS2_BY_FORMAT.get(fmt) or _PASS2_GENERIC
    return template.format(common="", language_instruction=lang_instr).strip()
# ---------------------------------------------------------------------------
# 自由发挥模式（默认）
# ---------------------------------------------------------------------------
#
# 帧 + 时间戳 + 用户说明 → 一次调用直接出提示词。
#
# ⚠️ 刻意**不写**「怎么观察」的规则。那些规则曾经占 Pass1 的 14000 字符，
# 而实测每收紧一次、输出就退化一次：
#   * 「只报告帧里真实存在的」压掉了合理推断 → 动作描述占比只剩 19%，读起来像照片说明
#   * 把「相机静止」当结论喂进去 → 模型外推成「人物也静止」，写出一堆 stands
#   * 加「动作占大头」的配比 → 又得再写一段「不许编动作」去中和它
#   * 告诉它「20 秒 MV 大约 5-20 个镜头」→ 快切素材被压到 20 以内
#
# **约束输出格式 ≠ 约束思考。** 这里只留两样：目标格式（输出契约），以及两条关于
# **产物**的硬约束 —— 静止动词会让生成的视频冻住（包括嘴），音频未知时编造比留空更糟。

_FREEFORM_OPENING = """You are a professional storyboard artist. The images below are this \
video's storyboard: contact sheets of consecutive frames, sampled at 2 per second, in order. \
The user message gives you the total duration and which time span each sheet covers.

**Infer the motion between them**: read what moved, how far, and in which direction, and write \
that motion into the shot. **A shot spans time and covers several frames — do not emit one shot \
per frame.** Group the frames into shots the way the video is actually cut, then describe what \
plays out inside each one.

Write the generation prompt for the target video model below, reproducing the original video's \
content as faithfully as you can.

Two constraints on the wording — they are about the artifact, not about how you reason:
* Never use static or terminal verbs (stays, holds, remains, freezes, pauses, ends, final pose) \
— they make the generated video freeze, including the mouth.
* If the audio report says the content is unknown, do not write dialogue, lyrics or specific \
sound effects. Leave the field empty or write N/A; a plausible-sounding guess is not."""


def build_system(fmt: str, language: str) -> str:
    """自由发挥模式：身份 + 要干嘛 + 目标格式规则。"""
    return _FREEFORM_OPENING + "\n\n" + build_format_block(fmt, language)


def build_closing() -> str:
    """收尾指令：明确只要提示词本身，不要前后缀。"""
    return (
        "Now write the final prompt described in your instructions, in the target format. "
        "Cover every shot you judge the segment to have, in order. Output only the prompt "
        "itself — no commentary, no preamble, no markdown fence."
    )
