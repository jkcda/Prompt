"""提示词引擎：两阶段反推的 Prompt 模板。

为什么分两阶段：
    一次性把 40 多张帧丢给模型让它「直接写一段提示词」，模型会把注意力花在
    措辞上，导致细节大面积丢失（尤其是次要镜头、背景细节、运镜方向）。
    拆成 Pass1「只看不写」的结构化观察 + Pass2「只写不看」的成文，
    观察质量立刻上一个档，而且同一份观察结果能出多种目标格式。

    Pass1: 帧 + 音频转写 → 逐镜头结构化 JSON（信息保全）
    Pass2: 观察 JSON + 音频 + 全局统计 → 目标格式提示词（措辞与结构）
"""

from __future__ import annotations

import json

from ..schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation

# ---------------------------------------------------------------------------
# Pass 1 —— 结构化镜头观察
# ---------------------------------------------------------------------------

PASS1_SYSTEM = """You are a senior film analyst and prompt engineer. You will receive a \
sequence of still frames sampled from ONE continuous video segment, each labelled with its \
exact timestamp, plus the audio transcript with timestamps.

Your ONLY job is faithful, exhaustive observation. You do NOT write marketing copy, you do \
NOT invent anything that is not visible or audible, and you do NOT summarise.

Hard rules:
1. Report ONLY what the frames actually show. If something is ambiguous or occluded, say so \
and lower the `confidence` value. Never fill gaps with plausible-sounding guesses.
2. Describe PHYSICS, not just nouns. A person who moves must have their motion described with \
its physical consequences: how weight shifts, how hair swings, how fabric ripples, how \
accessories react to gravity. Movement that has no physical description reads as frozen.
3. Use MOTION VERBS, never static verbs. Write "she rises and turns", "the camera tracks left", \
"his shoulders roll with the step". Never write "she stays", "the pose holds", "remains still", \
"hands rest" — those words make generated video freeze.
4. Estimate camera movement from how the framing changes between frames (subject position, \
background parallax, horizon tilt). Distinguish: static / pan / tilt / dolly-in / dolly-out / \
truck / crane / handheld shake / orbit / whip.
5. Transcribe any on-screen text, captions, subtitles or logos VERBATIM into `on_screen_text`. \
If a watermark or platform logo is present, note it. If there is none, write "none".
6. Align the audio transcript to the shots by timestamp. Put spoken words or lyrics that fall \
inside a shot into that shot's `dialogue` field, preserving the original language. Put \
non-verbal sound (footsteps, impact, whoosh, ambience) into `sfx`.
7. `timecode` must be the real timestamp of that shot's first supplied frame, formatted as \
MM:SS.mmm.

Output STRICT JSON only, no markdown fence, no commentary, matching exactly this shape:

{
  "shots": [
    {
      "shot": "1",
      "timecode": "00:00.000",
      "shot_size": "extreme close-up | close-up | medium close-up | medium | medium wide | wide | extreme wide",
      "camera": "movement type + amplitude + speed, e.g. 'static' / 'slow dolly-in, small amplitude'",
      "subject": "who or what is in frame, with appearance, wardrobe, key props",
      "action": "what happens, described with physical detail and motion verbs",
      "setting": "environment, location, background elements, depth",
      "lighting": "light source, direction, quality, contrast",
      "color": "palette, grade, contrast, saturation",
      "motion_energy": "low | medium | high, plus the rhythm or beat the motion follows",
      "on_screen_text": "verbatim text or 'none'",
      "dialogue": "spoken words or lyrics in this shot, original language, or empty string",
      "sfx": "non-verbal sounds in this shot",
      "transition": "how the shot ends / how it hands off to the next shot",
      "confidence": 0.0
    }
  ],
  "global_notes": "cross-shot observations: overall style, recurring subjects, wardrobe continuity, \
colour consistency, pacing pattern, anything the per-shot fields cannot capture"
}"""


def build_pass1_user(
    chunk_start: float,
    chunk_end: float,
    frame_marks: list[tuple[float, str]],
    audio_text: str,
    media: MediaInfo | None,
    chunk_index: int,
    chunk_total: int,
) -> str:
    """组织 Pass1 的用户消息文本（图片由调用方按顺序附在后面）。"""
    lines: list[str] = []
    lines.append(
        f"This is segment {chunk_index + 1} of {chunk_total}, covering "
        f"{chunk_start:.2f}s to {chunk_end:.2f}s of the source video."
    )
    if media:
        res = f"{media.width}x{media.height}" if media.width else "unknown"
        lines.append(f"Source video: {res}, {media.fps:.2f} fps, {media.duration:.2f}s total.")
    lines.append("")
    lines.append(f"The {len(frame_marks)} images attached AFTER this text are in this exact order:")
    for i, (t, role) in enumerate(frame_marks, start=1):
        lines.append(f"  Image {i} -> timestamp {t:.3f}s [role: {role}]")
    lines.append("")
    lines.append("Use the timestamp list above to group images into shots. A shot boundary is "
                 "where the framing, subject, location or lighting changes abruptly.")
    lines.append("")
    lines.append(audio_text)
    lines.append("")
    lines.append(
        "Now output the strict JSON described in your instructions. "
        "Be exhaustive about physical detail and camera movement. Output JSON only."
    )
    return "\n".join(lines)


def parse_pass1_json(raw: str) -> tuple[list[ShotObservation], str]:
    """宽容解析 Pass1 的 JSON 输出（模型偶尔会包 markdown 围栏或加前后缀）。"""
    text = (raw or "").strip()

    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return [], ""

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # 尝试修掉尾随逗号
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return [], ""

    shots_raw = data.get("shots") or []
    shots: list[ShotObservation] = []
    for item in shots_raw:
        if not isinstance(item, dict):
            continue
        try:
            shots.append(ShotObservation(**{
                k: (v if v is not None else "")
                for k, v in item.items()
                if k in ShotObservation.model_fields
            }))
        except Exception:  # noqa: BLE001
            continue

    return shots, str(data.get("global_notes") or "")


# ---------------------------------------------------------------------------
# Pass 2 —— 成文
# ---------------------------------------------------------------------------

_COMMON_RULES = """You are a prompt engineer writing the FINAL generation prompt for a video \
model, based on a shot-by-shot observation report of an existing video.

Absolute rules — these override everything else:
1. Use ONLY information present in the observation report and the audio report. Never invent \
new subjects, locations, props or events.
2. Describe MOTION with physical consequence. Every moving subject must carry its physics: \
weight shift, hair swing, fabric ripple, accessory sway, contact with the ground. A subject \
with no motion description renders as a frozen mannequin.
3. NEVER use static or terminal verbs: stop, freeze, hold, pause, rest, stay, remain, settle, \
end, ending, final frame, final pose, holds. Never declare that any motion is the only motion \
in frame. If you write "the only movement is X", everything else freezes, including the mouth.
4. When a person speaks or sings on camera, their mouth movement must be described explicitly \
in that shot: which words or syllables the lips move through, and that the movement is \
continuous through the line. Never place a dialogue line in a shot without stating the mouth \
movement.
5. Preserve the original language of dialogue and lyrics. Mark unintelligible spans as [unclear]. \
Never paraphrase dialogue.
6. Do not mention watermarks, logos, platform UI, or the fact that this is a reverse-engineered \
prompt. If the observation notes on-screen text, either transcribe it as diegetic text when it \
is part of the scene, or omit it.
7. Write in {language_instruction}."""


_PASS2_H3 = """{common}

TARGET FORMAT — MiniMax H3 T2VA (text-to-video with audio). Output exactly three fields, in \
this order, each starting at the beginning of a line with its bare name followed by a colon. \
Do NOT wrap the field names in angle brackets or any other markup.

integrated_multimodal_description: <the main body>
overall_soundscape: <ambience and physical sounds>
non_diegetic_music: <audience-only score, or N/A>

Rules for `integrated_multimodal_description`:
- Begin with one or two sentences establishing the overall style, format and grade of the video, \
BEFORE the first shot marker.
- Then write every shot as its own paragraph starting with `[Shot 1]`, `[Shot 2]`, and so on. \
`[Shot 1]` carries no timestamp; every later shot starts `[Shot N] At MM:SS.mmm, ...` using the \
cut time from the observation report.
- Inside each shot, establish in this order: framing and camera movement, subject appearance and \
position, environment and lighting, the action with its physical detail, and the current sound.
- Assign speakers stable IDs `(S1)`, `(S2)` in order of first vocal event. Write dialogue and \
lyrics as `<d>[Language] the exact words</d>`. When a speaker is on camera, state their mouth \
movement in the same shot.
- Length: 350-500 words. Distribute detail by information load, not evenly.
- The description must not end with any closing or resolution marker. It ends on the last \
described action, mid-flow.

Rules for `overall_soundscape`:
- Summarise ambience and physical action sounds across the whole video in one or two sentences.
- Do NOT repeat dialogue or lyrics here.

Rules for `non_diegetic_music`:
- Describe audience-only score with instrumentation, tempo and dynamics. Write `N/A` if the \
observation report indicates there is no score."""


_PASS2_H3_REF = """{common}

TARGET FORMAT — MiniMax H3 full-reference (Ref2VA) rewrite. The uploaded video is the single \
reference asset, labelled `<Video 1>`; if it has usable audio, that track is `<Audio 1>`. \
Output exactly six sections, in this order, each starting at the beginning of a line with its \
bare name followed by a colon. Do NOT wrap field names in angle brackets or any other markup.

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Rules:
- `subject_definitions`: one line per tracked item. Use `<Subject 1>`, `<Subject 2>` ... for \
reusable visible content (people, environments, props, wardrobe, style). Define `<Video 1>` as \
the source video. Define `<Audio 1>` only if its audio is actually reused. Each line states what \
the label denotes, its reference role, and its main features to follow.
- `summary`: one short paragraph beginning with a bracketed task-type prefix, e.g. \
`[video continuation + reference generation]` or `[reference generation + audio reference]`. \
Do not introduce new labels here.
- `retention_analysis`: one line per label, using the fixed markers — visible content: \
`fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`; audio: \
`fully_copy` / `partially_copy` / `reference` / `weak_reference`. Format: \
`<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...`
- `detailed_description`: the main body. One or two sentences of style before `[Shot 1]`. Then \
`[Shot 1]` with no timestamp, and `[Shot N] At MM:SS.mmm, ...` for later shots. Insert reference \
labels at first appearance and wherever their role applies. Speakers use `(Sx)` and dialogue uses \
`<d>[Language] ...</d>`. 350-500 words.
- `overall_soundscape` / `non_diegetic_music`: ambience and physical sound vs. audience-only \
score. Write `N/A` when a category is absent. Never repeat dialogue here."""


_PASS2_SEEDANCE = """{common}

TARGET FORMAT — Seedance 2.0 (doubao-seedance). Write ONE coherent natural-language prompt in \
CHINESE, following the official six-element order: 主体 → 动作 → 环境 → 风格 → 镜头 → 声音. \
Do NOT use field labels, do NOT use `[Shot N]` markers, do NOT use H3's colon structure.

Rules:
- Write it as flowing prose, not a bullet list. Roughly: 主体+动作 occupies about half the \
length, then one sentence each for 环境, 风格, 镜头, 声音.
- Be concrete: only describe what is visible or audible. Never write vague words like 漂亮、高级、\
电影感十足.
- 镜头 sentence: express shot sizes and camera movement as a continuous progression, e.g. \
「以全景开场，随后缓慢推轨至中近景，浅景深」. If the video does not cut at all, say 全片不切镜. \
If it does cut, state the shot count and beat map at the end of the camera sentence, e.g. \
「全片四个镜头，节拍为 0–2.5 秒、2.5–5 秒、5–7.5 秒、7.5–10 秒」.
- 声音 sentence: list ambience, action sounds, music and dialogue compactly, separated by 分号. \
If there is no dialogue, end with 无对白. Append the score description at the very end.
- Add explicit negative instructions on their own at the end when the observation report shows \
text, subtitles, logos or watermarks: 「不要出现字幕、文字、水印」.
- Do not write duration or aspect ratio into the prompt — those are separate parameters.
- Keep dialogue and lyrics in their original language, quoting them inline."""


_PASS2_GENERIC = """{common}

TARGET FORMAT — a neutral, tool-agnostic storyboard prompt sheet. Write in \
{language_instruction} using this structure:

【整体风格】one paragraph: medium, genre, grade, palette, lighting logic, aspect feel, pacing.

【镜头分镜】
For each shot, one block:
镜头 N｜MM:SS.mmm–MM:SS.mmm
  景别 / 角度：
  运镜：
  画面内容：subject, action with physical detail, environment, lighting
  台词 / 人声：exact words in original language, or 无
  音效：
  转场：

【声音设计】ambience, action sound, and audience-only score, described separately.

【负面提示词】a comma-separated list of things that must not appear, derived from the \
observation report (text overlays, watermarks, unwanted artefacts, identity drift, etc.).

Do not add a closing or summary section after the negative prompt."""


_PASS2_BY_FORMAT = {
    "h3": _PASS2_H3,
    "h3-ref": _PASS2_H3_REF,
    "seedance": _PASS2_SEEDANCE,
    "generic": _PASS2_GENERIC,
}

FORMAT_LABELS = {
    "h3": "MiniMax H3（T2VA 三字段）",
    "h3-ref": "MiniMax H3（Ref2VA 六段式）",
    "seedance": "Seedance 2.0（六要素中文）",
    "generic": "通用分镜表",
}

FORMAT_NOTES = {
    "h3": "从零生成用。三个字段：integrated_multimodal_description / overall_soundscape / "
          "non_diegetic_music，带 [Shot N] 切点时间戳。反推场景的默认选择。",
    "h3-ref": "带参考素材时用。六段式：subject_definitions / summary / retention_analysis / "
              "detailed_description / overall_soundscape / non_diegetic_music，"
              "把上传的视频作为 <Video 1> 引用。",
    "seedance": "Seedance 2.0 / 即梦。中文连贯段落，按 主体→动作→环境→风格→镜头→声音 顺序，"
                "不用字段名，末尾附负面指令。",
    "generic": "工具无关的分镜脚本，含整体风格、逐镜分镜表、声音设计、负面提示词。"
               "适合人工二次加工或投喂给其他工具。",
}


def build_pass2_system(fmt: str, language: str) -> str:
    if fmt == "seedance":
        lang_instr = "Chinese (简体中文)"
    elif language == "zh":
        lang_instr = "Chinese (简体中文), except that dialogue and lyrics stay in their original language"
    else:
        lang_instr = "English, except that dialogue and lyrics stay in their original language"

    template = _PASS2_BY_FORMAT.get(fmt) or _PASS2_GENERIC
    return template.format(common=_COMMON_RULES, language_instruction=lang_instr)


def build_pass2_user(
    observations: list[ChunkObservation],
    audio: AudioReport,
    media: MediaInfo | None,
    shots_summary: str,
    extra_instruction: str = "",
    target_duration: float | None = None,
) -> str:
    """把 Pass1 的观察结果整理成 Pass2 的输入。"""
    lines: list[str] = []

    lines.append("=== SOURCE VIDEO ===")
    if media:
        lines.append(
            f"duration {media.duration:.2f}s | resolution {media.width}x{media.height} | "
            f"{media.fps:.2f} fps | video codec {media.video_codec or 'unknown'} | "
            f"audio {'present' if media.has_audio else 'absent'}"
        )
    lines.append(f"shot structure: {shots_summary}")
    if target_duration:
        lines.append(f"target duration for the generated video: {target_duration:.2f}s")
    lines.append("")

    lines.append("=== SHOT OBSERVATION REPORT ===")
    total = 0
    for chunk in observations:
        if len(observations) > 1:
            lines.append(f"--- segment {chunk.chunk_index + 1} "
                         f"({chunk.start:.2f}s - {chunk.end:.2f}s) ---")
        for shot in chunk.shots:
            total += 1
            lines.append(_format_shot(shot))
        if chunk.global_notes:
            lines.append(f"  segment notes: {chunk.global_notes}")
        lines.append("")

    lines.append("=== AUDIO REPORT ===")
    from .asr import format_transcript_for_prompt
    lines.append(format_transcript_for_prompt(audio))
    lines.append("")

    if extra_instruction:
        lines.append("=== USER'S EXTRA INSTRUCTION ===")
        lines.append(extra_instruction)
        lines.append("")

    lines.append(
        f"Now write the final prompt covering ALL {total} observed shots, in the target format "
        "defined in your instructions. Follow every absolute rule. "
        "Do not add any commentary, preamble, or explanation — output only the prompt itself."
    )
    return "\n".join(lines)


def _format_shot(s: ShotObservation) -> str:
    parts = [f"[Shot {s.shot or '?'}] @ {s.timecode or '?'}"]
    for label, value in (
        ("size", s.shot_size),
        ("camera", s.camera),
        ("subject", s.subject),
        ("action", s.action),
        ("setting", s.setting),
        ("light", s.lighting),
        ("color", s.color),
        ("energy", s.motion_energy),
        ("on-screen text", s.on_screen_text),
        ("dialogue", s.dialogue),
        ("sfx", s.sfx),
        ("transition", s.transition),
    ):
        if value and str(value).strip():
            parts.append(f"    {label}: {str(value).strip()}")
    if s.confidence:
        parts.append(f"    confidence: {s.confidence}")
    return "\n".join(parts)


def merge_observations(chunks: list[ChunkObservation]) -> list[ShotObservation]:
    """把多个分块的观察结果合并，并重排镜号（分块会产生重复镜号）。"""
    merged: list[ShotObservation] = []
    for chunk in sorted(chunks, key=lambda c: c.start):
        for shot in chunk.shots:
            item = shot.model_copy()
            item.shot = str(len(merged) + 1)
            merged.append(item)
    return merged
