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
import re
from dataclasses import dataclass, field

from ..schemas import AudioReport, ChunkObservation, MediaInfo, ShotObservation, SubjectEntry

# ---------------------------------------------------------------------------
# Pass 1 —— 结构化镜头观察
# ---------------------------------------------------------------------------

_PASS1_EDIT_STRUCTURE_RULES = """
HOW TO DECIDE THE SHOT STRUCTURE — read this carefully

The frame timestamps you receive were chosen by an automated scene-change detector.
**That detector is a sampling heuristic, not ground truth.** It splits on pixel
difference, so it will:

  * split a single continuous take into several "shots" whenever the camera moves
    fast, something large enters frame, or the exposure changes
  * miss real hard cuts that happen during motion or a flash
  * fire several times around one soft transition

So do NOT treat the timestamps as the edit structure. **Judge it yourself from what
you see**, and report it in `edit_structure`:

  * `continuous` — one uninterrupted take. The camera may move, the subject may move,
    the framing may change — but there is no instant where the image jumps to a
    different setup. Continuous movement, continuous lighting, continuous subject
    position across the boundary = still one shot.
  * `multi_shot` — there are real cuts: an instant where the frame content changes
    discontinuously (different setup, jump in position, hard change of light).

If it is `continuous`, emit **ONE** entry in `shots` covering the whole segment, even
if the frames you were given span several detector groups. Then list `cut_points` as
an empty array. Getting this right matters: a `[Shot N]` marker in the final prompt
tells the video model to cut there, so over-splitting a continuous take produces a
choppy result that does not match the source.

If it is `multi_shot`, put the real cut timecodes in `cut_points` (MM:SS.mmm), and
make the `shots` array follow those cuts — again, not the detector's grouping.

When you genuinely cannot tell, use `unknown`, keep the detector's grouping, and say
so in `continuity_notes`. Never invent a cut you cannot see.
"""


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
   Also never declare that one motion is the ONLY motion in frame (e.g. "the only motion is the \
shifting bar"). Everything you do not mention as moving stops moving, including the mouth. \
Your `global_notes` is passed verbatim into the next stage, so forbidden phrasing written here \
contaminates the final prompt. If little is moving, say what IS moving — do not rank it as the \
sole motion.
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
8. After the per-shot list, build a SUBJECT REGISTRY. A subject is anything that must look the \
same if it reappears: a person, an environment, a prop, a wardrobe piece, or the overall visual \
style itself. Give each one a short lowercase label, list every shot index it appears in, and \
spell out the features that must stay consistent. If the same person or place appears in \
several shots, it must be ONE entry covering all of them — not one entry per shot. This registry \
is what lets the final prompt keep identities from drifting, so be precise about the traits that \
are easy to get wrong (eye colour, hair length and parting, garment cut and hardware, tattoo \
placement, the exact grade).
   A response whose `subjects` array is missing or empty is INCOMPLETE and will be rejected. \
Every distinct recurring subject in the footage must appear in it. Emit `subjects` BEFORE \
`shots` so you do not run out of output budget before writing it.
   Do NOT register watermarks, platform logos, channel bugs, UI overlays, subtitles or \
burned-in captions as subjects. They are artefacts of the source file, not content to \
reproduce — registering them invites the generator to render them into the new video. \
Mention them only in `on_screen_text`, never in `subjects`.

""" + _PASS1_EDIT_STRUCTURE_RULES + """

Output STRICT JSON only, no markdown fence, no commentary, matching exactly this shape \
(`subjects` first, then `shots`):

{
  "subjects": [
    {
      "label": "short lowercase label, e.g. performer / rooftop / jacket / grade",
      "kind": "person | environment | prop | wardrobe | style | other",
      "description": "appearance, material, colour, identifying features",
      "shots": ["1", "2", "5"],
      "notes": "what must stay consistent, and which traits are prone to drifting"
    }
  ],
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
  "edit_structure": "continuous | multi_shot",
  "cut_points": ["00:03.400", "00:07.900"],
  "continuity_notes": "why you judge the edit structure this way",
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
    content_hint: str = "",
) -> str:
    """组织 Pass1 的用户消息文本（图片由调用方按顺序附在后面）。

    `content_hint` 是用户自己写的画面说明。**这东西很有用**：静态帧看不出
    「这段是一镜到底还是多镜头切换」「这是什么作品/角色」「动作的前因后果」，
    而这些直接影响产出质量。让用户补一句话，比让模型瞎猜强得多。
    """
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
    lines.append(
        "These timestamps come from an automated scene-change detector. **They are a "
        "sampling aid, not the edit structure** — the detector splits on pixel "
        "difference, so it routinely breaks one continuous take into several groups "
        "and can miss real cuts. Decide the actual shot structure yourself from what "
        "you see, and report it in `edit_structure` / `cut_points`."
    )
    lines.append("")
    lines.append(audio_text)

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
    lines.append(
        "Now output the strict JSON described in your instructions. "
        "Be exhaustive about physical detail and camera movement. Output JSON only."
    )
    return "\n".join(lines)


@dataclass
class Pass1Parse:
    """Pass1 的解析结果。

    做成 dataclass 而不是越来越长的元组 —— 加「剪辑结构」这类字段时
    不用把所有调用点都改一遍解包顺序。
    """

    shots: list[ShotObservation] = field(default_factory=list)
    subjects: list[SubjectEntry] = field(default_factory=list)
    global_notes: str = ""
    # 模型自己判断的剪辑结构（不是我们切出来的）：
    #   continuous = 一镜到底；multi_shot = 有硬切；unknown = 判断不了
    edit_structure: str = "unknown"
    cut_points: list[str] = field(default_factory=list)
    continuity_notes: str = ""


def parse_pass1_json(raw: str) -> Pass1Parse:
    """宽容解析 Pass1 的 JSON 输出（模型偶尔会包 markdown 围栏或加前后缀）。

    返回 `(逐镜头观察, 主体登记表, 跨镜头备注)`。解析不出来时返回空列表，
    由调用方决定是报错还是降级——不要在这里抛异常，一块失败不该拖垮整条管线。
    """
    text = (raw or "").strip()

    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return Pass1Parse()

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # 尝试修掉尾随逗号
        import re
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return Pass1Parse()

    return Pass1Parse(
        shots=_parse_shots(data.get("shots")),
        subjects=_parse_subjects(data.get("subjects")),
        global_notes=str(data.get("global_notes") or ""),
        edit_structure=_normalize_structure(data.get("edit_structure")),
        cut_points=_parse_cut_points(data.get("cut_points")),
        continuity_notes=str(data.get("continuity_notes") or ""),
    )


def _normalize_structure(value: object) -> str:
    """把模型写的各种说法归一到 continuous / multi_shot / unknown。"""
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return "unknown"
    if any(k in s for k in ("continuous", "one_take", "single_take", "oners", "oner", "no_cut")):
        return "continuous"
    if any(k in s for k in ("multi", "cut", "edited", "montage")):
        return "multi_shot"
    return "unknown"


def _parse_cut_points(raw: object) -> list[str]:
    """切点列表。模型可能给字符串数组，也可能给逗号分隔的字符串。"""
    if isinstance(raw, str):
        parts = re.split(r"[,;\n]", raw)
    elif isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        return []
    out: list[str] = []
    for p in parts:
        p = p.strip()
        # 只要看起来像时间码的（MM:SS.mmm / HH:MM:SS.mmm）
        if p and re.match(r"^\d{1,2}:\d{2}(\.\d{1,3})?$", p):
            out.append(p)
    return out


def _parse_shots(shots_raw: object) -> list[ShotObservation]:
    if not isinstance(shots_raw, list):
        return []
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
    return shots


def _parse_subjects(subjects_raw: object) -> list[SubjectEntry]:
    """解析主体登记表。`shots` 可能是列表也可能是逗号串，两种都收。"""
    if not isinstance(subjects_raw, list):
        return []
    subjects: list[SubjectEntry] = []
    for item in subjects_raw:
        if not isinstance(item, dict):
            continue
        shots_field = item.get("shots")
        if isinstance(shots_field, str):
            shot_list = [s.strip() for s in shots_field.replace("，", ",").split(",") if s.strip()]
        elif isinstance(shots_field, list):
            shot_list = [str(s).strip() for s in shots_field if str(s).strip()]
        else:
            shot_list = []

        label = str(item.get("label") or "").strip()
        description = str(item.get("description") or "").strip()
        if not label and not description:
            continue  # 整条都是空的，丢掉比留着干净

        subjects.append(SubjectEntry(
            label=label,
            kind=str(item.get("kind") or "").strip(),
            description=description,
            shots=shot_list,
            notes=str(item.get("notes") or "").strip(),
        ))
    return subjects


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
7. Audio: distinguish CONTENT from SPECTRUM. If the audio report says the audio was not \
transcribed, you do not know what the audio contains — never write dialogue, lyrics, music \
instrumentation, tempo, or specific sound-effect types, not even hedged ("as if", "appears to \
be"). Spectral measurements (band energy, volume, silence, beat points) ARE valid input: you may \
write a hedged tendency based on them, e.g. "the track reads as voice-dominant with little \
low-frequency content", and you must make clear it is inferred from energy distribution, not \
identified. Never upgrade a spectral tendency into a concrete claim — "voice-dominant energy" \
does not license "someone is singing".
   Do NOT convert visual events into sound events either. Seeing a person walk does not license \
"footsteps"; seeing fabric move does not license "cloth rustle". Those are inferences about audio \
content, which is exactly what is unknown. Describe the visual action itself and leave the sound \
unspecified.
   Fabricated sound is worse than an empty field: the generated video will not match the source.
8. Keep each subject's appearance wording IDENTICAL across shots. A character described as "a \
performer in a dark quilted jacket" in shot 1 must not become "a woman in a leather coat" in \
shot 3 — inconsistent wording is read as a different person and the identity drifts.
9. Write in {language_instruction}."""


_PASS2_H3 = """{common}

TARGET MODE — MiniMax H3, text-to-video with audio (T2VA). There are NO reference assets in this \
task, so the prompt must be fully self-contained: everything the model needs is in the text.

Output exactly three fields, in this order, each starting at the beginning of a line with its \
bare name followed by a colon. Do NOT wrap the field names in angle brackets or any other markup.

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
- Because there are no reference assets, appearance must be stated in full the first time a \
subject appears, and the same wording must be reused for that subject afterwards. Never write \
`<Subject 1>`, `<Video 1>`, `<Audio 1>` or any other reference label — they are meaningless in \
T2VA and will be read as literal text.
- Assign speakers stable IDs `(S1)`, `(S2)` in order of first vocal event. Write dialogue and \
lyrics as `<d>[Language] the exact words</d>`. When a speaker is on camera, state their mouth \
movement in the same shot.
- **LENGTH BUDGET — hard requirement.** The generated video model has a limited prompt window. \
Keep `integrated_multimodal_description` to **420 words or fewer**. Oversized prompts get \
truncated or ignored by the generator. Write tight, information-dense sentences — no filler, no \
restating the style, no repeating a subject's appearance after its first mention. If you are \
running long, cut adjectives and scene-setting, never the action or the mouth movement.
- The description must not end with any closing or resolution marker. It ends on the last \
described action, mid-flow.

Rules for `overall_soundscape`:
- Summarise ambience and physical action sounds across the whole video in one or two sentences.
- Do NOT repeat dialogue or lyrics here.

Rules for `non_diegetic_music`:
- Describe audience-only score with instrumentation, tempo and dynamics. Write `N/A` if the \
observation report indicates there is no score."""


_PASS2_H3_REF = """{common}

TARGET MODE — MiniMax H3 full-reference (Ref2VA) FORMAT. "Reference" here means the prompt \
FORMAT, not the source video. The source video is only being MINED for content — it is NOT a \
reference asset and must NOT be cited as one.

That means:
- Do NOT define or mention `<Video 1>` or `<Audio 1>`. There is no video reference and no audio \
reference in this task.
- The `<Subject N>` labels are the reusable content you extracted from the footage. The user will \
supply their OWN reference images for these labels, so each definition must be self-contained \
enough to identify the thing from the text alone.

A SUBJECT REGISTRY is provided in the user message. Use it as the authoritative source for labels \
— do not invent subjects that are not in it, and do not split one registry entry into several.

Output exactly six sections, in this order, each starting at the beginning of a line with its \
bare name followed by a colon. Do NOT wrap field names in angle brackets or any other markup.

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

## LENGTH BUDGET — this is a hard requirement, not a suggestion

The generated video model has a limited prompt window. **The entire output must be under \
700 words.** Oversized prompts get truncated or ignored by the generator, which makes the whole \
rewrite useless. Hit these per-section budgets:

| section | budget |
|---|---|
| `subject_definitions` | **at most 6 entries**, 15 words each — see selection rule below |
| `summary` | 40 words |
| `retention_analysis` | one line per subject, 15 words each |
| `detailed_description` | **420 words** — the bulk of the budget belongs here |
| `overall_soundscape` | 40 words |
| `non_diegetic_music` | 25 words |

If you are running long, cut words from the definition and analysis lines, never from \
`detailed_description`. Write tight, information-dense sentences — no filler, no restating the \
style, no repeating a subject's appearance after its first mention.

**Subject selection rule**: the registry may list more than 6 items. Pick at most 6 — the ones \
that most need a reference image, in this priority: people > wardrobe/props > environment > \
style/grade. Drop the least important ones entirely rather than giving everyone a half-line.

## Section rules

- `subject_definitions`: one line per SELECTED registry entry, numbered in registry order as \
`<Subject 1>`, `<Subject 2>`, ... Each line: what the label denotes, then its identifying \
features, in 15 words or fewer. Registry entries whose `kind` is `style` should be defined as the \
look and grade to carry across, not as an object. Do not add a `<Video 1>` or `<Audio 1>` line.
- `summary`: one short paragraph beginning with a bracketed task-type prefix, e.g. \
`[reference generation]` or `[reference generation + style transfer]`. Do not introduce new \
labels here, and do not describe the source clip as a reference.
- `retention_analysis`: one line per `<Subject N>` label ONLY — no video or audio line. Use the \
fixed markers: `fully_preserved` / `partially_preserved` / `attribute_transfer` / \
`weak_reference`. Format: `<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...` \
with the shot list taken verbatim from the registry entry. The reason after the dash must be \
15 words or fewer. Every tracked subject is `fully_preserved` or `partially_preserved` unless \
the observation report says its appearance changes.
- `detailed_description`: the main body, 420 words. One or two sentences of style before \
`[Shot 1]`. Then `[Shot 1]` with no timestamp, and `[Shot N] At MM:SS.mmm, ...` for later shots. \
Insert subject labels at first appearance and wherever their role applies. Speakers use `(Sx)` \
and dialogue uses `<d>[Language] ...</d>`.
- `overall_soundscape` / `non_diegetic_music`: ambience and physical sound vs. audience-only \
score, 40 / 25 words. Write `N/A` when a category is absent. Never repeat dialogue here."""


_PASS2_SEEDANCE = """{common}

TARGET MODE — Seedance 2.0 / 即梦. Write ONE coherent natural-language prompt in CHINESE. The \
official element order is 主体 → 动作 → 环境 → 风格 → 镜头 → 声音.

Hard constraints for this mode:
- Write in Chinese only. Do not use English words except for proper nouns and on-screen text.
- Do NOT use field labels. Do NOT use `[Shot N]` markers. Do NOT use H3's colon structure or any \
`<Subject N>` / `<d>` markup. This mode is plain prose, not a structured document.
- Write flowing prose, not a bullet list. Roughly: 主体 + 动作 take about half the length, then \
one sentence each for 环境, 风格, 镜头, 声音.
- Be concrete. 主体 must be pinned down by colour, material, garment cut and identifying features, \
because there are no reference images in this mode — vague words like 漂亮、高级、电影感十足 \
carry no information and are forbidden.
- 动作 must carry its physical consequence: 重心转移、头发摆动、衣料起伏、配饰晃动、与地面的接触。\
A subject without motion description renders as a frozen mannequin.
- 镜头 sentence: express shot sizes and camera movement as a continuous progression, e.g. \
「以全景开场，随后缓慢推轨至中近景，浅景深」. If the video does not cut at all, say 全片不切镜. \
If it does cut, state the shot count and beat map at the end of the camera sentence, and the beat \
map must cover EVERY shot in the observation report without gaps or overlap, e.g. \
「全片四个镜头，节拍为 0–2.5 秒、2.5–5 秒、5–7.5 秒、7.5–10 秒」.
- 声音 sentence: list ambience, action sounds, music and dialogue compactly, separated by 分号. \
If there is no dialogue, end with 无对白. Append the score description at the very end.
- Add explicit negative instructions on their own at the end when the observation report shows \
text, subtitles, logos or watermarks: 「不要出现字幕、文字、水印」.
- Do not write duration or aspect ratio into the prompt — those are separate parameters.
- Keep dialogue and lyrics in their original language, quoting them inline. When a speaker is on \
camera, state their mouth movement in that part of the 动作 description.
- The prompt must not end with a closing or resolution marker."""


_PASS2_GENERIC = """{common}

TARGET MODE — a neutral, tool-agnostic storyboard prompt sheet. Write in \
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
    return f"""You are a ruthless text editor. Shorten the prompt the user sends to \
under {target} words — it must end up comfortably below {limit}.

Keep: the section names present in the original, in order, each on its own line \
as a bare `name:`; every `<Subject N>` label; every `[Shot N]` marker; all retention \
markers; all dialogue verbatim; and `N/A` where present.
Cut: adjectives, filler, repeated descriptions, hedging.

Output only the shortened prompt."""


def build_compress_user(prompt: str, limit: int) -> str:
    words = len(prompt.split())
    target = max(1, int(limit * 0.85))
    cut = max(0, words - target)
    return (
        f"The following prompt is {words} words. Cut at least {cut} words — "
        f"aim for {target} words or fewer (hard ceiling {limit}).\n\n"
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


def build_pass2_system(fmt: str, language: str) -> str:
    if fmt == "seedance":
        lang_instr = "Chinese (简体中文)"
    elif language == "zh":
        lang_instr = "Chinese (简体中文), except that dialogue and lyrics stay in their original language"
    else:
        lang_instr = "English, except that dialogue and lyrics stay in their original language"

    template = _PASS2_BY_FORMAT.get(fmt) or _PASS2_GENERIC
    # 注意：_COMMON_RULES 自己带 {language_instruction} 占位符，而 str.format 不会
    # 递归替换被代入的值——必须先单独 format 一次，否则这段会原样漏给模型。
    common = _COMMON_RULES.format(language_instruction=lang_instr)
    return template.format(common=common, language_instruction=lang_instr)


def build_pass2_user(
    observations: list[ChunkObservation],
    audio: AudioReport,
    media: MediaInfo | None,
    shots_summary: str,
    extra_instruction: str = "",
    target_duration: float | None = None,
    subjects: list[SubjectEntry] | None = None,
    fmt: str = "",
    content_hint: str = "",
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
    lines.append(f"target mode: {MODE_LABELS.get(mode_of(fmt), fmt or 'unknown')}")
    lines.append("")

    # 用户的画面说明也要传给成文阶段。观察结果可能漏掉或误判的东西
    # （尤其「一镜到底 vs 多镜头」这种静态帧判断不了的），成文时要用得上。
    if content_hint.strip():
        lines.append("=== CONTEXT FROM THE USER (reliable background) ===")
        lines.append(content_hint.strip())
        lines.append("")
        lines.append("Use it to inform the rewrite, but the observation report above is "
                     "authoritative about what is visible. If they conflict, follow the "
                     "observation report and do not invent.")
        lines.append("")

    registry = subjects or []
    if registry:
        lines.append("=== SUBJECT REGISTRY (authoritative labels) ===")
        for i, sub in enumerate(registry, start=1):
            shots_txt = ", ".join(f"[Shot {s}]" for s in sub.shots) or "unspecified shots"
            head = f"{i}. {sub.label or 'unnamed'} ({sub.kind or 'unspecified'}) - {shots_txt}"
            lines.append(head)
            if sub.description:
                lines.append(f"     appearance: {sub.description}")
            if sub.notes:
                lines.append(f"     must stay consistent: {sub.notes}")
        lines.append("")
        lines.append(
            "These entries are the only subjects you may reference. In reference-based formats "
            "they become <Subject 1..N> in registry order. In non-reference formats use the same "
            "wording for each one in every shot it appears in."
        )
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

    # 剪辑结构要单独、显眼地说一遍 —— 它决定成文时要不要输出切点标记。
    # 一镜到底的片子如果被写成一堆 [Shot N]，生成的视频就会在原本连续的地方
    # 硬切，和源片完全不是一回事（用户反馈过「镜头连贯性不对」）。
    lines.append("=== EDIT STRUCTURE (decided by the observer, authoritative) ===")
    structures = [c.edit_structure for c in observations if c.edit_structure]
    if "continuous" in structures and "multi_shot" not in structures:
        lines.append("CONTINUOUS — the footage is ONE uninterrupted take. There are no cuts.")
        lines.append("Therefore the description must NOT contain multiple `[Shot N]` markers "
                     "implying cuts. Describe the whole thing as a single continuous shot: "
                     "use one `[Shot 1]` (or none at all) and carry the camera movement, "
                     "subject motion and framing changes through as continuous evolution.")
    elif "multi_shot" in structures:
        lines.append("MULTI_SHOT — there are real cuts.")
        for c in observations:
            if c.cut_points:
                lines.append(f"  segment {c.chunk_index + 1} cut points: {', '.join(c.cut_points)}")
        lines.append("Use `[Shot N]` markers that follow THESE cuts — not the sampling "
                     "timestamps, and not the detector's grouping.")
    else:
        lines.append("UNKNOWN — the observer could not tell. Keep the shot markers "
                     "conservative: only mark a cut where the report describes an "
                     "abrupt change of setup.")
    for c in observations:
        if c.continuity_notes:
            lines.append(f"  segment {c.chunk_index + 1} reasoning: {c.continuity_notes}")
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


def merge_subjects(
    chunks: list[ChunkObservation],
    merged_shots: list[ShotObservation],
) -> list[SubjectEntry]:
    """跨分块合并主体登记表。

    分块会让同一个主体在多个块里各登记一次（例如主角在第 1 块和第 3 块都出现），
    直接拼起来会出现重复标签，Ref2VA 的 retention_analysis 就会写出两条
    `<Subject 1>`。所以这里按标签归一化后合并，并把镜号映射到重排后的全局镜号。

    归一化用 `_subject_key`：小写、去空格与非字母数字，这样 "The Performer" 与
    "performer" 会归成同一条。
    """
    # 块内镜号 → 全局镜号：merge_observations 是按块顺序、块内顺序重排的
    local_to_global: dict[tuple[int, str], str] = {}
    cursor = 0
    for chunk in sorted(chunks, key=lambda c: c.start):
        for shot in chunk.shots:
            cursor += 1
            local_to_global[(chunk.chunk_index, str(shot.shot).strip())] = str(cursor)

    # 全局镜号 → 该镜在重排结果里的下标，用于兜底按主体描述反查
    order = {str(i + 1): s for i, s in enumerate(merged_shots)}

    merged: dict[str, SubjectEntry] = {}
    for chunk in sorted(chunks, key=lambda c: c.start):
        for sub in chunk.subjects:
            key = _subject_key(sub.label) or _subject_key(sub.description[:40])
            if not key:
                continue

            shots: list[str] = []
            for raw in sub.shots:
                gid = local_to_global.get((chunk.chunk_index, raw.strip()))
                if gid is None:
                    gid = raw.strip() if raw.strip().isdigit() else ""
                if gid and gid not in shots:
                    shots.append(gid)

            if key not in merged:
                merged[key] = SubjectEntry(
                    label=sub.label,
                    kind=sub.kind,
                    description=sub.description,
                    shots=shots,
                    notes=sub.notes,
                )
                continue

            existing = merged[key]
            if len(sub.description) > len(existing.description):
                existing.description = sub.description
            if sub.notes and sub.notes not in existing.notes:
                existing.notes = (existing.notes + " " + sub.notes).strip()
            for gid in shots:
                if gid not in existing.shots:
                    existing.shots.append(gid)
            if not existing.kind:
                existing.kind = sub.kind
            if not existing.label:
                existing.label = sub.label

    result = list(merged.values())

    # 兜底：模型忘了写 shots 时，用主体描述里的关键词回查镜号，
    # 否则 retention_analysis 会写不出「appears in [Shot N]」。
    for sub in result:
        if sub.shots or not sub.description:
            continue
        needle = _subject_key(sub.description[:24])
        if not needle:
            continue
        hits = [sid for sid, shot in order.items()
                if needle and needle in _subject_key(shot.subject)]
        if hits:
            sub.shots = sorted(hits, key=lambda x: int(x))

    for sub in result:
        sub.shots = sorted(set(sub.shots), key=lambda x: int(x) if x.isdigit() else 9999)
    return result


def _subject_key(text: str) -> str:
    """归一化主体标签，用于跨块判重。

    先去掉前导冠词再压掉非字母数字：模型会在不同块里把同一个主体写成
    "performer" / "The Performer" / "a performer"，这三种要归成一条，
    否则 retention_analysis 会写出三条 <Subject N>。
    冠词只在后面紧跟空格时才剥，"anime style" 不能被吃成 "imestyle"。
    """
    import re
    raw = re.sub(r"^(the|a|an)\s+", "", (text or "").strip().lower())
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", raw)
