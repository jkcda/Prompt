"""两阶段反推管线。

完整流程：

  1. 探测媒体信息（时长/分辨率/帧率/有无音轨）
  2. 并行做两件事：
       a) 场景检测 → 镜头切分（ffmpeg select=gt(scene,T)）
       b) 音频分析 → 语音转写（带时间戳）+ 音量/静音曲线
  3. 按 token 预算把帧分配到各镜头（首/中/尾，长镜头多给）
  4. 长视频在**镜头边界**处逻辑分块（不物理切割，避免把镜头劈断）
  5. 逐块抽帧 → Pass1：每块并行送多模态模型，产出结构化镜头 JSON
  6. Pass2：合并全部观察 + 音频报告 → 一次性合成目标格式提示词

关键取舍说明：
  - 不按固定 fps 抽帧。60s 视频 fps=10 会产生 600 帧、约 66 万 tokens，必然超上下文。
    按镜头分配预算后是 48 帧、约 5.3 万 tokens，覆盖反而更全。
  - 分块只在镜头边界切，所以不会出现「一个镜头被劈成两半、两边都推不准」。
  - 音频必须进管线：画面帧推不出台词、口型、BGM、音效，而这些是目标格式的硬字段。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from ..core.config import get_settings
from ..schemas import (
    AnalyzeOptions,
    AudioReport,
    FrameRef,
    Job,
    JobResult,
    Shot,
)
from . import asr as asr_mod
from . import audio_features, selection, templates
from . import ffmpeg as ff
from .jobs import store
from .vlm import VLMClient, VLMError

log = logging.getLogger("pipeline")

STAGE_LABELS = {
    "probe": "探测媒体",
    "trim": "截取片段",
    "audio": "分析音频",
    "scenes": "镜头切分",
    "plan": "分配帧预算",
    "frames": "抽取关键帧",
    "compose": "生成提示词",
    "done": "完成",
}


async def run_pipeline(job: Job, video_path: Path) -> JobResult:
    """执行完整反推流程。异常由调用方捕获并写入 job.error。"""
    s = get_settings()
    opts: AnalyzeOptions = job.options
    job_id = job.id
    t0 = time.time()

    async def step(stage: str, percent: int, message: str = "") -> None:
        await store.progress(job_id, stage, percent, message, STAGE_LABELS.get(stage, stage))

    store.raise_if_cancelled(job_id)

    # ---------------- 1. 探测 ----------------
    await step("probe", 3, "读取视频信息")
    media = await asyncio.to_thread(ff.probe, video_path)
    if not media.has_video:
        raise RuntimeError("该文件不含视频流，无法反推")

    # ---------------- 1.5 片段截取（用户框选了区间时） ----------------
    #
    # 先切出片段再跑后面的流程，这样探测、镜头检测、抽帧、音频、时间戳
    # **全部天然是相对片段的**。另一种做法是把偏移量传遍整条管线、
    # 在七八个地方各自记得减 —— 那种改法每漏一处就是一个隐蔽 bug。
    trim_start = trim_end = None
    if opts.trim_start is not None and opts.trim_end is not None:
        trim_start = max(0.0, float(opts.trim_start))
        trim_end = min(float(opts.trim_end), media.duration)
        if trim_end - trim_start < 0.5:
            raise RuntimeError(
                f"选择的片段只有 {max(0.0, trim_end - trim_start):.2f} 秒，太短了（至少 0.5 秒）"
            )
        await step("trim", 5, f"截取片段 {trim_start:.1f}s - {trim_end:.1f}s")
        seg_path, why = await asyncio.to_thread(
            ff.extract_segment,
            video_path,
            trim_start,
            trim_end,
            ff.frame_dir_for(job_id) / "segment.mp4",
        )
        if seg_path is None:
            raise RuntimeError(why or "截取片段失败")
        log.info("已截取片段 %.2fs - %.2fs（%.2fs）", trim_start, trim_end, trim_end - trim_start)
        video_path = seg_path
        media = await asyncio.to_thread(ff.probe, video_path)
        store.raise_if_cancelled(job_id)

    if s.max_duration_seconds > 0 and media.duration > s.max_duration_seconds:
        log.warning("视频时长 %.1fs 超过上限 %.1fs，将只分析前段", media.duration, s.max_duration_seconds)
    eff_duration = media.duration
    if s.max_duration_seconds > 0:
        eff_duration = min(eff_duration, s.max_duration_seconds)

    # ---------------- 2. 场景检测 + 音频（并行） ----------------
    await step("scenes", 8, "检测镜头切换点")

    scene_task = asyncio.create_task(asyncio.to_thread(
        _safe_detect_shots, video_path, eff_duration, s.scene_threshold, s.min_shot_seconds
    ))
    audio_task = asyncio.create_task(asyncio.to_thread(
        _safe_audio, video_path, opts.enable_asr
    ))

    raw_shots, cut_count, used_threshold, adaptive_note = await scene_task
    store.raise_if_cancelled(job_id)
    await step("audio", 20, "转写语音与音量分析")

    audio: AudioReport = await audio_task
    store.raise_if_cancelled(job_id)

    # ---------------- 3. 镜头切分 ----------------
    if not opts.enable_scene_split:
        raw_shots = []
    elif adaptive_note:
        log.info("镜头检测：%s", adaptive_note)

    if len(raw_shots) < 2:
        # 没有明显切换（单镜头 / 长镜头访谈 / 纯色测试图案）→ 退化为均匀逻辑镜头，
        # 否则整段只能抽到 1~3 帧，信息量不足。
        target = max(2, min(12, int(eff_duration / 4) or 2))
        raw_shots = ff.uniform_shots(eff_duration, target)
        log.info("未检测到足够的场景切换，退化为 %d 个均匀逻辑镜头", len(raw_shots))

    shots = [Shot(index=i, start=a, end=b) for i, (a, b) in enumerate(raw_shots)]

    store.raise_if_cancelled(job_id)

    # ---------------- 4. 帧预算分配 ----------------
    budget = opts.max_total_frames or s.max_total_frames
    # 按次覆盖：前端高级选项里可以单独指定，留空用服务端默认
    interval = opts.frame_interval_seconds or s.frame_interval_seconds
    word_limit = opts.prompt_word_limit or s.prompt_word_limit
    await step("plan", 26, f"{len(shots)} 个镜头，预算 {budget} 帧")

    # ---------------- 5. 逻辑分块（镜头边界处切） ----------------
    chunks = _chunk_shots(shots, s.chunk_threshold_seconds, s.chunk_seconds)
    total_chunks = len(chunks)
    log.info("镜头 %d 个 → 分块 %d 个", len(shots), total_chunks)

    # 按块时长比例分配帧预算
    per_chunk_budget = _split_budget(
        chunks, budget, s.max_frames_per_shot, s.long_shot_seconds, interval
    )

    await step("frames", 32, f"抽帧（{total_chunks} 块）")
    frame_root = ff.frame_dir_for(job_id)

    chunk_frames: list[list[tuple[float, str, Path]]] = []
    # **实际**用到的帧规划，累计起来用于最后的说明。
    # 不要在末尾拿总预算重算一遍 —— 那样算出来的是「如果重新分配会怎样」，
    # 而不是「实际抽了多少」。实测过：界面显示 27 张，实际只有 21 张。
    actual_plan: list[selection.PlannedFrame] = []
    actual_shots: list[tuple[float, float]] = []
    for ci, chunk in enumerate(chunks):
        actual_shots.extend((sh.start, sh.end) for sh in chunk)
        store.raise_if_cancelled(job_id)
        local_shots = [(sh.start, sh.end) for sh in chunk]
        plan = selection.plan_frames(
            local_shots,
            per_chunk_budget[ci],
            max_per_shot=s.max_frames_per_shot,
            long_shot_seconds=s.long_shot_seconds,
            frame_interval=interval,
        )
        actual_plan.extend(plan)
        role_of = {round(p.time, 3): p.role for p in plan}
        out_dir = frame_root / f"c{ci:02d}"

        pairs = await asyncio.to_thread(
            ff.extract_frames_at,
            video_path,
            [p.time for p in plan],
            out_dir,
            workers=5,
        )
        items: list[tuple[float, str, Path]] = []
        for t, path in pairs:
            role = role_of.get(round(t, 3), "mid")
            items.append((t, role, path))

        chunk_frames.append(items)

        pct = 32 + int(18 * (ci + 1) / max(1, total_chunks))
        await step("frames", pct, f"已抽帧 {sum(len(x) for x in chunk_frames)} 张")

    total_frames = sum(len(x) for x in chunk_frames)
    if total_frames == 0:
        raise RuntimeError("抽帧失败，无法进行视觉分析（请检查 ffmpeg 是否可解码该文件）")
    log.info("共抽帧 %d 张", total_frames)

    flat: list[tuple[float, str, Path]] = [
        (ft, role, fp) for items in chunk_frames for ft, role, fp in items
    ]

    # ---------------- 5.5 拼图：把相邻帧并进一张网格图 ----------------
    #
    # 用户 09-23 提的方案（「一秒2帧 + 自动切镜 + 合成缩略图」）。
    # 为什么要把相邻帧放进同一张图：
    #   1) **模型能直接并排比较** —— 判断「这是切镜还是同一个镜头里的运动」，
    #      在一张图里比在几张图之间容易得多。这是分镜判断的关键。
    #   2) token 几乎不涨：实测网格从 2.7 Mpx 做到 9.2 Mpx（3.4 倍），
    #      token 只从 1156 涨到 1176。一格里放 6 帧还是 20 帧成本一样。
    #   3) 同预算下时间覆盖度翻几倍。
    #
    # ⚠️ 帧清单仍按**每一帧**给（不是按图给），否则帧间隔里的 `sampling boundary`
    # 切点标注就丢了。布局说明负责把「第几张图 = 哪几个时间点」讲清楚。
    images = [fp for _, _, fp in flat]
    sheet_note = ""
    if s.frame_sheet_cells >= 2 and len(flat) >= 2:
        sheets = await asyncio.to_thread(
            ff.build_contact_sheets,
            [(ft, fp) for ft, _, fp in flat],
            s.frame_sheet_cells,
            ff.frame_dir_for(job_id) / "sheets",
        )
        if sheets and any(len(times) > 1 for times, _ in sheets):
            sheet_note = ff.describe_sheet_layout(sheets, s.frame_sheet_cells)
            # ⚠️ 拼图会打乱「第几张图 = 哪个时间点」的直觉，实测模型会**自己编时间戳**
            # （写成 00:40.000 / 01:20.000 这种整十秒，而视频只有 15 秒）。
            # 所以把「只能用列出的时间」写死。
            all_times = sorted(t for times, _ in sheets for t in times)
            sheet_note += (
                f"\n\nThe segment is {eff_duration:.2f}s long. **Every `[Shot N] At ...` "
                "timestamp you write must be one of the times listed above** — those are the "
                "only real timestamps you have — copy them in the same MM:SS.mmm form. Do not "
                "invent timestamps, and do not round them to tidy numbers."
            )
            log.info("拼图覆盖时间：%.2fs - %.2fs", all_times[0], all_times[-1])
            images = [path for _, path in sheets]
            log.info(
                "拼图：%d 帧 → %d 张网格（每张最多 %d 格），送模型的图片数 %d → %d",
                total_frames, len(images), s.frame_sheet_cells, len(flat), len(images),
            )

    # ---------------- 6. 生成提示词 ----------------
    #
    # 帧 + 时间戳 + 用户说明 → 一次调用出稿。
    #
    # ⚠️ 这里**只给身份、任务和目标格式**，刻意不写「怎么观察」的规则。
    # 那些规则曾经占掉 14000 字符的系统提示词，而实测每收紧一次、输出就退化一次：
    # 防编造规则压掉合理推断（动作描述只剩 19%）、把「相机静止」当结论喂进去会被
    # 外推成「人物也静止」、写死「20 秒 MV 大约 5-20 个镜头」会把快切压到 20 以内。
    # **约束输出格式 ≠ 约束思考。**

    client = VLMClient()
    client.require_configured()

    await step("compose", 55, f"一次成稿（{total_frames} 帧，{len(images)} 张图）")
    freeform_user = templates.build_user(
        chunk_start=0.0,
        chunk_end=eff_duration,
        frame_marks=[(ft, role) for ft, role, _ in flat],
        audio_text=_audio_slice(audio, 0.0, eff_duration),
        media=media,
        chunk_index=0,
        chunk_total=1,
        content_hint=opts.content_hint,
        closing=templates.build_closing(),
        sheet_note=sheet_note,
    )
    if opts.extra_instruction.strip():
        freeform_user += (
            "\n\n--- EXTRA INSTRUCTION FROM THE USER ---\n"
            + opts.extra_instruction.strip()
        )
    if opts.target_duration:
        freeform_user += (
            f"\n\nThe generated video should be {opts.target_duration:.2f}s long."
        )
    try:
        prompt = await client.complete(
            templates.build_system(opts.format, opts.language),
            freeform_user,
            images=images,
            max_tokens=s.vlm_max_tokens,
        )
    except VLMError as exc:
        raise RuntimeError(f"提示词生成失败：{exc}") from exc
    total_chunks = 1

    # 长度检查与压缩。视频生成模型的提示词窗口有限，超长会被截断或忽略 ——
    # 实测不限长时六段式能写到 1795 词 / 11314 字符。
    #
    # 逐段给预算只能压到 918 词（模型对字数指令的服从度很差：给了「每条 15 词」
    # 仍然写出 26 词）。所以再加一道「压缩」——把超长文本交给模型改短，
    # 这是编辑任务，比「按预算生成」可靠得多。压缩后要校验结构没丢：
    # 残缺的提示词比超长的更糟。
    prompt = prompt.strip()
    word_count = len(prompt.split())
    limit = word_limit
    compressed = False
    if limit > 0 and word_count > limit:
        await step("compose", 92, f"提示词 {word_count} 词，压缩到 {limit} 词以内")
        try:
            shorter = (
                await client.complete(
                    templates.build_compress_system(limit),
                    templates.build_compress_user(prompt, limit),
                    images=[],
                    max_tokens=s.vlm_max_tokens,
                    # ⚠️ 压缩要**开思考**，和其他环节相反 ——
                    # 实测关思考时模型只肯砍 74 词（甚至原样返回），
                    # 开思考能砍 252 词直接达标。编辑需要先想清楚哪些能砍。
                    disable_thinking=False,
                )
            ).strip()
            ok, why = templates.check_prompt_integrity(prompt, shorter)
            new_words = len(shorter.split())
            if ok and new_words < word_count:
                log.info("提示词压缩：%d 词 → %d 词", word_count, new_words)
                prompt, word_count, compressed = shorter, new_words, True
            else:
                log.warning(
                    "提示词压缩未采用（%s；%d 词 → %d 词），保留原稿",
                    why or "没有变短", word_count, new_words,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("提示词压缩失败，保留原稿：%s", exc)

    if limit > 0 and word_count > limit:
        log.warning(
            "提示词仍有 %d 词，超过 %d 词上限 —— 视频模型可能截断或忽略。"
            "可调小 MAX_FRAMES_PER_SHOT / FRAME_INTERVAL_SECONDS 减少细节量，"
            "或收窄 PROMPT_WORD_LIMIT 逼模型更精简。",
            word_count, limit,
        )

    # 音频分析术语的强制清理。放在压缩之后 —— 压缩也是一次模型编辑，可能重新引入。
    #
    # ⚠️ 为什么用代码而不是提示词：试过在提示词里列禁用词，结果模型把这份清单
    # 本身抄进了输出（「不要写 content unanalysed」反而让它记住了这个词）。
    # 这类泄漏只能在输出侧堵。
    prompt, jargon_hits = audio_features.sanitize_audio_jargon(prompt)
    if jargon_hits:
        log.warning("提示词里出现了音频分析术语，已清理：%s", ", ".join(jargon_hits))
        word_count = len(prompt.split())

    # ---------------- 8. 汇总 ----------------
    frame_urls = [
        f"/api/media/frame/{job_id}/{p.parent.name}/{p.name}"
        for items in chunk_frames for _, _, p in items
    ]

    elapsed = time.time() - t0
    result = JobResult(
        prompt=prompt,
        media=media,
        audio=audio,
        shots=shots,
        frames_used=total_frames,
        frame_urls=frame_urls,
        # 有截取片段时，预览要用片段而不是原视频 ——
        # 抽帧时间戳是相对片段的，放原视频会让点帧跳转整条错位。
        analyzed_video_url=(
            f"/api/media/segment/{job_id}"
            if trim_start is not None and trim_end is not None
            else ""
        ),
        chunks=total_chunks,
        stats={
            "elapsed_sec": round(elapsed, 1),
            "shots": len(shots),
            "frames": total_frames,
            "chunks": total_chunks,
            "scene_cuts": cut_count,
            "scene_threshold": used_threshold,
            "scene_adaptive": adaptive_note or "",
            "est_tokens": selection.estimate_tokens(total_frames, long_edge=s.frame_long_edge),
            "images_sent": len(images),
            "sheet_cells": s.frame_sheet_cells if sheet_note else 0,
            "prompt_words": word_count,
            "prompt_chars": len(prompt),
            "prompt_word_limit": word_limit,
            "prompt_compressed": compressed,
            "prompt_over_limit": bool(
                word_limit > 0 and word_count > word_limit
            ),
            "format": opts.format,
            "trim": (
                {"start": round(trim_start, 3), "end": round(trim_end, 3),
                 "duration": round(trim_end - trim_start, 3)}
                if trim_start is not None and trim_end is not None else None
            ),
            "format_label": templates.format_display(opts.format),
            "mode": templates.mode_of(opts.format),
            "asr": audio.note or ("已转写" if audio.transcript else "无转写"),
            "plan": selection.describe_plan(
                actual_plan,
                actual_shots,
                s.frame_long_edge,
                actual_frames=total_frames,
            ),
            "audio_jargon_removed": jargon_hits,
            "vocal_isolation": audio.vocal_isolation,
            "music_bpm": audio.bpm,
        },
    )
    return result


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_TC_RE = re.compile(r"^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")
def _safe_detect_shots(
    path: Path, duration: float, threshold: float, min_shot_seconds: float
) -> tuple[list[tuple[float, float]], int, float, str]:
    """自适应镜头检测，失败时返回空（由调用方退化为均匀分镜）。

    ⚠️ 捕获 `BaseException` 而不是 `Exception` —— 某些运行环境在删除临时文件时
    会抛 `SystemExit`（批量删除保护），而 `SystemExit` 继承 `BaseException`，
    `except Exception` 拦不住。任务级的问题绝不该升级成进程级事故。
    """
    try:
        return ff.detect_shots_adaptive(
            path, duration, threshold=threshold, min_shot_seconds=min_shot_seconds
        )
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001
        log.warning("场景检测失败，将退化为均匀分镜: %s", exc)
        return [], 0, threshold, ""


def _safe_audio(path: Path, enable_asr: bool) -> AudioReport:
    """音频分析，失败时返回带原因的降级报告。

    ⚠️ 必须捕获 `BaseException`。踩过：本地 whisper 首次加载会走 HuggingFace
    的缓存流程，清理临时目录时触发了批量删除保护并抛 `SystemExit` ——
    而这里原来只捕获 `Exception`，`SystemExit` 直接穿过去把整个任务带走了，
    表现是「跑音频分析那一步进程就没了，日志里只有一句删除保护」。
    """
    try:
        return asr_mod.analyze_audio(path, enable_asr=enable_asr)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001
        log.warning("音频分析失败: %s", exc)
        report = AudioReport()
        report.note = f"音频分析失败：{exc}"
        return report


def _chunk_shots(
    shots: list[Shot],
    threshold_seconds: float,
    chunk_seconds: float,
) -> list[list[Shot]]:
    """按镜头边界把镜头分组为若干块。

    只在镜头之间切，绝不切开单个镜头——否则两边都推不准。
    单个镜头本身超过 chunk_seconds 时，它独占一块。
    """
    if not shots:
        return []
    total = shots[-1].end - shots[0].start
    if total <= threshold_seconds:
        return [shots]

    chunks: list[list[Shot]] = []
    current: list[Shot] = []
    chunk_start = shots[0].start

    for sh in shots:
        if current and (sh.end - chunk_start) > chunk_seconds:
            chunks.append(current)
            current = [sh]
            chunk_start = sh.start
        else:
            current.append(sh)

    if current:
        chunks.append(current)
    return chunks


def _split_budget(
    chunks: list[list[Shot]],
    budget: int,
    max_per_shot: int,
    long_shot_seconds: float,
    frame_interval: float = 1.0,
) -> list[int]:
    """按各块时长占比分配帧预算，保证每块至少能覆盖自己所有镜头。

    这里只负责「总预算怎么分给各块」，块内每个镜头具体几帧由
    `plan_frames` 按镜头时长决定。所以容量上限要按每块镜头的**目标帧数**算，
    不能用固定值 —— 原来硬编码 3，导致 max_frames_per_shot 调到 8 也不生效。
    """
    if not chunks:
        return []

    # 每块的硬下限：保证每个镜头至少 1 帧
    floors = [len(c) for c in chunks]

    if sum(floors) >= budget:
        # 预算连保底都不够，按镜头数比例压缩（plan_frames 内部会再均匀挑镜头）
        scale = budget / sum(floors)
        return [max(1, int(f * scale)) for f in floors]

    remaining = budget - sum(floors)
    alloc = list(floors)

    # 容量上限：每块内所有镜头的目标帧数之和
    def _target(shot: Shot) -> int:
        d = max(0.05, shot.end - shot.start)
        return max(1, min(max_per_shot, int(round(d / max(0.1, frame_interval))) or 1))

    caps = [max(0, sum(_target(sh) for sh in c) - floors[i]) for i, c in enumerate(chunks)]
    while remaining > 0 and sum(caps) > 0:
        total_cap = sum(caps)
        added = 0
        for i in range(len(chunks)):
            if remaining <= 0 or caps[i] <= 0:
                continue
            share = caps[i] / total_cap * remaining
            give = min(int(share), caps[i], remaining)
            if give <= 0 and remaining > 0 and caps[i] > 0:
                give = 1
            alloc[i] += give
            caps[i] -= give
            remaining -= give
            added += give
        if added == 0:
            break

    return alloc


def _audio_slice(audio: AudioReport, start: float, end: float) -> str:
    """截取落在该时间窗内的转写片段 + 整片音乐画像，附给对应分块。

    ⚠️ 这里同样**不给 dB 数字和频段名**。Pass1 的 `global_notes` 会原样喂进
    Pass2，写进去的措辞会被模型学去（实测最终输出里出现过
    `measured energy mostly voice band`）。只给人类可读的描述。
    """
    if not audio.has_audio:
        return "【音频】该视频没有音轨。所有音频字段请写 N/A。"

    lines = [f"【本段语音转写 {start:.2f}s - {end:.2f}s】"]
    hit = [s for s in audio.segments if s.end >= start - 0.5 and s.start <= end + 0.5]
    if hit:
        for seg in hit[:80]:
            lines.append(f"  [{seg.start:6.2f}s] {seg.text}")
    elif audio.transcript:
        lines.append("  （无逐句时间戳，以下为整片转写，请按语义对齐到镜头）")
        lines.append("  " + audio.transcript[:3000])
    else:
        # 这里必须明说「你听不到」，否则模型会照着画面编出
        # 「电子提示音与数字跳变同步」这类听起来合理的声音描述。
        lines.append("  （未做语音识别，或识别结果为空。）")
        lines.append("  ⚠ 音轨存在但音频内容未知：禁止写台词、歌词、BGM 乐器、")
        lines.append("    具体音效类型。音频字段只能留空或写 N/A。")

    # 音乐画像：实测得出的人类可读描述，可以直接用
    if audio.music_profile:
        lines.append("【整片音乐与声音（实测描述，可以直接引用或改写措辞）】")
        lines.append("  " + audio.music_profile)
        if audio.bpm and audio.has_beat:
            lines.append(f"  （节奏约 {audio.bpm:.0f} BPM）")

    return "\n".join(lines)
def build_frame_refs(job_id: str, frames: list[tuple[float, str, Path]], shot_index: int) -> list[FrameRef]:
    return [
        FrameRef(shot_index=shot_index, time=t, path=str(p), role=role)  # type: ignore[arg-type]
        for t, role, p in frames
    ]


def run_pipeline_sync(job: Job, video_path: Path) -> JobResult:
    """同步执行整条管线。

    供 CLI、脚本和测试使用（这些场景不方便自己管事件循环）。
    Web 路径走 `runner._spawn` 里的 `await run_pipeline(...)`，不要用这个。
    """
    return asyncio.run(run_pipeline(job, video_path))
