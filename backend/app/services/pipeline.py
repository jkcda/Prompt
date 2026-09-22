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
import time
from pathlib import Path

from ..core.config import get_settings
from ..schemas import (
    AnalyzeOptions,
    AudioReport,
    ChunkObservation,
    FrameRef,
    Job,
    JobResult,
    Shot,
)
from . import asr as asr_mod
from . import ffmpeg as ff
from . import selection, templates
from .jobs import store
from .vlm import VLMClient, VLMError

log = logging.getLogger("pipeline")

STAGE_LABELS = {
    "probe": "探测媒体",
    "audio": "分析音频",
    "scenes": "镜头切分",
    "plan": "分配帧预算",
    "frames": "抽取关键帧",
    "observe": "逐块视觉分析",
    "compose": "合成提示词",
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

    if s.max_duration_seconds > 0 and media.duration > s.max_duration_seconds:
        log.warning("视频时长 %.1fs 超过上限 %.1fs，将只分析前段", media.duration, s.max_duration_seconds)
    eff_duration = media.duration
    if s.max_duration_seconds > 0:
        eff_duration = min(eff_duration, s.max_duration_seconds)

    # ---------------- 2. 场景检测 + 音频（并行） ----------------
    await step("scenes", 8, "检测镜头切换点")

    scene_task = asyncio.create_task(asyncio.to_thread(
        _safe_scene_detect, video_path, s.scene_threshold, eff_duration
    ))
    audio_task = asyncio.create_task(asyncio.to_thread(
        _safe_audio, video_path, opts.enable_asr
    ))

    cuts = await scene_task
    store.raise_if_cancelled(job_id)
    await step("audio", 20, "转写语音与音量分析")

    audio: AudioReport = await audio_task
    store.raise_if_cancelled(job_id)

    # ---------------- 3. 镜头切分 ----------------
    if opts.enable_scene_split and cuts:
        raw_shots = ff.cuts_to_shots(cuts, eff_duration, s.min_shot_seconds)
    else:
        raw_shots = []

    if len(raw_shots) < 2:
        # 没有明显切换（单镜头 / 长镜头访谈）→ 退化为均匀逻辑镜头，
        # 否则整段只能抽到 1~3 帧，信息量不足。
        target = max(2, min(12, int(eff_duration / 4) or 2))
        raw_shots = ff.uniform_shots(eff_duration, target)
        log.info("未检测到场景切换，退化为 %d 个均匀逻辑镜头", len(raw_shots))

    shots = [Shot(index=i, start=a, end=b) for i, (a, b) in enumerate(raw_shots)]

    # ---------------- 4. 帧预算分配 ----------------
    budget = opts.max_total_frames or s.max_total_frames
    await step("plan", 26, f"{len(shots)} 个镜头，预算 {budget} 帧")

    # ---------------- 5. 逻辑分块（镜头边界处切） ----------------
    chunks = _chunk_shots(shots, s.chunk_threshold_seconds, s.chunk_seconds)
    total_chunks = len(chunks)
    log.info("镜头 %d 个 → 分块 %d 个", len(shots), total_chunks)

    # 按块时长比例分配帧预算
    per_chunk_budget = _split_budget(chunks, budget, s.max_frames_per_shot, s.long_shot_seconds)

    await step("frames", 32, f"抽帧（{total_chunks} 块）")
    frame_root = ff.frame_dir_for(job_id)

    chunk_frames: list[list[tuple[float, str, Path]]] = []
    for ci, chunk in enumerate(chunks):
        store.raise_if_cancelled(job_id)
        local_shots = [(sh.start, sh.end) for sh in chunk]
        plan = selection.plan_frames(
            local_shots,
            per_chunk_budget[ci],
            max_per_shot=s.max_frames_per_shot,
            long_shot_seconds=s.long_shot_seconds,
        )
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

    # ---------------- 6. Pass1：逐块视觉观察 ----------------
    await step("observe", 55, f"视觉分析 {total_chunks} 块（{total_frames} 帧）")

    client = VLMClient()
    client.require_configured()

    # 是否给模型附上音频片段。只在「开了开关」且「ASR 没给出转写」时才附——
    # 已经有逐句转写文本时再送音频纯属浪费 token，而且模型未必比转写更准。
    attach_audio = bool(s.vlm_audio_input) and audio.has_audio and not audio.segments
    if attach_audio:
        log.info("VLM_AUDIO_INPUT 已开启且无 ASR 转写，将为每块附上音频片段")

    tasks: list[tuple[str, str, list[Path]]] = []
    audio_per_task: list[list[Path]] = []
    for ci, items in enumerate(chunk_frames):
        cstart = chunks[ci][0].start
        cend = chunks[ci][-1].end
        marks = [(t, role) for t, role, _ in items]
        imgs = [p for _, _, p in items]
        prev_notes = ""
        if ci > 0:
            prev_notes = "\n\nContext from the previous segment (already analysed, do not repeat it):\n"

        audio_text = _audio_slice(audio, cstart, cend)
        audios: list[Path] = []
        if attach_audio:
            seg = await asyncio.to_thread(
                ff.extract_audio_segment, video_path, cstart, cend
            )
            if seg:
                audios = [seg]
                audio_text += (
                    f"\n【本段音频已随请求附上（{cend - cstart:.1f}s）】"
                    "你可以直接听。听出来的内容以你的听觉为准，"
                    "但只描述确实听到的，不确定就说不确定。"
                )

        user = templates.build_pass1_user(
            chunk_start=cstart,
            chunk_end=cend,
            frame_marks=marks,
            audio_text=audio_text,
            media=media,
            chunk_index=ci,
            chunk_total=total_chunks,
        )
        tasks.append((templates.PASS1_SYSTEM, prev_notes + user, imgs))
        audio_per_task.append(audios)

    raw_outputs = await client.complete_many(
        tasks, max_tokens=s.vlm_max_tokens, audio_per_task=audio_per_task
    )
    store.raise_if_cancelled(job_id)

    observations: list[ChunkObservation] = []
    for ci, raw in enumerate(raw_outputs):
        shots_obs, subjects_obs, notes = templates.parse_pass1_json(raw)
        chunk = chunks[ci]
        observations.append(ChunkObservation(
            chunk_index=ci,
            start=chunk[0].start,
            end=chunk[-1].end,
            shots=shots_obs,
            subjects=subjects_obs,
            global_notes=notes,
            raw=raw,
        ))
        await store.emit(job_id, {
            "type": "chunk",
            "index": ci,
            "shots": len(shots_obs),
            "subjects": len(subjects_obs),
            "total": total_chunks,
        })

    merged = templates.merge_observations(observations)
    if not merged:
        raise RuntimeError(_explain_empty_observations(raw_outputs, client.last_errors))

    subjects = templates.merge_subjects(observations, merged)
    log.info("观察完成：%d 个镜头，%d 个主体", len(merged), len(subjects))
    if not subjects:
        # 模型漏了 subjects 数组。不致命（Pass2 能从观察结果自己推参考标签，
        # 实测推得还行），但要显式记下来 —— 否则前端「主体」页空着，
        # 而提示词里却有 <Subject N>，看起来像 bug 却查不到原因。
        log.warning(
            "Pass1 未返回主体登记表（subjects 为空）。Ref2VA 的参考标签将由 Pass2 "
            "从镜头观察里自行推导，稳定性和镜号准确度会下降。"
        )

    # ---------------- 7. Pass2：合成目标格式 ----------------
    await step("compose", 82, f"合成 {templates.format_display(opts.format)} 提示词")

    shots_summary = _describe_shots(shots)
    pass2_system = templates.build_pass2_system(opts.format, opts.language)
    pass2_user = templates.build_pass2_user(
        observations=observations,
        audio=audio,
        media=media,
        shots_summary=shots_summary,
        extra_instruction=opts.extra_instruction,
        target_duration=opts.target_duration,
        subjects=subjects,
        fmt=opts.format,
    )

    try:
        prompt = await client.complete(
            pass2_system, pass2_user, images=[], max_tokens=s.vlm_max_tokens
        )
    except VLMError as exc:
        raise RuntimeError(f"提示词合成失败：{exc}") from exc

    # ---------------- 8. 汇总 ----------------
    frame_urls = [
        f"/api/media/frame/{job_id}/{p.parent.name}/{p.name}"
        for items in chunk_frames for _, _, p in items
    ]

    elapsed = time.time() - t0
    result = JobResult(
        prompt=prompt.strip(),
        observations=merged,
        subjects=subjects,
        media=media,
        audio=audio,
        shots=shots,
        frames_used=total_frames,
        frame_urls=frame_urls,
        chunks=total_chunks,
        stats={
            "elapsed_sec": round(elapsed, 1),
            "shots": len(shots),
            "frames": total_frames,
            "chunks": total_chunks,
            "scene_cuts": len(cuts),
            "subjects": len(subjects),
            "est_tokens": selection.estimate_tokens(total_frames),
            "format": opts.format,
            "format_label": templates.format_display(opts.format),
            "mode": templates.mode_of(opts.format),
            "asr": audio.note or ("已转写" if audio.transcript else "无转写"),
            "plan": selection.describe_plan(
                selection.plan_frames(
                    [(sh.start, sh.end) for sh in shots],
                    budget,
                    s.max_frames_per_shot,
                    s.long_shot_seconds,
                ),
                [(sh.start, sh.end) for sh in shots],
            ),
        },
    )
    return result


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _explain_empty_observations(raw_outputs: list[str], errors: list[str]) -> str:
    """Pass1 全军覆没时，拼一条能直接定位问题的报错。

    原来的文案是「请检查模型是否支持图片输入，或换用更强的视觉模型」——
    但真实原因常常不是这个。踩过：DeepSeek-V4.1-Flash 是推理模型，
    思考过程 5 万多字符把 token 预算吃光，正文一个字没写出来，
    结果报成「不支持图片」，方向完全带偏。
    """
    lines = ["视觉模型未能返回可解析的镜头观察结果。"]

    real = [e for e in errors if e]
    if real:
        lines.append("")
        lines.append("各分块的失败原因：")
        for i, e in enumerate(real[:3]):
            lines.append(f"  分块 {i + 1}: {' '.join(e.split())[:300]}")
    else:
        lines.append("")
        lines.append("模型有返回内容，但不是要求的 JSON 结构。各分块原始返回开头：")
        for i, raw in enumerate(raw_outputs[:3]):
            head = " ".join((raw or "").split())[:200] or "（空响应）"
            lines.append(f"  分块 {i + 1}: {head}")

    lines.append("")
    lines.append(
        "排查顺序：① 看上面有没有 finish_reason=length —— 那是 token 预算不够，"
        "调大 VLM_MAX_TOKENS；② 确认模型支持图片输入（GET /api/health/vlm 会发一张图实测）；"
        "③ 确认模型能按 JSON 输出，推理模型有时会把答案写成散文。"
    )
    return "\n".join(lines)


def _safe_scene_detect(path: Path, threshold: float, duration: float) -> list[float]:
    try:
        return ff.detect_scene_cuts(path, threshold=threshold, max_duration=duration)
    except Exception as exc:  # noqa: BLE001
        log.warning("场景检测失败，将退化为均匀分镜: %s", exc)
        return []


def _safe_audio(path: Path, enable_asr: bool) -> AudioReport:
    try:
        return asr_mod.analyze_audio(path, enable_asr=enable_asr)
    except Exception as exc:  # noqa: BLE001
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
) -> list[int]:
    """按各块时长占比分配帧预算，保证每块至少能覆盖自己所有镜头。"""
    if not chunks:
        return []

    durations = [max(0.1, c[-1].end - c[0].start) for c in chunks]
    sum(durations)

    # 每块的硬下限：保证每个镜头至少 1 帧
    floors = [len(c) for c in chunks]

    if sum(floors) >= budget:
        # 预算连保底都不够，按镜头数比例压缩（plan_frames 内部会再均匀挑镜头）
        scale = budget / sum(floors)
        return [max(1, int(f * scale)) for f in floors]

    remaining = budget - sum(floors)
    alloc = list(floors)

    # 按容量上限（每镜头最多 3 帧）加权分配剩余
    caps = [max(0, min(max_per_shot, 3) * len(c) - floors[i]) for i, c in enumerate(chunks)]
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
    """截取落在该时间窗内的转写片段，附给对应分块。"""
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

    meta: list[str] = []
    if audio.mean_volume_db is not None:
        meta.append(f"平均音量 {audio.mean_volume_db:.1f}dB")
    if audio.silence_ratio is not None:
        meta.append(f"静音占比 {audio.silence_ratio * 100:.0f}%")
    if meta:
        lines.append("【音频能量】" + "；".join(meta))
    return "\n".join(lines)


def _describe_shots(shots: list[Shot]) -> str:
    if not shots:
        return "无"
    total = shots[-1].end - shots[0].start
    avg = total / len(shots)
    return (
        f"{len(shots)} 个镜头，总时长 {total:.2f}s，平均镜头长度 {avg:.2f}s"
        f"（{'快剪' if avg < 2 else '常规' if avg < 5 else '长镜头为主'}）"
    )


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
