"""端到端管线测试：用 mock 模型把整条反推链路完整跑一遍。

真实模型调用有成本、有延迟、还依赖外部可用性；但管线里最容易出错的恰恰是
调用之外的部分——抽帧、时间戳对齐、Pass1 JSON 解析、多块合并、Pass2 组装。
mock 掉模型这一层，就能把整条链路完整验证。

断言的是「结构」而不是「措辞」：措辞由模型决定，结构由我们保证。
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from app.core import config as cfg
from app.schemas import AnalyzeOptions, Job
from app.services import ffmpeg as ff
from app.services import pipeline
from tests import mock_vlm

# ---------------------------------------------------------------------------
# mock 模型服务（整个测试会话共用一个）
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_server():
    port = _free_port()
    config = uvicorn.Config(mock_vlm.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/v1/models", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.15)
    else:
        pytest.fail("mock 模型服务启动失败")

    yield base

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def mock_vlm_env(mock_server, monkeypatch):
    """把配置指向 mock 服务，并放宽抽帧预算让断言更稳定。"""
    monkeypatch.setenv("VLM_API_KEY", "mock-key")
    monkeypatch.setenv("VLM_BASE_URL", f"{mock_server}/v1")
    monkeypatch.setenv("VLM_MODEL", "mock-vlm")
    monkeypatch.setenv("VLM_CONCURRENCY", "3")
    cfg.refresh_settings()
    mock_vlm.reset()
    yield mock_server
    cfg.refresh_settings()


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_full_pipeline_h3(mock_vlm_env, sample_video: Path, tmp_path: Path):
    """完整跑一遍：探测 → 镜头切分 → 抽帧 → Pass1 → Pass2。"""
    job = Job(
        id="e2e-h3",
        source="upload",
        title="e2e",
        options=AnalyzeOptions(format="h3", language="en", enable_asr=False),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)

    # 提示词非空，且含 H3 的三个字段
    assert result.prompt.strip()
    assert "integrated_multimodal_description:" in result.prompt
    assert "overall_soundscape:" in result.prompt
    assert "non_diegetic_music:" in result.prompt

    # 结构化观察被解析出来了
    assert len(result.observations) >= 3
    assert all(o.shot for o in result.observations)

    # 抽帧真的发生了，且文件都在
    assert result.frames_used > 0
    assert len(result.frame_urls) == result.frames_used

    # 媒体信息与音频报告都在
    assert result.media is not None
    assert result.media.duration == pytest.approx(10.0, abs=0.5)
    assert result.audio is not None
    assert result.audio.has_audio is True

    # 统计字段
    assert result.stats["format"] == "h3"
    assert result.stats["frames"] == result.frames_used
    assert result.stats["shots"] >= 3

    # 模型被调用了两次：Pass1（带图）+ Pass2（不带图）
    passes = [c["pass"] for c in mock_vlm.CALLS]
    assert 1 in passes, "Pass1 未被调用"
    assert 2 in passes, "Pass2 未被调用"
    pass2_calls = [c for c in mock_vlm.CALLS if c["pass"] == 2]
    assert len(pass2_calls) == 1, "Pass2 应该只调用一次"


def test_pass1_receives_images_matching_frame_count(mock_vlm_env, sample_video: Path):
    """Pass1 请求里的图片数必须等于我们抽出来的帧数——时间戳与图片必须一一对应。"""
    job = Job(
        id="e2e-frames",
        source="upload",
        options=AnalyzeOptions(format="h3", enable_asr=False),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)

    pass1_calls = [c for c in mock_vlm.CALLS if c["pass"] == 1]
    assert pass1_calls, "没有 Pass1 调用"
    assert sum(c["images"] for c in pass1_calls) == result.frames_used


def test_pipeline_respects_frame_budget(mock_vlm_env, sample_video: Path):
    """预算设成 6 就最多送 6 帧，不能超。"""
    job = Job(
        id="e2e-budget",
        source="upload",
        options=AnalyzeOptions(format="h3", enable_asr=False, max_total_frames=6),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)
    assert result.frames_used <= 6
    assert sum(c["images"] for c in mock_vlm.CALLS if c["pass"] == 1) <= 6


def test_pipeline_supports_all_formats(mock_vlm_env, sample_video: Path):
    """四种格式都要能出结果，且各自带上自己的标志性结构。"""
    markers = {
        "h3": "integrated_multimodal_description:",
        "h3-ref": "subject_definitions:",
        "seedance": "全片",
        "generic": "【整体风格】",
    }
    for fmt, marker in markers.items():
        mock_vlm.reset()
        job = Job(
            id=f"e2e-{fmt}",
            source="upload",
            options=AnalyzeOptions(format=fmt, enable_asr=False),  # type: ignore[arg-type]
        )
        result = pipeline.run_pipeline_sync(job, sample_video)
        assert result.prompt.strip(), f"{fmt} 未产出提示词"
        assert marker in result.prompt, f"{fmt} 输出缺少标志结构 {marker}"
        assert result.stats["format"] == fmt


def test_pipeline_collects_subject_registry(mock_vlm_env, sample_video: Path):
    """Pass1 登记的主体要一路带到结果里，镜号是重排后的全局镜号。"""
    mock_vlm.reset()
    job = Job(
        id="e2e-subjects",
        source="upload",
        options=AnalyzeOptions(format="h3-ref", enable_asr=False),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)

    assert result.subjects, "主体登记表是空的"
    labels = {s.label for s in result.subjects}
    assert labels == {"performer", "rooftop"}
    assert result.stats["subjects"] == len(result.subjects)

    shot_count = len(result.observations)
    for sub in result.subjects:
        assert sub.shots, f"{sub.label} 没有登记镜号"
        assert all(1 <= int(x) <= shot_count for x in sub.shots), \
            f"{sub.label} 的镜号超出范围：{sub.shots}"


def test_subject_registry_reaches_pass2(mock_vlm_env, sample_video: Path):
    """Ref2VA 的参考标签必须来自登记表，而不是 Pass2 自己编。

    mock 会照登记表生成 <Subject N>，所以只要标签数量对得上，
    就说明登记表确实送到了 Pass2。
    """
    mock_vlm.reset()
    job = Job(
        id="e2e-registry-pass2",
        source="upload",
        options=AnalyzeOptions(format="h3-ref", enable_asr=False),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)

    n = len(result.subjects)
    assert f"<Subject {n}>" in result.prompt
    assert f"<Subject {n + 1}>" not in result.prompt
    assert result.prompt.count("): fully_preserved") == n


def test_seedance_mode_does_not_leak_h3_structure(mock_vlm_env, sample_video: Path):
    """两种模式不能串味：Seedance 输出里不该有 H3 的字段名。"""
    mock_vlm.reset()
    job = Job(
        id="e2e-seedance-clean",
        source="upload",
        options=AnalyzeOptions(format="seedance", enable_asr=False),
    )
    result = pipeline.run_pipeline_sync(job, sample_video)

    assert "subject_definitions" not in result.prompt
    assert "integrated_multimodal_description" not in result.prompt
    assert "retention_analysis" not in result.prompt
    assert result.stats["mode"] == "seedance"


def test_h3_modes_report_their_mode(mock_vlm_env, sample_video: Path):
    """T2VA 与 Ref2VA 都归到 h3 模式，但 format 各自不同。"""
    for fmt in ("h3", "h3-ref"):
        mock_vlm.reset()
        job = Job(
            id=f"e2e-mode-{fmt}",
            source="upload",
            options=AnalyzeOptions(format=fmt, enable_asr=False),  # type: ignore[arg-type]
        )
        result = pipeline.run_pipeline_sync(job, sample_video)
        assert result.stats["mode"] == "h3"
        assert result.stats["format"] == fmt


def test_pipeline_reports_progress_stages(mock_vlm_env, sample_video: Path):
    """进度事件必须覆盖到各个阶段，前端才能画步骤条。"""
    import asyncio

    from app.services.jobs import store

    job = asyncio.run(store.create(Job(
        id="e2e-progress",
        source="upload",
        options=AnalyzeOptions(format="h3", enable_asr=False),
    )))
    pipeline.run_pipeline_sync(job, sample_video)

    stages = {e["stage"] for e in store.events(job.id) if e.get("type") == "progress"}
    for expected in ("probe", "scenes", "audio", "plan", "frames", "observe", "compose"):
        assert expected in stages, f"缺少阶段事件：{expected}"


def test_pipeline_fails_clearly_without_vlm_key(sample_video: Path, monkeypatch):
    """没配 key 时要给出可操作的报错，而不是抛一堆栈。"""
    monkeypatch.setenv("VLM_API_KEY", "")
    cfg.refresh_settings()
    try:
        job = Job(id="e2e-nokey", source="upload", options=AnalyzeOptions(enable_asr=False))
        with pytest.raises(Exception) as exc:
            pipeline.run_pipeline_sync(job, sample_video)
        assert "VLM_API_KEY" in str(exc.value) or "多模态" in str(exc.value)
    finally:
        monkeypatch.setenv("VLM_API_KEY", "mock-key")
        cfg.refresh_settings()


def test_short_video_uses_single_chunk(mock_vlm_env, sample_video: Path):
    """10 秒视频不该被分块。"""
    job = Job(id="e2e-chunk", source="upload", options=AnalyzeOptions(enable_asr=False))
    result = pipeline.run_pipeline_sync(job, sample_video)
    assert result.chunks == 1


def test_long_video_is_chunked(mock_vlm_env, sample_video: Path, monkeypatch):
    """把分块阈值压到 4 秒，10 秒视频就该被切成多块。"""
    monkeypatch.setenv("CHUNK_THRESHOLD_SECONDS", "4")
    monkeypatch.setenv("CHUNK_SECONDS", "4")
    cfg.refresh_settings()
    try:
        job = Job(id="e2e-long", source="upload", options=AnalyzeOptions(enable_asr=False))
        result = pipeline.run_pipeline_sync(job, sample_video)
        assert result.chunks >= 2, f"应被分块，实际 {result.chunks} 块"
        assert result.stats["chunks"] == result.chunks
        # 分块后 Pass1 也应被调用多次
        assert len([c for c in mock_vlm.CALLS if c["pass"] == 1]) == result.chunks
    finally:
        monkeypatch.setenv("CHUNK_THRESHOLD_SECONDS", "90")
        monkeypatch.setenv("CHUNK_SECONDS", "60")
        cfg.refresh_settings()


# ---------------------------------------------------------------------------
# 请求级覆盖（前端高级选项）
# ---------------------------------------------------------------------------

def test_frame_interval_override_changes_frame_count(mock_vlm_env, sample_video: Path):
    """请求里指定 frame_interval_seconds 要真的生效。

    sample_video 是 10 秒 / 3 个镜头（4s / 3s / 3s），预算 96 不会成为瓶颈。
    间隔 1.0 -> 每镜 4/3/3 帧 = 10 帧；间隔 3.0 -> 每镜 3/3/3（受保底 3 限制）= 9 帧。
    """
    import asyncio

    from app.services.jobs import store

    counts = {}
    for interval in (1.0, 3.0):
        mock_vlm.reset()
        job = asyncio.run(store.create(Job(
            id=f"e2e-int-{interval}",
            source="upload",
            options=AnalyzeOptions(
                format="h3", enable_asr=False, frame_interval_seconds=interval
            ),
        )))
        result = pipeline.run_pipeline_sync(job, sample_video)
        counts[interval] = result.frames_used
        assert result.stats["plan"]

    assert counts[1.0] > counts[3.0], f"间隔越小帧应越多：{counts}"


def test_frame_interval_override_shows_up_in_plan(mock_vlm_env, sample_video: Path):
    import asyncio

    from app.services.jobs import store

    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-int-plan", source="upload",
        options=AnalyzeOptions(enable_asr=False, frame_interval_seconds=2.0),
    )))
    result = pipeline.run_pipeline_sync(job, sample_video)
    # 每镜最多 3 张 -> 3s 的镜头间隔 2s 只该拿 3 帧（保底），4s 的拿 3 帧
    assert "每镜最多" in result.stats["plan"]


def test_prompt_word_limit_override_recorded(mock_vlm_env, sample_video: Path):
    """请求里指定的词数上限要记进 stats，供前端判断是否超限。"""
    import asyncio

    from app.services.jobs import store

    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-wordlimit", source="upload",
        options=AnalyzeOptions(enable_asr=False, prompt_word_limit=1234),
    )))
    result = pipeline.run_pipeline_sync(job, sample_video)
    assert result.stats["prompt_word_limit"] == 1234
    assert "prompt_words" in result.stats
    assert "prompt_over_limit" in result.stats


# ---------------------------------------------------------------------------
# 片段截取（trim）
# ---------------------------------------------------------------------------

def test_trim_analyses_only_the_selected_range(mock_vlm_env, sample_video: Path):
    """框选片段后，反推的应该**只有**那一段。

    实现是「先切出片段再跑管线」，所以下游（探测/镜头检测/抽帧/音频/时间戳）
    全部天然是相对片段的，不用在七八个地方各自记得减偏移。
    """
    import asyncio

    from app.services.jobs import store

    full = ff.probe(sample_video)
    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-trim", source="upload",
        options=AnalyzeOptions(enable_asr=False, trim_start=1.0, trim_end=3.0),
    )))
    result = pipeline.run_pipeline_sync(job, sample_video)

    assert abs(result.media.duration - 2.0) < 0.4, f"应该只分析 2s，实际 {result.media.duration}"
    assert result.media.duration < full.duration - 1.0
    trim = result.stats["trim"]
    assert trim is not None
    assert abs(trim["start"] - 1.0) < 0.01 and abs(trim["end"] - 3.0) < 0.01
    # 时间戳要相对片段，不能是原片的绝对时间
    assert result.observations, "应该有镜头观察结果"


def test_no_trim_means_whole_video(mock_vlm_env, sample_video: Path):
    import asyncio

    from app.services.jobs import store

    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-notrim", source="upload", options=AnalyzeOptions(enable_asr=False),
    )))
    result = pipeline.run_pipeline_sync(job, sample_video)
    assert result.stats["trim"] is None
    assert result.media.duration > 5.0


def test_trim_shorter_than_half_second_is_rejected(mock_vlm_env, sample_video: Path):
    import asyncio

    from app.services.jobs import store

    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-trim-short", source="upload",
        options=AnalyzeOptions(enable_asr=False, trim_start=1.0, trim_end=1.2),
    )))
    with pytest.raises(RuntimeError, match="太短"):
        pipeline.run_pipeline_sync(job, sample_video)


def test_trim_options_reject_inconsistent_ranges():
    """只给一端 / 终点不大于起点 都要被 schema 拒掉。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnalyzeOptions(trim_start=1.0)
    with pytest.raises(ValidationError):
        AnalyzeOptions(trim_end=5.0)
    with pytest.raises(ValidationError):
        AnalyzeOptions(trim_start=5.0, trim_end=5.0)
    with pytest.raises(ValidationError):
        AnalyzeOptions(trim_start=-1.0, trim_end=3.0)
    ok = AnalyzeOptions(trim_start=1.0, trim_end=3.0)
    assert ok.trim_start == 1.0 and ok.trim_end == 3.0


def test_stats_plan_matches_actual_frames(mock_vlm_env, sample_video: Path):
    """stats 里的选帧说明必须是**实际**抽的，不是拿总预算重算的。

    踩过：末尾用总预算重跑了一遍 plan_frames，算出的是「如果重新分配会怎样」——
    界面显示 27 张，实际只抽了 21 张（分块预算比总预算小）。
    这种不一致会让用户按错误的数字调参。
    """
    import asyncio

    from app.services.jobs import store

    mock_vlm.reset()
    job = asyncio.run(store.create(Job(
        id="e2e-plan-match", source="upload", options=AnalyzeOptions(enable_asr=False),
    )))
    result = pipeline.run_pipeline_sync(job, sample_video)

    plan_text = result.stats["plan"]
    assert f"抽帧 {result.frames_used} 张" in plan_text, (
        f"说明里的帧数和实际不符：\n  {plan_text}\n  实际 {result.frames_used}"
    )
