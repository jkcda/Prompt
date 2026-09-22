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
