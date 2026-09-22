"""API 层集成测试（TestClient，不调用真实模型）。

覆盖：健康检查、上传校验、反推任务创建、任务查询、Range 视频流、
提示词库增删。模型调用被刻意排除——那是外部依赖，属于端到端验证范畴。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services import storage


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["ffmpeg"]
    assert body["db"].endswith(".db")


def test_formats(client):
    r = client.get("/api/formats")
    assert r.status_code == 200
    values = {f["value"] for f in r.json()}
    assert values == {"h3", "h3-ref", "seedance", "generic"}
    assert all(f["label"] for f in r.json())


def test_settings_masks_api_key(client):
    body = client.get("/api/settings").json()
    assert "api_key_masked" in body["vlm"]
    assert "test-key-not-real" not in str(body)


def test_upload_rejects_non_video(client):
    r = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "不支持" in r.json()["detail"]


def test_upload_rejects_empty_file(client):
    r = client.post(
        "/api/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r.status_code == 400


def test_upload_accepts_video(client, sample_video: Path):
    r = client.post(
        "/api/upload",
        files={"file": ("clip.mp4", sample_video.read_bytes(), "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file_id"].endswith(".mp4")
    assert body["duration"] == pytest.approx(10.0, abs=0.5)
    assert body["has_audio"] is True
    assert body["video_url"].startswith("/api/media/upload/")


def test_upload_keeps_chinese_display_name(client, sample_video: Path):
    """中文名要保留在展示字段里，但落盘文件名必须是纯 ASCII。

    原因：中文出现在 URL 路径里，只要有一环没做百分号编码（curl、部分代理、
    Content-Disposition）就会 400。展示与落盘分离，两边各取所需。
    """
    r = client.post(
        "/api/upload",
        files={"file": ("我的测试视频.mp4", sample_video.read_bytes(), "video/mp4")},
    )
    assert r.status_code == 200
    body = r.json()

    assert body["name"] == "我的测试视频.mp4"
    assert "我的测试视频" not in body["file_id"]
    assert body["file_id"].isascii()
    assert body["file_id"].endswith(".mp4")
    assert body["video_url"].isascii()


def test_media_stream_supports_range(client, sample_video: Path):
    up = client.post(
        "/api/upload",
        files={"file": ("range.mp4", sample_video.read_bytes(), "video/mp4")},
    ).json()

    # 完整请求
    full = client.get(up["video_url"])
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"

    # Range 请求 → 206
    part = client.get(up["video_url"], headers={"Range": "bytes=0-1023"})
    assert part.status_code == 206
    assert part.headers["content-range"].startswith("bytes 0-1023/")
    assert len(part.content) == 1024


def test_media_stream_missing_file(client):
    assert client.get("/api/media/upload/nope.mp4").status_code == 404


def test_range_works_for_chinese_named_upload(client, sample_video: Path):
    """中文名上传后，返回的 video_url 必须能直接喂给 <video>（即纯 ASCII 且支持 Range）。"""
    up = client.post(
        "/api/upload",
        files={"file": ("产品演示视频.mp4", sample_video.read_bytes(), "video/mp4")},
    ).json()
    assert up["video_url"].isascii()

    part = client.get(up["video_url"], headers={"Range": "bytes=100-1099"})
    assert part.status_code == 206
    assert len(part.content) == 1000
    assert part.headers["content-range"].startswith("bytes 100-1099/")


def test_analyze_requires_existing_file(client):
    r = client.post("/api/analyze", json={"file_id": "missing.mp4"})
    assert r.status_code == 404


def test_analyze_creates_job(client, sample_video: Path):
    """任务会被创建；后台管线随后会因为模型不可用而失败，这里只验证接口契约。"""
    up = client.post(
        "/api/upload",
        files={"file": ("job.mp4", sample_video.read_bytes(), "video/mp4")},
    ).json()

    r = client.post("/api/analyze", json={
        "file_id": up["file_id"],
        "name": "job.mp4",
        "options": {"format": "seedance", "language": "zh", "enable_asr": False},
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert len(job_id) == 16

    detail = client.get(f"/api/jobs/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == job_id
    assert body["source"] == "upload"
    assert body["options"]["format"] == "seedance"
    assert body["options"]["language"] == "zh"


def test_analyze_ignores_unknown_option_keys(client, sample_video: Path):
    up = client.post(
        "/api/upload",
        files={"file": ("opt.mp4", sample_video.read_bytes(), "video/mp4")},
    ).json()
    r = client.post("/api/analyze", json={
        "file_id": up["file_id"],
        "options": {"format": "h3", "bogus_key": 123},
    })
    assert r.status_code == 200


def test_get_missing_job_returns_404(client):
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_cancel_missing_job(client):
    r = client.post("/api/jobs/doesnotexist/cancel")
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_fetch_probe_requires_url(client):
    assert client.post("/api/fetch/probe", json={"url": ""}).status_code == 400


def test_fetch_probe_extracts_url_from_share_text(client):
    """分享文案里混着中文和表情，也要能抠出链接。"""
    r = client.post("/api/fetch/probe", json={
        "url": "7.32 复制打开抖音，看看【某某】的作品 https://v.douyin.com/iABCDEF/ 好有意思",
    })
    # 不要求真能下载成功，只要不是「缺少链接」的 400
    assert r.status_code == 200
    assert r.json()["platform"] == "douyin"


def test_jobs_list(client):
    r = client.get("/api/jobs?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_jobs_list_rejects_bad_limit(client):
    assert client.get("/api/jobs?limit=0").status_code == 422
    assert client.get("/api/jobs?limit=9999").status_code == 422


def test_prompt_library_roundtrip(client):
    created = client.post("/api/prompts", json={
        "job_id": "abc",
        "title": "测试提示词",
        "content": "[Shot 1] a woman walks...",
        "format": "h3",
        "tags": "测试,走路",
    })
    assert created.status_code == 200
    pid = created.json()["id"]

    items = client.get("/api/prompts").json()
    assert any(i["id"] == pid for i in items)

    filtered = client.get("/api/prompts?keyword=走路").json()
    assert any(i["id"] == pid for i in filtered)

    assert client.delete(f"/api/prompts/{pid}").json()["ok"] is True
    assert not any(i["id"] == pid for i in client.get("/api/prompts").json())


def test_prompt_requires_content(client):
    assert client.post("/api/prompts", json={"title": "empty"}).status_code == 400


def test_storage_persists_job_roundtrip(client, sample_video: Path):
    """任务落库后，即使内存里没有，也能从数据库读回来。"""
    from app.schemas import AnalyzeOptions, Job, JobProgress, JobResult
    from app.services.jobs import store

    job = Job(
        id="persist00000001",
        state="succeeded",
        title="持久化测试",
        source="upload",
        options=AnalyzeOptions(format="h3-ref"),
        progress=JobProgress(stage="done", percent=100),
        result=JobResult(prompt="hello prompt", frames_used=12, chunks=2),
    )
    storage.save_job(job)

    # 从内存里踢掉，强制走数据库
    store._jobs.pop(job.id, None)

    loaded = store.get(job.id)
    assert loaded is not None
    assert loaded.title == "持久化测试"
    assert loaded.options.format == "h3-ref"
    assert loaded.result is not None
    assert loaded.result.prompt == "hello prompt"
    assert loaded.result.frames_used == 12

    summaries = storage.list_jobs(limit=50)
    assert any(s.id == job.id for s in summaries)


def test_settings_whitelist_rejects_unknown_keys(client):
    r = client.post("/api/settings", json={"SOME_RANDOM_KEY": "x"})
    assert r.status_code == 200
    assert r.json()["ok"] is False
