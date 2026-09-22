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


def test_formats_returns_two_primary_modes(client):
    """对外只有两种模式：H3 与 Seedance。generic 作为兜底，primary=False。"""
    r = client.get("/api/formats")
    assert r.status_code == 200
    modes = r.json()

    primary = [m for m in modes if m["primary"]]
    assert [m["value"] for m in primary] == ["h3", "seedance"]
    assert all(m["label"] and m["description"] for m in modes)

    # 每个变体都要归属到自己的模式
    for m in modes:
        assert m["variants"], f"{m['value']} 没有变体"
        for v in m["variants"]:
            assert v["mode"] == m["value"]
            assert v["label"] and v["description"]
        assert sum(1 for v in m["variants"] if v["default"]) == 1


def test_formats_h3_mode_has_two_variants(client):
    """T2VA 与 Ref2VA 写法不同（有没有参考素材），必须在 H3 模式下显式选。"""
    modes = {m["value"]: m for m in client.get("/api/formats").json()}
    h3 = modes["h3"]
    assert [v["value"] for v in h3["variants"]] == ["h3", "h3-ref"]
    assert [v["value"] for v in h3["variants"] if v["default"]] == ["h3"]
    assert [v["value"] for v in modes["seedance"]["variants"]] == ["seedance"]


def test_formats_covers_all_formats(client):
    values = {
        v["value"]
        for m in client.get("/api/formats").json()
        for v in m["variants"]
    }
    assert values == {"h3", "h3-ref", "seedance", "generic"}


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


def test_storage_persists_subject_registry(client):
    """主体登记表也要落库，否则从历史打开 Ref2VA 结果会丢参考标签依据。"""
    from app.schemas import AnalyzeOptions, Job, JobResult, SubjectEntry
    from app.services.jobs import store

    job = Job(
        id="persist00000002",
        state="succeeded",
        source="upload",
        options=AnalyzeOptions(format="h3-ref"),
        result=JobResult(
            prompt="subject_definitions:\n<Subject 1> ...",
            subjects=[
                SubjectEntry(label="performer", kind="person",
                             description="dark jacket", shots=["1", "3"],
                             notes="hair tied back"),
                SubjectEntry(label="rooftop", kind="environment", shots=["1"]),
            ],
        ),
    )
    storage.save_job(job)
    store._jobs.pop(job.id, None)

    loaded = store.get(job.id)
    assert loaded is not None and loaded.result is not None
    assert [s.label for s in loaded.result.subjects] == ["performer", "rooftop"]
    assert loaded.result.subjects[0].shots == ["1", "3"]
    assert loaded.result.subjects[0].notes == "hair tied back"


def test_schema_adds_missing_columns_on_existing_db(client, tmp_path):
    """老库升级后缺列时，init_db 要能补上，而不是报 no such column。"""
    from sqlalchemy import inspect, text

    from app.core.db import get_engine, init_db

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS jobs"))
        conn.execute(text(
            "CREATE TABLE jobs ("
            "id VARCHAR PRIMARY KEY, state VARCHAR, created_at FLOAT, "
            "observations_json VARCHAR, prompt VARCHAR)"
        ))

    init_db()

    columns = {c["name"] for c in inspect(engine).get_columns("jobs")}
    assert "subjects_json" in columns
    assert "observations_json" in columns


def test_settings_whitelist_rejects_unknown_keys(client):
    r = client.post("/api/settings", json={"SOME_RANDOM_KEY": "x"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_mark_interrupted_clears_zombie_jobs(client):
    """服务被 kill 时留在库里的 running 任务，启动时要收成 failed。

    不然界面上会一直显示「进行中」，其实进程早没了，用户白等。
    """
    from app.schemas import AnalyzeOptions, Job, JobResult
    from app.services import storage

    zombie = Job(id="zombie00000001", state="running", source="upload",
                 options=AnalyzeOptions())
    # 真正的成功任务要有产出，否则会被当成「假成功」一起修掉
    done = Job(id="zombie00000002", state="succeeded", source="upload",
               options=AnalyzeOptions(),
               result=JobResult(prompt="a real prompt", frames_used=9))
    storage.save_job(zombie)
    storage.save_job(done)

    n = storage.mark_interrupted()
    assert n >= 1

    reloaded = storage.load_job("zombie00000001")
    assert reloaded is not None
    assert reloaded.state == "failed"
    assert "中断" in reloaded.error

    # 有产出的成功任务不能被误伤
    ok = storage.load_job("zombie00000002")
    assert ok is not None and ok.state == "succeeded"


def test_mark_interrupted_is_idempotent(client):
    from app.services import storage

    assert storage.mark_interrupted() == 0


def test_mark_interrupted_repairs_fake_success(client):
    """succeeded 但 prompt 为空 = 假成功，必须修掉。

    服务关闭时 asyncio 任务被取消，旧逻辑把「无结果」判成成功，
    界面显示「完成」但点进去没提示词 —— 比报失败更难查。
    """
    from app.schemas import AnalyzeOptions, Job, JobResult
    from app.services import storage

    ghost = Job(id="ghost000000001", state="succeeded", source="upload",
                options=AnalyzeOptions(),
                result=JobResult(prompt="", frames_used=0))
    storage.save_job(ghost)

    storage.mark_interrupted()

    reloaded = storage.load_job("ghost000000001")
    assert reloaded is not None
    assert reloaded.state == "failed"
    assert "未产出结果" in reloaded.error


def test_finish_without_result_is_not_success(client):
    """核心回归：finish(id, None, "") 不能判成 succeeded。"""
    import asyncio

    from app.schemas import AnalyzeOptions, Job
    from app.services.jobs import store

    async def scenario():
        job = await store.create(Job(id="noresult000001", source="upload",
                                     options=AnalyzeOptions()))
        await store.finish(job.id, None, "")
        return store.get(job.id)

    got = asyncio.run(scenario())
    assert got is not None
    assert got.state == "failed", "没有结果却记成了成功"
    assert got.error


# ---------------------------------------------------------------------------
# 模型热切换（用户要「自主切换模型」）
# ---------------------------------------------------------------------------

def test_settings_hot_switch_reflects_in_health(client):
    """改模型不该需要重启服务 —— 写配置后 /api/health 必须立刻反映。"""
    from app.core.config import get_settings

    original = get_settings().vlm_model
    try:
        r = client.post("/api/settings", json={"VLM_MODEL": "Vendor/Another-Model"})
        assert r.json()["ok"] is True
        assert "VLM_MODEL" in r.json()["updated"]

        assert client.get("/api/health").json()["vlm_model"] == "Vendor/Another-Model"
        assert client.get("/api/settings").json()["vlm"]["model"] == "Vendor/Another-Model"
    finally:
        client.post("/api/settings", json={"VLM_MODEL": original})
    assert client.get("/api/health").json()["vlm_model"] == original


def test_settings_hot_switch_base_url(client):
    from app.core.config import get_settings

    original = get_settings().vlm_base_url
    try:
        client.post("/api/settings", json={"VLM_BASE_URL": "https://example.com/v1"})
        assert client.get("/api/health").json()["vlm_model"]  # 仍然可用
        assert client.get("/api/settings").json()["vlm"]["base_url"] == "https://example.com/v1"
    finally:
        client.post("/api/settings", json={"VLM_BASE_URL": original})


def test_settings_omitting_key_does_not_wipe_it(client):
    """切换模型时不该被迫重填 key —— 不传就不能动它。"""
    from app.core.config import get_settings

    before = get_settings().vlm_api_key
    assert before, "测试前提：应该已有 key"

    client.post("/api/settings", json={"VLM_MODEL": "Vendor/X"})
    assert get_settings().vlm_api_key == before

    masked = client.get("/api/settings").json()["vlm"]["api_key_masked"]
    assert masked and before not in masked


def test_settings_audio_input_toggle(client):
    from app.core.config import get_settings

    original = get_settings().vlm_audio_input
    try:
        client.post("/api/settings", json={"VLM_AUDIO_INPUT": "true"})
        assert get_settings().vlm_audio_input is True
        assert client.get("/api/settings").json()["vlm"]["audio_input"] is True
    finally:
        client.post("/api/settings", json={"VLM_AUDIO_INPUT": "false" if not original else "true"})
    assert get_settings().vlm_audio_input == original


def test_models_endpoint_returns_shape_on_failure(client):
    """拉不到列表时要给出原因，不能 500。

    conftest 把 base_url 指到 127.0.0.1:9（discard 端口），必然连不上。
    """
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["models"] == []
    assert body["message"]


def test_models_endpoint_accepts_base_url_override(client):
    r = client.get("/api/models", params={"base_url": "http://127.0.0.1:9/v1"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_test_vlm_config_endpoint_accepts_overrides(client):
    """POST /api/health/vlm 要能试一组未保存的配置。"""
    r = client.post("/api/health/vlm", json={
        "model": "Vendor/Probe-Target",
        "base_url": "http://127.0.0.1:9/v1",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "Vendor/Probe-Target"
    assert body["base_url"] == "http://127.0.0.1:9/v1"
    assert body["ok"] is False
    assert body["vision"] is False
    assert body["inconclusive"] is False
    assert body["message"]


def test_test_vlm_config_falls_back_to_saved_key(client):
    """只改模型名时不该要求重填 key。"""
    from app.core.config import get_settings

    r = client.post("/api/health/vlm", json={"model": get_settings().vlm_model})
    assert r.status_code == 200
    assert r.json()["configured"] is True


def test_vision_probe_image_is_generated(ffmpeg_bin: str):
    """自检图片必须真的能生成，否则自检会假阳性。"""
    from app.services.vlm import _vision_probe_image

    p = _vision_probe_image()
    assert p is not None and p.is_file()
    assert p.stat().st_size > 0
    assert p.read_bytes()[:2] == b"\xff\xd8"  # JPEG SOI


def test_settings_write_does_not_touch_real_env_file(client):
    """跑测试不能改真实的 backend/.env。

    踩过：测试把 VLM_MODEL 写成 Vendor/X，测试全绿但服务起不来了。
    现在写入目标由 ENV_FILE 指向临时文件，真实 .env 必须纹丝不动。
    """
    import os

    from app.core.config import BACKEND_DIR, env_file_path

    real_env = BACKEND_DIR / ".env"
    before = real_env.read_bytes() if real_env.is_file() else None

    client.post("/api/settings", json={"VLM_MODEL": "Vendor/Pollution-Probe"})

    after = real_env.read_bytes() if real_env.is_file() else None
    assert before == after, "测试污染了真实的 backend/.env！"
    assert "Pollution-Probe" not in (after or b"").decode("utf-8", "replace")

    # 写入应该落在临时文件上
    target = env_file_path()
    assert target != real_env
    assert target.is_file()
    assert "Pollution-Probe" in target.read_text(encoding="utf-8")
    assert os.environ["ENV_FILE"] == str(target)
