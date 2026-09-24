"""pytest 公共夹具。

注意：必须在导入 app 之前设置环境变量，因为 Settings 是单例缓存的。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---- 先设环境，再导入应用 ----
_TMP = Path(tempfile.mkdtemp(prefix="vpr-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["MAX_UPLOAD_MB"] = "50"
os.environ["HISTORY_KEEP_DAYS"] = "0"
# 关键：把配置写回目标指到临时文件。
# 否则跑测试会往真实的 backend/.env 里写假模型名，
# 测试全绿但服务起不来（踩过）。
os.environ["ENV_FILE"] = str(_TMP / "test.env")
os.environ.setdefault("VLM_API_KEY", "test-key-not-real")
os.environ.setdefault("VLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("VLM_MODEL", "test-model")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import resolve_ffmpeg  # noqa: E402
from app.core.db import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database() -> None:
    init_db()


@pytest.fixture(autouse=True)
def _no_local_whisper(monkeypatch):
    """测试里绝不加载本地 whisper 模型。

    ⚠️ 装了 faster-whisper 之后，只要走 enable_asr=True 的测试都会真的去
    HuggingFace 下载 460MB 的 small 权重 —— 测试会挂几分钟，还依赖外网，
    表现成「pytest 卡住不动」。所以默认把它关掉；
    真要测转写的地方自己 monkeypatch 覆盖这个 fixture。
    """
    from app.services import asr

    monkeypatch.setattr(asr, "_local_whisper_available", lambda: False)
    yield


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    path = resolve_ffmpeg()
    if not path:
        pytest.skip("未找到 ffmpeg，跳过依赖抽帧的测试")
    return path


@pytest.fixture(scope="session")
def sample_video(ffmpeg_bin: str) -> Path:
    """生成一段 10 秒、含 3 个明显场景切换、带音轨的测试视频。"""
    out = _TMP / "sample.mp4"
    if out.is_file():
        return out

    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=4",
        "-f", "lavfi", "-i", "smptebars=size=320x180:rate=24:duration=3",
        "-f", "lavfi", "-i", "color=c=blue:size=320x180:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a",
        "-pix_fmt", "yuv420p", "-shortest", "-y", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return out


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
