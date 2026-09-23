"""抓取模块的单元测试。

只测不联网的部分：链接提取、平台识别、yt-dlp 选项构造。
真实下载依赖外网和平台状态，属于端到端验证范畴，不放进来。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.services import downloader as dl

# ---------------------------------------------------------------------------
# 链接提取
# ---------------------------------------------------------------------------

def test_extract_url_from_plain_link():
    assert dl.extract_url("https://www.bilibili.com/video/BV1xx411c7mD") == \
        "https://www.bilibili.com/video/BV1xx411c7mD"


def test_extract_url_from_douyin_share_text():
    """抖音的分享文案是中文夹链接，正则必须能剥掉两边的中文和标点。"""
    text = "7.32 复制打开抖音，看看【某某】的作品 https://v.douyin.com/iRNBho6u/ 内容不错！"
    assert dl.extract_url(text) == "https://v.douyin.com/iRNBho6u/"


def test_extract_url_strips_trailing_bracket():
    """中文括号紧贴链接时不能把括号吃进去。"""
    assert dl.extract_url("看这个（https://v.douyin.com/abc/）") == "https://v.douyin.com/abc/"


def test_extract_url_returns_none_without_link():
    assert dl.extract_url("这段文字里没有任何链接") is None


def test_extract_url_ignores_trailing_chinese_comma():
    assert dl.extract_url("链接：https://b23.tv/abcdefg，记得看") == "https://b23.tv/abcdefg"


# ---------------------------------------------------------------------------
# 平台识别
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.bilibili.com/video/BV1xx411c7mD", "bilibili"),
    ("https://b23.tv/abcdefg", "bilibili"),
    ("https://m.bilibili.com/video/BV1xx", "bilibili"),
    ("https://v.douyin.com/iRNBho6u/", "douyin"),
    ("https://www.douyin.com/video/7234567890", "douyin"),
    ("https://www.iesdouyin.com/share/video/7234567890/", "douyin"),
])
def test_detect_platform(url: str, expected: str):
    assert dl.detect_platform(url) == expected


def test_detect_platform_unknown():
    assert dl.detect_platform("https://example.com/video/1") not in ("bilibili", "douyin")


# ---------------------------------------------------------------------------
# 抖音 aweme_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.douyin.com/video/7234567890123456789",
    "https://www.iesdouyin.com/share/video/7234567890123456789/?region=CN",
    "https://www.douyin.com/?modal_id=7234567890123456789",
])
def test_extract_aweme_id(url: str):
    assert dl._extract_aweme_id(url) == "7234567890123456789"


def test_extract_aweme_id_none_when_absent():
    assert dl._extract_aweme_id("https://www.douyin.com/user/xyz") is None


# ---------------------------------------------------------------------------
# yt-dlp 选项
# ---------------------------------------------------------------------------

def test_yt_dlp_options_includes_ffmpeg_location(ffmpeg_bin: str):
    """回归测试：`bv*+ba` 合并音视频必须调用 ffmpeg。

    本机 ffmpeg 常常不在 PATH 里（只有 ffmpeg-static 那种单文件）。
    不显式告诉 yt-dlp 位置，合并步骤必然失败，表现为「解析成功但下载失败」，
    很容易被误判成平台反爬，排查方向完全跑偏。
    """
    opts = dl._yt_dlp_options()
    assert opts.get("ffmpeg_location"), "没传 ffmpeg_location，音视频合并会失败"
    assert Path(opts["ffmpeg_location"]).exists()


def test_yt_dlp_options_defaults_are_quiet():
    opts = dl._yt_dlp_options()
    assert opts["quiet"] is True
    assert opts["noplaylist"] is True
    assert "User-Agent" in opts["http_headers"]


def test_yt_dlp_options_reads_cookie_settings(monkeypatch):
    from app.core import config as cfg

    monkeypatch.setenv("COOKIES_FILE", "C:/tmp/cookies.txt")
    cfg.refresh_settings()
    try:
        assert dl._yt_dlp_options()["cookiefile"] == "C:/tmp/cookies.txt"
    finally:
        monkeypatch.delenv("COOKIES_FILE", raising=False)
        cfg.refresh_settings()


def test_ytdlp_format_caps_resolution_by_default():
    """反推会把帧缩到长边 896，下 1080p 是纯浪费。默认必须封顶。"""
    from app.core.config import get_settings

    fmt = get_settings().ytdlp_format
    assert "height<=720" in fmt
    assert "height<=720" in fmt.split("/")[0], "第一优先项就该是 720p 封顶"
    assert fmt.endswith("bv*+ba/b"), "最后要有不封顶的兜底，否则低清源会直接失败"


# ---------------------------------------------------------------------------
# 下载通道
# ---------------------------------------------------------------------------

def test_download_passes_configured_format(monkeypatch, tmp_path: Path):
    """下载时要真的用上 YTDLP_FORMAT，不能写死。"""
    captured: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            out = Path(captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4"))
            out.write_bytes(b"\x00")
            return {"id": "fake", "ext": "mp4"}

        def prepare_filename(self, info):
            return captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4")

    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = FakeYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    from app.core.config import get_settings
    path, reason = dl.download_with_ytdlp("https://example.com/v", tmp_path)

    assert reason == ""
    assert path is not None and path.is_file()
    assert captured["format"] == get_settings().ytdlp_format
    assert captured["merge_output_format"] == "mp4"


def test_download_returns_reason_on_failure(monkeypatch, tmp_path: Path):
    """失败原因必须往上传，否则界面只能给「会员内容/需要登录」这种通用猜测。"""

    class BoomYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            raise RuntimeError("ffmpeg not found\nplease install")

    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = BoomYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    path, reason = dl.download_with_ytdlp("https://example.com/v", tmp_path)
    assert path is None
    assert "ffmpeg not found" in reason
    assert "\n" not in reason, "换行会破坏错误提示的排版"


def test_download_hint_includes_ytdlp_reason(monkeypatch, tmp_path: Path):
    """最终错误提示里要带上 yt-dlp 原始报错，否则排查全靠猜。"""
    from app.core import config as cfg

    monkeypatch.setattr(dl, "TMP_DIR", tmp_path)
    monkeypatch.setattr(dl, "download_with_ytdlp",
                        lambda url, out: (None, "Requested format is not available"))

    path, info, hint = dl.download("https://www.bilibili.com/video/BV1xx411c7mD", "job1")
    assert path is None
    assert info.platform == "bilibili"
    assert "Requested format is not available" in hint
    assert "COOKIES_FROM_BROWSER" in hint, "仍要保留可操作建议"
    cfg.refresh_settings()


def test_download_skips_ytdlp_when_unavailable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(dl, "TMP_DIR", tmp_path)
    monkeypatch.setattr(dl, "yt_dlp_available", lambda: False)

    path, info, hint = dl.download("https://www.bilibili.com/video/BV1xx411c7mD", "job2")
    assert path is None
    assert "未安装 yt-dlp" in hint


# ---------------------------------------------------------------------------
# 删除保护环境下的容错
# ---------------------------------------------------------------------------

def _fake_ytdlp(monkeypatch, behavior):
    """装一个假 yt_dlp 模块。behavior(captured, out_dir) 决定 extract_info 干什么。"""
    captured: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            return behavior(captured, self)

        def prepare_filename(self, info):
            return captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4")

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = FakeYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    return captured


def test_ytdlp_opts_keep_intermediate_files(monkeypatch, tmp_path: Path):
    """必须传 keepvideo=True。

    yt-dlp 合并完成后默认会 os.remove 掉分离的音视频流，而某些运行环境
    在 os.remove 上装了删除保护，一删就抛 SystemExit 把下载任务带走 ——
    实测报「SystemExit: 1」，而视频其实早就下好、也合并完了。
    """
    def behavior(captured, ydl):
        out = Path(captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4"))
        out.write_bytes(b"\x00" * 2048)
        return {"id": "fake", "ext": "mp4"}

    captured = _fake_ytdlp(monkeypatch, behavior)
    dl.download_with_ytdlp("https://example.com/v", tmp_path)
    assert captured.get("keepvideo") is True


def test_download_recovers_when_delete_guard_kills_postprocess(monkeypatch, tmp_path: Path):
    """yt-dlp 抛 SystemExit 但文件已产出时，要能认出来并继续。

    这是本机真实故障：视频下载+合并都成功，死在删除原始流那一步。
    报「下载失败」是误报 —— 要用文件系统的事实判断，别相信异常。
    """
    def behavior(captured, ydl):
        out = Path(captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4"))
        out.write_bytes(b"\x00" * 4096)
        raise SystemExit(1)

    _fake_ytdlp(monkeypatch, behavior)
    path, reason = dl.download_with_ytdlp("https://example.com/v", tmp_path)

    assert reason == "", f"不该报失败，实际：{reason}"
    assert path is not None and path.is_file()


def test_download_recovers_from_plain_exception_too(monkeypatch, tmp_path: Path):
    """普通异常也一样：先看磁盘再决定报不报失败。"""
    def behavior(captured, ydl):
        out = Path(captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4"))
        out.write_bytes(b"\x00" * 4096)
        raise RuntimeError("post-processing blew up")

    _fake_ytdlp(monkeypatch, behavior)
    path, reason = dl.download_with_ytdlp("https://example.com/v", tmp_path)
    assert reason == ""
    assert path is not None


def test_download_still_reports_failure_when_nothing_on_disk(monkeypatch, tmp_path: Path):
    """真失败（磁盘上什么都没有）时仍要如实报错。"""
    def behavior(captured, ydl):
        raise SystemExit(1)

    _fake_ytdlp(monkeypatch, behavior)
    path, reason = dl.download_with_ytdlp("https://example.com/v", tmp_path)
    assert path is None
    assert "SystemExit" in reason, f"要带上真实异常类型，实际：{reason}"


def test_find_downloaded_video_picks_largest(tmp_path: Path):
    """兜底：目录里有多个视频文件时挑最大的。

    keepvideo=True 会留下分离的音视频流，音频流通常小得多，最大的就是成片。
    """
    (tmp_path / "a.f30280.m4a").write_bytes(b"\x00" * 100)
    small = tmp_path / "a.f100026.mp4"
    small.write_bytes(b"\x00" * 500)
    big = tmp_path / "a.mp4"
    big.write_bytes(b"\x00" * 5000)

    got = dl._find_downloaded_video(tmp_path, tmp_path / "nonexistent.mp4")
    assert got == big


def test_find_downloaded_video_prefers_expected_path(tmp_path: Path):
    """期望路径存在时优先用它，不做多余的扫描。"""
    want = tmp_path / "v.mp4"
    want.write_bytes(b"\x00" * 10)
    (tmp_path / "other.mp4").write_bytes(b"\x00" * 99999)
    assert dl._find_downloaded_video(tmp_path, want) == want


def test_find_downloaded_video_ignores_audio_only(tmp_path: Path):
    """只有音频流时不该当成成片返回。"""
    (tmp_path / "a.m4a").write_bytes(b"\x00" * 9999)
    assert dl._find_downloaded_video(tmp_path, tmp_path / "x.mp4") is None


# ---------------------------------------------------------------------------
# 删除保护环境下的容错
# ---------------------------------------------------------------------------

def _fake_ytdlp(monkeypatch, behavior):
    """装一个假 yt_dlp 模块。behavior(captured, out_dir) 决定 extract_info 干什么。"""
    captured: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            return behavior(captured, self)

        def prepare_filename(self, info):
            return captured["outtmpl"].replace("%(id)s", "fake").replace("%(ext)s", "mp4")

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = FakeYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    return captured
