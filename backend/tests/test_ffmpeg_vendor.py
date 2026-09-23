"""ffmpeg 打包与跨平台解析的测试。

仓库里带了两份二进制（`backend/vendor/ffmpeg/`）：
Windows 用 `ffmpeg.exe`，Linux 用 `ffmpeg`（无扩展名）。

查找优先级：显式配置 > vendor > runtime > 系统 PATH > 常见安装位置。
**自带排在 PATH 之前**是为了让部署可复现 —— 服务器上装了什么版本
不该影响这个服务的输出。
"""

from __future__ import annotations

from pathlib import Path

from app.core import config as cfg


def test_vendor_dir_is_under_backend():
    assert cfg.VENDOR_FFMPEG_DIR == cfg.BACKEND_DIR / "vendor" / "ffmpeg"


def test_vendor_is_searched_before_system_path():
    """自带二进制必须排在 PATH 之前，否则部署机上的版本会盖掉它。"""
    cands = [str(p).replace("\\", "/") for p in cfg._candidates(("ffmpeg",))]
    vendor_idx = next((i for i, c in enumerate(cands) if "vendor/ffmpeg" in c), None)
    assert vendor_idx is not None, "候选列表里没有 vendor/ffmpeg"

    # PATH 命中的路径不在 vendor 之后（没装 ffmpeg 时 PATH 里没有，也算通过）
    import shutil

    on_path = shutil.which("ffmpeg")
    if on_path:
        path_idx = next((i for i, c in enumerate(cands) if Path(c) == Path(on_path)), None)
        assert path_idx is None or vendor_idx < path_idx, "vendor 应该排在 PATH 之前"


def test_no_hardcoded_machine_specific_path():
    """不该再硬编码开发机上别的项目的路径。

    原来有 `D:/nexus/aiconnent/.../ffmpeg-static`。那种路径在服务器上必然不存在，
    留着只会让「本地能跑、上线找不到 ffmpeg」这种问题更难查。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cfg._candidates))
    # 只看代码里的字符串常量，不看注释和文档字符串
    # （注释里提到 nexus 是解释为什么删掉的）
    literals = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    joined = " ".join(literals).lower()
    assert "nexus" not in joined, f"还有硬编码路径：{[s for s in literals if 'nexus' in s.lower()]}"
    assert "aiconnent" not in joined


def test_candidates_use_platform_suffix():
    """Windows 找 .exe，Linux 找无扩展名的 —— 两份二进制要各自被找到。"""
    import inspect

    src = inspect.getsource(cfg._candidates)
    assert 'exe = ".exe" if os.name == "nt" else ""' in src

    # 仓库里两份都在
    vendor = cfg.VENDOR_FFMPEG_DIR
    if vendor.exists():
        names = {p.name for p in vendor.iterdir()}
        assert "ffmpeg.exe" in names, "缺 Windows 版"
        assert "ffmpeg" in names, "缺 Linux 版"


def test_resolved_ffmpeg_exists_and_is_executable_file():
    got = cfg.resolve_ffmpeg()
    if got is None:
        return  # 环境里完全没有 ffmpeg 时跳过（CI 上可能如此）
    p = Path(got)
    assert p.is_file(), f"解析到 {got} 但不是文件"
    assert p.stat().st_size > 1024 * 1024, "ffmpeg 二进制不该这么小"


def test_ffmpeg_available_matches_resolver():
    from app.services import ffmpeg as ff

    assert ff.ffmpeg_available() == (cfg.resolve_ffmpeg() is not None)


def test_ffprobe_is_optional():
    """项目不依赖 ffprobe —— 缺失时用 `ffmpeg -i` 解析媒体信息。"""
    import inspect

    from app.services import ffmpeg as ff

    assert "resolve_ffprobe" in inspect.getsource(cfg)
    # probe 里要有 ffprobe 缺失时的回退路径
    src = inspect.getsource(ff.probe)
    assert "_probe_with_ffmpeg" in src or "ffmpeg -i" in src.lower()
