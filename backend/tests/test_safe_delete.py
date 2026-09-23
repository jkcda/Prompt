"""安全删除与「删除保护」环境的兼容性测试。

本机运行环境在 `os.remove` / `shutil.rmtree` 上装了删除保护，
超过阈值会抛 `SystemExit`（继承 BaseException，`except Exception` 拦不住）。
踩过两次：临时目录清理干掉 uvicorn、yt-dlp 删中间文件把下载任务带走。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.core.safe_delete import safe_delete


def test_safe_delete_removes_file(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert safe_delete(f) is True
    assert not f.exists()


def test_safe_delete_removes_dir(tmp_path: Path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x")
    assert safe_delete(d) is True
    assert not d.exists()


def test_safe_delete_missing_path_is_ok(tmp_path: Path):
    """删不存在的路径算成功（幂等）。"""
    assert safe_delete(tmp_path / "nope.txt") is True


def test_safe_delete_survives_systemexit(monkeypatch, tmp_path: Path):
    """删除保护抛 SystemExit 时不能穿出去。

    这是本机的真实行为：sitecustomize.py 拦截 os.remove，
    累计删除超阈值就 raise SystemExit(1)。
    """
    f = tmp_path / "a.txt"
    f.write_text("x")

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(Path, "unlink", boom)
    assert safe_delete(f) is False, "应该返回 False 而不是抛异常"


def test_safe_delete_survives_rmtree_systemexit(monkeypatch, tmp_path: Path):
    d = tmp_path / "sub"
    d.mkdir()

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(shutil, "rmtree", boom)
    assert safe_delete(d) is False


def test_safe_delete_survives_keyboardinterrupt(monkeypatch, tmp_path: Path):
    """连 KeyboardInterrupt 也不该从删除动作里漏出去。"""
    f = tmp_path / "a.txt"
    f.write_text("x")

    def boom(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "unlink", boom)
    assert safe_delete(f) is False


def test_ffmpeg_cleanup_delegates_to_safe_delete(monkeypatch, tmp_path: Path):
    """ffmpeg.cleanup 必须走安全删除，不能自己 try/except OSError。"""
    import ast
    import inspect

    from app.services import ffmpeg as ff

    tree = ast.parse(inspect.getsource(ff.cleanup))
    # 只看代码，不看 docstring（注释里提到 except OSError 是解释用的）
    code = "\n".join(
        line for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        for line in [node.func.id]
    )
    assert "safe_delete" in code, "cleanup 应该委托给 safe_delete"

    handlers = [
        h.type for node in ast.walk(tree) if isinstance(node, ast.Try)
        for h in node.handlers
    ]
    names = {
        h.id for t in handlers if t is not None
        for h in ast.walk(t) if isinstance(h, ast.Name)
    }
    assert "OSError" not in names, "cleanup 不该自己捕获 OSError —— 挡不住 SystemExit"

    d = tmp_path / "tmpdir"
    d.mkdir()
    ff.cleanup(d)  # 不该抛
    assert not d.exists()


def test_cleanup_survives_delete_guard(monkeypatch, tmp_path: Path):
    """清理失败只记日志，绝不把异常抛给调用方。"""
    from app.services import ffmpeg as ff

    def boom(*a, **kw):
        raise SystemExit(1)

    monkeypatch.setattr(shutil, "rmtree", boom)
    d = tmp_path / "x"
    d.mkdir()
    ff.cleanup(d)  # 不该抛


@pytest.mark.parametrize("mod", ["app.routers.upload", "app.services.downloader"])
def test_modules_do_not_call_bare_unlink(mod: str):
    """回归：这些模块里不该再出现裸的 .unlink()。

    裸 unlink 会被删除保护打成 SystemExit，把请求或任务带走。
    """
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(mod))
    assert ".unlink(" not in src, f"{mod} 里还有裸 unlink，应改用 safe_delete"
