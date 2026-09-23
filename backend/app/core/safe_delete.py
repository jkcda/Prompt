"""安全删除：删除动作**绝不抛异常**。

为什么需要这个模块 —— 踩过两次，代价都是任务白跑或服务挂掉：

某些运行环境（本机实测是 `sitecustomize.py` 注入的删除保护）会拦截
`os.remove` / `shutil.rmtree`，在「本轮删除次数超过阈值」时抛 `SystemExit`。
`SystemExit` 继承的是 **`BaseException` 而不是 `Exception`**，所以：

- `except OSError` 拦不住
- `except Exception` 拦不住
- 一路穿到线程池 / uvicorn，把整个进程带走

实测两次具体的翻车：

1. 临时目录清理把 uvicorn 干掉了（`cleanup()` 只捕获 `OSError`）。
2. B站视频下载后 yt-dlp 要 `os.remove` 掉合并前的原始音视频流，
   一删就抛 `SystemExit: 1`，任务报失败 —— 而**视频其实早就下好、也合并完了**。

结论：**删除是收尾动作，失败了最多留几个临时文件，绝不该让任务或服务挂掉。**
所有删除都必须走这里。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def safe_delete(path: str | Path, *, quiet: bool = True) -> bool:
    """删除文件或目录，返回是否删掉了。**任何情况下都不抛异常。**

    `quiet=False` 时会打 warning，用于「删不掉就该被看见」的场景。
    """
    p = Path(path)
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink(missing_ok=True)
        else:
            return True
        return not p.exists()
    except BaseException as exc:  # noqa: BLE001
        # 刻意捕获 BaseException：删除保护抛的是 SystemExit，
        # 用 except Exception 会让它穿出去把进程带走。
        if quiet:
            log.debug("删除 %s 失败（已忽略）：%s", p, exc)
        else:
            log.warning("删除 %s 失败（已忽略）：%s", p, exc)
        return False
