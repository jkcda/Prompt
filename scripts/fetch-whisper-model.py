#!/usr/bin/env python3
"""下载 faster-whisper 的模型权重到 backend/vendor/whisper/<规格>/。

为什么需要这个脚本
------------------
模型权重单个 460MB（small），**会被远端 git 仓库的大文件限制直接拒掉**，
所以它不进版本库。但它是「本地转写」功能的必需品，而服务器常常不方便联网。

所以约定是：
  * 本地开发 → 跑一次这个脚本，模型落在 `backend/vendor/whisper/<规格>/`；
  * 打包到服务器 → 用 rsync / tar / docker COPY 时把 `backend/vendor/` 一起带上，
    服务器上不用再下载；
  * 纯 git 部署 → 在服务器上跑一次这个脚本。

用法：
    python scripts/fetch-whisper-model.py                # 默认 small
    python scripts/fetch-whisper-model.py --model base   # 4G 内存的服务器可以换 base
    python scripts/fetch-whisper-model.py --list         # 看各规格的体积与内存开销
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 各规格的仓库名与体积。体积是实测/官方值，用来给部署选型做参考。
# 内存是 int8 推理的**峰值**（含模型常驻），4G 的机器要照着这个选。
SPECS: dict[str, dict[str, object]] = {
    "tiny":   {"repo": "Systran/faster-whisper-tiny",   "disk": "75 MB",  "ram": "~250 MB"},
    "base":   {"repo": "Systran/faster-whisper-base",   "disk": "145 MB", "ram": "~350 MB"},
    "small":  {"repo": "Systran/faster-whisper-small",  "disk": "480 MB", "ram": "~1.2 GB"},
    "medium": {"repo": "Systran/faster-whisper-medium", "disk": "1.5 GB", "ram": "~3 GB"},
}

# faster-whisper 真正会读的文件。仓库里的 README / .gitattributes 不用下。
FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "backend" / "vendor" / "whisper"
# 国内直连 HuggingFace 很慢，默认走镜像；海外部署用 --endpoint 覆盖。
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dst: Path, retries: int = 3) -> None:
    """带进度地下载。先写 .part 再改名，中断了不会留下半截的正式文件。"""
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with part.open("wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done / total * 100
                            sys.stdout.write(
                                f"\r    {dst.name:<16} {pct:5.1f}%  "
                                f"{human(done)} / {human(total)}   "
                            )
                            sys.stdout.flush()
            sys.stdout.write("\n")
            if total and part.stat().st_size != total:
                raise OSError(f"下载不完整：{part.stat().st_size} != {total}")
            os.replace(part, dst)
            return
        except (urllib.error.URLError, OSError) as exc:
            sys.stdout.write("\n")
            print(f"    第 {attempt} 次失败：{exc}")
            if attempt == retries:
                raise
    # 走到这里说明重试用尽


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 faster-whisper 模型到 vendor 目录")
    ap.add_argument("--model", default="small", choices=sorted(SPECS), help="模型规格")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="目标根目录")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT,
                    help="下载源，默认国内镜像")
    ap.add_argument("--list", action="store_true", help="只列出各规格的体积与内存开销")
    args = ap.parse_args()

    if args.list:
        print(f"{'规格':<8} {'磁盘':>9} {'推理峰值内存':>14}   仓库")
        print("-" * 62)
        for name, info in SPECS.items():
            print(f"{name:<8} {info['disk']:>9} {info['ram']:>14}   {info['repo']}")
        print("\n4 核 4G 的服务器：base 稳，small 是准确率与内存的平衡点，medium 不要碰。")
        return 0

    spec = SPECS[args.model]
    repo = str(spec["repo"])
    out = Path(args.dest) / args.model

    if (out / "model.bin").is_file():
        size = (out / "model.bin").stat().st_size
        print(f"模型已存在，跳过：{out}（model.bin {human(size)}）")
        print("要重新下载就先删掉这个目录。")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    print(f"规格   : {args.model}（磁盘约 {spec['disk']}，推理峰值内存 {spec['ram']}）")
    print(f"来源   : {args.endpoint}")
    print(f"目标   : {out}")
    print()

    try:
        for name in FILES:
            url = f"{args.endpoint.rstrip('/')}/{repo}/resolve/main/{name}"
            download(url, out / name)
    except Exception as exc:  # noqa: BLE001
        print(f"\n下载失败：{exc}")
        print("可以换一个源重试，例如 --endpoint https://huggingface.co")
        print("或者在有网的机器上下好后，把整个 backend/vendor/whisper/ 目录拷过来。")
        return 1

    total = sum((out / n).stat().st_size for n in FILES)
    print(f"\n完成，共 {human(total)}。")
    print("打包时记得带上 backend/vendor/whisper/，服务器上就不用再下了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
