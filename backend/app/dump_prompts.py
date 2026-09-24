"""把管线实际用到的提示词导出成文本，方便查看和调参。

用法：
    python -m app.dump_prompts [输出目录]

导出内容：
    system_<格式>.txt  各目标格式的系统提示词（身份 + 任务 + 格式规则）
    user.txt           用户消息样例（帧时间戳清单 + 音频 + 用户说明）
    payload.json       实际发出去的 HTTP body 骨架（图片用占位符替换）

为什么要做成命令：提示词是这个项目的核心资产，改一个字都会影响产出质量。
把它导出来对照着读，比在 Python 字符串里翻要快得多。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .core.config import get_settings, refresh_settings
from .schemas import MediaInfo
from .services import templates
from .services.vlm import VLMClient


def _user_sample() -> str:
    """用真实参数造一条用户消息 —— 帧列表 + 音频 + 用户画面说明。"""
    return templates.build_user(
        chunk_start=0.0,
        chunk_end=12.0,
        frame_marks=[
            (0.0, "head"), (0.5, "mid"), (1.0, "mid"), (1.5, "mid"),
            (2.0, "mid"), (2.5, "mid"), (3.0, "tail"),
            (3.5, "head"), (4.0, "mid"), (4.5, "tail"),
        ],
        audio_text="【音频】无转写（未配置 ASR）。音频内容对你完全未知。",
        media=MediaInfo(path="demo.mp4", duration=12.0, width=1920, height=1080, fps=30.0),
        chunk_index=0,
        chunk_total=1,
        content_hint="一镜到底的跟拍运镜，全程没有切镜；主角是白发少女，穿黑色风衣。",
        closing=templates.build_closing(),
    )


def _payload_skeleton() -> dict:
    """实际发出去的 body 骨架 —— 图片用占位符替换，不然文件太大。"""
    refresh_settings()
    client = VLMClient()
    payload = client._openai_payload(
        "<系统提示词见 system_h3.txt>",
        "<用户消息见 user.txt>",
        images=[],  # 不塞真图，只保留结构
        max_tokens=get_settings().vlm_max_tokens,
    )
    payload["messages"][1]["content"] = [
        {"type": "text", "text": "<用户消息文本>"},
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64,<每帧一张，约 75KB → base64 后约 100KB>"}},
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64,<第二帧...>"}},
    ]
    return payload


def dump(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(name: str, text: str) -> None:
        p = out_dir / name
        p.write_text(text, encoding="utf-8")
        written.append(p)

    for fmt in ("h3", "h3-ref", "seedance", "generic"):
        write(f"system_{fmt}.txt", templates.build_system(fmt, "en"))

    write("user.txt", _user_sample())
    write("payload.json", json.dumps(_payload_skeleton(), ensure_ascii=False, indent=2))
    return written


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("prompts-dump")
    files = dump(out)
    print(f"已导出 {len(files)} 个文件到 {out.resolve()}")
    for f in files:
        print(f"  {f.name:26s} {len(f.read_text(encoding='utf-8')):>7d} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
