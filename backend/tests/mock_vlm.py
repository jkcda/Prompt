"""最小 OpenAI 兼容 mock 服务，用于端到端测试反推管线。

为什么需要它：
    真实模型调用有成本、有延迟、还依赖外部可用性。但管线里最容易出错的恰恰是
    调用之外的部分——抽帧、时间戳对齐、Pass1 JSON 解析、多块合并、Pass2 组装。
    mock 掉模型这一层，就能在 CI 里把整条链路完整跑一遍。

判定逻辑：
    - 请求体里带 image_url → 视为 Pass1（视觉观察），返回结构化镜头 JSON
    - 不带图片            → 视为 Pass2（成文），返回一段目标格式提示词

可以单独运行：
    python -m tests.mock_vlm 8899
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI, Request

app = FastAPI(title="Mock VLM")

# 记录收到的请求，便于测试断言（例如确认帧数、确认 Pass2 只调用一次）
CALLS: list[dict[str, Any]] = []


def reset() -> None:
    CALLS.clear()


def _count_images(payload: dict) -> int:
    n = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def _extract_timestamps(text: str) -> list[str]:
    return re.findall(r"Image \d+ -> timestamp ([0-9.]+)s", text)


def _make_pass1_response(payload: dict, user_text: str) -> str:
    """根据请求里实际给的时间戳生成对应的镜头列表——这样测试能验证对齐是否正确。"""
    stamps = _extract_timestamps(user_text)
    if not stamps:
        stamps = ["0.000"]

    shots = []
    for i, ts in enumerate(stamps):
        secs = float(ts)
        m, s = divmod(secs, 60)
        # 时间区间：用下一个时间戳当本条的结束。最后一条故意给一个**偏小**的值，
        # 用来验证代码兜底会把末镜的 end_ms 拉齐到片段总时长。
        start_ms = int(round(secs * 1000))
        if i + 1 < len(stamps):
            end_ms = int(round(float(stamps[i + 1]) * 1000))
        else:
            end_ms = start_ms + 200
        shots.append({
            "shot": str(i + 1),
            "timecode": f"{int(m):02d}:{s:06.3f}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "shot_size": "medium close-up" if i % 2 == 0 else "wide",
            "camera": "slow dolly-in, small amplitude" if i % 3 else "static",
            "subject": f"a performer in frame at {ts}s, wearing a dark jacket",
            "action": (
                "she turns toward the lens, her weight shifting onto the front foot "
                "and the motion carrying up through her shoulders"
            ),
            "setting": "an industrial rooftop at dusk, vents and cables in the background",
            "lighting": "low warm sun from camera left, soft falloff",
            "color": "teal and amber, slightly desaturated",
            "motion_energy": "medium, following a steady beat",
            "on_screen_text": "none",
            "dialogue": "still here" if i == 0 else "",
            "sfx": "cloth movement",
            "transition": "cut",
            "confidence": 0.86,
        })

    return json.dumps({
        "shots": shots,
        "subjects": [
            {
                "label": "performer",
                "kind": "person",
                "description": "a performer in a dark jacket, dark hair tied back",
                "shots": [str(i + 1) for i in range(len(shots))],
                "notes": "the dark jacket and the tied-back hair must not change between shots",
            },
            {
                "label": "rooftop",
                "kind": "environment",
                "description": "an industrial rooftop at dusk with vents and cables",
                "shots": [str(i + 1) for i in range(len(shots))],
                "notes": "the vent positions and the low warm sun direction stay fixed",
            },
        ],
        "global_notes": f"mock 观察：共 {len(shots)} 个镜头，手持轻微晃动，暖色调统一。",
    }, ensure_ascii=False)


def _extract_registry(user_text: str) -> list[dict[str, str]]:
    """从 Pass2 的用户消息里抽出主体登记表。

    mock 按真实登记表生成 <Subject N> 与 retention_analysis 行，
    这样测试才能验证「登记表确实送到了 Pass2」，而不是只看字段名在不在。
    """
    block = ""
    if "=== SUBJECT REGISTRY" in user_text:
        block = user_text.split("=== SUBJECT REGISTRY", 1)[1].split("===", 1)[1]
    entries: list[dict[str, str]] = []
    for line in block.splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.+?)\s+\(([^)]*)\)\s+-\s+(.+)$", line)
        if m:
            entries.append({
                "index": m.group(1),
                "label": m.group(2).strip(),
                "kind": m.group(3).strip(),
                "shots": m.group(4).strip(),
            })
    return entries


def _make_pass2_response(payload: dict, user_text: str) -> str:
    """按 Pass2 提示里要求的字段名，返回一段结构合法的提示词。"""
    system = ""
    for msg in payload.get("messages") or []:
        if msg.get("role") == "system":
            system = msg.get("content") or ""
            break

    if "subject_definitions:" in system:
        registry = _extract_registry(user_text) or [
            {"index": "1", "label": "performer", "kind": "person", "shots": "[Shot 1]"},
        ]
        definitions = "\n".join(
            f"<Subject {e['index']}> is the {e['label']} ({e['kind']}) from the source video."
            for e in registry
        )
        retention = "\n".join(
            f"<Subject {e['index']}> (appears in {e['shots']}): fully_preserved - "
            f"the {e['label']} is carried over unchanged."
            for e in registry
        )
        return (
            "subject_definitions:\n"
            f"{definitions}\n"
            "<Video 1> is the source video for the target edit.\n\n"
            "summary:\n"
            "[video continuation + reference generation] The target video continues "
            "<Subject 1>'s rooftop performance using the pacing of <Video 1>.\n\n"
            "retention_analysis:\n"
            f"{retention}\n"
            "<Video 1> (cut and pacing structure): weak_reference - the edit follows the "
            "original rhythm.\n\n"
            "detailed_description:\n"
            "The target video is a cinematic rooftop performance with teal and amber grading.\n"
            "[Shot 1] The shot opens on a medium close-up of <Subject 1>, who turns toward "
            "the lens, her weight shifting onto her front foot and the motion carrying up "
            "through her shoulders. She sings, her lips moving through every syllable of "
            "the line.\n"
            "[Shot 2] At 00:03.400, the shot cuts to a wide view of <Subject 2>.\n\n"
            "overall_soundscape:\n"
            "Low rooftop wind and light cloth movement continue throughout.\n\n"
            "non_diegetic_music:\n"
            "A restrained synth pad at a slow tempo with no swell."
        )

    if "Seedance" in system:
        return (
            "一位身穿深色夹克的表演者站在黄昏的工业天台边缘，转身面向镜头，重心前移，"
            "动作顺着肩膀向上传导。背景有通风管道与线缆，暖色侧光从画面左侧打来。"
            "电影级写实风格，青橙色调，略微去饱和。以中近景开场，随后缓慢推轨至特写，浅景深。"
            "全片两个镜头，节拍为 0–3.4 秒、3.4–6.8 秒。"
            "天台风声与衣料摩擦声；无对白；末尾一段缓慢的合成器铺底配乐。"
            "不要出现字幕、文字、水印。"
        )

    if "【整体风格】" in system:
        return (
            "【整体风格】\n"
            "写实电影风格，青橙色调，黄昏暖侧光，节奏中速。\n\n"
            "【镜头分镜】\n"
            "镜头 1｜00:00.000–00:03.400\n"
            "  景别 / 角度：中近景，平视\n"
            "  运镜：缓慢推轨\n"
            "  画面内容：表演者转身面向镜头，重心前移，动作传导至肩膀。\n"
            "  台词 / 人声：still here\n"
            "  音效：衣料摩擦\n"
            "  转场：切\n\n"
            "【声音设计】天台风声持续；无对白段落以配乐铺底。\n\n"
            "【负面提示词】字幕, 文字, 水印, 平台 logo"
        )

    # 默认：H3 T2VA 三字段
    return (
        "integrated_multimodal_description: "
        "The target video is a cinematic rooftop performance with teal and amber grading "
        "and a slightly desaturated palette. "
        "[Shot 1] The shot opens on a medium close-up of a performer in a dark jacket who "
        "turns toward the lens, her weight shifting onto her front foot and the motion "
        "carrying up through her shoulders. She sings, her lips moving through every "
        "syllable of the line. "
        "[Shot 2] At 00:03.400, the shot cuts to a wide view of the industrial rooftop, "
        "vents and cables silhouetted against a low warm sun.\n\n"
        "overall_soundscape: Low rooftop wind and light cloth movement continue throughout.\n\n"
        "non_diegetic_music: A restrained synth pad at a slow tempo with no swell."
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> dict:
    payload = await request.json()

    user_text = ""
    for msg in payload.get("messages") or []:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_text += content
            elif isinstance(content, list):
                user_text += "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )

    n_images = _count_images(payload)
    system_text = ""
    for msg in payload.get("messages") or []:
        if msg.get("role") == "system":
            system_text = msg.get("content") or ""
            break

    # 自由发挥模式也带图，但它直接出提示词，不出 Pass1 的 JSON。
    # 靠系统提示词里那句「镜头数由你自己判断」识别 —— 那是该模式独有的。
    is_freeform = "professional storyboard artist" in system_text
    is_pass1 = n_images > 0 and not is_freeform

    CALLS.append({
        "images": n_images,
        # pass 的语义保持不变（1 = 带图调用，2 = 不带图调用），老断言继续有效；
        # 用 kind 区分「这次带图到底是要 JSON 还是要提示词」。
        "pass": 1 if n_images else 2,
        "kind": "freeform" if is_freeform else ("pass1" if is_pass1 else "pass2"),
        "model": payload.get("model"),
        "chars": len(user_text),
    })

    text = (
        _make_pass1_response(payload, user_text)
        if is_pass1
        else _make_pass2_response(payload, user_text)
    )

    return {
        "id": "mock-1",
        "object": "chat.completion",
        "model": payload.get("model") or "mock",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": n_images * 1100, "completion_tokens": 400, "total_tokens": 0},
    }


@app.get("/v1/models")
async def models() -> dict:
    return {"data": [{"id": "mock-vlm", "object": "model"}]}


def serve(port: int = 8899) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    import sys

    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8899)
