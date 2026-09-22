"""多模态模型客户端。

只做一件事：把「一段文本 + N 张帧」发给视觉模型，拿回文本。
支持两种协议：
  - `openai`    : OpenAI /v1/chat/completions（ModelScope、百炼、方舟、OpenRouter、
                  本地 vLLM 都兼容）—— 默认
  - `anthropic` : Anthropic /v1/messages

内建：并发限流、指数退避重试、超时、以及「请求体过大自动降帧重试」的保护。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from ..core.config import get_settings, resolve_ffmpeg
from .ffmpeg import run as ffmpeg_run

log = logging.getLogger("vlm")


class VLMError(RuntimeError):
    pass


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# OpenAI 协议的音频内容块只认这几种格式，其余要报错而不是硬塞
_AUDIO_FORMATS = {"wav", "mp3", "m4a", "flac", "ogg", "webm", "aac"}
_AUDIO_MAX_BYTES = 20 * 1024 * 1024   # 官方上限 25MB，留点余量


def _audio_block(path: Path) -> dict:
    """构造 OpenAI 的 input_audio 内容块。

    格式字段只接受官方白名单里的值；超出体积上限时直接抛错，
    让调用方降级——静默发一个必然失败的请求，比明确报错更难排查。
    """
    fmt = path.suffix.lstrip(".").lower()
    if fmt not in _AUDIO_FORMATS:
        raise ValueError(f"不支持的音频格式 {fmt!r}，可选：{sorted(_AUDIO_FORMATS)}")
    raw = path.read_bytes()
    if len(raw) > _AUDIO_MAX_BYTES:
        raise ValueError(
            f"音频片段 {len(raw) / 1048576:.1f}MB 超过 {_AUDIO_MAX_BYTES // 1048576}MB 上限"
        )
    return {"data": base64.b64encode(raw).decode("ascii"), "format": fmt}


class VLMClient:
    """带并发限流的视觉模型客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        concurrency: int | None = None,
        protocol: str = "openai",
    ) -> None:
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.vlm_api_key
        self.base_url = (base_url or s.vlm_base_url).rstrip("/")
        self.model = model or s.vlm_model
        self.timeout = timeout or s.vlm_timeout
        self.protocol = protocol
        self._sem = asyncio.Semaphore(max(1, concurrency or s.vlm_concurrency))

    # -- 基础可用性 --------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def require_configured(self) -> None:
        if not self.configured:
            raise VLMError(
                "未配置多模态模型。请在 backend/.env 中设置 VLM_API_KEY、VLM_BASE_URL、VLM_MODEL。"
            )

    # -- 请求体构造 --------------------------------------------------------

    def _openai_payload(
        self,
        system: str,
        user: str,
        images: list[Path],
        max_tokens: int,
        audio: list[Path] | None = None,
    ) -> dict:
        content: list[dict] = [{"type": "text", "text": user}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(img)}})
        for clip in audio or []:
            content.append({"type": "input_audio", "input_audio": _audio_block(clip)})
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        }

    def _anthropic_payload(
        self,
        system: str,
        user: str,
        images: list[Path],
        max_tokens: int,
        audio: list[Path] | None = None,
    ) -> dict:
        if audio:
            # Anthropic 的 messages 协议目前没有音频内容块。
            # 静默丢弃比报错好——音频只是增强，不该让整个任务失败。
            log.warning("Anthropic 协议不支持音频输入，已忽略 %d 个音频片段", len(audio))
        blocks: list[dict] = [{"type": "text", "text": user}]
        for img in images:
            mime = mimetypes.guess_type(img.name)[0] or "image/jpeg"
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": base64.b64encode(img.read_bytes()).decode("ascii"),
                },
            })
        return {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": blocks}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

    def _endpoint_and_headers(self) -> tuple[str, dict]:
        if self.protocol == "anthropic":
            base = self.base_url
            url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
            return url, {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        base = self.base_url
        url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        return url, {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    @staticmethod
    def _extract_text(payload: dict, protocol: str) -> str:
        if protocol == "anthropic":
            parts = payload.get("content") or []
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
        return ""

    @staticmethod
    def _diagnose_empty(payload: dict, n_images: int, n_audio: int = 0) -> str:
        """HTTP 200 但内容为空时，把原因说清楚。

        最常见的两种「空」：
          1. `choices: null` + `usage` 全为 0 —— 请求根本没被处理。
             典型原因是模型不支持该模态的输入，或该模型在当前账号下没有开通推理服务。
             这个特征很好认，但只报「模型返回空内容」会让人往网络、超时方向查。
          2. `finish_reason: content_filter` / `length` —— 被截断或拦截。

        报错必须可操作，否则等于没报。
        """
        choices = payload.get("choices")
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        model = payload.get("model") or "(未知)"

        if choices is None or (isinstance(choices, list) and not choices):
            if prompt_tokens == 0:
                hint = (
                    f"模型 {model} 返回了空 choices，且 usage 显示 0 tokens——"
                    "请求没有被真正处理。通常是：\n"
                    "  1) 该模型不支持本次请求用到的输入模态"
                    "（图片或音频——注意很多「多模态」模型只支持图片，不支持音频）；\n"
                    "  2) 该模型在当前账号下没有开通推理服务（服务未部署/未订阅）；\n"
                    "  3) 模型名拼写有误。\n"
                    "建议：换一个确认支持所需模态的模型，或先用 GET /api/health/vlm 自检。"
                )
                parts = []
                if n_images:
                    parts.append(f"{n_images} 张图片")
                if n_audio:
                    parts.append(f"{n_audio} 段音频")
                if parts:
                    hint += f"\n本次请求带了 {' 和 '.join(parts)}。"
                    if n_audio:
                        hint += (
                            "\n如果这个模型本来能读图，那问题很可能出在音频上——"
                            "试试关掉 VLM_AUDIO_INPUT。"
                        )
                return hint
            return f"模型 {model} 返回空 choices（usage: {usage}）"

        if isinstance(choices, list) and choices:
            reason = choices[0].get("finish_reason")
            if reason == "content_filter":
                return f"模型 {model} 的内容过滤拦截了本次请求（finish_reason=content_filter）"
            if reason == "length":
                return (
                    f"模型 {model} 的输出被 max_tokens 截断（finish_reason=length）。"
                    "调大 max_tokens 或减少输入帧数。"
                )
            if reason:
                return f"模型 {model} 返回空内容（finish_reason={reason}）"

        return f"模型 {model} 返回空内容"

    # -- 调用 --------------------------------------------------------------

    async def complete(
        self,
        system: str,
        user: str,
        images: list[Path] | None = None,
        max_tokens: int = 4096,
        retries: int = 3,
        audio: list[Path] | None = None,
    ) -> str:
        """单次调用。失败自动重试；请求体过大时自动减半图片重试。"""
        self.require_configured()
        images = list(images or [])
        audio = list(audio or [])

        url, headers = self._endpoint_and_headers()
        async with self._sem:
            return await self._complete_inner(
                url, headers, system, user, images, max_tokens, retries, audio
            )

    async def _complete_inner(
        self,
        url: str,
        headers: dict,
        system: str,
        user: str,
        images: list[Path],
        max_tokens: int,
        retries: int,
        audio: list[Path] | None = None,
    ) -> str:
        attempt = 0
        working = list(images)
        working_audio = list(audio or [])

        while True:
            attempt += 1
            payload = (
                self._anthropic_payload(system, user, working, max_tokens, working_audio)
                if self.protocol == "anthropic"
                else self._openai_payload(system, user, working, max_tokens, working_audio)
            )

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= retries:
                    raise VLMError(f"模型请求失败（网络）：{exc}") from exc
                wait = 2 ** attempt
                log.warning("网络异常，%ds 后重试 (%d/%d)：%s", wait, attempt, retries, exc)
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 200:
                data = resp.json()
                text = self._extract_text(data, self.protocol)
                if not text:
                    if attempt >= retries:
                        raise VLMError(self._diagnose_empty(data, len(working), len(working_audio)))
                    await asyncio.sleep(2 ** attempt)
                    continue
                return text

            body = resp.text[:600]

            # 音频不被支持时，先丢掉音频再试——音频只是增强，
            # 不该因为它一个人把整个任务打死。
            if working_audio and resp.status_code in (400, 413, 422) and (
                "audio" in body.lower()
                or "modality" in body.lower()
                or "unsupported" in body.lower()
            ):
                log.warning("模型不接受音频输入，丢弃 %d 个片段后重试：%s",
                            len(working_audio), body[:200])
                working_audio = []
                continue

            # 请求体过大 / 上下文超限 → 减帧重试
            if resp.status_code in (400, 413, 422) and working and (
                "too large" in body.lower()
                or "context" in body.lower()
                or "token" in body.lower()
                or "exceed" in body.lower()
            ):
                keep = max(1, len(working) // 2)
                log.warning("请求体超限（%s），帧数 %d → %d 重试", resp.status_code, len(working), keep)
                step = max(1, len(working) // keep)
                working = working[::step][:keep]
                if attempt >= retries + 2:
                    raise VLMError(f"模型拒绝请求：{resp.status_code} {body}")
                continue

            # 限流 / 服务端错误 → 退避重试
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= retries:
                    raise VLMError(f"模型服务不可用：{resp.status_code} {body}")
                wait = min(30, 2 ** attempt * 2)
                log.warning("HTTP %s，%ds 后重试 (%d/%d)", resp.status_code, wait, attempt, retries)
                await asyncio.sleep(wait)
                continue

            raise VLMError(f"模型请求被拒绝：HTTP {resp.status_code} {body}")

    async def complete_many(
        self,
        tasks: list[tuple[str, str, list[Path]]],
        max_tokens: int = 4096,
        audio_per_task: list[list[Path]] | None = None,
    ) -> list[str]:
        """并发执行多个 (system, user, images) 任务，返回等长结果列表。

        单个任务失败不会中断其他任务——失败位置返回错误标记文本。
        `audio_per_task` 与 tasks 等长时，给每个任务附带自己的音频片段（可选）。
        """
        audios = audio_per_task or [[] for _ in tasks]

        async def _one(
            idx: int, sys_p: str, usr_p: str, imgs: list[Path], aud: list[Path]
        ) -> tuple[int, str]:
            try:
                text = await self.complete(sys_p, usr_p, imgs, max_tokens=max_tokens, audio=aud)
                return idx, text
            except Exception as exc:  # noqa: BLE001
                log.error("Pass1 分块 %d 失败: %s", idx, exc)
                return idx, ""

        results = await asyncio.gather(
            *(_one(i, s, u, im, audios[i] if i < len(audios) else [])
              for i, (s, u, im) in enumerate(tasks))
        )
        out = [""] * len(tasks)
        for idx, text in results:
            out[idx] = text
        return out


# ---------------------------------------------------------------------------
# 连通性自检
# ---------------------------------------------------------------------------

async def healthcheck(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """连通性自检。**会真的发一张图**，所以通过就代表这个模型能读图。

    为什么要发图：只发文本的话，纯文本模型也能回「OK」，
    用户会以为配好了，真跑反推时才发现收到图片就报错。
    自检必须覆盖「图片输入」这个前提，否则等于没检。

    为什么用纯色图：这是对照测试的思路——给一张纯红图，
    回答里必须出现 red 才算真读到了。答案对不上说明模型在猜。
    """
    s = get_settings()
    info = {
        "configured": bool(api_key if api_key is not None else s.vlm_api_key),
        "base_url": (base_url or s.vlm_base_url),
        "model": (model or s.vlm_model),
        "ok": False,
        "vision": False,
        "inconclusive": False,
        "message": "",
    }
    if not info["configured"]:
        info["message"] = "未配置 VLM_API_KEY"
        return info

    probe = _vision_probe_image()
    if not probe:
        info["message"] = "无法生成自检用图片（ffmpeg 不可用）"
        return info

    client = VLMClient(
        api_key=api_key if api_key is not None else None,
        base_url=base_url if base_url is not None else None,
        model=model if model is not None else None,
    )
    try:
        text = await client.complete(
            "You are a connectivity probe for a vision model. Answer in one word.",
            "What is the dominant colour of this image? Answer with a single English colour word.",
            images=[probe],
            # 不能给太小：推理模型（DeepSeek-V4 / GLM 这类）会先输出一大段思考，
            # max_tokens=16 会被思考过程吃光，然后报 finish_reason=length，
            # 看起来像「模型不可用」，其实只是额度不够。
            max_tokens=512,
            retries=1,
        )
        info["ok"] = True
        info["message"] = text[:120]
        # 纯红图，答案里必须出现 red。答案跑偏说明它没真在看图。
        info["vision"] = "red" in text.lower()
        if not info["vision"]:
            info["message"] = (
                f"接口通了，但回答是 {text[:60]!r}，没有识别出图片内容。"
                "该模型可能只支持文本输入，或图片被忽略了。"
            )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        info["message"] = msg[:400]
        # 输出被截断 ≠ 不可用。推理模型的思考过程可能极长，
        # 这种情况只能判「无法确定」，不能判「不支持」。
        if "finish_reason=length" in msg:
            info["inconclusive"] = True
            info["message"] = (
                "无法判定：模型有输出但被 max_tokens 截断，可能是推理模型"
                "（思考过程很长）。请换一张图或在真实任务里试。"
            )
    return info


_PROBE_IMAGE: Path | None = None


def _vision_probe_image() -> Path | None:
    """生成（并缓存）一张纯红小图，用于验证模型真的能读图。"""
    global _PROBE_IMAGE
    if _PROBE_IMAGE and _PROBE_IMAGE.is_file():
        return _PROBE_IMAGE

    from ..core.config import TMP_DIR

    out = Path(TMP_DIR) / "vision_probe.jpg"
    try:
        ff = resolve_ffmpeg()
        if not ff:
            return None
        ffmpeg_run([
            ff, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:size=64x64:duration=1",
            "-frames:v", "1", "-y", str(out),
        ], timeout=30)
    except Exception as exc:  # noqa: BLE001
        log.warning("生成自检图片失败: %s", exc)
        return None

    if out.is_file() and out.stat().st_size > 0:
        _PROBE_IMAGE = out
        return out
    return None


async def list_models(
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """拉取服务商声明的模型列表。

    ⚠ 返回的列表**不代表账号真实可用范围**——很多服务商只返回精选列表，
    里面有些模型没有部署推理服务，调用会报 has no provider supported。
    所以列表只能当候选来源，必须逐个测试才能确定哪个真能用。
    """
    s = get_settings()
    base = (base_url or s.vlm_base_url).rstrip("/")
    key = api_key if api_key is not None else s.vlm_api_key
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"

    if not key:
        return {"ok": False, "models": [], "message": "未配置 VLM_API_KEY"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {key}"})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "models": [], "message": f"请求失败：{exc}"}

    if resp.status_code != 200:
        return {
            "ok": False,
            "models": [],
            "message": f"HTTP {resp.status_code}：{' '.join(resp.text[:200].split())}",
        }

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return {"ok": False, "models": [], "message": "返回的不是 JSON"}

    raw = data.get("data") if isinstance(data, dict) else data
    ids: list[str] = []
    for item in raw or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)

    return {"ok": True, "models": sorted(ids), "message": f"共 {len(ids)} 个"}
