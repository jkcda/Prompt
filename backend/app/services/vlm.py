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

from ..core.config import get_settings

log = logging.getLogger("vlm")


class VLMError(RuntimeError):
    pass


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


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

    def _openai_payload(self, system: str, user: str, images: list[Path], max_tokens: int) -> dict:
        content: list[dict] = [{"type": "text", "text": user}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(img)}})
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

    def _anthropic_payload(self, system: str, user: str, images: list[Path], max_tokens: int) -> dict:
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
    def _diagnose_empty(payload: dict, n_images: int) -> str:
        """HTTP 200 但内容为空时，把原因说清楚。

        最常见的两种「空」：
          1. `choices: null` + `usage` 全为 0 —— 请求根本没被处理。
             典型原因是模型不支持图片输入，或该模型在当前账号下没有开通推理服务。
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
                    "  1) 该模型不支持图片输入（不是视觉语言模型）；\n"
                    "  2) 该模型在当前账号下没有开通推理服务（服务未部署/未订阅）；\n"
                    "  3) 模型名拼写有误。\n"
                    "建议：换一个确认支持图片输入的 VL 模型，或先用 GET /api/health/vlm 自检。"
                )
                if n_images:
                    hint += f"\n本次请求带了 {n_images} 张图片。"
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
    ) -> str:
        """单次调用。失败自动重试；请求体过大时自动减半图片重试。"""
        self.require_configured()
        images = list(images or [])

        url, headers = self._endpoint_and_headers()
        async with self._sem:
            return await self._complete_inner(url, headers, system, user, images, max_tokens, retries)

    async def _complete_inner(
        self,
        url: str,
        headers: dict,
        system: str,
        user: str,
        images: list[Path],
        max_tokens: int,
        retries: int,
    ) -> str:
        attempt = 0
        working = list(images)

        while True:
            attempt += 1
            payload = (
                self._anthropic_payload(system, user, working, max_tokens)
                if self.protocol == "anthropic"
                else self._openai_payload(system, user, working, max_tokens)
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
                        raise VLMError(self._diagnose_empty(data, len(working)))
                    await asyncio.sleep(2 ** attempt)
                    continue
                return text

            body = resp.text[:600]

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
    ) -> list[str]:
        """并发执行多个 (system, user, images) 任务，返回等长结果列表。

        单个任务失败不会中断其他任务——失败位置返回错误标记文本。
        """
        async def _one(idx: int, sys_p: str, usr_p: str, imgs: list[Path]) -> tuple[int, str]:
            try:
                text = await self.complete(sys_p, usr_p, imgs, max_tokens=max_tokens)
                return idx, text
            except Exception as exc:  # noqa: BLE001
                log.error("Pass1 分块 %d 失败: %s", idx, exc)
                return idx, ""

        results = await asyncio.gather(
            *(_one(i, s, u, im) for i, (s, u, im) in enumerate(tasks))
        )
        out = [""] * len(tasks)
        for idx, text in results:
            out[idx] = text
        return out


# ---------------------------------------------------------------------------
# 连通性自检
# ---------------------------------------------------------------------------

async def healthcheck() -> dict:
    s = get_settings()
    info = {
        "configured": bool(s.vlm_api_key),
        "base_url": s.vlm_base_url,
        "model": s.vlm_model,
        "ok": False,
        "message": "",
    }
    if not s.vlm_api_key:
        info["message"] = "未配置 VLM_API_KEY"
        return info

    client = VLMClient()
    try:
        text = await client.complete(
            "You are a connectivity probe. Reply with exactly: OK",
            "Reply with exactly: OK",
            max_tokens=16,
            retries=1,
        )
        info["ok"] = True
        info["message"] = text[:80]
    except Exception as exc:  # noqa: BLE001
        info["message"] = str(exc)[:300]
    return info
