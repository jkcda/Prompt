"""模型客户端（VLMClient）的单元测试。

只测不联网的部分：请求体构造、响应解析、空响应的原因诊断。
真实调用属于端到端范畴，不放进来。
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from app.services.vlm import VLMClient, VLMError


@pytest.fixture
def jpeg(tmp_path: Path) -> Path:
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9")
    return p


# ---------------------------------------------------------------------------
# 响应解析
# ---------------------------------------------------------------------------

def test_extract_text_openai_string_content():
    payload = {"choices": [{"message": {"content": "  hello  "}}]}
    assert VLMClient._extract_text(payload, "openai") == "hello"


def test_extract_text_openai_list_content():
    payload = {"choices": [{"message": {"content": [
        {"type": "text", "text": "part one "},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "part two"},
    ]}}]}
    assert VLMClient._extract_text(payload, "openai") == "part one part two"


def test_extract_text_anthropic():
    payload = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert VLMClient._extract_text(payload, "anthropic") == "ab"


def test_extract_text_handles_null_choices():
    """有些服务对不支持的请求返回 choices: null 而不是报错。"""
    assert VLMClient._extract_text({"choices": None}, "openai") == ""


def test_extract_text_handles_missing_message():
    assert VLMClient._extract_text({"choices": [{}]}, "openai") == ""


# ---------------------------------------------------------------------------
# 空响应诊断
# ---------------------------------------------------------------------------

def test_diagnose_null_choices_zero_tokens():
    """这个特征组合 = 请求根本没被处理，必须说清楚，不能只报「返回空内容」。"""
    payload = {
        "model": "Vendor/Some-VL-Model",
        "choices": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    msg = VLMClient._diagnose_empty(payload, n_images=3)
    assert "Vendor/Some-VL-Model" in msg
    assert "不支持本次请求用到的输入模态" in msg
    assert "没有开通推理服务" in msg
    assert "3 张图片" in msg
    assert "/api/health/vlm" in msg, "要给出下一步动作"


def test_diagnose_null_choices_without_images():
    payload = {"model": "m", "choices": None, "usage": {"prompt_tokens": 0}}
    msg = VLMClient._diagnose_empty(payload, n_images=0)
    assert "张图片" not in msg


def test_diagnose_content_filter():
    payload = {"model": "m", "choices": [{"finish_reason": "content_filter"}]}
    assert "内容过滤" in VLMClient._diagnose_empty(payload, 1)


def test_diagnose_length_truncation():
    payload = {"model": "m", "choices": [{"finish_reason": "length"}]}
    msg = VLMClient._diagnose_empty(payload, 1)
    assert "max_tokens" in msg
    assert "截断" in msg


def test_diagnose_empty_choices_list_with_tokens_used():
    """有 token 消耗但没内容，说明确实调用了模型，不能误导成「没被处理」。"""
    payload = {"model": "m", "choices": [], "usage": {"prompt_tokens": 900}}
    msg = VLMClient._diagnose_empty(payload, 2)
    assert "没有被真正处理" not in msg
    assert "900" in msg


def test_diagnose_unknown_finish_reason():
    payload = {"model": "m", "choices": [{"finish_reason": "stop"}]}
    assert "stop" in VLMClient._diagnose_empty(payload, 1)


def test_diagnose_no_choices_key_at_all():
    msg = VLMClient._diagnose_empty({"model": "m"}, 0)
    assert "m" in msg


# ---------------------------------------------------------------------------
# 请求体构造
# ---------------------------------------------------------------------------

def test_openai_payload_embeds_images_as_data_uri(jpeg: Path):
    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    payload = client._openai_payload("sys", "usr", [jpeg], 100)

    assert payload["model"] == "m"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    content = payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "usr"}
    assert content[1]["type"] == "image_url"
    uri = content[1]["image_url"]["url"]
    assert uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == jpeg.read_bytes()
    assert payload["stream"] is False


def test_openai_payload_without_images_has_only_text(jpeg: Path):
    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    content = client._openai_payload("sys", "usr", [], 100)["messages"][1]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


# ---------------------------------------------------------------------------
# 音频输入
# ---------------------------------------------------------------------------

def test_openai_payload_embeds_audio_as_input_audio(tmp_path: Path):
    """OpenAI 协议的音频块是 input_audio，不是 image_url。"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 2048)

    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    payload = client._openai_payload("sys", "usr", [], 100, audio=[wav])
    content = payload["messages"][1]["content"]

    assert content[0]["type"] == "text"
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    assert base64.b64decode(content[1]["input_audio"]["data"]).startswith(b"RIFF")


def test_audio_and_image_can_coexist(tmp_path: Path, jpeg: Path):
    wav = tmp_path / "a.mp3"
    wav.write_bytes(b"ID3" + b"\x00" * 1024)

    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    content = client._openai_payload("s", "u", [jpeg], 100, audio=[wav])["messages"][1]["content"]
    types = [c["type"] for c in content]
    assert types == ["text", "image_url", "input_audio"]


def test_audio_block_rejects_unknown_format(tmp_path: Path):
    """格式白名单之外要明确报错，而不是发一个必然失败的请求。"""
    from app.services.vlm import _audio_block

    bad = tmp_path / "a.txt"
    bad.write_bytes(b"hello")
    with pytest.raises(ValueError, match="不支持的音频格式"):
        _audio_block(bad)


def test_audio_block_rejects_oversized_clip(tmp_path: Path):
    from app.services.vlm import _AUDIO_MAX_BYTES, _audio_block

    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * (_AUDIO_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="超过"):
        _audio_block(big)


def test_anthropic_payload_drops_audio_without_failing(tmp_path: Path, jpeg: Path):
    """Anthropic 协议没有音频块，应该丢掉音频继续跑，而不是让整个任务失败。"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 1024)

    client = VLMClient(api_key="k", base_url="http://x/v1", model="m", protocol="anthropic")
    payload = client._anthropic_payload("s", "u", [jpeg], 100, audio=[wav])
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image"]


def test_diagnose_mentions_audio_when_audio_attached():
    """带音频却空响应时，要提示可能是音频不被支持，而不是笼统说图片问题。"""
    payload = {"model": "m", "choices": None, "usage": {"prompt_tokens": 0}}
    msg = VLMClient._diagnose_empty(payload, n_images=3, n_audio=1)
    assert "1 段音频" in msg
    assert "VLM_AUDIO_INPUT" in msg, "要给出下一步动作"


def test_diagnose_omits_audio_hint_without_audio():
    """没带音频时不要出现音频相关的排查提示（会误导排查方向）。"""
    payload = {"model": "m", "choices": None, "usage": {"prompt_tokens": 0}}
    msg = VLMClient._diagnose_empty(payload, n_images=3, n_audio=0)
    assert "段音频" not in msg
    assert "VLM_AUDIO_INPUT" not in msg


def test_diagnose_says_multimodal_may_be_image_only():
    """「多模态」不等于支持音频，报错里要点明这个常见误解。"""
    payload = {"model": "m", "choices": None, "usage": {"prompt_tokens": 0}}
    msg = VLMClient._diagnose_empty(payload, n_images=1, n_audio=0)
    assert "只支持图片" in msg


# ---------------------------------------------------------------------------
# 音频不被支持时自动降级
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """第一次带音频返回空 choices，第二次不带音频返回正常内容。"""

    calls: list[dict] = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):  # noqa: A002
        content = json["messages"][1]["content"]
        has_audio = any(c.get("type") == "input_audio" for c in content)
        _FakeAsyncClient.calls.append({"has_audio": has_audio})
        if has_audio:
            # 这就是实测中 DeepSeek-V4.1-Flash 的行为：
            # HTTP 200 + choices: null + usage 全 0，不报错但请求没被处理
            return _FakeResponse(
                {"model": "m", "choices": None,
                 "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            )
        return _FakeResponse({
            "model": "m",
            "choices": [{"message": {"content": "described fine"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900},
        })


def test_audio_dropped_and_retried_on_empty_response(monkeypatch, jpeg: Path, tmp_path: Path):
    """模型不认音频时，要自动丢掉音频重试，而不是让整个任务失败。

    实测：DeepSeek-V4.1-Flash 单发图片正常，图片+音频则整个请求失效
    （HTTP 200 + choices: null）。如果这里直接报错，用户开着
    VLM_AUDIO_INPUT 就会每次反推都失败，而且看不出原因。
    """
    import httpx

    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 2048)

    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    text = asyncio.run(client.complete("sys", "usr", images=[jpeg], audio=[wav]))

    assert text == "described fine", "应该丢掉音频后成功，而不是失败"
    assert len(_FakeAsyncClient.calls) == 2
    assert _FakeAsyncClient.calls[0]["has_audio"] is True
    assert _FakeAsyncClient.calls[1]["has_audio"] is False, "重试时必须不带音频"


def test_no_audio_means_no_retry(monkeypatch, jpeg: Path):
    """本来就没带音频时，空响应要正常报错，不能假装能重试。"""
    import httpx

    class _AlwaysEmpty(_FakeAsyncClient):
        async def post(self, url, headers=None, json=None):  # noqa: A002
            return _FakeResponse(
                {"model": "m", "choices": None, "usage": {"prompt_tokens": 0}}
            )

    monkeypatch.setattr(httpx, "AsyncClient", _AlwaysEmpty)
    client = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    with pytest.raises(VLMError):
        asyncio.run(client.complete("sys", "usr", images=[jpeg], retries=1))


def test_audio_unsupported_is_remembered_across_clients(monkeypatch, jpeg: Path, tmp_path: Path):
    """「这个模型不认音频」要跨任务记住，否则每个任务都白跑一次。

    Pass1 的分块是并发发出的，同一批会同时撞墙；而 VLMClient 每个任务
    新建一次，只记在实例上等于每个任务都要重新踩一遍。
    """
    import httpx

    from app.services import vlm as vlm_mod

    vlm_mod._AUDIO_UNSUPPORTED_MODELS.clear()
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 2048)

    # 第一个客户端：带音频 → 空响应 → 丢音频重试成功
    c1 = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    assert asyncio.run(c1.complete("s", "u", images=[jpeg], audio=[wav])) == "described fine"
    assert [c["has_audio"] for c in _FakeAsyncClient.calls] == [True, False]

    # 第二个客户端（模拟下一个任务）：不该再试音频
    _FakeAsyncClient.calls = []
    c2 = VLMClient(api_key="k", base_url="http://x/v1", model="m")
    assert asyncio.run(c2.complete("s", "u", images=[jpeg], audio=[wav])) == "described fine"
    assert [c["has_audio"] for c in _FakeAsyncClient.calls] == [False], "又白跑了一次带音频的请求"

    # 不同模型不受影响
    assert vlm_mod.audio_supported("other-model") is True
    vlm_mod._AUDIO_UNSUPPORTED_MODELS.clear()


def test_anthropic_payload_uses_image_blocks(jpeg: Path):
    client = VLMClient(api_key="k", base_url="http://x/v1", model="m", protocol="anthropic")
    payload = client._anthropic_payload("sys", "usr", [jpeg], 100)
    blocks = payload["messages"][0]["content"]
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["media_type"] == "image/jpeg"
    assert blocks[1]["source"]["type"] == "base64"
    assert payload["system"] == "sys"


# ---------------------------------------------------------------------------
# 端点构造
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base,expected", [
    ("https://api-inference.modelscope.cn/v1", "https://api-inference.modelscope.cn/v1/chat/completions"),
    ("https://api-inference.modelscope.cn", "https://api-inference.modelscope.cn/v1/chat/completions"),
    ("http://127.0.0.1:8899/v1/", "http://127.0.0.1:8899/v1/chat/completions"),
])
def test_endpoint_normalizes_v1_suffix(base: str, expected: str):
    """用户填 base_url 时带不带 /v1 都要能work，不能让他猜。"""
    client = VLMClient(api_key="k", base_url=base, model="m")
    url, headers = client._endpoint_and_headers()
    assert url == expected
    assert headers["Authorization"] == "Bearer k"


def test_endpoint_anthropic():
    client = VLMClient(api_key="k", base_url="https://api.anthropic.com/v1", model="m",
                       protocol="anthropic")
    url, headers = client._endpoint_and_headers()
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "k"
    assert "anthropic-version" in headers


def test_require_configured_raises_actionable_error():
    from app.services.vlm import VLMError

    client = VLMClient(api_key="", base_url="http://x/v1", model="")
    with pytest.raises(VLMError) as exc:
        client.require_configured()
    assert "VLM_API_KEY" in str(exc.value)
    assert "backend/.env" in str(exc.value)
