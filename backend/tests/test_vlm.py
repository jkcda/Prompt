"""模型客户端（VLMClient）的单元测试。

只测不联网的部分：请求体构造、响应解析、空响应的原因诊断。
真实调用属于端到端范畴，不放进来。
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.services.vlm import VLMClient


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
    assert "不支持图片输入" in msg
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
