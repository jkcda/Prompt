"""访问认证测试。

**为什么要认证**：这个服务没有别的鉴权，谁能连上就能提交反推任务 ——
攻击者不需要偷 VLM 密钥，直接拿服务器当免费代理，每次调用都烧额度。
「换端口」「加 Nginx 反代」都拦不住，反代只是转发不是权限。

测试策略：直接用最小 FastAPI 应用挂中间件，覆盖全部逻辑分支。
（不走 `app.main` 的模块级注册，那样要 reload 模块 + 清 settings 缓存，脆。）
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.auth import BasicAuthMiddleware

USER = "alice"
PASSWORD = "s3cret-pw"


def _app(username: str = USER, password: str = PASSWORD) -> TestClient:
    a = FastAPI()
    a.add_middleware(BasicAuthMiddleware, username=username, password=password)

    @a.get("/api/health")
    async def health():
        return {"ok": True}

    @a.get("/api/settings")
    async def settings_():
        return {"vlm": "x"}

    @a.get("/")
    async def index():
        return {"page": True}

    return TestClient(a)


def _creds(user: str, pw: str) -> dict[str, str]:
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


def test_health_is_open_without_credentials():
    """⚠️ 健康检查必须放行 —— Docker HEALTHCHECK 是无凭据请求。

    拦掉它容器会永远 unhealthy，被 restart 策略反复重启，
    表现是「服务明明起来了却一直在重启」。
    """
    assert _app().get("/api/health").status_code == 200


def test_protected_route_rejects_without_credentials():
    r = _app().get("/api/settings")
    assert r.status_code == 401


def test_401_carries_www_authenticate():
    """⚠️ 少了这个头浏览器不会弹登录框，只显示光秃秃的 401。

    而且 `EventSource`（SSE 进度推送）**不能自定义 header**，
    它靠的就是浏览器缓存了这份凭据 —— 没有这个头，进度条永远不动。
    """
    r = _app().get("/api/settings")
    assert "www-authenticate" in {k.lower() for k in r.headers}
    assert "Basic" in r.headers["WWW-Authenticate"]


def test_index_page_is_also_protected():
    """SPA 入口也要拦 —— 否则前端页面能打开，只是接口报错，体验很怪。"""
    assert _app().get("/").status_code == 401


@pytest.mark.parametrize(
    ("user", "pw"),
    [
        (USER, "wrong-password"),
        ("wrong-user", PASSWORD),
        ("", PASSWORD),
        (USER, ""),
    ],
)
def test_wrong_credentials_rejected(user: str, pw: str):
    assert _app().get("/api/settings", headers=_creds(user, pw)).status_code == 401


def test_correct_credentials_accepted():
    r = _app().get("/api/settings", headers=_creds(USER, PASSWORD))
    assert r.status_code == 200
    assert r.json()["vlm"] == "x"


def test_health_is_open_regardless_of_credentials():
    """健康检查是**无条件放行** —— 带了错凭据也照样 200。

    这是刻意的：`_OPEN_PATHS` 在解析 Authorization 之前就 return 了。
    Docker 的 HEALTHCHECK 只关心进程活着，不该因为凭据问题判死。
    """
    assert _app().get("/api/health", headers=_creds("x", "y")).status_code == 200
    assert _app().get("/api/health", headers={"Authorization": "garbage"}).status_code == 200
    assert _app().get("/api/health").status_code == 200


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Basic",
        "Basic ",
        "Bearer abc",
        "Basic not-base64!!!",
        "Basic " + base64.b64encode(b"no-colon-here").decode(),
    ],
)
def test_malformed_authorization_header_rejected(header: str):
    """各种畸形头都要 401，而且**不能把失败原因回给客户端** ——
    那等于告诉攻击者「格式对了但内容不对」，缩小猜测范围。"""
    r = _app().get("/api/settings", headers={"Authorization": header})
    assert r.status_code == 401
    assert r.text == "Unauthorized", "不能泄露具体失败原因"


def test_username_and_password_are_both_checked():
    """用户名对、密码错，和密码对、用户名错，都要拒绝。

    ⚠️ 实现里两个 compare_digest 都要跑完，不能用 `and` 短路 ——
    短路会泄露「用户名对不对」。
    """
    assert _app().get("/api/settings", headers=_creds(USER, "x")).status_code == 401
    assert _app().get("/api/settings", headers=_creds("x", PASSWORD)).status_code == 401


def test_main_wires_auth_when_password_is_set():
    """`app.main` 必须在设了密码时挂上中间件。

    这条是源码级检查 —— 中间件挂漏了的话上面所有测试照样全绿，
    但线上完全没有防护，属于最危险的一类回归。
    """
    src = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text(encoding="utf-8")
    assert "BasicAuthMiddleware" in src, "main.py 没引用认证中间件"
    assert "if _settings.auth_password:" in src, "认证必须是「设了密码才启用」的条件挂载"
    # 没设密码时要留下警告 —— 对外暴露却忘了设，日志是唯一的提示
    assert "未启用访问认证" in src
