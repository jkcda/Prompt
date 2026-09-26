"""HTTP Basic Auth 中间件。

**为什么做进应用而不是靠 Nginx**：
「反代」这个动作本身不提供任何防护 —— Nginx 不配 `auth_basic`、Vite 的 proxy、
换端口，三者安全性**完全一样，都是零**：谁能连上，谁就能用。
区别只在于是 `auth_basic` 那三行在拦，还是别的什么东西在拦。
做进应用的好处是不依赖外部组件，`git pull` 完就生效。

**默认关闭**（`AUTH_PASSWORD` 为空）。本地开发不该被登录框挡路。

⚠️ `/api/health` **必须放行** —— Docker 的 HEALTHCHECK 是无凭据请求的，
拦掉会让容器永远处于 unhealthy，进而被 restart 策略反复重启。
"""

from __future__ import annotations

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("auth")

# 不需要认证的路径。只放行健康检查 —— 别的都该保护。
_OPEN_PATHS = frozenset({"/api/health"})

_REALM = "Video Prompt Reverse"


def _unauthorized() -> Response:
    """401 + WWW-Authenticate。

    ⚠️ `WWW-Authenticate` 头是**必需的** —— 少了它浏览器不会弹登录框，
    只会显示一个光秃秃的 401。有了它，浏览器会弹框、并在之后自动带上凭据，
    **包括 `EventSource`**（SSE 不能自定义 header，靠的就是浏览器缓存的这份凭据）。
    """
    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": f'Basic realm="{_REALM}", charset="UTF-8"'},
    )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """用户名 + 密码都对才放行。

    ⚠️ 用 `secrets.compare_digest` 而不是 `==`。
    `==` 是短路比较，逐字节返回的时间差可以被用来逐位猜出密码
    （时序攻击）。`compare_digest` 是常数时间。
    """

    def __init__(self, app, username: str, password: str) -> None:
        super().__init__(app)
        self._user = username.encode("utf-8")
        self._pass = password.encode("utf-8")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "basic" or not token:
            return _unauthorized()

        try:
            import base64

            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except Exception:
            # 解不出来就是格式不对，直接 401。
            # ⚠️ 这里**不能把异常内容回给客户端** —— 那等于告诉攻击者
            # 他试的这个串「格式对了但内容不对」，缩小了猜测范围。
            return _unauthorized()

        user, sep, pw = decoded.partition(":")
        if not sep:
            return _unauthorized()

        # 两个比较都要跑完，不要用 `and` 短路 —— 短路会泄露「用户名对不对」。
        user_ok = secrets.compare_digest(user.encode("utf-8"), self._user)
        pass_ok = secrets.compare_digest(pw.encode("utf-8"), self._pass)
        if not (user_ok and pass_ok):
            log.warning("认证失败：来自 %s 的 %s", request.client.host if request.client else "?", request.url.path)
            return _unauthorized()

        return await call_next(request)
