"""安全响应头中间件（M9d）：给所有响应加传输层加固头。

为什么放 FastAPI 而不是全交给 Caddy？这些头是【应用自身的安全契约】：可单测、可移植——
换掉边缘（Caddy→Nginx→云 LB）也不会丢；让 Caddy 专注做它最擅长的 TLS 终止与反代。

HSTS 只在 https 下下发：明文 http 发 HSTS 无意义、规范也不建议。scope["scheme"] 在 uvicorn
`--proxy-headers` 下会依 X-Forwarded-Proto 变成 "https"——这正是【信任边界】的落地点：只有
可信反代（uvicorn `--forwarded-allow-ips` 限定来源）传的 proto 才被采信，否则外部客户端能
伪造 `X-Forwarded-Proto: https` 骗过这里的判断。生产务必把 forwarded-allow-ips 收窄到反代 IP。

其余三个头是低成本高收益的通用加固：
  - X-Content-Type-Options: nosniff —— 禁 MIME 嗅探（防把上传文本当脚本执行）；
  - X-Frame-Options: DENY —— 禁被 iframe 内嵌（防点击劫持）；
  - Referrer-Policy —— 跨源只发 origin，不泄露完整 URL（可能含敏感 query）。
"""

from __future__ import annotations

from app.core.config import settings


class SecurityHeadersMiddleware:
    """纯 ASGI 中间件：在 http.response.start 时给响应头追加安全头。"""

    def __init__(self, app):  # noqa: ANN001
        self.app = app

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        if scope["type"] != "http" or not settings.security_headers_enabled:
            await self.app(scope, receive, send)
            return

        # uvicorn --proxy-headers 下，反代传 X-Forwarded-Proto: https 会把 scheme 置为 https。
        is_https = scope.get("scheme") == "https"

        async def send_wrapper(message):  # noqa: ANN001
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                if is_https:
                    value = f"max-age={settings.hsts_max_age}; includeSubDomains"
                    if settings.hsts_preload:
                        value += "; preload"
                    headers.append((b"strict-transport-security", value.encode()))
            await send(message)

        await self.app(scope, receive, send_wrapper)
