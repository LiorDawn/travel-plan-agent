"""安全 — API 管理令牌鉴权依赖（最小防线，方案 A）。

只用 FastAPI 依赖注入保护敏感端点（/audits、/monitor），/chat 与静态页放行：
- 请求头 `Authorization: Bearer <token>` 或 `X-API-Key: <token>` 二选一；
- 未配置 api_admin_token（空串）→ 鉴权关（本地演示零障碍）；
- 校验用 hmac.compare_digest 恒定时间比较，避免时序攻击。
"""
import hmac

from fastapi import Header, HTTPException

from app.core.config import get_settings


def _token_matches(given: str | None, expect: str) -> bool:
    if not given or not expect:
        return False
    try:
        return hmac.compare_digest(given.encode("utf-8"), expect.encode("utf-8"))
    except Exception:
        return False


async def require_api_token(
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> None:
    """敏感端点依赖：与配置令牌比对，不匹配抛 401。空令牌关闭鉴权。"""
    expect = get_settings().api_admin_token
    if not expect:  # 未配置 → 鉴权关闭，放行
        return
    given = None
    if x_api_key:
        given = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        given = authorization[7:].strip()
    if not _token_matches(given, expect):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")