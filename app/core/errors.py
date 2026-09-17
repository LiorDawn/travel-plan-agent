"""统一异常体系 — services 用例层抛出的业务异常，由 middleware 统一映射为 HTTP 响应"""
from app.core.logging import logger


class AppError(Exception):
    """应用级业务异常基类。默认映射为 500。"""

    status_code: int = 500
    message: str = "服务器内部错误"

    def __init__(self, message: str = "", code: int = None):
        if message:
            self.message = message
        self.code = code or self.status_code
        super().__init__(self.message)


class NotFoundError(AppError):
    """资源不存在 → 404"""

    status_code = 404
    message = "资源不存在"


class BusinessError(AppError):
    """业务校验失败 → 400"""

    status_code = 400
    message = "请求不合法"

    def __init__(self, message: str = ""):
        super().__init__(message, code=400)
        logger.warning("business_error", message=self.message)