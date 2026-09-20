"""Error hierarchy and error body construction (spec 5.9; POC_DESIGN 4.5).

Every failure that reaches the client is one of these. The framework's own error bodies
(FastAPI's ``{"detail": ...}``) must never be exposed, so the handlers registered here
cover 404, 405, request validation and uncaught exceptions as well.
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from jevbert.api.encoding import json_response

logger = logging.getLogger("jevbert.api")

#: Path elements are object keys (str) or array indices (int).
ErrorPath = list[str | int]


class JevBERTError(Exception):
    """Base class for every failure with a defined HTTP contract."""

    status_code: int = 500
    code: str = "internal_error"
    retryable: bool = False
    retry_after: int | None = None

    def __init__(self, message: str, *, path: ErrorPath | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def body(self, request_id: str) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            error["path"] = self.path
        error["retryable"] = self.retryable
        return {"error": error, "request_id": request_id}

    def headers(self) -> dict[str, str]:
        if self.retry_after is None:
            return {}
        return {"Retry-After": str(self.retry_after)}


class UnauthorizedError(JevBERTError):
    status_code = 401
    code = "unauthorized"


class UnsupportedMediaTypeError(JevBERTError):
    status_code = 415
    code = "unsupported_media_type"


class RequestTooLargeError(JevBERTError):
    status_code = 413
    code = "request_too_large"


class InvalidJSONError(JevBERTError):
    status_code = 400
    code = "invalid_json"


class ValidationError(JevBERTError):
    status_code = 422
    code = "validation_error"


class ModelNotFoundError(JevBERTError):
    status_code = 422
    code = "model_not_found"


class ContextLengthExceededError(JevBERTError):
    status_code = 422
    code = "context_length_exceeded"


class RateLimitExceededError(JevBERTError):
    """Not raised in the PoC: per-caller rate limiting is unimplemented (N18)."""

    status_code = 429
    code = "rate_limit_exceeded"
    retryable = True
    retry_after = 1


class OverloadedError(JevBERTError):
    status_code = 529
    code = "overloaded"
    retryable = True
    retry_after = 1


class ModelUnavailableError(JevBERTError):
    status_code = 503
    code = "model_unavailable"
    retryable = True
    retry_after = 1


class DeadlineExceededError(JevBERTError):
    status_code = 504
    code = "deadline_exceeded"
    retryable = True


class InferenceError(JevBERTError):
    status_code = 500
    code = "inference_error"


class InternalError(JevBERTError):
    status_code = 500
    code = "internal_error"


class NotFoundError(JevBERTError):
    status_code = 404
    code = "not_found"


class MethodNotAllowedError(JevBERTError):
    status_code = 405
    code = "method_not_allowed"


_STATUS_TO_ERROR: dict[int, type[JevBERTError]] = {
    400: InvalidJSONError,
    401: UnauthorizedError,
    404: NotFoundError,
    405: MethodNotAllowedError,
    413: RequestTooLargeError,
    415: UnsupportedMediaTypeError,
    422: ValidationError,
    429: RateLimitExceededError,
    503: ModelUnavailableError,
    504: DeadlineExceededError,
    529: OverloadedError,
}

_DEFAULT_MESSAGES: dict[int, str] = {
    404: "The requested path does not exist.",
    405: "The method is not supported for this path.",
}


def error_response(request: Request, error: JevBERTError) -> Response:
    """Render a JevBERTError as the wire body defined in spec 5.9."""
    request_id = str(getattr(request.state, "request_id", ""))
    return json_response(
        error.body(request_id),
        status_code=error.status_code,
        headers=error.headers(),
    )


async def jevbert_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, JevBERTError)
    return error_response(request, exc)


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Translate Starlette's own 404/405/... into the JevBERT error body."""
    assert isinstance(exc, StarletteHTTPException)
    factory = _STATUS_TO_ERROR.get(exc.status_code, InternalError)
    message = _DEFAULT_MESSAGES.get(exc.status_code) or "The request could not be processed."
    error = factory(message)
    error.status_code = exc.status_code
    return error_response(request, error)


async def request_validation_handler(request: Request, exc: Exception) -> Response:
    """FastAPI's own parameter validation must not leak ``{"detail": ...}``."""
    logger.warning("request validation rejected by the framework: %s", type(exc).__name__)
    return error_response(request, ValidationError("The request could not be validated."))
