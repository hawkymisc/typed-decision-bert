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

#: Question types, for the ``loc`` tag Jev's 422 schema puts after the question ID.
_QUESTION_TYPE_TAGS = frozenset({"noul", "choice", "score"})


class JevBERTError(Exception):
    """Base class for every failure with a defined HTTP contract."""

    status_code: int = 500
    code: str = "internal_error"
    retryable: bool = False
    retry_after: int | None = None

    def __init__(
        self,
        message: str,
        *,
        path: ErrorPath | None = None,
        question_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        #: Only set once the question's type has been established, because the tag in
        #: ``detail[].loc`` must not be guessed (spec 5.9).
        self.question_type = question_type

    def body(self, request_id: str) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            error["path"] = self.path
        error["retryable"] = self.retryable
        payload: dict[str, Any] = {"error": error}
        if self.status_code == 422:
            payload["detail"] = self.detail()
        payload["request_id"] = request_id
        return payload

    def detail(self) -> list[dict[str, Any]]:
        """Jev's 422 shape, alongside JevBERT's own ``error`` object (spec 5.9, 3.4).

        The official SDK's wire schema - generated from Jev's OpenAPI document -
        defines a 422 as ``{"detail":[{"loc","msg","type"}]}``, so a client that reads
        ``detail`` directly keeps working against this server. ``input`` and ``ctx``
        are part of that schema and are deliberately left out: both carry the caller's
        own value back out in an error body (spec 15.2).
        """
        return [{"loc": self.detail_loc(), "msg": self.message, "type": self.code}]

    def detail_loc(self) -> list[str | int]:
        """``["body"] + path``, with the question type inserted after the question ID.

        The tag is only inserted when the path reaches *inside* a question and the type
        was actually determined; a question whose ``type`` is the thing being rejected
        has no type to name.
        """
        path = list(self.path or [])
        if (
            len(path) > 2
            and path[0] == "questions"
            and self.question_type in _QUESTION_TYPE_TAGS
        ):
            return ["body", path[0], path[1], self.question_type, *path[2:]]
        return ["body", *path]

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


def request_log(request: Request) -> dict[str, Any]:
    """The per-request log record the middleware will emit, if there is one."""
    state = request.scope.get("state") or {}
    log = state.get("log")
    return log if isinstance(log, dict) else {}


def error_response(
    request: Request, error: JevBERTError, *, headers: dict[str, str] | None = None
) -> Response:
    """Render a JevBERTError as the wire body defined in spec 5.9.

    Also records ``error_code`` on the access log line and, for a 5xx, emits an
    operator-facing line of its own. Without this a NaN logit or a broken invariant is
    a silent 500: the body deliberately says nothing, so the log is the only place the
    failure can show up (A-F1). Neither line carries request data (spec 15.2).
    """
    request_log(request)["error_code"] = error.code
    if error.status_code >= 500:
        logger.error(
            "request failed: status=%d error_code=%s error_class=%s",
            error.status_code,
            error.code,
            type(error).__name__,
        )
    merged = dict(error.headers())
    if headers:
        merged.update(headers)
    request_id = str(getattr(request.state, "request_id", ""))
    return json_response(error.body(request_id), status_code=error.status_code, headers=merged)


async def jevbert_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, JevBERTError)
    return error_response(request, exc)


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Translate Starlette's own 404/405/... into the JevBERT error body."""
    assert isinstance(exc, StarletteHTTPException)
    factory = _STATUS_TO_ERROR.get(exc.status_code)
    if factory is None:
        # A status outside the contract of spec 5.9 is a defect in this server, not a
        # code to invent on the wire. It is reported as what it is rather than passed
        # through with a mislabelled code (A-F12).
        logger.error("unmapped HTTP status raised internally: status=%d", exc.status_code)
        return error_response(request, InternalError("The request could not be processed."))
    message = _DEFAULT_MESSAGES.get(exc.status_code) or "The request could not be processed."
    # Starlette attaches ``Allow`` to the 405 it raises from routing; dropping it would
    # leave the caller without the one thing a 405 is supposed to tell them (A-F12).
    return error_response(request, factory(message), headers=dict(exc.headers or {}))


async def request_validation_handler(request: Request, exc: Exception) -> Response:
    """FastAPI's own parameter validation must not leak ``{"detail": ...}``."""
    logger.warning("request validation rejected by the framework: %s", type(exc).__name__)
    return error_response(request, ValidationError("The request could not be validated."))
