"""Request ID, response headers and structured logging (spec 5.7, 15.2; POC_DESIGN 7.1).

Implemented as raw ASGI rather than ``BaseHTTPMiddleware`` for two reasons: the headers
have to be attached to every response including the ones produced by exception handlers,
and an exception escaping the router has to become a JevBERT 500 here, where the request
ID is still in scope, rather than in Starlette's outermost error middleware where it is
not.

The log line never carries state, instructions, criteria, option keys, question IDs or
credentials (spec 15.2).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from jevbert import CONFIDENCE_DEFINITION, CONTRACT_PROFILE
from jevbert.api.encoding import dumps

logger = logging.getLogger("jevbert.access")

#: Headers that only exist once a bundle has been resolved (POC_DESIGN 4.4).
BUNDLE_HEADERS = ("X-JevBERT-Bundle", "X-JevBERT-Usage", "X-JevBERT-Calibration")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, contract_profile: str = CONTRACT_PROFILE) -> None:
        self.app = app
        self.contract_profile = contract_profile

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        started_at = time.monotonic()
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["started_at"] = started_at
        state["extra_headers"] = {}
        state["log"] = {}

        response_started = False
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                # The official SDK reads this one as `response.request_id` (spec 3.4).
                headers["x-typesafe-request-id"] = request_id
                headers["X-JevBERT-Contract"] = self.contract_profile
                headers["X-JevBERT-Confidence"] = CONFIDENCE_DEFINITION
                for name, value in state["extra_headers"].items():
                    headers[name] = value
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            # Never let a traceback or a framework body reach the client (spec 4.3).
            logger.error(
                "unhandled error while serving request %s: %s",
                request_id,
                type(exc).__name__,
                exc_info=exc,
            )
            state["log"]["error_code"] = "internal_error"
            if response_started:
                raise
            status_code = 500
            await _send_internal_error(send_wrapper, request_id)
        finally:
            self._log(scope, state, status_code, time.monotonic() - started_at)

    def _log(
        self, scope: Scope, state: dict[str, Any], status_code: int, elapsed: float
    ) -> None:
        record = {
            "request_id": state["request_id"],
            "method": scope.get("method"),
            "path": scope.get("path"),
            "status": status_code,
            "total_ms": round(elapsed * 1000, 3),
        }
        record.update(state["log"])
        logger.info(dumps(record))


async def _send_internal_error(send: Send, request_id: str) -> None:
    body = dumps(
        {
            "error": {
                "code": "internal_error",
                "message": "The request could not be processed.",
                "retryable": False,
            },
            "request_id": request_id,
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
