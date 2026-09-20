"""Bearer authentication (spec 5.1; POC_DESIGN 4.3; ADR-015).

Static keys compared in constant time. There is no unauthenticated mode: a server with
no configured key refuses to start (``jevbert.config``). Keys never reach a log line or
an error body (spec 15.2).

Authentication is applied by :class:`AuthenticationMiddleware` in front of routing
rather than by each handler, so that a route added later is closed by default and an
unauthenticated caller cannot tell an existing path from a missing one (S-M3).
"""

from __future__ import annotations

import hmac

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from jevbert.api.encoding import json_response
from jevbert.api.errors import UnauthorizedError

_SCHEME = "bearer"

#: The only paths served without credentials: liveness and readiness, both of which
#: answer with a fixed word and disclose nothing about the models (spec 16.1, S-L2).
PUBLIC_PATHS = frozenset({"/healthz", "/readyz"})


def _credential_bytes(value: str) -> bytes:
    """Recover the bytes a client actually sent in a header.

    ASGI hands header values over already decoded as latin-1, so a UTF-8 credential
    arrives mojibake'd. Encoding it back with latin-1 reproduces the original bytes,
    which is what a configured key is compared against. ``hmac.compare_digest`` refuses
    non-ASCII ``str`` with a ``TypeError``; comparing bytes removes that whole class of
    failure, so a hostile header is a 401 rather than a 500 (S-H1).
    """
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        # Not reachable through ASGI, but a direct caller may pass astral characters.
        return value.encode("utf-8")


def authenticate(request: Request, api_keys: tuple[str, ...]) -> None:
    """Raise ``UnauthorizedError`` unless the request carries a configured key."""
    header = request.headers.get("authorization")
    if not header:
        raise UnauthorizedError("An Authorization header with a Bearer token is required.")

    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != _SCHEME:
        raise UnauthorizedError("The Authorization header must use the Bearer scheme.")

    presented = _credential_bytes(parts[1].strip())
    if not presented:
        raise UnauthorizedError("The Bearer token is empty.")

    # Every key is compared so that the time taken does not depend on which one matches.
    matched = False
    for key in api_keys:
        matched |= hmac.compare_digest(presented, key.encode("utf-8"))
    if not matched:
        raise UnauthorizedError("The provided credentials are not valid.")


class AuthenticationMiddleware:
    """Default-deny gate in front of routing.

    Placed inside ``RequestContextMiddleware`` so that the 401 it produces still gets a
    request ID, the standard headers and an access log line, and outside the router so
    that it answers before 404 and 405 do.
    """

    def __init__(self, app: ASGIApp, *, api_keys: tuple[str, ...]) -> None:
        self.app = app
        self.api_keys = tuple(api_keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        try:
            authenticate(request, self.api_keys)
        except UnauthorizedError as exc:
            state = scope.get("state") or {}
            log = state.get("log")
            if isinstance(log, dict):
                log["error_code"] = exc.code
            response = json_response(
                exc.body(str(state.get("request_id", ""))),
                status_code=exc.status_code,
                headers=exc.headers(),
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
