"""Bearer authentication (spec 5.1; POC_DESIGN 4.3; ADR-015).

Static keys compared in constant time. There is no unauthenticated mode: a server with
no configured key refuses to start (``jevbert.config``). Keys never reach a log line or
an error body (spec 15.2).
"""

from __future__ import annotations

import hmac
from collections.abc import Sequence

from starlette.requests import Request

from jevbert.api.errors import UnauthorizedError

_SCHEME = "bearer"


def authenticate(request: Request, api_keys: Sequence[str]) -> None:
    """Raise ``UnauthorizedError`` unless the request carries a configured key."""
    header = request.headers.get("authorization")
    if not header:
        raise UnauthorizedError("An Authorization header with a Bearer token is required.")

    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != _SCHEME:
        raise UnauthorizedError("The Authorization header must use the Bearer scheme.")

    presented = parts[1].strip()
    if not presented:
        raise UnauthorizedError("The Bearer token is empty.")

    # Every key is compared so that the time taken does not depend on which one matches.
    matched = False
    for key in api_keys:
        matched |= hmac.compare_digest(presented, key)
    if not matched:
        raise UnauthorizedError("The provided credentials are not valid.")
