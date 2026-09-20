"""Response JSON encoding.

The official SDK validates responses with pydantic in strict mode (spec 3.4), so a
probability that happens to be a whole number must not reach the wire as ``1`` and must
not be rounded on the way out (I03, spec 5.5). ``json.dumps`` already writes floats with
``repr``; this module additionally expands exponent notation so that every float is
written in plain decimal form, exactly and without rounding.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

from starlette.responses import Response

#: Insertion order is meaningful (spec 5.5), so nothing here sorts keys.
_STRING_ESCAPER = json.JSONEncoder(ensure_ascii=False).encode


def format_float(value: float) -> str:
    """Return the shortest exact decimal representation, never an integer literal."""
    if not math.isfinite(value):
        raise ValueError("Non-finite numbers must never reach the response body")
    text = repr(value)
    if "e" in text or "E" in text:
        # Decimal(repr(v)) is exact for the shortest round-trip repr, so expanding to
        # positional notation neither rounds nor loses the value.
        text = format(Decimal(text), "f")
    if "." not in text:
        text += ".0"
    return text


def dumps(value: Any) -> str:
    """Serialize a JSON value, preserving key order and float formatting."""
    out: list[str] = []
    _write(value, out)
    return "".join(out)


def _write(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(_STRING_ESCAPER(value))
    elif isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, float):
        out.append(format_float(value))
    elif isinstance(value, list | tuple):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out)
        out.append("]")
    elif isinstance(value, dict):
        out.append("{")
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            if index:
                out.append(",")
            out.append(_STRING_ESCAPER(key))
            out.append(":")
            _write(item, out)
        out.append("}")
    else:
        raise TypeError(f"{type(value).__name__} is not JSON serializable")


def json_response(
    payload: Any, *, status_code: int = 200, headers: dict[str, str] | None = None
) -> Response:
    return Response(
        content=dumps(payload),
        status_code=status_code,
        media_type="application/json",
        headers=headers,
    )
