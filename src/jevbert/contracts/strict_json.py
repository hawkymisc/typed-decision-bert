"""Strict JSON parser (POC_DESIGN 4.1; spec 5.2).

The framework's own body parsing is not used. A request body is raw bytes here, and
everything a permissive parser would accept silently - a BOM, duplicate keys, ``NaN``,
``1e999``, a lone surrogate - is refused before any value reaches the validator.
"""

from __future__ import annotations

import codecs
import json
from typing import Any

from jevbert.api.errors import InvalidJSONError, ValidationError

_SURROGATE_START = 0xD800
_SURROGATE_END = 0xDFFF


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class _NonFiniteNumberError(ValueError):
    pass


def parse_strict_json(body: bytes, *, max_depth: int = 32) -> Any:
    """Parse a request body.

    Raises:
        ValidationError: the nesting depth exceeds ``max_depth`` (422, spec 4.2).
        InvalidJSONError: anything else that makes the body unacceptable (400).
    """
    if body.startswith(codecs.BOM_UTF8):
        raise InvalidJSONError("The request body must not start with a byte order mark.")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidJSONError("The request body is not valid UTF-8.") from exc
    if text.startswith("﻿"):
        raise InvalidJSONError("The request body must not start with a byte order mark.")

    # Checked on the raw text so that a deeply nested body never reaches the recursive
    # decoder and trips Python's recursion limit.
    _check_depth(text, max_depth)

    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs_hook,
            parse_constant=_parse_constant,
            parse_float=_parse_float,
        )
    except _DuplicateKeyError as exc:
        # The key is not quoted back: the error body is readable by whoever sent the
        # request, and a duplicated key name is the caller's own data (S-L1).
        raise InvalidJSONError("The request body contains a duplicate object key.") from exc
    except _NonFiniteNumberError as exc:
        raise InvalidJSONError("JSON numbers must be finite.") from exc
    except json.JSONDecodeError as exc:
        raise InvalidJSONError("The request body is not valid JSON.") from exc
    except RecursionError as exc:
        raise InvalidJSONError("The request body is nested too deeply.") from exc
    except ValueError as exc:
        # CPython refuses ``int()`` on literals longer than ``sys.int_info`` allows
        # (4300 digits by default) with a plain ValueError, which would otherwise make
        # a caller-controlled literal a 500 (S-M1). Any other ValueError from the
        # decoder is likewise a malformed body, not a server fault.
        raise InvalidJSONError("The request body is not valid JSON.") from exc

    _reject_lone_surrogates(value)
    return value


def _check_depth(text: str, max_depth: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > max_depth:
                raise ValidationError(
                    f"The request body is nested deeper than {max_depth} levels."
                )
        elif char in "]}":
            depth -= 1


def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _parse_constant(constant: str) -> Any:
    raise _NonFiniteNumberError(constant)


def _parse_float(text: str) -> float:
    value = float(text)
    # `1e999` parses without error but is infinite, which spec 5.2 forbids.
    if value in (float("inf"), float("-inf")):
        raise _NonFiniteNumberError(text)
    return value


def _reject_lone_surrogates(value: Any) -> None:
    if isinstance(value, str):
        _check_string(value)
    elif isinstance(value, list):
        for item in value:
            _reject_lone_surrogates(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_string(key)
            _reject_lone_surrogates(item)


def _check_string(text: str) -> None:
    for char in text:
        if _SURROGATE_START <= ord(char) <= _SURROGATE_END:
            raise InvalidJSONError("JSON strings must not contain unpaired surrogates.")
