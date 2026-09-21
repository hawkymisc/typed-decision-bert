"""Canonical JSON representation (spec 6.1; POC_DESIGN 5.1)."""

from __future__ import annotations

import json
from typing import Any

#: JSON text form fed to the model: sorted object keys, preserved array order, compact
#: separators, no ASCII escaping and no Unicode normalisation of the contents.
_DUMP = json.JSONEncoder(
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
    check_circular=False,
).encode


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON text of ``value``.

    Object keys are sorted by Unicode code point so that HTTP insertion order cannot
    reach the model; arrays keep their order because it carries meaning.
    """
    return _DUMP(value)


def render(content: Any) -> str:
    """Render Content as model input text.

    A string is passed through unchanged so that natural language reaches an NLI model
    as natural language. Arrays and objects use :func:`canonical_json`.

    Known limitation (POC_DESIGN 5.1): the top level string ``"[1]"`` and the array
    ``[1]`` render identically. This is a serializer-nli-v1 (PoC) decision recorded in
    ``compat/differences.md``; ``serializer-v1`` must not inherit it.
    """
    if isinstance(content, str):
        return content
    return canonical_json(content)
