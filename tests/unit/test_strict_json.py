"""Strict JSON parser (POC_DESIGN 4.1; spec 5.2, CT05, CT06)."""

from __future__ import annotations

import pytest

from jevbert.api.errors import InvalidJSONError, ValidationError
from jevbert.contracts.strict_json import parse_strict_json


def _deep(depth: int) -> bytes:
    """Body whose top level object counts as depth 1."""
    inner = "1"
    for _ in range(depth - 1):
        inner = "[" + inner + "]"
    return ('{"a":' + inner + "}").encode()


class TestAcceptedInput:
    def test_parses_an_object(self) -> None:
        assert parse_strict_json(b'{"a": 1}') == {"a": 1}

    def test_keeps_integers_as_int_and_decimals_as_float(self) -> None:
        value = parse_strict_json(b'{"i": 1, "f": 1.0}')
        assert isinstance(value["i"], int)
        assert isinstance(value["f"], float)

    def test_does_not_convert_booleans_to_numbers(self) -> None:
        # CT05: no implicit boolean -> number conversion.
        value = parse_strict_json(b'{"b": true}')
        assert value["b"] is True
        assert not isinstance(value["b"], int) or value["b"] is True

    def test_accepts_multibyte_utf8(self) -> None:
        assert parse_strict_json('{"a": "返金"}'.encode()) == {"a": "返金"}

    def test_accepts_escaped_surrogate_pair(self) -> None:
        escaped = "".join(f"\\u{code}" for code in ("d83d", "de00"))
        body = ('{"a": "' + escaped + '"}').encode()
        assert parse_strict_json(body) == {"a": chr(0x1F600)}

    def test_object_key_order_is_preserved(self) -> None:
        value = parse_strict_json(b'{"b": 1, "a": 2}')
        assert list(value) == ["b", "a"]


class TestRejectedInput:
    @pytest.mark.parametrize(
        "body",
        [
            b"",
            b"{",
            b'{"a": }',
            b'{"a": 1,}',
            b"{'a': 1}",
        ],
    )
    def test_malformed_json_is_rejected(self, body: bytes) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(body)

    def test_bom_is_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b"\xef\xbb\xbf" + b'{"a": 1}')

    def test_invalid_utf8_is_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b'{"a": "\xff\xfe"}')

    def test_lone_surrogate_escape_is_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(rb'{"a": "\ud800"}')

    def test_lone_surrogate_in_a_key_is_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(rb'{"\udc00": 1}')

    @pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
    def test_non_finite_constants_are_rejected(self, literal: bytes) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b'{"a": ' + literal + b"}")

    @pytest.mark.parametrize("literal", [b"1e999", b"-1e999"])
    def test_numbers_that_overflow_to_infinity_are_rejected(self, literal: bytes) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b'{"a": ' + literal + b"}")

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b'{"a": 1, "a": 2}')

    def test_duplicate_keys_nested_are_rejected(self) -> None:
        with pytest.raises(InvalidJSONError):
            parse_strict_json(b'{"outer": {"a": 1, "a": 2}}')

    def test_duplicate_key_inside_a_string_value_is_not_a_duplicate(self) -> None:
        assert parse_strict_json(b'{"a": "\\"a\\": 1"}') == {"a": '"a": 1'}


class TestDepth:
    def test_depth_at_the_limit_is_accepted(self) -> None:
        # CT06: 32 is accepted, 33 is refused.
        assert parse_strict_json(_deep(32), max_depth=32) is not None

    def test_depth_above_the_limit_is_a_validation_error(self) -> None:
        # POC_DESIGN 4.1: depth is 422, not 400, and is judged before json.loads so
        # that a deep body never reaches Python's recursion limit.
        with pytest.raises(ValidationError):
            parse_strict_json(_deep(33), max_depth=32)

    def test_very_deep_body_does_not_raise_recursion_error(self) -> None:
        with pytest.raises(ValidationError):
            parse_strict_json(_deep(5000), max_depth=32)

    def test_brackets_inside_strings_do_not_count(self) -> None:
        body = b'{"a": "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["}'
        assert parse_strict_json(body, max_depth=32)["a"].startswith("[")

    def test_escaped_quote_does_not_end_the_string_for_the_depth_scan(self) -> None:
        body = rb'{"a": "\"[[[", "b": 1}'
        assert parse_strict_json(body, max_depth=32)["b"] == 1


class TestTopLevelShape:
    @pytest.mark.parametrize("body", [b"null", b"1", b"true", b'"text"', b"[1]"])
    def test_non_object_top_level_parses_here_and_is_refused_by_the_validator(
        self, body: bytes
    ) -> None:
        # The parser only guarantees JSON validity; shape belongs to the validator.
        parse_strict_json(body)
