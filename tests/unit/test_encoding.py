"""Response JSON encoding (spec 5.5; K5).

The SDK validates responses in strict mode, so the wire form of a float matters.
"""

from __future__ import annotations

import json

import pytest

from jevbert.api.encoding import dumps, format_float


class TestFormatFloat:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, "1.0"),
            (0.0, "0.0"),
            (0.5, "0.5"),
            (1.6, "1.6"),
            (1.5999999999999999, "1.5999999999999999"),
        ],
    )
    def test_plain_values(self, value: float, expected: str) -> None:
        assert format_float(value) == expected

    def test_whole_numbers_never_become_integer_literals(self) -> None:
        for value in (0.0, 1.0, 2.0, 9.0):
            assert "." in format_float(value)

    def test_small_values_are_expanded_out_of_exponent_notation(self) -> None:
        # repr(1e-30) is "1e-30"; the wire form must be positional and still exact.
        text = format_float(1e-30)
        assert "e" not in text
        assert text.startswith("0.000000")
        assert float(text) == 1e-30

    def test_expansion_does_not_round(self) -> None:
        value = 1.2345678901234567e-15
        assert float(format_float(value)) == value

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_refused(self, bad: float) -> None:
        with pytest.raises(ValueError):
            format_float(bad)


class TestDumps:
    def test_key_order_is_preserved(self) -> None:
        assert dumps({"z": 1, "a": 2}) == '{"z":1,"a":2}'

    def test_non_ascii_is_not_escaped(self) -> None:
        assert dumps({"k": "返金"}) == '{"k":"返金"}'

    def test_booleans_are_not_numbers(self) -> None:
        assert dumps({"a": True, "b": 1}) == '{"a":true,"b":1}'

    def test_integers_stay_integers(self) -> None:
        # usage.input_tokens is an integer in appendix B.
        assert dumps({"input_tokens": 500}) == '{"input_tokens":500}'

    def test_nested_structures_round_trip(self) -> None:
        value = {"a": [1, 2.5, None, {"b": "c"}], "d": True}
        assert json.loads(dumps(value)) == value

    def test_non_string_keys_are_refused(self) -> None:
        with pytest.raises(TypeError):
            dumps({1: "a"})

    def test_unsupported_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            dumps({"a": {1, 2}})
