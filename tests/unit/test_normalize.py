"""Canonical JSON representation (spec 6.1; POC_DESIGN 5.1)."""

from __future__ import annotations

import json

from jevbert.compiler.normalize import canonical_json, render


class TestCanonicalJson:
    def test_object_keys_are_sorted_by_code_point(self) -> None:
        assert canonical_json({"b": 1, "a": 2, "A": 3}) == '{"A":3,"a":2,"b":1}'

    def test_nested_objects_are_sorted_recursively(self) -> None:
        assert canonical_json({"z": {"y": 1, "x": 2}}) == '{"z":{"x":2,"y":1}}'

    def test_array_order_is_preserved(self) -> None:
        assert canonical_json(["c", "a", "b"]) == '["c","a","b"]'

    def test_compact_separators(self) -> None:
        assert canonical_json({"a": [1, 2]}) == '{"a":[1,2]}'

    def test_non_ascii_is_not_escaped(self) -> None:
        assert canonical_json({"msg": "返金"}) == '{"msg":"返金"}'

    def test_insertion_order_does_not_change_the_result(self) -> None:
        # spec 6.1: HTTP object insertion order must not reach the model.
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_types_are_preserved(self) -> None:
        # spec 6.1: "1" and 1, "null" and null stay distinct.
        assert canonical_json({"v": "1"}) != canonical_json({"v": 1})
        assert canonical_json({"v": "null"}) != canonical_json({"v": None})
        assert canonical_json({"v": True}) != canonical_json({"v": 1})

    def test_strings_are_not_unicode_normalised(self) -> None:
        # spec 6.1: no NFKC. Half width and full width stay different.
        assert canonical_json("ｱ") != canonical_json("ア")
        assert canonical_json("  a\n") == '"  a\\n"'

    def test_round_trips_through_a_json_parser(self) -> None:
        value = {"b": [1, {"d": None, "c": True}], "a": "x"}
        assert json.loads(canonical_json(value)) == value


class TestRender:
    def test_string_content_is_passed_through_unchanged(self) -> None:
        assert render("返金してください") == "返金してください"

    def test_structured_content_uses_canonical_json(self) -> None:
        assert render({"b": 1, "a": 2}) == '{"a":2,"b":1}'
        assert render([1, "a"]) == '[1,"a"]'

    def test_known_poc_limitation_string_and_array_collide(self) -> None:
        # POC_DESIGN 5.1: documented limitation of serializer-nli-v1, recorded in
        # compat/differences.md. Asserted so that a silent change is noticed.
        assert render("[1]") == render([1])
