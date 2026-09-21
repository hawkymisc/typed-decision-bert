"""PT03: appendix A and the runtime validator must agree (POC_DESIGN 8.2).

The schema is the published structural contract; the server decides on parsed Python
values. If the two disagree, one of them is wrong - the schema is not decoration.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from jsonschema import Draft202012Validator

from jevbert.api.errors import ValidationError
from jevbert.config import Limits
from jevbert.contracts import load_schema
from jevbert.contracts.validator import validate_request
from tests.property.strategies import loose_requests

SCHEMA = load_schema("request")
SCHEMA_VALIDATOR = Draft202012Validator(SCHEMA)
LIMITS = Limits()


def accepted_by_schema(value: Any) -> bool:
    return SCHEMA_VALIDATOR.is_valid(value)


def accepted_by_validator(value: Any) -> bool:
    try:
        validate_request(value, LIMITS)
    except ValidationError:
        return False
    return True


class TestPT03Agreement:
    @given(loose_requests())
    @settings(max_examples=400)
    def test_verdicts_match(self, value: Any) -> None:
        assert accepted_by_schema(value) == accepted_by_validator(value), value

    @pytest.mark.parametrize(
        "value",
        [
            {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": [], "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": {}, "questions": {"q": {"type": "noul"}}},
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": "B"}}},
            },
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": ["a", {"b": 1}]}},
            },
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "noul", "criteria": {"true": "t"}}},
            },
            # ADR-016: appendix A now says additionalProperties: true at the top level,
            # and the validator's default agrees by ignoring the field.
            {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "extra": 1},
        ],
    )
    def test_known_valid_bodies(self, value: Any) -> None:
        assert accepted_by_schema(value)
        assert accepted_by_validator(value)

    @pytest.mark.parametrize(
        "value",
        [
            {},
            {"model": "", "state": "s", "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": None, "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": 1, "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": True, "questions": {"q": {"type": "noul"}}},
            {"model": "m", "state": "s", "questions": {}},
            {"model": "m", "state": "s", "questions": {"q": {"type": "other"}}},
            {"model": "m", "state": "s", "questions": {"q": {"type": "noul", "x": 1}}},
            {"model": "m", "state": "s", "questions": {"q": {"type": "choice"}}},
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "choice", "criteria": {"a": "A"}}},
            },
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": ["a"]}},
            },
            # U02: a null Score level is refused by both.
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "score", "criteria": ["a", None]}},
            },
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "noul", "criteria": {"maybe": "m"}}},
            },
        ],
    )
    def test_known_invalid_bodies(self, value: Any) -> None:
        assert not accepted_by_schema(value)
        assert not accepted_by_validator(value)

    def test_the_reject_setting_is_deliberately_stricter_than_the_schema(self) -> None:
        """The published schema describes the default; ``reject`` narrows it.

        Appendix A is the contract a caller codes against, so it has to describe what
        the server accepts out of the box. An operator who turns on ``reject`` is
        choosing to accept less than the published contract, which is the one direction
        that cannot surprise a caller into sending something valid and being refused
        without the operator having asked for it.
        """
        body = {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "extra": 1}
        assert accepted_by_schema(body)
        with pytest.raises(ValidationError):
            validate_request(body, LIMITS, unknown_top_level_fields="reject")


class TestBoundaryAgreement:
    @pytest.mark.parametrize("count", [1, 2, 255, 256])
    def test_choice_option_counts(self, count: int) -> None:
        body = {
            "model": "m",
            "state": "s",
            "questions": {
                "q": {"type": "choice", "criteria": {f"k{i:03d}": None for i in range(count)}}
            },
        }
        assert accepted_by_schema(body) == accepted_by_validator(body)

    @pytest.mark.parametrize("count", [1, 2, 10, 11])
    def test_score_level_counts(self, count: int) -> None:
        body = {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(count)]}},
        }
        assert accepted_by_schema(body) == accepted_by_validator(body)

    @pytest.mark.parametrize("count", [1, 256, 257])
    def test_question_counts(self, count: int) -> None:
        body = {
            "model": "m",
            "state": "s",
            "questions": {f"q{i}": {"type": "noul"} for i in range(count)},
        }
        assert accepted_by_schema(body) == accepted_by_validator(body)
