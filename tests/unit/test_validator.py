"""Contract validator (spec 5.2, 5.3, appendix A; POC_DESIGN 4.2; CT01-CT05)."""

from __future__ import annotations

from typing import Any

import pytest

from jevbert.api.errors import ValidationError
from jevbert.config import Limits
from jevbert.contracts.validator import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    validate_request,
)

LIMITS = Limits()


def _request(**questions: Any) -> dict[str, Any]:
    return {"model": "m", "state": "s", "questions": questions}


def _validate(value: Any) -> Any:
    return validate_request(value, LIMITS)


class TestTopLevel:
    def test_minimal_valid_request(self) -> None:
        result = _validate(_request(q={"type": "noul"}))
        assert result.model == "m"
        assert result.state == "s"
        assert len(result.questions) == 1

    def test_question_order_is_preserved(self) -> None:
        result = _validate(_request(b={"type": "noul"}, a={"type": "noul"}))
        assert [q.question_id for q in result.questions] == ["b", "a"]

    @pytest.mark.parametrize("value", [None, 1, True, "text", [1]])
    def test_non_object_body_is_rejected(self, value: Any) -> None:
        with pytest.raises(ValidationError):
            _validate(value)

    @pytest.mark.parametrize("missing", ["model", "state", "questions"])
    def test_required_fields(self, missing: str) -> None:
        body = _request(q={"type": "noul"})
        del body[missing]
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == [missing]

    def test_unknown_top_level_field_is_rejected(self) -> None:
        # spec 5.3 / SDK extra_body.
        body = _request(q={"type": "noul"}) | {"temperature": 0.5}
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["temperature"]

    @pytest.mark.parametrize("value", ["", 1, None, [], {}])
    def test_model_must_be_a_non_empty_string(self, value: Any) -> None:
        body = _request(q={"type": "noul"})
        body["model"] = value
        with pytest.raises(ValidationError):
            _validate(body)

    @pytest.mark.parametrize("value", ["", [], {}, ["a"], {"k": None}])
    def test_state_accepts_every_content_shape_including_empty(self, value: Any) -> None:
        body = _request(q={"type": "noul"})
        body["state"] = value
        assert _validate(body).state == value

    @pytest.mark.parametrize("value", [None, 1, 1.5, True])
    def test_state_top_level_scalars_other_than_string_are_rejected(self, value: Any) -> None:
        # spec 5.2: null/boolean/number are refused at the top level of state.
        body = _request(q={"type": "noul"})
        body["state"] = value
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["state"]

    def test_scalars_are_allowed_inside_state(self) -> None:
        body = _request(q={"type": "noul"})
        body["state"] = {"a": None, "b": True, "c": 1.5}
        assert _validate(body).state == {"a": None, "b": True, "c": 1.5}


class TestQuestionCount:
    def test_empty_questions_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate(_request())

    def test_thirty_two_questions_are_accepted(self) -> None:
        body = _request(**{f"q{i}": {"type": "noul"} for i in range(32)})
        assert len(_validate(body).questions) == 32

    def test_thirty_three_questions_are_rejected(self) -> None:
        body = _request(**{f"q{i}": {"type": "noul"} for i in range(33)})
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["questions"]

    def test_questions_must_be_an_object(self) -> None:
        body = {"model": "m", "state": "s", "questions": [{"type": "noul"}]}
        with pytest.raises(ValidationError):
            _validate(body)

    def test_question_must_be_an_object(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q="noul"))
        assert excinfo.value.path == ["questions", "q"]


class TestQuestionType:
    @pytest.mark.parametrize("bad", ["Noul", "boolean", "", None, 1])
    def test_unknown_type_is_rejected(self, bad: Any) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q={"type": bad}))
        assert excinfo.value.path == ["questions", "q", "type"]

    def test_missing_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _validate(_request(q={"instructions": "x"}))

    def test_unknown_question_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q={"type": "noul", "weight": 1}))
        assert excinfo.value.path == ["questions", "q", "weight"]


class TestInstructions:
    def test_absent_and_null_instructions_are_the_same(self) -> None:
        # U01 / spec 3.4: SDK 0.7.0 omits unset fields from the wire.
        absent = _validate(_request(q={"type": "noul"})).questions[0]
        explicit = _validate(_request(q={"type": "noul", "instructions": None})).questions[0]
        assert absent.instructions is None
        assert explicit.instructions is None

    @pytest.mark.parametrize("value", ["text", ["a", "b"], {"k": "v"}])
    def test_structured_instructions_are_kept(self, value: Any) -> None:
        question = _validate(_request(q={"type": "noul", "instructions": value})).questions[0]
        assert question.instructions == value

    @pytest.mark.parametrize("value", [1, True, 1.5])
    def test_scalar_instructions_other_than_string_are_rejected(self, value: Any) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q={"type": "noul", "instructions": value}))
        assert excinfo.value.path == ["questions", "q", "instructions"]


class TestNoulCriteria:
    def test_criteria_may_be_absent(self) -> None:
        question = _validate(_request(q={"type": "noul"})).questions[0]
        assert isinstance(question, NoulQuestion)
        assert question.criteria_true is None
        assert question.criteria_false is None

    @pytest.mark.parametrize("value", [None, {}])
    def test_null_and_empty_criteria_are_accepted(self, value: Any) -> None:
        question = _validate(_request(q={"type": "noul", "criteria": value})).questions[0]
        assert question.criteria_true is None
        assert question.criteria_false is None

    def test_one_sided_criteria_is_accepted(self) -> None:
        body = _request(q={"type": "noul", "criteria": {"true": "a refund is requested"}})
        question = _validate(body).questions[0]
        assert question.criteria_true == "a refund is requested"
        assert question.criteria_false is None

    def test_explicit_null_side_is_the_same_as_an_absent_side(self) -> None:
        body = _request(q={"type": "noul", "criteria": {"true": "yes", "false": None}})
        question = _validate(body).questions[0]
        assert question.criteria_false is None

    def test_structured_criteria_sides_are_kept(self) -> None:
        body = _request(q={"type": "noul", "criteria": {"true": {"k": [1, 2]}}})
        assert _validate(body).questions[0].criteria_true == {"k": [1, 2]}

    @pytest.mark.parametrize("key", ["yes", "True", "1", ""])
    def test_only_true_and_false_keys_are_allowed(self, key: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q={"type": "noul", "criteria": {key: "x"}}))
        assert excinfo.value.path == ["questions", "q", "criteria", key]

    @pytest.mark.parametrize("value", ["text", [1], 1])
    def test_criteria_must_be_an_object_or_null(self, value: Any) -> None:
        with pytest.raises(ValidationError):
            _validate(_request(q={"type": "noul", "criteria": value}))


class TestChoiceCriteria:
    def test_two_options_are_accepted(self) -> None:
        body = _request(q={"type": "choice", "criteria": {"a": "A", "b": "B"}})
        question = _validate(body).questions[0]
        assert isinstance(question, ChoiceQuestion)
        assert question.criteria == (("a", "A"), ("b", "B"))

    def test_request_key_order_is_preserved(self) -> None:
        body = _request(q={"type": "choice", "criteria": {"z": "Z", "a": "A"}})
        assert [key for key, _ in _validate(body).questions[0].criteria] == ["z", "a"]

    def test_null_description_is_kept_as_a_candidate(self) -> None:
        # spec 5.3: a null description means "the key alone names the option".
        body = _request(q={"type": "choice", "criteria": {"a": None, "b": "B"}})
        assert _validate(body).questions[0].criteria == (("a", None), ("b", "B"))

    def test_empty_string_key_is_structurally_accepted(self) -> None:
        body = _request(q={"type": "choice", "criteria": {"": "empty", "b": "B"}})
        assert _validate(body).questions[0].criteria[0][0] == ""

    def test_two_hundred_fifty_five_options_are_accepted(self) -> None:
        criteria = {f"k{i:03d}": None for i in range(255)}
        body = _request(q={"type": "choice", "criteria": criteria})
        assert len(_validate(body).questions[0].criteria) == 255

    @pytest.mark.parametrize("count", [0, 1, 256])
    def test_option_count_outside_two_to_255_is_rejected(self, count: int) -> None:
        criteria = {f"k{i:03d}": None for i in range(count)}
        with pytest.raises(ValidationError) as excinfo:
            _validate(_request(q={"type": "choice", "criteria": criteria}))
        assert excinfo.value.path == ["questions", "q", "criteria"]

    def test_criteria_is_required(self) -> None:
        with pytest.raises(ValidationError):
            _validate(_request(q={"type": "choice"}))

    @pytest.mark.parametrize("value", [None, [], "a"])
    def test_criteria_must_be_an_object(self, value: Any) -> None:
        with pytest.raises(ValidationError):
            _validate(_request(q={"type": "choice", "criteria": value}))

    @pytest.mark.parametrize("bad", [1, True, 1.5])
    def test_scalar_description_other_than_string_is_rejected(self, bad: Any) -> None:
        body = _request(q={"type": "choice", "criteria": {"a": bad, "b": "B"}})
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["questions", "q", "criteria", "a"]


class TestScoreCriteria:
    def test_two_levels_are_accepted(self) -> None:
        body = _request(q={"type": "score", "criteria": ["low", "high"]})
        question = _validate(body).questions[0]
        assert isinstance(question, ScoreQuestion)
        assert question.criteria == ("low", "high")

    def test_level_order_is_preserved(self) -> None:
        body = _request(q={"type": "score", "criteria": ["z", "a", "m"]})
        assert _validate(body).questions[0].criteria == ("z", "a", "m")

    def test_structured_levels_are_kept(self) -> None:
        body = _request(q={"type": "score", "criteria": [{"b": 1, "a": 2}, ["x"]]})
        assert _validate(body).questions[0].criteria == ({"b": 1, "a": 2}, ["x"])

    def test_ten_levels_are_accepted(self) -> None:
        body = _request(q={"type": "score", "criteria": [str(i) for i in range(10)]})
        assert len(_validate(body).questions[0].criteria) == 10

    @pytest.mark.parametrize("count", [0, 1, 11])
    def test_level_count_outside_two_to_ten_is_rejected(self, count: int) -> None:
        body = _request(q={"type": "score", "criteria": [str(i) for i in range(count)]})
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["questions", "q", "criteria"]

    def test_null_level_is_rejected(self) -> None:
        # U02: Score levels are non-null even though Advanced docs allow null.
        body = _request(q={"type": "score", "criteria": ["low", None]})
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["questions", "q", "criteria", 1]

    @pytest.mark.parametrize("value", [None, {}, "ab"])
    def test_criteria_must_be_an_array(self, value: Any) -> None:
        with pytest.raises(ValidationError):
            _validate(_request(q={"type": "score", "criteria": value}))


class TestMixedRequests:
    def test_three_types_in_one_request(self) -> None:
        # CT01
        body = _request(
            n={"type": "noul"},
            c={"type": "choice", "criteria": {"a": "A", "b": "B"}},
            s={"type": "score", "criteria": ["low", "high"]},
        )
        questions = _validate(body).questions
        assert [type(q).__name__ for q in questions] == [
            "NoulQuestion",
            "ChoiceQuestion",
            "ScoreQuestion",
        ]

    def test_first_violation_is_reported_and_nothing_is_partially_accepted(self) -> None:
        # spec 5.9: one invalid question rejects the whole request.
        body = _request(
            ok={"type": "noul"},
            bad={"type": "score", "criteria": ["only-one"]},
        )
        with pytest.raises(ValidationError) as excinfo:
            _validate(body)
        assert excinfo.value.path == ["questions", "bad", "criteria"]


class TestStateInternalKeysAreNotSchemaChecked:
    def test_business_keys_named_like_schema_fields_are_kept(self) -> None:
        # spec 5.3: the unknown-field rule does not reach into state.
        body = _request(q={"type": "noul"})
        body["state"] = {"type": "choice", "criteria": {"weird": 1}}
        assert _validate(body).state == {"type": "choice", "criteria": {"weird": 1}}

    def test_business_keys_inside_criteria_descriptions_are_kept(self) -> None:
        body = _request(
            q={"type": "choice", "criteria": {"a": {"type": "x"}, "b": "B"}},
        )
        assert _validate(body).questions[0].criteria[0][1] == {"type": "x"}
