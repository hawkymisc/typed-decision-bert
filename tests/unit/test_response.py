"""Q-H1: ``verify_invariants`` is exercised branch by branch (spec 5.5, I01-I09).

Every other test in this suite feeds the checker bodies the server built itself, which
are correct by construction: replacing ``verify_invariants`` with ``pass`` left all of
them green. The bodies here are hand-assembled and each one is wrong in exactly one
way, so a hole in the checker shows up as a test that stops failing.

The checker is the last thing between a numeric anomaly and a caller who cannot tell
one (spec 4.3, N02): a response that looks like a confident answer but is not derived
from the distribution is worse than a 500.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from jevbert.api.errors import InferenceError
from jevbert.compiler.compiled import CompiledQuestion, EncodedQuestion, EncodedRequest
from jevbert.contracts.response import build_response, verify_invariants
from jevbert.scoring.numeric import (
    NOUL_OPTION_ORDER,
    expected_score,
    normalized_entropy_confidence,
)

MODEL = "jevbert-fake-0.0.0"
TOTAL_TOKENS = 42


def noul_question(question_id: str = "q") -> CompiledQuestion:
    return CompiledQuestion(
        question_id=question_id,
        question_type="noul",
        option_keys=NOUL_OPTION_ORDER,
        output_keys=NOUL_OPTION_ORDER,
        pairs=(),
        legend=None,
    )


def choice_question(
    question_id: str = "q", keys: tuple[str, ...] = ("b", "a")
) -> CompiledQuestion:
    return CompiledQuestion(
        question_id=question_id,
        question_type="choice",
        option_keys=tuple(sorted(keys)),
        # Wire order follows the request, which is deliberately not sorted here.
        output_keys=keys,
        pairs=(),
        legend=None,
    )


def score_question(
    question_id: str = "q", legend: tuple[Any, ...] = ("low", "high")
) -> CompiledQuestion:
    keys = tuple(str(index) for index in range(len(legend)))
    return CompiledQuestion(
        question_id=question_id,
        question_type="score",
        option_keys=keys,
        output_keys=keys,
        pairs=(),
        legend=legend,
    )


def encoded(*questions: CompiledQuestion, total_tokens: int = TOTAL_TOKENS) -> EncodedRequest:
    return EncodedRequest(
        questions=tuple(
            EncodedQuestion(compiled=question, sequences=()) for question in questions
        ),
        total_tokens=total_tokens,
    )


def body(answers: dict[str, Any], *, total_tokens: int = TOTAL_TOKENS) -> dict[str, Any]:
    return {
        "model": MODEL,
        "answers": answers,
        "usage": {"input_tokens": total_tokens, "output_tokens": 0},
    }


def choice_answer(
    probabilities: dict[str, float], choice: str | None = None
) -> dict[str, Any]:
    values = list(probabilities.values())
    best = max(values)
    return {
        "type": "choice",
        "choice": choice
        or min(key for key, value in probabilities.items() if value == best),
        "probabilities": dict(probabilities),
        "confidence": normalized_entropy_confidence(values),
    }


def score_answer(
    probabilities: list[float], legend: tuple[Any, ...] = ("low", "high")
) -> dict[str, Any]:
    return {
        "type": "score",
        "score": expected_score(probabilities),
        "legend": {str(index): value for index, value in enumerate(legend)},
        "probabilities": {
            str(index): value for index, value in enumerate(probabilities)
        },
        "confidence": normalized_entropy_confidence(probabilities),
    }


def expect_rejected(payload: dict[str, Any], request: EncodedRequest) -> None:
    with pytest.raises(InferenceError):
        verify_invariants(payload, request)


class TestTheBaselineIsAccepted:
    """Without this, every assertion below could be passing for the wrong reason."""

    def test_a_correct_noul_body_passes(self) -> None:
        verify_invariants(body({"q": {"type": "noul", "noul": 0.25}}), encoded(noul_question()))

    def test_a_correct_choice_body_passes(self) -> None:
        payload = body({"q": choice_answer({"b": 0.75, "a": 0.25})})
        verify_invariants(payload, encoded(choice_question()))

    def test_a_correct_score_body_passes(self) -> None:
        payload = body({"q": score_answer([0.25, 0.75])})
        verify_invariants(payload, encoded(score_question()))

    def test_a_correct_mixed_body_passes(self) -> None:
        payload = body(
            {
                "n": {"type": "noul", "noul": 0.5},
                "c": choice_answer({"b": 0.5, "a": 0.5}),
                "s": score_answer([0.5, 0.5]),
            }
        )
        verify_invariants(
            payload,
            encoded(noul_question("n"), choice_question("c"), score_question("s")),
        )


class TestI01QuestionIdentity:
    def test_a_missing_answer_is_rejected(self) -> None:
        expect_rejected(body({}), encoded(noul_question()))

    def test_an_extra_answer_is_rejected(self) -> None:
        answers = {
            "q": {"type": "noul", "noul": 0.5},
            "invented": {"type": "noul", "noul": 0.5},
        }
        expect_rejected(body(answers), encoded(noul_question()))

    def test_a_renamed_answer_is_rejected(self) -> None:
        expect_rejected(
            body({"other": {"type": "noul", "noul": 0.5}}), encoded(noul_question())
        )


class TestI02AnswerType:
    def test_a_noul_question_answered_as_choice_is_rejected(self) -> None:
        expect_rejected(
            body({"q": choice_answer({"b": 0.5, "a": 0.5})}), encoded(noul_question())
        )

    def test_a_choice_question_answered_as_score_is_rejected(self) -> None:
        expect_rejected(body({"q": score_answer([0.5, 0.5])}), encoded(choice_question()))

    def test_an_unknown_type_label_is_rejected(self) -> None:
        answer = {"type": "boolean", "noul": 0.5}
        expect_rejected(body({"q": answer}), encoded(noul_question()))


class TestI09NothingUnrequested:
    def test_a_reasoning_field_on_an_answer_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5}) | {"reasoning": "because"}
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_reasoning_field_at_the_top_level_is_rejected(self) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}}) | {"reasoning": "because"}
        expect_rejected(payload, encoded(noul_question()))

    @pytest.mark.parametrize("field", ["explanation", "routing", "debug", "citations"])
    def test_other_invented_top_level_fields_are_rejected(self, field: str) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}}) | {field: "x"}
        expect_rejected(payload, encoded(noul_question()))

    def test_a_missing_usage_block_is_rejected(self) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}})
        del payload["usage"]
        expect_rejected(payload, encoded(noul_question()))

    def test_an_extra_usage_counter_is_rejected(self) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}})
        payload["usage"]["cached_tokens"] = 0
        expect_rejected(payload, encoded(noul_question()))

    def test_a_missing_answer_field_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5})
        del answer["confidence"]
        expect_rejected(body({"q": answer}), encoded(choice_question()))


class TestTheModelIsNamed:
    @pytest.mark.parametrize("model", ["", None, 1, ["jevbert-fake-0.0.0"]])
    def test_a_body_without_a_usable_model_name_is_rejected(self, model: Any) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}})
        payload["model"] = model
        expect_rejected(payload, encoded(noul_question()))


class TestI03Finiteness:
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_noul_is_rejected(self, value: float) -> None:
        expect_rejected(body({"q": {"type": "noul", "noul": value}}), encoded(noul_question()))

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_a_noul_outside_zero_and_one_is_rejected(self, value: float) -> None:
        expect_rejected(body({"q": {"type": "noul", "noul": value}}), encoded(noul_question()))

    def test_an_integer_noul_is_rejected(self) -> None:
        # The SDK declares a float; an int on the wire is a contract drift, not a value.
        expect_rejected(body({"q": {"type": "noul", "noul": 1}}), encoded(noul_question()))

    def test_a_boolean_noul_is_rejected(self) -> None:
        expect_rejected(body({"q": {"type": "noul", "noul": True}}), encoded(noul_question()))

    def test_a_non_finite_probability_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5})
        answer["probabilities"]["a"] = float("nan")
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_an_integer_probability_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5})
        answer["probabilities"] = {"b": 1, "a": 0}
        answer["choice"] = "b"
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_non_finite_confidence_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5})
        answer["confidence"] = float("nan")
        expect_rejected(body({"q": answer}), encoded(choice_question()))


class TestI04Normalisation:
    @pytest.mark.parametrize("total", [0.9, 1.1, 0.999])
    def test_a_distribution_that_does_not_sum_to_one_is_rejected(self, total: float) -> None:
        # Assembled by hand: the confidence helper refuses an unnormalised input, and
        # the point here is that the checker refuses it too rather than trusting it.
        answer = {
            "type": "choice",
            "choice": "a",
            "probabilities": {"b": total / 2, "a": total / 2},
            "confidence": 0.0,
        }
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_distribution_inside_the_tolerance_is_accepted(self) -> None:
        answer = choice_answer({"b": 0.5, "a": 0.5 + 1e-9})
        verify_invariants(body({"q": answer}), encoded(choice_question()))


class TestI05ChoiceIsTheArgmax:
    def test_a_choice_that_is_not_the_argmax_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.75, "a": 0.25}, choice="a")
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_the_wrong_side_of_a_tie_is_rejected(self) -> None:
        # Ties resolve to the smallest key by code point (spec 5.4).
        answer = choice_answer({"b": 0.5, "a": 0.5}, choice="b")
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_choice_outside_the_candidate_keys_is_rejected(self) -> None:
        answer = choice_answer({"b": 0.75, "a": 0.25}, choice="c")
        expect_rejected(body({"q": answer}), encoded(choice_question()))


class TestI06ScoreIsTheExpectation:
    def test_a_score_off_by_a_thousandth_is_rejected(self) -> None:
        answer = score_answer([0.25, 0.75])
        answer["score"] += 1e-3
        expect_rejected(body({"q": answer}), encoded(score_question()))

    def test_a_score_inside_the_tolerance_is_accepted(self) -> None:
        answer = score_answer([0.25, 0.75])
        answer["score"] += 1e-9
        verify_invariants(body({"q": answer}), encoded(score_question()))

    def test_a_nan_score_is_rejected(self) -> None:
        # A NaN compares false against every tolerance, so an expectation check alone
        # would let it through as "not different enough to notice".
        answer = score_answer([0.25, 0.75])
        answer["score"] = float("nan")
        expect_rejected(body({"q": answer}), encoded(score_question()))

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_an_infinite_score_is_rejected(self, value: float) -> None:
        answer = score_answer([0.25, 0.75])
        answer["score"] = value
        expect_rejected(body({"q": answer}), encoded(score_question()))

    @pytest.mark.parametrize("value", [-0.5, 1.5])
    def test_a_score_outside_zero_and_k_minus_one_is_rejected(self, value: float) -> None:
        # Both of these are finite and wrong; the expectation check happens to catch
        # them too, but the range is what the contract states.
        answer = score_answer([0.5, 0.5])
        answer["score"] = value
        expect_rejected(body({"q": answer}), encoded(score_question()))

    def test_an_integer_score_is_rejected(self) -> None:
        answer = score_answer([1.0, 0.0])
        answer["score"] = 0
        expect_rejected(body({"q": answer}), encoded(score_question()))

    def test_a_whole_number_score_stays_a_float(self) -> None:
        answer = score_answer([1.0, 0.0])
        assert answer["score"] == 0.0
        assert isinstance(answer["score"], float)
        verify_invariants(body({"q": answer}), encoded(score_question()))


class TestConfidenceIsDerivedFromTheDistribution:
    """A-F2: a confidence nobody recomputed is a number the caller cannot rely on."""

    @pytest.mark.parametrize("wrong", [0.0, 1.0, 0.5])
    def test_a_choice_confidence_that_does_not_match_is_rejected(self, wrong: float) -> None:
        answer = choice_answer({"b": 0.75, "a": 0.25})
        assert not math.isclose(answer["confidence"], wrong)
        answer["confidence"] = wrong
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_score_confidence_that_does_not_match_is_rejected(self) -> None:
        answer = score_answer([0.25, 0.75])
        answer["confidence"] = answer["confidence"] / 2 + 0.25
        expect_rejected(body({"q": answer}), encoded(score_question()))

    def test_a_confidence_taken_from_a_different_distribution_is_rejected(self) -> None:
        # The failure mode this guards: confidence computed once and reused, or taken
        # from a backend that reports its own (spec 12.1, probability adapter).
        answer = choice_answer({"b": 0.9, "a": 0.1})
        answer["confidence"] = normalized_entropy_confidence([0.5, 0.5])
        expect_rejected(body({"q": answer}), encoded(choice_question()))

    def test_a_negligible_rounding_difference_is_accepted(self) -> None:
        answer = choice_answer({"b": 0.75, "a": 0.25})
        answer["confidence"] += 1e-12
        verify_invariants(body({"q": answer}), encoded(choice_question()))


class TestI07LegendPreservesTheRubric:
    def test_a_boolean_level_is_not_satisfied_by_one(self) -> None:
        question = score_question(legend=(True, "high"))
        answer = score_answer([0.5, 0.5], legend=(1, "high"))
        expect_rejected(body({"q": answer}), encoded(question))

    def test_a_boolean_level_round_trips(self) -> None:
        question = score_question(legend=(True, "high"))
        answer = score_answer([0.5, 0.5], legend=(True, "high"))
        verify_invariants(body({"q": answer}), encoded(question))

    def test_a_stringified_structured_level_is_rejected(self) -> None:
        question = score_question(legend=({"label": "low"}, "high"))
        answer = score_answer([0.5, 0.5], legend=('{"label":"low"}', "high"))
        expect_rejected(body({"q": answer}), encoded(question))

    def test_a_reordered_legend_is_rejected(self) -> None:
        question = score_question(legend=("low", "high"))
        answer = score_answer([0.5, 0.5], legend=("high", "low"))
        expect_rejected(body({"q": answer}), encoded(question))

    def test_a_legend_with_the_wrong_keys_is_rejected(self) -> None:
        answer = score_answer([0.5, 0.5])
        answer["legend"] = {"1": "high", "2": "low"}
        expect_rejected(body({"q": answer}), encoded(score_question()))


class TestI08NoulCarriesNoConfidence:
    def test_a_noul_answer_with_a_confidence_is_rejected(self) -> None:
        answer = {"type": "noul", "noul": 0.5, "confidence": 0.9}
        expect_rejected(body({"q": answer}), encoded(noul_question()))

    def test_a_noul_answer_with_probabilities_is_rejected(self) -> None:
        answer = {"type": "noul", "noul": 0.5, "probabilities": {"true": 0.5, "false": 0.5}}
        expect_rejected(body({"q": answer}), encoded(noul_question()))


class TestCandidateKeysAndTheirOrder:
    def test_probability_keys_that_are_not_in_request_order_are_rejected(self) -> None:
        # The wire order is the caller's order, not the code point order the model saw
        # (spec 5.5). Sorting on the way out would silently reorder every response.
        question = choice_question(keys=("b", "a"))
        answer = choice_answer({"a": 0.25, "b": 0.75})
        assert list(answer["probabilities"]) == ["a", "b"]
        expect_rejected(body({"q": answer}), encoded(question))

    def test_a_missing_candidate_is_rejected(self) -> None:
        question = choice_question(keys=("b", "a", "c"))
        answer = choice_answer({"b": 0.5, "a": 0.5})
        expect_rejected(body({"q": answer}), encoded(question))

    def test_an_invented_candidate_is_rejected(self) -> None:
        question = choice_question(keys=("b", "a"))
        answer = choice_answer({"b": 0.4, "a": 0.3, "c": 0.3})
        expect_rejected(body({"q": answer}), encoded(question))

    def test_score_levels_out_of_order_are_rejected(self) -> None:
        answer = score_answer([0.25, 0.75])
        answer["probabilities"] = {"1": 0.75, "0": 0.25}
        expect_rejected(body({"q": answer}), encoded(score_question()))


class TestUsageMatchesTheCompiledRequest:
    @pytest.mark.parametrize("reported", [TOTAL_TOKENS - 1, TOTAL_TOKENS + 1, 0])
    def test_an_input_token_count_that_does_not_match_is_rejected(
        self, reported: int
    ) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}}, total_tokens=reported)
        expect_rejected(payload, encoded(noul_question(), total_tokens=TOTAL_TOKENS))

    def test_a_non_zero_output_token_count_is_rejected(self) -> None:
        payload = body({"q": {"type": "noul", "noul": 0.5}})
        payload["usage"]["output_tokens"] = 7
        expect_rejected(payload, encoded(noul_question()))


class TestBuildResponseRejectsBrokenLogits:
    """The same guarantees, reached the way the server reaches them."""

    def test_a_nan_logit_never_becomes_an_answer(self) -> None:
        with pytest.raises(InferenceError):
            build_response(
                model_id=MODEL,
                encoded=encoded(choice_question()),
                grouped_logits=[[float("nan"), 0.0]],
                temperatures={"choice": 1.0},
            )

    def test_a_logit_count_mismatch_never_becomes_an_answer(self) -> None:
        with pytest.raises(InferenceError):
            build_response(
                model_id=MODEL,
                encoded=encoded(choice_question()),
                grouped_logits=[[0.0]],
                temperatures={"choice": 1.0},
            )

    def test_a_question_count_mismatch_never_becomes_an_answer(self) -> None:
        with pytest.raises(InferenceError):
            build_response(
                model_id=MODEL,
                encoded=encoded(choice_question("a"), choice_question("b")),
                grouped_logits=[[0.0, 1.0]],
                temperatures={"choice": 1.0},
            )

    def test_a_well_formed_result_is_built_and_verified(self) -> None:
        payload = build_response(
            model_id=MODEL,
            encoded=encoded(choice_question()),
            grouped_logits=[[1.0, 0.0]],
            temperatures={"choice": 1.0},
        )
        assert list(payload["answers"]["q"]["probabilities"]) == ["b", "a"]
        assert payload["usage"] == {"input_tokens": TOTAL_TOKENS, "output_tokens": 0}
