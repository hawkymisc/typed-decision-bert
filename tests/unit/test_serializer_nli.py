"""serializer-nli-v1 compiler (POC_DESIGN 5; spec 6.1, CT02, CT03, CT06)."""

from __future__ import annotations

from typing import Any

import pytest

from jevbert.api.errors import ContextLengthExceededError
from jevbert.backends.fake import FakeBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.serializer_nli import (
    DEFAULT_NOUL_FALSE,
    DEFAULT_NOUL_TRUE,
    compile_request,
    encode_request,
)
from jevbert.config import Limits
from jevbert.contracts.validator import validate_request

LIMITS = Limits()


def _compile(state: Any, **questions: Any) -> Any:
    body = {"model": "m", "state": state, "questions": questions}
    return compile_request(validate_request(body, LIMITS))


def _one(state: Any, question: Any) -> Any:
    return _compile(state, q=question).questions[0]


class TestPremise:
    def test_premise_is_the_rendered_state(self) -> None:
        compiled = _one("顧客からの連絡", {"type": "noul"})
        assert all(pair.premise == "顧客からの連絡" for pair in compiled.pairs)

    def test_structured_state_is_canonicalised(self) -> None:
        compiled = _one({"b": 1, "a": 2}, {"type": "noul"})
        assert compiled.pairs[0].premise == '{"a":2,"b":1}'


class TestNoul:
    def test_internal_order_is_false_then_true(self) -> None:
        compiled = _one("s", {"type": "noul"})
        assert compiled.option_keys == ("false", "true")

    def test_default_yes_no_sentences_are_used_when_criteria_are_absent(self) -> None:
        compiled = _one("s", {"type": "noul"})
        assert compiled.pairs[0].hypothesis == DEFAULT_NOUL_FALSE
        assert compiled.pairs[1].hypothesis == DEFAULT_NOUL_TRUE

    def test_one_sided_criteria_keeps_the_default_on_the_other_side(self) -> None:
        compiled = _one("s", {"type": "noul", "criteria": {"true": "返金を求めている"}})
        assert compiled.pairs[0].hypothesis == DEFAULT_NOUL_FALSE
        assert compiled.pairs[1].hypothesis == "返金を求めている"

    def test_absent_and_null_criteria_compile_identically(self) -> None:
        absent = _one("s", {"type": "noul"})
        explicit = _one("s", {"type": "noul", "criteria": None})
        empty = _one("s", {"type": "noul", "criteria": {}})
        assert absent.pairs == explicit.pairs == empty.pairs

    def test_instructions_are_combined_with_the_candidate(self) -> None:
        compiled = _one("s", {"type": "noul", "instructions": "返金要求か"})
        assert compiled.pairs[1].hypothesis == f"返金要求か — {DEFAULT_NOUL_TRUE}"

    def test_no_instructions_leaves_the_candidate_alone(self) -> None:
        compiled = _one("s", {"type": "noul"})
        assert compiled.pairs[1].hypothesis == DEFAULT_NOUL_TRUE


class TestChoice:
    def test_candidates_are_ordered_by_code_point(self) -> None:
        # spec 6.1: HTTP insertion order must not reach the model.
        compiled = _one("s", {"type": "choice", "criteria": {"z": "Z", "a": "A", "M": "m"}})
        assert compiled.option_keys == ("M", "a", "z")

    def test_key_and_description_are_both_used(self) -> None:
        compiled = _one("s", {"type": "choice", "criteria": {"billing": "請求", "b": "B"}})
        assert compiled.pairs[1].hypothesis == "billing: 請求"

    def test_null_description_uses_the_key_alone(self) -> None:
        compiled = _one("s", {"type": "choice", "criteria": {"a": None, "b": "B"}})
        assert compiled.pairs[0].hypothesis == "a"

    def test_structured_description_is_canonicalised(self) -> None:
        compiled = _one("s", {"type": "choice", "criteria": {"a": {"y": 1, "x": 2}, "b": "B"}})
        assert compiled.pairs[0].hypothesis == 'a: {"x":2,"y":1}'

    def test_wire_order_follows_the_request_not_the_model_order(self) -> None:
        compiled = _one("s", {"type": "choice", "criteria": {"z": "Z", "a": "A"}})
        assert compiled.option_keys == ("a", "z")
        assert compiled.output_keys == ("z", "a")

    def test_insertion_order_does_not_change_the_compiled_sequences(self) -> None:
        # PT04 at unit level: reordering criteria must not change model input.
        first = _one("s", {"type": "choice", "criteria": {"a": "A", "b": "B"}})
        second = _one("s", {"type": "choice", "criteria": {"b": "B", "a": "A"}})
        assert first.pairs == second.pairs


class TestScore:
    def test_levels_keep_the_array_order(self) -> None:
        compiled = _one("s", {"type": "score", "criteria": ["z", "a"]})
        assert compiled.option_keys == ("0", "1")
        assert [pair.hypothesis for pair in compiled.pairs] == ["z", "a"]

    def test_level_number_is_not_part_of_the_model_input(self) -> None:
        # POC_DESIGN 5.2 / spec 6.1: the stage index is external metadata.
        compiled = _one("s", {"type": "score", "criteria": ["low", "high"]})
        assert all("0" not in pair.hypothesis for pair in compiled.pairs)

    def test_structured_level_is_canonicalised(self) -> None:
        compiled = _one("s", {"type": "score", "criteria": [{"b": 1, "a": 2}, "x"]})
        assert compiled.pairs[0].hypothesis == '{"a":2,"b":1}'

    def test_legend_keeps_the_original_values_and_types(self) -> None:
        # I07
        compiled = _one("s", {"type": "score", "criteria": [{"b": 1}, ["x"]]})
        assert compiled.legend == ({"b": 1}, ["x"])


class TestQuestionIdIsolation:
    def test_question_id_never_appears_in_the_model_input(self) -> None:
        # F04 / spec 6.1 / PT04
        compiled = _compile(
            "s", very_unique_question_id_42={"type": "choice", "criteria": {"a": "A", "b": "B"}}
        )
        text = "".join(p.premise + p.hypothesis for p in compiled.questions[0].pairs)
        assert "very_unique_question_id_42" not in text

    def test_renaming_a_question_does_not_change_its_sequences(self) -> None:
        first = _compile("s", alpha={"type": "noul", "instructions": "i"})
        second = _compile("s", omega={"type": "noul", "instructions": "i"})
        assert first.questions[0].pairs == second.questions[0].pairs

    def test_adding_an_unrelated_question_does_not_change_the_others(self) -> None:
        first = _compile("s", a={"type": "noul", "instructions": "i"})
        second = _compile(
            "s",
            a={"type": "noul", "instructions": "i"},
            b={"type": "score", "criteria": ["x", "y"]},
        )
        assert first.questions[0].pairs == second.questions[0].pairs


class TestTokenBudget:
    def _encode(self, compiled: Any, budget: TokenBudget) -> Any:
        return encode_request(compiled, FakeBackend(), budget)

    def test_sequence_at_the_limit_is_accepted(self) -> None:
        # FakeBackend counts utf-8 bytes + 4, so the arithmetic is exact.
        compiled = _compile("x" * 10, q={"type": "noul", "criteria": {"true": "y", "false": "n"}})
        budget = TokenBudget(max_sequence_tokens=15, max_request_tokens=1000)
        encoded = self._encode(compiled, budget)
        assert encoded.total_tokens == 30

    def test_sequence_above_the_limit_is_refused(self) -> None:
        compiled = _compile("x" * 10, q={"type": "noul", "criteria": {"true": "y", "false": "n"}})
        budget = TokenBudget(max_sequence_tokens=14, max_request_tokens=1000)
        with pytest.raises(ContextLengthExceededError) as excinfo:
            self._encode(compiled, budget)
        assert excinfo.value.path == ["questions", "q"]

    def test_request_total_above_the_limit_is_refused(self) -> None:
        compiled = _compile("x" * 10, q={"type": "noul", "criteria": {"true": "y", "false": "n"}})
        budget = TokenBudget(max_sequence_tokens=1000, max_request_tokens=29)
        with pytest.raises(ContextLengthExceededError):
            self._encode(compiled, budget)

    def test_nothing_is_truncated_when_the_request_is_accepted(self) -> None:
        # CT06 / spec 6.3: overflow_policy is reject, never truncate.
        state = "y" * 100
        compiled = _compile(state, q={"type": "noul", "criteria": {"true": "a", "false": "b"}})
        encoded = self._encode(compiled, TokenBudget(1000, 10000))
        assert encoded.questions[0].sequences[0].token_count == len(state) + 1 + 4

    def test_usage_is_the_sum_over_every_candidate_sequence(self) -> None:
        # spec 5.8: expanded-input-a0-v1 counts one sequence per candidate.
        compiled = _compile("s", q={"type": "choice", "criteria": {"a": None, "b": None}})
        encoded = self._encode(compiled, TokenBudget(1000, 10000))
        assert encoded.total_tokens == sum(
            seq.token_count for q in encoded.questions for seq in q.sequences
        )
        assert len(encoded.questions[0].sequences) == 2
