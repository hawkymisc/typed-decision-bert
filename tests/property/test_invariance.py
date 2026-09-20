"""PT04: question IDs and ordering must not reach the model (spec 13.3, F04).

The fake backend's logit depends only on the compiled premise and hypothesis, so if a
question ID or an insertion order leaked into the model input, the distribution would
move and these properties would fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jevbert.compiler.serializer_nli import compile_request
from jevbert.config import Limits
from jevbert.contracts.validator import validate_request
from tests.conftest import MODEL, build_client, systemone
from tests.property.strategies import CONTENT, valid_questions

LIMITS = Limits()
TARGET = "target"

IDENTIFIERS = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=122), min_size=1, max_size=10
)


@pytest.fixture(scope="module")
def module_client() -> Iterator[TestClient]:
    with build_client() as client:
        yield client


def _compiled_pairs(body: dict[str, Any], question_id: str) -> tuple[Any, ...]:
    compiled = compile_request(validate_request(body, LIMITS))
    for question in compiled.questions:
        if question.question_id == question_id:
            return question.pairs
    raise AssertionError(f"{question_id} was not compiled")


def _body(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    return {"model": MODEL, "state": state, "questions": questions}


class TestPT04CompilerInvariance:
    @given(CONTENT, valid_questions(), IDENTIFIERS)
    def test_renaming_a_question_does_not_change_its_input(
        self, state: Any, question: dict[str, Any], new_id: str
    ) -> None:
        base = _compiled_pairs(_body(state, {TARGET: question}), TARGET)
        renamed = _compiled_pairs(_body(state, {new_id: question}), new_id)
        assert base == renamed

    @given(CONTENT, valid_questions(), valid_questions())
    def test_adding_an_unrelated_question_does_not_change_the_others(
        self, state: Any, question: dict[str, Any], other: dict[str, Any]
    ) -> None:
        base = _compiled_pairs(_body(state, {TARGET: question}), TARGET)
        with_other = _compiled_pairs(_body(state, {TARGET: question, "other": other}), TARGET)
        assert base == with_other

    @given(CONTENT, valid_questions(), valid_questions())
    def test_question_order_does_not_change_the_input(
        self, state: Any, question: dict[str, Any], other: dict[str, Any]
    ) -> None:
        first = _compiled_pairs(_body(state, {TARGET: question, "other": other}), TARGET)
        second = _compiled_pairs(_body(state, {"other": other, TARGET: question}), TARGET)
        assert first == second

    @given(
        CONTENT,
        st.dictionaries(
            st.text(max_size=4), st.none() | st.text(max_size=6), min_size=2, max_size=5
        ),
        st.randoms(use_true_random=True),
    )
    def test_choice_criteria_insertion_order_does_not_change_the_input(
        self, state: Any, criteria: dict[str, Any], random: Any
    ) -> None:
        # spec 13.3: after normalisation the input tokens must match exactly.
        items = list(criteria.items())
        random.shuffle(items)
        shuffled = dict(items)
        base = _compiled_pairs(
            _body(state, {TARGET: {"type": "choice", "criteria": criteria}}), TARGET
        )
        reordered = _compiled_pairs(
            _body(state, {TARGET: {"type": "choice", "criteria": shuffled}}), TARGET
        )
        assert base == reordered


class TestPT04EndToEndInvariance:
    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(CONTENT, valid_questions(), valid_questions(), IDENTIFIERS)
    def test_the_answer_survives_renaming_reordering_and_additions(
        self,
        module_client: TestClient,
        state: Any,
        question: dict[str, Any],
        other: dict[str, Any],
        new_id: str,
    ) -> None:
        base = systemone(module_client, _body(state, {TARGET: question})).json()
        expected = base["answers"][TARGET]

        renamed = systemone(module_client, _body(state, {new_id: question})).json()
        assert renamed["answers"][new_id] == expected

        extended = systemone(
            module_client, _body(state, {"other": other, TARGET: question})
        ).json()
        assert extended["answers"][TARGET] == expected

    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        CONTENT,
        st.dictionaries(
            st.text(max_size=4), st.none() | st.text(max_size=6), min_size=2, max_size=5
        ),
        st.randoms(use_true_random=True),
    )
    def test_choice_distribution_survives_criteria_reordering(
        self, module_client: TestClient, state: Any, criteria: dict[str, Any], random: Any
    ) -> None:
        items = list(criteria.items())
        random.shuffle(items)
        shuffled = dict(items)

        base = systemone(
            module_client, _body(state, {TARGET: {"type": "choice", "criteria": criteria}})
        ).json()["answers"][TARGET]
        reordered = systemone(
            module_client, _body(state, {TARGET: {"type": "choice", "criteria": shuffled}})
        ).json()["answers"][TARGET]

        # The wire order of `probabilities` follows the request, but the mapping and
        # the selected choice must be identical.
        assert base["probabilities"] == reordered["probabilities"]
        assert base["choice"] == reordered["choice"]
        assert base["confidence"] == reordered["confidence"]
