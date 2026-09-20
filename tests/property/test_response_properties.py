"""PT02: every response satisfies appendix B and I01-I09 (POC_DESIGN 8.2).

The invariants are re-derived here from the HTTP body with the appendix C functions,
independently of the server's own check, so that a defect in the checker cannot hide a
defect in the response.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from jsonschema import Draft202012Validator

from jevbert.contracts import load_schema
from jevbert.scoring.numeric import expected_score, normalized_entropy_confidence, select_choice
from tests.conftest import MODEL, build_client, systemone
from tests.property.strategies import valid_requests

RESPONSE_VALIDATOR = Draft202012Validator(load_schema("response"))

ANSWER_FIELDS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score": {"type", "score", "legend", "probabilities", "confidence"},
}


@pytest.fixture(scope="module")
def module_client() -> Iterator[TestClient]:
    with build_client() as client:
        yield client


def check_invariants(body: dict[str, Any], payload: dict[str, Any]) -> None:
    questions = body["questions"]
    # I01
    assert set(payload["answers"]) == set(questions)

    for question_id, answer in payload["answers"].items():
        question = questions[question_id]
        # I02
        assert answer["type"] == question["type"]
        # I09
        assert set(answer) == ANSWER_FIELDS[question["type"]]

        if question["type"] == "noul":
            # I03, I08
            assert isinstance(answer["noul"], float)
            assert 0.0 <= answer["noul"] <= 1.0
            continue

        probabilities = answer["probabilities"]
        # I03
        assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in probabilities.values())
        assert 0.0 <= answer["confidence"] <= 1.0
        # I04
        assert abs(math.fsum(probabilities.values()) - 1.0) <= 1e-6
        assert answer["confidence"] == pytest.approx(
            normalized_entropy_confidence(list(probabilities.values())), abs=1e-12
        )

        if question["type"] == "choice":
            # I05
            assert set(probabilities) == set(question["criteria"])
            assert answer["choice"] == select_choice(probabilities)
        else:
            levels = question["criteria"]
            assert list(probabilities) == [str(i) for i in range(len(levels))]
            # I06
            assert abs(answer["score"] - expected_score(list(probabilities.values()))) <= 1e-6
            # I07
            assert answer["legend"] == {str(i): level for i, level in enumerate(levels)}

    assert payload["usage"]["output_tokens"] == 0
    assert payload["usage"]["input_tokens"] >= 0
    assert payload["model"] == MODEL


class TestPT02Responses:
    @given(valid_requests(MODEL))
    @settings(
        max_examples=120,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_requests_produce_conforming_responses(
        self, module_client: TestClient, body: dict[str, Any]
    ) -> None:
        response = systemone(module_client, body)
        assert response.status_code == 200, response.text
        payload = response.json()
        RESPONSE_VALIDATOR.validate(payload)
        check_invariants(body, payload)

    @given(valid_requests(MODEL))
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_responses_are_deterministic(
        self, module_client: TestClient, body: dict[str, Any]
    ) -> None:
        first = systemone(module_client, body).json()
        second = systemone(module_client, body).json()
        assert first == second
