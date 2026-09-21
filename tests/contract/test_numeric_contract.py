"""CT07-CT09: numerics over HTTP (spec 13.2, 5.5; POC_DESIGN 8.1).

Logits are injected through the fake backend's ``logit_fn``, which is a constructor
argument: nothing reachable over HTTP can choose it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevbert.backends.base import TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.compiler.serializer_nli import DEFAULT_NOUL_FALSE, DEFAULT_NOUL_TRUE
from tests.conftest import build_client, build_registry, request_body, systemone

Produce = Callable[[Sequence[TextPair]], Sequence[float]]


def healthy_warmup(produce: Produce) -> Produce:
    """Keep warmup sound so that the bundle becomes ready, then misbehave on real input.

    Readiness is a real gate (spec 16.1): a backend that fails warmup must stay 503, so
    a fault that only appears at request time has to be modelled as exactly that.
    """

    def fn(pairs: Sequence[TextPair]) -> Sequence[float]:
        if all(pair.premise == "warmup" for pair in pairs):
            return [0.0] * len(pairs)
        return produce(pairs)

    return fn


def client_with(produce: Produce) -> TestClient:
    backend = FakeBackend(logit_fn=healthy_warmup(produce))
    return build_client(registry=build_registry(backend))


class TestCT07Distributions:
    def test_exact_tie_selects_the_smallest_key_and_zero_confidence(self) -> None:
        with client_with(lambda pairs: [1.0] * len(pairs)) as client:
            response = systemone(
                client,
                request_body({"q": {"type": "choice", "criteria": {"b": "B", "a": "A", "c": "C"}}}),
            )
        answer = response.json()["answers"]["q"]
        assert answer["choice"] == "a"
        assert answer["confidence"] == pytest.approx(0.0, abs=1e-12)
        assert list(answer["probabilities"].values()) == pytest.approx([1 / 3] * 3)

    def test_extreme_logits_stay_finite(self) -> None:
        produce = [1e4, -1e4]
        with client_with(lambda pairs: produce) as client:
            response = systemone(
                client, request_body({"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
            )
        answer = response.json()["answers"]["q"]
        assert response.status_code == 200
        assert math.isclose(sum(answer["probabilities"].values()), 1.0, abs_tol=1e-9)
        assert answer["probabilities"]["a"] == pytest.approx(1.0)
        assert answer["confidence"] == pytest.approx(1.0)

    def test_uniform_logits_give_a_uniform_distribution(self) -> None:
        with client_with(lambda pairs: [0.0] * len(pairs)) as client:
            response = systemone(
                client, request_body({"q": {"type": "score", "criteria": ["a", "b", "c", "d"]}})
            )
        answer = response.json()["answers"]["q"]
        assert all(p == pytest.approx(0.25) for p in answer["probabilities"].values())
        assert answer["score"] == pytest.approx(1.5)
        assert answer["confidence"] == pytest.approx(0.0, abs=1e-12)


class TestCT08NumericFailures:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_logits_produce_500_without_a_distribution(self, bad: float) -> None:
        with client_with(lambda pairs: [bad] + [0.0] * (len(pairs) - 1)) as client:
            response = systemone(
                client, request_body({"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
            )
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"]["code"] == "inference_error"
        # N02: no uniform distribution, no 0.5, nothing that looks like an answer.
        assert "answers" not in payload
        assert "probabilities" not in response.text

    def test_logit_count_mismatch_produces_500(self) -> None:
        with client_with(lambda pairs: [0.0] * (len(pairs) - 1)) as client:
            response = systemone(
                client, request_body({"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
            )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_error"

    def test_a_failing_question_does_not_yield_a_partial_response(self) -> None:
        def produce(pairs: Sequence[TextPair]) -> Sequence[float]:
            return [float("nan") if "poison" in p.hypothesis else 0.0 for p in pairs]

        with client_with(produce) as client:
            response = systemone(
                client,
                request_body(
                    {
                        "good": {"type": "noul"},
                        "bad": {"type": "choice", "criteria": {"poison": "x", "b": "B"}},
                    }
                ),
            )
        assert response.status_code == 500
        assert "good" not in response.text


class TestCT09TypeSemantics:
    def test_noul_direction_is_not_inverted(self) -> None:
        def produce(pairs: Sequence[TextPair]) -> Sequence[float]:
            return [3.0 if DEFAULT_NOUL_TRUE in p.hypothesis else -3.0 for p in pairs]

        with client_with(produce) as client:
            response = systemone(client, request_body({"q": {"type": "noul"}}))
        assert response.json()["answers"]["q"]["noul"] > 0.99

        def inverted(pairs: Sequence[TextPair]) -> Sequence[float]:
            return [3.0 if DEFAULT_NOUL_FALSE in p.hypothesis else -3.0 for p in pairs]

        with client_with(inverted) as client:
            response = systemone(client, request_body({"q": {"type": "noul"}}))
        assert response.json()["answers"]["q"]["noul"] < 0.01

    def test_score_is_the_expectation_of_the_returned_distribution(self) -> None:
        with client_with(lambda pairs: [0.0] * len(pairs)) as client:
            response = systemone(
                client, request_body({"q": {"type": "score", "criteria": ["a", "b", "c"]}})
            )
        answer = response.json()["answers"]["q"]
        expected = sum(int(k) * v for k, v in answer["probabilities"].items())
        assert abs(answer["score"] - expected) <= 1e-6

    def test_score_keys_are_strings_zero_to_k_minus_one(self) -> None:
        with client_with(lambda pairs: [0.0] * len(pairs)) as client:
            response = systemone(
                client, request_body({"q": {"type": "score", "criteria": ["a", "b", "c"]}})
            )
        answer = response.json()["answers"]["q"]
        assert list(answer["probabilities"]) == ["0", "1", "2"]
        assert list(answer["legend"]) == ["0", "1", "2"]

    def test_choice_probability_order_follows_the_request(self) -> None:
        with client_with(lambda pairs: [0.0] * len(pairs)) as client:
            response = systemone(
                client,
                request_body({"q": {"type": "choice", "criteria": {"z": "Z", "a": "A", "m": "M"}}}),
            )
        ordered = json.loads(response.text, object_pairs_hook=list)
        answers = dict(ordered)["answers"]
        probabilities = dict(dict(answers)["q"])["probabilities"]
        assert [key for key, _ in probabilities] == ["z", "a", "m"]


class TestWireNumberFormat:
    def test_floats_are_written_in_decimal_notation(self) -> None:
        # K5 / spec 3.4: the SDK validates strictly, so a whole-number probability must
        # not reach the wire as an integer literal.
        with client_with(lambda pairs: [1e4, -1e4]) as client:
            response = systemone(
                client, request_body({"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
            )
        compact = response.text.replace(" ", "")
        assert '"a":1.0' in compact
        assert '"b":0.0' in compact
        assert '"confidence":1.0' in compact

    def test_probabilities_are_not_rounded(self) -> None:
        with client_with(lambda pairs: [0.0] * len(pairs)) as client:
            response = systemone(
                client, request_body({"q": {"type": "score", "criteria": ["a", "b", "c"]}})
            )
        answer: dict[str, Any] = response.json()["answers"]["q"]
        assert answer["probabilities"]["0"] == pytest.approx(1 / 3, abs=1e-15)
        assert len(repr(answer["probabilities"]["0"])) > 6
