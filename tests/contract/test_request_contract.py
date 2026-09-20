"""CT01-CT06: request contract over HTTP (spec 13.2; POC_DESIGN 8.1)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import MODEL, request_body, systemone


def _ok(client: TestClient, questions: dict[str, Any], state: Any = "s") -> dict[str, Any]:
    response = systemone(client, request_body(questions, state))
    assert response.status_code == 200, response.text
    return response.json()


class TestCT01TypesAndCounts:
    def test_single_noul(self, client: TestClient) -> None:
        payload = _ok(client, {"q": {"type": "noul"}})
        assert payload["answers"]["q"]["type"] == "noul"
        assert 0.0 <= payload["answers"]["q"]["noul"] <= 1.0

    def test_single_choice(self, client: TestClient) -> None:
        payload = _ok(client, {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
        answer = payload["answers"]["q"]
        assert answer["choice"] in {"a", "b"}
        assert set(answer["probabilities"]) == {"a", "b"}

    def test_single_score(self, client: TestClient) -> None:
        payload = _ok(client, {"q": {"type": "score", "criteria": ["low", "mid", "high"]}})
        answer = payload["answers"]["q"]
        assert set(answer["probabilities"]) == {"0", "1", "2"}
        assert 0.0 <= answer["score"] <= 2.0

    def test_three_types_mixed(self, client: TestClient) -> None:
        payload = _ok(
            client,
            {
                "refund_requested": {"type": "noul", "instructions": "返金要求か"},
                "department": {
                    "type": "choice",
                    "criteria": {"billing": "請求", "technical": "不具合", "other": "その他"},
                },
                "urgency": {"type": "score", "criteria": ["低", "中", "高"]},
            },
        )
        assert set(payload["answers"]) == {"refund_requested", "department", "urgency"}
        assert payload["model"] == MODEL

    def test_answer_order_follows_the_request(self, client: TestClient) -> None:
        questions = {f"q{i}": {"type": "noul"} for i in range(5)}
        response = systemone(client, request_body(questions))
        ordered = json.loads(response.text, object_pairs_hook=list)
        answers = dict(ordered)["answers"]
        assert [key for key, _ in answers] == list(questions)

    def test_thirty_two_questions(self, client: TestClient) -> None:
        payload = _ok(client, {f"q{i}": {"type": "noul"} for i in range(32)})
        assert len(payload["answers"]) == 32

    def test_thirty_three_questions_are_refused(self, client: TestClient) -> None:
        response = systemone(
            client, request_body({f"q{i}": {"type": "noul"} for i in range(33)})
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestCT02StructuredValues:
    @pytest.mark.parametrize(
        "state",
        ["文字列", ["a", 1, None], {"message": {"body": "本文", "flags": [True, False]}}],
    )
    def test_state_shapes(self, client: TestClient, state: Any) -> None:
        assert _ok(client, {"q": {"type": "noul"}}, state)["answers"]["q"]["type"] == "noul"

    def test_structured_instructions_and_criteria(self, client: TestClient) -> None:
        payload = _ok(
            client,
            {
                "q": {
                    "type": "choice",
                    "instructions": {"task": "分類", "hints": ["注意"]},
                    "criteria": {"a": {"desc": "A", "weight": 1}, "b": ["B"]},
                }
            },
        )
        assert set(payload["answers"]["q"]["probabilities"]) == {"a", "b"}

    def test_legend_preserves_the_rubric_types(self, client: TestClient) -> None:
        # I07 / U03: the rubric comes back with its JSON types intact, not stringified.
        criteria = ["文字列", {"level": 1, "note": None}, [1, 2, 3]]
        payload = _ok(client, {"q": {"type": "score", "criteria": criteria}})
        legend = payload["answers"]["q"]["legend"]
        assert legend == {"0": criteria[0], "1": criteria[1], "2": criteria[2]}
        assert legend["1"]["level"] == 1
        assert isinstance(legend["2"], list)


class TestCT03OptionalFields:
    def test_choice_null_description_keeps_the_option(self, client: TestClient) -> None:
        payload = _ok(client, {"q": {"type": "choice", "criteria": {"a": None, "b": "B"}}})
        assert set(payload["answers"]["q"]["probabilities"]) == {"a", "b"}

    @pytest.mark.parametrize(
        "criteria", [None, {}, {"true": "はい"}, {"false": "いいえ"}, {"true": None}]
    )
    def test_noul_criteria_variants(self, client: TestClient, criteria: Any) -> None:
        question: dict[str, Any] = {"type": "noul"}
        if criteria is not None or True:
            question["criteria"] = criteria
        payload = _ok(client, {"q": question})
        assert "confidence" not in payload["answers"]["q"]  # I08

    def test_absent_and_null_instructions_give_the_same_answer(self, client: TestClient) -> None:
        # U01: SDK 0.7.0 omits unset fields; both forms must mean "no extra guidance".
        absent = _ok(client, {"q": {"type": "noul"}})
        explicit = _ok(client, {"q": {"type": "noul", "instructions": None}})
        assert absent["answers"]["q"] == explicit["answers"]["q"]


class TestCT04OptionCountBoundaries:
    @pytest.mark.parametrize("count", [2, 255])
    def test_accepted_choice_counts(self, client: TestClient, count: int) -> None:
        criteria = {f"k{i:03d}": None for i in range(count)}
        payload = _ok(client, {"q": {"type": "choice", "criteria": criteria}})
        assert len(payload["answers"]["q"]["probabilities"]) == count

    @pytest.mark.parametrize("count", [1, 256])
    def test_refused_choice_counts(self, client: TestClient, count: int) -> None:
        criteria = {f"k{i:03d}": None for i in range(count)}
        response = systemone(client, request_body({"q": {"type": "choice", "criteria": criteria}}))
        assert response.status_code == 422

    @pytest.mark.parametrize("count", [2, 10])
    def test_accepted_score_counts(self, client: TestClient, count: int) -> None:
        criteria = [f"level {i}" for i in range(count)]
        payload = _ok(client, {"q": {"type": "score", "criteria": criteria}})
        assert len(payload["answers"]["q"]["probabilities"]) == count

    @pytest.mark.parametrize("count", [1, 11])
    def test_refused_score_counts(self, client: TestClient, count: int) -> None:
        criteria = [f"level {i}" for i in range(count)]
        response = systemone(client, request_body({"q": {"type": "score", "criteria": criteria}}))
        assert response.status_code == 422


class TestCT05MalformedInput:
    def test_empty_questions(self, client: TestClient) -> None:
        assert systemone(client, request_body({})).status_code == 422

    def test_unknown_question_type(self, client: TestClient) -> None:
        response = systemone(client, request_body({"q": {"type": "boolean"}}))
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["questions", "q", "type"]

    def test_unknown_top_level_field(self, client: TestClient) -> None:
        body = request_body({"q": {"type": "noul"}}) | {"temperature": 0.7}
        assert systemone(client, body).status_code == 422

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
    def test_non_finite_numbers(self, client: TestClient, literal: str) -> None:
        questions = '"questions":{"q":{"type":"noul"}}'
        raw = f'{{"model":"{MODEL}","state":{{"v":{literal}}},{questions}}}'
        response = systemone(client, content=raw.encode())
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_json"

    def test_duplicate_keys(self, client: TestClient) -> None:
        raw = b'{"model":"m","model":"n","state":"s","questions":{"q":{"type":"noul"}}}'
        response = systemone(client, content=raw)
        assert response.status_code == 400

    def test_byte_order_mark(self, client: TestClient) -> None:
        body = b'{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}'
        assert systemone(client, content=b"\xef\xbb\xbf" + body).status_code == 400

    def test_lone_surrogate(self, client: TestClient) -> None:
        raw = rb'{"model":"m","state":"\ud800","questions":{"q":{"type":"noul"}}}'
        assert systemone(client, content=raw).status_code == 400

    @pytest.mark.parametrize("state", ["null", "1", "true"])
    def test_scalar_state_at_the_top_level(self, client: TestClient, state: str) -> None:
        raw = f'{{"model":"{MODEL}","state":{state},"questions":{{"q":{{"type":"noul"}}}}}}'
        response = systemone(client, content=raw.encode())
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["state"]

    def test_boolean_is_not_converted_to_a_number(self, client: TestClient) -> None:
        # A boolean where Content is required stays a boolean and is refused.
        raw = f'{{"model":"{MODEL}","state":"s","questions":{{"q":{{"type":"noul",'
        raw += '"instructions":true}}}'
        assert systemone(client, content=raw.encode()).status_code == 422

    def test_one_invalid_question_rejects_the_whole_request(self, client: TestClient) -> None:
        # spec 5.9: no partial 200.
        response = systemone(
            client,
            request_body(
                {
                    "good": {"type": "noul"},
                    "bad": {"type": "score", "criteria": ["only one"]},
                }
            ),
        )
        assert response.status_code == 422
        assert "answers" not in response.json()


class TestCT06Limits:
    def test_depth_at_the_limit(self, client: TestClient) -> None:
        nested: Any = "leaf"
        for _ in range(29):
            nested = [nested]
        # body(1) + state(1) + 29 arrays = 31 levels, inside the limit of 32.
        assert systemone(client, request_body({"q": {"type": "noul"}}, nested)).status_code == 200

    def test_depth_above_the_limit(self, client: TestClient) -> None:
        nested: Any = "leaf"
        for _ in range(40):
            nested = [nested]
        response = systemone(client, request_body({"q": {"type": "noul"}}, nested))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    @staticmethod
    def _body_of_exactly(size: int) -> bytes:
        body = request_body({"q": {"type": "noul"}}, "")
        raw = json.dumps(body, separators=(",", ":")).encode()
        body["state"] = "x" * (size - len(raw))
        raw = json.dumps(body, separators=(",", ":")).encode()
        assert len(raw) == size
        return raw

    def test_body_at_the_size_limit_passes_the_size_gate(self, client: TestClient) -> None:
        # A 2 MiB body is not too large; it is then refused by the token budget, which
        # is the next gate. Asserting "not 413" is what this boundary is about.
        response = systemone(client, content=self._body_of_exactly(2 * 1024 * 1024))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"

    def test_body_one_byte_above_the_size_limit(self, client: TestClient) -> None:
        response = systemone(client, content=self._body_of_exactly(2 * 1024 * 1024 + 1))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    def test_size_gate_runs_before_json_parsing(self, client: TestClient) -> None:
        # An oversized body must not be parsed at all, valid JSON or not.
        response = systemone(client, content=b"{" + b"x" * (2 * 1024 * 1024))
        assert response.status_code == 413

    def test_token_budget_overflow_is_refused_not_truncated(self, client: TestClient) -> None:
        # The fake backend counts utf-8 bytes + 4, so 2048 bytes of state overflows the
        # 2048-token sequence limit of the fake manifest.
        response = systemone(
            client, request_body({"q": {"type": "noul"}}, "x" * 4000)
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "context_length_exceeded"
        assert payload["error"]["path"] == ["questions", "q"]

    def test_accepted_request_counts_every_candidate_sequence(self, client: TestClient) -> None:
        state = "s"
        questions = {"q": {"type": "choice", "criteria": {"a": None, "bb": None}}}
        response = systemone(client, request_body(questions, state))
        assert response.status_code == 200
        # (1 + 1 + 4) + (1 + 2 + 4) = 13, nothing truncated.
        assert response.json()["usage"]["input_tokens"] == 13
