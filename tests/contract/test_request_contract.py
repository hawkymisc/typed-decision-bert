"""CT01-CT06: request contract over HTTP (spec 13.2; POC_DESIGN 8.1)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevbert.backends.fake import FakeBackend
from jevbert.compiler.serializer_nli import compile_request
from jevbert.config import Limits
from jevbert.contracts.validator import validate_request
from jevbert.inference.registry import Bundle, BundleLimits, ModelRegistry, read_manifest
from tests.conftest import (
    FAKE_MANIFEST,
    MODEL,
    build_client,
    build_registry,
    request_body,
    systemone,
)

#: Distinguishes "the key is absent" from "the key is present and null"; the two are
#: treated alike by the contract but only one of them is what SDK 0.7.0 sends (U01).
OMITTED = object()


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
        "criteria",
        [OMITTED, None, {}, {"true": "はい"}, {"false": "いいえ"}, {"true": None}],
        ids=["omitted", "null", "empty", "true-only", "false-only", "true-null"],
    )
    def test_noul_criteria_variants(self, client: TestClient, criteria: Any) -> None:
        # The omitted case used to be unreachable: `if criteria is not None or True`
        # is always true, so `criteria: null` was sent six times and the "no criteria
        # key at all" shape - the one SDK 0.7.0 actually produces - was never tested.
        question: dict[str, Any] = {"type": "noul"}
        if criteria is not OMITTED:
            question["criteria"] = criteria
        payload = _ok(client, {"q": question})
        assert "confidence" not in payload["answers"]["q"]  # I08
        assert set(payload["answers"]["q"]) == {"type", "noul"}

    def test_omitted_and_null_criteria_give_the_same_answer(
        self, client: TestClient
    ) -> None:
        omitted = _ok(client, {"q": {"type": "noul"}})
        explicit = _ok(client, {"q": {"type": "noul", "criteria": None}})
        empty = _ok(client, {"q": {"type": "noul", "criteria": {}}})
        assert omitted["answers"] == explicit["answers"] == empty["answers"]

    def test_a_one_sided_criterion_changes_only_that_side(
        self, client: TestClient
    ) -> None:
        # The default sentence stands in for the missing side (spec 5.4), so naming
        # one side must move the answer rather than be ignored.
        default = _ok(client, {"q": {"type": "noul"}})["answers"]["q"]["noul"]
        one_sided = _ok(
            client, {"q": {"type": "noul", "criteria": {"true": "返金を求めている"}}}
        )["answers"]["q"]["noul"]
        assert one_sided != default

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
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["questions", "q", "criteria"]

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
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["questions", "q", "criteria"]

    def test_a_null_score_level_is_refused_at_its_index(self, client: TestClient) -> None:
        # U02: the Advanced docs allow a null level; this server does not, and says
        # which level it objected to.
        criteria = ["low", None, "high"]
        response = systemone(client, request_body({"q": {"type": "score", "criteria": criteria}}))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["questions", "q", "criteria", 1]


class TestCT05MalformedInput:
    def test_empty_questions(self, client: TestClient) -> None:
        response = systemone(client, request_body({}))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["questions"]

    def test_unknown_question_type(self, client: TestClient) -> None:
        response = systemone(client, request_body({"q": {"type": "boolean"}}))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["questions", "q", "type"]

    def test_unknown_top_level_field(self, client: TestClient) -> None:
        body = request_body({"q": {"type": "noul"}}) | {"temperature": 0.7}
        response = systemone(client, body)
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert payload["error"]["path"] == ["temperature"]

    def test_unknown_question_field(self, client: TestClient) -> None:
        body = request_body({"q": {"type": "noul", "threshold": 0.5}})
        response = systemone(client, body)
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["questions", "q", "threshold"]

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
    @staticmethod
    def _body_nested(levels: int) -> dict[str, Any]:
        """A body whose deepest point is exactly ``levels``.

        The body object is level 1 and each array around the state adds one, so
        ``levels`` needs ``levels - 1`` arrays. The questions branch is only three
        levels deep and never competes.
        """
        nested: Any = "leaf"
        for _ in range(levels - 1):
            nested = [nested]
        return request_body({"q": {"type": "noul"}}, nested)

    def test_depth_exactly_at_the_limit(self, client: TestClient) -> None:
        assert systemone(client, self._body_nested(32)).status_code == 200

    def test_depth_one_past_the_limit(self, client: TestClient) -> None:
        response = systemone(client, self._body_nested(33))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_depth_far_past_the_limit(self, client: TestClient) -> None:
        # Well past Python's recursion limit, and assembled as raw bytes because any
        # recursive encoder would give up first. The depth is counted on the raw text
        # before the decoder ever sees it, which is the whole point (POC_DESIGN 4.1).
        depth = 5_000
        raw = (
            f'{{"model":"{MODEL}","state":'.encode()
            + b"[" * depth
            + b'"leaf"'
            + b"]" * depth
            + b',"questions":{"q":{"type":"noul"}}}'
        )
        response = systemone(client, content=raw)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_sequence_tokens_exactly_at_the_limit(self, client: TestClient) -> None:
        # The fake backend counts UTF-8 bytes plus four special tokens, and the longer
        # of the two Noul defaults ("...is yes.") is 34 bytes: 2010 + 34 + 4 is exactly
        # the 2048 the manifest declares.
        response = systemone(client, request_body({"q": {"type": "noul"}}, "x" * 2010))
        assert response.status_code == 200, response.text

    def test_sequence_tokens_one_past_the_limit(self, client: TestClient) -> None:
        response = systemone(client, request_body({"q": {"type": "noul"}}, "x" * 2011))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "context_length_exceeded"
        assert payload["error"]["path"] == ["questions", "q"]

    def test_request_tokens_exactly_at_the_limit(self) -> None:
        # 10 characters of state: (10+33+4) + (10+34+4) = 95 tokens across both
        # candidates, which is what a request budget of 95 must still accept.
        with _client_with_budget(max_request_tokens=95) as client:
            response = systemone(client, request_body({"q": {"type": "noul"}}, "x" * 10))
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] == 95

    def test_request_tokens_one_past_the_limit(self) -> None:
        with _client_with_budget(max_request_tokens=94) as client:
            response = systemone(client, request_body({"q": {"type": "noul"}}, "x" * 10))
            assert response.status_code == 422
            payload = response.json()
            assert payload["error"]["code"] == "context_length_exceeded"
            # The request as a whole is too long, so no single question is named.
            assert "path" not in payload["error"]

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


class TestCandidateKeyOrdering:
    """Code point order for the model, request order for the wire (POC_DESIGN 5.2)."""

    KEYS = ("😀", "z", "ア", "あ")

    def test_the_model_sees_candidates_in_code_point_order(self) -> None:
        compiled = _compile(dict.fromkeys(self.KEYS))
        assert compiled.option_keys == tuple(sorted(self.KEYS))
        assert compiled.option_keys == ("z", "あ", "ア", "😀")
        # Sorting is by code point, not by any locale or normalisation rule.
        assert [ord(key[0]) for key in compiled.option_keys] == sorted(
            ord(key[0]) for key in self.KEYS
        )

    def test_the_wire_keeps_the_request_order(self, client: TestClient) -> None:
        criteria = dict.fromkeys(self.KEYS)
        response = systemone(
            client, request_body({"q": {"type": "choice", "criteria": criteria}})
        )
        ordered = json.loads(response.text, object_pairs_hook=list)
        answers = dict(dict(ordered)["answers"])
        probabilities = dict(answers["q"])["probabilities"]
        assert [key for key, _ in probabilities] == list(self.KEYS)

    def test_a_tie_resolves_to_the_smallest_key(self) -> None:
        # Equal logits for every candidate leave the tie-break as the only thing that
        # can decide the answer (spec 5.4, CT07).
        criteria = dict.fromkeys(self.KEYS)
        backend = FakeBackend(logit_fn=lambda pairs: [1.0] * len(pairs))
        registry = build_registry(backend)
        with build_client(registry=registry) as client:
            payload = systemone(
                client, request_body({"q": {"type": "choice", "criteria": criteria}})
            ).json()
        answer = payload["answers"]["q"]
        assert set(answer["probabilities"].values()) == {0.25}
        assert answer["choice"] == min(self.KEYS) == "z"
        assert answer["confidence"] == 0.0

    def test_nfc_and_nfd_are_different_candidates(self, client: TestClient) -> None:
        # No Unicode normalisation anywhere (spec 6.1): two spellings of the same
        # glyph are two options, kept apart and both answered for.
        nfc, nfd = "が", "が"
        assert nfc != nfd and len(nfc) != len(nfd)
        criteria = {nfc: None, nfd: None}
        response = systemone(
            client, request_body({"q": {"type": "choice", "criteria": criteria}})
        )
        assert response.status_code == 200
        probabilities = response.json()["answers"]["q"]["probabilities"]
        assert set(probabilities) == {nfc, nfd}

    def test_nfc_and_nfd_reach_the_model_as_written(self) -> None:
        nfc, nfd = "が", "が"
        compiled = _compile({nfc: None, nfd: None})
        hypotheses = [pair.hypothesis for pair in compiled.pairs]
        assert sorted(hypotheses) == sorted([nfc, nfd])

    def test_an_empty_key_keeps_its_place_in_the_order(self) -> None:
        compiled = _compile({"": None, "a": None})
        assert compiled.option_keys == ("", "a")


def _compile(criteria: dict[str, Any]) -> Any:
    body = {
        "model": MODEL,
        "state": "s",
        "questions": {"q": {"type": "choice", "criteria": criteria}},
    }
    return compile_request(validate_request(body, Limits())).questions[0]


@contextmanager
def _client_with_budget(*, max_request_tokens: int) -> Iterator[TestClient]:
    """A client whose bundle declares a deliberately tiny token budget."""
    manifest, digest = read_manifest(FAKE_MANIFEST)
    narrowed = manifest.model_copy(
        update={
            "limits": BundleLimits(
                max_sequence_tokens=2048, max_request_tokens=max_request_tokens
            )
        }
    )
    registry = ModelRegistry([Bundle(narrowed, digest, FakeBackend())])
    registry.load_all()
    with build_client(registry=registry) as client:
        yield client
