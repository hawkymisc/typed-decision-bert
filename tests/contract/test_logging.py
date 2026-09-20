"""Q-H3 / N03 / A-F1: what the access log does and does not contain (spec 15.2).

N03 ("no raw input or credentials in the standard log") was asserted by nothing before
phase 1.5. These tests plant a unique sentinel in every field a caller controls - state,
instructions, criteria, option keys, question IDs, the credential itself - drive each
response path, and then read back everything every logger emitted.

Two properties are checked on every path: the access line parses as JSON and carries the
full set of fields, and no sentinel appears anywhere in any record. The first is what
makes the log usable; the second is what makes it safe to keep.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from jevbert.api.middleware import LOG_FIELDS
from jevbert.backends.base import WARMUP_PREMISE
from jevbert.backends.fake import FakeBackend
from tests.conftest import API_KEY, build_client, build_registry, request_body, systemone

#: One unmistakable marker per caller-controlled field. Anything that reaches a log
#: line brings its sentinel with it, so a leak names its own source.
SENTINELS = {
    "state": "CANARY-STATE-8f21",
    "instructions": "CANARY-INSTRUCTIONS-3b7c",
    "criteria": "CANARY-CRITERIA-c1d9",
    "option_key": "CANARY-OPTION-KEY-77aa",
    "question_id": "CANARY_QUESTION_ID_5e04",
    "legend": "CANARY-LEGEND-90fe",
}

#: The full record shape: the header fields the middleware always writes, plus the
#: per-stage fields that are present as null until a stage fills them in.
REQUIRED_FIELDS = {"request_id", "method", "path", "status", "total_ms", *LOG_FIELDS}


def sentinel_body() -> dict[str, Any]:
    question_id = SENTINELS["question_id"]
    return request_body(
        {
            question_id: {
                "type": "choice",
                "instructions": SENTINELS["instructions"],
                "criteria": {
                    SENTINELS["option_key"]: SENTINELS["criteria"],
                    "plain": None,
                },
            },
            f"{question_id}_score": {
                "type": "score",
                "criteria": [SENTINELS["legend"], "other"],
            },
        },
        SENTINELS["state"],
    )


@pytest.fixture
def captured(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture every logger, not only jevbert's: uvicorn's count too (spec 15.2)."""
    caplog.set_level(logging.DEBUG)
    return caplog


def access_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    lines = []
    for record in caplog.records:
        if record.name != "jevbert.access":
            continue
        lines.append(json.loads(record.getMessage()))
    return lines


def all_text(caplog: pytest.LogCaptureFixture) -> str:
    """Everything every logger produced, message and arguments alike."""
    parts: list[str] = []
    for record in caplog.records:
        parts.append(record.name)
        parts.append(record.getMessage())
        if record.exc_info is not None:
            parts.append(repr(record.exc_info[1]))
        if record.exc_text:
            parts.append(record.exc_text)
    return "\n".join(parts)


def assert_no_sentinels(caplog: pytest.LogCaptureFixture) -> None:
    text = all_text(caplog)
    for name, sentinel in SENTINELS.items():
        assert sentinel not in text, f"the {name} sentinel reached a log line"
    assert API_KEY not in text
    assert "Bearer" not in text


def assert_record_shape(record: dict[str, Any]) -> None:
    assert set(record) == REQUIRED_FIELDS
    assert record["request_id"]
    assert isinstance(record["status"], int)
    assert isinstance(record["total_ms"], float)


class TestSuccessPath:
    def test_the_access_line_is_json_with_every_field(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with build_client() as client:
            assert systemone(client, sentinel_body()).status_code == 200

        [record] = access_lines(captured)
        assert_record_shape(record)
        assert record["status"] == 200
        assert record["error_code"] is None
        assert record["bundle"].startswith("sha256:")
        # Counts per type, never the question IDs themselves (spec 15.2).
        assert record["questions"] == {"noul": 0, "choice": 1, "score": 1}
        assert record["sequences"] == 4
        assert record["input_tokens"] > 0
        for stage in ("parse_ms", "compile_ms", "inference_ms"):
            assert isinstance(record[stage], float)

    def test_no_caller_data_reaches_any_logger(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with build_client() as client:
            assert systemone(client, sentinel_body()).status_code == 200
        assert_no_sentinels(captured)


class TestValidationFailurePath:
    def test_a_422_is_logged_with_its_error_code(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        body = sentinel_body()
        body["questions"][SENTINELS["question_id"]]["criteria"] = {
            SENTINELS["option_key"]: SENTINELS["criteria"]
        }  # one option: below the minimum
        with build_client() as client:
            assert systemone(client, body).status_code == 422

        [record] = access_lines(captured)
        assert_record_shape(record)
        assert record["status"] == 422
        assert record["error_code"] == "validation_error"
        # Nothing downstream of validation ran, so those fields stay null.
        assert record["sequences"] is None
        assert record["input_tokens"] is None

    def test_the_rejected_values_do_not_reach_any_logger(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        body = sentinel_body()
        body["questions"][SENTINELS["question_id"]]["criteria"] = {
            SENTINELS["option_key"]: SENTINELS["criteria"]
        }
        with build_client() as client:
            assert systemone(client, body).status_code == 422
        assert_no_sentinels(captured)

    def test_a_context_length_refusal_is_logged_with_its_code(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        body = request_body({"q": {"type": "noul"}}, SENTINELS["state"] + "x" * 4000)
        with build_client() as client:
            assert systemone(client, body).status_code == 422
        [record] = access_lines(captured)
        assert record["error_code"] == "context_length_exceeded"
        assert_no_sentinels(captured)


class TestUnauthorizedPath:
    def test_a_401_is_logged_with_its_error_code(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with build_client() as client:
            response = client.post(
                "/v1/systemone",
                content=json.dumps(sentinel_body()).encode(),
                headers={
                    "Authorization": "Bearer CANARY-PRESENTED-KEY-2266",
                    "Content-Type": "application/json",
                },
            )
        assert response.status_code == 401

        [record] = access_lines(captured)
        assert_record_shape(record)
        assert record["status"] == 401
        assert record["error_code"] == "unauthorized"
        assert record["bundle"] is None

    def test_neither_the_presented_nor_the_configured_key_is_logged(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        presented = "CANARY-PRESENTED-KEY-2266"
        with build_client() as client:
            client.post(
                "/v1/systemone",
                content=json.dumps(sentinel_body()).encode(),
                headers={
                    "Authorization": f"Bearer {presented}",
                    "Content-Type": "application/json",
                },
            )
        text = all_text(captured)
        assert presented not in text
        assert_no_sentinels(captured)

    def test_the_body_of_an_unauthenticated_request_is_never_logged(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with build_client() as client:
            client.post(
                "/v1/systemone",
                content=b"this is not json " + SENTINELS["state"].encode(),
                headers={"Content-Type": "application/json"},
            )
        assert_no_sentinels(captured)


class TestInternalFailurePath:
    """A NaN logit produces a deliberately empty body; the log is the only evidence."""

    @staticmethod
    def _nan_after_warmup(pairs: Any) -> list[float]:
        # Sound at warmup, broken afterwards: a bundle that fails to warm up never
        # becomes ready, and a 503 would not exercise the path under test.
        return [
            0.0 if pair.premise == WARMUP_PREMISE else float("nan") for pair in pairs
        ]

    def _nan_client(self) -> Any:
        backend = FakeBackend(logit_fn=self._nan_after_warmup)
        return build_client(registry=build_registry(backend))

    def test_a_500_is_logged_with_its_error_code(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with self._nan_client() as client:
            response = systemone(client, sentinel_body())
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_error"

        [record] = access_lines(captured)
        assert_record_shape(record)
        assert record["status"] == 500
        assert record["error_code"] == "inference_error"
        # The request got far enough to be measured, and the log says so.
        assert record["sequences"] == 4
        assert record["input_tokens"] > 0

    def test_an_operator_facing_line_accompanies_the_500(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        # Without this the only trace of a broken invariant is a 500 with an empty
        # body, which is indistinguishable from any other failure (A-F1).
        with self._nan_client() as client:
            assert systemone(client, sentinel_body()).status_code == 500

        errors = [r for r in captured.records if r.levelno >= logging.ERROR]
        assert errors, "a 5xx must leave an error-level line behind"
        assert any("inference_error" in r.getMessage() for r in errors)

    def test_the_failing_input_does_not_reach_any_logger(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        with self._nan_client() as client:
            assert systemone(client, sentinel_body()).status_code == 500
        assert_no_sentinels(captured)

    def test_an_unexpected_backend_exception_does_not_log_its_message(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        # The most likely leak of all: a real tokenizer or torch error quotes the input
        # it choked on, and a traceback repeats that message at the end (S-M2).
        class Exploding(FakeBackend):
            def score(self, sequences: Any, cancel: Any) -> list[float]:
                if any(s.data.premise != WARMUP_PREMISE for s in sequences):
                    raise RuntimeError(f"cannot score {SENTINELS['state']}")
                return super().score(sequences, cancel)

        with build_client(registry=build_registry(Exploding())) as client:
            assert systemone(client, sentinel_body()).status_code == 500

        [record] = access_lines(captured)
        assert record["error_code"] == "inference_error"
        assert_no_sentinels(captured)

    def test_an_exception_escaping_the_router_does_not_log_its_message(
        self, captured: pytest.LogCaptureFixture
    ) -> None:
        class Exploding(FakeBackend):
            def count_and_encode(self, pairs: Any) -> list[Any]:
                if any(pair.premise != WARMUP_PREMISE for pair in pairs):
                    raise KeyError(f"unexpected {SENTINELS['state']}")
                return super().count_and_encode(pairs)

        with build_client(registry=build_registry(Exploding())) as client:
            assert systemone(client, sentinel_body()).status_code == 500

        [record] = access_lines(captured)
        assert record["error_code"] == "internal_error"
        assert_no_sentinels(captured)


class TestHealthPaths:
    @pytest.mark.parametrize("path", ["/healthz", "/readyz"])
    def test_unauthenticated_health_checks_still_log_a_full_record(
        self, captured: pytest.LogCaptureFixture, path: str
    ) -> None:
        with build_client() as client:
            client.get(path)
        [record] = access_lines(captured)
        assert_record_shape(record)
        assert record["path"] == path
        assert record["error_code"] is None
