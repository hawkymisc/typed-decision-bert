"""S-H2 / A-F7: the cost of a request is bounded before the tokenizer is asked.

An A0 backend scores one sequence per candidate, so a single small body expands into
``questions x candidates`` sequences that all repeat the state and the instruction. Three
things keep that from turning into a denial of service: a character ceiling checked
before any text is expanded or tokenized, tokenization that runs off the event loop
under the request deadline, and admission control that counts the encoding step.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from jevbert.backends.base import CancelToken, EncodedSequence, TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.serializer_nli import MIN_TOKENS_PER_SEQUENCE, max_sequences_for
from jevbert.config import Limits, ServingSettings
from jevbert.inference.registry import Bundle, ModelRegistry, read_manifest
from tests.conftest import (
    AUTH,
    FAKE_MANIFEST,
    build_client,
    build_registry,
    build_settings,
    request_body,
    systemone,
)

NOUL = {"q": {"type": "noul"}}

#: build_settings() mirrors configs/jevbert.test.yaml: 131,072 request tokens, so the
#: derived character ceiling is four times that.
DEFAULT_MAX_CHARS = 4 * 131_072


class RecordingBackend(FakeBackend):
    """Records what the encoder was asked to do and on which thread."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encode_calls: list[int] = []
        self.encode_threads: list[str] = []
        self.score_threads: list[str] = []

    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
        self.encode_calls.append(len(pairs))
        self.encode_threads.append(threading.current_thread().name)
        return super().count_and_encode(pairs)

    def score(self, sequences: Sequence[EncodedSequence], cancel: CancelToken) -> list[float]:
        self.score_threads.append(threading.current_thread().name)
        return super().score(sequences, cancel)

    def forget_warmup(self) -> None:
        self.encode_calls.clear()
        self.encode_threads.clear()
        self.score_threads.clear()


def _amplifying_body(state_chars: int, options: int) -> dict[str, Any]:
    criteria = {f"k{index:03d}": None for index in range(options)}
    return request_body({"q": {"type": "choice", "criteria": criteria}}, "x" * state_chars)


class TestCharacterCeiling:
    def test_candidate_amplification_is_refused_before_tokenization(self) -> None:
        backend = RecordingBackend()
        registry = build_registry(backend)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            # 255 candidates x 10,000 characters of state is 2.5M characters of model
            # input from a 15 KiB body.
            response = systemone(client, _amplifying_body(10_000, 255))

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"
        # The gate is worth nothing if the tokenizer ran anyway.
        assert backend.encode_calls == []

    def test_the_refusal_names_the_question(self) -> None:
        with build_client() as client:
            payload = systemone(client, _amplifying_body(10_000, 255)).json()
        assert payload["error"]["path"] == ["questions", "q"]

    def test_a_request_just_under_the_ceiling_is_served(self) -> None:
        backend = RecordingBackend()
        registry = build_registry(backend)
        backend.forget_warmup()
        settings = build_settings(limits=Limits(max_request_tokens=131_072))
        with build_client(settings=settings, registry=registry) as client:
            # 2 candidates, both well inside every other limit.
            response = systemone(client, request_body(NOUL, "x" * 1_000))
        assert response.status_code == 200
        assert backend.encode_calls == [2]

    def test_an_explicit_ceiling_is_honoured(self) -> None:
        settings = build_settings(
            limits=Limits(max_request_tokens=131_072, max_request_chars=500)
        )
        with build_client(settings=settings) as client:
            assert systemone(client, request_body(NOUL, "x" * 100)).status_code == 200
            over = systemone(client, request_body(NOUL, "x" * 400))
            assert over.status_code == 422
            assert over.json()["error"]["code"] == "context_length_exceeded"

    def test_the_instruction_counts_once_per_candidate(self) -> None:
        # The template repeats the instruction for every candidate, so a short body
        # with a long instruction and many options amplifies just as much as the state.
        settings = build_settings(
            limits=Limits(max_request_tokens=131_072, max_request_chars=10_000)
        )
        criteria = {f"k{index:03d}": None for index in range(100)}
        body = request_body(
            {"q": {"type": "choice", "instructions": "i" * 1_000, "criteria": criteria}}, "s"
        )
        with build_client(settings=settings) as client:
            response = systemone(client, body)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"

    def test_the_ceiling_is_published(self, client: TestClient) -> None:
        limits = client.get("/jevbert/v1/capabilities", headers=AUTH).json()["limits"]
        assert limits["max_request_chars"] == DEFAULT_MAX_CHARS

    def test_the_default_is_derived_from_the_token_limit(self) -> None:
        assert Limits(max_request_tokens=1_000).effective_max_request_chars == 4_000
        explicit = Limits(max_request_tokens=1_000, max_request_chars=7)
        assert explicit.effective_max_request_chars == 7


class TestSequenceCeiling:
    """S-L1: the count of sequences is known before a single one is tokenized.

    256 questions x 255 options is 65,280 sequences, and every one of them carries at
    least the four control tokens plus a token of data. The character gate lets that
    through when the texts are short, so the whole set used to be tokenized and then
    refused on the token total.
    """

    #: 4 control tokens + at least 1 of data, so no sequence can cost less than 5.
    MIN_TOKENS_PER_SEQUENCE = 5

    def _many_sequences_body(self, questions: int, options: int) -> dict[str, Any]:
        criteria = {f"k{index:03d}": None for index in range(options)}
        return request_body(
            {
                f"q{index}": {"type": "choice", "criteria": dict(criteria)}
                for index in range(questions)
            },
            "s",
        )

    def test_too_many_sequences_are_refused_before_tokenization(self) -> None:
        backend = RecordingBackend()
        registry = build_registry(backend)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            response = systemone(client, self._many_sequences_body(256, 255))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert backend.encode_calls == []

    def _narrow_registry(self, backend: RecordingBackend, max_request_tokens: int) -> Any:
        """A bundle whose own budget is small; the bundle's limits are what applies."""
        manifest, digest = read_manifest(FAKE_MANIFEST)
        narrowed = manifest.model_copy(
            update={
                "limits": manifest.limits.model_copy(
                    update={
                        "max_request_tokens": max_request_tokens,
                        "max_sequence_tokens": max_request_tokens,
                    }
                )
            }
        )
        registry = ModelRegistry([Bundle(narrowed, digest, backend)])
        registry.load_all()
        return registry

    def test_the_ceiling_follows_the_token_budget(self) -> None:
        backend = RecordingBackend()
        # 500 tokens buys at most 100 sequences; 102 are asked for.
        registry = self._narrow_registry(backend, 500)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            response = systemone(client, self._many_sequences_body(1, 102))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert backend.encode_calls == []

    def test_a_request_inside_the_ceiling_still_reaches_the_backend(self) -> None:
        backend = RecordingBackend()
        registry = self._narrow_registry(backend, 500)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            response = systemone(client, self._many_sequences_body(1, 20))
        assert response.status_code == 200
        assert backend.encode_calls == [20]

    def test_the_ceiling_is_the_budget_over_the_cheapest_possible_sequence(self) -> None:
        budget = TokenBudget(max_sequence_tokens=2048, max_request_tokens=131_072)
        assert max_sequences_for(budget) == 131_072 // self.MIN_TOKENS_PER_SEQUENCE
        assert MIN_TOKENS_PER_SEQUENCE == self.MIN_TOKENS_PER_SEQUENCE


class TestEncodingRunsOffTheEventLoop:
    def test_compilation_and_tokenization_use_the_encoder_thread(self) -> None:
        backend = RecordingBackend()
        registry = build_registry(backend)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            assert systemone(client, request_body(NOUL)).status_code == 200

        [encoder_thread] = backend.encode_threads
        [scorer_thread] = backend.score_threads
        assert encoder_thread.startswith("jevbert-encoder")
        assert scorer_thread.startswith("jevbert-inference")
        # A shared tokenizer must never be touched by two threads at once, and GPU work
        # must not wait behind it, so the two are different single-thread pools.
        assert encoder_thread != scorer_thread

    def test_a_still_running_encode_hits_the_deadline(self) -> None:
        # A-F7: before phase 1.5 tokenization ran on the event loop, outside every
        # deadline, so a slow tokenizer stalled the whole server instead of timing out.
        # The gate is still closed when the assertion runs, so the 504 is produced
        # while the encode is genuinely in flight rather than after it finished.
        gate = _Gate()

        class GatedEncoder(FakeBackend):
            def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
                if any(pair.premise != "warmup" for pair in pairs):
                    gate.arrive()
                return super().count_and_encode(pairs)

        settings = build_settings(serving=ServingSettings(request_deadline_seconds=0.05))
        registry = build_registry(GatedEncoder())
        try:
            with build_client(settings=settings, registry=registry) as client:
                response = systemone(client, request_body(NOUL))
                assert gate.started.is_set()
                assert response.status_code == 504
                payload = response.json()
                assert payload["error"]["code"] == "deadline_exceeded"
                assert payload["error"]["retryable"] is True
        finally:
            gate.release()

    def test_the_event_loop_keeps_serving_while_an_encode_is_in_flight(self) -> None:
        gate = _Gate()

        class GatedEncoder(FakeBackend):
            def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
                if any(pair.premise != "warmup" for pair in pairs):
                    gate.arrive()
                return super().count_and_encode(pairs)

        settings = build_settings(serving=ServingSettings(max_pending_requests=4))
        registry = build_registry(GatedEncoder())
        with build_client(settings=settings, registry=registry) as client:
            with _in_background(systemone, client, request_body(NOUL)) as pending:
                assert gate.started.wait(5.0)
                # The loop is free: liveness still answers while the encoder blocks.
                assert client.get("/healthz").status_code == 200
                gate.release()
            assert pending.result().status_code == 200


class TestEncodingIsAdmissionControlled:
    def test_a_blocked_encoder_makes_the_next_request_overloaded(self) -> None:
        gate = _Gate()

        class GatedEncoder(FakeBackend):
            def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
                if any(pair.premise != "warmup" for pair in pairs):
                    gate.arrive()
                return super().count_and_encode(pairs)

        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        registry = build_registry(GatedEncoder())
        with build_client(settings=settings, registry=registry) as client:
            with _in_background(systemone, client, request_body(NOUL)) as pending:
                assert gate.started.wait(5.0)
                refused = systemone(client, request_body(NOUL))
                gate.release()
            assert refused.status_code == 529
            assert refused.json()["error"]["code"] == "overloaded"
            assert pending.result().status_code == 200


class TestValidationOrderIsPreserved:
    def test_token_overflow_is_still_a_422_after_the_encoder_runs(
        self, client: TestClient
    ) -> None:
        # The character ceiling is generous on purpose: a request that passes it and
        # then overruns the token budget must still be 422, not 529 or 500.
        response = systemone(client, request_body(NOUL, "x" * 4_000))
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "context_length_exceeded"
        assert payload["error"]["path"] == ["questions", "q"]

    def test_an_unknown_model_is_refused_before_any_encoding(self) -> None:
        backend = RecordingBackend()
        registry = build_registry(backend)
        backend.forget_warmup()
        with build_client(registry=registry) as client:
            body = request_body(NOUL)
            body["model"] = "jev-1.13.0"
            assert systemone(client, body).status_code == 422
        assert backend.encode_calls == []


class _Gate:
    """Deterministic rendezvous: the worker waits until the test lets it through."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self._release = threading.Event()

    def arrive(self) -> None:
        self.started.set()
        assert self._release.wait(10.0), "the gate was never released"

    def release(self) -> None:
        self._release.set()


@contextmanager
def _in_background(call: Any, *args: Any) -> Iterator[Future[Any]]:
    """Run ``call`` on a worker thread for the duration of the ``with`` block."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield pool.submit(call, *args)
