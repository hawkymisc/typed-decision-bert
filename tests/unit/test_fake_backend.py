"""fake-deterministic-v1 backend (POC_DESIGN 6.2)."""

from __future__ import annotations

import math
import time

import pytest

from jevbert.backends import fake
from jevbert.backends.base import WARMUP_PREMISE, CancelToken, InferenceCancelled, TextPair
from jevbert.backends.fake import FakeBackend


def _score(backend: FakeBackend, pairs: list[TextPair]) -> list[float]:
    return backend.score(backend.count_and_encode(pairs), CancelToken())


class TestDeterminism:
    def test_same_pair_gives_the_same_logit(self) -> None:
        backend = FakeBackend()
        pair = [TextPair("premise", "hypothesis")]
        assert _score(backend, pair) == _score(FakeBackend(), pair)

    def test_logits_stay_inside_the_declared_range(self) -> None:
        backend = FakeBackend()
        pairs = [TextPair(f"p{i}", f"h{i}") for i in range(200)]
        logits = _score(backend, pairs)
        assert all(-4.0 <= z <= 4.0 and math.isfinite(z) for z in logits)

    def test_logit_depends_on_both_premise_and_hypothesis(self) -> None:
        backend = FakeBackend()
        assert _score(backend, [TextPair("a", "b")]) != _score(backend, [TextPair("a", "c")])
        assert _score(backend, [TextPair("a", "b")]) != _score(backend, [TextPair("c", "b")])

    def test_the_separator_prevents_boundary_collisions(self) -> None:
        # sha256("ab" || 0x1f || "") must differ from sha256("a" || 0x1f || "b").
        backend = FakeBackend()
        assert _score(backend, [TextPair("ab", "")]) != _score(backend, [TextPair("a", "b")])

    def test_logit_does_not_depend_on_position_in_the_batch(self) -> None:
        # The invariance tests in spec 13.3 rely on this.
        backend = FakeBackend()
        alone = _score(backend, [TextPair("p", "h")])
        together = _score(backend, [TextPair("x", "y"), TextPair("p", "h")])
        assert together[1] == alone[0]


class TestTokenCounting:
    def test_token_count_is_utf8_bytes_plus_four(self) -> None:
        backend = FakeBackend()
        [encoded] = backend.count_and_encode([TextPair("abc", "de")])
        assert encoded.token_count == 3 + 2 + 4

    def test_multibyte_characters_count_as_their_utf8_length(self) -> None:
        backend = FakeBackend()
        [encoded] = backend.count_and_encode([TextPair("返金", "")])
        assert encoded.token_count == 6 + 4


class TestPremiseIsMeasuredOnce:
    """S-H2: an A0 backend sees the same premise once per candidate.

    ``count_and_encode`` is contracted to tokenize a repeated premise once per call
    (``backends/base.py``). With a real tokenizer the difference between honouring that
    and ignoring it is the difference between one pass over the state and 255 of them.
    """

    def test_a_repeated_premise_is_measured_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        measured: list[str] = []
        original = fake.utf8_length
        monkeypatch.setattr(
            fake, "utf8_length", lambda text: (measured.append(text), original(text))[1]
        )
        backend = FakeBackend()
        backend.count_and_encode(
            [TextPair("state", f"h{index}") for index in range(8)]
        )
        assert measured.count("state") == 1
        assert sorted(t for t in measured if t != "state") == [f"h{i}" for i in range(8)]

    def test_distinct_premises_are_each_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        measured: list[str] = []
        original = fake.utf8_length
        monkeypatch.setattr(
            fake, "utf8_length", lambda text: (measured.append(text), original(text))[1]
        )
        backend = FakeBackend()
        backend.count_and_encode([TextPair("a", "h"), TextPair("b", "h")])
        assert measured.count("a") == 1
        assert measured.count("b") == 1

    def test_caching_does_not_change_the_counts(self) -> None:
        backend = FakeBackend()
        pairs = [TextPair("返金", "x"), TextPair("返金", "yy"), TextPair("other", "x")]
        counts = [sequence.token_count for sequence in backend.count_and_encode(pairs)]
        assert counts == [6 + 1 + 4, 6 + 2 + 4, 5 + 1 + 4]

    def test_the_cache_does_not_survive_the_call(self) -> None:
        # Per call, not per instance: a long-lived map keyed by user text would be a
        # second place where request data lives (spec 15.2).
        backend = FakeBackend()
        first = backend.count_and_encode([TextPair("p", "h")])
        second = backend.count_and_encode([TextPair("p", "h")])
        assert first[0].token_count == second[0].token_count
        assert not hasattr(backend, "_premise_lengths")


class TestWarmupIsNotDelayed:
    def test_the_simulated_latency_does_not_apply_to_the_warmup_fixture(self) -> None:
        # A slow fake is how 529 and 504 are reproduced; paying that delay at warmup
        # too would just make every suite run slower for nothing.
        backend = FakeBackend(delay_seconds=30.0)
        pairs = [TextPair(WARMUP_PREMISE, "The answer to the question is no.")]
        started = time.monotonic()
        backend.score(backend.count_and_encode(pairs), CancelToken())
        assert time.monotonic() - started < 1.0

    def test_a_served_request_still_pays_the_latency(self) -> None:
        backend = FakeBackend(delay_seconds=0.2)
        pairs = [TextPair("real state", "h")]
        started = time.monotonic()
        backend.score(backend.count_and_encode(pairs), CancelToken())
        assert time.monotonic() - started >= 0.2


class TestInjection:
    def test_logit_fn_replaces_the_hash(self) -> None:
        backend = FakeBackend(logit_fn=lambda pairs: [1.0] * len(pairs))
        assert _score(backend, [TextPair("a", "b"), TextPair("c", "d")]) == [1.0, 1.0]

    def test_logit_fn_can_inject_a_count_mismatch(self) -> None:
        # CT08 relies on being able to produce a broken candidate mapping.
        backend = FakeBackend(logit_fn=lambda pairs: [0.0])
        assert len(_score(backend, [TextPair("a", "b"), TextPair("c", "d")])) == 1

    def test_logit_fn_can_inject_non_finite_values(self) -> None:
        backend = FakeBackend(logit_fn=lambda pairs: [float("nan")] * len(pairs))
        assert all(math.isnan(z) for z in _score(backend, [TextPair("a", "b")]))


class TestCancellation:
    def test_a_cancelled_token_stops_the_work(self) -> None:
        backend = FakeBackend(delay_seconds=5.0)
        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(InferenceCancelled):
            backend.score(backend.count_and_encode([TextPair("a", "b")]), cancel)

    def test_cancellation_is_checked_even_without_a_simulated_delay(self) -> None:
        backend = FakeBackend()
        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(InferenceCancelled):
            backend.score(backend.count_and_encode([TextPair("a", "b")]), cancel)


class TestLoad:
    def test_load_flips_the_backend_from_unloaded_to_loaded(self) -> None:
        backend = FakeBackend()
        assert backend.loaded is False
        backend.load()
        assert backend.loaded is True
        assert backend.backend_id == "fake-deterministic-v1"

    def test_the_configured_load_delay_is_honoured(self) -> None:
        backend = FakeBackend(load_delay_seconds=0.2)
        started = time.monotonic()
        backend.load()
        assert time.monotonic() - started >= 0.2
        assert backend.loaded is True
