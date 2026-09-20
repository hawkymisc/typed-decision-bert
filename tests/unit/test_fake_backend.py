"""fake-deterministic-v1 backend (POC_DESIGN 6.2)."""

from __future__ import annotations

import math

import pytest

from jevbert.backends.base import CancelToken, TextPair
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
        with pytest.raises(Exception, match="cancel"):
            backend.score(backend.count_and_encode([TextPair("a", "b")]), cancel)


class TestLoad:
    def test_load_is_required_before_scoring_is_considered_ready(self) -> None:
        backend = FakeBackend()
        backend.load()
        assert backend.backend_id == "fake-deterministic-v1"
