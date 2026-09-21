"""Backend contract (spec 12.2; POC_DESIGN 6.1).

A backend turns (premise, hypothesis) text pairs into one finite logit per candidate.
It never decides probabilities or confidence - that is ``jevbert.scoring`` alone
(spec 12.1) - and it owns everything that needs a tokenizer, so that the compiler can
stay free of model specifics.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: Premise of the fixed warmup fixture (spec 16.1). It is not user data, so a backend
#: is free to recognise it - the fake one skips its simulated latency for it.
WARMUP_PREMISE = "warmup"


class InferenceCancelled(Exception):
    """Raised inside a backend when the request was cancelled or timed out."""

    def __init__(self, message: str = "The request was cancelled before it completed.") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class TextPair:
    """One candidate: the premise and the hypothesis built by the compiler."""

    premise: str
    hypothesis: str


@dataclass(frozen=True)
class EncodedSequence:
    """One tokenized candidate sequence.

    Attributes:
        token_count: Non-padding token count, including special tokens (spec 5.8).
        data: Payload private to the backend that produced it (token IDs for the NLI
            backend, the text pair for the fake backend). Callers must not interpret it.
    """

    token_count: int
    data: Any


class CancelToken:
    """Cooperative cancellation shared between the engine and a backend."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise InferenceCancelled()


@runtime_checkable
class Backend(Protocol):
    """The interface the registry builds and the engine calls."""

    backend_id: str

    def load(self) -> None:
        """Load weights and tokenizer. Failure raises; readiness is the caller's concern."""

    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
        """Tokenize each pair and report its non-padding token count.

        Never truncates: the caller enforces the token budget and rejects overflow
        (spec 6.3).

        An A0 backend receives one pair per candidate, and every pair of one request
        carries the *same* premise: up to 256 questions x 255 options is 65,280 copies
        of the state. An implementation SHOULD therefore tokenize each distinct premise
        once per call and reuse the result, rather than paying for the state once per
        candidate (S-H2). Any cache is per call: a map keyed by user text that outlives
        the call would be a second place request data lives (spec 15.2, 12.4).

        Called on the engine's single encoder thread, so an implementation does not
        have to make a tokenizer safe for concurrent use, and must not assume it runs
        on the event loop.
        """

    def score(self, sequences: Sequence[EncodedSequence], cancel: CancelToken) -> list[float]:
        """Return one finite logit per sequence, in the order given.

        A non-finite value or a length mismatch is turned into a 500 by the caller; it
        must never be smoothed into a plausible distribution.
        """
