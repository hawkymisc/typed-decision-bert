"""fake-deterministic-v1 (POC_DESIGN 6.2).

The logit of a candidate is a hash of its premise and hypothesis, so it depends on the
compiled model input and on nothing else - not on the question ID, not on the position
in the batch, not on the number of candidates. That is what lets the invariance tests
(spec 13.3, PT04) detect a compiler defect rather than hide it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence

from jevbert.backends.base import WARMUP_PREMISE, CancelToken, EncodedSequence, TextPair

#: Separator between premise and hypothesis, so that ("ab", "") and ("a", "b") differ.
_SEPARATOR = b"\x1f"
_UINT64_MAX = 2**64 - 1
_LOGIT_RANGE = 4.0

#: Stand-in for a tokenizer: UTF-8 byte length plus the four special tokens.
_SPECIAL_TOKEN_COUNT = 4


def utf8_length(text: str) -> int:
    """This backend's stand-in for tokenizing one side of a pair."""
    return len(text.encode("utf-8"))

LogitFn = Callable[[Sequence[TextPair]], Sequence[float]]


class FakeBackend:
    """Deterministic backend for tests and for exercising the API without a model.

    ``logit_fn`` replaces the hash so that tests can inject ties, extreme values, NaN or
    a candidate/logit count mismatch (CT07, CT08). It is a constructor argument only:
    nothing reachable over HTTP can select it.
    """

    backend_id = "fake-deterministic-v1"

    def __init__(
        self,
        *,
        logit_fn: LogitFn | None = None,
        delay_seconds: float = 0.0,
        load_delay_seconds: float = 0.0,
    ) -> None:
        self._logit_fn = logit_fn
        self._delay_seconds = delay_seconds
        self._load_delay_seconds = load_delay_seconds
        self.loaded = False

    def load(self) -> None:
        if self._load_delay_seconds > 0.0:
            time.sleep(self._load_delay_seconds)
        self.loaded = True

    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
        """Measure each pair, paying for a repeated premise only once (S-H2).

        The map lives for the length of this call only, as ``backends/base.py``
        requires.
        """
        premise_lengths: dict[str, int] = {}
        sequences: list[EncodedSequence] = []
        for pair in pairs:
            premise_length = premise_lengths.get(pair.premise)
            if premise_length is None:
                premise_length = utf8_length(pair.premise)
                premise_lengths[pair.premise] = premise_length
            sequences.append(
                EncodedSequence(
                    token_count=(
                        premise_length + utf8_length(pair.hypothesis) + _SPECIAL_TOKEN_COUNT
                    ),
                    data=pair,
                )
            )
        return sequences

    def score(self, sequences: Sequence[EncodedSequence], cancel: CancelToken) -> list[float]:
        cancel.raise_if_cancelled()
        pairs = [sequence.data for sequence in sequences]
        self._sleep(cancel, pairs)
        if self._logit_fn is not None:
            return list(self._logit_fn(pairs))
        return [_hash_logit(pair) for pair in pairs]

    def _sleep(self, cancel: CancelToken, pairs: Sequence[TextPair]) -> None:
        """Simulate a slow model, staying responsive to cancellation (CT10).

        The warmup fixture is exempt: a slow fake is how 529 and 504 are reproduced,
        and charging that delay to every bundle load would only slow the suite down.
        """
        remaining = self._delay_seconds
        if all(pair.premise == WARMUP_PREMISE for pair in pairs):
            remaining = 0.0
        while remaining > 0.0:
            cancel.raise_if_cancelled()
            step = min(0.01, remaining)
            time.sleep(step)
            remaining -= step
        cancel.raise_if_cancelled()


def _hash_logit(pair: TextPair) -> float:
    digest = hashlib.sha256(
        pair.premise.encode("utf-8") + _SEPARATOR + pair.hypothesis.encode("utf-8")
    ).digest()
    unsigned = int.from_bytes(digest[:8], "big")
    return -_LOGIT_RANGE + 2.0 * _LOGIT_RANGE * unsigned / _UINT64_MAX
