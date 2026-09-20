"""Numeric processing.

The first section is the reference implementation from the specification
(JevBERT_spec_design.md, appendix C) reproduced verbatim. It MUST NOT be modified: the
production path calls exactly these functions so that there is no second definition of
the probability maths to drift from.

The second section adds the per-type adapters described in POC_DESIGN 7. They only
arrange inputs and outputs; every number they return comes out of the reference
functions above.
"""

# --- BEGIN verbatim copy of JevBERT_spec_design.md appendix C -------------------------
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real


def _finite_real(value: Real, name: str) -> float:
    """Reject booleans, implicit string conversions, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def probabilities_from_logits(
    logits: Sequence[float], temperature: float = 1.0
) -> tuple[float, ...]:
    """Normalize valid-option logits only; padded options must be removed."""
    if len(logits) < 2:
        raise ValueError("At least two valid options are required")
    t = _finite_real(temperature, "temperature")
    if t <= 0.0:
        raise ValueError("temperature must be positive")
    values = tuple(_finite_real(v, "logit") for v in logits)
    maximum = max(values)
    # Center before division to avoid overflow from large positive logits.
    weights = tuple(math.exp((v - maximum) / t) for v in values)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid softmax normalizer")
    return tuple(w / total for w in weights)


def _checked_probabilities(
    probabilities: Sequence[float], tolerance: float = 1e-6
) -> tuple[float, ...]:
    if len(probabilities) < 2:
        raise ValueError("At least two probabilities are required")
    p = tuple(_finite_real(v, "probability") for v in probabilities)
    if any(v < 0.0 or v > 1.0 for v in p):
        raise ValueError("Probabilities must be in [0, 1]")
    total = math.fsum(p)
    if abs(total - 1.0) > tolerance:
        raise ValueError("Probabilities must sum to one")
    return tuple(v / total for v in p)


def normalized_entropy_confidence(probabilities: Sequence[float]) -> float:
    """Concentration statistic, not a calibrated probability of correctness."""
    p = _checked_probabilities(probabilities)
    entropy = -math.fsum(v * math.log(v) for v in p if v > 0.0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(p))))


def expected_score(probabilities: Sequence[float]) -> float:
    """Return the expectation of zero-based rubric indices."""
    p = _checked_probabilities(probabilities)
    if len(p) > 10:
        raise ValueError("Score supports at most ten levels")
    return math.fsum(index * value for index, value in enumerate(p))


def select_choice(probabilities: Mapping[str, float]) -> str:
    """Resolve exact ties by Unicode code point order of option keys."""
    if not all(isinstance(key, str) for key in probabilities):
        raise ValueError("Option keys must be strings")
    if not 2 <= len(probabilities) <= 255:
        raise ValueError("Choice supports two to 255 options")
    keys = sorted(probabilities)
    p = _checked_probabilities([probabilities[key] for key in keys])
    index = max(range(len(keys)), key=p.__getitem__)
    return keys[index]


# --- END verbatim copy ---------------------------------------------------------------

#: Internal candidate order for Noul (spec 8.1). Never reordered.
NOUL_OPTION_ORDER = ("false", "true")


def noul_probability(logits: Sequence[float], temperature: float = 1.0) -> float:
    """Return p(yes) from ``[z_false, z_true]`` (spec 5.4, appendix C note)."""
    if len(logits) != 2:
        raise ValueError("Noul requires exactly two logits in false, true order")
    return probabilities_from_logits(logits, temperature)[1]


def choice_scores(
    option_keys: Sequence[str], logits: Sequence[float], temperature: float = 1.0
) -> tuple[str, dict[str, float], float]:
    """Return ``(choice, probabilities by key, confidence)`` for a Choice question.

    ``option_keys[i]`` must be the candidate scored by ``logits[i]``. The returned
    mapping is keyed in that same order; the response adapter reorders it for the wire.
    """
    if len(option_keys) != len(logits):
        raise ValueError("Choice option count does not match the logit count")
    if len(set(option_keys)) != len(option_keys):
        raise ValueError("Choice option keys must be unique")
    probabilities = probabilities_from_logits(logits, temperature)
    by_key = dict(zip(option_keys, probabilities, strict=True))
    return select_choice(by_key), by_key, normalized_entropy_confidence(probabilities)


def score_scores(
    logits: Sequence[float], temperature: float = 1.0
) -> tuple[float, tuple[float, ...], float]:
    """Return ``(score, probabilities by level order, confidence)`` for a Score question."""
    probabilities = probabilities_from_logits(logits, temperature)
    return (
        expected_score(probabilities),
        probabilities,
        normalized_entropy_confidence(probabilities),
    )
