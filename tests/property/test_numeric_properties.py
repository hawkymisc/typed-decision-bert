"""PT01/PT05: numeric and normalisation properties (POC_DESIGN 8.2)."""

from __future__ import annotations

import json
import math
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from jevbert.compiler.normalize import canonical_json, render
from jevbert.scoring.numeric import (
    choice_scores,
    expected_score,
    normalized_entropy_confidence,
    noul_probability,
    probabilities_from_logits,
    score_scores,
    select_choice,
)
from tests.property.strategies import JSON_VALUES

# Bounded well inside the exponent range so that generation stays meaningful; CT07
# covers the extreme values explicitly.
LOGITS = st.lists(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=255,
)
SMALL_LOGITS = st.lists(
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=10,
)


class TestPT01Distributions:
    @given(LOGITS)
    def test_distribution_is_a_probability_vector(self, logits: list[float]) -> None:
        # I03, I04
        p = probabilities_from_logits(logits)
        assert len(p) == len(logits)
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in p)
        assert abs(math.fsum(p) - 1.0) <= 1e-6

    @given(LOGITS)
    def test_confidence_is_within_zero_and_one(self, logits: list[float]) -> None:
        confidence = normalized_entropy_confidence(probabilities_from_logits(logits))
        assert 0.0 <= confidence <= 1.0

    @given(LOGITS)
    def test_confidence_is_zero_only_for_a_flat_distribution(
        self, logits: list[float]
    ) -> None:
        p = probabilities_from_logits(logits)
        confidence = normalized_entropy_confidence(p)
        if confidence > 0.5:
            assert max(p) > 1.0 / len(p)

    @given(LOGITS)
    def test_choice_selects_the_argmax_with_code_point_tie_break(
        self, logits: list[float]
    ) -> None:
        # I05
        keys = tuple(f"k{i:03d}" for i in range(len(logits)))
        choice, probabilities, _ = choice_scores(keys, logits)
        best = max(probabilities.values())
        assert probabilities[choice] == best
        assert choice == min(k for k, v in probabilities.items() if v == best)

    @given(SMALL_LOGITS)
    def test_score_is_the_expectation_and_stays_in_range(self, logits: list[float]) -> None:
        # I06
        score, p, _ = score_scores(logits)
        assert 0.0 <= score <= len(p) - 1
        assert abs(score - expected_score(p)) <= 1e-6

    @given(
        st.floats(min_value=-50, max_value=50, allow_nan=False),
        st.floats(min_value=-50, max_value=50, allow_nan=False),
    )
    def test_noul_is_monotonic_in_the_logit_difference(
        self, z_false: float, z_true: float
    ) -> None:
        # Non-strict: a difference far below float resolution at 0.5 (say 1e-264)
        # legitimately produces exactly 0.5.
        value = noul_probability([z_false, z_true])
        assert 0.0 <= value <= 1.0
        if z_true >= z_false:
            assert value >= 0.5
        else:
            assert value <= 0.5

    @given(
        st.floats(min_value=-50, max_value=50, allow_nan=False),
        st.floats(min_value=1e-3, max_value=50),
    )
    def test_a_representable_gap_moves_noul_off_the_midpoint(
        self, z_false: float, gap: float
    ) -> None:
        assert noul_probability([z_false, z_false + gap]) > 0.5
        assert noul_probability([z_false, z_false - gap]) < 0.5

    @given(LOGITS, st.floats(min_value=0.1, max_value=10.0))
    def test_temperature_preserves_the_ordering(
        self, logits: list[float], temperature: float
    ) -> None:
        base = probabilities_from_logits(logits)
        scaled = probabilities_from_logits(logits, temperature)
        order = sorted(range(len(logits)), key=base.__getitem__)
        scaled_order = sorted(range(len(logits)), key=scaled.__getitem__)
        assert [base[i] for i in order] == sorted(base)
        assert [scaled[i] for i in scaled_order] == sorted(scaled)

    @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=20))
    def test_select_choice_agrees_with_the_distribution(self, logits: list[float]) -> None:
        assume(all(math.isfinite(v) for v in logits))
        keys = [f"key{i}" for i in range(len(logits))]
        probabilities = dict(zip(keys, probabilities_from_logits(logits), strict=True))
        assert select_choice(probabilities) in keys


class TestPT05CanonicalJson:
    @given(JSON_VALUES)
    def test_is_deterministic(self, value: Any) -> None:
        assert canonical_json(value) == canonical_json(value)

    @given(JSON_VALUES)
    def test_round_trips_to_the_same_value_and_types(self, value: Any) -> None:
        restored = json.loads(canonical_json(value))
        assert restored == value
        assert _same_types(restored, value)

    @given(JSON_VALUES)
    def test_reparsing_is_a_fixed_point(self, value: Any) -> None:
        once = canonical_json(value)
        assert canonical_json(json.loads(once)) == once

    @given(st.dictionaries(st.text(max_size=5), JSON_VALUES, max_size=5))
    def test_key_insertion_order_does_not_matter(self, mapping: dict[str, Any]) -> None:
        reversed_mapping = dict(reversed(list(mapping.items())))
        assert canonical_json(mapping) == canonical_json(reversed_mapping)

    @given(st.text(max_size=20))
    def test_render_passes_strings_through_unchanged(self, text: str) -> None:
        assert render(text) == text


def _same_types(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return all(_same_types(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return all(_same_types(a, b) for a, b in zip(left, right, strict=True))
    return True
