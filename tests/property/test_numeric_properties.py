"""PT01/PT05: numeric and normalisation properties (POC_DESIGN 8.2)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from hypothesis import assume, example, given
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

#: A perfectly flat distribution computes 1 - H/log K to within a summation ulp of
#: zero, not to zero itself, so "flat" is read at this scale rather than exactly.
FLAT_CONFIDENCE = 1e-12


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
    @example([0.0, 1e-6])
    def test_confidence_is_zero_exactly_when_the_distribution_is_flat(
        self, logits: list[float]
    ) -> None:
        # The definition (spec 8.2) is 1 - H/log K, so a zero says "no concentration
        # at all" and nothing else. The old assertion here only fired when confidence
        # exceeded 0.5, which every non-degenerate distribution satisfies anyway.
        #
        # Both directions need a tolerance. A perfectly uniform vector sums its entropy
        # to within an ulp of log K, landing a hair either side of zero. And the
        # statistic is quadratic near the uniform point - its gradient there is zero -
        # so a confidence at 1e-12 still admits a spread around 1e-6. The bound below
        # is that relation with room to spare; it still rules out anything a reader
        # would call concentrated.
        p = probabilities_from_logits(logits)
        confidence = normalized_entropy_confidence(p)
        spread = max(p) - min(p)
        if spread == 0.0:
            assert confidence <= FLAT_CONFIDENCE
        if confidence <= FLAT_CONFIDENCE:
            assert spread <= 1e-4

    @given(LOGITS)
    def test_a_visibly_concentrated_distribution_has_a_visible_confidence(
        self, logits: list[float]
    ) -> None:
        # The other half of the same statement, stated where the quadratic does not
        # bite: a distribution that is plainly not flat reports so.
        p = probabilities_from_logits(logits)
        if max(p) - min(p) >= 0.01:
            assert normalized_entropy_confidence(p) > 1e-6

    @given(st.integers(min_value=2, max_value=64), st.floats(min_value=-20, max_value=20))
    def test_equal_logits_give_a_confidence_of_zero(self, count: int, logit: float) -> None:
        p = probabilities_from_logits([logit] * count)
        assert normalized_entropy_confidence(p) <= FLAT_CONFIDENCE

    @given(st.integers(min_value=2, max_value=32))
    def test_a_one_hot_distribution_gives_a_confidence_of_one(self, count: int) -> None:
        p = probabilities_from_logits([1e4] + [-1e4] * (count - 1))
        assert normalized_entropy_confidence(p) == pytest.approx(1.0, abs=1e-12)

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

    @given(SMALL_LOGITS, st.floats(min_value=0.1, max_value=10.0))
    def test_temperature_preserves_the_ordering(
        self, logits: list[float], temperature: float
    ) -> None:
        # The old version of this test sorted a list and asserted it was sorted, which
        # is true of every list. The property that matters is that scaling never
        # reorders candidates: a temperature change may flatten or sharpen the
        # distribution but must not change which option wins.
        base = probabilities_from_logits(logits)
        scaled = probabilities_from_logits(logits, temperature)
        for i in range(len(logits)):
            for j in range(len(logits)):
                if logits[i] < logits[j]:
                    assert base[i] <= base[j]
                    assert scaled[i] <= scaled[j]

    @given(SMALL_LOGITS, st.floats(min_value=1.0, max_value=10.0))
    def test_raising_the_temperature_never_raises_confidence(
        self, logits: list[float], temperature: float
    ) -> None:
        # A higher temperature flattens the distribution, so the concentration
        # statistic cannot go up. This is the direction calibration would move it.
        base = normalized_entropy_confidence(probabilities_from_logits(logits))
        hotter = normalized_entropy_confidence(
            probabilities_from_logits(logits, temperature)
        )
        assert hotter <= base + 1e-9

    @given(SMALL_LOGITS, st.floats(min_value=0.1, max_value=1.0))
    def test_lowering_the_temperature_never_lowers_confidence(
        self, logits: list[float], temperature: float
    ) -> None:
        base = normalized_entropy_confidence(probabilities_from_logits(logits))
        colder = normalized_entropy_confidence(
            probabilities_from_logits(logits, temperature)
        )
        assert colder >= base - 1e-9

    @given(st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=20))
    def test_select_choice_agrees_with_the_distribution(self, logits: list[float]) -> None:
        # "in keys" cannot fail: select_choice returns one of the keys it was given.
        # The contract is argmax with a code point tie-break (I05, spec 5.4).
        assume(all(math.isfinite(v) for v in logits))
        keys = [f"key{i}" for i in range(len(logits))]
        probabilities = dict(zip(keys, probabilities_from_logits(logits), strict=True))
        best = max(probabilities.values())
        expected = min(key for key, value in probabilities.items() if value == best)
        assert select_choice(probabilities) == expected

    @given(st.lists(st.sampled_from(["a", "z", "あ", "ア", "😀"]), min_size=2, max_size=5))
    def test_ties_resolve_to_the_smallest_key_by_code_point(self, keys: list[str]) -> None:
        assume(len(set(keys)) == len(keys))
        probabilities = dict.fromkeys(keys, 1.0 / len(keys))
        assert select_choice(probabilities) == min(keys)


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
