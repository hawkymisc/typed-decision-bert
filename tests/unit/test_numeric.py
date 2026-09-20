"""Numeric processing (spec 8, appendix C; POC_DESIGN 7)."""

from __future__ import annotations

import math

import pytest

from jevbert.scoring.numeric import (
    NOUL_OPTION_ORDER,
    choice_scores,
    expected_score,
    normalized_entropy_confidence,
    noul_probability,
    probabilities_from_logits,
    score_scores,
    select_choice,
)


class TestAppendixCFixtures:
    """The check values tabulated at the end of appendix C (CT09)."""

    def test_choice_distribution_confidence(self) -> None:
        assert normalized_entropy_confidence([0.8, 0.15, 0.05]) == 0.44214218356782187

    def test_score_distribution(self) -> None:
        # Appendix C tabulates `score = 1.6`, which is the exact decimal value. The
        # reference implementation cannot return it: fsum(0*0.1 + 1*0.2 + 2*0.7) is the
        # double one ulp below 1.6. The reference code is reproduced unmodified, so the
        # actual value is pinned here and the deviation is recorded in POC_DESIGN 12.
        score = expected_score([0.1, 0.2, 0.7])
        assert score == 1.5999999999999999
        assert abs(score - 1.6) <= math.ulp(1.6)
        assert normalized_entropy_confidence([0.1, 0.2, 0.7]) == 0.27015330083790245

    def test_uniform_distribution_has_zero_confidence(self) -> None:
        assert normalized_entropy_confidence([0.5, 0.5]) == 0.0

    def test_concentrated_distribution_has_unit_confidence(self) -> None:
        assert normalized_entropy_confidence([0.0, 1.0]) == 1.0

    def test_tie_is_broken_by_code_point_order(self) -> None:
        assert select_choice({"b": 0.5, "a": 0.5}) == "a"


class TestProbabilitiesFromLogits:
    def test_uniform_logits_give_uniform_distribution(self) -> None:
        assert probabilities_from_logits([1.0, 1.0, 1.0]) == pytest.approx([1 / 3] * 3)

    def test_extreme_logits_do_not_overflow(self) -> None:
        # CT07: +-1e4 must not produce inf/NaN.
        p = probabilities_from_logits([1e4, -1e4])
        assert math.isfinite(sum(p))
        assert p[0] == pytest.approx(1.0)
        assert p[1] == pytest.approx(0.0)

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            probabilities_from_logits([0.0, 1.0], 0.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_logits_are_rejected(self, bad: float) -> None:
        # CT08: no uniform distribution, no 0.5, just a failure.
        with pytest.raises(ValueError):
            probabilities_from_logits([0.0, bad])

    def test_single_option_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            probabilities_from_logits([1.0])

    def test_booleans_are_not_silently_treated_as_numbers(self) -> None:
        with pytest.raises(ValueError):
            probabilities_from_logits([True, False])


class TestNoulAdapter:
    def test_noul_internal_order_is_false_true(self) -> None:
        assert NOUL_OPTION_ORDER == ("false", "true")

    def test_returns_probability_of_yes(self) -> None:
        # Higher true-logit must raise p(yes) above 0.5, never invert it (CT09).
        assert noul_probability([0.0, 2.0]) > 0.5
        assert noul_probability([2.0, 0.0]) < 0.5

    def test_equals_sigmoid_of_the_logit_difference(self) -> None:
        z_false, z_true = -0.7, 1.3
        expected = 1.0 / (1.0 + math.exp(-(z_true - z_false)))
        assert noul_probability([z_false, z_true]) == pytest.approx(expected, abs=1e-12)

    def test_requires_exactly_two_logits(self) -> None:
        with pytest.raises(ValueError):
            noul_probability([0.1, 0.2, 0.3])


class TestChoiceAdapter:
    def test_maps_logits_onto_the_given_keys(self) -> None:
        choice, probabilities, confidence = choice_scores(("a", "b", "c"), [0.0, 5.0, 0.0])
        assert choice == "b"
        assert set(probabilities) == {"a", "b", "c"}
        assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= confidence <= 1.0

    def test_exact_tie_selects_the_smallest_key(self) -> None:
        choice, _, confidence = choice_scores(("b", "a"), [1.0, 1.0])
        assert choice == "a"
        assert confidence == 0.0

    def test_key_count_must_match_logit_count(self) -> None:
        # CT08: candidate mapping and logits must correspond one to one.
        with pytest.raises(ValueError):
            choice_scores(("a", "b", "c"), [0.0, 1.0])


class TestScoreAdapter:
    def test_score_is_the_expectation_of_zero_based_indices(self) -> None:
        score, probabilities, _ = score_scores([0.0, 0.0])
        assert probabilities == pytest.approx([0.5, 0.5])
        assert score == pytest.approx(0.5)

    def test_score_stays_within_zero_and_k_minus_one(self) -> None:
        score, probabilities, _ = score_scores([0.0, 1.0, 9.0, 0.0])
        assert 0.0 <= score <= len(probabilities) - 1

    def test_score_is_not_normalised_to_zero_one(self) -> None:
        # spec 5.4: score must not be rescaled to [0, 1] nor start at 1.
        score, _, _ = score_scores([-10.0, -10.0, 10.0])
        assert score == pytest.approx(2.0, abs=1e-6)
