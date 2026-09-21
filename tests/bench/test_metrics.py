"""The numbers the comparison report publishes (docs/BENCHMARK.md 4).

Every expected value below is worked out by hand from the definition, not read back
from the implementation.
"""

from __future__ import annotations

import math

import pytest

from bench.metrics import (
    classification_metrics,
    mcnemar_exact_p,
    paired_comparison,
    percentile,
    probabilistic_metrics,
    top_choice,
)


class TestClassificationMetrics:
    def test_accuracy_and_per_label_scores(self) -> None:
        gold = ["a", "a", "b", "b", "c"]
        pred = ["a", "b", "b", "b", "a"]
        result = classification_metrics(gold, pred, labels=["a", "b", "c"])

        assert result["n"] == 5
        assert result["accuracy"] == pytest.approx(3 / 5)
        # a: tp=1 fp=1 fn=1 -> p=r=f=0.5; b: tp=2 fp=1 fn=0 -> p=2/3 r=1 f=0.8; c: tp=0 -> 0
        assert result["per_label"]["a"] == pytest.approx(
            {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1 + 1}
        )
        assert result["per_label"]["b"]["precision"] == pytest.approx(2 / 3)
        assert result["per_label"]["b"]["recall"] == pytest.approx(1.0)
        assert result["per_label"]["b"]["f1"] == pytest.approx(0.8)
        assert result["per_label"]["c"]["f1"] == 0.0
        assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)

    def test_a_label_absent_from_gold_and_prediction_does_not_dilute_macro_f1(self) -> None:
        result = classification_metrics(["a", "b"], ["a", "b"], labels=["a", "b", "never"])
        assert result["macro_f1"] == pytest.approx(1.0)
        assert result["per_label"]["never"]["support"] == 0

    def test_a_predicted_label_absent_from_gold_counts_against_macro_f1(self) -> None:
        # "c" is predicted but never correct: it is in the average with f1 = 0.
        result = classification_metrics(["a", "b"], ["a", "c"], labels=["a", "b", "c"])
        assert result["macro_f1"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)

    def test_confusion_is_indexed_gold_then_prediction(self) -> None:
        result = classification_metrics(["a", "a", "b"], ["b", "a", "b"], labels=["a", "b"])
        assert result["confusion"] == {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 1}}

    def test_empty_input_reports_zero_count_and_no_accuracy(self) -> None:
        result = classification_metrics([], [], labels=["a", "b"])
        assert result["n"] == 0
        assert result["accuracy"] is None
        assert result["macro_f1"] is None

    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError):
            classification_metrics(["a"], ["a", "b"], labels=["a", "b"])


class TestProbabilisticMetrics:
    def test_nll_and_brier(self) -> None:
        gold = ["a", "b"]
        probabilities = [{"a": 0.8, "b": 0.2}, {"a": 0.4, "b": 0.6}]
        result = probabilistic_metrics(gold, probabilities, labels=["a", "b"], bins=10)
        assert result["nll"] == pytest.approx((-math.log(0.8) - math.log(0.6)) / 2)
        # (0.2^2 + 0.2^2 + 0.4^2 + 0.4^2) / 2
        assert result["brier"] == pytest.approx((0.08 + 0.32) / 2)

    def test_ece_uses_the_top_probability_of_the_distribution(self) -> None:
        # Two answers in the (0.8, 0.9] bin: confidence 0.8 twice... placed by the top
        # probability. One right, one wrong -> |0.5 - 0.85| weighted 2/3; one answer at
        # 0.6 right -> |1 - 0.6| weighted 1/3.
        gold = ["a", "a", "b"]
        probabilities = [
            {"a": 0.85, "b": 0.15},
            {"a": 0.15, "b": 0.85},
            {"a": 0.4, "b": 0.6},
        ]
        result = probabilistic_metrics(gold, probabilities, labels=["a", "b"], bins=10)
        assert result["ece"] == pytest.approx(2 / 3 * abs(0.5 - 0.85) + 1 / 3 * abs(1.0 - 0.6))

    def test_a_zero_probability_on_the_gold_label_is_floored_not_infinite(self) -> None:
        result = probabilistic_metrics(["a"], [{"a": 0.0, "b": 1.0}], labels=["a", "b"])
        assert math.isfinite(result["nll"])
        assert result["nll"] > 20

    def test_empty_input(self) -> None:
        result = probabilistic_metrics([], [], labels=["a"])
        assert result == {"n": 0, "nll": None, "brier": None, "ece": None}


class TestMcNemar:
    def test_no_discordant_pairs_is_p_one(self) -> None:
        assert mcnemar_exact_p(0, 0) == 1.0

    def test_exact_binomial_two_sided(self) -> None:
        # b=0, c=5: 2 * 0.5^5
        assert mcnemar_exact_p(0, 5) == pytest.approx(2 * 0.5**5)
        # b=2, c=8: 2 * sum_{i<=2} C(10,i) / 2^10 = 2 * (1 + 10 + 45) / 1024
        assert mcnemar_exact_p(2, 8) == pytest.approx(2 * 56 / 1024)

    def test_is_symmetric_and_capped_at_one(self) -> None:
        assert mcnemar_exact_p(3, 3) == 1.0
        assert mcnemar_exact_p(7, 2) == mcnemar_exact_p(2, 7)


class TestPairedComparison:
    def test_counts_and_difference(self) -> None:
        gold = ["a", "a", "b", "b"]
        pred_a = ["a", "a", "a", "b"]  # 3 right
        pred_b = ["a", "b", "b", "a"]  # 2 right
        result = paired_comparison(gold, pred_a, pred_b, seed=1, n_boot=200)
        assert result["n"] == 4
        assert result["accuracy_a"] == pytest.approx(0.75)
        assert result["accuracy_b"] == pytest.approx(0.5)
        assert result["difference"] == pytest.approx(0.25)
        assert result["both_correct"] == 1
        assert result["only_a_correct"] == 2
        assert result["only_b_correct"] == 1
        assert result["both_wrong"] == 0
        assert result["agreement"] == pytest.approx(1 / 4)
        assert result["mcnemar_p"] == pytest.approx(mcnemar_exact_p(2, 1))
        low, high = result["difference_ci95"]
        assert low <= result["difference"] <= high

    def test_bootstrap_is_reproducible_for_a_seed(self) -> None:
        gold = ["a", "b"] * 20
        pred_a = ["a"] * 40
        pred_b = ["b"] * 40
        first = paired_comparison(gold, pred_a, pred_b, seed=7, n_boot=300)
        second = paired_comparison(gold, pred_a, pred_b, seed=7, n_boot=300)
        assert first["difference_ci95"] == second["difference_ci95"]

    def test_empty(self) -> None:
        result = paired_comparison([], [], [], seed=1)
        assert result["n"] == 0
        assert result["difference"] is None


class TestPercentile:
    def test_linear_interpolation(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([10.0], 95) == 10.0
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 100) == 5.0

    def test_empty_is_none(self) -> None:
        assert percentile([], 50) is None


class TestEdges:
    def test_a_prediction_outside_the_label_set_counts_against_macro_f1(self) -> None:
        # Pinned: an unexpected label is scored like sklearn scores it, F1 = 0.
        result = classification_metrics(["a", "b"], ["a", "d"], labels=["a", "b"])
        assert result["macro_f1"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)

    def test_a_tie_in_the_distribution_goes_to_the_smallest_key(self) -> None:
        assert top_choice({"b": 0.5, "a": 0.5, "c": 0.0}) == "a"
        assert top_choice({"ab": 0.4, "a": 0.4, "z": 0.2}) == "a"

    def test_a_probability_on_a_bin_edge_goes_to_the_lower_bin(self) -> None:
        # 0.8 belongs to (0.7, 0.8]; with 0.75 beside it the bin mean is 0.775.
        result = probabilistic_metrics(
            ["a", "a"], [{"a": 0.8, "b": 0.2}, {"a": 0.75, "b": 0.25}], labels=["a", "b"], bins=10
        )
        assert result["ece"] == pytest.approx(1.0 - 0.775)

    def test_mcnemar_values(self) -> None:
        assert mcnemar_exact_p(2, 1) == 1.0
        assert mcnemar_exact_p(7, 2) == pytest.approx(2 * (1 + 9 + 36) / 512)

    def test_a_constant_difference_has_a_zero_width_interval(self) -> None:
        # a is right and b wrong on every pair: every resample differs by exactly 1.
        result = paired_comparison(["a"] * 10, ["a"] * 10, ["b"] * 10, seed=3, n_boot=100)
        assert result["difference_ci95"] == [1.0, 1.0]

    def test_the_interval_has_the_width_of_a_resampled_proportion(self) -> None:
        # Half the pairs favour a, half are ties: d in {1, 0}, mean 0.5, SE = 0.5/sqrt(n).
        n = 400
        gold = ["a"] * n
        pred_a = ["a"] * n
        pred_b = ["a", "b"] * (n // 2)
        low, high = paired_comparison(gold, pred_a, pred_b, seed=1, n_boot=2000)[
            "difference_ci95"
        ]
        se = 0.5 / n**0.5
        assert low == pytest.approx(0.5 - 1.96 * se, abs=0.01)
        assert high == pytest.approx(0.5 + 1.96 * se, abs=0.01)

    def test_percentile_zero_is_the_minimum(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 0) == 1.0
