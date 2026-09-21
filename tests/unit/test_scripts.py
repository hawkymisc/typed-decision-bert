"""The scripts that produce the evidence for AC2 and AC6 (Q-M4).

``scripts/sdk_demo.py`` is where I01-I09 are declared satisfied and
``scripts/bench_latency.py`` is where p95 is declared measured, and POC_RESULTS quotes
both. Neither had a test, so the two functions that decide what those documents say -
the invariant checker and the percentile - were the only untested code producing
published numbers. No server and no model is needed: both take data.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from scripts.bench_latency import (
    MAX_SEQUENCE_TOKENS,
    OPTION_COUNT,
    QUESTION_COUNT,
    _percentile,
    build_premise_probe,
    longest_sequence_upper_bound,
    premise_sequence_tokens,
)
from scripts.sdk_demo import check_invariants

#: A body of the shape the demo sends, with answers that satisfy every invariant.
GOOD: dict[str, Any] = {
    "model": "jevbert-poc-nli-ja-en-0.2.0",
    "answers": {
        "refund_requested": {"type": "noul", "noul": 0.6689449430817159},
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.95, "technical": 0.01, "other": 0.04},
            "confidence": 0.8,
        },
        "urgency": {
            "type": "score",
            "score": 1.9,
            "legend": {
                "0": "対応期限の指定がない",
                "1": "数日以内の対応を求めている",
                "2": "当日中または直ちに対応することを求めている",
            },
            "probabilities": {"0": 0.05, "1": 0.0, "2": 0.95},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 476, "output_tokens": 0},
}


def _mutate(**changes: Any) -> dict[str, Any]:
    payload = json.loads(json.dumps(GOOD, ensure_ascii=False))
    for path, value in changes.items():
        cursor: Any = payload
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        if value is _DELETE:
            del cursor[parts[-1]]
        else:
            cursor[parts[-1]] = value
    return payload


_DELETE = object()


def _failed(lines: list[str]) -> list[str]:
    return [line for line in lines if "FAILED" in line]


class TestCheckInvariants:
    """The checker has to say OK for a good body and FAILED for each broken one."""

    def test_a_correct_body_passes_every_invariant(self) -> None:
        lines = check_invariants(GOOD)
        assert len(lines) == 9
        assert _failed(lines) == []
        assert all(line.startswith("I0") for line in lines)

    def test_i01_notices_a_missing_answer(self) -> None:
        payload = _mutate(**{"answers.urgency": _DELETE})
        assert any("I01" in line for line in _failed(check_invariants(payload)))

    def test_i02_notices_an_answer_of_the_wrong_type(self) -> None:
        payload = _mutate(**{"answers.department.type": "noul"})
        assert any("I02" in line for line in _failed(check_invariants(payload)))

    @pytest.mark.parametrize("value", [1.5, -0.1])
    def test_i03_notices_a_probability_outside_the_unit_interval(self, value: float) -> None:
        payload = _mutate(**{"answers.refund_requested.noul": value})
        assert any("I03" in line for line in _failed(check_invariants(payload)))

    def test_i04_notices_a_distribution_that_does_not_sum_to_one(self) -> None:
        payload = _mutate(
            **{"answers.department.probabilities": {"billing": 0.5, "technical": 0.1, "other": 0.1}}
        )
        assert any("I04" in line for line in _failed(check_invariants(payload)))

    def test_i05_notices_a_choice_that_is_not_the_argmax(self) -> None:
        payload = _mutate(**{"answers.department.choice": "other"})
        assert any("I05" in line for line in _failed(check_invariants(payload)))

    def test_i06_notices_a_score_that_is_not_the_expected_value(self) -> None:
        payload = _mutate(**{"answers.urgency.score": 0.5})
        assert any("I06" in line for line in _failed(check_invariants(payload)))

    def test_i07_notices_a_rewritten_legend(self) -> None:
        payload = _mutate(**{"answers.urgency.legend": {"0": "a", "1": "b", "2": "c"}})
        assert any("I07" in line for line in _failed(check_invariants(payload)))

    def test_i08_notices_a_confidence_on_a_noul(self) -> None:
        payload = _mutate(**{"answers.refund_requested.confidence": 0.9})
        assert any("I08" in line for line in _failed(check_invariants(payload)))

    def test_i09_notices_an_invented_field(self) -> None:
        payload = _mutate(**{"answers.department.reasoning": "because"})
        assert any("I09" in line for line in _failed(check_invariants(payload)))

    def test_i09_notices_an_invented_top_level_field(self) -> None:
        payload = _mutate(**{"trace_id": "abc"})
        assert any("I09" in line for line in _failed(check_invariants(payload)))

    def test_one_broken_invariant_does_not_hide_the_others(self) -> None:
        # Every line is still reported, so the demo output says what held as well as
        # what did not.
        payload = _mutate(**{"answers.department.choice": "other"})
        assert len(check_invariants(payload)) == 9


class TestPercentile:
    """Nearest-rank: with 100 samples p95 is the 95th smallest, not an interpolation."""

    SAMPLES = [float(value) for value in range(1, 101)]

    def test_p95_of_a_hundred_samples_is_the_ninety_fifth(self) -> None:
        assert _percentile(self.SAMPLES, 95) == 95.0

    def test_p50_of_a_hundred_samples_is_the_fiftieth(self) -> None:
        assert _percentile(self.SAMPLES, 50) == 50.0

    def test_p99_of_a_hundred_samples_is_the_ninety_ninth(self) -> None:
        assert _percentile(self.SAMPLES, 99) == 99.0

    def test_p100_is_the_maximum(self) -> None:
        assert _percentile(self.SAMPLES, 100) == 100.0

    def test_p0_is_the_minimum_rather_than_an_index_error(self) -> None:
        assert _percentile(self.SAMPLES, 0) == 1.0

    def test_a_single_sample_is_every_percentile(self) -> None:
        assert _percentile([7.0], 95) == 7.0

    def test_ten_samples_round_to_a_rank(self) -> None:
        samples = [float(value) for value in range(1, 11)]
        assert _percentile(samples, 95) == 10.0
        assert _percentile(samples, 50) == 5.0

    def test_no_samples_is_an_error_rather_than_a_number(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            _percentile([], 95)


class TestLongestSequenceBound:
    """Q-M4: the benchmark point is per sequence, so an average does not check it."""

    def test_the_probe_halves_its_two_identical_sequences(self) -> None:
        assert premise_sequence_tokens(120) == 60

    @pytest.mark.parametrize("tokens", [0, -2, 61])
    def test_an_impossible_probe_result_is_refused(self, tokens: int) -> None:
        with pytest.raises(ValueError):
            premise_sequence_tokens(tokens)

    def test_the_probe_body_has_no_instruction_and_empty_candidates(self) -> None:
        question = build_premise_probe("m")["questions"]["probe"]
        assert question == {"type": "noul", "criteria": {"true": "", "false": ""}}

    def test_the_bound_is_the_premise_plus_every_candidate(self) -> None:
        # 2 questions x 3 options, premise 10 per sequence, candidates summing to 12.
        total = 2 * (3 * 10 + 12)
        assert longest_sequence_upper_bound(total, 2, 3, 10) == 22

    def test_the_bound_is_never_below_the_average(self) -> None:
        total = 2 * (3 * 10 + 12)
        average = total / (2 * 3)
        assert longest_sequence_upper_bound(total, 2, 3, 10) >= average

    def test_identical_candidates_still_give_a_valid_bound(self) -> None:
        # 4 candidates of 5 tokens each: the true longest is 10 + 5 = 15 and the bound
        # is 10 + 20 = 30. Loose, but it is a bound; the mean, 10 + 5, is not.
        total = 1 * (4 * 10 + 20)
        assert longest_sequence_upper_bound(total, 1, 4, 10) == 30

    def test_a_total_that_does_not_divide_by_the_questions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="divide evenly"):
            longest_sequence_upper_bound(101, 4, 8, 10)

    def test_a_premise_larger_than_the_question_is_refused(self) -> None:
        with pytest.raises(ValueError, match="premise probe"):
            longest_sequence_upper_bound(80, 1, 8, 20)

    def test_the_benchmark_point_is_the_one_the_spec_names(self) -> None:
        assert (QUESTION_COUNT, OPTION_COUNT, MAX_SEQUENCE_TOKENS) == (4, 8, 512)
