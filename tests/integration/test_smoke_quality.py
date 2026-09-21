"""The smoke evaluation as a regression guard (POC_DESIGN 8.4).

These floors are **not** a quality bar. The numbers behind them are a trend on 52
hand-written examples and say nothing about whether the bundle is any good (spec 7.7,
14.1). What they catch is a change that makes the answers worse - a broken escape, a
template edit, a dtype change, a compiler regression - which would otherwise pass every
structural test in the suite untouched, because every one of those tests is satisfied
by a well-formed wrong answer.

The floors are the first measurement minus 0.15, recorded in ``docs/POC_RESULTS.md``
with the measurement itself. **They are never lowered to make a run pass.** A number
below the floor is a finding to investigate and record, not a line to edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jevbert.backends.nli import NliZeroShotBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.config import Limits
from jevbert.evaluation.smoke import SmokeReport, load_cases, pipeline_answerer, run_smoke
from tests.conftest import require_nli_weights

pytestmark = pytest.mark.model

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"

#: First measurement, 2026-09-21, nli-template-v1, float32 (POC_RESULTS).
BASELINE = {"noul": 0.650, "choice": 1.000, "score": 0.261}

#: Baseline -/+ 0.15. Accuracy floors; the Score entry is a ceiling on the error.
NOUL_FLOOR = 0.50
CHOICE_FLOOR = 0.85
SCORE_ERROR_CEILING = 0.41


@pytest.fixture(scope="module")
def report(nli_backend: NliZeroShotBackend) -> SmokeReport:
    manifest, _, _ = require_nli_weights()
    answerer = pipeline_answerer(
        nli_backend,
        model_id=manifest.public_id,
        budget=TokenBudget(
            max_sequence_tokens=manifest.limits.max_sequence_tokens,
            max_request_tokens=manifest.limits.max_request_tokens,
        ),
        temperatures=dict(manifest.calibration.temperature),
        limits=Limits(),
    )
    return run_smoke(load_cases(FIXTURES), answerer)


class TestSmokeRegressionFloors:
    def test_noul_accuracy_is_above_the_floor(self, report: SmokeReport) -> None:
        value = report.metric("noul")
        assert value is not None and value >= NOUL_FLOOR, report.format_report()

    def test_choice_accuracy_is_above_the_floor(self, report: SmokeReport) -> None:
        value = report.metric("choice")
        assert value is not None and value >= CHOICE_FLOOR, report.format_report()

    def test_score_error_is_below_the_ceiling(self, report: SmokeReport) -> None:
        value = report.metric("score")
        assert value is not None and value <= SCORE_ERROR_CEILING, report.format_report()

    @pytest.mark.parametrize("language", ["ja", "en"])
    def test_neither_language_collapses(self, report: SmokeReport, language: str) -> None:
        # A template or an escape that works in one script and not the other would
        # otherwise hide inside an average over both.
        noul = report.metric("noul", language)
        choice = report.metric("choice", language)
        assert noul is not None and noul >= NOUL_FLOOR - 0.1, report.format_report()
        assert choice is not None and choice >= CHOICE_FLOOR - 0.1, report.format_report()

    def test_every_case_was_actually_answered(self, report: SmokeReport) -> None:
        # The floors mean nothing if the fixture set silently shrank.
        assert len(report.results) == len(load_cases(FIXTURES)) >= 45

    def test_the_report_states_that_it_is_not_a_quality_claim(
        self, report: SmokeReport
    ) -> None:
        summary: dict[str, Any] = report.summary()
        assert "not a quality claim" in summary["disclaimer"]
