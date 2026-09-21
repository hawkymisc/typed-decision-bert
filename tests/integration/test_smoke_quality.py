"""The smoke evaluation as a regression guard (POC_DESIGN 8.4).

These floors are **not** a quality bar. The numbers behind them are a trend on 52
hand-written examples and say nothing about whether the bundle is any good (spec 7.7,
14.1). What they catch is a change that makes the answers worse - a broken escape, a
template edit, a dtype change, a compiler regression - which would otherwise pass every
structural test in the suite untouched, because every one of those tests is satisfied
by a well-formed wrong answer.

A floor only catches that if it is above what an answer-free predictor would score
(Q-H1). The first floors were the first measurement minus 0.15, and mutation testing
showed what that bought: a Noul backend pinned to the constant 0.99 scored exactly
0.500 on this balanced fixture and passed, and a Score backend that always answered the
middle level scored 0.375 and passed. Both floors now sit between the uninformative
baseline and the first measurement, so a predictor that ignores the question fails.
The baselines are computed from the fixture rather than written down, so adding cases
cannot quietly move them.

**The floors are never lowered to make a run pass.** A number below a floor is a
finding to investigate and record, not a line to edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jevbert.backends.nli import NliZeroShotBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.config import Limits
from jevbert.evaluation.smoke import (
    SmokeCase,
    SmokeReport,
    load_cases,
    pipeline_answerer,
    run_smoke,
    uninformative_baselines,
)
from tests.conftest import load_nli_backend, require_nli_weights

pytestmark = pytest.mark.model

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"

#: First measurement, 2026-09-21, nli-template-v1, float32 (POC_RESULTS 4.1).
BASELINE = {"noul": 0.650, "choice": 1.000, "score": 0.261}

#: Between the uninformative baseline and the first measurement, in that order:
#: Noul 0.500 < 0.55 <= 0.650, Choice 0.25 < 0.85 <= 1.000, Score 0.261 <= 0.36 < 0.375.
#: Choice keeps its original floor - the majority-class predictor is already far below
#: it - while Noul and Score are *tightened* from the first release's 0.50 / 0.41.
NOUL_FLOOR = 0.55
CHOICE_FLOOR = 0.85
SCORE_ERROR_CEILING = 0.36

#: How far a single language may fall below the overall floor before it counts as one
#: script collapsing while the average hides it.
LANGUAGE_SLACK = 0.1


@pytest.fixture(scope="module")
def smoke_backend() -> NliZeroShotBackend:
    """A backend of this module's own (A-H1).

    The regression guard must not depend on what any other test left behind, and it
    must not be the instance another test is driving from its own thread.
    """
    return load_nli_backend()


@pytest.fixture(scope="module")
def cases() -> list[SmokeCase]:
    return load_cases(FIXTURES)


@pytest.fixture(scope="module")
def baselines(cases: list[SmokeCase]) -> dict[str, float]:
    """What a predictor that never reads the question would score on this fixture."""
    return uninformative_baselines(cases)


@pytest.fixture(scope="module")
def report(smoke_backend: NliZeroShotBackend, cases: list[SmokeCase]) -> SmokeReport:
    manifest, _, _ = require_nli_weights()
    answerer = pipeline_answerer(
        smoke_backend,
        model_id=manifest.public_id,
        budget=TokenBudget(
            max_sequence_tokens=manifest.limits.max_sequence_tokens,
            max_request_tokens=manifest.limits.max_request_tokens,
        ),
        temperatures=dict(manifest.calibration.temperature),
        limits=Limits(),
    )
    return run_smoke(cases, answerer)


class TestTheFloorsBeatAnUninformativePredictor:
    """Q-H1: a floor an answer-free predictor clears is not a regression guard."""

    def test_the_noul_floor_is_above_guessing(self, baselines: dict[str, float]) -> None:
        # The fixture is 10 yes / 10 no, so a constant answer scores exactly 0.500 and
        # the old `>= 0.50` accepted it by equality.
        assert baselines["noul"] == pytest.approx(0.5)
        assert NOUL_FLOOR > baselines["noul"]

    def test_the_choice_floor_is_above_the_majority_class(
        self, baselines: dict[str, float]
    ) -> None:
        assert CHOICE_FLOOR > baselines["choice"]

    def test_the_score_ceiling_is_below_always_answering_the_middle(
        self, baselines: dict[str, float]
    ) -> None:
        assert SCORE_ERROR_CEILING < baselines["score"]

    def test_every_floor_is_still_reachable_by_the_first_measurement(self) -> None:
        # A floor above the measured value would fail on the very run it was written
        # for, which is a different mistake from a floor that is too low.
        assert BASELINE["noul"] >= NOUL_FLOOR
        assert BASELINE["choice"] >= CHOICE_FLOOR
        assert BASELINE["score"] <= SCORE_ERROR_CEILING

    def test_the_floors_were_tightened_not_loosened(self) -> None:
        # The phase 2 release shipped 0.50 / 0.85 / 0.41 (POC_RESULTS 4.2). Recording
        # the direction here makes a future loosening a visible edit to a test that
        # says loosening is not allowed.
        assert NOUL_FLOOR > 0.50
        assert CHOICE_FLOOR >= 0.85
        assert SCORE_ERROR_CEILING < 0.41


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
        assert noul is not None and noul >= NOUL_FLOOR - LANGUAGE_SLACK, report.format_report()
        assert choice is not None and choice >= CHOICE_FLOOR - LANGUAGE_SLACK, (
            report.format_report()
        )

    @pytest.mark.parametrize("language", ["ja", "en"])
    def test_neither_language_collapses_on_score(
        self, report: SmokeReport, language: str
    ) -> None:
        # Score had no per-language guard at all, so a template change that broke the
        # rubric in one script could be absorbed by the other half of the fixture.
        value = report.metric("score", language)
        assert value is not None and value <= SCORE_ERROR_CEILING + LANGUAGE_SLACK, (
            report.format_report()
        )

    def test_every_case_was_actually_answered(
        self, report: SmokeReport, cases: list[SmokeCase]
    ) -> None:
        # The floors mean nothing if the fixture set silently shrank.
        assert len(report.results) == len(cases) >= 45

    def test_the_report_states_that_it_is_not_a_quality_claim(
        self, report: SmokeReport
    ) -> None:
        summary: dict[str, Any] = report.summary()
        assert "not a quality claim" in summary["disclaimer"]

    def test_the_report_can_show_what_went_wrong(self, report: SmokeReport) -> None:
        # Whatever the run does, `failures()` has to be able to describe it: the
        # assertion messages above are the only place a future regression is explained.
        for failure in report.failures():
            assert failure.case_id
            assert (failure.correct is False) or (failure.error is not None)
