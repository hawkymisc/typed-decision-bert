"""The smoke evaluation harness (POC_DESIGN 8.4).

The harness has to be trustworthy even though its *numbers* are not a quality claim:
a metric that silently averages the wrong thing, or a fixture set that quietly lost
half its cases, would make the regression floor meaningless. None of this needs the
model - the answers are supplied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevbert.evaluation.smoke import (
    DISCLAIMER,
    QUESTION_TYPES,
    SmokeCase,
    SmokeReport,
    SmokeResult,
    load_cases,
    parse_case,
    run_smoke,
    score_case,
    uninformative_baselines,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def noul_case(expected: bool) -> SmokeCase:
    return SmokeCase("c", "ja", "noul", "s", "i?", expected)


def choice_case(expected: str) -> SmokeCase:
    return SmokeCase("c", "en", "choice", "s", "i?", expected, {"a": "A", "b": "B"})


def score_case_fixture(expected: int) -> SmokeCase:
    return SmokeCase("c", "ja", "score", "s", "i?", expected, ["low", "mid", "high"])


def choice_case_over(criteria: dict[str, Any], expected: str) -> SmokeCase:
    return SmokeCase("c", "en", "choice", "s", "i?", expected, criteria)


class TestFixtures:
    """POC_DESIGN 8.4 asks for at least 15 cases per type in both languages."""

    def test_the_fixtures_load(self) -> None:
        assert len(load_cases(FIXTURES)) >= 45

    @pytest.mark.parametrize("question_type", QUESTION_TYPES)
    def test_each_type_has_at_least_fifteen_cases(self, question_type: str) -> None:
        cases = [c for c in load_cases(FIXTURES) if c.question_type == question_type]
        assert len(cases) >= 15

    @pytest.mark.parametrize("language", ["ja", "en"])
    def test_both_languages_cover_every_type(self, language: str) -> None:
        cases = [c for c in load_cases(FIXTURES) if c.language == language]
        assert {c.question_type for c in cases} == set(QUESTION_TYPES)

    def test_noul_cases_come_in_opposing_pairs(self) -> None:
        # POC_DESIGN 8.4: the same state with a different instruction must have a
        # different answer, or the set cannot tell instruction-following from a prior.
        cases = [c for c in load_cases(FIXTURES) if c.question_type == "noul"]
        by_state: dict[str, set[bool]] = {}
        for case in cases:
            by_state.setdefault(json.dumps(case.state, ensure_ascii=False), set()).add(
                bool(case.expected)
            )
        assert by_state, "no noul cases"
        assert all(answers == {True, False} for answers in by_state.values())

    def test_choice_cases_offer_between_three_and_six_options(self) -> None:
        for case in load_cases(FIXTURES):
            if case.question_type == "choice":
                assert 3 <= len(case.criteria) <= 6, case.case_id
                assert case.expected in case.criteria, case.case_id

    def test_score_cases_have_three_to_five_levels(self) -> None:
        for case in load_cases(FIXTURES):
            if case.question_type == "score":
                assert 3 <= case.level_count <= 5, case.case_id
                assert 0 <= case.expected < case.level_count, case.case_id

    def test_every_case_id_is_unique(self) -> None:
        cases = load_cases(FIXTURES)
        assert len({case.case_id for case in cases}) == len(cases)

    def test_a_repeated_case_id_is_refused(self, tmp_path: Path) -> None:
        line = json.dumps(
            {"id": "x", "language": "ja", "type": "noul", "state": "s", "expected": True}
        )
        (tmp_path / "a.jsonl").write_text(f"{line}\n{line}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="repeats the case ID"):
            load_cases(tmp_path)

    def test_an_empty_directory_is_refused(self, tmp_path: Path) -> None:
        # A silently empty fixture set would make every metric a perfect score.
        with pytest.raises(ValueError, match="no smoke fixtures"):
            load_cases(tmp_path)

    def test_a_malformed_line_names_the_file_and_line(self, tmp_path: Path) -> None:
        (tmp_path / "a.jsonl").write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="a.jsonl:1"):
            load_cases(tmp_path)

    def test_a_case_missing_a_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            parse_case({"id": "x", "language": "ja", "type": "noul", "state": "s"})

    def test_an_unknown_question_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown question type"):
            parse_case(
                {"id": "x", "language": "ja", "type": "rank", "state": "s", "expected": 1}
            )


class TestRequestConstruction:
    def test_a_missing_instruction_is_omitted_rather_than_null(self) -> None:
        # U01: absent and null mean the same thing, and the SDK omits the field, so the
        # smoke path sends what the SDK would send.
        case = SmokeCase("c", "ja", "noul", "s", None, True)
        assert "instructions" not in case.question()

    def test_the_criteria_travel_when_present(self) -> None:
        question = choice_case("a").question()
        assert question["criteria"] == {"a": "A", "b": "B"}
        assert question["type"] == "choice"

    def test_the_request_names_the_model_and_one_question(self) -> None:
        body = noul_case(True).request("bundle-1", "only")
        assert body["model"] == "bundle-1"
        assert list(body["questions"]) == ["only"]


class TestScoring:
    @pytest.mark.parametrize(
        ("noul", "expected", "correct"),
        [(0.9, True, True), (0.1, True, False), (0.1, False, True), (0.9, False, False)],
    )
    def test_noul_uses_the_half_threshold(
        self, noul: float, expected: bool, correct: bool
    ) -> None:
        result = score_case(noul_case(expected), {"type": "noul", "noul": noul})
        assert result.correct is correct

    def test_exactly_half_counts_as_yes(self) -> None:
        # 0.5 is not a third class (spec 5.4); the threshold has to fall somewhere and
        # the choice is written down rather than left to a floating point accident.
        assert score_case(noul_case(True), {"type": "noul", "noul": 0.5}).correct is True

    def test_choice_compares_the_returned_key(self) -> None:
        answer = {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}
        assert score_case(choice_case("a"), answer).correct is True
        assert score_case(choice_case("b"), answer).correct is False

    @pytest.mark.parametrize(
        ("score", "expected", "error"),
        [(2.0, 2, 0.0), (0.0, 2, 1.0), (1.0, 2, 0.5), (1.5, 1, 0.25)],
    )
    def test_score_error_is_normalised_by_the_span(
        self, score: float, expected: int, error: float
    ) -> None:
        result = score_case(score_case_fixture(expected), {"type": "score", "score": score})
        assert result.error == pytest.approx(error)

    def test_an_answer_of_the_wrong_type_is_an_error(self) -> None:
        # A type mismatch means the harness is measuring something other than what it
        # thinks; averaging it in would hide the defect.
        with pytest.raises(ValueError, match="answered as"):
            score_case(noul_case(True), {"type": "choice", "choice": "a"})


class TestReport:
    def _report(self) -> SmokeReport:
        return SmokeReport(
            (
                SmokeResult("a", "ja", "noul", correct=True),
                SmokeResult("b", "ja", "noul", correct=False),
                SmokeResult("c", "en", "noul", correct=True),
                SmokeResult("d", "en", "choice", correct=True),
                SmokeResult("e", "ja", "score", error=0.25),
                SmokeResult("f", "en", "score", error=0.75),
            )
        )

    def test_accuracy_is_per_type(self) -> None:
        assert self._report().metric("noul") == pytest.approx(2 / 3)
        assert self._report().metric("choice") == 1.0

    def test_accuracy_is_per_language(self) -> None:
        assert self._report().metric("noul", "ja") == 0.5
        assert self._report().metric("noul", "en") == 1.0

    def test_score_reports_mean_error_not_accuracy(self) -> None:
        assert self._report().metric("score") == pytest.approx(0.5)

    def test_an_empty_slice_is_none_rather_than_zero(self) -> None:
        # Zero accuracy and "nothing was measured" must not look the same.
        assert self._report().metric("choice", "ja") is None

    def test_counts_are_reported(self) -> None:
        assert self._report().count("noul") == 3
        assert self._report().count("noul", "en") == 1

    def test_the_summary_carries_the_disclaimer(self) -> None:
        summary = self._report().summary()
        assert summary["disclaimer"] == DISCLAIMER
        assert summary["by_type"]["score"]["better"] == "lower"
        assert summary["by_type"]["noul"]["better"] == "higher"

    def test_the_text_report_carries_the_disclaimer(self) -> None:
        text = self._report().format_report()
        assert DISCLAIMER in text
        assert "noul" in text and "score" in text

    def test_failures_list_wrong_answers_and_distant_scores(self) -> None:
        # A-L: Score has no boolean to be false, so before this it contributed nothing
        # to an assertion message about a Score regression.
        assert [result.case_id for result in self._report().failures()] == ["b", "f"]

    def test_the_score_threshold_can_be_moved(self) -> None:
        assert [result.case_id for result in self._report().failures(score_error_above=0.9)] == [
            "b"
        ]
        assert [
            result.case_id for result in self._report().failures(score_error_above=0.1)
        ] == ["b", "e", "f"]

    def test_the_text_report_shows_the_score_error(self) -> None:
        text = self._report().format_report()
        assert "0.750" in text
        assert "f " in text


class TestAResultSetsExactlyOneOutcome:
    """A-L: the invariant in the docstring is now the one the constructor enforces."""

    def test_neither_is_refused(self) -> None:
        with pytest.raises(ValueError, match="neither"):
            SmokeResult("a", "ja", "noul")

    def test_both_is_refused(self) -> None:
        with pytest.raises(ValueError, match="both"):
            SmokeResult("a", "ja", "score", correct=True, error=0.1)

    def test_a_false_correct_is_still_an_outcome(self) -> None:
        # `correct=False` is falsy, so a check written with `if not correct` would read
        # a wrong answer as a missing one.
        assert SmokeResult("a", "ja", "noul", correct=False).correct is False

    def test_a_zero_error_is_still_an_outcome(self) -> None:
        assert SmokeResult("a", "ja", "score", error=0.0).error == 0.0


class TestUninformativeBaselines:
    """Q-H1: what a predictor that never reads the question would score."""

    def test_a_balanced_noul_fixture_gives_one_half(self) -> None:
        cases = [noul_case(True), noul_case(False), noul_case(True), noul_case(False)]
        assert uninformative_baselines(cases)["noul"] == pytest.approx(0.5)

    def test_an_unbalanced_noul_fixture_gives_the_majority(self) -> None:
        cases = [noul_case(True), noul_case(True), noul_case(True), noul_case(False)]
        assert uninformative_baselines(cases)["noul"] == pytest.approx(0.75)

    def test_choice_is_at_least_uniform_guessing(self) -> None:
        options = {"a": None, "b": None, "c": None, "d": None}
        # Four options, and every key is right exactly once, so no constant answer beats
        # guessing and the baseline is 1/K.
        cases = [choice_case_over(dict(options), key) for key in options]
        assert uninformative_baselines(cases)["choice"] == pytest.approx(0.25)

    def test_choice_notices_a_key_that_is_usually_right(self) -> None:
        options = {"a": None, "b": None, "c": None, "d": None}
        cases = [choice_case_over(dict(options), "a") for _ in range(3)] + [
            choice_case_over(dict(options), "b")
        ]
        # Always answering "a" scores 0.75, well above uniform guessing.
        assert uninformative_baselines(cases)["choice"] == pytest.approx(0.75)

    def test_score_is_the_best_constant_position(self) -> None:
        # Targets 0, 0.5, 1 normalised: the median is 0.5 and the mean distance to it
        # is 1/3. Nothing constant does better.
        cases = [score_case_fixture(level) for level in (0, 1, 2)]
        assert uninformative_baselines(cases)["score"] == pytest.approx(1 / 3)

    def test_a_type_with_no_cases_has_no_baseline(self) -> None:
        assert "score" not in uninformative_baselines([noul_case(True)])

    def test_the_shipped_fixture_baselines_are_the_recorded_ones(self) -> None:
        # POC_RESULTS 4.2 publishes these beside the measurements, so a fixture change
        # that moves them has to move the document too.
        cases = load_cases(FIXTURES)
        baselines = uninformative_baselines(cases)
        assert baselines["noul"] == pytest.approx(0.500)
        assert baselines["choice"] == pytest.approx(0.260, abs=5e-4)
        assert baselines["score"] == pytest.approx(0.375)


class TestRunSmoke:
    def test_it_asks_about_every_case_in_order(self) -> None:
        cases = [noul_case(True), noul_case(False)]
        asked: list[Any] = []

        def answerer(case: SmokeCase) -> dict[str, Any]:
            asked.append(case)
            return {"type": "noul", "noul": 0.9}

        report = run_smoke(cases, answerer)
        assert asked == cases
        assert [result.correct for result in report.results] == [True, False]
