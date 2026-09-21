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
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def noul_case(expected: bool) -> SmokeCase:
    return SmokeCase("c", "ja", "noul", "s", "i?", expected)


def choice_case(expected: str) -> SmokeCase:
    return SmokeCase("c", "en", "choice", "s", "i?", expected, {"a": "A", "b": "B"})


def score_case_fixture(expected: int) -> SmokeCase:
    return SmokeCase("c", "ja", "score", "s", "i?", expected, ["low", "mid", "high"])


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

    def test_failures_list_only_wrong_answers(self) -> None:
        assert [result.case_id for result in self._report().failures()] == ["b"]


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
