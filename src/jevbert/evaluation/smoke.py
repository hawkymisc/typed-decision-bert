"""PoC quality smoke evaluation (POC_DESIGN 8.4; spec 7.7, 13.4).

**This is a record of a trend, not a quality claim.** The fixtures are a few dozen
hand-written examples, they were written by the same person who chose the template, and
they carry no inter-annotator agreement, no sampling frame and no confidence interval.
Nothing here may be cited as evidence for G2, and no number here says the bundle is fit
for any purpose (spec 14.1). What it *is* good for is noticing that something moved:
the numbers are pinned in an integration test so that a change to the compiler, the
template or the backend cannot quietly make the answers worse.

Metrics, per question type and per language:

* Noul - accuracy at the 0.5 threshold. Higher is better.
* Choice - accuracy of the returned ``choice``. Higher is better.
* Score - mean of ``|score - expected| / (K - 1)``. **Lower** is better.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevbert.backends.base import Backend, CancelToken
from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.serializer_nli import (
    DEFAULT_TEMPLATE,
    HypothesisTemplate,
    all_sequences,
    compile_and_encode,
    split_by_question,
)
from jevbert.config import Limits
from jevbert.contracts.response import build_response
from jevbert.contracts.validator import validate_request

#: Cases are answered through the same path the API uses, so an answer is the wire
#: object of spec 5.4.
Answer = dict[str, Any]
Answerer = Callable[["SmokeCase"], Answer]

DISCLAIMER = (
    "This smoke evaluation is a record of a trend on a few dozen hand-written "
    "examples. It is not a quality claim, not a calibration check and not evidence "
    "for gate G2 (spec 7.7, 14.1)."
)

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class SmokeCase:
    """One fixture line: a question, and what a careful reader would answer."""

    case_id: str
    language: str
    question_type: str
    state: Any
    instructions: Any
    expected: Any
    criteria: Any = None

    @property
    def level_count(self) -> int:
        if self.question_type != "score":
            raise ValueError("level_count is only defined for a Score case")
        return len(self.criteria)

    def question(self) -> dict[str, Any]:
        question: dict[str, Any] = {"type": self.question_type}
        if self.instructions is not None:
            question["instructions"] = self.instructions
        if self.criteria is not None:
            question["criteria"] = self.criteria
        return question

    def request(self, model: str, question_id: str = "q") -> dict[str, Any]:
        return {"model": model, "state": self.state, "questions": {question_id: self.question()}}


def parse_case(payload: dict[str, Any]) -> SmokeCase:
    missing = {"id", "language", "type", "state", "expected"} - set(payload)
    if missing:
        raise ValueError(f"a smoke case is missing {', '.join(sorted(missing))}")
    question_type = payload["type"]
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {question_type!r}")
    return SmokeCase(
        case_id=payload["id"],
        language=payload["language"],
        question_type=question_type,
        state=payload["state"],
        instructions=payload.get("instructions"),
        expected=payload["expected"],
        criteria=payload.get("criteria"),
    )


def load_cases(directory: Path) -> list[SmokeCase]:
    """Read every ``*.jsonl`` fixture in ``directory``, sorted by file then by line."""
    cases: list[SmokeCase] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number} is not valid JSON") from exc
            case = parse_case(payload)
            if case.case_id in seen:
                raise ValueError(f"{path.name}:{number} repeats the case ID {case.case_id!r}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no smoke fixtures found in {directory}")
    return cases


@dataclass(frozen=True)
class SmokeResult:
    """One case's outcome. Exactly one of ``correct`` and ``error`` is set.

    Enforced rather than documented (A-L): ``metric`` averages one field and
    ``failures`` reads the other, so a result carrying neither is silently dropped from
    an accuracy and a result carrying both is counted twice over.
    """

    case_id: str
    language: str
    question_type: str
    #: Noul and Choice: did the answer match? ``None`` for Score.
    correct: bool | None = None
    #: Score: ``|score - expected| / (K - 1)``. ``None`` for Noul and Choice.
    error: float | None = None

    def __post_init__(self) -> None:
        if (self.correct is None) == (self.error is None):
            raise ValueError(
                f"{self.case_id}: a result sets exactly one of correct and error, "
                f"not {'both' if self.correct is not None else 'neither'}"
            )


def score_case(case: SmokeCase, answer: Answer) -> SmokeResult:
    """Turn one wire answer into a result, refusing an answer of the wrong type."""
    if answer.get("type") != case.question_type:
        raise ValueError(
            f"{case.case_id}: answered as {answer.get('type')!r}, expected "
            f"{case.question_type!r}"
        )
    if case.question_type == "noul":
        return SmokeResult(
            case.case_id,
            case.language,
            case.question_type,
            correct=(answer["noul"] >= 0.5) is bool(case.expected),
        )
    if case.question_type == "choice":
        return SmokeResult(
            case.case_id,
            case.language,
            case.question_type,
            correct=answer["choice"] == case.expected,
        )
    span = case.level_count - 1
    return SmokeResult(
        case.case_id,
        case.language,
        case.question_type,
        error=abs(answer["score"] - float(case.expected)) / span,
    )


@dataclass(frozen=True)
class SmokeReport:
    results: tuple[SmokeResult, ...]

    def select(
        self, question_type: str | None = None, language: str | None = None
    ) -> list[SmokeResult]:
        return [
            result
            for result in self.results
            if (question_type is None or result.question_type == question_type)
            and (language is None or result.language == language)
        ]

    def languages(self) -> list[str]:
        return sorted({result.language for result in self.results})

    def metric(self, question_type: str, language: str | None = None) -> float | None:
        """Accuracy for Noul and Choice, mean normalised error for Score.

        ``None`` when there is nothing to average, so that an empty slice is visibly
        empty rather than a confident zero.
        """
        selected = self.select(question_type, language)
        if not selected:
            return None
        if question_type == "score":
            return statistics.fmean(
                result.error for result in selected if result.error is not None
            )
        return statistics.fmean(
            1.0 if result.correct else 0.0 for result in selected
        )

    def count(self, question_type: str, language: str | None = None) -> int:
        return len(self.select(question_type, language))

    def failures(self, *, score_error_above: float = 0.5) -> list[SmokeResult]:
        """Cases worth looking at: a wrong answer, or a Score a long way off.

        Score had no failures at all before, because it has no boolean to be false, so
        an assertion message about a Score regression named no case (A-L). Half the
        rubric's span is the default line: at that distance the answer is closer to a
        different level than to the right one.
        """
        return [
            result
            for result in self.results
            if result.correct is False
            or (result.error is not None and result.error > score_error_above)
        ]

    def summary(self) -> dict[str, Any]:
        """A JSON-serialisable summary, disclaimer included."""
        languages = self.languages()
        by_type: dict[str, Any] = {}
        for question_type in QUESTION_TYPES:
            if not self.count(question_type):
                continue
            by_type[question_type] = {
                "metric": "mean_normalised_error" if question_type == "score" else "accuracy",
                "better": "lower" if question_type == "score" else "higher",
                "overall": self.metric(question_type),
                "count": self.count(question_type),
                "by_language": {
                    language: {
                        "value": self.metric(question_type, language),
                        "count": self.count(question_type, language),
                    }
                    for language in languages
                    if self.count(question_type, language)
                },
            }
        return {"disclaimer": DISCLAIMER, "cases": len(self.results), "by_type": by_type}

    def format_report(self) -> str:
        lines = [f"smoke evaluation: {len(self.results)} cases", DISCLAIMER, ""]
        languages = self.languages()
        header = f"{'type':<8}{'metric':<24}{'all':>10}" + "".join(
            f"{language:>10}" for language in languages
        )
        lines.append(header)
        lines.append("-" * len(header))
        for question_type in QUESTION_TYPES:
            if not self.count(question_type):
                continue
            metric = "mean |err| / (K-1)" if question_type == "score" else "accuracy"
            row = f"{question_type:<8}{metric:<24}{_cell(self.metric(question_type)):>10}"
            for language in languages:
                row += f"{_cell(self.metric(question_type, language)):>10}"
            lines.append(row)
        lines.append("")
        for question_type in QUESTION_TYPES:
            counts = ", ".join(
                f"{language}={self.count(question_type, language)}"
                for language in languages
                if self.count(question_type, language)
            )
            if counts:
                lines.append(f"{question_type}: {self.count(question_type)} cases ({counts})")
        failures = self.failures()
        if failures:
            lines.append("")
            lines.append(f"{len(failures)} case(s) to look at:")
            for result in failures:
                if result.error is None:
                    lines.append(f"  {result.case_id:<28} {result.question_type} answered wrongly")
                else:
                    lines.append(
                        f"  {result.case_id:<28} {result.question_type} "
                        f"|err|/(K-1) = {result.error:.3f}"
                    )
        return "\n".join(lines)


def uninformative_baselines(cases: Sequence[SmokeCase]) -> dict[str, float]:
    """What the best predictor that never reads the question scores on these cases.

    A floor below this number is not a regression guard: mutation testing showed a Noul
    backend pinned to a constant and a Score backend pinned to the middle level passing
    every floor the first release shipped (Q-H1). Computed from the fixture rather than
    written down, so adding cases moves the baseline instead of quietly invalidating it.

    * **noul** - the better of always-yes and always-no.
    * **choice** - the better of guessing uniformly (``mean 1/K``) and always naming the
      single key that is most often right.
    * **score** - the smallest mean ``|score - expected| / (K - 1)`` any constant answer
      can reach. A constant is a fixed *position* in the rubric, because ``K`` varies;
      the minimum is at the median of the normalised targets. **Lower is better here**,
      so a ceiling has to sit below this number.
    """
    baselines: dict[str, float] = {}

    noul = [case for case in cases if case.question_type == "noul"]
    if noul:
        yes = sum(1 for case in noul if bool(case.expected))
        baselines["noul"] = max(yes, len(noul) - yes) / len(noul)

    choice = [case for case in cases if case.question_type == "choice"]
    if choice:
        guessing = statistics.fmean(1.0 / len(case.criteria) for case in choice)
        keys = {key for case in choice for key in case.criteria}
        constant = max(
            (
                statistics.fmean(1.0 if case.expected == key else 0.0 for case in choice)
                for key in keys
            ),
            default=0.0,
        )
        baselines["choice"] = max(guessing, constant)

    score = [case for case in cases if case.question_type == "score"]
    if score:
        targets = [float(case.expected) / (case.level_count - 1) for case in score]
        best = statistics.median(targets)
        baselines["score"] = statistics.fmean(abs(best - target) for target in targets)

    return baselines


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def run_smoke(cases: Iterable[SmokeCase], answer_for: Answerer) -> SmokeReport:
    """Ask ``answer_for`` about every case and collect the results."""
    return SmokeReport(tuple(score_case(case, answer_for(case)) for case in cases))


def compare_templates(
    cases: Sequence[SmokeCase], answerers: dict[str, Answerer]
) -> dict[str, SmokeReport]:
    """Run the same cases through several answerers (K4, POC_DESIGN 5.3)."""
    return {name: run_smoke(cases, answerer) for name, answerer in answerers.items()}


def pipeline_answerer(
    backend: Backend,
    *,
    model_id: str,
    budget: TokenBudget,
    temperatures: dict[str, float],
    limits: Limits,
    template: HypothesisTemplate = DEFAULT_TEMPLATE,
    question_id: str = "q",
) -> Answerer:
    """Answer a case through the server's own compile - encode - score - build path.

    Not a shortcut past the API: the same validator, the same serializer, the same
    scoring and the same invariant check run, so a smoke number cannot be better than
    what the endpoint would return. Only HTTP is missing.
    """

    def answer(case: SmokeCase) -> Answer:
        validated = validate_request(case.request(model_id, question_id), limits)
        encoded = compile_and_encode(
            validated,
            backend,
            budget,
            max_request_chars=limits.effective_max_request_chars,
            template=template,
        )
        logits = backend.score(all_sequences(encoded), CancelToken())
        payload = build_response(
            model_id=model_id,
            encoded=encoded,
            grouped_logits=split_by_question(encoded, logits),
            temperatures=temperatures,
        )
        return payload["answers"][question_id]

    return answer
