"""Response construction and the I01-I09 invariant check (spec 5.5; POC_DESIGN 7).

Every successful body is built here and checked here. A numeric anomaly is a 500, never
a uniform distribution or a 0.5 dressed up as an answer (spec 4.3, N02).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from jevbert.api.errors import InferenceError
from jevbert.compiler.compiled import EncodedRequest
from jevbert.scoring.numeric import (
    NOUL_OPTION_ORDER,
    choice_scores,
    expected_score,
    noul_probability,
    score_scores,
    select_choice,
)

#: Tolerances from spec 5.5 (I04, I06).
SUM_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-6

_ANSWER_FIELDS: dict[str, frozenset[str]] = {
    "noul": frozenset({"type", "noul"}),
    "choice": frozenset({"type", "choice", "probabilities", "confidence"}),
    "score": frozenset({"type", "score", "legend", "probabilities", "confidence"}),
}


def build_response(
    *,
    model_id: str,
    encoded: EncodedRequest,
    grouped_logits: Sequence[Sequence[float]],
    temperatures: Mapping[str, float],
) -> dict[str, Any]:
    """Build the success body and verify it before it can be sent."""
    if len(grouped_logits) != len(encoded.questions):
        raise InferenceError("The backend returned results for a different question count.")

    answers: dict[str, Any] = {}
    for question, logits in zip(encoded.questions, grouped_logits, strict=True):
        compiled = question.compiled
        if len(logits) != len(compiled.option_keys):
            raise InferenceError(
                f"The backend returned {len(logits)} logits for "
                f"{len(compiled.option_keys)} candidates."
            )
        temperature = temperatures.get(compiled.question_type, 1.0)
        try:
            answers[compiled.question_id] = _build_answer(compiled, logits, temperature)
        except ValueError as exc:
            # Non-finite logits and broken normalisers land here (CT08).
            raise InferenceError(
                f"The {compiled.question_type} answer could not be computed: {exc}"
            ) from exc

    payload = {
        "model": model_id,
        "answers": answers,
        "usage": {"input_tokens": encoded.total_tokens, "output_tokens": 0},
    }
    verify_invariants(payload, encoded)
    return payload


def _build_answer(compiled: Any, logits: Sequence[float], temperature: float) -> dict[str, Any]:
    if compiled.question_type == "noul":
        # I08: Noul carries no confidence.
        return {"type": "noul", "noul": noul_probability(logits, temperature)}

    if compiled.question_type == "choice":
        choice, by_key, confidence = choice_scores(compiled.option_keys, logits, temperature)
        # Wire order follows the request, not the code point order fed to the model.
        probabilities = {key: by_key[key] for key in compiled.output_keys}
        return {
            "type": "choice",
            "choice": choice,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    if compiled.question_type == "score":
        score, distribution, confidence = score_scores(logits, temperature)
        legend = compiled.legend or ()
        return {
            "type": "score",
            "score": score,
            # I07: the original level descriptions keep their JSON type.
            "legend": {str(index): value for index, value in enumerate(legend)},
            "probabilities": {
                str(index): value for index, value in enumerate(distribution)
            },
            "confidence": confidence,
        }

    raise InferenceError(f"Unsupported question type: {compiled.question_type}")


def verify_invariants(payload: Mapping[str, Any], encoded: EncodedRequest) -> None:
    """Check I01-I09. Any violation is a 500, never a silently shipped body."""
    answers = payload["answers"]
    compiled_by_id = {q.compiled.question_id: q.compiled for q in encoded.questions}

    # I01
    if set(answers) != set(compiled_by_id):
        raise InferenceError("The answer IDs do not match the question IDs.")

    for question_id, answer in answers.items():
        compiled = compiled_by_id[question_id]
        # I02
        if answer.get("type") != compiled.question_type:
            raise InferenceError("An answer type does not match its question type.")
        # I09
        if set(answer) != _ANSWER_FIELDS[compiled.question_type]:
            raise InferenceError("An answer carries fields outside the contract.")

        if compiled.question_type == "noul":
            _check_probability(answer["noul"], "noul")
            continue

        probabilities = answer["probabilities"]
        keys = tuple(probabilities)
        if keys != compiled.output_keys:
            raise InferenceError("The answer candidates do not match the question candidates.")
        for key in keys:
            _check_probability(probabilities[key], f"probabilities[{key}]")
        _check_probability(answer["confidence"], "confidence")

        # I04
        total = math.fsum(probabilities.values())
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise InferenceError(f"The distribution sums to {total}, not to one.")

        if compiled.question_type == "choice":
            # I05
            if answer["choice"] != select_choice(dict(probabilities)):
                raise InferenceError("The selected choice is not the argmax of the distribution.")
        else:
            # I06
            values = list(probabilities.values())
            if abs(answer["score"] - expected_score(values)) > SCORE_TOLERANCE:
                raise InferenceError(
                    "The score does not match the expectation of the distribution."
                )
            # I07
            legend = answer["legend"]
            if tuple(legend) != compiled.output_keys:
                raise InferenceError("The legend keys do not match the score levels.")
            for index, original in enumerate(compiled.legend or ()):
                if not _same_json(legend[str(index)], original):
                    raise InferenceError("The legend does not preserve the original rubric.")

    usage = payload["usage"]
    if usage["input_tokens"] != encoded.total_tokens or usage["output_tokens"] != 0:
        raise InferenceError("The usage counters do not match the compiled request.")


def _check_probability(value: Any, name: str) -> None:
    # I03
    if isinstance(value, bool) or not isinstance(value, float):
        raise InferenceError(f"{name} must be a float.")
    if not math.isfinite(value):
        raise InferenceError(f"{name} is not finite.")
    if not 0.0 <= value <= 1.0:
        raise InferenceError(f"{name} is outside [0, 1].")


def _same_json(left: Any, right: Any) -> bool:
    """Value and type equality, so that ``True`` never passes for ``1``."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def noul_option_order() -> tuple[str, ...]:
    """Exposed so that tests can assert the false/true order is never flipped."""
    return NOUL_OPTION_ORDER
