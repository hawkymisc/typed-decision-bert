"""Procedural contract validator (spec 5.2, 5.3, appendix A; POC_DESIGN 4.2).

Appendix A is the structural contract; this module decides the same thing on already
parsed Python values and reports the path of the first violation. Nothing is coerced:
a boolean never becomes a number and a number never becomes a string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jevbert.api.errors import ErrorPath, ValidationError
from jevbert.config import Limits

#: ``Content`` and ``Entry`` from spec 5.2.
Content = str | list[Any] | dict[str, Any]
Entry = Content | None

QUESTION_TYPES = ("noul", "choice", "score")
_COMMON_FIELDS = frozenset({"type", "instructions", "criteria"})
_TOP_LEVEL_FIELDS = ("model", "state", "questions")
_NOUL_CRITERIA_KEYS = frozenset({"true", "false"})


@dataclass(frozen=True)
class NoulQuestion:
    question_id: str
    instructions: Entry
    #: ``None`` means "absent or null"; the serializer substitutes the default sentence.
    criteria_true: Entry
    criteria_false: Entry
    type: str = "noul"


@dataclass(frozen=True)
class ChoiceQuestion:
    question_id: str
    instructions: Entry
    #: ``(key, description)`` in request order. The description may be ``None``.
    criteria: tuple[tuple[str, Entry], ...]
    type: str = "choice"


@dataclass(frozen=True)
class ScoreQuestion:
    question_id: str
    instructions: Entry
    #: Level descriptions in array order; the index is the level number.
    criteria: tuple[Content, ...]
    type: str = "score"


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class ValidatedRequest:
    model: str
    state: Content
    questions: tuple[Question, ...]


def validate_request(value: Any, limits: Limits) -> ValidatedRequest:
    """Validate a parsed body, raising ``ValidationError`` on the first violation."""
    if not isinstance(value, dict):
        raise ValidationError("The request body must be a JSON object.", path=[])

    for field in _TOP_LEVEL_FIELDS:
        if field not in value:
            raise ValidationError(f"`{field}` is required.", path=[field])
    for field in value:
        if field not in _TOP_LEVEL_FIELDS:
            raise ValidationError(f"Unknown field `{field}`.", path=[field])

    model = value["model"]
    if not isinstance(model, str) or not model:
        raise ValidationError("`model` must be a non-empty string.", path=["model"])

    state = value["state"]
    if not _is_content(state):
        raise ValidationError(
            "`state` must be a string, an array or an object.", path=["state"]
        )

    raw_questions = value["questions"]
    if not isinstance(raw_questions, dict):
        raise ValidationError("`questions` must be an object.", path=["questions"])
    if not 1 <= len(raw_questions) <= limits.max_questions:
        raise ValidationError(
            f"`questions` must hold between 1 and {limits.max_questions} entries.",
            path=["questions"],
        )

    questions = tuple(
        _validate_question(question_id, raw, limits)
        for question_id, raw in raw_questions.items()
    )
    return ValidatedRequest(model=model, state=state, questions=questions)


def _validate_question(question_id: str, raw: Any, limits: Limits) -> Question:
    base: ErrorPath = ["questions", question_id]
    if not isinstance(raw, dict):
        raise ValidationError("A question must be an object.", path=base)

    question_type = raw.get("type")
    if question_type not in QUESTION_TYPES:
        raise ValidationError(
            f"`type` must be one of {', '.join(QUESTION_TYPES)}.", path=[*base, "type"]
        )
    for field in raw:
        if field not in _COMMON_FIELDS:
            raise ValidationError(f"Unknown field `{field}`.", path=[*base, field])

    instructions = raw.get("instructions")
    if not _is_entry(instructions):
        raise ValidationError(
            "`instructions` must be a string, an array, an object or null.",
            path=[*base, "instructions"],
        )

    if question_type == "noul":
        return _validate_noul(question_id, raw, instructions, base)
    if question_type == "choice":
        return _validate_choice(question_id, raw, instructions, base, limits)
    return _validate_score(question_id, raw, instructions, base, limits)


def _validate_noul(
    question_id: str, raw: dict[str, Any], instructions: Entry, base: ErrorPath
) -> NoulQuestion:
    criteria = raw.get("criteria")
    if criteria is None:
        return NoulQuestion(question_id, instructions, None, None)
    if not isinstance(criteria, dict):
        raise ValidationError(
            "Noul `criteria` must be an object or null.", path=[*base, "criteria"]
        )
    for key in criteria:
        if key not in _NOUL_CRITERIA_KEYS:
            raise ValidationError(
                'Noul `criteria` only accepts the keys "true" and "false".',
                path=[*base, "criteria", key],
            )
    sides: dict[str, Entry] = {}
    for key in ("true", "false"):
        side = criteria.get(key)
        if not _is_entry(side):
            raise ValidationError(
                f'Noul criterion "{key}" must be a string, an array, an object or null.',
                path=[*base, "criteria", key],
            )
        sides[key] = side
    return NoulQuestion(question_id, instructions, sides["true"], sides["false"])


def _validate_choice(
    question_id: str,
    raw: dict[str, Any],
    instructions: Entry,
    base: ErrorPath,
    limits: Limits,
) -> ChoiceQuestion:
    if "criteria" not in raw:
        raise ValidationError("Choice `criteria` is required.", path=[*base, "criteria"])
    criteria = raw["criteria"]
    if not isinstance(criteria, dict):
        raise ValidationError("Choice `criteria` must be an object.", path=[*base, "criteria"])
    if not limits.min_choice_options <= len(criteria) <= limits.max_choice_options:
        raise ValidationError(
            f"Choice `criteria` must hold between {limits.min_choice_options} and "
            f"{limits.max_choice_options} options.",
            path=[*base, "criteria"],
        )
    for key, description in criteria.items():
        if not _is_entry(description):
            raise ValidationError(
                "A Choice option description must be a string, an array, an object or null.",
                path=[*base, "criteria", key],
            )
    return ChoiceQuestion(question_id, instructions, tuple(criteria.items()))


def _validate_score(
    question_id: str,
    raw: dict[str, Any],
    instructions: Entry,
    base: ErrorPath,
    limits: Limits,
) -> ScoreQuestion:
    if "criteria" not in raw:
        raise ValidationError("Score `criteria` is required.", path=[*base, "criteria"])
    criteria = raw["criteria"]
    if not isinstance(criteria, list):
        raise ValidationError("Score `criteria` must be an array.", path=[*base, "criteria"])
    if not limits.min_score_levels <= len(criteria) <= limits.max_score_levels:
        raise ValidationError(
            f"Score `criteria` must hold between {limits.min_score_levels} and "
            f"{limits.max_score_levels} levels.",
            path=[*base, "criteria"],
        )
    for index, level in enumerate(criteria):
        # U02: a null level is refused even though the Advanced docs allow it.
        if not _is_content(level):
            raise ValidationError(
                "A Score level must be a string, an array or an object.",
                path=[*base, "criteria", index],
            )
    return ScoreQuestion(question_id, instructions, tuple(criteria))


def _is_content(value: Any) -> bool:
    return isinstance(value, str | list | dict)


def _is_entry(value: Any) -> bool:
    return value is None or _is_content(value)
