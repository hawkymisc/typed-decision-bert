"""serializer-nli-v1: question -> (premise, hypothesis) pairs (POC_DESIGN 5).

The compiler stops at text. Token counting and ID assembly belong to the backend
(``count_and_encode``), which is what lets the same compiler serve the fake backend and
a real tokenizer without knowing either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jevbert.api.errors import ContextLengthExceededError, InferenceError
from jevbert.backends.base import Backend, EncodedSequence, TextPair
from jevbert.compiler.compiled import (
    CompiledQuestion,
    CompiledRequest,
    EncodedQuestion,
    EncodedRequest,
    TokenBudget,
)
from jevbert.compiler.normalize import render
from jevbert.contracts.validator import (
    ChoiceQuestion,
    Entry,
    NoulQuestion,
    Question,
    ScoreQuestion,
    ValidatedRequest,
)
from jevbert.scoring.numeric import NOUL_OPTION_ORDER

SERIALIZER_VERSION = "serializer-nli-v1"
NLI_TEMPLATE_ID = "nli-template-v1"

#: Generic yes/no candidate text used when a Noul criterion is absent or null (spec 5.4).
DEFAULT_NOUL_TRUE = "The answer to the question is yes."
DEFAULT_NOUL_FALSE = "The answer to the question is no."


def compile_request(
    request: ValidatedRequest, *, max_request_chars: int | None = None
) -> CompiledRequest:
    """Expand every question into one (premise, hypothesis) pair per candidate.

    ``max_request_chars`` bounds the expansion before it happens (S-H2). The template
    repeats both the state and the instruction once per candidate, so a 15 KiB body
    with 32 questions of 255 options each would otherwise build gigabytes of
    hypotheses before anything downstream had a chance to refuse it. The check is
    arithmetic on lengths: no large string is built to measure it.
    """
    premise = render(request.state)
    prepared = tuple(_prepare_question(question) for question in request.questions)
    if max_request_chars is not None:
        _check_request_chars(premise, prepared, max_request_chars)
    return CompiledRequest(
        model=request.model,
        questions=tuple(_expand(question, premise) for question in prepared),
    )


@dataclass(frozen=True)
class _PreparedQuestion:
    """A question resolved down to candidate texts, before the template is applied."""

    question_id: str
    question_type: str
    option_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    instruction: str | None
    candidates: tuple[str, ...]
    legend: tuple[Any, ...] | None


def _check_request_chars(
    premise: str, questions: Sequence[_PreparedQuestion], max_request_chars: int
) -> None:
    premise_chars = len(premise)
    total = 0
    for question in questions:
        instruction_chars = 0 if question.instruction is None else len(question.instruction)
        # One sequence per candidate, each carrying the premise and the instruction.
        total += (premise_chars + instruction_chars) * len(question.candidates)
        total += sum(len(candidate) for candidate in question.candidates)
        if total > max_request_chars:
            raise ContextLengthExceededError(
                "The encoded request exceeds the per-request input limit.",
                path=["questions", question.question_id],
            )


def _prepare_question(question: Question) -> _PreparedQuestion:
    rendered = None if question.instructions is None else render(question.instructions)
    # An instruction that renders to nothing means "no extra guidance": keeping it
    # would put a bare separator in front of every candidate (A-F11). `[]` and `{}`
    # render to "[]" and "{}" and are therefore kept as written.
    instruction = None if rendered == "" else rendered

    if isinstance(question, NoulQuestion):
        candidates = (
            _noul_side(question.criteria_false, DEFAULT_NOUL_FALSE),
            _noul_side(question.criteria_true, DEFAULT_NOUL_TRUE),
        )
        option_keys = NOUL_OPTION_ORDER
        output_keys = NOUL_OPTION_ORDER
        legend = None
    elif isinstance(question, ChoiceQuestion):
        descriptions = dict(question.criteria)
        # Code point order for the model, request order for the wire (spec 6.1, 5.5).
        option_keys = tuple(sorted(descriptions))
        output_keys = tuple(key for key, _ in question.criteria)
        candidates = tuple(_choice_candidate(key, descriptions[key]) for key in option_keys)
        legend = None
    elif isinstance(question, ScoreQuestion):
        option_keys = tuple(str(index) for index in range(len(question.criteria)))
        output_keys = option_keys
        # The level number is external metadata and stays out of the model input.
        candidates = tuple(render(level) for level in question.criteria)
        legend = question.criteria
    else:  # pragma: no cover - the validator only produces the three types above
        raise InferenceError(f"Unsupported question type: {type(question).__name__}")

    return _PreparedQuestion(
        question_id=question.question_id,
        question_type=question.type,
        option_keys=option_keys,
        output_keys=output_keys,
        instruction=instruction,
        candidates=candidates,
        legend=legend,
    )


def _expand(question: _PreparedQuestion, premise: str) -> CompiledQuestion:
    pairs = tuple(
        TextPair(premise=premise, hypothesis=_hypothesis(question.instruction, candidate))
        for candidate in question.candidates
    )
    return CompiledQuestion(
        question_id=question.question_id,
        question_type=question.question_type,
        option_keys=question.option_keys,
        output_keys=question.output_keys,
        pairs=pairs,
        legend=question.legend,
    )


def _noul_side(criterion: Entry, default: str) -> str:
    return default if criterion is None else render(criterion)


def _choice_candidate(key: str, description: Entry) -> str:
    """A null description means the key alone names the option (spec 5.3)."""
    if description is None:
        return key
    return f"{key}: {render(description)}"


def _hypothesis(instruction: str | None, candidate: str) -> str:
    """Template ``nli-template-v1`` (POC_DESIGN 5.3)."""
    if instruction is None:
        return candidate
    return f"{instruction} — {candidate}"


def encode_request(
    compiled: CompiledRequest, backend: Backend, budget: TokenBudget
) -> EncodedRequest:
    """Tokenize every candidate and enforce the token budget.

    Nothing is truncated: overflow is refused with 422 ``context_length_exceeded``
    (spec 6.3, ``overflow_policy: reject``).
    """
    flat: list[TextPair] = [pair for question in compiled.questions for pair in question.pairs]
    encoded = backend.count_and_encode(flat)
    if len(encoded) != len(flat):
        raise InferenceError("The backend returned a different number of encoded sequences.")

    questions: list[EncodedQuestion] = []
    total = 0
    offset = 0
    for question in compiled.questions:
        sequences = tuple(encoded[offset : offset + len(question.pairs)])
        offset += len(question.pairs)
        for sequence in sequences:
            if sequence.token_count > budget.max_sequence_tokens:
                raise ContextLengthExceededError(
                    "The encoded question exceeds the model input limit.",
                    path=["questions", question.question_id],
                )
            total += sequence.token_count
        questions.append(EncodedQuestion(compiled=question, sequences=sequences))

    if total > budget.max_request_tokens:
        raise ContextLengthExceededError(
            "The encoded request exceeds the per-request token limit."
        )
    return EncodedRequest(questions=tuple(questions), total_tokens=total)


def compile_and_encode(
    request: ValidatedRequest,
    backend: Backend,
    budget: TokenBudget,
    *,
    max_request_chars: int,
) -> EncodedRequest:
    """Compile and tokenize one request.

    Called as a unit on the engine's encoder thread so that neither step runs on the
    event loop and both are covered by the request deadline (S-H2, A-F7).
    """
    compiled = compile_request(request, max_request_chars=max_request_chars)
    return encode_request(compiled, backend, budget)


def all_sequences(encoded: EncodedRequest) -> list[EncodedSequence]:
    """Flatten every candidate sequence in dispatch order."""
    return [sequence for question in encoded.questions for sequence in question.sequences]


def split_by_question(encoded: EncodedRequest, values: Sequence[float]) -> list[list[float]]:
    """Regroup a flat result list back onto the questions it came from."""
    grouped: list[list[float]] = []
    offset = 0
    for question in encoded.questions:
        count = len(question.sequences)
        grouped.append(list(values[offset : offset + count]))
        offset += count
    return grouped
