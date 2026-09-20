"""serializer-nli-v1: question -> (premise, hypothesis) pairs (POC_DESIGN 5).

The compiler stops at text. Token counting and ID assembly belong to the backend
(``count_and_encode``), which is what lets the same compiler serve the fake backend and
a real tokenizer without knowing either.
"""

from __future__ import annotations

from collections.abc import Sequence

from jevbert.api.errors import ContextLengthExceededError, InferenceError
from jevbert.backends.base import Backend, TextPair
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


def compile_request(request: ValidatedRequest) -> CompiledRequest:
    premise = render(request.state)
    return CompiledRequest(
        model=request.model,
        questions=tuple(_compile_question(question, premise) for question in request.questions),
    )


def _compile_question(question: Question, premise: str) -> CompiledQuestion:
    instruction = None if question.instructions is None else render(question.instructions)

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

    pairs = tuple(
        TextPair(premise=premise, hypothesis=_hypothesis(instruction, candidate))
        for candidate in candidates
    )
    return CompiledQuestion(
        question_id=question.question_id,
        question_type=question.type,
        option_keys=option_keys,
        output_keys=output_keys,
        pairs=pairs,
        legend=legend,
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


def all_sequences(encoded: EncodedRequest) -> list[object]:
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
