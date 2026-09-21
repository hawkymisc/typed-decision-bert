"""serializer-nli-v1: question -> (premise, hypothesis) pairs (POC_DESIGN 5).

The compiler stops at text. Token counting and ID assembly belong to the backend
(``count_and_encode``), which is what lets the same compiler serve the fake backend and
a real tokenizer without knowing either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

#: Generic yes/no candidate text used when a Noul criterion is absent or null (spec 5.4).
DEFAULT_NOUL_TRUE = "The answer to the question is yes."
DEFAULT_NOUL_FALSE = "The answer to the question is no."

#: Hiragana, katakana and CJK ideographs. Used by ``nli-template-v3`` only.
_JAPANESE_RANGES = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))


def _looks_japanese(text: str) -> bool:
    return any(
        any(low <= ord(character) <= high for low, high in _JAPANESE_RANGES)
        for character in text
    )


@dataclass(frozen=True)
class HypothesisTemplate:
    """How an instruction and a candidate become one NLI hypothesis (POC_DESIGN 5.3).

    The three candidates below are the ones K4 compared; the bundle's
    ``serializer_version`` names the one in force and the registry refuses a manifest
    that names another, so the template a bundle ID promises is the template it gets.

    User text is concatenated, never passed through ``str.format``: a criterion that
    renders to ``"{}"`` - which an empty JSON object does - would otherwise be read as
    a field reference.
    """

    template_id: str
    render: Callable[[str | None, str], str]


def _render_separator(instruction: str | None, candidate: str) -> str:
    """nli-template-v1: ``"{I} — {C}"``, or the bare candidate without an instruction."""
    if instruction is None:
        return candidate
    return f"{instruction} — {candidate}"


def _render_question_answer(instruction: str | None, candidate: str) -> str:
    """nli-template-v2: ``"Question: {I} Answer: {C}"``."""
    if instruction is None:
        return f"Answer: {candidate}"
    return f"Question: {instruction} Answer: {candidate}"


def _render_natural_sentence(instruction: str | None, candidate: str) -> str:
    """nli-template-v3: a full sentence, in the script the question is written in.

    The frame has to agree with the surrounding text or the hypothesis reads as two
    languages spliced together, so the script of the instruction and the candidate
    decides it. The rule is a character-range test, not a language model: it is
    deterministic, it cannot fail, and it makes the compiled input depend on the
    *script* of the input, which is itself a reason to prefer a frame that needs no
    such rule (recorded with the K4 comparison in POC_RESULTS).
    """
    japanese = _looks_japanese(candidate) or (
        instruction is not None and _looks_japanese(instruction)
    )
    if instruction is None:
        return f"答えは「{candidate}」である。" if japanese else f'The answer is "{candidate}".'
    if japanese:
        return f"「{instruction}」の答えは「{candidate}」である。"
    return f'The answer to "{instruction}" is "{candidate}".'


TEMPLATE_V1 = HypothesisTemplate("nli-template-v1", _render_separator)
TEMPLATE_V2 = HypothesisTemplate("nli-template-v2", _render_question_answer)
TEMPLATE_V3 = HypothesisTemplate("nli-template-v3", _render_natural_sentence)

#: The K4 candidates, by ID. ``scripts/compare_templates.py`` re-runs the comparison.
TEMPLATES: dict[str, HypothesisTemplate] = {
    template.template_id: template for template in (TEMPLATE_V1, TEMPLATE_V2, TEMPLATE_V3)
}

#: The template this build compiles with. Changing it changes every compiled input, so
#: it also changes ``serializer_version`` and therefore the bundle ID (spec 5.7).
DEFAULT_TEMPLATE = TEMPLATE_V1
NLI_TEMPLATE_ID = DEFAULT_TEMPLATE.template_id

#: What a manifest's ``serializer_version`` looks like for this build.
SERIALIZER_VERSION_FULL = f"{SERIALIZER_VERSION}+{NLI_TEMPLATE_ID}"


def template_id_of(serializer_version: str) -> str | None:
    """The template ID a ``serializer_version`` names, or ``None`` if it names none."""
    family, separator, template = serializer_version.partition("+")
    if family != SERIALIZER_VERSION or not separator:
        return None
    return template


def compile_request(
    request: ValidatedRequest,
    *,
    max_request_chars: int | None = None,
    template: HypothesisTemplate = DEFAULT_TEMPLATE,
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
        questions=tuple(_expand(question, premise, template) for question in prepared),
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


def _expand(
    question: _PreparedQuestion, premise: str, template: HypothesisTemplate
) -> CompiledQuestion:
    pairs = tuple(
        TextPair(premise=premise, hypothesis=template.render(question.instruction, candidate))
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


#: The cheapest a sequence can possibly be: the four control tokens the backend inserts
#: plus one token of data. Dividing the request's token budget by it gives the largest
#: number of sequences that could ever fit, which is decidable by counting (S-L1).
MIN_TOKENS_PER_SEQUENCE = 5


def max_sequences_for(budget: TokenBudget) -> int:
    """How many sequences could fit in the token budget in the best case."""
    return budget.max_request_tokens // MIN_TOKENS_PER_SEQUENCE


def encode_request(
    compiled: CompiledRequest, backend: Backend, budget: TokenBudget
) -> EncodedRequest:
    """Tokenize every candidate and enforce the token budget.

    Nothing is truncated: overflow is refused with 422 ``context_length_exceeded``
    (spec 6.3, ``overflow_policy: reject``).

    The *count* of sequences is checked first, because it is known without tokenizing
    anything. 256 questions x 255 options is 65,280 sequences, and short texts carry
    that straight through the character ceiling, so the whole set used to be tokenized
    and only then refused on a total it could never have met (S-L1).
    """
    flat: list[TextPair] = [pair for question in compiled.questions for pair in question.pairs]
    if len(flat) > max_sequences_for(budget):
        raise ContextLengthExceededError(
            "The encoded request exceeds the per-request token limit."
        )
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
    template: HypothesisTemplate = DEFAULT_TEMPLATE,
) -> EncodedRequest:
    """Compile and tokenize one request.

    Called as a unit on the engine's encoder thread so that neither step runs on the
    event loop and both are covered by the request deadline (S-H2, A-F7).
    """
    compiled = compile_request(
        request, max_request_chars=max_request_chars, template=template
    )
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
