"""Compiled request representation (spec 12.2; POC_DESIGN 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jevbert.backends.base import EncodedSequence, TextPair


@dataclass(frozen=True)
class TokenBudget:
    """Effective token limits of a bundle (spec 5.8, POC_DESIGN 5.5)."""

    max_sequence_tokens: int
    max_request_tokens: int


@dataclass(frozen=True)
class CompiledQuestion:
    """One question expanded into one candidate sequence each.

    Attributes:
        option_keys: Candidate order as given to the model. Code point order for
            Choice, array order for Score, ``false, true`` for Noul.
        output_keys: Candidate order for the wire: request order for Choice,
            ``"0"..."K-1"`` for Score. Independent of ``option_keys`` (POC_DESIGN 5.2).
        legend: Original Score level descriptions, kept to restore types (I07).
    """

    question_id: str
    question_type: str
    option_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    pairs: tuple[TextPair, ...]
    legend: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class CompiledRequest:
    model: str
    questions: tuple[CompiledQuestion, ...]


@dataclass(frozen=True)
class EncodedQuestion:
    compiled: CompiledQuestion
    sequences: tuple[EncodedSequence, ...]


@dataclass(frozen=True)
class EncodedRequest:
    questions: tuple[EncodedQuestion, ...]
    #: ``usage.input_tokens`` under ``expanded-input-a0-v1`` (spec 5.8).
    total_tokens: int
