"""Hypothesis strategies shared by the property tests."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

#: Keys and text stay small so that generated bodies remain cheap to compile.
KEYS = st.text(min_size=0, max_size=6)
SCALARS = st.none() | st.booleans() | st.integers(-1000, 1000) | st.floats(-1e6, 1e6) | KEYS

JSON_VALUES = st.recursive(
    SCALARS,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(KEYS, children, max_size=3),
    max_leaves=6,
)

#: ``Content`` from spec 5.2: string, array or object - never a bare null/bool/number.
CONTENT = (
    st.text(max_size=12)
    | st.lists(JSON_VALUES, max_size=3)
    | st.dictionaries(KEYS, JSON_VALUES, max_size=3)
)
ENTRY = st.none() | CONTENT


def noul_questions() -> st.SearchStrategy[dict[str, Any]]:
    criteria = st.none() | st.dictionaries(st.sampled_from(["true", "false"]), ENTRY, max_size=2)
    return st.fixed_dictionaries(
        {"type": st.just("noul")},
        optional={"instructions": ENTRY, "criteria": criteria},
    )


def choice_questions(
    min_options: int = 2, max_options: int = 4
) -> st.SearchStrategy[dict[str, Any]]:
    criteria = st.dictionaries(KEYS, ENTRY, min_size=min_options, max_size=max_options)
    return st.fixed_dictionaries(
        {"type": st.just("choice"), "criteria": criteria},
        optional={"instructions": ENTRY},
    )


def score_questions(
    min_levels: int = 2, max_levels: int = 4
) -> st.SearchStrategy[dict[str, Any]]:
    criteria = st.lists(CONTENT, min_size=min_levels, max_size=max_levels)
    return st.fixed_dictionaries(
        {"type": st.just("score"), "criteria": criteria},
        optional={"instructions": ENTRY},
    )


def valid_questions() -> st.SearchStrategy[dict[str, Any]]:
    return noul_questions() | choice_questions() | score_questions()


def valid_requests(model: str, max_questions: int = 3) -> st.SearchStrategy[dict[str, Any]]:
    return st.fixed_dictionaries(
        {
            "model": st.just(model),
            "state": CONTENT,
            "questions": st.dictionaries(
                KEYS, valid_questions(), min_size=1, max_size=max_questions
            ),
        }
    )


def loose_questions() -> st.SearchStrategy[Any]:
    """Question-shaped values, valid or not, for the schema agreement test."""
    return (
        noul_questions()
        # Option and level counts reach outside the permitted range.
        | choice_questions(min_options=0, max_options=3)
        | score_questions(min_levels=0, max_levels=3)
        # A Score level that is null (U02) and other malformed shapes.
        | st.fixed_dictionaries(
            {"type": st.just("score"), "criteria": st.lists(ENTRY, min_size=2, max_size=3)}
        )
        | st.fixed_dictionaries({"type": st.sampled_from(["noul", "choice", "score", "bool", ""])})
        | st.fixed_dictionaries(
            {"type": st.just("noul"), "weight": st.integers()}  # unknown field
        )
        | st.fixed_dictionaries(
            {
                "type": st.just("noul"),
                "criteria": st.dictionaries(
                    st.sampled_from(["true", "false", "maybe"]), ENTRY, min_size=1, max_size=3
                ),
            }
        )
        | JSON_VALUES
    )


def loose_requests() -> st.SearchStrategy[Any]:
    """Request-shaped values, valid or not."""
    base = st.fixed_dictionaries(
        {
            "model": st.text(max_size=6),
            "state": CONTENT | st.none() | st.integers() | st.booleans(),
            "questions": st.dictionaries(KEYS, loose_questions(), max_size=3),
        },
        optional={"extra": st.integers()},
    )
    return base | JSON_VALUES
