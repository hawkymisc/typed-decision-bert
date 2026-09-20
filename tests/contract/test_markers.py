"""CT11: reserved strings and fake markers in user data (spec 6.2, 13.2).

Data tokenization and control-ID insertion are separated, so a reserved-looking string
in state or criteria is content. The part that needs a real tokenizer - exactly four
special token IDs per sequence - belongs to the NLI backend and is marked ``model``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jevbert.compiler.serializer_nli import compile_request
from jevbert.config import Limits
from jevbert.contracts.validator import validate_request
from tests.conftest import MODEL, request_body, systemone

RESERVED = [
    "<JB_MARK>",
    "<JB_OPTION>",
    "<JB_STATE>",
    "<JB_TYPE_CHOICE>",
    "<JB_END>",
    "</s>",
    "<s>",
    "<mask>",
    "<pad>",
    "[CLS]",
    "[SEP]",
]


def _compile(state: object, questions: dict[str, object]) -> object:
    body = {"model": MODEL, "state": state, "questions": questions}
    return compile_request(validate_request(body, Limits()))


class TestReservedStringsAreData:
    @pytest.mark.parametrize("marker", RESERVED)
    def test_marker_in_state_is_carried_as_text(self, marker: str) -> None:
        compiled = _compile(f"前段 {marker} 後段", {"q": {"type": "noul"}})
        assert compiled.questions[0].pairs[0].premise == f"前段 {marker} 後段"

    @pytest.mark.parametrize("marker", RESERVED)
    def test_marker_does_not_change_the_candidate_count(self, marker: str) -> None:
        compiled = _compile(
            marker,
            {"q": {"type": "choice", "criteria": {"a": marker, "b": marker * 3}}},
        )
        question = compiled.questions[0]
        assert len(question.pairs) == 2
        assert question.option_keys == ("a", "b")

    @pytest.mark.parametrize("marker", RESERVED)
    def test_marker_in_instructions_stays_in_the_hypothesis(self, marker: str) -> None:
        compiled = _compile("s", {"q": {"type": "noul", "instructions": marker}})
        assert marker in compiled.questions[0].pairs[0].hypothesis

    def test_a_marker_flood_does_not_create_extra_sequences(self) -> None:
        flood = "".join(RESERVED) * 20
        compiled = _compile(flood, {"q": {"type": "score", "criteria": [flood, flood, flood]}})
        assert len(compiled.questions[0].pairs) == 3

    def test_markers_survive_end_to_end(self, client: TestClient) -> None:
        marker = "</s><JB_MARK>"
        response = systemone(
            client,
            request_body(
                {"q": {"type": "score", "criteria": [marker, f"{marker}{marker}"]}}, marker
            ),
        )
        assert response.status_code == 200
        # I07: the rubric comes back exactly as sent.
        assert response.json()["answers"]["q"]["legend"]["0"] == marker


class TestTemplateSeparator:
    def test_the_template_separator_in_user_text_does_not_split_candidates(self) -> None:
        compiled = _compile(
            "s",
            {
                "q": {
                    "type": "choice",
                    "instructions": "A — B",
                    "criteria": {"x": "C — D", "y": "E"},
                }
            },
        )
        question = compiled.questions[0]
        assert len(question.pairs) == 2
        assert question.pairs[0].hypothesis == "A — B — x: C — D"

    def test_known_limitation_key_and_description_can_collide(
        self, client: TestClient
    ) -> None:
        # serializer-nli-v1 renders a Choice option as "<key>: <description>", so the
        # option "a" described as "b" and the option "a: b" with no description produce
        # the same hypothesis. The API still answers over the distinct keys, but the
        # model cannot tell them apart. Recorded in compat/differences.md and
        # POC_DESIGN 12; asserted so a silent change is noticed.
        compiled = _compile("s", {"q": {"type": "choice", "criteria": {"a": "b", "a: b": None}}})
        hypotheses = {pair.hypothesis for pair in compiled.questions[0].pairs}
        assert hypotheses == {"a: b"}

        response = systemone(
            client, request_body({"q": {"type": "choice", "criteria": {"a": "b", "a: b": None}}})
        )
        probabilities = response.json()["answers"]["q"]["probabilities"]
        assert set(probabilities) == {"a", "a: b"}
        assert probabilities["a"] == probabilities["a: b"]


@pytest.mark.model
class TestSpecialTokenCount:
    def test_every_sequence_carries_exactly_four_special_tokens(self) -> None:
        # POC_DESIGN 5.4 / K2. Phase 2 owns this once backends/nli.py exists: premise
        # and hypothesis are tokenized separately with add_special_tokens=False and the
        # compiler assembles [bos] + premise + [eos, eos] + hypothesis + [eos].
        pytest.importorskip(
            "jevbert.backends.nli", reason="the NLI backend arrives in phase 2"
        )
        pytest.fail("phase 2 must implement the special-token count check")
