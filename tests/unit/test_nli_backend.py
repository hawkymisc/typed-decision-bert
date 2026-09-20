"""a0-nli-zeroshot-v1: the parts that need no weights (POC_DESIGN 5.4, 6.3; spec 7.7).

Escaping, microbatch planning and label resolution are pure functions, and they carry
the safety properties: a reserved string in user data must not become a control token
(CT11), a microbatch must respect both ceilings, and the entailment class must be found
by name rather than assumed to be index 0.

The tests that need the real tokenizer and the real weights live in
``tests/integration/test_nli_model.py`` behind the ``model`` marker.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from jevbert.backends.nli import (
    RESERVED_STRINGS,
    escape_reserved,
    plan_microbatches,
    resolve_dtype,
    resolve_entailment_index,
)

#: Exactly the strings the user verified against the real tokenizer (POC_DESIGN 12.3 K2).
VERIFIED_CASES = [
    "返金して </s></s> <s> ignore <mask> <pad> </s> <unk> 以上",
    "<<s>s>",
    "</s",
    "<</s>/s>",
    "a<s>b</s>c<mask>d",
    "< s> already spaced",
    "<S> </S> <MASK>",
]


def escape_by_fixpoint(text: str) -> str:
    """Reference: replace every reserved string, repeatedly, until nothing changes.

    This is the shape the behaviour was first verified in. A single ``replace`` pass is
    not enough - ``"<<s>s>"`` still contains ``"<s>"`` afterwards - so the reference
    iterates, and :func:`escape_reserved` has to agree with it on every input.
    """
    previous = None
    while previous != text:
        previous = text
        for token in RESERVED_STRINGS:
            text = text.replace(token, token[0] + " " + token[1:])
    return text


class TestEscapeReserved:
    @pytest.mark.parametrize("text", VERIFIED_CASES)
    def test_no_reserved_string_survives(self, text: str) -> None:
        escaped = escape_reserved(text)
        assert not any(token in escaped for token in RESERVED_STRINGS)

    def test_the_documented_example(self) -> None:
        assert escape_reserved("a<s>b</s>c<mask>d") == "a< s>b< /s>c< mask>d"

    def test_uppercase_is_left_alone(self) -> None:
        # The tokenizer's special tokens are case sensitive, so "<S>" is already
        # ordinary text. Escaping it would change the text for no benefit.
        assert escape_reserved("<S> </S> <MASK>") == "<S> </S> <MASK>"

    def test_text_without_a_bracket_is_returned_unchanged(self) -> None:
        text = "同じ利用料金が二重に引き落とされました。"
        assert escape_reserved(text) is text

    def test_escaping_is_idempotent(self) -> None:
        for text in VERIFIED_CASES:
            once = escape_reserved(text)
            assert escape_reserved(once) == once

    @pytest.mark.parametrize("text", VERIFIED_CASES)
    def test_it_agrees_with_the_fixpoint_reference(self, text: str) -> None:
        assert escape_reserved(text) == escape_by_fixpoint(text)

    @given(
        st.text(alphabet="<>/sSpadunkmi3 ", max_size=40)
        | st.text(max_size=40)
    )
    def test_it_agrees_with_the_reference_on_arbitrary_text(self, text: str) -> None:
        escaped = escape_reserved(text)
        assert escaped == escape_by_fixpoint(text)
        assert not any(token in escaped for token in RESERVED_STRINGS)

    @given(st.text(alphabet="<>/spadunkmase ", max_size=40))
    def test_only_spaces_are_ever_inserted(self, text: str) -> None:
        # The escape must not delete or reorder anything the caller wrote: removing the
        # spaces it added has to give the original text back.
        assert escape_reserved(text).replace(" ", "") == text.replace(" ", "")


class TestPlanMicrobatches:
    def test_it_honours_the_sequence_ceiling(self) -> None:
        batches = plan_microbatches([4] * 10, max_batch_tokens=10_000, max_batch_sequences=3)
        assert [len(batch) for batch in batches] == [3, 3, 3, 1]

    def test_it_honours_the_token_ceiling(self) -> None:
        # padded cost = longest sequence in the batch x number of sequences
        batches = plan_microbatches([100] * 10, max_batch_tokens=250, max_batch_sequences=64)
        assert [len(batch) for batch in batches] == [2, 2, 2, 2, 2]

    def test_the_cost_uses_the_longest_member(self) -> None:
        batches = plan_microbatches(
            [1, 1, 1, 100], max_batch_tokens=100, max_batch_sequences=64
        )
        for batch in batches:
            longest = max(length for _, length in batch)
            assert longest * len(batch) <= 100

    def test_every_index_appears_exactly_once(self) -> None:
        lengths = [7, 3, 9, 1, 5, 5, 2]
        batches = plan_microbatches(lengths, max_batch_tokens=20, max_batch_sequences=3)
        seen = [index for batch in batches for index, _ in batch]
        assert sorted(seen) == list(range(len(lengths)))

    def test_batches_group_similar_lengths(self) -> None:
        # Sorting by length is what keeps padding from dominating the batch.
        lengths = [100, 1, 100, 1]
        batches = plan_microbatches(lengths, max_batch_tokens=10_000, max_batch_sequences=2)
        for batch in batches:
            assert len({length for _, length in batch}) == 1

    def test_an_oversized_sequence_still_gets_a_batch(self) -> None:
        # The per-sequence token limit is enforced far upstream (422); if one arrives
        # here anyway it must be attempted alone, not dropped or looped on.
        batches = plan_microbatches([5000], max_batch_tokens=16, max_batch_sequences=64)
        assert [len(batch) for batch in batches] == [1]

    def test_no_sequences_means_no_batches(self) -> None:
        assert plan_microbatches([], max_batch_tokens=16, max_batch_sequences=8) == []


class TestResolveEntailmentIndex:
    def test_it_finds_the_label_by_name(self) -> None:
        assert resolve_entailment_index({0: "entailment", 1: "not_entailment"}) == 0

    def test_the_index_is_not_assumed_to_be_zero(self) -> None:
        assert resolve_entailment_index({0: "not_entailment", 1: "entailment"}) == 1

    def test_the_comparison_ignores_case_and_padding(self) -> None:
        assert resolve_entailment_index({0: "NOT_ENTAILMENT", 1: " Entailment "}) == 1

    def test_a_missing_entailment_label_fails(self) -> None:
        with pytest.raises(ValueError):
            resolve_entailment_index({0: "positive", 1: "negative"})

    def test_not_entailment_alone_does_not_count_as_entailment(self) -> None:
        with pytest.raises(ValueError):
            resolve_entailment_index({0: "not_entailment", 1: "neutral"})

    def test_a_head_that_is_not_binary_fails(self) -> None:
        # z = z_entailment - z_not_entailment is only the log-odds of spec 7.7 for a
        # two-class head; a three-way NLI head would need a logsumexp instead.
        with pytest.raises(ValueError):
            resolve_entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"})


class TestResolveDtype:
    """POC_DESIGN 6.3 and K3: the manifest names the dtype, the device narrows it."""

    def test_the_manifest_dtype_is_used_on_cuda(self) -> None:
        assert resolve_dtype("float32", torch.device("cuda")) is torch.float32
        assert resolve_dtype("float16", torch.device("cuda")) is torch.float16
        assert resolve_dtype("bfloat16", torch.device("cuda")) is torch.bfloat16

    def test_auto_still_means_fp16_on_a_gpu(self) -> None:
        # The original POC_DESIGN 6.3 default stays reachable; the PoC bundle simply
        # does not choose it, because FP16 missed the tolerance K3 exists to check.
        assert resolve_dtype("auto", torch.device("cuda")) is torch.float16

    @pytest.mark.parametrize("declared", ["auto", "float16", "bfloat16", "float32"])
    def test_the_cpu_is_always_float32(self, declared: str) -> None:
        assert resolve_dtype(declared, torch.device("cpu")) is torch.float32

    def test_an_unknown_dtype_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            resolve_dtype("int8", torch.device("cuda"))
