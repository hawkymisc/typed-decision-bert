"""a0-nli-zeroshot-v1: a public zero-shot NLI classifier in the A0 shape (spec 7.7).

One sequence per candidate. The scalar the backend returns is the entailment log-odds

    z = z_entailment - z_not_entailment

which is the two-class form of spec 7.7. Turning that into probabilities, confidence or
a score is ``jevbert.scoring`` alone (spec 12.1): this module never sees a temperature.

Two things here are load-bearing for safety rather than for quality:

* **Control IDs are assembled, never parsed out of text** (spec 6.2, CT11). Premise and
  hypothesis are tokenized separately with ``add_special_tokens=False`` and the
  sequence is built as ``[bos] + premise + [eos, eos] + hypothesis + [eos]``. Whatever
  the user wrote stays data.
* **Nothing is truncated** (spec 6.3). The token budget is enforced upstream and
  overflow is a 422, so a sequence that arrives here is one the server already accepted.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from jevbert.backends.base import CancelToken, EncodedSequence, TextPair
from jevbert.fetch import mismatched_files

logger = logging.getLogger("jevbert.backend.nli")

#: Special token strings that must never reach the tokenizer as data.
#:
#: ``split_special_tokens=True`` looks like the answer and is not: measured against this
#: tokenizer it still produced six control IDs from user text (POC_DESIGN 12.3, K2).
#: What does work is inserting a space after the leading ``<``, which the tokenizer then
#: reads as ordinary characters. Lower case only - the tokenizer matches its special
#: tokens case sensitively, so ``<MASK>`` is already just text.
RESERVED_STRINGS: tuple[str, ...] = ("</s>", "<s>", "<pad>", "<unk>", "<mask>")

#: Control tokens the compiler inserts, and the exact count every sequence must carry.
_STRUCTURAL_TOKEN_COUNT = 4

_ENTAILMENT_LABEL = "entailment"


def escape_reserved(text: str) -> str:
    """Disarm reserved special-token strings by putting a space after the ``<``.

    ``"a<s>b</s>c"`` becomes ``"a< s>b< /s>c"``. Only spaces are inserted: nothing is
    removed or reordered, so the text the model sees still contains everything the
    caller wrote. That it is *not identical* to what the caller wrote is a recorded
    difference (``compat/differences.md`` L08).

    One left-to-right pass is enough, and terminates for the same reason: a ``<`` that
    has been given a space can never start a reserved string again, and no ``<`` is
    ever added. Scanning continues from the character after the disarmed ``<``, so a
    nested ``"<<s>s>"`` is handled without a second pass.
    """
    index = text.find("<")
    if index < 0:
        return text

    parts: list[str] = []
    start = 0
    while index >= 0:
        if any(text.startswith(token, index) for token in RESERVED_STRINGS):
            parts.append(text[start : index + 1])
            parts.append(" ")
            start = index + 1
        index = text.find("<", index + 1)
    if not parts:
        return text
    parts.append(text[start:])
    return "".join(parts)


def resolve_entailment_index(id2label: Mapping[int, str]) -> int:
    """Find the entailment class by name (POC_DESIGN 6.3).

    Index 0 is not assumed: a checkpoint that happens to order its labels the other way
    would otherwise return the negated score for every candidate, silently, and every
    invariant in spec 5.5 would still hold.
    """
    if len(id2label) != 2:
        raise ValueError(
            f"a0-nli-zeroshot-v1 needs a two-class NLI head, found {len(id2label)} labels. "
            "z = z_entailment - z_not_entailment is the log-odds of spec 7.7 only for two "
            "classes."
        )
    matches = [
        index
        for index, label in id2label.items()
        if str(label).strip().casefold() == _ENTAILMENT_LABEL
    ]
    if len(matches) != 1:
        raise ValueError(
            "the model config has no single label named 'entailment'; "
            f"labels were {sorted(str(label) for label in id2label.values())}"
        )
    return int(matches[0])


def plan_microbatches(
    lengths: Sequence[int], *, max_batch_tokens: int, max_batch_sequences: int
) -> list[list[tuple[int, int]]]:
    """Group sequence indices into microbatches (POC_DESIGN 6.3).

    Sequences are sorted by length so that a batch pads to something close to its own
    members, and a batch is closed when either ceiling would be crossed: the padded
    cost ``longest x count`` against ``max_batch_tokens``, or the count against
    ``max_batch_sequences``. A single sequence longer than the token ceiling still gets
    a batch of its own - refusing it here would be a 500 for something the request path
    already accepted, and looping on it would be worse.

    Returns:
        Batches of ``(original index, length)``, every index exactly once.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    batches: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    longest = 0
    for index in order:
        length = lengths[index]
        candidate_longest = max(longest, length)
        if current and (
            len(current) + 1 > max_batch_sequences
            or candidate_longest * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current = []
            candidate_longest = length
        current.append((index, length))
        longest = candidate_longest
    if current:
        batches.append(current)
    return batches


def resolve_device(requested: str) -> torch.device:
    """``auto`` means CUDA when there is one (POC_DESIGN 6.3)."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


#: What a manifest may declare in ``dtype``. ``auto`` is the original POC_DESIGN 6.3
#: rule, FP16 on the GPU; the PoC bundle declares ``float32`` instead because FP16
#: missed the numeric tolerance K3 was written to test (POC_DESIGN 12.3).
_DTYPES: Mapping[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(declared: str, device: torch.device) -> torch.dtype:
    """The compute dtype for one device, from what the manifest declares.

    The CPU is always FP32: a half-precision matmul there is either unsupported or
    emulated, so it would be neither the fast path nor a usable reference.
    """
    if device.type == "cpu":
        return torch.float32
    if declared == "auto":
        return torch.float16
    dtype = _DTYPES.get(declared)
    if dtype is None:
        raise ValueError(
            f"the manifest declares dtype={declared!r}; expected one of "
            f"{', '.join(sorted(_DTYPES))} or 'auto'"
        )
    return dtype


class NliZeroShotBackend:
    """Zero-shot NLI backend over a locally pinned checkpoint.

    The weights are whatever ``python -m jevbert fetch-model`` put in ``model_dir``;
    nothing here resolves a repo ID, reaches the network or runs repository code
    (spec 15.3).
    """

    backend_id = "a0-nli-zeroshot-v1"

    def __init__(
        self,
        model_dir: Path,
        *,
        max_batch_tokens: int = 16_384,
        max_batch_sequences: int = 64,
        device: str = "auto",
        dtype: str = "float32",
        expected_file_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_sequences = max_batch_sequences
        self._requested_device = device
        self._declared_dtype = dtype
        self._expected_file_hashes = dict(expected_file_hashes or {})

        self._tokenizer: object | None = None
        self._model: object | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._entailment_index = 0
        self._other_index = 1
        self._pad_token_id = 1
        self._bos_token_id = 0
        self._eos_token_id = 2
        self._structural_ids: frozenset[int] = frozenset()
        self._premise_tokenizations = 0

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def tokenizer(self) -> Any:
        """The loaded tokenizer. Tests compare against its own pair encoding."""
        return self._require_tokenizer()

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        return self._dtype

    @property
    def premise_tokenizations(self) -> int:
        """How many premise strings have been handed to the tokenizer.

        An observability counter for the SHOULD in ``backends/base.py``: one request can
        carry 8,160 copies of the same state, and this is how a test sees that the state
        was tokenized once rather than once per candidate. It counts strings, never
        stores them.
        """
        return self._premise_tokenizations

    def load(self) -> None:
        """Verify the pinned files, then load the tokenizer and the weights.

        The hash check is the readiness step of spec 16.1: the manifest records a
        SHA-256 per file and the bundle digest is the hash of the manifest, so loading
        bytes that do not match would serve a different model under a published ID.
        """
        self._verify_files()

        config = AutoConfig.from_pretrained(
            self._model_dir, local_files_only=True, trust_remote_code=False
        )
        self._entailment_index = resolve_entailment_index(config.id2label)
        self._other_index = 1 - self._entailment_index

        device = resolve_device(self._requested_device)
        # The manifest names the dtype, so the bundle digest moves when it changes -
        # it is part of what the bundle ID identifies (spec 5.7). What it should be was
        # measured, not assumed (K3, POC_DESIGN 12.3).
        dtype = resolve_dtype(self._declared_dtype, device)

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
        )
        model.eval()
        model.to(device)

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype
        self._bos_token_id = _required_token_id(tokenizer, "bos_token_id", "cls_token_id")
        self._eos_token_id = _required_token_id(tokenizer, "eos_token_id", "sep_token_id")
        self._pad_token_id = _required_token_id(tokenizer, "pad_token_id")
        # The unknown token is data, not structure: it stands for a character the
        # vocabulary cannot encode, and counting it would turn an exotic but perfectly
        # valid input into a 500.
        self._structural_ids = frozenset(
            token_id
            for token_id in tokenizer.all_special_ids
            if token_id != tokenizer.unk_token_id
        )
        logger.info(
            "backend %s loaded: device=%s dtype=%s entailment_index=%d labels=%d",
            self.backend_id,
            device,
            str(dtype).removeprefix("torch."),
            self._entailment_index,
            len(config.id2label),
        )

    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
        """Escape, tokenize both sides separately, and assemble the pair sequence.

        Every distinct premise is tokenized once per call, as ``backends/base.py``
        requires: an A0 request repeats the same state for every candidate, so doing
        otherwise would pay for the state up to 8,160 times. The map dies with the call.
        """
        tokenizer = self._require_tokenizer()
        if not pairs:
            return []

        distinct_premises: dict[str, int] = {}
        for pair in pairs:
            distinct_premises.setdefault(pair.premise, len(distinct_premises))
        premise_texts = [escape_reserved(text) for text in distinct_premises]
        premise_ids = tokenizer(premise_texts, add_special_tokens=False)["input_ids"]
        self._premise_tokenizations += len(premise_texts)

        hypothesis_ids = tokenizer(
            [escape_reserved(pair.hypothesis) for pair in pairs], add_special_tokens=False
        )["input_ids"]

        bos, eos = self._bos_token_id, self._eos_token_id
        sequences: list[EncodedSequence] = []
        for pair, hypothesis in zip(pairs, hypothesis_ids, strict=True):
            premise = premise_ids[distinct_premises[pair.premise]]
            ids = [bos, *premise, eos, eos, *hypothesis, eos]
            self._check_structural_tokens(ids)
            sequences.append(EncodedSequence(token_count=len(ids), data=ids))
        return sequences

    def _check_structural_tokens(self, ids: Sequence[int]) -> None:
        """CT11: the four control IDs are the ones this method just inserted.

        A reserved string in the state or in a criterion has to arrive as data. If the
        escape ever stops working, the request must fail loudly rather than be scored
        against a sequence whose structure the caller wrote.
        """
        found = sum(1 for token_id in ids if token_id in self._structural_ids)
        if found != _STRUCTURAL_TOKEN_COUNT:
            raise ValueError(
                f"the encoded sequence carries {found} control tokens, expected "
                f"{_STRUCTURAL_TOKEN_COUNT}"
            )

    def score(self, sequences: Sequence[EncodedSequence], cancel: CancelToken) -> list[float]:
        """One finite entailment log-odds per sequence, in the order given."""
        model = self._require_model()
        cancel.raise_if_cancelled()
        if not sequences:
            return []

        lengths = [len(sequence.data) for sequence in sequences]
        results: list[float] = [0.0] * len(sequences)
        batches = plan_microbatches(
            lengths,
            max_batch_tokens=self._max_batch_tokens,
            max_batch_sequences=self._max_batch_sequences,
        )
        for batch in batches:
            # Between microbatches, not inside one: a forward pass cannot be stopped
            # half way, but the next one does not have to start (spec 12.3).
            cancel.raise_if_cancelled()
            for index, value in zip(
                [index for index, _ in batch], self._score_batch(model, sequences, batch),
                strict=True,
            ):
                results[index] = value
        return results

    def _score_batch(
        self,
        model: object,
        sequences: Sequence[EncodedSequence],
        batch: Sequence[tuple[int, int]],
    ) -> list[float]:
        device = self._device
        assert device is not None  # load() sets both together
        width = max(length for _, length in batch)
        input_ids = torch.full((len(batch), width), self._pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (index, length) in enumerate(batch):
            input_ids[row, :length] = torch.tensor(sequences[index].data, dtype=torch.long)
            attention_mask[row, :length] = 1

        try:
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                ).logits
            # FP32 before any arithmetic (spec 8.1): the difference of two FP16 logits
            # loses resolution exactly where the two classes are close.
            logits = logits.float().cpu()
        except torch.cuda.OutOfMemoryError:
            # Context only - no input, no exception message (spec 15.2). Freeing the
            # cache is what keeps the *next* request serviceable (POC_DESIGN 6.3).
            logger.error(
                "CUDA out of memory in %s: batch_sequences=%d padded_width=%d "
                "max_batch_tokens=%d",
                self.backend_id,
                len(batch),
                width,
                self._max_batch_tokens,
            )
            torch.cuda.empty_cache()
            raise

        differences = logits[:, self._entailment_index] - logits[:, self._other_index]
        values = [float(value) for value in differences.tolist()]
        for value in values:
            if not math.isfinite(value):
                # Never smoothed into a plausible distribution (spec 5.5, N02).
                raise ValueError("the model produced a non-finite logit")
        return values

    def _verify_files(self) -> None:
        if not self._expected_file_hashes:
            raise ValueError(
                "the manifest records no source_model.files, so the weights cannot be "
                "verified. Run `python -m jevbert fetch-model`."
            )
        bad = mismatched_files(self._model_dir, self._expected_file_hashes)
        if bad:
            raise ValueError(
                f"{len(bad)} model files are missing or do not match the manifest: "
                f"{', '.join(sorted(bad))}"
            )

    def _require_tokenizer(self) -> object:
        if self._tokenizer is None:
            raise RuntimeError("the NLI backend has not been loaded")
        return self._tokenizer

    def _require_model(self) -> object:
        if self._model is None:
            raise RuntimeError("the NLI backend has not been loaded")
        return self._model


def _required_token_id(tokenizer: object, *names: str) -> int:
    for name in names:
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"the tokenizer defines none of {', '.join(names)}")
