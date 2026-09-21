"""a0-nli-zeroshot-v2: a public zero-shot NLI classifier in the A0 shape (spec 7.7).

One sequence per candidate. The scalar the backend returns is the entailment log-odds

    z = z_entailment - z_not_entailment

which is the two-class form of spec 7.7. Turning that into probabilities, confidence or
a score is ``jevbert.scoring`` alone (spec 12.1): this module never sees a temperature.

Two things here are load-bearing for safety rather than for quality:

* **Control IDs are assembled, never parsed out of text** (spec 6.2, CT11). Premise and
  hypothesis are tokenized separately with ``add_special_tokens=False`` and the
  sequence is built as ``[bos] + premise + [eos, eos] + hypothesis + [eos]``. Whatever
  the user wrote stays data. The escape that makes that true is decided on the
  *normalized* text, because the tokenizer folds compatibility characters before it
  matches its special tokens (S-H1, POC_DESIGN 12.4).
* **Nothing is truncated** (spec 6.3). The token budget is enforced upstream and
  overflow is a 422, so a sequence that arrives here is one the server already accepted.

The file hashes are verified at :meth:`NliZeroShotBackend.load` and the weights are read
immediately afterwards. Nothing re-reads them, so a file replaced between the two steps
would be loaded unverified: a TOCTOU window the PoC accepts because both steps run
inside one startup on a single-user machine (S-L5, README N08).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from jevbert.backends.base import CancelToken, EncodedSequence, TextPair
from jevbert.fetch import SOURCE_FILES, mismatched_files, unexpected_files

logger = logging.getLogger("jevbert.backend.nli")

#: Every special token string this build knows how to keep out of the data.
#:
#: ``split_special_tokens=True`` looks like the answer and is not: measured against this
#: tokenizer it still produced six control IDs from user text (POC_DESIGN 12.3, K2).
#: A checkpoint whose special tokens are not all in this list is refused at load time
#: (A-M5): escaping is only as complete as this list.
RESERVED_STRINGS: tuple[str, ...] = ("</s>", "<s>", "<pad>", "<unk>", "<mask>")

#: A tokenizer's normalizer, as a plain function of text.
Normalize = Callable[[str], str]

_ENTAILMENT_LABEL = "entailment"


def normalizer_of(tokenizer: Any) -> Normalize:
    """The tokenizer's own normalizer, as a function (S-H1).

    Not ``unicodedata.normalize("NFKC", ...)``: the two disagree on 181 code points for
    this checkpoint, and the disagreement is exactly where it hurts. The charsmap
    *deletes* the C0 control characters, so ``"<\\x01s>"`` normalizes to ``"<s>"`` and
    NFKC leaves it alone (POC_DESIGN 12.4). Matching the tokenizer means asking it.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend, "normalizer", None)
    if normalizer is None:
        raise ValueError(
            "the tokenizer exposes no normalizer, so the escape cannot be decided on "
            "the text the tokenizer will actually match against"
        )
    normalize: Normalize = normalizer.normalize_str
    return normalize


def reserved_strings_for(tokenizer: Any) -> tuple[str, ...]:
    """The checkpoint's special tokens, refusing any this build does not escape (A-M5).

    ``RESERVED_STRINGS`` is a constant, and a constant that silently fails to cover a
    checkpoint is worse than one that is missing: the escape would pass, the structural
    check would fire, and every request carrying that token would be a 500. Refusing to
    load makes it a readiness failure instead (spec 16.1).
    """
    declared = [str(token) for token in getattr(tokenizer, "all_special_tokens", ())]
    unknown = sorted(set(declared) - set(RESERVED_STRINGS))
    if unknown:
        raise ValueError(
            f"the checkpoint declares {len(unknown)} special token(s) this build does "
            "not know how to escape; a0-nli-zeroshot-v2 escapes "
            f"{', '.join(RESERVED_STRINGS)}"
        )
    if not declared:
        raise ValueError("the checkpoint declares no special token to protect")
    return tuple(declared)


def escape_reserved(
    text: str, normalize: Normalize, reserved: Sequence[str] = RESERVED_STRINGS
) -> str:
    """Disarm reserved strings, judged on the text the tokenizer will match against.

    ``"a<s>b</s>c"`` becomes ``"a< s>b< /s>c"``, and so does its full-width spelling:
    the decision is taken on ``normalize(text)``, so ``"\\uff1cs\\uff1e"`` - which folds
    to ``"<s>"`` - is disarmed too. Only spaces are inserted, and they are inserted into
    the caller's own characters, so the model still reads what the caller wrote
    (``compat/differences.md`` L08).

    Two stages, in this order:

    1. If the normalized text spells no reserved string, return the text **unchanged**.
       Ordinary prose containing ``"<"`` is therefore never rewritten.
    2. Otherwise put a space after every character that normalizes into a ``"<"``. Every
       ``"<"`` in the normalized output is then followed by a space, and no reserved
       string begins with ``"<"`` followed by a space, so none can survive.

    Locating the reserved string in the *original* text was tried and rejected: the
    charsmap deletes 30 control characters, so ``"<"`` plus arbitrarily many of them
    plus ``"s>"`` normalizes to ``"<s>"`` and no window of bounded width can see it
    (POC_DESIGN 12.4).

    The character-to-``"<"`` question is memoised for the length of one call only. The
    memo is a table of characters, it never leaves the call, and nothing derived from
    the caller's text outlives it (spec 15.2).
    """
    if not any(token in normalize(text) for token in reserved):
        return text

    folds_to_angle: dict[str, bool] = {}
    parts: list[str] = []
    for character in text:
        parts.append(character)
        angle = folds_to_angle.get(character)
        if angle is None:
            angle = "<" in normalize(character)
            folds_to_angle[character] = angle
        if angle:
            parts.append(" ")
    return "".join(parts)


def escape_every_angle(text: str, normalize: Normalize) -> str:
    """The stronger fallback: normalize the text, then space every ``"<"`` in it.

    :func:`escape_reserved` has to know which characters fold into a ``"<"``; this one
    does not, because it hands the tokenizer the normalizer's own output and edits that.
    It changes the caller's text further - compatibility characters are folded away -
    which is why it is a fallback and not the rule.
    """
    if "<" not in normalize(text):
        return text
    return normalize(text).replace("<", "< ")


def misplaced_control_token(
    ids: Sequence[int],
    premise_length: int,
    *,
    bos_token_id: int,
    eos_token_id: int,
    control_ids: frozenset[int],
) -> int | None:
    """The first index holding a control token the assembly did not put there (S-H1).

    ``[bos] premise [eos, eos] hypothesis [eos]``. Counting four control IDs is not
    enough: a sequence in which a separator moved into the caller's data and one of the
    caller's tokens took its place still carries four. The positions are known exactly -
    this method chose them - so they are what is checked.

    Returns:
        The lowest index that violates the structure, or ``None`` when it is intact.
    """
    if len(ids) < premise_length + 4:
        return len(ids)
    separator = premise_length + 1
    expected = {
        0: bos_token_id,
        separator: eos_token_id,
        separator + 1: eos_token_id,
        len(ids) - 1: eos_token_id,
    }
    for index, token_id in enumerate(ids):
        wanted = expected.get(index)
        if wanted is None:
            if token_id in control_ids:
                return index
        elif token_id != wanted:
            return index
    return None


def resolve_entailment_index(id2label: Mapping[int, str]) -> int:
    """Find the entailment class by name (POC_DESIGN 6.3).

    Index 0 is not assumed: a checkpoint that happens to order its labels the other way
    would otherwise return the negated score for every candidate, silently, and every
    invariant in spec 5.5 would still hold.
    """
    if len(id2label) != 2:
        raise ValueError(
            f"a0-nli-zeroshot-v2 needs a two-class NLI head, found {len(id2label)} labels. "
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

    ``longest x count`` is a memory model for scaled dot-product attention, which does
    not materialise the ``length^2`` score matrix. It is not a safe budget for an eager
    attention implementation, where a batch at the ceiling would allocate a temporary
    proportional to ``count x heads x longest^2`` (S-L5). transformers picks sdpa for
    this checkpoint; a build that forced eager would need this ceiling re-derived.

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

#: Everything a manifest may declare for a backend that computes with tensors.
COMPUTE_DTYPES: frozenset[str] = frozenset({"auto", *_DTYPES})


def resolve_dtype(declared: str, device: torch.device) -> torch.dtype:
    """The compute dtype for one device, from what the manifest declares.

    The CPU is always FP32: a half-precision matmul there is either unsupported or
    emulated, so it would be neither the fast path nor a usable reference. The
    declaration is still *validated* first (A-M4): short-circuiting on the device made
    a misspelled dtype silent on a CPU host and a load failure on a GPU host, which is
    the worst of both - the machine that would have caught it is the one in production.
    """
    if declared not in COMPUTE_DTYPES:
        raise ValueError(
            f"the manifest declares dtype={declared!r}; expected one of "
            f"{', '.join(sorted(COMPUTE_DTYPES))}"
        )
    if device.type == "cpu":
        return torch.float32
    if declared == "auto":
        return torch.float16
    return _DTYPES[declared]


class NliZeroShotBackend:
    """Zero-shot NLI backend over a locally pinned checkpoint.

    The weights are whatever ``python -m jevbert fetch-model`` put in ``model_dir``;
    nothing here resolves a repo ID, reaches the network or runs repository code
    (spec 15.3).
    """

    backend_id = "a0-nli-zeroshot-v2"

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
        self._control_ids: frozenset[int] = frozenset()
        self._normalize: Normalize | None = None
        self._reserved: tuple[str, ...] = RESERVED_STRINGS
        self._premise_tokenizations = 0
        self._fallback_escapes = 0

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
        carry 65,280 copies of the same state, and this is how a test sees that the state
        was tokenized once rather than once per candidate. It counts strings, never
        stores them.
        """
        return self._premise_tokenizations

    @property
    def fallback_escapes(self) -> int:
        """How many pairs needed the stronger escape (S-H1). Counts pairs, not text."""
        return self._fallback_escapes

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
        # Both before the weights: a checkpoint this build cannot escape safely must
        # fail readiness, not 500 on the requests that happen to exercise it (A-M5).
        self._normalize = normalizer_of(tokenizer)
        self._reserved = reserved_strings_for(tokenizer)
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
        # vocabulary cannot encode, and treating it as control would turn an exotic but
        # perfectly valid input into a 500.
        self._control_ids = frozenset(
            token_id
            for token_id in tokenizer.all_special_ids
            if token_id != tokenizer.unk_token_id
        )
        logger.info(
            "backend %s loaded: device=%s dtype=%s entailment_index=%d labels=%d "
            "reserved_strings=%d",
            self.backend_id,
            device,
            str(dtype).removeprefix("torch."),
            self._entailment_index,
            len(config.id2label),
            len(self._reserved),
        )

    def count_and_encode(self, pairs: Sequence[TextPair]) -> list[EncodedSequence]:
        """Escape, tokenize both sides separately, and assemble the pair sequence.

        Every distinct premise is tokenized once per call, as ``backends/base.py``
        requires: an A0 request repeats the same state for every candidate, so doing
        otherwise would pay for the state up to 65,280 times. The map dies with the call.

        A pair whose assembled sequence still carries a control token in a data position
        is re-encoded with the stronger escape rather than refused: a caller's text must
        not be able to turn a 200 into a 500 (S-H1).
        """
        tokenizer = self._require_tokenizer()
        if not pairs:
            return []

        distinct_premises: dict[str, int] = {}
        for pair in pairs:
            distinct_premises.setdefault(pair.premise, len(distinct_premises))
        premise_texts = [self._escape(text) for text in distinct_premises]
        premise_ids = tokenizer(premise_texts, add_special_tokens=False)["input_ids"]
        self._premise_tokenizations += len(premise_texts)

        hypothesis_ids = tokenizer(
            [self._escape(pair.hypothesis) for pair in pairs], add_special_tokens=False
        )["input_ids"]

        sequences: list[EncodedSequence] = []
        for pair, hypothesis in zip(pairs, hypothesis_ids, strict=True):
            premise = premise_ids[distinct_premises[pair.premise]]
            ids = self._assemble(premise, hypothesis)
            if misplaced_control_token(
                ids,
                len(premise),
                bos_token_id=self._bos_token_id,
                eos_token_id=self._eos_token_id,
                control_ids=self._control_ids,
            ) is not None:
                ids = self._reencode(pair)
            sequences.append(EncodedSequence(token_count=len(ids), data=ids))
        return sequences

    def _escape(self, text: str) -> str:
        """The rule of POC_DESIGN 5.4, with this checkpoint's own normalizer."""
        return escape_reserved(text, self._require_normalize(), self._reserved)

    def _escape_hard(self, text: str) -> str:
        return escape_every_angle(text, self._require_normalize())

    def _assemble(self, premise: Sequence[int], hypothesis: Sequence[int]) -> list[int]:
        """XLM-RoBERTa's pair form, built from IDs rather than parsed out of text."""
        bos, eos = self._bos_token_id, self._eos_token_id
        return [bos, *premise, eos, eos, *hypothesis, eos]

    def _reencode(self, pair: TextPair) -> list[int]:
        """Second attempt at one pair, with the stronger escape (S-H1).

        Reaching here means the targeted escape did not disarm something this
        tokenizer still read as a control token. That is a defect in the escape, not in
        the request, so it is logged for an operator - without any of the text - and the
        pair is encoded again from the normalizer's own output. Only if *that* still
        leaves a control token in a data position does the request fail, which is the
        one case where a 500 is the honest answer.
        """
        tokenizer = self._require_tokenizer()
        self._fallback_escapes += 1
        logger.warning(
            "backend %s: the targeted escape left a control token in a data position; "
            "re-encoding the pair with the stronger escape",
            self.backend_id,
        )
        premise = tokenizer(self._escape_hard(pair.premise), add_special_tokens=False)[
            "input_ids"
        ]
        hypothesis = tokenizer(
            self._escape_hard(pair.hypothesis), add_special_tokens=False
        )["input_ids"]
        self._premise_tokenizations += 1
        ids = self._assemble(premise, hypothesis)
        index = misplaced_control_token(
            ids,
            len(premise),
            bos_token_id=self._bos_token_id,
            eos_token_id=self._eos_token_id,
            control_ids=self._control_ids,
        )
        if index is not None:
            # Never scored: a sequence whose structure the caller wrote would make the
            # answer a reply to a question this server did not ask (spec 6.2, CT11).
            raise ValueError(
                "the encoded sequence carries a control token in a data position that "
                "neither escape could remove"
            )
        return ids

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
        """Every file the loader may open, checked before any of them is opened.

        The verified set has to *cover* the loaded set (S-M3): a manifest recording one
        hash for ``config.json`` used to be enough to start, and ``model.safetensors`` -
        the bytes that decide every answer - was then read without being checked at all.
        """
        if not self._expected_file_hashes:
            raise ValueError(
                "the manifest records no source_model.files, so the weights cannot be "
                "verified. Run `python -m jevbert fetch-model`."
            )
        unverified = sorted(set(SOURCE_FILES) - set(self._expected_file_hashes))
        if unverified:
            raise ValueError(
                f"the manifest records no hash for {', '.join(unverified)}, which this "
                "backend loads. Run `python -m jevbert fetch-model`."
            )
        extra = unexpected_files(self._model_dir)
        if extra:
            raise ValueError(
                f"{self._model_dir} holds {len(extra)} file(s) outside the fetch "
                f"allow-list: {', '.join(extra)}. The loader reads this directory by "
                "name, so nothing the manifest does not vouch for may sit in it."
            )
        bad = mismatched_files(self._model_dir, self._expected_file_hashes)
        if bad:
            raise ValueError(
                f"{len(bad)} model files are missing or do not match the manifest: "
                f"{', '.join(sorted(bad))}"
            )

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("the NLI backend has not been loaded")
        return self._tokenizer

    def _require_normalize(self) -> Normalize:
        if self._normalize is None:
            raise RuntimeError("the NLI backend has not been loaded")
        return self._normalize

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
