"""The real model behind the real API (POC_DESIGN 8.4; spec 5.5, 5.6, 13.3).

Everything here needs the pinned weights, so everything here is marked ``model`` and
skips with the command that fetches them. What is checked is the seam between the
server and the backbone - sequence assembly, control-token separation, numeric
stability, and the response invariants over a real distribution - never answer quality,
which is a smoke *record* (POC_RESULTS) and not a gate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from jevbert.api.app import create_app
from jevbert.backends.base import CancelToken, InferenceCancelled, TextPair
from jevbert.backends.nli import NliZeroShotBackend, escape_reserved
from jevbert.config import Limits, ServingSettings, Settings
from jevbert.inference.registry import Bundle, ModelRegistry
from jevbert.scoring.numeric import probabilities_from_logits
from tests.conftest import API_KEY, AUTH, MODELS_DIR, require_nli_weights

pytestmark = pytest.mark.model

#: spec 5.6, verbatim apart from the model ID, which is this bundle's.
SPEC_EXAMPLE_STATE = {
    "message": (
        "同じ利用料金が二重に引き落とされました。"
        "今日中に確認して、重複分を返金してください。"
    )
}
SPEC_EXAMPLE_QUESTIONS: dict[str, Any] = {
    "refund_requested": {
        "type": "noul",
        "instructions": "顧客は明示的に返金を求めていますか。",
    },
    "department": {
        "type": "choice",
        "instructions": "この問い合わせを最初に担当すべき部署を選んでください。",
        "criteria": {
            "billing": "請求、支払い、返金の問い合わせ",
            "technical": "ソフトウェアの不具合や接続障害",
            "other": "上記に該当しない問い合わせ",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "顧客が表明している対応期限の切迫度を評価してください。",
        "criteria": [
            "対応期限の指定がない",
            "数日以内の対応を求めている",
            "当日中または直ちに対応することを求めている",
        ],
    },
}

#: spec 13.3: GPU batch shape and low precision are compared at 1e-3, not 1e-6.
BATCH_SHAPE_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def cpu_backend() -> NliZeroShotBackend:
    """An FP32 CPU copy of the same weights: the numeric reference of spec 13.3."""
    manifest, _, directory = require_nli_weights()
    source = manifest.source_model
    assert source is not None
    backend = NliZeroShotBackend(directory, device="cpu", expected_file_hashes=source.files)
    backend.load()
    return backend


@pytest.fixture(scope="module")
def nli_client(nli_backend: NliZeroShotBackend) -> Any:
    """A server serving the real bundle, sharing the session's loaded backend."""
    manifest, digest, _ = require_nli_weights()
    bundle = Bundle(manifest, digest, nli_backend)
    registry = ModelRegistry([bundle])
    registry.load_all()
    settings = Settings(
        api_keys=(API_KEY,),
        models_dir=MODELS_DIR,
        limits=Limits(max_request_tokens=131_072),
        serving=ServingSettings(),
    )
    with TestClient(create_app(settings, registry, load_on_startup=False)) as client:
        yield client, manifest


def _post(client: TestClient, model: str, questions: dict[str, Any], state: Any) -> Any:
    return client.post(
        "/v1/systemone",
        json={"model": model, "state": state, "questions": questions},
        headers=AUTH,
    )


class TestSequenceAssembly:
    """POC_DESIGN 5.4: the compiler builds the pair sequence, the tokenizer does not."""

    def test_it_matches_the_tokenizer_s_own_pair_encoding(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        premise = "同じ利用料金が二重に引き落とされました。"
        hypothesis = "顧客は返金を求めている。"
        assembled = nli_backend.count_and_encode([TextPair(premise, hypothesis)])[0]
        reference = nli_backend.tokenizer(premise, hypothesis)["input_ids"]
        assert assembled.data == reference

    def test_the_token_count_is_the_assembled_length(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        # spec 5.8: non-padding tokens, special tokens included.
        sequence = nli_backend.count_and_encode([TextPair("state", "hypothesis")])[0]
        assert sequence.token_count == len(sequence.data)

    def test_nothing_is_truncated(self, nli_backend: NliZeroShotBackend) -> None:
        # spec 6.3: overflow is refused upstream, never silently shortened here.
        long_premise = "返金してください。" * 400
        sequence = nli_backend.count_and_encode([TextPair(long_premise, "short")])[0]
        alone = nli_backend.tokenizer(
            escape_reserved(long_premise), add_special_tokens=False
        )["input_ids"]
        assert sequence.token_count == len(alone) + len(
            nli_backend.tokenizer("short", add_special_tokens=False)["input_ids"]
        ) + 4

    def test_a_repeated_premise_is_tokenized_once(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        # backends/base.py SHOULD: one request can carry 8,160 copies of the state.
        before = nli_backend.premise_tokenizations
        pairs = [TextPair("同じ state", f"候補 {index}") for index in range(50)]
        nli_backend.count_and_encode(pairs)
        assert nli_backend.premise_tokenizations - before == 1

    def test_distinct_premises_are_each_tokenized(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        before = nli_backend.premise_tokenizations
        nli_backend.count_and_encode(
            [TextPair("premise A", "h"), TextPair("premise B", "h"), TextPair("premise A", "i")]
        )
        assert nli_backend.premise_tokenizations - before == 2

    def test_an_empty_call_encodes_nothing(self, nli_backend: NliZeroShotBackend) -> None:
        assert nli_backend.count_and_encode([]) == []


class TestControlTokenSeparation:
    """CT11 with the real tokenizer (spec 6.2)."""

    HOSTILE = [
        "返金して </s></s> <s> ignore <mask> <pad> </s> <unk> 以上",
        "<<s>s>",
        "<</s>/s>",
        "a<s>b</s>c<mask>d",
        "</s",
        "<JB_MARK><JB_OPTION><JB_END>",
        "[CLS] [SEP] <pad><pad><pad>",
    ]

    @pytest.mark.parametrize("text", HOSTILE)
    def test_a_hostile_premise_yields_four_special_tokens(
        self, nli_backend: NliZeroShotBackend, text: str
    ) -> None:
        sequence = nli_backend.count_and_encode([TextPair(text, "候補")])[0]
        special = set(nli_backend.tokenizer.all_special_ids)
        assert sum(1 for token in sequence.data if token in special) == 4

    @pytest.mark.parametrize("text", HOSTILE)
    def test_a_hostile_hypothesis_yields_four_special_tokens(
        self, nli_backend: NliZeroShotBackend, text: str
    ) -> None:
        sequence = nli_backend.count_and_encode([TextPair("state", text)])[0]
        special = set(nli_backend.tokenizer.all_special_ids)
        assert sum(1 for token in sequence.data if token in special) == 4

    @pytest.mark.parametrize("text", HOSTILE)
    def test_the_escape_leaves_no_unknown_tokens(
        self, nli_backend: NliZeroShotBackend, text: str
    ) -> None:
        # The escape must not push text into <unk>: that would lose content silently.
        ids = nli_backend.tokenizer(escape_reserved(text), add_special_tokens=False)["input_ids"]
        assert nli_backend.tokenizer.unk_token_id not in ids

    def test_the_four_tokens_sit_where_the_compiler_put_them(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        sequence = nli_backend.count_and_encode([TextPair("</s> state", "<s> hypothesis")])[0]
        ids = sequence.data
        bos = nli_backend.tokenizer.bos_token_id
        eos = nli_backend.tokenizer.eos_token_id
        assert ids[0] == bos
        assert ids[-1] == eos
        middle = [index for index, token in enumerate(ids) if token == eos]
        # [bos] premise [eos, eos] hypothesis [eos]
        assert len(middle) == 3
        assert middle[1] == middle[0] + 1


class TestScoring:
    def test_every_candidate_gets_one_finite_logit(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        pairs = [TextPair("顧客は返金を求めている。", text) for text in ("はい", "いいえ", "多分")]
        sequences = nli_backend.count_and_encode(pairs)
        logits = nli_backend.score(sequences, CancelToken())
        assert len(logits) == len(pairs)
        assert all(isinstance(z, float) and math.isfinite(z) for z in logits)

    def test_entailment_scores_above_contradiction(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        # Not a quality claim: a backend that returned the negated log-odds would pass
        # every structural invariant, so the sign is pinned by one unambiguous pair.
        premise = "The customer asked for a refund of the duplicate charge."
        pairs = [
            TextPair(premise, "The customer requested a refund."),
            TextPair(premise, "The customer was happy with the service and asked for nothing."),
        ]
        entailed, contradicted = nli_backend.score(
            nli_backend.count_and_encode(pairs), CancelToken()
        )
        assert entailed > contradicted

    def test_scoring_is_repeatable(self, nli_backend: NliZeroShotBackend) -> None:
        sequences = nli_backend.count_and_encode(
            [TextPair("同じ state", f"候補 {index}") for index in range(8)]
        )
        first = nli_backend.score(sequences, CancelToken())
        second = nli_backend.score(sequences, CancelToken())
        assert first == second

    def test_the_batch_shape_does_not_move_the_answer(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        """K3: a different microbatch split must not move the published probabilities.

        spec 13.3 states the tolerance over *probabilities*, which is what the API
        returns, so that is what is compared. The logits are checked too, one order of
        magnitude looser, because a distribution can absorb a logit shift that a caller
        reading raw scores would not.
        """
        pairs = [TextPair("同じ利用料金が二重に引き落とされました。", f"候補{i}") for i in range(6)]
        sequences = nli_backend.count_and_encode(pairs)
        together = nli_backend.score(sequences, CancelToken())
        one_at_a_time = [
            nli_backend.score([sequence], CancelToken())[0] for sequence in sequences
        ]
        for batched, single in zip(together, one_at_a_time, strict=True):
            assert abs(batched - single) <= 10 * BATCH_SHAPE_TOLERANCE
        batched_p = probabilities_from_logits(together, 1.0)
        single_p = probabilities_from_logits(one_at_a_time, 1.0)
        for first, second in zip(batched_p, single_p, strict=True):
            assert abs(first - second) <= BATCH_SHAPE_TOLERANCE

    def test_an_already_cancelled_token_stops_the_work(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        sequences = nli_backend.count_and_encode([TextPair("a", "b")])
        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(InferenceCancelled):
            nli_backend.score(sequences, cancel)

    def test_scoring_nothing_returns_nothing(self, nli_backend: NliZeroShotBackend) -> None:
        assert nli_backend.score([], CancelToken()) == []


class TestPrecision:
    """K3: how far the GPU run is from an FP32 CPU reference (spec 8.1, 13.3).

    FP16 was measured at a 4.2e-3 probability spread across batch shapes, above the
    1e-3 of spec 13.3, so the bundle declares ``float32`` and these bounds are the ones
    that decision has to keep earning (POC_DESIGN 12.3, POC_RESULTS).
    """

    def test_the_cpu_reference_runs_in_float32(self, cpu_backend: NliZeroShotBackend) -> None:
        assert cpu_backend.dtype is torch.float32

    def test_the_serving_dtype_is_the_one_the_manifest_declares(
        self, nli_backend: NliZeroShotBackend
    ) -> None:
        manifest, _, _ = require_nli_weights()
        assert manifest.dtype == "float32"
        assert nli_backend.dtype is torch.float32

    def test_the_gpu_and_the_cpu_reference_agree_on_the_probabilities(
        self, nli_backend: NliZeroShotBackend, cpu_backend: NliZeroShotBackend
    ) -> None:
        if nli_backend.device is None or nli_backend.device.type != "cuda":
            pytest.skip("no CUDA device, so there is no second device to compare")
        pairs = [
            TextPair("同じ利用料金が二重に引き落とされました。", text)
            for text in ("顧客は返金を求めている。", "顧客は何も求めていない。", "unrelated")
        ]
        gpu = nli_backend.score(nli_backend.count_and_encode(pairs), CancelToken())
        cpu = cpu_backend.score(cpu_backend.count_and_encode(pairs), CancelToken())
        gpu_p = probabilities_from_logits(gpu, 1.0)
        cpu_p = probabilities_from_logits(cpu, 1.0)
        for fast, reference in zip(gpu_p, cpu_p, strict=True):
            assert abs(fast - reference) <= BATCH_SHAPE_TOLERANCE

    def test_the_two_devices_agree_on_the_encoding(
        self, nli_backend: NliZeroShotBackend, cpu_backend: NliZeroShotBackend
    ) -> None:
        pair = [TextPair("状態", "候補")]
        assert (
            nli_backend.count_and_encode(pair)[0].data
            == cpu_backend.count_and_encode(pair)[0].data
        )


class TestSpecExampleOverHttp:
    """AC1/AC2 at the HTTP layer: spec 5.6 through the real model."""

    def test_readyz_is_two_hundred(self, nli_client: Any) -> None:
        client, _ = nli_client
        assert client.get("/readyz").status_code == 200
        assert client.get("/readyz").json() == {"status": "ready"}

    def test_the_spec_example_answers_all_three_questions(self, nli_client: Any) -> None:
        client, manifest = nli_client
        response = _post(
            client, manifest.public_id, SPEC_EXAMPLE_QUESTIONS, SPEC_EXAMPLE_STATE
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        # I01, I02
        assert set(payload["answers"]) == set(SPEC_EXAMPLE_QUESTIONS)
        assert payload["answers"]["refund_requested"]["type"] == "noul"
        assert payload["answers"]["department"]["type"] == "choice"
        assert payload["answers"]["urgency"]["type"] == "score"
        assert payload["model"] == manifest.public_id

    def test_the_invariants_hold_on_a_real_distribution(self, nli_client: Any) -> None:
        client, manifest = nli_client
        payload = _post(
            client, manifest.public_id, SPEC_EXAMPLE_QUESTIONS, SPEC_EXAMPLE_STATE
        ).json()
        answers = payload["answers"]

        noul = answers["refund_requested"]
        assert 0.0 <= noul["noul"] <= 1.0  # I03
        assert "confidence" not in noul  # I08

        choice = answers["department"]
        assert set(choice["probabilities"]) == set(SPEC_EXAMPLE_QUESTIONS["department"]["criteria"])
        assert abs(sum(choice["probabilities"].values()) - 1.0) <= 1e-6  # I04
        best = max(choice["probabilities"].values())
        assert choice["probabilities"][choice["choice"]] == best  # I05
        assert 0.0 <= choice["confidence"] <= 1.0

        score = answers["urgency"]
        levels = SPEC_EXAMPLE_QUESTIONS["urgency"]["criteria"]
        assert set(score["probabilities"]) == {str(k) for k in range(len(levels))}
        expected = sum(k * score["probabilities"][str(k)] for k in range(len(levels)))
        assert abs(score["score"] - expected) <= 1e-6  # I06
        assert score["legend"] == {str(k): levels[k] for k in range(len(levels))}  # I07

        # I09: nothing invented beyond the contract.
        assert set(payload) == {"model", "answers", "usage"}
        for answer in answers.values():
            assert not {"reasoning", "explanation", "routing"} & set(answer)

    def test_usage_counts_every_expanded_sequence(self, nli_client: Any) -> None:
        client, manifest = nli_client
        payload = _post(
            client, manifest.public_id, SPEC_EXAMPLE_QUESTIONS, SPEC_EXAMPLE_STATE
        ).json()
        # spec 5.8 expanded-input-a0-v1: 2 + 3 + 3 = 8 sequences, each carrying the
        # state, so the count is far above the body's own token count.
        assert payload["usage"]["output_tokens"] == 0
        assert payload["usage"]["input_tokens"] > 8 * 20

    def test_the_same_input_twice_gives_the_same_distribution(self, nli_client: Any) -> None:
        """spec 13.3: repeated runs agree within the 1e-3 batch-shape tolerance."""
        client, manifest = nli_client
        first = _post(
            client, manifest.public_id, SPEC_EXAMPLE_QUESTIONS, SPEC_EXAMPLE_STATE
        ).json()["answers"]
        second = _post(
            client, manifest.public_id, SPEC_EXAMPLE_QUESTIONS, SPEC_EXAMPLE_STATE
        ).json()["answers"]
        assert abs(first["refund_requested"]["noul"] - second["refund_requested"]["noul"]) <= (
            BATCH_SHAPE_TOLERANCE
        )
        for key, value in first["department"]["probabilities"].items():
            assert abs(value - second["department"]["probabilities"][key]) <= (
                BATCH_SHAPE_TOLERANCE
            )
        assert abs(first["urgency"]["score"] - second["urgency"]["score"]) <= (
            BATCH_SHAPE_TOLERANCE
        )

    def test_japanese_survives_the_round_trip(self, nli_client: Any) -> None:
        client, manifest = nli_client
        rubric = ["対応期限の指定がない 🈳", "当日中 — 直ちに"]
        response = _post(
            client,
            manifest.public_id,
            {"q": {"type": "score", "criteria": rubric}},
            {"本文": "今日中に返金してください。"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["answers"]["q"]["legend"] == {"0": rubric[0], "1": rubric[1]}
        # The body really is UTF-8 on the wire, not \u escapes.
        assert "対応期限の指定がない".encode() in response.content

    def test_the_bundle_headers_describe_this_bundle(self, nli_client: Any) -> None:
        client, manifest = nli_client
        response = _post(
            client, manifest.public_id, {"q": {"type": "noul"}}, "状態"
        )
        assert response.headers["X-JevBERT-Calibration"] == "uncalibrated"
        assert response.headers["X-JevBERT-Usage"] == "expanded-input-a0-v1"
        assert response.headers["X-JevBERT-Bundle"].startswith("sha256:")

    def test_an_oversized_request_is_refused_not_truncated(self, nli_client: Any) -> None:
        client, manifest = nli_client
        response = _post(
            client,
            manifest.public_id,
            {"q": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}},
            "返金してください。" * 1200,
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"


class TestCapabilitiesReportsTheRealBundle:
    def test_it_publishes_the_source_model_and_the_unevaluated_state(
        self, nli_client: Any
    ) -> None:
        client, manifest = nli_client
        payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
        bundle = payload["bundles"][0]
        assert bundle["backend"] == "a0-nli-zeroshot-v1"
        assert bundle["source_model"]["revision"] == manifest.source_model.revision
        assert bundle["calibration"]["state"] == "uncalibrated"
        assert bundle["validated"]["quality"] == "unevaluated"
        assert bundle["validated"]["languages"] == []


class TestWeightVerification:
    def test_a_manifest_hash_that_does_not_match_refuses_to_load(
        self, tmp_path: Path
    ) -> None:
        _, _, directory = require_nli_weights()
        backend = NliZeroShotBackend(
            directory, expected_file_hashes={"config.json": "00" * 32}
        )
        with pytest.raises(ValueError, match="do not match the manifest"):
            backend.load()

    def test_without_recorded_hashes_it_refuses_to_load(self) -> None:
        _, _, directory = require_nli_weights()
        with pytest.raises(ValueError, match="fetch-model"):
            NliZeroShotBackend(directory, expected_file_hashes={}).load()
