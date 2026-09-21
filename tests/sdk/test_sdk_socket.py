"""Official Python SDK against a real socket (spec 13.7; POC_DESIGN 8.3).

Pinned to ``typesafe-sdk==0.7.0``. Retries are disabled so that a test observes the one
response the server actually sent: the SDK retries 408, 429 and every 5xx by default
(U06), which would otherwise hide a single 503 or 529 behind a later success.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeUnprocessableEntityError,
)

from jevbert.backends.base import WARMUP_PREMISE, CancelToken
from jevbert.backends.fake import FakeBackend
from jevbert.config import ServingSettings
from tests.conftest import API_KEY, MODEL, build_settings
from tests.sdk.conftest import make_app, running_server

NO_RETRY = RetryPolicy(max_retries=0)

STATE = {"message": "同じ利用料金が二重に引き落とされました。今日中に重複分を返金してください。"}

QUESTIONS: dict[str, Any] = {
    "refund_requested": Noul(instructions="顧客は明示的に返金を求めていますか。"),
    "department": Choice(
        instructions="この問い合わせを最初に担当すべき部署を選んでください。",
        criteria={
            "billing": "請求、支払い、返金の問い合わせ",
            "technical": "ソフトウェアの不具合や接続障害",
            "other": "上記に該当しない問い合わせ",
        },
    ),
    "urgency": Score(
        instructions="顧客が表明している対応期限の切迫度を評価してください。",
        criteria=[
            "対応期限の指定がない",
            "数日以内の対応を求めている",
            "当日中または直ちに対応することを求めている",
        ],
    ),
}


def client_for(url: str, *, api_key: str = API_KEY) -> TypeSafeClient:
    return TypeSafeClient(base_url=url, api_key=api_key, model=MODEL, retry=NO_RETRY)


class TestSyncHappyPath:
    def test_three_types_mixed_in_japanese(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(state=STATE, questions=QUESTIONS)

        assert response.model == MODEL
        assert set(response.answers) == {"refund_requested", "department", "urgency"}

        noul = response.answers["refund_requested"]
        assert isinstance(noul, NoulAnswer)
        assert 0.0 <= noul.noul <= 1.0
        assert not hasattr(noul, "confidence")  # I08

        choice = response.answers["department"]
        assert isinstance(choice, ChoiceAnswer)
        assert choice.choice in {"billing", "technical", "other"}
        assert set(choice.probabilities) == {"billing", "technical", "other"}

        score = response.answers["urgency"]
        assert isinstance(score, ScoreAnswer)
        assert 0.0 <= score.score <= 2.0

        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens == 0

    def test_request_id_is_restored_from_the_header(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(state="s", questions={"q": Noul()})
        # spec 3.4 / ADR-013: reading this attribute raises when the header is missing.
        assert response.request_id
        assert isinstance(response.request_id, str)

    def test_score_keys_are_restored_as_integers(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(
                state="s", questions={"q": Score(criteria=["低", "中", "高"])}
            )
        answer = response.answers["q"]
        assert isinstance(answer, ScoreAnswer)
        # The wire carries "0".."K-1" as strings; the SDK converts them to ints.
        assert sorted(answer.probabilities) == [0, 1, 2]
        assert sorted(answer.legend) == [0, 1, 2]
        assert answer.legend[0] == "低"

    def test_structured_score_criteria_keep_their_type_in_the_legend(
        self, server_url: str
    ) -> None:
        # U03 / I07: the SDK's legend accepts str, object and array values.
        criteria: list[Any] = ["文字列", {"label": "中", "note": ["補足"]}, ["高", "至急"]]
        with client_for(server_url) as client:
            response = client.system_one(state="s", questions={"q": Score(criteria=criteria)})
        answer = response.answers["q"]
        assert isinstance(answer, ScoreAnswer)
        assert answer.legend[1] == {"label": "中", "note": ["補足"]}
        assert answer.legend[2] == ["高", "至急"]

    def test_one_sided_noul_criteria(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(
                state="s", questions={"q": Noul(criteria={"true": "返金を求めている"})}
            )
        assert isinstance(response.answers["q"], NoulAnswer)

    def test_dict_questions_are_accepted(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(
                state="s",
                questions={"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            )
        assert isinstance(response.answers["q"], ChoiceAnswer)

    def test_models_list(self, server_url: str) -> None:
        with client_for(server_url) as client:
            listing = client.models.list()
        names = [model.name for model in listing.models]
        assert MODEL in names
        entry = next(model for model in listing.models if model.name == MODEL)
        assert entry.release_date == "2026-09-21"
        assert entry.description


class TestSyncErrors:
    def test_bad_credentials_raise_authentication_error(self, server_url: str) -> None:
        with client_for(server_url, api_key="not-the-key") as client:
            with pytest.raises(TypeSafeAuthenticationError) as excinfo:
                client.system_one(state="s", questions={"q": Noul()})
        assert excinfo.value.status == 401
        assert excinfo.value.request_id
        # spec 5.9: error.message becomes the exception message.
        assert "credentials" in str(excinfo.value)

    def test_unknown_model_raises_unprocessable_entity(self, server_url: str) -> None:
        with TypeSafeClient(
            base_url=server_url, api_key=API_KEY, model="jev-1.13.0", retry=NO_RETRY
        ) as client:
            with pytest.raises(TypeSafeUnprocessableEntityError) as excinfo:
                client.system_one(state="s", questions={"q": Noul()})
        assert excinfo.value.status == 422

    def test_extra_body_unknown_field_is_accepted_by_default(self, server_url: str) -> None:
        # spec 3.4 / ADR-016: extra_body merges into the top level body, and an unknown
        # top level field is ignored by default rather than refused, because a caller
        # that works against Jev must keep working here.
        with client_for(server_url) as client:
            answer = client.system_one(
                state="s", questions={"q": Noul()}, extra_body={"temperature": 0.5}
            )
        assert set(answer.answers) == {"q"}

    def test_extra_body_unknown_field_is_refused_when_configured(
        self, strict_server_url: str
    ) -> None:
        with client_for(strict_server_url) as client:
            with pytest.raises(TypeSafeUnprocessableEntityError):
                client.system_one(
                    state="s", questions={"q": Noul()}, extra_body={"temperature": 0.5}
                )

    def test_too_many_score_levels_is_rejected(self, server_url: str) -> None:
        with client_for(server_url) as client:
            with pytest.raises(TypeSafeUnprocessableEntityError):
                client.system_one(
                    state="s", questions={"q": Score(criteria=[str(i) for i in range(11)])}
                )

    def test_model_unavailable_surfaces_as_internal_server_error(
        self, unready_server_url: str
    ) -> None:
        # spec 3.4: 503, 504 and 529 all look like TypeSafeInternalServerError; only
        # the `status` attribute tells them apart (the spec calls it `status`; there is
        # no `status_code` on SDK 0.7.0 exceptions).
        with client_for(unready_server_url) as client:
            with pytest.raises(TypeSafeInternalServerError) as excinfo:
                client.system_one(state="s", questions={"q": Noul()})
        assert excinfo.value.status == 503
        assert excinfo.value.request_id

    def test_overload_surfaces_as_529_through_the_same_exception(self) -> None:
        # The three statuses an operator most needs to tell apart are the three the
        # SDK collapses into one type, so the distinction is worth a test each.
        backend = GatedBackend()
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        app = make_app(backend=backend, settings=settings)
        with running_server(app) as url, ThreadPoolExecutor(max_workers=1) as pool:
            busy = pool.submit(_ask, url)
            assert backend.started.wait(10.0)
            with client_for(url) as client:
                with pytest.raises(TypeSafeInternalServerError) as excinfo:
                    client.system_one(state="s", questions={"q": Noul()})
            backend.release()
            busy.result()

        assert excinfo.value.status == 529
        assert excinfo.value.request_id

    def test_a_deadline_surfaces_as_504_through_the_same_exception(self) -> None:
        backend = GatedBackend()
        settings = build_settings(
            serving=ServingSettings(request_deadline_seconds=0.05)
        )
        app = make_app(backend=backend, settings=settings)
        try:
            with running_server(app) as url, client_for(url) as client:
                with pytest.raises(TypeSafeInternalServerError) as excinfo:
                    client.system_one(state="s", questions={"q": Noul()})
                assert backend.started.is_set()
        finally:
            backend.release()

        assert excinfo.value.status == 504
        assert excinfo.value.request_id


class GatedBackend(FakeBackend):
    """Holds one request inside ``score`` so that "busy" is a fact, not a race."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self._release = threading.Event()

    def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
        if any(sequence.data.premise != WARMUP_PREMISE for sequence in sequences):
            self.started.set()
            assert self._release.wait(20.0), "the gated backend was never released"
        return super().score(sequences, cancel)

    def release(self) -> None:
        self._release.set()


def _ask(url: str) -> None:
    with client_for(url) as client:
        client.system_one(state="s", questions={"q": Noul()})


class TestAsyncClient:
    async def test_three_types_mixed(self, server_url: str) -> None:
        async with AsyncTypeSafeClient(
            base_url=server_url, api_key=API_KEY, model=MODEL, retry=NO_RETRY
        ) as client:
            response = await client.system_one(state=STATE, questions=QUESTIONS)
        assert set(response.answers) == {"refund_requested", "department", "urgency"}
        assert response.request_id

    async def test_models_list(self, server_url: str) -> None:
        async with AsyncTypeSafeClient(
            base_url=server_url, api_key=API_KEY, model=MODEL, retry=NO_RETRY
        ) as client:
            listing = await client.models.list()
        assert MODEL in [model.name for model in listing.models]

    async def test_authentication_error(self, server_url: str) -> None:
        async with AsyncTypeSafeClient(
            base_url=server_url, api_key="wrong", model=MODEL, retry=NO_RETRY
        ) as client:
            with pytest.raises(TypeSafeAuthenticationError):
                await client.system_one(state="s", questions={"q": Noul()})

    async def test_sync_and_async_agree(self, server_url: str) -> None:
        with client_for(server_url) as sync_client:
            expected = sync_client.system_one(state=STATE, questions=QUESTIONS)
        async with AsyncTypeSafeClient(
            base_url=server_url, api_key=API_KEY, model=MODEL, retry=NO_RETRY
        ) as client:
            actual = await client.system_one(state=STATE, questions=QUESTIONS)
        assert actual.answers == expected.answers
        assert actual.usage == expected.usage


class TestStrictValidationObservations:
    """K5: what the SDK's ``strict=True`` response validation actually accepts."""

    def test_the_server_response_passes_strict_validation(self, server_url: str) -> None:
        with client_for(server_url) as client:
            response = client.system_one(
                state="s", questions={"q": Choice(criteria={"a": "A", "b": "B"})}
            )
        answer = response.answers["q"]
        assert isinstance(answer, ChoiceAnswer)
        assert all(isinstance(value, float) for value in answer.probabilities.values())

    def test_strict_mode_still_accepts_integer_json_numbers_for_float_fields(self) -> None:
        # Measured, not assumed: pydantic strict mode treats an int as acceptable for a
        # float field in both python and JSON mode. Writing decimals is therefore belt
        # and braces rather than a requirement of this SDK version.
        raw = b'{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}'
        assert ChoiceAnswer.model_validate_json(raw).confidence == 1.0

    def test_strict_mode_rejects_a_string_where_a_float_is_declared(self) -> None:
        raw = b'{"type":"noul","noul":"0.5"}'
        with pytest.raises(ValidationError) as excinfo:
            NoulAnswer.model_validate_json(raw)
        # The SDK validates with pydantic, so the rejection is a pydantic
        # ValidationError naming the field - not just "some exception happened".
        assert excinfo.value.error_count() == 1
        assert excinfo.value.errors()[0]["loc"] == ("noul",)


class TestJevCompatibilityOverTheSocket:
    """ADR-016: what a caller written against Jev sees when it is pointed here."""

    def test_the_sdk_default_model_resolves_through_the_alias(
        self, alias_server_url: str
    ) -> None:
        """The drop-in case (C2, spec 18.3): only base_url and api_key change.

        ``TypeSafeClient`` defaults ``model`` to ``jev-latest``, so a caller that never
        names a model is the caller most likely to be pointed at this server unchanged.
        """
        with TypeSafeClient(
            base_url=alias_server_url, api_key=API_KEY, retry=NO_RETRY
        ) as client:
            response = client.system_one(state="s", questions={"q": Noul()})
        # D01: the answer names the immutable bundle, never the alias that was asked for.
        assert response.model == MODEL
        assert isinstance(response.answers["q"], NoulAnswer)

    def test_the_preview_alias_resolves_too(self, alias_server_url: str) -> None:
        with TypeSafeClient(
            base_url=alias_server_url, api_key=API_KEY, model="jev-preview", retry=NO_RETRY
        ) as client:
            assert client.system_one(state="s", questions={"q": Noul()}).model == MODEL

    def test_a_422_body_carries_jev_s_detail_array(self, server_url: str) -> None:
        with client_for(server_url) as client:
            with pytest.raises(TypeSafeUnprocessableEntityError) as excinfo:
                client.system_one(
                    state="s", questions={"urgency": Score(criteria=["only one"])}
                )
        body = excinfo.value.body
        assert body["detail"][0]["loc"][0] == "body"
        # spec 3.4: Jev's own example puts the question type after the question ID.
        assert body["detail"][0]["loc"] == ["body", "questions", "urgency", "score", "criteria"]
        assert body["detail"][0]["type"] == "validation_error"

    def test_the_detail_array_does_not_change_the_exception_message(
        self, server_url: str
    ) -> None:
        # spec 5.9: the SDK extracts error.message first, so adding `detail` must not
        # move the message a caller already logs.
        with client_for(server_url) as client:
            with pytest.raises(TypeSafeUnprocessableEntityError) as excinfo:
                client.system_one(
                    state="s", questions={"urgency": Score(criteria=["only one"])}
                )
        # The SDK frames the message it extracted; what matters is that the text it
        # extracted is error.message and not something from the new `detail` array.
        assert excinfo.value.body["error"]["message"] in str(excinfo.value)
        assert "levels" in str(excinfo.value)

    def test_many_questions_are_accepted(self, server_url: str) -> None:
        # spec 4.2: 256, up from 32, so a count Jev would take is not refused here.
        with client_for(server_url) as client:
            response = client.system_one(
                state="s", questions={f"q{index}": Noul() for index in range(256)}
            )
        assert len(response.answers) == 256
