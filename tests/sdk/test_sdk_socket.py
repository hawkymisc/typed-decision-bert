"""Official Python SDK against a real socket (spec 13.7; POC_DESIGN 8.3).

Pinned to ``typesafe-sdk==0.7.0``. Retries are disabled so that a test observes the one
response the server actually sent: the SDK retries 408, 429 and every 5xx by default
(U06), which would otherwise hide a single 503 or 529 behind a later success.
"""

from __future__ import annotations

from typing import Any

import pytest
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

from tests.conftest import API_KEY, MODEL

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

    def test_extra_body_unknown_field_is_rejected(self, server_url: str) -> None:
        # spec 3.4: extra_body merges into the top level body, which is a 422 here.
        with client_for(server_url) as client:
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
        with pytest.raises(Exception, match="noul"):
            NoulAnswer.model_validate_json(raw)
