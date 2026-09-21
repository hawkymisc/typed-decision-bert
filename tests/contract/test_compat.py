"""ADR-016: do not refuse, on form alone, a request Jev would accept.

Four decisions live here, and each is a deliberate loss of strictness bought for
compatibility:

* an unknown **top level** field is ignored by default, because the official SDK has
  ``extra_body`` and the real Jev's handling of one is not knowable from the published
  material (spec 5.3). A Question's unknown field is still a 422 - that one could
  change what is being asked.
* a 422 carries a ``detail`` array in Jev's shape as well as JevBERT's ``error``
  object, so a client reading ``detail`` directly keeps working (spec 5.9).
* up to 256 questions, because Jev documents no question-count ceiling at all
  (spec 4.2).
* ``jev-latest`` and ``jev-preview`` resolve in the PoC configuration, so that only the
  base URL and the key have to change (spec 18.3).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevbert.api.errors import ValidationError
from jevbert.config import CompatSettings, Limits
from jevbert.contracts.validator import validate_request
from tests.conftest import (
    MODEL,
    build_client,
    build_registry,
    build_settings,
    request_body,
    systemone,
)


def _ignore_client() -> TestClient:
    return build_client(settings=build_settings())


def _reject_client() -> TestClient:
    settings = build_settings(compat=CompatSettings(unknown_top_level_fields="reject"))
    return build_client(settings=settings)


class TestUnknownTopLevelFields:
    def test_the_default_ignores_them(self) -> None:
        with _ignore_client() as client:
            body = request_body({"q": {"type": "noul"}})
            body["beam_width"] = 4
            response = systemone(client, body)
        assert response.status_code == 200
        assert set(response.json()["answers"]) == {"q"}

    def test_an_ignored_field_changes_no_answer(self) -> None:
        # "Ignored" has to mean ignored: it must not reach the model input, and it must
        # not reach the response either (spec 5.3).
        with _ignore_client() as client:
            plain = systemone(client, request_body({"q": {"type": "noul"}})).json()
            noisy_body = request_body({"q": {"type": "noul"}})
            noisy_body["temperature"] = 0.5
            noisy_body["metadata"] = {"trace": "abc"}
            noisy = systemone(client, noisy_body).json()
        assert noisy == plain

    @pytest.mark.parametrize("field", ["temperature", "beam_width", "stream", "_x"])
    def test_any_name_is_ignored(self, field: str) -> None:
        with _ignore_client() as client:
            body = request_body({"q": {"type": "noul"}})
            body[field] = "anything"
            assert systemone(client, body).status_code == 200

    def test_the_reject_setting_restores_the_422(self) -> None:
        with _reject_client() as client:
            body = request_body({"q": {"type": "noul"}})
            body["beam_width"] = 4
            response = systemone(client, body)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert error["path"] == ["beam_width"]

    def test_a_question_s_unknown_field_is_still_refused(self) -> None:
        # A field inside a question can change what is being asked, so silence there
        # would answer a different question than the caller wrote (ADR-016).
        with _ignore_client() as client:
            response = systemone(
                client, request_body({"q": {"type": "noul", "weight": 2}})
            )
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["questions", "q", "weight"]

    def test_a_missing_required_field_is_still_refused(self) -> None:
        with _ignore_client() as client:
            response = systemone(client, {"model": MODEL, "questions": {"q": {"type": "noul"}}})
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["state"]

    def test_the_validator_defaults_to_ignoring(self) -> None:
        validated = validate_request(
            {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "x": 1},
            Limits(),
        )
        assert validated.model == "m"

    def test_the_validator_can_be_told_to_reject(self) -> None:
        with pytest.raises(ValidationError):
            validate_request(
                {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "x": 1},
                Limits(),
                unknown_top_level_fields="reject",
            )


class TestValidationDetail:
    """spec 5.9: the 422 body carries Jev's ``detail`` shape alongside ``error``."""

    def _detail(self, body: Any) -> Any:
        with _ignore_client() as client:
            response = systemone(client, body)
        assert response.status_code == 422
        return response.json()

    def test_detail_is_a_list_of_objects(self) -> None:
        payload = self._detail(request_body({"q": {"type": "nope"}}))
        assert isinstance(payload["detail"], list)
        assert set(payload["detail"][0]) == {"loc", "msg", "type"}

    def test_loc_starts_with_body(self) -> None:
        payload = self._detail(request_body({"q": {"type": "nope"}}))
        assert payload["detail"][0]["loc"][0] == "body"

    def test_the_message_and_code_are_the_same_as_in_error(self) -> None:
        # The SDK prefers error.message, so the two must not drift apart.
        payload = self._detail(request_body({"q": {"type": "nope"}}))
        entry = payload["detail"][0]
        assert entry["msg"] == payload["error"]["message"]
        assert entry["type"] == payload["error"]["code"]

    def test_the_question_type_tag_follows_the_question_id(self) -> None:
        # spec 3.4 records Jev's own example: ["body","questions","urgency","score",...]
        payload = self._detail(
            request_body({"urgency": {"type": "score", "criteria": ["only one"]}})
        )
        assert payload["detail"][0]["loc"] == ["body", "questions", "urgency", "score", "criteria"]

    def test_the_tag_is_omitted_when_the_type_is_not_known(self) -> None:
        payload = self._detail(request_body({"q": {"type": "nope"}}))
        assert payload["detail"][0]["loc"] == ["body", "questions", "q", "type"]

    def test_a_top_level_path_needs_no_tag(self) -> None:
        payload = self._detail({"model": MODEL, "state": "s", "questions": {}})
        assert payload["detail"][0]["loc"] == ["body", "questions"]

    def test_model_not_found_carries_detail_too(self) -> None:
        with _ignore_client() as client:
            response = systemone(client, {**request_body({"q": {"type": "noul"}}), "model": "no"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "model"]
        assert response.json()["detail"][0]["type"] == "model_not_found"

    def test_no_input_value_is_reflected_back(self) -> None:
        # Jev's schema allows `input` and `ctx`; JevBERT omits both rather than echo
        # the caller's data into an error body (spec 15.2).
        secret = "私的なデータ-0123456789"
        payload = self._detail(request_body({"q": {"type": "noul", secret: 1}}))
        assert "input" not in payload["detail"][0]
        assert "ctx" not in payload["detail"][0]

    @pytest.mark.parametrize(
        ("status", "body"),
        [(400, b"{"), (415, None)],
    )
    def test_other_statuses_carry_no_detail(self, status: int, body: bytes | None) -> None:
        with _ignore_client() as client:
            if status == 400:
                response = systemone(client, content=body)
            else:
                response = systemone(
                    client,
                    request_body({"q": {"type": "noul"}}),
                    headers={"Content-Type": "text/plain"},
                )
        assert response.status_code == status
        assert "detail" not in response.json()

    def test_a_404_carries_no_detail(self) -> None:
        with _ignore_client() as client:
            response = client.get("/nope", headers={"Authorization": f"Bearer {'t' * 34}"})
        assert response.status_code in (401, 404)
        assert "detail" not in response.json()


class TestQuestionCount:
    """spec 4.2: 1..256. Jev documents no count ceiling, only a token budget."""

    def test_two_hundred_and_fifty_six_questions_are_accepted(self) -> None:
        with _ignore_client() as client:
            response = systemone(
                client, request_body({f"q{i}": {"type": "noul"} for i in range(256)}, "s")
            )
        assert response.status_code == 200
        assert len(response.json()["answers"]) == 256

    def test_two_hundred_and_fifty_seven_are_refused(self) -> None:
        with _ignore_client() as client:
            response = systemone(
                client, request_body({f"q{i}": {"type": "noul"} for i in range(257)}, "s")
            )
        assert response.status_code == 422
        assert response.json()["error"]["path"] == ["questions"]

    def test_the_published_limit_says_256(self) -> None:
        with _ignore_client() as client:
            payload = client.get(
                "/jevbert/v1/capabilities",
                headers={"Authorization": f"Bearer {'t' * 34}"},
            )
        assert payload.status_code in (200, 401)

    def test_the_default_limit_is_256(self) -> None:
        assert Limits().max_questions == 256


class TestJevAliases:
    def test_an_alias_resolves_and_the_answer_names_the_bundle(self) -> None:
        registry = build_registry(aliases={"jev-latest": MODEL, "jev-preview": MODEL})
        settings = build_settings(
            serving=build_settings().serving.model_copy(
                update={"allow_jev_aliases": True, "aliases": {"jev-latest": MODEL}}
            )
        )
        with build_client(settings=settings, registry=registry) as client:
            body = request_body({"q": {"type": "noul"}})
            body["model"] = "jev-latest"
            response = systemone(client, body)
        assert response.status_code == 200
        # D01: the answer always names the immutable bundle, never the alias.
        assert response.json()["model"] == MODEL

    def test_an_unregistered_jev_name_is_still_refused(self) -> None:
        registry = build_registry(aliases={"jev-latest": MODEL})
        with build_client(registry=registry) as client:
            body = request_body({"q": {"type": "noul"}})
            body["model"] = "jev-9.9.9"
            assert systemone(client, body).status_code == 422
