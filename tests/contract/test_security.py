"""Security review findings (phase 1.5 S-H1..S-L2, phase 2.5 S-L3).

Each test here fixes a behaviour that the review found missing, not an implementation
detail: a hostile header must not become a 500, a hostile literal must not become a 500,
a new route must not be reachable without credentials, and no error body may reflect the
caller's own data back at an unauthenticated reader.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jevbert.config import CompatSettings, Limits, ServingSettings
from tests.conftest import (
    API_KEY,
    AUTH,
    MODEL,
    build_client,
    build_registry,
    build_settings,
    request_body,
    systemone,
)

NOUL = {"q": {"type": "noul"}}

#: ASGI decodes request headers as latin-1, so this is exactly what a caller sending
#: the raw bytes ``b"Bearer t\xe9st"`` produces on the server side (S-H1).
LATIN1_TOKEN = b"Bearer t\xe9st"

#: Non-ASCII key of at least the minimum length, to prove the comparison is byte based.
UNICODE_KEY = "鍵" * 32


class TestSH1NonAsciiCredentials:
    def test_a_non_ascii_bearer_token_is_unauthorized_not_a_crash(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/systemone",
            content=b'{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}',
            headers={"Authorization": LATIN1_TOKEN, "Content-Type": "application/json"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    @pytest.mark.parametrize(
        "token",
        [
            b"Bearer t\xe9st",
            ("Bearer " + UNICODE_KEY).encode(),
            ("Bearer " + "ÿ" * 40).encode("latin-1"),
            "Bearer 𝐤𝐞𝐲".encode(),
        ],
    )
    def test_no_credential_shape_can_produce_a_500(
        self, client: TestClient, token: bytes
    ) -> None:
        # An HTTP client sends header bytes; the ASGI server decodes them as latin-1.
        response = client.get("/v1/models", headers={"Authorization": token})
        assert response.status_code == 401

    def test_a_configured_non_ascii_key_still_authenticates(self) -> None:
        settings = build_settings(api_keys=(UNICODE_KEY,))
        with build_client(settings=settings) as client:
            response = systemone(
                client,
                request_body(NOUL),
                headers={"Authorization": ("Bearer " + UNICODE_KEY).encode()},
            )
            assert response.status_code == 200

    def test_a_configured_non_ascii_key_still_rejects_the_wrong_token(self) -> None:
        settings = build_settings(api_keys=(UNICODE_KEY,))
        with build_client(settings=settings) as client:
            response = systemone(client, request_body(NOUL))
            assert response.status_code == 401


class TestSM1OversizedIntegerLiteral:
    @staticmethod
    def _body_with_literal(literal: str) -> bytes:
        return (
            f'{{"model":"{MODEL}","state":{{"n":{literal}}},'
            '"questions":{"q":{"type":"noul"}}}'
        ).encode()

    def test_an_integer_above_the_digit_limit_is_a_400(self, client: TestClient) -> None:
        # CPython refuses int() on literals longer than 4300 digits with a bare
        # ValueError; without handling that is a 500 for a body the caller controls.
        response = systemone(client, content=self._body_with_literal("9" * 4301))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_json"

    def test_an_integer_just_inside_the_digit_limit_still_parses(
        self, client: TestClient
    ) -> None:
        # Not a 400: the literal is decoded, and the request then fails further down
        # the pipeline on the token budget, which is a different gate entirely.
        response = systemone(client, content=self._body_with_literal("9" * 4300))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "context_length_exceeded"

    def test_the_error_body_does_not_echo_the_literal(self, client: TestClient) -> None:
        response = systemone(client, content=self._body_with_literal("9" * 4301))
        assert "9999" not in response.text


class TestSM3DefaultDeny:
    """Authentication is a gate in front of routing, not a call each handler remembers."""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/v1/models"),
            ("GET", "/jevbert/v1/capabilities"),
            ("POST", "/v1/systemone"),
            ("GET", "/nope"),
            ("POST", "/jevbert/v1/anything"),
            ("GET", "/v1/systemone"),
            ("DELETE", "/v1/models"),
        ],
    )
    def test_every_path_needs_credentials(
        self, client: TestClient, method: str, path: str
    ) -> None:
        response = client.request(method, path)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_401_precedes_404(self, client: TestClient) -> None:
        # An unauthenticated caller must not learn which paths exist.
        assert client.get("/nope").status_code == 401
        assert client.get("/nope", headers=AUTH).status_code == 404

    def test_401_precedes_405(self, client: TestClient) -> None:
        assert client.get("/v1/systemone").status_code == 401
        assert client.get("/v1/systemone", headers=AUTH).status_code == 405

    @pytest.mark.parametrize("path", ["/healthz", "/readyz"])
    def test_the_health_allow_list_stays_open(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code in (200, 503)

    def test_the_allow_list_is_exact(self, client: TestClient) -> None:
        # A path that merely starts with an allowed one is not allowed.
        assert client.get("/healthz/../v1/models").status_code == 401
        assert client.get("/healthzz").status_code == 401


class TestSL1DuplicateKeyMessage:
    def test_the_duplicate_key_is_not_reflected(self, client: TestClient) -> None:
        raw = (
            b'{"model":"m","state":{"CANARY-DUPLICATE-KEY":1,"CANARY-DUPLICATE-KEY":2},'
            b'"questions":{"q":{"type":"noul"}}}'
        )
        response = systemone(client, content=raw)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_json"
        assert "CANARY-DUPLICATE-KEY" not in response.text


class TestSL3UnknownFieldNameIsNotReflected:
    """S-L3: the 422 said which field, by quoting the caller's own key.

    ``path`` and ``detail[].loc`` already point at it, and they are structure rather
    than prose - a log line or an error tracker that keeps the message keeps the key
    (spec 15.2). The location stays; the quotation goes.
    """

    def test_a_question_s_unknown_field_name_stays_out_of_the_message(self) -> None:
        with build_client(settings=build_settings()) as client:
            response = systemone(
                client, request_body({"q": {"type": "noul", "CANARY-FIELD": 2}})
            )
        assert response.status_code == 422
        payload = response.json()
        assert "CANARY-FIELD" not in payload["error"]["message"]
        assert "CANARY-FIELD" not in payload["detail"][0]["msg"]
        # The caller can still find it: the path and the loc say exactly where.
        assert payload["error"]["path"] == ["questions", "q", "CANARY-FIELD"]
        assert payload["detail"][0]["loc"] == ["body", "questions", "q", "noul", "CANARY-FIELD"]

    def test_a_top_level_unknown_field_name_stays_out_of_the_message(self) -> None:
        settings = build_settings(compat=CompatSettings(unknown_top_level_fields="reject"))
        body = request_body(NOUL)
        body["CANARY-FIELD"] = 1
        with build_client(settings=settings) as client:
            response = systemone(client, body)
        assert response.status_code == 422
        payload = response.json()
        assert "CANARY-FIELD" not in payload["error"]["message"]
        assert "CANARY-FIELD" not in payload["detail"][0]["msg"]
        assert payload["error"]["path"] == ["CANARY-FIELD"]


class TestSL2ReadyzDisclosure:
    def test_the_not_ready_body_names_nothing(self) -> None:
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            response = client.get("/readyz")
            assert response.status_code == 503
            assert response.json() == {"status": "not_ready"}

    def test_the_ready_body_names_nothing(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_the_detail_is_available_to_an_authenticated_caller(self) -> None:
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
            assert payload["bundles"][0]["state"] == "loading"
            assert payload["bundles"][0]["id"] == MODEL


class TestCredentialsNeverReachTheWire:
    def test_no_response_body_contains_a_configured_key(self, client: TestClient) -> None:
        for response in (
            client.get("/jevbert/v1/capabilities", headers=AUTH),
            client.get("/v1/models", headers=AUTH),
            systemone(client, request_body(NOUL)),
            client.get("/readyz"),
        ):
            assert API_KEY not in response.text


def test_settings_used_here_are_the_shipped_ones() -> None:
    # Guards the fixtures above: a limit or a serving default that drifts from the
    # configuration files would make every assertion in this module vacuous.
    settings = build_settings()
    assert settings.limits == Limits(max_request_tokens=131_072)
    assert settings.serving == ServingSettings()
