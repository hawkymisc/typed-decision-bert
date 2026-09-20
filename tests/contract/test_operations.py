"""CT10/CT12 and the cross-cutting HTTP contract (spec 5.1, 5.7, 5.9; POC_DESIGN 4)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevbert import CONFIDENCE_DEFINITION, CONTRACT_PROFILE
from jevbert.backends.base import CancelToken, EncodedSequence, TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.config import ServingSettings
from jevbert.inference.registry import build_registry as registry_from_settings
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


class BrokenBackend(FakeBackend):
    """A backend whose weights never load (spec 16.1: stay 503, do not crash)."""

    def load(self) -> None:
        raise RuntimeError("weights are missing")


class TestAuthentication:
    def test_valid_key_is_accepted(self, client: TestClient) -> None:
        assert systemone(client, request_body(NOUL)).status_code == 200

    @pytest.mark.parametrize(
        "header",
        [None, "", "Bearer", "Bearer ", "Bearer wrong-key", "Basic " + API_KEY, API_KEY],
    )
    def test_missing_or_wrong_credentials(self, client: TestClient, header: str | None) -> None:
        headers = {"Content-Type": "application/json"}
        if header is not None:
            headers["Authorization"] = header
        response = client.post(
            "/v1/systemone",
            content=b'{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}',
            headers=headers,
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_credentials_are_never_echoed(self, client: TestClient) -> None:
        response = client.post(
            "/v1/systemone",
            content=b"{}",
            headers={"Authorization": "Bearer super-secret", "Content-Type": "application/json"},
        )
        assert "super-secret" not in response.text

    def test_authentication_precedes_body_interpretation(self, client: TestClient) -> None:
        # POC_DESIGN 3: never interpret the body before authenticating.
        response = client.post(
            "/v1/systemone",
            content=b"not json at all",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 401

    def test_health_endpoints_need_no_credentials(self, client: TestClient) -> None:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code in (200, 503)

    @pytest.mark.parametrize("path", ["/v1/models", "/jevbert/v1/capabilities"])
    def test_other_endpoints_need_credentials(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 401


class TestMediaType:
    @pytest.mark.parametrize(
        "content_type", ["text/plain", "application/xml", "", "application/json-patch+json"]
    )
    def test_unsupported_content_type(self, client: TestClient, content_type: str) -> None:
        response = systemone(
            client,
            request_body(NOUL),
            headers={"Content-Type": content_type},
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_media_type"

    @pytest.mark.parametrize(
        "content_type", ["application/json", "application/json; charset=utf-8", "APPLICATION/JSON"]
    )
    def test_accepted_content_types(self, client: TestClient, content_type: str) -> None:
        response = systemone(client, request_body(NOUL), headers={"Content-Type": content_type})
        assert response.status_code == 200

    def test_foreign_charset_is_refused(self, client: TestClient) -> None:
        response = systemone(
            client,
            request_body(NOUL),
            headers={"Content-Type": "application/json; charset=shift_jis"},
        )
        assert response.status_code == 415

    def test_content_encoding_is_refused(self, client: TestClient) -> None:
        response = systemone(
            client, request_body(NOUL), headers={"Content-Encoding": "gzip"}
        )
        assert response.status_code == 415


class TestModelResolution:
    def test_unknown_model(self, client: TestClient) -> None:
        body = request_body(NOUL)
        body["model"] = "jev-1.13.0"
        response = systemone(client, body)
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "model_not_found"
        assert payload["error"]["path"] == ["model"]

    def test_alias_resolves_when_enabled(self) -> None:
        settings = build_settings(
            serving=ServingSettings(allow_jev_aliases=True, aliases={"jev-latest": MODEL})
        )
        registry = registry_from_settings(settings)
        registry.load_all()
        with build_client(settings=settings, registry=registry) as client:
            body = request_body(NOUL)
            body["model"] = "jev-latest"
            response = systemone(client, body)
            assert response.status_code == 200
            # spec 2.3: the response names the real bundle, never the alias.
            assert response.json()["model"] == MODEL

    def test_alias_is_rejected_when_disabled(self) -> None:
        settings = build_settings(serving=ServingSettings(allow_jev_aliases=False))
        registry = registry_from_settings(settings)
        registry.load_all()
        with build_client(settings=settings, registry=registry) as client:
            body = request_body(NOUL)
            body["model"] = "jev-latest"
            assert systemone(client, body).status_code == 422

    def test_unregistered_jev_names_are_not_wildcarded(self, client: TestClient) -> None:
        for name in ("jev-latest", "jev-2.0.0", "jev-anything"):
            body = request_body(NOUL)
            body["model"] = name
            assert systemone(client, body).status_code == 422


class TestModelsEndpoint:
    def test_shape(self, client: TestClient) -> None:
        payload = client.get("/v1/models", headers=AUTH).json()
        assert list(payload) == ["models"]
        entry = payload["models"][0]
        assert set(entry) == {"name", "description", "release_date"}
        assert entry["name"] == MODEL
        assert entry["release_date"] == "2026-09-21"

    def test_enabled_aliases_are_listed(self) -> None:
        settings = build_settings(
            serving=ServingSettings(allow_jev_aliases=True, aliases={"jev-latest": MODEL})
        )
        registry = registry_from_settings(settings)
        with build_client(settings=settings, registry=registry) as client:
            names = [m["name"] for m in client.get("/v1/models", headers=AUTH).json()["models"]]
        assert names == [MODEL, "jev-latest"]


class TestCapabilities:
    def test_shape(self, client: TestClient) -> None:
        payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
        assert payload["contract"] == CONTRACT_PROFILE
        assert payload["stage"] == "P0.5-poc"
        bundle = payload["bundles"][0]
        assert bundle["digest"].startswith("sha256:")
        assert bundle["calibration"]["state"] == "uncalibrated"
        assert bundle["usage"] == "expanded-input-a0-v1"
        assert bundle["confidence"] == CONFIDENCE_DEFINITION

    def test_unevaluated_ranges_are_not_presented_as_validated(
        self, client: TestClient
    ) -> None:
        payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
        validated = payload["bundles"][0]["validated"]
        assert validated["languages"] == []
        assert validated["domains"] == []
        assert validated["quality"] == "unevaluated"

    def test_known_differences_are_published(self, client: TestClient) -> None:
        payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
        assert any("confidence" in item for item in payload["known_differences"])
        assert any("usage" in item for item in payload["known_differences"])


class TestReadiness:
    def test_not_ready_until_bundles_load(self) -> None:
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            assert client.get("/readyz").status_code == 503
            assert client.get("/healthz").status_code == 200  # liveness is separate
            response = systemone(client, request_body(NOUL))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_unavailable"
            assert response.headers["Retry-After"] == "1"

            registry.load_all()
            assert client.get("/readyz").status_code == 200
            assert systemone(client, request_body(NOUL)).status_code == 200

    def test_a_failed_bundle_keeps_the_server_alive_and_not_ready(self) -> None:
        registry = build_registry(BrokenBackend())
        with build_client(registry=registry) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 503
            assert systemone(client, request_body(NOUL)).status_code == 503

    def test_readyz_does_not_leak_model_internals(self) -> None:
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            assert "sha256:" not in client.get("/readyz").text
            assert client.get("/healthz").json() == {"status": "ok"}


class TestOverloadAndDeadline:
    def test_queue_full_returns_529(self) -> None:
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        registry = build_registry(FakeBackend(delay_seconds=1.0))
        with build_client(settings=settings, registry=registry) as client:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(systemone, client, request_body(NOUL)) for _ in range(2)
                ]
                statuses = sorted(future.result().status_code for future in futures)
        assert statuses == [200, 529]

    def test_529_carries_retry_after_and_is_marked_retryable(self) -> None:
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        registry = build_registry(FakeBackend(delay_seconds=1.0))
        with build_client(settings=settings, registry=registry) as client:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(systemone, client, request_body(NOUL)) for _ in range(2)
                ]
                responses = [future.result() for future in futures]
        overloaded = [r for r in responses if r.status_code == 529]
        assert len(overloaded) == 1
        payload = overloaded[0].json()
        assert payload["error"]["code"] == "overloaded"
        assert payload["error"]["retryable"] is True
        assert overloaded[0].headers["Retry-After"] == "1"

    def test_deadline_returns_504(self) -> None:
        settings = build_settings(serving=ServingSettings(request_deadline_seconds=0.05))
        registry = build_registry(FakeBackend(delay_seconds=5.0))
        with build_client(settings=settings, registry=registry) as client:
            response = systemone(client, request_body(NOUL))
        assert response.status_code == 504
        payload = response.json()
        assert payload["error"]["code"] == "deadline_exceeded"
        assert payload["error"]["retryable"] is True


class TestErrorContract:
    def test_unknown_path_is_a_jevbert_error(self, client: TestClient) -> None:
        response = client.get("/nope")
        assert response.status_code == 404
        payload = response.json()
        assert payload["error"]["code"] == "not_found"
        assert "detail" not in payload  # no FastAPI default body

    def test_wrong_method_is_a_jevbert_error(self, client: TestClient) -> None:
        response = client.get("/v1/systemone")
        assert response.status_code == 405
        payload = response.json()
        assert payload["error"]["code"] == "method_not_allowed"
        assert "detail" not in payload

    def test_error_body_shape(self, client: TestClient) -> None:
        payload = systemone(client, request_body({})).json()
        assert set(payload) == {"error", "request_id"}
        assert set(payload["error"]) <= {"code", "message", "path", "retryable"}
        assert payload["error"]["retryable"] is False
        assert payload["request_id"]

    def test_request_id_in_body_matches_the_header(self, client: TestClient) -> None:
        response = systemone(client, request_body({}))
        assert response.json()["request_id"] == response.headers["x-typesafe-request-id"]

    def test_unhandled_errors_become_internal_error(self) -> None:
        class Exploding(FakeBackend):
            """Sound at warmup, broken afterwards: readiness must not mask this."""

            def count_and_encode(self, pairs: Any) -> list[EncodedSequence]:
                if any(pair.premise != "warmup" for pair in pairs):
                    raise KeyError("unexpected internal failure")
                return super().count_and_encode(pairs)

        registry = build_registry(Exploding())
        with build_client(registry=registry) as client:
            response = systemone(client, request_body(NOUL))
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"]["code"] == "internal_error"
        # The internal failure text must not reach the caller.
        assert "unexpected internal failure" not in response.text
        assert "Traceback" not in response.text


class TestHeaders:
    def test_success_headers(self, client: TestClient) -> None:
        response = systemone(client, request_body(NOUL))
        headers = response.headers
        assert headers["X-Request-ID"] == headers["x-typesafe-request-id"]
        assert headers["X-JevBERT-Contract"] == CONTRACT_PROFILE
        assert headers["X-JevBERT-Confidence"] == CONFIDENCE_DEFINITION
        assert headers["X-JevBERT-Bundle"].startswith("sha256:")
        assert headers["X-JevBERT-Usage"] == "expanded-input-a0-v1"
        assert headers["X-JevBERT-Calibration"] == "uncalibrated"

    def test_bundle_headers_are_omitted_before_a_bundle_is_known(
        self, client: TestClient
    ) -> None:
        response = systemone(client, request_body({}))
        assert response.status_code == 422
        assert "X-JevBERT-Bundle" not in response.headers
        assert response.headers["X-JevBERT-Contract"] == CONTRACT_PROFILE

    @pytest.mark.parametrize("path", ["/healthz", "/readyz", "/nope"])
    def test_every_response_carries_a_request_id(self, client: TestClient, path: str) -> None:
        response = client.get(path)
        assert response.headers["x-typesafe-request-id"]
        assert response.headers["X-Request-ID"] == response.headers["x-typesafe-request-id"]

    def test_request_ids_are_server_generated_and_unique(self, client: TestClient) -> None:
        first = client.get("/healthz", headers={"X-Request-ID": "client-supplied"})
        second = client.get("/healthz")
        assert first.headers["X-Request-ID"] != "client-supplied"
        assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


class TestCT12RequestIsolation:
    def test_concurrent_requests_do_not_mix_answers(self) -> None:
        registry = build_registry(FakeBackend(delay_seconds=0.02))
        settings = build_settings(serving=ServingSettings(max_pending_requests=16))
        bodies = [
            request_body({f"q{i}": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}, f"s{i}")
            for i in range(8)
        ]
        with build_client(settings=settings, registry=registry) as client:
            with ThreadPoolExecutor(max_workers=8) as pool:
                concurrent = [r.json() for r in pool.map(lambda b: systemone(client, b), bodies)]
            sequential = [systemone(client, body).json() for body in bodies]

        for index, (parallel, serial) in enumerate(zip(concurrent, sequential, strict=True)):
            assert set(parallel["answers"]) == {f"q{index}"}
            assert parallel == serial


class TestBackendContractGuard:
    def test_a_backend_returning_too_few_logits_is_a_500(self) -> None:
        class ShortBackend(FakeBackend):
            def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
                pairs = [s.data for s in sequences]
                assert isinstance(pairs[0], TextPair)
                return [0.0] * (len(sequences) - 1) if len(sequences) > 2 else [0.0, 0.0]

        registry = build_registry(ShortBackend())
        with build_client(registry=registry) as client:
            response = systemone(
                client,
                request_body({"q": {"type": "score", "criteria": ["a", "b", "c"]}}),
            )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "inference_error"
