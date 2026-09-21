"""CT10/CT12 and the cross-cutting HTTP contract (spec 5.1, 5.7, 5.9; POC_DESIGN 4)."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from jevbert import CONFIDENCE_DEFINITION, CONTRACT_PROFILE
from jevbert.api.app import create_app
from jevbert.backends.base import (
    WARMUP_PREMISE,
    CancelToken,
    EncodedSequence,
    InferenceCancelled,
    TextPair,
)
from jevbert.backends.fake import FakeBackend
from jevbert.compiler.serializer_nli import (
    DEFAULT_TEMPLATE,
    NLI_TEMPLATE_ID,
    SERIALIZER_VERSION_FULL,
)
from jevbert.config import ConfigurationError, ServingSettings, Settings
from jevbert.inference.registry import build_registry as registry_from_settings
from tests.conftest import (
    API_KEY,
    AUTH,
    FAKE_MANIFEST,
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

    def test_the_published_limits_are_the_ones_in_force(self, client: TestClient) -> None:
        limits = client.get("/jevbert/v1/capabilities", headers=AUTH).json()["limits"]
        configured = build_settings().limits
        assert limits["max_body_bytes"] == configured.max_body_bytes == 2_097_152
        assert limits["max_json_depth"] == configured.max_json_depth == 32
        assert limits["max_questions"] == configured.max_questions == 256
        assert limits["min_choice_options"] == 2
        assert limits["max_choice_options"] == 255
        assert limits["min_score_levels"] == 2
        assert limits["max_score_levels"] == 10
        assert limits["overflow_policy"] == "reject"
        assert limits["max_request_chars"] == configured.effective_max_request_chars
        assert limits["request_deadline_seconds"] == 30.0
        assert limits["max_pending_requests"] == 8

    def test_the_bundle_token_limits_come_from_the_manifest(
        self, client: TestClient
    ) -> None:
        # The per-bundle values are the ones the request path enforces, and POC_DESIGN
        # 5.5 requires them to be published rather than assumed from spec 4.2.
        bundle = client.get("/jevbert/v1/capabilities", headers=AUTH).json()["bundles"][0]
        assert bundle["limits"] == {
            "max_sequence_tokens": 2048,
            "max_request_tokens": 131_072,
        }

    def test_aliases_are_empty_unless_enabled(self, client: TestClient) -> None:
        assert client.get("/jevbert/v1/capabilities", headers=AUTH).json()["aliases"] == {}

    def test_enabled_aliases_are_published(self) -> None:
        settings = build_settings(
            serving=ServingSettings(allow_jev_aliases=True, aliases={"jev-latest": MODEL})
        )
        registry = registry_from_settings(settings)
        with build_client(settings=settings, registry=registry) as client:
            payload = client.get("/jevbert/v1/capabilities", headers=AUTH).json()
        assert payload["aliases"] == {"jev-latest": MODEL}


class TestFakeInjectionIsUnreachable:
    """POC_DESIGN 6.2: ``logit_fn`` and the delays are constructor arguments only.

    They exist so that ties, extreme values, NaN and count mismatches can be produced
    (CT07, CT08). Nothing a caller or an operator writes may select them: a fake that
    could be steered from a manifest or a request body would be a way to make the
    server answer anything at all.
    """

    def test_a_manifest_cannot_carry_them(self, tmp_path: Path) -> None:
        raw = json.loads(FAKE_MANIFEST.read_text(encoding="utf-8"))
        for field in ("logit_fn", "delay_seconds", "load_delay_seconds"):
            path = tmp_path / "jevbert-fake-0.0.0.json"
            path.write_text(json.dumps(raw | {field: 5}), encoding="utf-8")
            with pytest.raises(ConfigurationError):
                registry_from_settings(
                    build_settings(manifests_dir=tmp_path, enable_fake_bundle=True)
                )

    def test_a_configuration_cannot_carry_them(self) -> None:
        for field in ("logit_fn", "delay_seconds"):
            with pytest.raises(ValidationError):
                ServingSettings(**{field: 5})
            with pytest.raises(ValidationError):
                Settings(api_keys=(API_KEY,), **{field: 5})

    def test_a_request_body_cannot_carry_them(self, client: TestClient) -> None:
        # An unknown top level field is ignored rather than refused now (ADR-016), so
        # what matters is that it changes nothing: the answer is the one the same body
        # produces without it.
        body = request_body(NOUL) | {"logit_fn": "anything"}
        response = systemone(client, body)
        assert response.status_code == 200
        assert response.json() == systemone(client, request_body(NOUL)).json()

        nested = request_body({"q": {"type": "noul", "delay_seconds": 5}})
        assert systemone(client, nested).status_code == 422

    def test_the_built_fake_backend_has_neither(self) -> None:
        registry = registry_from_settings(build_settings())
        backend = registry.bundles[0].backend
        assert isinstance(backend, FakeBackend)
        # Read through the public behaviour rather than the private attributes: an
        # injected logit_fn would change the answer, a delay would change the timing.
        pairs = [TextPair("p", "h1"), TextPair("p", "h2")]
        logits = backend.score(backend.count_and_encode(pairs), CancelToken())
        assert logits == FakeBackend().score(
            FakeBackend().count_and_encode(pairs), CancelToken()
        )
        assert logits[0] != logits[1]


class TestValidationOrder:
    """POC_DESIGN 3 / D3: the gates are ordered, and the order is observable."""

    def test_media_type_is_checked_before_size(self, client: TestClient) -> None:
        # Both are wrong; 415 is the one the caller is told about, because the body is
        # never interpreted as JSON at all.
        response = client.post(
            "/v1/systemone",
            content=b"x" * (3 * 1024 * 1024),
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "text/plain"},
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_media_type"

    def test_size_is_checked_before_parsing(self, client: TestClient) -> None:
        response = systemone(client, content=b"{" + b"x" * (3 * 1024 * 1024))
        assert response.status_code == 413

    def test_structure_is_checked_before_the_model_is_resolved(
        self, client: TestClient
    ) -> None:
        body = request_body({"q": {"type": "boolean"}})
        body["model"] = "not-a-registered-bundle"
        payload = systemone(client, body).json()
        assert payload["error"]["code"] == "validation_error"

    def test_an_unknown_model_is_refused_before_readiness(self) -> None:
        # D3: readiness is a property of a bundle, so a name that resolves to no
        # bundle is 422, never 503, whether or not anything has loaded.
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            body = request_body(NOUL)
            body["model"] = "jev-1.13.0"
            response = systemone(client, body)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "model_not_found"
            # The same request against a registered bundle is the 503 case.
            assert systemone(client, request_body(NOUL)).status_code == 503


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

    def test_model_unavailable_is_marked_retryable(self) -> None:
        registry = build_registry(load=False)
        with build_client(registry=registry) as client:
            payload = systemone(client, request_body(NOUL)).json()
        assert payload["error"]["code"] == "model_unavailable"
        assert payload["error"]["retryable"] is True

    def test_the_production_startup_path_is_unready_and_then_ready(self) -> None:
        # Every other readiness test drives registry.load_all() by hand. This one uses
        # the path the server actually takes: create_app(load_on_startup=True) with a
        # backend that takes a moment, so the 503 window is real rather than staged.
        registry = build_registry(FakeBackend(load_delay_seconds=0.3), load=False)
        settings = build_settings()
        app = create_app(settings, registry, load_on_startup=True)
        with TestClient(app) as client:
            assert client.get("/readyz").status_code == 503
            response = systemone(client, request_body(NOUL))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_unavailable"

            deadline = time.monotonic() + 10.0
            while client.get("/readyz").status_code != 200:
                assert time.monotonic() < deadline, "the bundle never became ready"
                time.sleep(0.02)
            assert systemone(client, request_body(NOUL)).status_code == 200

    def test_warmup_runs_on_the_engine_worker(self) -> None:
        # A-F10: a warmup forward pass on the loading thread would touch the device
        # beside whatever the worker is serving.
        threads: list[str] = []

        class Observing(FakeBackend):
            def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
                if all(s.data.premise == WARMUP_PREMISE for s in sequences):
                    threads.append(threading.current_thread().name)
                return super().score(sequences, cancel)

        registry = build_registry(Observing(), load=False)
        app = create_app(build_settings(), registry, load_on_startup=True)
        with TestClient(app) as client:
            deadline = time.monotonic() + 10.0
            while client.get("/readyz").status_code != 200:
                assert time.monotonic() < deadline, "the bundle never became ready"
                time.sleep(0.02)

        assert threads and all(name.startswith("jevbert-inference") for name in threads)


class GatedBackend(FakeBackend):
    """Blocks inside ``score`` until the test lets it go.

    The 529 tests used to race two requests against a one-second sleep and hope the
    scheduler cooperated. A rendezvous makes "the worker is busy" a fact rather than a
    guess, and takes the sleep out of the suite.
    """

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self._release = threading.Event()

    def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
        if any(sequence.data.premise != WARMUP_PREMISE for sequence in sequences):
            self.started.set()
            assert self._release.wait(10.0), "the gated backend was never released"
        return super().score(sequences, cancel)

    def release(self) -> None:
        self._release.set()


class TestOverloadAndDeadline:
    def test_queue_full_returns_529(self) -> None:
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        backend = GatedBackend()
        with build_client(settings=settings, registry=build_registry(backend)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                busy = pool.submit(systemone, client, request_body(NOUL))
                assert backend.started.wait(10.0)
                refused = systemone(client, request_body(NOUL))
                backend.release()
                assert busy.result().status_code == 200
        assert refused.status_code == 529

    def test_529_carries_retry_after_and_is_marked_retryable(self) -> None:
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        backend = GatedBackend()
        with build_client(settings=settings, registry=build_registry(backend)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                busy = pool.submit(systemone, client, request_body(NOUL))
                assert backend.started.wait(10.0)
                refused = systemone(client, request_body(NOUL))
                backend.release()
                busy.result()

        payload = refused.json()
        assert refused.status_code == 529
        assert payload["error"]["code"] == "overloaded"
        assert payload["error"]["retryable"] is True
        assert refused.headers["Retry-After"] == "1"

    def test_capacity_returns_once_the_worker_finishes(self) -> None:
        settings = build_settings(serving=ServingSettings(max_pending_requests=1))
        backend = GatedBackend()
        with build_client(settings=settings, registry=build_registry(backend)) as client:
            with ThreadPoolExecutor(max_workers=1) as pool:
                busy = pool.submit(systemone, client, request_body(NOUL))
                assert backend.started.wait(10.0)
                assert systemone(client, request_body(NOUL)).status_code == 529
                backend.release()
                assert busy.result().status_code == 200
            # A 529 is a statement about right now, not a latch.
            assert systemone(client, request_body(NOUL)).status_code == 200

    def test_deadline_returns_504(self) -> None:
        settings = build_settings(serving=ServingSettings(request_deadline_seconds=0.05))
        backend = GatedBackend()
        try:
            with build_client(settings=settings, registry=build_registry(backend)) as client:
                response = systemone(client, request_body(NOUL))
                # Still inside the backend: the 504 is produced while the work is
                # genuinely in flight, not after it quietly finished.
                assert backend.started.is_set()
                assert response.status_code == 504
                payload = response.json()
                assert payload["error"]["code"] == "deadline_exceeded"
                assert payload["error"]["retryable"] is True
        finally:
            backend.release()

    def test_the_cancel_token_reaches_a_backend_that_watches_it(self) -> None:
        # CT10: a request that gave up must stop the work, not just stop waiting.
        observed = threading.Event()

        class Watchful(FakeBackend):
            def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
                if any(s.data.premise != WARMUP_PREMISE for s in sequences):
                    for _ in range(1000):
                        if cancel.cancelled:
                            observed.set()
                            raise InferenceCancelled()
                        time.sleep(0.01)
                return super().score(sequences, cancel)

        settings = build_settings(serving=ServingSettings(request_deadline_seconds=0.05))
        with build_client(settings=settings, registry=build_registry(Watchful())) as client:
            assert systemone(client, request_body(NOUL)).status_code == 504
            assert observed.wait(10.0)


class TestErrorContract:
    def test_unknown_path_is_a_jevbert_error(self, client: TestClient) -> None:
        # Credentials first: an unauthenticated caller gets a 401 and learns nothing
        # about which paths exist (S-M3, tests/contract/test_security.py).
        response = client.get("/nope", headers=AUTH)
        assert response.status_code == 404
        payload = response.json()
        assert payload["error"]["code"] == "not_found"
        assert "detail" not in payload  # no FastAPI default body

    def test_wrong_method_is_a_jevbert_error(self, client: TestClient) -> None:
        response = client.get("/v1/systemone", headers=AUTH)
        assert response.status_code == 405
        payload = response.json()
        assert payload["error"]["code"] == "method_not_allowed"
        assert "detail" not in payload
        # A-F12: a 405 must say what the path does accept.
        assert response.headers["Allow"] == "POST"

    def test_error_body_shape(self, client: TestClient) -> None:
        payload = systemone(client, request_body({})).json()
        # spec 5.9: a 422 also carries Jev's own ``detail`` shape (ADR-016).
        assert set(payload) == {"error", "detail", "request_id"}
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

    @pytest.mark.parametrize(
        "header", ["X-JevBERT-Bundle", "X-JevBERT-Usage", "X-JevBERT-Calibration"]
    )
    def test_bundle_headers_are_omitted_before_a_bundle_is_known(
        self, client: TestClient, header: str
    ) -> None:
        # POC_DESIGN 4.4: all three describe a bundle, so none of them may carry a
        # default when the request failed before one was resolved.
        response = systemone(client, request_body({}))
        assert response.status_code == 422
        assert header not in response.headers

    def test_the_contract_headers_are_present_even_then(self, client: TestClient) -> None:
        response = systemone(client, request_body({}))
        assert response.headers["X-JevBERT-Contract"] == CONTRACT_PROFILE
        assert response.headers["x-typesafe-request-id"]

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


class TestCapabilitiesNamesTheTemplate:
    """spec 5.7: the serializer a bundle publishes includes its template (K4)."""

    def test_the_template_is_published_not_null(self, client: TestClient) -> None:
        bundle = client.get("/jevbert/v1/capabilities", headers=AUTH).json()["bundles"][0]
        assert bundle["serializer"] == SERIALIZER_VERSION_FULL
        assert bundle["template"] == NLI_TEMPLATE_ID

    def test_the_published_template_is_the_one_that_compiles(
        self, client: TestClient
    ) -> None:
        # A published template ID that is not the one in force would be a claim about
        # the model input that the compiled input does not honour.
        bundle = client.get("/jevbert/v1/capabilities", headers=AUTH).json()["bundles"][0]
        assert bundle["template"] == DEFAULT_TEMPLATE.template_id
