"""Shared fixtures: a server wired to the fake backend, with no GPU and no weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevbert.api.app import create_app
from jevbert.api.encoding import dumps
from jevbert.backends.base import Backend
from jevbert.backends.fake import FakeBackend
from jevbert.config import Limits, ServingSettings, Settings
from jevbert.inference.registry import Bundle, ModelRegistry, read_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
MODELS_DIR = PROJECT_ROOT / "models"
FAKE_MANIFEST = MANIFESTS_DIR / "jevbert-fake-0.0.0.json"

#: At least ``MIN_API_KEY_LENGTH`` characters, like every key the server accepts (S-L4).
API_KEY = "test-api-key-0123456789-abcdefghij"
MODEL = "jevbert-fake-0.0.0"

AUTH: dict[str, str] = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def build_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "api_keys": (API_KEY,),
        "manifests_dir": MANIFESTS_DIR,
        "models_dir": MODELS_DIR,
        "enable_fake_bundle": True,
        # Matches configs/jevbert.test.yaml. The fake manifest declares 131,072
        # request tokens and a manifest may never widen the server limits (A-F6).
        "limits": Limits(max_request_tokens=131_072),
        "serving": ServingSettings(),
    }
    values.update(overrides)
    return Settings(**values)


def build_registry(
    backend: Backend | None = None,
    *,
    load: bool = True,
    aliases: Mapping[str, str] | None = None,
) -> ModelRegistry:
    manifest, digest = read_manifest(FAKE_MANIFEST)
    bundle = Bundle(manifest, digest, backend or FakeBackend())
    registry = ModelRegistry([bundle], aliases)
    if load:
        registry.load_all()
    return registry


def build_client(
    *,
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    load_on_startup: bool = False,
) -> TestClient:
    settings = settings or build_settings()
    registry = registry or build_registry()
    app = create_app(settings, registry, load_on_startup=load_on_startup)
    return TestClient(app)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with build_client() as test_client:
        yield test_client


def systemone(
    client: TestClient, body: Any = None, *, content: bytes | None = None, **kwargs: Any
) -> Any:
    """POST /v1/systemone with the default credentials unless overridden."""
    headers = dict(AUTH)
    headers.update(kwargs.pop("headers", {}) or {})
    if content is None:
        content = dumps(body).encode("utf-8")
    return client.post("/v1/systemone", content=content, headers=headers, **kwargs)


def request_body(questions: dict[str, Any], state: Any = "顧客からの問い合わせ") -> dict[str, Any]:
    return {"model": MODEL, "state": state, "questions": questions}
