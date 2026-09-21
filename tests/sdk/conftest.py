"""Real uvicorn servers on ephemeral ports for the SDK tests (POC_DESIGN 8.3).

The SDK is exercised over a real socket, not an in-process transport, so that URL
joining, headers, status codes and connection handling are the real ones.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi import FastAPI

from jevbert.api.app import create_app
from jevbert.backends.base import Backend
from jevbert.config import CompatSettings, Settings
from jevbert.inference.registry import ModelRegistry
from tests.conftest import MODEL, build_registry, build_settings


@contextmanager
def running_server(app: FastAPI) -> Iterator[str]:
    """Serve ``app`` on a free port, yielding its base URL."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True, name="jevbert-test-server"
    )
    thread.start()

    deadline = time.monotonic() + 20.0
    while not server.started:
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError("the test server did not start in time")
        time.sleep(0.02)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20.0)
        listener.close()


def make_app(
    *,
    backend: Backend | None = None,
    load: bool = True,
    settings: Settings | None = None,
    aliases: dict[str, str] | None = None,
) -> FastAPI:
    settings = settings or build_settings()
    registry: ModelRegistry = build_registry(backend, load=load, aliases=aliases)
    return create_app(settings, registry, load_on_startup=False)


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    with running_server(make_app()) as url:
        yield url


@pytest.fixture
def unready_server_url() -> Iterator[str]:
    with running_server(make_app(load=False)) as url:
        yield url


@pytest.fixture(scope="module")
def strict_server_url() -> Iterator[str]:
    """A server configured to refuse unknown top level fields (ADR-016)."""
    settings = build_settings(compat=CompatSettings(unknown_top_level_fields="reject"))
    with running_server(make_app(settings=settings)) as url:
        yield url


@pytest.fixture(scope="module")
def alias_server_url() -> Iterator[str]:
    """A server with the PoC's aliases on, so the SDK's default model resolves.

    This is the drop-in case of spec 18.3: the caller changes ``base_url`` and
    ``api_key`` and nothing else, leaving the SDK's own default ``jev-latest``.
    """
    settings = build_settings()
    settings = settings.model_copy(
        update={
            "serving": settings.serving.model_copy(
                update={
                    "allow_jev_aliases": True,
                    "aliases": {"jev-latest": MODEL, "jev-preview": MODEL},
                }
            )
        }
    )
    aliases = {"jev-latest": MODEL, "jev-preview": MODEL}
    with running_server(make_app(settings=settings, aliases=aliases)) as url:
        yield url
