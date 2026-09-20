"""Q-H2: the body size gate over a real socket (spec 5.9, 413; K7).

``_read_body`` has two halves: a check on the declared ``Content-Length`` and a running
total over the bytes actually received. Deleting the second half left every phase 1 test
green, because an in-process test client always declares a length. A caller does not
have to: chunked transfer encoding declares nothing at all, which is precisely the case
the running total exists for.

These tests speak HTTP/1.1 down a socket by hand. Nothing else gives that much control
over the framing.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from tests.conftest import API_KEY, MODEL, build_settings
from tests.sdk.conftest import make_app, running_server

#: Comfortably over the 2 MiB ceiling, and large enough to span many chunks.
OVERSIZED = 3 * 1024 * 1024
VALID_BODY = (
    f'{{"model":"{MODEL}","state":"s","questions":{{"q":{{"type":"noul"}}}}}}'
).encode()


@pytest.fixture
def socket_server_url() -> Iterator[str]:
    """One server per test.

    A deliberately malformed framing leaves unread bytes that the server reads as the
    start of a pipelined request, which is realistic and which the next test must not
    inherit.
    """
    with running_server(make_app(settings=build_settings())) as url:
        yield url


def address_of(url: str) -> tuple[str, int]:
    host, _, port = url.removeprefix("http://").partition(":")
    return host, int(port)


def send_raw(url: str, head: bytes, body_chunks: Iterator[bytes]) -> tuple[int, bytes]:
    """Send a hand-built request and read the status line and body back.

    The server may answer and close before the whole body has been sent, which is the
    correct behaviour for a 413 and would otherwise show up as a broken pipe.
    """
    host, port = address_of(url)
    with socket.create_connection((host, port), timeout=20.0) as connection:
        connection.sendall(head)
        try:
            for chunk in body_chunks:
                connection.sendall(chunk)
        except OSError:
            pass  # The refusal arrived before we finished; the response says so.

        connection.shutdown(socket.SHUT_WR)
        received = bytearray()
        while True:
            try:
                part = connection.recv(65536)
            except OSError:
                break
            if not part:
                break
            received.extend(part)

    raw = bytes(received)
    assert raw, "the server closed the connection without answering"
    status = int(raw.split(b" ", 2)[1])
    _, _, body = raw.partition(b"\r\n\r\n")
    return status, body


def chunked(payload: bytes, chunk_size: int = 65536) -> Iterator[bytes]:
    for start in range(0, len(payload), chunk_size):
        piece = payload[start : start + chunk_size]
        yield f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
    yield b"0\r\n\r\n"


def head(url: str, *extra: str) -> bytes:
    host, port = address_of(url)
    lines = [
        "POST /v1/systemone HTTP/1.1",
        f"Host: {host}:{port}",
        f"Authorization: Bearer {API_KEY}",
        "Content-Type: application/json",
        *extra,
        "",
        "",
    ]
    return "\r\n".join(lines).encode()


class TestChunkedBodies:
    def test_a_chunked_body_past_the_limit_is_refused(self, socket_server_url: str) -> None:
        # No Content-Length at all: only the running total can stop this.
        payload = b'{"model":"m","state":"' + b"x" * OVERSIZED + b'"}'
        status, body = send_raw(
            socket_server_url, head(socket_server_url, "Transfer-Encoding: chunked"),
            chunked(payload),
        )
        assert status == 413
        assert b"request_too_large" in body

    def test_a_chunked_body_inside_the_limit_is_served(self, socket_server_url: str) -> None:
        status, body = send_raw(
            socket_server_url,
            head(socket_server_url, "Transfer-Encoding: chunked"),
            chunked(VALID_BODY, chunk_size=8),
        )
        assert status == 200, body
        assert b'"answers"' in body

    def test_the_refusal_does_not_depend_on_the_chunk_size(
        self, socket_server_url: str
    ) -> None:
        payload = b'{"model":"m","state":"' + b"x" * OVERSIZED + b'"}'
        status, _ = send_raw(
            socket_server_url,
            head(socket_server_url, "Transfer-Encoding: chunked"),
            chunked(payload, chunk_size=1024),
        )
        assert status == 413


class TestDeclaredLengths:
    def test_an_honest_oversized_length_is_refused_without_reading_the_body(
        self, socket_server_url: str
    ) -> None:
        status, body = send_raw(
            socket_server_url,
            head(socket_server_url, f"Content-Length: {OVERSIZED}"),
            iter([b"x" * 4096]),
        )
        assert status == 413
        assert b"request_too_large" in body

    def test_an_understated_length_cannot_smuggle_the_rest_of_the_body(
        self, socket_server_url: str
    ) -> None:
        """A short ``Content-Length`` in front of a huge body.

        h11 delivers exactly the declared number of bytes and treats the remainder as
        the start of the next request on the connection, so the handler never sees the
        3 MiB and the response is decided by the 100 bytes it did see. What matters is
        that the excess is neither served nor accumulated: the answer must not be a
        200, and the body must not be interpreted past the declaration.
        """
        declared = 100
        payload = b'{"model":"m","state":"' + b"x" * OVERSIZED + b'"}'
        status, body = send_raw(
            socket_server_url,
            head(socket_server_url, f"Content-Length: {declared}"),
            iter([payload]),
        )
        assert status in (400, 413), body
        assert b'"answers"' not in body

    def test_a_length_exactly_at_the_limit_passes_the_size_gate(
        self, socket_server_url: str
    ) -> None:
        # 2 MiB is not too large; the token budget is the next gate, and it is a 422.
        limit = build_settings().limits.max_body_bytes
        filler = limit - len(VALID_BODY) + len(b'"s"') - len(b'""')
        payload = (
            f'{{"model":"{MODEL}","state":"'.encode()
            + b"x" * filler
            + b'","questions":{"q":{"type":"noul"}}}'
        )
        assert len(payload) == limit
        status, body = send_raw(
            socket_server_url,
            head(socket_server_url, f"Content-Length: {len(payload)}"),
            iter([payload]),
        )
        assert status == 422, body
        assert b"context_length_exceeded" in body

    def test_one_byte_past_the_limit_is_refused(self, socket_server_url: str) -> None:
        limit = build_settings().limits.max_body_bytes
        status, _ = send_raw(
            socket_server_url,
            head(socket_server_url, f"Content-Length: {limit + 1}"),
            iter([b"x" * 4096]),
        )
        assert status == 413
