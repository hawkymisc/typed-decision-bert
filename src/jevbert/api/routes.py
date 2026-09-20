"""The five endpoints (spec 5.1; POC_DESIGN 4.4).

The validation order of POC_DESIGN 3 is fixed here and must not be reordered:
401 -> 415 -> 413 -> 400 -> 422 (structure, depth) -> 422 (model) -> 503 ->
422 (context length) -> inference. The body is never interpreted before authentication.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response

from jevbert import CONFIDENCE_DEFINITION
from jevbert.api.encoding import json_response
from jevbert.api.errors import (
    DeadlineExceededError,
    InferenceError,
    ModelUnavailableError,
    RequestTooLargeError,
    UnsupportedMediaTypeError,
)
from jevbert.backends.base import CancelToken, EncodedSequence, InferenceCancelled
from jevbert.compiler.compiled import EncodedRequest
from jevbert.compiler.serializer_nli import (
    NLI_TEMPLATE_ID,
    SERIALIZER_VERSION,
    all_sequences,
    compile_and_encode,
    split_by_question,
)
from jevbert.config import Limits
from jevbert.contracts.response import build_response
from jevbert.contracts.strict_json import parse_strict_json
from jevbert.contracts.validator import ValidatedRequest, validate_request
from jevbert.inference.registry import Bundle, ModelRegistry

logger = logging.getLogger("jevbert.api")

router = APIRouter()

_JSON_MEDIA_TYPE = "application/json"
_ALLOWED_CHARSETS = frozenset({"utf-8", "utf8"})


@router.post("/v1/systemone")
async def system_one(request: Request) -> Response:
    settings = request.app.state.settings
    registry: ModelRegistry = request.app.state.registry
    engine = request.app.state.engine
    log: dict[str, Any] = request.scope["state"]["log"]

    _check_media_type(request)
    body = await _read_body(request, settings.limits.max_body_bytes)

    parse_started = time.monotonic()
    value = parse_strict_json(body, max_depth=settings.limits.max_json_depth)
    validated = validate_request(value, settings.limits)
    log["parse_ms"] = round((time.monotonic() - parse_started) * 1000, 3)
    log["questions"] = _question_type_counts(validated)

    bundle = registry.resolve(validated.model)
    _set_bundle_headers(request, bundle)
    log["bundle"] = bundle.digest
    if not bundle.is_ready:
        raise ModelUnavailableError(
            "The model bundle is not ready to serve requests yet."
        )

    deadline = engine.deadline_from(request.scope["state"]["started_at"])
    encoded = await _run_encoding(engine, bundle, validated, settings.limits, deadline, log)
    logits = await _run_inference(engine, bundle, encoded, deadline, log)

    payload = build_response(
        model_id=bundle.public_id,
        encoded=encoded,
        grouped_logits=split_by_question(encoded, logits),
        temperatures=bundle.temperatures,
    )
    return json_response(payload)


async def _run_encoding(
    engine: Any,
    bundle: Bundle,
    validated: ValidatedRequest,
    limits: Limits,
    deadline: float,
    log: dict[str, Any],
) -> EncodedRequest:
    """Compile and tokenize on the encoder thread (S-H2).

    Held under the same deadline and the same admission counter as inference, so a
    tokenizer that takes minutes on a hostile input ends as a 504 instead of stalling
    the event loop for every other caller.
    """
    max_request_chars = limits.effective_max_request_chars

    def work(cancel: CancelToken) -> EncodedRequest:
        return compile_and_encode(
            validated, bundle.backend, bundle.token_budget, max_request_chars=max_request_chars
        )

    compile_started = time.monotonic()
    try:
        encoded = await engine.run_encoding(work, deadline=deadline)
    except InferenceCancelled as exc:
        raise DeadlineExceededError("The request deadline passed during encoding.") from exc
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        _log_backend_failure("encode", bundle, type(exc).__name__, sequences=None, tokens=None)
        raise InferenceError("The model failed to encode the request.") from exc
    log["compile_ms"] = round((time.monotonic() - compile_started) * 1000, 3)
    log["sequences"] = sum(len(q.sequences) for q in encoded.questions)
    log["input_tokens"] = encoded.total_tokens
    return encoded


async def _run_inference(
    engine: Any,
    bundle: Bundle,
    encoded: EncodedRequest,
    deadline: float,
    log: dict[str, Any],
) -> list[float]:
    sequences: list[EncodedSequence] = all_sequences(encoded)
    expected = len(sequences)

    def work(cancel: CancelToken) -> list[float]:
        return bundle.backend.score(sequences, cancel)

    queued_at = time.monotonic()
    try:
        logits = await engine.run(work, deadline=deadline)
    except InferenceCancelled as exc:
        # The only thing that cancels a job is the deadline or a dropped connection,
        # and the contract for that is 504, not 503 (A-F9, spec 5.9).
        raise DeadlineExceededError("The request deadline passed during inference.") from exc
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        # Includes CUDA OOM once the real backend is in place (POC_DESIGN 6.3).
        _log_backend_failure(
            "score", bundle, type(exc).__name__, sequences=expected, tokens=encoded.total_tokens
        )
        raise InferenceError("The model failed to score the request.") from exc
    log["inference_ms"] = round((time.monotonic() - queued_at) * 1000, 3)

    if len(logits) != expected:
        raise InferenceError(
            f"The backend returned {len(logits)} logits for {expected} candidates."
        )
    return logits


def _log_backend_failure(
    stage: str, bundle: Bundle, error_class: str, *, sequences: int | None, tokens: int | None
) -> None:
    """Report a backend failure with context but without the exception message.

    A real tokenizer or torch error quotes the offending input, and a traceback ends
    with that same message, so neither the message nor ``exc_info`` may be logged
    (S-M2, spec 15.2). What an operator needs - which stage, which backend, which
    bundle and how much work was in flight - carries no request data.
    """
    logger.error(
        "backend failure: stage=%s backend=%s bundle=%s error_class=%s sequences=%s tokens=%s",
        stage,
        bundle.manifest.backend,
        bundle.digest,
        error_class,
        "unknown" if sequences is None else sequences,
        "unknown" if tokens is None else tokens,
    )


@router.get("/v1/models")
async def list_models(request: Request) -> Response:
    registry: ModelRegistry = request.app.state.registry
    return json_response({"models": registry.list_models()})


@router.get("/healthz")
async def healthz(request: Request) -> Response:
    """Liveness only. Never reports model detail (spec 5.1, 16.1)."""
    return json_response({"status": "ok"})


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    """Readiness only.

    Served without credentials, so the body says whether the server is ready and
    nothing else. Which bundle is in which state is model detail and belongs to the
    authenticated capabilities endpoint (S-L2, spec 16.1).
    """
    registry: ModelRegistry = request.app.state.registry
    if registry.is_ready():
        return json_response({"status": "ready"})
    return json_response(
        {"status": "not_ready"}, status_code=503, headers={"Retry-After": "1"}
    )


@router.get("/jevbert/v1/capabilities")
async def capabilities(request: Request) -> Response:
    settings = request.app.state.settings
    registry: ModelRegistry = request.app.state.registry

    limits = settings.limits
    payload = {
        "contract": settings.contract_profile,
        "stage": "P0.5-poc",
        "bundles": [_bundle_capabilities(bundle) for bundle in registry.bundles],
        "aliases": registry.aliases,
        "limits": {
            "max_body_bytes": limits.max_body_bytes,
            "max_json_depth": limits.max_json_depth,
            "max_questions": limits.max_questions,
            "min_choice_options": limits.min_choice_options,
            "max_choice_options": limits.max_choice_options,
            "min_score_levels": limits.min_score_levels,
            "max_score_levels": limits.max_score_levels,
            "max_request_chars": limits.effective_max_request_chars,
            "overflow_policy": limits.overflow_policy,
            "request_deadline_seconds": settings.serving.request_deadline_seconds,
            "max_pending_requests": settings.serving.max_pending_requests,
        },
        "known_differences": KNOWN_DIFFERENCES,
    }
    return json_response(payload)


#: Summary of compat/differences.md, surfaced so that a caller sees it without the repo.
KNOWN_DIFFERENCES = [
    "confidence is a JevBERT-specific normalized-entropy statistic, not Jev's value",
    "usage.input_tokens counts JevBERT's expanded input and is not a Jev billing token count",
    "the response model is always an immutable JevBERT bundle ID, never a Jev version",
    "probabilities are uncalibrated (T=1); Jev thresholds do not transfer",
    "tie-breaking, error bodies and unknown-field handling are JevBERT decisions, "
    "not verified against the real Jev API",
]


def _bundle_capabilities(bundle: Bundle) -> dict[str, Any]:
    manifest = bundle.manifest
    source = manifest.source_model
    return {
        "id": manifest.public_id,
        "digest": bundle.digest,
        "state": bundle.state.value,
        "backend": manifest.backend,
        "serializer": manifest.serializer_version,
        "template": NLI_TEMPLATE_ID if manifest.serializer_version == SERIALIZER_VERSION else None,
        "source_model": (
            None if source is None else {"repo": source.repo, "revision": source.revision}
        ),
        "calibration": {
            "state": manifest.calibration.state,
            "temperature": manifest.calibration.temperature,
        },
        "confidence": manifest.confidence,
        "usage": manifest.usage_semantics,
        "dtype": manifest.dtype,
        "limits": {
            "max_sequence_tokens": manifest.limits.max_sequence_tokens,
            "max_request_tokens": manifest.limits.max_request_tokens,
        },
        "validated": {
            "languages": manifest.validated.languages,
            "domains": manifest.validated.domains,
            "max_choice_options": manifest.validated.max_choice_options,
            "quality": manifest.validated.quality,
        },
    }


def _check_media_type(request: Request) -> None:
    raw = request.headers.get("content-type", "")
    parts = [part.strip() for part in raw.split(";")]
    if not parts or parts[0].lower() != _JSON_MEDIA_TYPE:
        raise UnsupportedMediaTypeError(
            f"The request body must be sent as {_JSON_MEDIA_TYPE}."
        )
    for parameter in parts[1:]:
        name, _, value = parameter.partition("=")
        if name.strip().lower() == "charset" and value.strip().strip('"').lower() not in (
            _ALLOWED_CHARSETS
        ):
            raise UnsupportedMediaTypeError("The request body must be encoded as UTF-8.")

    encoding = request.headers.get("content-encoding")
    if encoding is not None and encoding.strip().lower() not in ("", "identity"):
        raise UnsupportedMediaTypeError("Content-Encoding is not supported.")


async def _read_body(request: Request, max_bytes: int) -> bytes:
    """Read the body, refusing anything past the limit by actually received bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise RequestTooLargeError(f"The request body exceeds {max_bytes} bytes.")

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > max_bytes:
            raise RequestTooLargeError(f"The request body exceeds {max_bytes} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _set_bundle_headers(request: Request, bundle: Bundle) -> None:
    headers = request.scope["state"]["extra_headers"]
    headers["X-JevBERT-Bundle"] = bundle.digest
    headers["X-JevBERT-Usage"] = bundle.manifest.usage_semantics
    headers["X-JevBERT-Calibration"] = bundle.manifest.calibration.state
    headers["X-JevBERT-Confidence"] = bundle.manifest.confidence or CONFIDENCE_DEFINITION


def _question_type_counts(validated: Any) -> dict[str, int]:
    counts = {"noul": 0, "choice": 0, "score": 0}
    for question in validated.questions:
        counts[question.type] += 1
    return counts
