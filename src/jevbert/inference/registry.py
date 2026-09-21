"""Model registry, manifests and bundle digests (spec 5.7, 12.1; POC_DESIGN 6.5).

A bundle is identified by the digest of its whole manifest, so a changed weight hash, a
changed serializer version or a changed temperature all produce a different bundle ID
(spec 5.7, 16.3). The registry never downloads anything: weights are fetched ahead of
time by ``python -m jevbert fetch-model`` (ADR-014).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jevbert.api.errors import ModelNotFoundError
from jevbert.backends.base import (
    WARMUP_PREMISE,
    Backend,
    CancelToken,
    EncodedSequence,
    TextPair,
)
from jevbert.backends.fake import FakeBackend
from jevbert.backends.nli import NliZeroShotBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.normalize import canonical_json
from jevbert.compiler.serializer_nli import (
    NLI_TEMPLATE_ID,
    SERIALIZER_VERSION_FULL,
    template_id_of,
)
from jevbert.config import ConfigurationError, Settings
from jevbert.fetch import SOURCE_FILES, FetchError, model_directory

logger = logging.getLogger("jevbert.registry")

#: The fake bundle only ever loads when the configuration enables it (POC_DESIGN 6.2).
FAKE_MANIFEST_FILENAME = "jevbert-fake-0.0.0.json"

#: Fixed warmup input: a minimum fixture, not a user request (spec 16.1).
_WARMUP_PAIRS = (
    TextPair(premise=WARMUP_PREMISE, hypothesis="The answer to the question is no."),
    TextPair(premise=WARMUP_PREMISE, hypothesis="The answer to the question is yes."),
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceModel(_Strict):
    repo: str
    revision: str
    #: File name -> SHA-256, verified at load time by the backend (spec 15.3).
    files: dict[str, str] = Field(default_factory=dict)


class Calibration(_Strict):
    state: Literal["calibrated", "uncalibrated"]
    temperature: dict[str, float]


class BundleLimits(_Strict):
    max_sequence_tokens: int = Field(gt=0)
    max_request_tokens: int = Field(gt=0)


class Validated(_Strict):
    """Never filled in from assumption: unevaluated ranges stay visibly unevaluated."""

    languages: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    max_choice_options: int | str = "unknown"
    quality: str = "unevaluated"


class BundleManifest(_Strict):
    public_id: str = Field(min_length=1)
    description: str
    release_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    backend: str
    serializer_version: str
    source_model: SourceModel | None = None
    calibration: Calibration
    confidence: str
    usage_semantics: str
    dtype: str
    limits: BundleLimits
    #: Longest sequence the backbone can encode at all. ``limits.max_sequence_tokens``
    #: above it refuses startup rather than failing per request (POC_DESIGN 5.5).
    #: ``None`` where there is no model, as in the fake bundle.
    model_max_sequence_tokens: int | None = Field(default=None, gt=0)
    validated: Validated = Field(default_factory=Validated)


class BundleState(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@runtime_checkable
class WarmupRunner(Protocol):
    """The part of ``InferenceEngine`` a bundle needs in order to warm up (A-F10)."""

    def run_blocking(self, work: Callable[[CancelToken], Any]) -> Any: ...

    def run_encoding_blocking(self, work: Callable[[CancelToken], Any]) -> Any: ...


def bundle_digest(raw_manifest: Mapping[str, Any]) -> str:
    """``sha256`` of the manifest's canonical JSON (POC_DESIGN 6.5)."""
    encoded = canonical_json(dict(raw_manifest)).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class Bundle:
    """One registered inference bundle and its readiness state."""

    def __init__(self, manifest: BundleManifest, digest: str, backend: Backend) -> None:
        self.manifest = manifest
        self.digest = digest
        self.backend = backend
        self.state = BundleState.LOADING
        self.error: str | None = None

    @property
    def public_id(self) -> str:
        return self.manifest.public_id

    @property
    def is_ready(self) -> bool:
        return self.state is BundleState.READY

    @property
    def token_budget(self) -> TokenBudget:
        return TokenBudget(
            max_sequence_tokens=self.manifest.limits.max_sequence_tokens,
            max_request_tokens=self.manifest.limits.max_request_tokens,
        )

    @property
    def temperatures(self) -> dict[str, float]:
        return dict(self.manifest.calibration.temperature)

    def load(self, runner: WarmupRunner | None = None) -> None:
        """Load, then warm up on a fixed fixture.

        Failure marks the bundle, not the process (spec 16.1). ``runner`` is the
        inference engine: when it is given, warmup runs on the very threads that serve
        requests, so a warmup forward pass can never touch the device beside a served
        one (A-F10). Without it the calls are direct, which is what a unit test that
        has no engine wants.
        """
        try:
            self.backend.load()
            self._warmup(runner)
        except Exception as exc:
            self.state = BundleState.FAILED
            self.error = type(exc).__name__
            # Readiness failure must not be mistaken for liveness failure (spec 16.1).
            # The exception message is not logged: a backend failure can quote the
            # model path or the input it choked on (spec 15.2).
            logger.error(
                "bundle %s failed to load: error_class=%s", self.public_id, self.error
            )
            return
        self.state = BundleState.READY
        logger.info("bundle %s is ready (%s)", self.public_id, self.digest)

    def _warmup(self, runner: WarmupRunner | None) -> None:
        backend = self.backend

        def encode(cancel: CancelToken) -> list[EncodedSequence]:
            return backend.count_and_encode(_WARMUP_PAIRS)

        def score(cancel: CancelToken) -> list[float]:
            return backend.score(sequences, cancel)

        if runner is None:
            sequences = encode(CancelToken())
        else:
            sequences = runner.run_encoding_blocking(encode)
        if len(sequences) != len(_WARMUP_PAIRS):
            raise RuntimeError("warmup encoding returned the wrong number of sequences")

        logits = score(CancelToken()) if runner is None else runner.run_blocking(score)
        if len(logits) != len(_WARMUP_PAIRS):
            raise RuntimeError("warmup scoring returned the wrong number of logits")
        if not all(isinstance(z, float) and math.isfinite(z) for z in logits):
            raise RuntimeError("warmup scoring returned a non-finite logit")


class ModelRegistry:
    """Resolves a request's ``model`` to an immutable bundle (spec 2.3, 12.1)."""

    def __init__(self, bundles: Sequence[Bundle], aliases: Mapping[str, str] | None = None) -> None:
        self._bundles = tuple(bundles)
        by_id: dict[str, Bundle] = {}
        for bundle in self._bundles:
            if bundle.public_id in by_id:
                raise ConfigurationError(f"Duplicate bundle ID: {bundle.public_id}")
            by_id[bundle.public_id] = bundle
        self._by_id = by_id

        self._aliases: dict[str, str] = {}
        for alias, target in (aliases or {}).items():
            if target not in by_id:
                raise ConfigurationError(
                    f"Alias {alias!r} points at unregistered bundle {target!r}"
                )
            if alias in by_id:
                raise ConfigurationError(f"Alias {alias!r} shadows a bundle ID")
            self._aliases[alias] = target

    @property
    def bundles(self) -> tuple[Bundle, ...]:
        return self._bundles

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def resolve(self, name: str) -> Bundle:
        """Resolve an immutable ID or an enabled alias. No wildcards (spec 18.3)."""
        if name in self._by_id:
            return self._by_id[name]
        target = self._aliases.get(name)
        if target is not None:
            return self._by_id[target]
        raise ModelNotFoundError(
            "The requested model is not registered on this server.", path=["model"]
        )

    def is_ready(self) -> bool:
        return bool(self._bundles) and all(bundle.is_ready for bundle in self._bundles)

    def list_models(self) -> list[dict[str, str]]:
        """``GET /v1/models`` payload entries (spec 5.7). Aliases appear as their own rows."""
        models = [
            {
                "name": bundle.public_id,
                "description": bundle.manifest.description,
                "release_date": bundle.manifest.release_date,
            }
            for bundle in self._bundles
        ]
        for alias, target in self._aliases.items():
            bundle = self._by_id[target]
            models.append(
                {
                    "name": alias,
                    "description": f"Alias for {target}. {bundle.manifest.description}",
                    "release_date": bundle.manifest.release_date,
                }
            )
        return models

    def load_all(self, runner: WarmupRunner | None = None) -> None:
        for bundle in self._bundles:
            bundle.load(runner)


@dataclass(frozen=True)
class BackendContext:
    """Everything a backend needs from the server configuration (A-F5).

    Passing the whole context rather than a directory is what lets phase 2 add
    ``backends/nli.py`` plus one entry in ``_BACKEND_FACTORIES`` and change nothing
    else: the microbatch ceilings of POC_DESIGN 6.3 and the device choice arrive here
    instead of being settings that nothing reads.
    """

    models_dir: Path
    max_batch_tokens: int
    max_batch_sequences: int
    device: str


BackendFactory = Callable[[BundleManifest, BackendContext], Backend]


def backend_context(settings: Settings) -> BackendContext:
    return BackendContext(
        models_dir=settings.models_dir,
        max_batch_tokens=settings.serving.max_batch_tokens,
        max_batch_sequences=settings.serving.max_batch_sequences,
        device=settings.serving.device,
    )


def _build_fake(manifest: BundleManifest, context: BackendContext) -> Backend:
    """The fake backend has no weights, no device and no batching to configure.

    Nothing from the manifest or the configuration selects ``logit_fn`` or a delay:
    those are constructor arguments of the test double alone (POC_DESIGN 6.2).
    """
    return FakeBackend()


def _build_nli(manifest: BundleManifest, context: BackendContext) -> Backend:
    """Build the zero-shot NLI backend from the manifest's pinned source model.

    The directory is derived from the manifest's repo ID, not from anything a caller
    can influence, and the per-file hashes travel with it so that the backend refuses
    to load bytes the manifest does not vouch for (spec 15.3).
    """
    source = manifest.source_model
    if source is None:
        raise ConfigurationError(
            f"{manifest.public_id}: backend {manifest.backend!r} needs a source_model "
            "with a pinned revision and per-file hashes."
        )
    if not source.files:
        raise ConfigurationError(
            f"{manifest.public_id}: the manifest records no source_model.files. "
            "Run `python -m jevbert fetch-model` before starting the server."
        )
    # The hashes have to cover every file the backend will open, or the check is
    # decoration: one recorded config.json used to let the server start and load
    # unverified weights (S-M3). Refusing at startup beats failing at readiness.
    unverified = sorted(set(SOURCE_FILES) - set(source.files))
    if unverified:
        raise ConfigurationError(
            f"{manifest.public_id}: the manifest records no hash for "
            f"{', '.join(unverified)}, which the backend loads. "
            "Run `python -m jevbert fetch-model` before starting the server."
        )
    try:
        directory = model_directory(context.models_dir, source.repo)
    except FetchError as exc:
        raise ConfigurationError(f"{manifest.public_id}: {exc}") from exc
    return NliZeroShotBackend(
        directory,
        max_batch_tokens=context.max_batch_tokens,
        max_batch_sequences=context.max_batch_sequences,
        device=context.device,
        dtype=manifest.dtype,
        expected_file_hashes=source.files,
    )


#: Backend ID -> factory.
_BACKEND_FACTORIES: dict[str, BackendFactory] = {
    FakeBackend.backend_id: _build_fake,
    NliZeroShotBackend.backend_id: _build_nli,
}


def register_backend_factory(backend_id: str, factory: BackendFactory) -> None:
    if backend_id in _BACKEND_FACTORIES:
        raise ConfigurationError(f"Backend {backend_id!r} is already registered")
    _BACKEND_FACTORIES[backend_id] = factory


def build_backend(manifest: BundleManifest, context: BackendContext) -> Backend:
    factory = _BACKEND_FACTORIES.get(manifest.backend)
    if factory is None:
        raise ConfigurationError(
            f"Manifest {manifest.public_id} requests unknown backend {manifest.backend!r}. "
            f"Known backends: {', '.join(sorted(_BACKEND_FACTORIES))}"
        )
    return factory(manifest, context)


def read_manifest(path: Path) -> tuple[BundleManifest, str]:
    """Parse a manifest file and compute its bundle digest."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read the manifest {path}") from exc
    try:
        raw: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    try:
        manifest = BundleManifest.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} is not a valid bundle manifest:\n{exc}") from exc
    return manifest, bundle_digest(raw)


def build_registry(settings: Settings) -> ModelRegistry:
    """Build the registry described by the configuration, without loading weights."""
    filenames = list(settings.bundles)
    if settings.enable_fake_bundle and FAKE_MANIFEST_FILENAME not in filenames:
        filenames.append(FAKE_MANIFEST_FILENAME)

    context = backend_context(settings)
    bundles: list[Bundle] = []
    for filename in filenames:
        manifest, digest = read_manifest(settings.manifests_dir / filename)
        if manifest.backend == FakeBackend.backend_id and not settings.enable_fake_bundle:
            raise ConfigurationError(
                f"{filename} uses the fake backend but enable_fake_bundle is false. "
                "The fake backend never registers implicitly."
            )
        _check_manifest_against_limits(manifest, settings)
        backend = build_backend(manifest, context)
        bundles.append(Bundle(manifest, digest, backend))

    if not bundles:
        raise ConfigurationError("No bundle configured; the server would answer nothing.")
    return ModelRegistry(bundles, settings.resolved_aliases())


def _check_manifest_against_limits(manifest: BundleManifest, settings: Settings) -> None:
    """A manifest may narrow the configured limits; it may never widen them (A-F6).

    The bundle's limits are what the request path actually enforces, so a manifest
    declaring more than the server was configured for would quietly raise the ceiling
    the operator set, and one declaring more than the backbone can encode would turn
    into a failure on every long request instead of a refusal to start.
    """
    if manifest.limits.max_sequence_tokens > manifest.limits.max_request_tokens:
        raise ConfigurationError(
            f"{manifest.public_id}: max_sequence_tokens exceeds max_request_tokens"
        )
    for name in ("max_sequence_tokens", "max_request_tokens"):
        declared = getattr(manifest.limits, name)
        configured = getattr(settings.limits, name)
        if declared > configured:
            raise ConfigurationError(
                f"{manifest.public_id}: the manifest declares {name}={declared} but the "
                f"server is configured for {configured}. A manifest cannot widen a limit."
            )
    model_limit = manifest.model_max_sequence_tokens
    if model_limit is not None and manifest.limits.max_sequence_tokens > model_limit:
        raise ConfigurationError(
            f"{manifest.public_id}: max_sequence_tokens="
            f"{manifest.limits.max_sequence_tokens} exceeds "
            f"model_max_sequence_tokens={model_limit}."
        )
    declared_template = template_id_of(manifest.serializer_version)
    if declared_template is None:
        raise ConfigurationError(
            f"{manifest.public_id}: serializer_version={manifest.serializer_version!r} "
            f"does not name a template. Expected {SERIALIZER_VERSION_FULL!r}."
        )
    if declared_template != NLI_TEMPLATE_ID:
        # The template decides every compiled model input, so a manifest that names a
        # different one is promising answers this build cannot produce. Refusing to
        # start is the only way the bundle ID keeps meaning what it says (spec 5.7).
        raise ConfigurationError(
            f"{manifest.public_id}: the manifest names template {declared_template!r} "
            f"but this build compiles with {NLI_TEMPLATE_ID!r}."
        )
    if manifest.confidence != "normalized-entropy-v1":
        raise ConfigurationError(
            f"{manifest.public_id}: this server only implements normalized-entropy-v1"
        )
    for question_type in ("noul", "choice", "score"):
        temperature = manifest.calibration.temperature.get(question_type)
        if temperature is None or not temperature > 0.0:
            raise ConfigurationError(
                f"{manifest.public_id}: a positive temperature is required for {question_type}"
            )
