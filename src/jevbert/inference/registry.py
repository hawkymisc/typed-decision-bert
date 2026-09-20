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
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jevbert.api.errors import ModelNotFoundError
from jevbert.backends.base import Backend, CancelToken, TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.normalize import canonical_json
from jevbert.config import ConfigurationError, Settings

logger = logging.getLogger("jevbert.registry")

#: The fake bundle only ever loads when the configuration enables it (POC_DESIGN 6.2).
FAKE_MANIFEST_FILENAME = "jevbert-fake-0.0.0.json"

#: Fixed warmup input: a minimum fixture, not a user request (spec 16.1).
_WARMUP_PAIRS = (
    TextPair(premise="warmup", hypothesis="The answer to the question is no."),
    TextPair(premise="warmup", hypothesis="The answer to the question is yes."),
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
    validated: Validated = Field(default_factory=Validated)


class BundleState(Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


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

    def load(self) -> None:
        """Load, then warm up on a fixed fixture. Failure marks the bundle, not the process."""
        try:
            self.backend.load()
            self._warmup()
        except Exception as exc:
            self.state = BundleState.FAILED
            self.error = f"{type(exc).__name__}: {exc}"
            # Readiness failure must not be mistaken for liveness failure (spec 16.1).
            logger.error(
                "bundle %s failed to load: %s", self.public_id, self.error, exc_info=exc
            )
            return
        self.state = BundleState.READY
        logger.info("bundle %s is ready (%s)", self.public_id, self.digest)

    def _warmup(self) -> None:
        sequences = self.backend.count_and_encode(_WARMUP_PAIRS)
        if len(sequences) != len(_WARMUP_PAIRS):
            raise RuntimeError("warmup encoding returned the wrong number of sequences")
        logits = self.backend.score(sequences, CancelToken())
        if len(logits) != len(_WARMUP_PAIRS):
            raise RuntimeError("warmup scoring returned the wrong number of logits")
        if not all(isinstance(z, float) and z == z and abs(z) != float("inf") for z in logits):
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

    def load_all(self) -> None:
        for bundle in self._bundles:
            bundle.load()


BackendFactory = Callable[[BundleManifest, Path], Backend]


def _build_fake(manifest: BundleManifest, models_dir: Path) -> Backend:
    return FakeBackend()


#: Backend ID -> factory. Phase 2 adds ``a0-nli-zeroshot-v1`` here and nothing else in
#: this module has to change.
_BACKEND_FACTORIES: dict[str, BackendFactory] = {
    FakeBackend.backend_id: _build_fake,
}


def register_backend_factory(backend_id: str, factory: BackendFactory) -> None:
    if backend_id in _BACKEND_FACTORIES:
        raise ConfigurationError(f"Backend {backend_id!r} is already registered")
    _BACKEND_FACTORIES[backend_id] = factory


def build_backend(manifest: BundleManifest, models_dir: Path) -> Backend:
    factory = _BACKEND_FACTORIES.get(manifest.backend)
    if factory is None:
        raise ConfigurationError(
            f"Manifest {manifest.public_id} requests unknown backend {manifest.backend!r}. "
            f"Known backends: {', '.join(sorted(_BACKEND_FACTORIES))}"
        )
    return factory(manifest, models_dir)


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

    bundles: list[Bundle] = []
    for filename in filenames:
        manifest, digest = read_manifest(settings.manifests_dir / filename)
        if manifest.backend == FakeBackend.backend_id and not settings.enable_fake_bundle:
            raise ConfigurationError(
                f"{filename} uses the fake backend but enable_fake_bundle is false. "
                "The fake backend never registers implicitly."
            )
        _check_manifest_against_limits(manifest, settings)
        backend = build_backend(manifest, settings.models_dir)
        bundles.append(Bundle(manifest, digest, backend))

    if not bundles:
        raise ConfigurationError("No bundle configured; the server would answer nothing.")
    return ModelRegistry(bundles, settings.resolved_aliases())


def _check_manifest_against_limits(manifest: BundleManifest, settings: Settings) -> None:
    if manifest.limits.max_sequence_tokens > manifest.limits.max_request_tokens:
        raise ConfigurationError(
            f"{manifest.public_id}: max_sequence_tokens exceeds max_request_tokens"
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
