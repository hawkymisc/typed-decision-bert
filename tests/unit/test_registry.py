"""Registry, manifests and bundle digests (spec 5.7, 16.3; POC_DESIGN 6.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevbert.api.errors import ModelNotFoundError
from jevbert.backends.base import Backend, CancelToken, EncodedSequence, TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.backends.nli import NliZeroShotBackend
from jevbert.compiler.serializer_nli import NLI_TEMPLATE_ID, template_id_of
from jevbert.config import ConfigurationError, Limits, ServingSettings
from jevbert.fetch import SOURCE_FILES, model_directory
from jevbert.inference.registry import (
    _BACKEND_FACTORIES,
    BackendContext,
    Bundle,
    BundleManifest,
    BundleState,
    ModelRegistry,
    backend_context,
    build_backend,
    build_registry,
    bundle_digest,
    read_manifest,
    register_backend_factory,
)
from tests.conftest import FAKE_MANIFEST, MODEL, build_settings

RAW_MANIFEST: dict[str, Any] = json.loads(FAKE_MANIFEST.read_text(encoding="utf-8"))


def write_manifest(tmp_path: Path, name: str, **overrides: Any) -> Path:
    data = dict(RAW_MANIFEST)
    data.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestBundleDigest:
    def test_digest_is_prefixed_and_stable(self) -> None:
        digest = bundle_digest(RAW_MANIFEST)
        assert digest.startswith("sha256:")
        assert digest == bundle_digest(RAW_MANIFEST)

    def test_key_order_does_not_change_the_digest(self) -> None:
        reordered = dict(reversed(list(RAW_MANIFEST.items())))
        assert bundle_digest(reordered) == bundle_digest(RAW_MANIFEST)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("serializer_version", "serializer-nli-v2"),
            ("dtype", "float16"),
            ("usage_semantics", "expanded-input-v1"),
            ("description", "different"),
        ],
    )
    def test_any_manifest_change_changes_the_digest(self, field: str, value: str) -> None:
        # spec 5.7: recalibration or a serializer change must not reuse a bundle ID.
        changed = dict(RAW_MANIFEST)
        changed[field] = value
        assert bundle_digest(changed) != bundle_digest(RAW_MANIFEST)

    def test_a_changed_temperature_changes_the_digest(self) -> None:
        changed = dict(RAW_MANIFEST)
        changed["calibration"] = {
            "state": "calibrated",
            "temperature": {"noul": 1.5, "choice": 1.0, "score": 1.0},
        }
        assert bundle_digest(changed) != bundle_digest(RAW_MANIFEST)


class TestReadManifest:
    def test_reads_the_fake_manifest(self) -> None:
        manifest, digest = read_manifest(FAKE_MANIFEST)
        assert manifest.public_id == MODEL
        assert manifest.backend == "fake-deterministic-v1"
        assert manifest.calibration.state == "uncalibrated"
        assert digest.startswith("sha256:")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            read_manifest(tmp_path / "absent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            read_manifest(path)

    def test_unknown_manifest_field_is_refused(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, "extra.json", surprise=1)
        with pytest.raises(ConfigurationError):
            read_manifest(path)

    @pytest.mark.parametrize("release_date", ["2026/09/21", "21-09-2026", "today", ""])
    def test_release_date_must_be_iso(self, tmp_path: Path, release_date: str) -> None:
        path = write_manifest(tmp_path, "date.json", release_date=release_date)
        with pytest.raises(ConfigurationError):
            read_manifest(path)


class TestBackendFactory:
    """A-F5: a factory receives everything a real backend needs to configure itself."""

    def test_known_backend_is_built(self, tmp_path: Path) -> None:
        manifest, _ = read_manifest(FAKE_MANIFEST)
        assert isinstance(build_backend(manifest, _context(tmp_path)), FakeBackend)

    def test_unknown_backend_names_the_known_ones(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, "unknown.json", backend="a0-nli-zeroshot-v2")
        manifest, _ = read_manifest(path)
        with pytest.raises(ConfigurationError, match="a0-nli-zeroshot-v2"):
            build_backend(manifest, _context(tmp_path))

    def test_the_context_carries_the_serving_batch_settings(self) -> None:
        # Before phase 1.5 these three were configurable and read by nobody, which is
        # the kind of setting that looks like it does something and does not.
        settings = build_settings(
            serving=ServingSettings(
                max_batch_tokens=1234, max_batch_sequences=7, device="cpu"
            )
        )
        context = backend_context(settings)
        assert context.max_batch_tokens == 1234
        assert context.max_batch_sequences == 7
        assert context.device == "cpu"
        assert context.models_dir == settings.models_dir

    def test_the_context_defaults_match_the_configuration_files(self) -> None:
        context = backend_context(build_settings())
        assert (context.max_batch_tokens, context.max_batch_sequences) == (16_384, 64)
        assert context.device == "auto"

    def test_a_registered_factory_receives_the_context(self) -> None:
        seen: list[BackendContext] = []

        def factory(manifest: BundleManifest, context: BackendContext) -> FakeBackend:
            seen.append(context)
            return FakeBackend()

        backend_id = "test-only-context-probe-v1"
        register_backend_factory(backend_id, factory)
        try:
            manifest, _ = read_manifest(FAKE_MANIFEST)
            probe = manifest.model_copy(update={"backend": backend_id})
            context = backend_context(build_settings())
            assert isinstance(build_backend(probe, context), FakeBackend)
            assert seen == [context]
        finally:
            _BACKEND_FACTORIES.pop(backend_id, None)

    def test_a_backend_id_cannot_be_registered_twice(self) -> None:
        def factory(manifest: BundleManifest, context: BackendContext) -> FakeBackend:
            return FakeBackend()

        with pytest.raises(ConfigurationError, match="already registered"):
            register_backend_factory(FakeBackend.backend_id, factory)


def _context(models_dir: Path) -> BackendContext:
    return BackendContext(
        models_dir=models_dir, max_batch_tokens=16_384, max_batch_sequences=64, device="auto"
    )


class TestManifestLimitsNeverWidenTheServer:
    """A-F6: a manifest may narrow the configured limits, never widen them."""

    def _settings(self, tmp_path: Path, **limit_overrides: int) -> Any:
        return build_settings(
            manifests_dir=tmp_path,
            enable_fake_bundle=True,
            limits=Limits(max_sequence_tokens=2048, max_request_tokens=131_072, **limit_overrides),
        )

    def test_a_manifest_sequence_limit_above_the_server_limit_refuses_startup(
        self, tmp_path: Path
    ) -> None:
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            limits={"max_sequence_tokens": 4096, "max_request_tokens": 131_072},
        )
        with pytest.raises(ConfigurationError, match="max_sequence_tokens"):
            build_registry(self._settings(tmp_path))

    def test_a_manifest_request_limit_above_the_server_limit_refuses_startup(
        self, tmp_path: Path
    ) -> None:
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            limits={"max_sequence_tokens": 2048, "max_request_tokens": 262_144},
        )
        with pytest.raises(ConfigurationError, match="max_request_tokens"):
            build_registry(self._settings(tmp_path))

    def test_a_narrower_manifest_is_accepted(self, tmp_path: Path) -> None:
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            limits={"max_sequence_tokens": 512, "max_request_tokens": 1024},
        )
        registry = build_registry(self._settings(tmp_path))
        budget = registry.bundles[0].token_budget
        assert (budget.max_sequence_tokens, budget.max_request_tokens) == (512, 1024)

    def test_a_sequence_limit_above_the_model_limit_refuses_startup(
        self, tmp_path: Path
    ) -> None:
        # POC_DESIGN 5.5: a configuration above what the backbone can encode must not
        # start, because the overflow would surface as a runtime failure per request.
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            model_max_sequence_tokens=1024,
            limits={"max_sequence_tokens": 2048, "max_request_tokens": 131_072},
        )
        with pytest.raises(ConfigurationError, match="model_max_sequence_tokens"):
            build_registry(self._settings(tmp_path))

    def test_a_sequence_limit_at_the_model_limit_is_accepted(self, tmp_path: Path) -> None:
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            model_max_sequence_tokens=2048,
            limits={"max_sequence_tokens": 2048, "max_request_tokens": 131_072},
        )
        registry = build_registry(self._settings(tmp_path))
        assert registry.bundles[0].manifest.model_max_sequence_tokens == 2048

    def test_the_model_limit_is_optional(self) -> None:
        manifest, _ = read_manifest(FAKE_MANIFEST)
        assert manifest.model_max_sequence_tokens is None


class TestModelRegistry:
    def _registry(self, backend: Backend | None = None, aliases: Any = None) -> ModelRegistry:
        manifest, digest = read_manifest(FAKE_MANIFEST)
        return ModelRegistry([Bundle(manifest, digest, backend or FakeBackend())], aliases)

    def test_resolves_the_immutable_id(self) -> None:
        assert self._registry().resolve(MODEL).public_id == MODEL

    def test_unknown_name_raises_model_not_found(self) -> None:
        with pytest.raises(ModelNotFoundError):
            self._registry().resolve("something-else")

    def test_alias_resolves_to_the_bundle(self) -> None:
        registry = self._registry(aliases={"jev-latest": MODEL})
        assert registry.resolve("jev-latest").public_id == MODEL

    def test_alias_to_an_unknown_bundle_is_refused(self) -> None:
        with pytest.raises(ConfigurationError):
            self._registry(aliases={"jev-latest": "absent"})

    def test_alias_cannot_shadow_a_bundle_id(self) -> None:
        with pytest.raises(ConfigurationError):
            self._registry(aliases={MODEL: MODEL})

    def test_duplicate_bundle_ids_are_refused(self) -> None:
        manifest, digest = read_manifest(FAKE_MANIFEST)
        with pytest.raises(ConfigurationError):
            ModelRegistry(
                [
                    Bundle(manifest, digest, FakeBackend()),
                    Bundle(manifest, digest, FakeBackend()),
                ]
            )

    def test_readiness_follows_loading(self) -> None:
        registry = self._registry()
        assert not registry.is_ready()
        assert registry.bundles[0].state is BundleState.LOADING
        registry.load_all()
        assert registry.is_ready()
        assert registry.bundles[0].state is BundleState.READY

    def test_token_budget_comes_from_the_manifest(self) -> None:
        budget = self._registry().bundles[0].token_budget
        assert budget.max_sequence_tokens == 2048
        assert budget.max_request_tokens == 131072


class TestWarmup:
    def test_a_backend_that_fails_to_load_is_marked_failed(self) -> None:
        class Failing(FakeBackend):
            def load(self) -> None:
                raise RuntimeError("no weights")

        manifest, digest = read_manifest(FAKE_MANIFEST)
        bundle = Bundle(manifest, digest, Failing())
        bundle.load()
        assert bundle.state is BundleState.FAILED
        # The class name, not the message: a real backend quotes the model path or the
        # input it choked on, and this string is one step from a log line (S-M2).
        assert bundle.error == "RuntimeError"
        assert "no weights" not in (bundle.error or "")

    def test_a_backend_returning_a_non_finite_warmup_logit_is_marked_failed(self) -> None:
        class NotFinite(FakeBackend):
            def score(self, sequences: Any, cancel: CancelToken) -> list[float]:
                return [float("nan")] * len(sequences)

        manifest, digest = read_manifest(FAKE_MANIFEST)
        bundle = Bundle(manifest, digest, NotFinite())
        bundle.load()
        assert bundle.state is BundleState.FAILED

    def test_a_backend_returning_the_wrong_warmup_count_is_marked_failed(self) -> None:
        class WrongCount(FakeBackend):
            def count_and_encode(self, pairs: Any) -> list[EncodedSequence]:
                return [EncodedSequence(token_count=1, data=TextPair("a", "b"))]

        manifest, digest = read_manifest(FAKE_MANIFEST)
        bundle = Bundle(manifest, digest, WrongCount())
        bundle.load()
        assert bundle.state is BundleState.FAILED


class TestBuildRegistryFromSettings:
    def test_fake_bundle_is_registered_when_enabled(self) -> None:
        registry = build_registry(build_settings(enable_fake_bundle=True))
        assert [b.public_id for b in registry.bundles] == [MODEL]

    def test_fake_bundle_is_not_registered_implicitly(self) -> None:
        with pytest.raises(ConfigurationError, match="No bundle configured"):
            build_registry(build_settings(enable_fake_bundle=False))

    def test_listing_the_fake_manifest_without_the_flag_is_refused(self) -> None:
        settings = build_settings(
            enable_fake_bundle=False, bundles=("jevbert-fake-0.0.0.json",)
        )
        with pytest.raises(ConfigurationError, match="enable_fake_bundle"):
            build_registry(settings)

    def test_aliases_are_applied_only_when_enabled(self) -> None:
        enabled = build_settings(
            serving=ServingSettings(allow_jev_aliases=True, aliases={"jev-latest": MODEL})
        )
        assert build_registry(enabled).aliases == {"jev-latest": MODEL}
        assert build_registry(build_settings()).aliases == {}

    def test_a_manifest_with_a_foreign_confidence_definition_is_refused(
        self, tmp_path: Path
    ) -> None:
        write_manifest(tmp_path, "jevbert-fake-0.0.0.json", confidence="max-probability-v1")
        settings = build_settings(manifests_dir=tmp_path, enable_fake_bundle=True)
        with pytest.raises(ConfigurationError, match="normalized-entropy-v1"):
            build_registry(settings)

    def test_a_manifest_with_a_non_positive_temperature_is_refused(
        self, tmp_path: Path
    ) -> None:
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            calibration={
                "state": "uncalibrated",
                "temperature": {"noul": 0.0, "choice": 1.0, "score": 1.0},
            },
        )
        settings = build_settings(manifests_dir=tmp_path, enable_fake_bundle=True)
        with pytest.raises(ConfigurationError, match="temperature"):
            build_registry(settings)


class TestModelListing:
    def test_entries_come_from_the_manifest(self) -> None:
        registry = build_registry(build_settings())
        entry = registry.list_models()[0]
        assert entry["name"] == MODEL
        assert entry["release_date"] == RAW_MANIFEST["release_date"]
        assert entry["description"] == RAW_MANIFEST["description"]


class TestNliBackendFactory:
    """a0-nli-zeroshot-v2 is built from the manifest alone (A-F5, spec 15.3)."""

    def _manifest(self, **source_overrides: Any) -> BundleManifest:
        source = {
            "repo": "MoritzLaurer/bge-m3-zeroshot-v2.0",
            "revision": "9abf1c8aaeb82a2447809c20753ed0b106b76652",
            "files": {"config.json": "ab" * 32},
        }
        source.update(source_overrides)
        payload = dict(RAW_MANIFEST)
        payload.update(backend="a0-nli-zeroshot-v2", source_model=source)
        return BundleManifest.model_validate(payload)

    def test_the_backend_id_is_registered(self) -> None:
        assert "a0-nli-zeroshot-v2" in _BACKEND_FACTORIES

    def test_the_model_directory_comes_from_the_manifest_repo(self, tmp_path: Path) -> None:
        manifest = self._manifest(files=dict.fromkeys(SOURCE_FILES, "ab" * 32))
        backend = build_backend(manifest, _context(tmp_path))
        assert isinstance(backend, NliZeroShotBackend)
        assert backend.model_dir == model_directory(tmp_path, "MoritzLaurer/bge-m3-zeroshot-v2.0")

    def test_a_manifest_without_a_source_model_refuses_startup(self, tmp_path: Path) -> None:
        payload = dict(RAW_MANIFEST)
        payload.update(backend="a0-nli-zeroshot-v2", source_model=None)
        manifest = BundleManifest.model_validate(payload)
        with pytest.raises(ConfigurationError, match="source_model"):
            build_backend(manifest, _context(tmp_path))

    def test_a_manifest_without_file_hashes_refuses_startup(self, tmp_path: Path) -> None:
        # Unverifiable weights are worse than absent ones: the bundle digest would
        # promise an identity the loaded bytes never had to match (spec 5.7, 15.3).
        with pytest.raises(ConfigurationError, match="fetch-model"):
            build_backend(self._manifest(files={}), _context(tmp_path))

    def test_hashes_that_do_not_cover_every_loaded_file_refuse_startup(
        self, tmp_path: Path
    ) -> None:
        # S-M3: the verified set and the loaded set were unrelated. One hashed
        # config.json was enough to start, and model.safetensors - the file that
        # decides every answer - was then read without ever being checked.
        with pytest.raises(ConfigurationError, match="model.safetensors"):
            build_backend(self._manifest(), _context(tmp_path))

    def test_the_full_source_file_set_is_accepted(self, tmp_path: Path) -> None:
        files = dict.fromkeys(SOURCE_FILES, "ab" * 32)
        backend = build_backend(self._manifest(files=files), _context(tmp_path))
        assert isinstance(backend, NliZeroShotBackend)

    def test_extra_recorded_hashes_are_allowed(self, tmp_path: Path) -> None:
        # Narrower than the loaded set is the failure; wider only means more is
        # verified than is opened.
        files = dict.fromkeys(SOURCE_FILES, "ab" * 32) | {"extra.json": "cd" * 32}
        assert build_backend(self._manifest(files=files), _context(tmp_path)) is not None

    @pytest.mark.parametrize(
        "repo", ["..\\..\\evil", "../../evil", "a/b/c", "", "owner/", "own er/name", "C:/x/y"]
    )
    def test_a_repo_id_that_is_not_a_repo_id_refuses_startup(
        self, tmp_path: Path, repo: str
    ) -> None:
        # S-M5: `repo_id.replace("/", "--")` leaves a Windows separator intact, so a
        # manifest could name a directory outside models/.
        files = dict.fromkeys(SOURCE_FILES, "ab" * 32)
        with pytest.raises(ConfigurationError, match="repo"):
            build_backend(self._manifest(repo=repo, files=files), _context(tmp_path))


class TestSerializerVersionNamesTheTemplate:
    """spec 5.7: the bundle ID identifies the serializer, template included."""

    def _settings(self, tmp_path: Path) -> Any:
        return build_settings(manifests_dir=tmp_path, enable_fake_bundle=True)

    def test_the_shipped_manifests_name_this_build_s_template(self) -> None:
        manifest, _ = read_manifest(FAKE_MANIFEST)
        assert template_id_of(manifest.serializer_version) == NLI_TEMPLATE_ID

    def test_a_serializer_version_without_a_template_refuses_startup(
        self, tmp_path: Path
    ) -> None:
        write_manifest(tmp_path, "jevbert-fake-0.0.0.json", serializer_version="serializer-nli-v1")
        with pytest.raises(ConfigurationError, match="does not name a template"):
            build_registry(self._settings(tmp_path))

    def test_a_manifest_naming_another_template_refuses_startup(self, tmp_path: Path) -> None:
        # The template decides every compiled input, so serving one while promising the
        # other would make the bundle digest a claim the answers do not honour.
        write_manifest(
            tmp_path,
            "jevbert-fake-0.0.0.json",
            serializer_version="serializer-nli-v1+nli-template-v9",
        )
        with pytest.raises(ConfigurationError, match="nli-template-v9"):
            build_registry(self._settings(tmp_path))

    def test_changing_the_template_changes_the_digest(self) -> None:
        changed = dict(RAW_MANIFEST, serializer_version="serializer-nli-v1+nli-template-v2")
        assert bundle_digest(changed) != bundle_digest(RAW_MANIFEST)
