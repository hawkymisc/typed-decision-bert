"""Registry, manifests and bundle digests (spec 5.7, 16.3; POC_DESIGN 6.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevbert.api.errors import ModelNotFoundError
from jevbert.backends.base import Backend, CancelToken, EncodedSequence, TextPair
from jevbert.backends.fake import FakeBackend
from jevbert.config import ConfigurationError, ServingSettings
from jevbert.inference.registry import (
    Bundle,
    BundleState,
    ModelRegistry,
    build_backend,
    build_registry,
    bundle_digest,
    read_manifest,
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
    def test_known_backend_is_built(self, tmp_path: Path) -> None:
        manifest, _ = read_manifest(FAKE_MANIFEST)
        assert isinstance(build_backend(manifest, tmp_path), FakeBackend)

    def test_unknown_backend_names_the_known_ones(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, "unknown.json", backend="a0-nli-zeroshot-v1")
        manifest, _ = read_manifest(path)
        with pytest.raises(ConfigurationError, match="a0-nli-zeroshot-v1"):
            build_backend(manifest, tmp_path)


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
        assert "no weights" in (bundle.error or "")

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
