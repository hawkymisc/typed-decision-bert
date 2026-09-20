"""`python -m jevbert fetch-model` (POC_DESIGN 6.5, ADR-014; spec 15.3).

The fetch step is the only part of the system that touches the network, and it is the
step that pins what the server will later load: a fixed revision, safetensors only, no
remote code, and a SHA-256 per file recorded in the manifest so that a changed byte
changes the bundle digest. None of these tests download anything - the downloader is a
parameter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from jevbert.fetch import (
    MANIFEST_FILENAME,
    REPO_ID,
    REVISION,
    SOURCE_FILES,
    FetchError,
    fetch_model,
    hash_files,
    mismatched_files,
    model_directory,
    record_source_hashes,
    sha256_file,
)
from jevbert.inference.registry import read_manifest

MANIFEST_TEMPLATE: dict[str, Any] = {
    "public_id": "jevbert-poc-nli-ja-en-0.1.0",
    "description": "test manifest",
    "release_date": "2026-09-21",
    "backend": "a0-nli-zeroshot-v1",
    "serializer_version": "serializer-nli-v1+nli-template-v1",
    "source_model": {"repo": REPO_ID, "revision": REVISION, "files": {}},
    "calibration": {
        "state": "uncalibrated",
        "temperature": {"noul": 1.0, "choice": 1.0, "score": 1.0},
    },
    "confidence": "normalized-entropy-v1",
    "usage_semantics": "expanded-input-a0-v1",
    "dtype": "float16",
    "limits": {"max_sequence_tokens": 2048, "max_request_tokens": 131072},
    "model_max_sequence_tokens": 8192,
    "validated": {
        "languages": [],
        "domains": [],
        "max_choice_options": "unknown",
        "quality": "unevaluated",
    },
}

#: Stand-ins for the real repository's files, one byte string per name the fetch step
#: insists on (``jevbert.fetch.SOURCE_FILES``).
FILES = {name: f"content of {name}".encode() for name in SOURCE_FILES}


def write_manifest(directory: Path, **overrides: Any) -> Path:
    payload = json.loads(json.dumps(MANIFEST_TEMPLATE))
    payload.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_weights(directory: Path, files: dict[str, bytes] | None = None) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = FILES if files is None else files
    for name, content in payload.items():
        (directory / name).write_bytes(content)
    return {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()}


class _RecordingDownloader:
    """Stands in for ``snapshot_download``; writes the files it is told to allow."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._files = FILES if files is None else files

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        allowed = set(kwargs["allow_patterns"])
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self._files.items():
            if name in allowed:
                (destination / name).write_bytes(content)
        return str(destination)


class TestHashing:
    def test_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        path.write_bytes(b"jevbert")
        assert sha256_file(path) == hashlib.sha256(b"jevbert").hexdigest()

    def test_hash_files_reports_every_name(self, tmp_path: Path) -> None:
        expected = write_weights(tmp_path)
        assert hash_files(tmp_path, FILES) == expected

    def test_hash_files_refuses_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FetchError):
            hash_files(tmp_path, ["absent.json"])


class TestMismatchDetection:
    def test_identical_content_has_no_mismatch(self, tmp_path: Path) -> None:
        expected = write_weights(tmp_path)
        assert mismatched_files(tmp_path, expected) == []

    def test_a_changed_byte_is_a_mismatch(self, tmp_path: Path) -> None:
        expected = write_weights(tmp_path)
        (tmp_path / "config.json").write_bytes(b'{"a": 2}')
        assert mismatched_files(tmp_path, expected) == ["config.json"]

    def test_a_missing_file_is_a_mismatch(self, tmp_path: Path) -> None:
        expected = write_weights(tmp_path)
        (tmp_path / "model.safetensors").unlink()
        assert mismatched_files(tmp_path, expected) == ["model.safetensors"]


class TestRecordSourceHashes:
    def test_an_empty_file_map_is_filled_in(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path)
        record_source_hashes(path, {"config.json": "ab" * 32})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["source_model"]["files"] == {"config.json": "ab" * 32}

    def test_the_rest_of_the_manifest_is_untouched(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path)
        before = json.loads(path.read_text(encoding="utf-8"))
        record_source_hashes(path, {"config.json": "cd" * 32})
        after = json.loads(path.read_text(encoding="utf-8"))
        before["source_model"]["files"] = after["source_model"]["files"]
        assert after == before

    def test_the_result_is_still_a_valid_manifest(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path)
        record_source_hashes(path, {"config.json": "ef" * 32})
        manifest, digest = read_manifest(path)
        assert manifest.source_model is not None
        assert manifest.source_model.files == {"config.json": "ef" * 32}
        assert digest.startswith("sha256:")

    def test_recording_the_same_hashes_again_is_a_no_op(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path)
        record_source_hashes(path, {"config.json": "ab" * 32})
        first = path.read_text(encoding="utf-8")
        record_source_hashes(path, {"config.json": "ab" * 32})
        assert path.read_text(encoding="utf-8") == first

    def test_a_conflicting_hash_refuses_to_overwrite(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path)
        record_source_hashes(path, {"config.json": "ab" * 32})
        before = path.read_text(encoding="utf-8")
        with pytest.raises(FetchError):
            record_source_hashes(path, {"config.json": "ba" * 32})
        # The recorded hash is the bundle's identity: a silent rewrite would move the
        # digest under a bundle ID that has already been published (spec 5.7).
        assert path.read_text(encoding="utf-8") == before


class TestFetchModel:
    def test_it_downloads_and_records_hashes(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        models = tmp_path / "models"
        path = write_manifest(manifests)
        downloader = _RecordingDownloader()

        assert fetch_model(manifests, models, download=downloader) == 0

        recorded = json.loads(path.read_text(encoding="utf-8"))["source_model"]["files"]
        expected = {
            name: hashlib.sha256(content).hexdigest() for name, content in FILES.items()
        }
        assert recorded == expected

    def test_it_pins_the_revision_and_refuses_pickled_weights(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        write_manifest(manifests)
        downloader = _RecordingDownloader()
        fetch_model(manifests, tmp_path / "models", download=downloader)

        call = downloader.calls[0]
        assert call["repo_id"] == REPO_ID
        assert call["revision"] == REVISION
        # spec 15.3: a *.bin checkpoint is a pickle, and an allow-list is the only way
        # to be sure one is never written next to the weights the server loads.
        assert all(not pattern.endswith(".bin") for pattern in call["allow_patterns"])
        assert "model.safetensors" in call["allow_patterns"]

    def test_a_second_run_downloads_nothing(self, tmp_path: Path) -> None:
        manifests = tmp_path / "manifests"
        models = tmp_path / "models"
        write_manifest(manifests)
        downloader = _RecordingDownloader()

        fetch_model(manifests, models, download=downloader)
        assert fetch_model(manifests, models, download=downloader) == 0
        assert len(downloader.calls) == 1

    def test_a_tampered_file_is_reported_and_not_silently_refetched(
        self, tmp_path: Path
    ) -> None:
        manifests = tmp_path / "manifests"
        models = tmp_path / "models"
        write_manifest(manifests)
        downloader = _RecordingDownloader()
        fetch_model(manifests, models, download=downloader)

        (model_directory(models) / "config.json").write_bytes(b'{"a": 999}')
        # The manifest already pins this file, so a differing local copy is a
        # supply-chain question, not a cache miss to paper over.
        assert fetch_model(manifests, models, download=downloader) == 4
        assert len(downloader.calls) == 1

    def test_a_missing_manifest_is_an_error(self, tmp_path: Path) -> None:
        assert fetch_model(tmp_path / "manifests", tmp_path / "models") == 4

    def test_it_reports_where_the_files_landed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifests = tmp_path / "manifests"
        models = tmp_path / "models"
        write_manifest(manifests)
        fetch_model(manifests, models, download=_RecordingDownloader())
        out = capsys.readouterr().out
        assert str(model_directory(models)) in out
        assert REVISION in out


class TestModelDirectory:
    def test_the_repo_id_becomes_one_directory_level(self, tmp_path: Path) -> None:
        directory = model_directory(tmp_path)
        assert directory.parent == tmp_path
        assert "/" not in directory.name
