"""`python -m jevbert fetch-model`: the pinned-revision download (ADR-014, spec 15.3).

This is the only place in the project that reaches the network, and it is deliberately
not the server: the serving path loads from ``models/`` with ``local_files_only=True``
and never resolves a repo ID a caller supplied (spec 12.1).

What the step pins:

* one revision, by commit SHA, never a branch name;
* safetensors only - a ``*.bin`` checkpoint is a pickle, so it is not in the allow-list;
* a SHA-256 per file, written into the manifest. Because the bundle digest is the hash
  of the whole manifest (POC_DESIGN 6.5), a changed weight byte changes the bundle ID.

A hash that is already recorded is never overwritten. A local file that stops matching
the manifest is reported, not re-downloaded: the manifest is the claim the bundle's
identity rests on, and quietly repairing the difference would hide exactly the event
the hashes exist to catch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

#: The zero-shot NLI model behind ``a0-nli-zeroshot-v1`` (spec 7.7, ADR-011). MIT
#: licensed, safetensors, no custom modelling code.
REPO_ID = "MoritzLaurer/bge-m3-zeroshot-v2.0"

#: Pinned commit. A tag or a branch would let the bytes move under a fixed bundle ID.
REVISION = "9abf1c8aaeb82a2447809c20753ed0b106b76652"

MANIFEST_FILENAME = "jevbert-poc-nli-ja-en-0.1.0.json"

#: Exactly the files the backend opens, and nothing else. An allow-list rather than an
#: ignore-list, so a file added to the repo later cannot arrive unnoticed.
SOURCE_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

_CHUNK_BYTES = 1 << 20

#: ``snapshot_download``-shaped callable, so the tests never touch the network.
Downloader = Callable[..., str]


class FetchError(Exception):
    """Raised when the model cannot be fetched or the manifest cannot be trusted."""


def model_directory(models_dir: Path, repo_id: str = REPO_ID) -> Path:
    """Where one repo's files live. The owner is kept, flattened into the name."""
    return models_dir / repo_id.replace("/", "--")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FetchError(f"Cannot read {path}") from exc
    return digest.hexdigest()


def hash_files(directory: Path, filenames: Iterable[str]) -> dict[str, str]:
    """SHA-256 of each named file. A missing file is an error, not an empty entry."""
    hashes: dict[str, str] = {}
    for name in filenames:
        path = directory / name
        if not path.is_file():
            raise FetchError(f"{path} is missing.")
        hashes[name] = sha256_file(path)
    return hashes


def mismatched_files(directory: Path, expected: Mapping[str, str]) -> list[str]:
    """Names whose local bytes are absent or differ from the recorded hash."""
    bad: list[str] = []
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            bad.append(name)
    return bad


def record_source_hashes(manifest_path: Path, hashes: Mapping[str, str]) -> None:
    """Write ``source_model.files``, refusing to change a hash already recorded."""
    payload = _read_manifest_payload(manifest_path)
    source = payload.get("source_model")
    if not isinstance(source, dict):
        raise FetchError(f"{manifest_path} has no source_model to record hashes into.")

    recorded = source.get("files") or {}
    if not isinstance(recorded, dict):
        raise FetchError(f"{manifest_path}: source_model.files must be an object.")

    conflicts = sorted(
        name for name, digest in recorded.items() if name in hashes and hashes[name] != digest
    )
    if conflicts:
        raise FetchError(
            f"{manifest_path} already records different hashes for: {', '.join(conflicts)}. "
            "The manifest was left unchanged. Either the local files are not the pinned "
            "revision, or the bundle needs a new ID (spec 5.7)."
        )
    if recorded == dict(hashes):
        return

    source["files"] = {name: hashes[name] for name in sorted(hashes)}
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def fetch_model(
    manifests_dir: Path,
    models_dir: Path,
    *,
    download: Downloader = snapshot_download,
    manifest_filename: str = MANIFEST_FILENAME,
) -> int:
    """Fetch the pinned revision if it is not already there, then record its hashes.

    Returns a process exit code: 0 on success, 4 on any refusal.
    """
    manifest_path = manifests_dir / manifest_filename
    try:
        payload = _read_manifest_payload(manifest_path)
        source = payload.get("source_model") or {}
        repo_id = source.get("repo", REPO_ID)
        revision = source.get("revision", REVISION)
        recorded: Mapping[str, str] = source.get("files") or {}
        destination = model_directory(models_dir, repo_id)

        if recorded:
            bad = mismatched_files(destination, recorded)
            if not bad:
                print(
                    f"jevbert: {len(recorded)} files already match the manifest in "
                    f"{destination} (revision {revision}). Nothing to download."
                )
                _report_size(destination, recorded)
                return 0
            if destination.exists():
                raise FetchError(
                    f"{destination} exists but these files do not match the manifest: "
                    f"{', '.join(bad)}. Nothing was downloaded and nothing was changed. "
                    "Delete the directory to fetch the pinned revision again."
                )

        print(f"jevbert: downloading {repo_id} at revision {revision} into {destination} ...")
        download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(SOURCE_FILES),
        )
        hashes = hash_files(destination, SOURCE_FILES)
        record_source_hashes(manifest_path, hashes)
    except FetchError as exc:
        print(f"jevbert: {exc}")
        return 4
    except OSError as exc:
        print(f"jevbert: the download failed ({type(exc).__name__}).")
        return 4

    print(f"jevbert: wrote {len(hashes)} file hashes to {manifest_path}.")
    for name in sorted(hashes):
        print(f"  {name}  sha256:{hashes[name]}")
    _report_size(destination, hashes)
    return 0


def _report_size(directory: Path, files: Mapping[str, str]) -> None:
    total = sum((directory / name).stat().st_size for name in files if (directory / name).is_file())
    print(f"jevbert: {directory} holds {len(files)} files, {total / (1024 * 1024):.1f} MiB total.")


def _read_manifest_payload(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FetchError(
            f"Cannot read {path}. The manifest is part of the repository; "
            "fetch-model fills in its hashes but does not create it."
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FetchError(f"{path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise FetchError(f"{path} must contain a JSON object.")
    return payload
