"""From a pinned dataset file to the sample both targets are sent (BENCHMARK 2).

``prepare`` is the only step that touches the network or the source data. It writes
the sample as the exact ``state`` objects to send, and a manifest recording where they
came from and the file's SHA-256. The runner reads nothing else, so every target is
sent the same thing, and a sample edited after the fact is refused rather than sent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from bench.config import ArxivSnapshotSource, FieldSpec, HfParquetSource, TaskSpec

#: (repo, revision, filename) -> a local path to that file at that revision.
Fetch = Callable[[str, str, str], Path]

_NEW_STYLE_ARXIV_ID = r"^\d{4}\.\d{4,5}$"
_WHITESPACE = re.compile(r"\s+")


class SampleError(Exception):
    """The sample cannot be drawn, or the one on disk cannot be trusted."""


@dataclass(frozen=True)
class Example:
    sample_id: str
    label: str
    state: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedTask:
    task_id: str
    examples: list[Example]
    path: Path
    manifest: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.examples)


def hf_fetch(repo: str, revision: str, filename: str) -> Path:
    """Download (or reuse from the HF cache) one dataset file at a pinned commit."""
    return Path(
        hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision)
    )


def render_field(value: Any, spec: FieldSpec) -> str:
    """Pattern removal, whitespace, then truncation - always in that order."""
    text = "" if value is None else str(value)
    for pattern in spec.remove_patterns:
        text = re.sub(pattern, "", text)
    if spec.collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    if spec.max_chars is not None:
        text = text[: spec.max_chars].rstrip()
    return text


def load_examples(task: TaskSpec, paths: Sequence[Path]) -> list[Example]:
    """Every row of the source that can be a sample, in file and row order."""
    source = task.source
    if len(paths) != len(source.files):
        raise SampleError(f"{task.id}: {len(source.files)} files declared, {len(paths)} given")
    if isinstance(source, HfParquetSource):
        return list(_hf_parquet_examples(task, source, paths))
    return list(_arxiv_examples(task, source, paths))


def _render_state(task: TaskSpec, row: dict[str, Any]) -> dict[str, str] | None:
    state = {key: render_field(row.get(spec.column), spec) for key, spec in task.state.items()}
    return state if all(state.values()) else None


def _hf_parquet_examples(
    task: TaskSpec, source: HfParquetSource, paths: Sequence[Path]
) -> Iterable[Example]:
    columns = sorted({source.label_column, *(spec.column for spec in task.state.values())})
    for filename, path in zip(source.files, paths, strict=True):
        rows = pq.read_table(path, columns=columns).to_pylist()
        for index, row in enumerate(rows):
            label = source.label_values.get(str(row[source.label_column]))
            if label is None:
                continue
            state = _render_state(task, row)
            if state is None:
                continue
            yield Example(f"{filename}#{index}", label, state, {})


def _arxiv_examples(
    task: TaskSpec, source: ArxivSnapshotSource, paths: Sequence[Path]
) -> Iterable[Example]:
    columns = sorted({"id", "categories", *(spec.column for spec in task.state.values())})
    for path in paths:
        table = pq.read_table(path, columns=columns)
        ids = table.column("id")
        recent = pc.and_(
            pc.match_substring_regex(ids, _NEW_STYLE_ARXIV_ID),
            pc.greater_equal(pc.utf8_slice_codeunits(ids, 0, 4), source.min_yymm),
        )
        for row in table.filter(recent).to_pylist():
            categories = str(row["categories"] or "").split()
            if not categories or categories[0] not in task.labels:
                continue
            state = _render_state(task, row)
            if state is None:
                continue
            yield Example(row["id"], categories[0], state, {"categories": categories})


def sample_key(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def stratified_sample(examples: Sequence[Example], task: TaskSpec, seed: int) -> list[Example]:
    """``per_label`` examples from each label, interleaved label by label.

    Within a label the examples are ranked by ``sha256(seed:sample_id)``, so the draw
    depends on neither row order nor file layout. The result is ordered rank by rank
    over the labels in their declared order, so any prefix (``--limit``) stays balanced.
    """
    by_label: dict[str, list[Example]] = {label: [] for label in task.labels}
    seen: set[str] = set()
    for example in examples:
        if example.sample_id in seen:
            raise SampleError(f"{task.id}: sample id {example.sample_id!r} appears twice")
        seen.add(example.sample_id)
        by_label.setdefault(example.label, []).append(example)

    per_label = task.sample.per_label
    short = {
        label: len(by_label[label]) for label in task.labels if len(by_label[label]) < per_label
    }
    if short:
        raise SampleError(
            f"{task.id}: fewer than {per_label} examples for {short}; lower sample.per_label "
            "or widen the source"
        )
    drawn = {
        label: sorted(by_label[label], key=lambda e: sample_key(seed, e.sample_id))[:per_label]
        for label in task.labels
    }
    return [drawn[label][rank] for rank in range(per_label) for label in task.labels]


def samples_path(output_dir: Path, task_id: str) -> Path:
    return output_dir / "samples" / f"{task_id}.jsonl"


def manifest_path(output_dir: Path, task_id: str) -> Path:
    return output_dir / "samples" / f"{task_id}.manifest.json"


def task_fingerprint(task: TaskSpec) -> str:
    """What the sample depends on. The runner refuses a sample drawn under another one."""
    relevant = task.model_dump(mode="json", include={"source", "state", "labels", "sample"})
    relevant["labels"] = sorted(relevant["labels"])  # descriptions do not change the sample
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def prepare_task(
    task: TaskSpec,
    seed: int,
    output_dir: Path,
    fetch: Fetch = hf_fetch,
    log: Callable[[str], None] = print,
) -> PreparedTask:
    source = task.source
    paths = []
    for filename in source.files:
        log(f"[{task.id}] fetching {source.repo}@{source.revision[:12]}:{filename}")
        paths.append(fetch(source.repo, source.revision, filename))
    examples = load_examples(task, paths)
    eligible: dict[str, int] = {}
    for example in examples:
        eligible[example.label] = eligible.get(example.label, 0) + 1
    log(f"[{task.id}] {len(examples)} eligible rows; drawing {task.sample.per_label} per label")
    sample = stratified_sample(examples, task, seed)
    return write_samples(
        output_dir, task, sample, seed=seed, source_files=list(source.files), eligible=eligible
    )


def write_samples(
    output_dir: Path,
    task: TaskSpec,
    examples: Sequence[Example],
    *,
    seed: int,
    source_files: Sequence[str],
    eligible: dict[str, int] | None = None,
) -> PreparedTask:
    lines = [
        json.dumps(
            {"sample_id": e.sample_id, "label": e.label, "state": e.state, "meta": e.meta},
            ensure_ascii=False,
        )
        + "\n"
        for e in examples
    ]
    content = "".join(lines).encode("utf-8")
    path = samples_path(output_dir, task.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    counts: dict[str, int] = {}
    for example in examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    manifest = {
        "task": task.id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": task.source.model_dump(mode="json"),
        "source_files": list(source_files),
        "seed": seed,
        "per_label": task.sample.per_label,
        "eligible_counts": eligible,
        "counts": counts,
        "count": len(examples),
        "task_fingerprint": task_fingerprint(task),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest_path(output_dir, task.id).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedTask(task.id, list(examples), path, manifest)


def read_manifest(output_dir: Path, task_id: str) -> dict[str, Any]:
    path = manifest_path(output_dir, task_id)
    if not path.is_file():
        raise SampleError(
            f"no sample for task {task_id!r} in {output_dir}; run `python -m bench prepare` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_samples(output_dir: Path, task_id: str) -> list[Example]:
    manifest = read_manifest(output_dir, task_id)
    path = samples_path(output_dir, task_id)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise SampleError(f"{path} is missing; run `python -m bench prepare` again") from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != manifest["sha256"]:
        raise SampleError(
            f"{path} does not match the SHA-256 in its manifest (edited after prepare?); "
            "run `python -m bench prepare` again"
        )
    examples = []
    for line in content.decode("utf-8").splitlines():
        record = json.loads(line)
        examples.append(
            Example(record["sample_id"], record["label"], record["state"], record["meta"])
        )
    return examples
