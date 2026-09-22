"""The experiment file: one YAML / JSON / TOML file is the whole experiment (BENCHMARK 1).

Everything that the two targets must share - the data, the wording of the question,
the model name, the client's retry and timeout - lives here once, and a target entry
can only say *where* to send and *with which key*. That is what keeps a comparison
apple-to-apple by construction rather than by care (BENCHMARK 4).

Unknown keys are refused, not ignored: a misspelt ``max_retry`` that silently fell back
to the default would change the experiment without anyone noticing.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: The SDK's own default model; a caller written against Jev sends this.
DEFAULT_MODEL = "jev-latest"

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_YYMM = re.compile(r"^\d{4}$")
_NAME = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"

#: Hosts that are this machine. Anything else is treated as someone else's service.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ExperimentError(Exception):
    """The experiment file cannot be used as written."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldSpec(_Strict):
    """How one key of the ``state`` object is made from a source column."""

    column: str
    max_chars: Annotated[int, Field(gt=0)] | None = None
    remove_patterns: tuple[str, ...] = ()
    collapse_whitespace: bool = True

    @field_validator("remove_patterns")
    @classmethod
    def _patterns_compile(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"remove_patterns: {pattern!r} is not a regex ({exc})") from exc
        return patterns


class _SourceBase(_Strict):
    repo: str
    revision: str
    files: tuple[str, ...] = Field(min_length=1)

    @field_validator("revision")
    @classmethod
    def _revision_is_a_commit(cls, revision: str) -> str:
        if not _COMMIT_SHA.match(revision):
            raise ValueError(
                "revision must be a 40-character commit SHA, so that the sample does not "
                "change when the dataset does"
            )
        return revision


class HfParquetSource(_SourceBase):
    """Rows of parquet files; the label is a column mapped through ``label_values``."""

    kind: Literal["hf_parquet"]
    label_column: str
    #: Raw label value (as text) -> label key. Rows whose value is absent are dropped.
    label_values: dict[str, str]


class ArxivSnapshotSource(_SourceBase):
    """The arXiv metadata snapshot; the label is the first (primary) category."""

    kind: Literal["arxiv_snapshot"]
    #: Keep new-style IDs whose ``YYMM`` prefix is at least this (BENCHMARK 2.1).
    min_yymm: str

    @field_validator("min_yymm")
    @classmethod
    def _yymm(cls, value: str) -> str:
        if not _YYMM.match(value):
            raise ValueError("min_yymm must be four digits, YYMM (e.g. '2606')")
        return value


Source = Annotated[HfParquetSource | ArxivSnapshotSource, Field(discriminator="kind")]


class SampleSpec(_Strict):
    per_label: Annotated[int, Field(gt=0)]


class TaskSpec(_Strict):
    """One dataset turned into one Choice question per sample."""

    id: Annotated[str, Field(pattern=_NAME)]
    source: Source
    #: ``state`` key -> how to make it. The keys are what both targets see.
    state: dict[str, FieldSpec] = Field(min_length=1)
    instructions: str
    #: Choice candidates: key -> description (the ``criteria`` of the question).
    labels: dict[str, str]
    sample: SampleSpec

    @field_validator("labels")
    @classmethod
    def _choice_size(cls, labels: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(labels) <= 255:
            raise ValueError(f"labels: a Choice needs 2 to 255 candidates, got {len(labels)}")
        return labels

    @model_validator(mode="after")
    def _label_values_are_labels(self) -> TaskSpec:
        if isinstance(self.source, HfParquetSource):
            unknown = sorted(set(self.source.label_values.values()) - set(self.labels))
            if unknown:
                raise ValueError(f"label_values name labels that are not declared: {unknown}")
        return self


class TargetSpec(_Strict):
    """Where to send, and with which key. Nothing about *what* is sent."""

    base_url: str
    api_key_env: str
    #: A ``KEY=VALUE`` file read when the environment does not hold ``api_key_env``.
    dotenv: Path | None = None
    #: A billable target is sent to only with ``--allow-billable`` (BENCHMARK 6). Left
    #: out, a target is billable unless its host is this machine: forgetting the flag
    #: must not be what lets paid requests through.
    billable: bool | None = None

    @field_validator("base_url")
    @classmethod
    def _https_off_this_machine(cls, base_url: str) -> str:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"base_url must be an http(s) URL with a host, got {base_url!r}")
        if parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
            raise ValueError(
                "base_url must use https for a host other than this machine; "
                "the API key would otherwise travel in clear text"
            )
        return base_url

    @property
    def is_local(self) -> bool:
        return urlsplit(self.base_url).hostname in LOOPBACK_HOSTS

    @property
    def is_billable(self) -> bool:
        return (not self.is_local) if self.billable is None else self.billable


class ClientSpec(_Strict):
    """The official SDK client's settings, shared by every target."""

    timeout_seconds: Annotated[float, Field(gt=0)] = 60.0
    max_retries: Annotated[int, Field(ge=0)] = 2
    #: Total retry budget per call (the SDK's ``RetryPolicy.timeout``).
    retry_budget_seconds: Annotated[float, Field(gt=0)] = 180.0
    concurrency: Annotated[int, Field(ge=1, le=64)] = 1


class Experiment(_Strict):
    name: Annotated[str, Field(pattern=_NAME)]
    output_dir: Path
    seed: int
    model: str = DEFAULT_MODEL
    question_id: str = "label"
    client: ClientSpec = ClientSpec()
    targets: dict[str, TargetSpec] = Field(min_length=1)
    tasks: list[TaskSpec] = Field(min_length=1)

    @field_validator("targets")
    @classmethod
    def _target_names(cls, targets: dict[str, TargetSpec]) -> dict[str, TargetSpec]:
        for name in targets:
            if not re.match(_NAME, name) or name in (".", ".."):
                raise ValueError(
                    f"target name {name!r} must match {_NAME}; it becomes a directory name"
                )
        return targets

    @model_validator(mode="after")
    def _unique_task_ids(self) -> Experiment:
        seen: set[str] = set()
        for task in self.tasks:
            if task.id in seen:
                raise ValueError(f"task id {task.id!r} appears more than once")
            seen.add(task.id)
        return self

    def task(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.id == task_id:
                return task
        known = ", ".join(task.id for task in self.tasks)
        raise ExperimentError(f"unknown task {task_id!r}; the file declares {known}")

    def target(self, name: str) -> TargetSpec:
        if name not in self.targets:
            known = ", ".join(self.targets)
            raise ExperimentError(f"unknown target {name!r}; the file declares {known}")
        return self.targets[name]


def load_experiment(path: str | os.PathLike[str]) -> Experiment:
    """Read an experiment file; relative paths in it are relative to the file."""
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(f"cannot read {path}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data: Any = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            data = tomllib.loads(text)
        else:
            raise ExperimentError(
                f"{path}: unknown extension {suffix!r}; use .yaml, .yml, .json or .toml"
            )
    except (yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentError(f"{path} does not parse as {suffix[1:]}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperimentError(f"{path} must contain a mapping at the top level")

    base = path.parent
    data = dict(data)
    name = data.get("name")
    data["output_dir"] = (base / data.get("output_dir", f"bench_runs/{name}")).resolve()
    targets = data.get("targets")
    if isinstance(targets, dict):
        data["targets"] = {
            key: _resolve_dotenv(value, base) for key, value in targets.items()
        }

    try:
        return Experiment.model_validate(data)
    except ValidationError as exc:
        raise ExperimentError(f"{path}:\n{_describe(exc)}") from exc


def _resolve_dotenv(target: Any, base: Path) -> Any:
    if isinstance(target, dict) and target.get("dotenv") is not None:
        return {**target, "dotenv": (base / target["dotenv"]).resolve()}
    return target


def _describe(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def resolve_api_key(target: TargetSpec, env: Mapping[str, str] | None = None) -> str:
    """The first key in ``api_key_env``, from the environment or the target's dotenv.

    A comma separated value (the local server's ``JEVBERT_API_KEYS``) yields its first
    key. The error names the variable, never a value.
    """
    env = os.environ if env is None else env
    raw = env.get(target.api_key_env, "")
    if not raw.strip() and target.dotenv is not None and target.dotenv.is_file():
        raw = _read_dotenv(target.dotenv).get(target.api_key_env, "")
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if not keys:
        where = f" or in {target.dotenv}" if target.dotenv is not None else ""
        raise ExperimentError(f"no API key: set {target.api_key_env} in the environment{where}")
    return keys[0]


def _read_dotenv(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
