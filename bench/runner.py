"""Send the prepared sample through the official SDK, and record every outcome (BENCHMARK 3).

Both targets go through ``typesafe_sdk.TypeSafeClient`` with the same retry policy and
timeout; only the base URL and the key differ. A recording transport under the SDK
keeps, per attempt, the SHA-256 of the body bytes the SDK actually sent and the raw
response, so the report can prove the two targets were sent the same bytes and can
read the wire answer rather than the SDK's typed restoration of it.

One JSON line per sample per run is appended to ``runs/<target>/<task>.jsonl``. A rerun
sends only the samples whose latest record is not ``ok``, so a run that stopped part
way resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeClient,
    TypeSafeError,
)

from bench.config import Experiment, TaskSpec, resolve_api_key
from bench.datasets import Example, SampleError, read_manifest, read_samples, task_fingerprint

#: The error text kept per record; enough to tell 422 reasons apart.
_ERROR_CHARS = 500


class BillableTargetRefused(Exception):
    """A billable target was asked to run without ``--allow-billable``."""


@dataclass
class Attempt:
    request_sha256: str
    status_code: int | None
    response_body: bytes | None
    elapsed_ms: float


class RecordingTransport(httpx2.BaseTransport):
    """Wraps the real transport and records each attempt on the calling thread.

    The SDK's sync client runs the transport on the thread that called it, so a
    thread-local list is enough to attribute attempts to the sample being sent.
    """

    def __init__(self, inner: httpx2.BaseTransport) -> None:
        self._inner = inner
        self._local = threading.local()

    def begin(self) -> list[Attempt]:
        self._local.attempts = []
        return self._local.attempts

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        body = request.read()
        attempts: list[Attempt] = getattr(self._local, "attempts", [])
        attempt = Attempt(hashlib.sha256(body).hexdigest(), None, None, 0.0)
        attempts.append(attempt)
        started = time.perf_counter()
        try:
            response = self._inner.handle_request(request)
            content = response.read()
        finally:
            attempt.elapsed_ms = (time.perf_counter() - started) * 1000
        attempt.status_code = response.status_code
        attempt.response_body = content
        return response

    def close(self) -> None:
        self._inner.close()


@dataclass
class RunSummary:
    target: str
    sent: int = 0
    skipped: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


def build_body(experiment: Experiment, task: TaskSpec, example: Example) -> dict[str, Any]:
    """The wire body, in the key order the SDK writes it (state, model, questions)."""
    return {
        "state": example.state,
        "model": experiment.model,
        "questions": {experiment.question_id: _question(task)},
    }


def _question(task: TaskSpec) -> dict[str, Any]:
    return {"type": "choice", "instructions": task.instructions, "criteria": dict(task.labels)}


def runs_path(output_dir: Path, target: str, task_id: str) -> Path:
    return output_dir / "runs" / target / f"{task_id}.jsonl"


def read_records(output_dir: Path, target: str, task_id: str) -> list[dict[str, Any]]:
    path = runs_path(output_dir, target, task_id)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def latest_records(output_dir: Path, target: str, task_id: str) -> dict[str, dict[str, Any]]:
    """The last record of each sample: a retried sample is judged by its latest outcome."""
    return {record["sample_id"]: record for record in read_records(output_dir, target, task_id)}


def pending_samples(
    experiment: Experiment, target: str, task: TaskSpec, limit: int | None
) -> tuple[list[Example], int]:
    """The samples still to send for one task, and how many are already done."""
    manifest = read_manifest(experiment.output_dir, task.id)
    if manifest.get("task_fingerprint") != task_fingerprint(task):
        raise SampleError(
            f"the sample for {task.id!r} was drawn under a different source, state, label set "
            "or sample size than the experiment file now says; run `python -m bench prepare` "
            "again (and discard the runs made on the old sample)"
        )
    examples = read_samples(experiment.output_dir, task.id)
    if limit is not None:
        examples = examples[:limit]
    done = {
        sample_id
        for sample_id, record in latest_records(experiment.output_dir, target, task.id).items()
        if record["status"] == "ok"
    }
    todo = [example for example in examples if example.sample_id not in done]
    return todo, len(examples) - len(todo)


def run_target(
    experiment: Experiment,
    target_name: str,
    *,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    allow_billable: bool = False,
    env: Mapping[str, str] | None = None,
    transport: httpx2.BaseTransport | None = None,
    log: Callable[[str], None] = print,
) -> RunSummary:
    target = experiment.targets[target_name]
    tasks = [experiment.task(task_id) for task_id in task_ids] if task_ids else experiment.tasks

    plan = [(task, *pending_samples(experiment, target_name, task, limit)) for task in tasks]
    to_send = sum(len(todo) for _, todo, _ in plan)
    if target.billable and not allow_billable and to_send:
        raise BillableTargetRefused(
            f"{target_name} is billable and {to_send} requests would be sent; "
            "pass --allow-billable to send them"
        )
    api_key = resolve_api_key(target, env)

    summary = RunSummary(target_name)
    for task, todo, done in plan:
        summary.skipped += done
        log(f"[{target_name}/{task.id}] {len(todo)} to send, {done} already ok")
        if not todo:
            continue
        path = runs_path(experiment.output_dir, target_name, task.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _send_all(experiment, target_name, api_key, task, todo, path, transport, summary, log)
    return summary


def _send_all(
    experiment: Experiment,
    target_name: str,
    api_key: str,
    task: TaskSpec,
    todo: list[Example],
    path: Path,
    transport: httpx2.BaseTransport | None,
    summary: RunSummary,
    log: Callable[[str], None],
) -> None:
    work: queue.Queue[Example] = queue.Queue()
    for example in todo:
        work.put(example)
    write_lock = threading.Lock()
    errors: list[BaseException] = []
    progress = {"count": 0}

    recorder = RecordingTransport(transport or httpx2.HTTPTransport())
    target = experiment.targets[target_name]

    def worker() -> None:
        client = TypeSafeClient(
            base_url=target.base_url,
            api_key=api_key,
            model=experiment.model,
            retry=RetryPolicy(
                max_retries=experiment.client.max_retries,
                timeout=experiment.client.retry_budget_seconds,
            ),
            timeout=experiment.client.timeout_seconds,
            http_client=httpx2.Client(
                transport=_Unclosed(recorder), timeout=experiment.client.timeout_seconds
            ),
        )
        with client:
            while True:
                try:
                    example = work.get_nowait()
                except queue.Empty:
                    return
                record = _send_one(experiment, target_name, task, example, client, recorder)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with write_lock:
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(line)
                    summary.sent += 1
                    summary.by_status[record["status"]] = (
                        summary.by_status.get(record["status"], 0) + 1
                    )
                    progress["count"] += 1
                    if progress["count"] % 20 == 0 or progress["count"] == len(todo):
                        log(f"[{target_name}/{task.id}] {progress['count']}/{len(todo)}")

    def guarded() -> None:
        try:
            worker()
        except BaseException as exc:  # recorded and re-raised on the calling thread below
            errors.append(exc)

    threads = [
        threading.Thread(target=guarded, name=f"bench-{target_name}-{index}")
        for index in range(min(experiment.client.concurrency, len(todo)))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recorder.close()
    if errors:
        raise RuntimeError(f"a worker for {target_name}/{task.id} failed") from errors[0]


class _Unclosed(httpx2.BaseTransport):
    """Lets several per-thread clients share one recorder without closing it early."""

    def __init__(self, inner: httpx2.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        return self._inner.handle_request(request)

    def close(self) -> None:
        """The recorder is closed once, by ``_send_all``, after every worker is done."""


def _send_one(
    experiment: Experiment,
    target_name: str,
    task: TaskSpec,
    example: Example,
    client: TypeSafeClient,
    recorder: RecordingTransport,
) -> dict[str, Any]:
    body = build_body(experiment, task, example)
    attempts = recorder.begin()
    record: dict[str, Any] = {
        "sample_id": example.sample_id,
        "task": task.id,
        "target": target_name,
        "gold": example.label,
        "started_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "status": "ok",
        "http_status": None,
        "error": None,
        "choice": None,
        "probabilities": None,
        "confidence": None,
        "response_model": None,
        "request_id": None,
        "usage": None,
    }
    started = time.perf_counter()
    try:
        response = client.system_one(
            state=body["state"], questions=body["questions"], model=body["model"]
        )
        record["request_id"] = response.request_id
    except TypeSafeAPIResponseValidationError as exc:
        record.update(status="response_invalid", error=_error_text(exc))
    except TypeSafeAPIError as exc:
        record.update(status="api_error", http_status=exc.status, error=_error_text(exc))
    except TypeSafeAPIConnectionError as exc:
        record.update(status="connection_error", error=_error_text(exc))
    except TypeSafeError as exc:
        record.update(status="sdk_error", error=_error_text(exc))
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    record["attempts"] = len(attempts)
    record["request_sha256"] = attempts[0].request_sha256 if attempts else None
    last = attempts[-1] if attempts else None
    record["last_attempt_ms"] = last.elapsed_ms if last else None

    wire = _parse(last.response_body) if last and last.response_body else None
    record["response"] = wire
    if record["status"] == "ok":
        record["http_status"] = last.status_code if last else None
        answer = (wire or {}).get("answers", {}).get(experiment.question_id, {})
        record["choice"] = answer.get("choice")
        record["probabilities"] = answer.get("probabilities")
        record["confidence"] = answer.get("confidence")
        record["response_model"] = (wire or {}).get("model")
        record["usage"] = (wire or {}).get("usage")
    return record


def _parse(content: bytes) -> Any:
    try:
        return json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return {"unparsed": content[:_ERROR_CHARS].decode("utf-8", errors="replace")}


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]
