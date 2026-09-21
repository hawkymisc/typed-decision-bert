"""Sending the prepared sample through the official SDK and recording it (BENCHMARK 3, 7).

The servers here are ``httpx2.MockTransport`` handlers standing in for either target:
what is under test is the harness - that both targets get the same bytes, that every
outcome is recorded, that a billable target is not sent to by accident, and that a run
resumes - not either model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bench.config import Experiment, load_experiment
from bench.datasets import prepare_task, read_samples
from bench.runner import BillableTargetRefused, build_body, read_records, run_target

REVISION = "0" * 40
ENV = {"LOCAL_KEY": "local-key", "JEV_KEY": "jev-key"}


def experiment_file(tmp_path: Path) -> Path:
    data = {
        "name": "t",
        "seed": 1,
        "output_dir": "out",
        "client": {"max_retries": 2, "retry_budget_seconds": 5, "timeout_seconds": 5},
        "targets": {
            "local": {"base_url": "http://local.test", "api_key_env": "LOCAL_KEY"},
            "jev": {"base_url": "https://jev.test", "api_key_env": "JEV_KEY", "billable": True},
        },
        "tasks": [
            {
                "id": "news",
                "source": {
                    "kind": "hf_parquet",
                    "repo": "example/news",
                    "revision": REVISION,
                    "files": ["a.parquet"],
                    "label_column": "label",
                    "label_values": {"0": "world", "1": "sports"},
                },
                "state": {"text": {"column": "text"}},
                "instructions": "Which section?",
                "labels": {"world": "World news", "sports": "Sports"},
                "sample": {"per_label": 3},
            }
        ],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    source = tmp_path / "src"
    source.mkdir()
    rows = [{"text": f"article number {i}", "label": i % 2} for i in range(10)]
    pq.write_table(pa.Table.from_pylist(rows), source / "a.parquet")
    loaded = load_experiment(experiment_file(tmp_path))
    prepare_task(
        loaded.tasks[0],
        seed=loaded.seed,
        output_dir=loaded.output_dir,
        fetch=lambda repo, revision, filename: source / filename,
    )
    return loaded


class FakeServer:
    """Answers every Choice with its first candidate, and remembers what it was sent."""

    def __init__(self, model: str = "fake-model", script: list[Any] | None = None) -> None:
        self.model = model
        self.bodies: list[bytes] = []
        self.authorizations: list[str] = []
        self.script = list(script or [])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = request.read()
        self.bodies.append(body)
        self.authorizations.append(request.headers.get("authorization", ""))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if isinstance(step, httpx2.Response):
                return step
        payload = json.loads(body)
        answers = {}
        for question_id, question in payload["questions"].items():
            keys = list(question["criteria"])
            probabilities = dict.fromkeys(keys, 0.0)
            probabilities[keys[0]] = 0.9
            probabilities[keys[1]] = 0.1
            answers[question_id] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": probabilities,
                "confidence": 0.8,
            }
        return httpx2.Response(
            200,
            json={
                "model": self.model,
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
            headers={"x-typesafe-request-id": f"req-{len(self.bodies)}"},
        )


def run(experiment: Experiment, target: str, server: FakeServer, **kwargs: Any):
    return run_target(
        experiment,
        target,
        env=ENV,
        transport=httpx2.MockTransport(server),
        log=lambda line: None,
        **kwargs,
    )


class TestBody:
    def test_one_choice_question_with_the_shared_wording(self, experiment: Experiment) -> None:
        task = experiment.tasks[0]
        example = read_samples(experiment.output_dir, "news")[0]
        body = build_body(experiment, task, example)
        assert body == {
            "state": example.state,
            "model": "jev-latest",
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Which section?",
                    "criteria": {"world": "World news", "sports": "Sports"},
                }
            },
        }


class TestRun:
    def test_every_sample_is_recorded_with_the_wire_answer(self, experiment: Experiment) -> None:
        server = FakeServer()
        summary = run(experiment, "local", server)

        records = read_records(experiment.output_dir, "local", "news")
        assert summary.sent == 6 and len(records) == 6
        first = records[0]
        assert first["status"] == "ok"
        assert first["choice"] == "world"
        assert first["probabilities"] == {"world": 0.9, "sports": 0.1}
        assert first["confidence"] == 0.8
        assert first["response_model"] == "fake-model"
        assert first["request_id"] == "req-1"
        assert first["attempts"] == 1
        assert first["response"]["answers"]["label"]["choice"] == "world"
        assert first["request_sha256"] == hashlib.sha256(server.bodies[0]).hexdigest()
        assert server.authorizations[0] == "Bearer local-key"

    def test_both_targets_are_sent_identical_bytes(self, experiment: Experiment) -> None:
        local, jev = FakeServer(), FakeServer(model="jev-x")
        run(experiment, "local", local)
        run(experiment, "jev", jev, allow_billable=True)
        assert local.bodies == jev.bodies
        assert local.authorizations[0] != jev.authorizations[0]

    def test_a_billable_target_is_not_sent_to_without_permission(
        self, experiment: Experiment
    ) -> None:
        server = FakeServer()
        with pytest.raises(BillableTargetRefused, match="6"):
            run(experiment, "jev", server)
        assert server.bodies == []

    def test_limit_takes_the_same_prefix_for_every_target(self, experiment: Experiment) -> None:
        local, jev = FakeServer(), FakeServer()
        run(experiment, "local", local, limit=2)
        run(experiment, "jev", jev, allow_billable=True, limit=2)
        assert len(local.bodies) == 2
        assert local.bodies == jev.bodies

    def test_a_422_is_recorded_not_raised(self, experiment: Experiment) -> None:
        refusal = httpx2.Response(
            422,
            json={"error": {"code": "context_length_exceeded", "message": "too long"}},
        )
        server = FakeServer(script=[refusal])
        run(experiment, "local", server)
        records = read_records(experiment.output_dir, "local", "news")
        assert records[0]["status"] == "api_error"
        assert records[0]["http_status"] == 422
        assert "too long" in records[0]["error"]
        assert records[0]["choice"] is None
        assert all(record["status"] == "ok" for record in records[1:])

    def test_a_429_is_retried_by_the_sdk_and_the_attempts_are_counted(
        self, experiment: Experiment
    ) -> None:
        busy = httpx2.Response(429, json={"error": {"message": "slow down"}},
                               headers={"retry-after-ms": "1"})
        server = FakeServer(script=[busy])
        run(experiment, "local", server)
        first = read_records(experiment.output_dir, "local", "news")[0]
        assert first["status"] == "ok"
        assert first["attempts"] == 2

    def test_a_connection_failure_is_recorded(self, experiment: Experiment) -> None:
        failures = [httpx2.ConnectError("refused")] * 3
        server = FakeServer(script=failures)
        run(experiment, "local", server)
        first = read_records(experiment.output_dir, "local", "news")[0]
        assert first["status"] == "connection_error"
        assert first["attempts"] == 3

    def test_a_rerun_sends_only_what_has_not_succeeded(self, experiment: Experiment) -> None:
        refusal = httpx2.Response(503, json={"error": {"message": "down"}})
        run(experiment, "local", FakeServer(script=[refusal] * 3))
        again = FakeServer()
        summary = run(experiment, "local", again)
        assert summary.sent == 1
        assert summary.skipped == 5
        latest = {r["sample_id"]: r for r in read_records(experiment.output_dir, "local", "news")}
        assert all(record["status"] == "ok" for record in latest.values())

    def test_concurrency_records_each_sample_once(self, experiment: Experiment) -> None:
        concurrent = experiment.model_copy(
            update={"client": experiment.client.model_copy(update={"concurrency": 4})}
        )
        server = FakeServer()
        run(concurrent, "local", server)
        records = read_records(experiment.output_dir, "local", "news")
        assert len({r["sample_id"] for r in records}) == 6
        sent = {hashlib.sha256(body).hexdigest() for body in server.bodies}
        assert {r["request_sha256"] for r in records} == sent

    def test_an_unknown_target_is_refused(self, experiment: Experiment) -> None:
        with pytest.raises(KeyError):
            run(experiment, "nowhere", FakeServer())
