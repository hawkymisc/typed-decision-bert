"""Sending the prepared sample through the official SDK and recording it (BENCHMARK 3, 7).

The servers here are ``httpx2.MockTransport`` handlers standing in for either target:
what is under test is the harness - that both targets get the same bytes, that every
outcome is recorded, that a billable target is not sent to by accident, and that a run
resumes - not either model.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bench.config import Experiment, ExperimentError, load_experiment
from bench.datasets import SampleError, prepare_task, read_samples
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
            "local": {"base_url": "http://127.0.0.1:8765", "api_key_env": "LOCAL_KEY"},
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

    def __init__(
        self,
        model: str = "fake-model",
        script: list[Any] | None = None,
        request_id: bool = True,
        on_request: Any = None,
    ) -> None:
        self.model = model
        self.request_id = request_id
        self.on_request = on_request
        self.bodies: list[bytes] = []
        self.authorizations: list[str] = []
        self.script = list(script or [])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = request.read()
        self.bodies.append(body)
        self.authorizations.append(request.headers.get("authorization", ""))
        if self.on_request is not None:
            self.on_request(self)
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
            headers=(
                {"x-typesafe-request-id": f"req-{len(self.bodies)}"} if self.request_id else {}
            ),
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

    def test_a_rerun_resends_a_retryable_failure(self, experiment: Experiment) -> None:
        refusal = httpx2.Response(503, json={"error": {"message": "down"}})
        run(experiment, "local", FakeServer(script=[refusal] * 3))
        again = FakeServer()
        summary = run(experiment, "local", again)
        assert summary.sent == 1
        assert summary.skipped == 5
        latest = {r["sample_id"]: r for r in read_records(experiment.output_dir, "local", "news")}
        assert all(record["status"] == "ok" for record in latest.values())

    def test_a_rerun_does_not_resend_a_final_refusal_unless_asked(
        self, experiment: Experiment
    ) -> None:
        # A 422 was answered (and may have been billed); sending it again gets the same
        # answer. Only --retry-failed sends it again.
        refusal = httpx2.Response(422, json={"error": {"message": "too long"}})
        run(experiment, "local", FakeServer(script=[refusal]))
        assert run(experiment, "local", FakeServer()).sent == 0
        assert run(experiment, "local", FakeServer(), retry_failed=True).sent == 1

    def test_concurrency_records_each_sample_once_with_its_own_bytes(
        self, experiment: Experiment
    ) -> None:
        concurrent = experiment.model_copy(
            update={"client": experiment.client.model_copy(update={"concurrency": 4})}
        )
        server = FakeServer()
        summary = run(concurrent, "local", server)
        records = read_records(experiment.output_dir, "local", "news")
        assert summary.sent == 6 and len(records) == 6 and len(server.bodies) == 6
        states = {e.sample_id: e.state for e in read_samples(experiment.output_dir, "news")}
        sent = {hashlib.sha256(body).hexdigest(): json.loads(body) for body in server.bodies}
        for record in records:
            assert sent[record["request_sha256"]]["state"] == states[record["sample_id"]]

    def test_a_missing_request_id_header_does_not_fail_a_good_answer(
        self, experiment: Experiment
    ) -> None:
        run(experiment, "local", FakeServer(request_id=False), limit=1)
        first = read_records(experiment.output_dir, "local", "news")[0]
        assert first["status"] == "ok"
        assert first["request_id"] is None

    def test_an_answer_without_a_choice_is_not_ok(self, experiment: Experiment) -> None:
        empty = httpx2.Response(200, json={"model": "m", "answers": {}, "usage": {}})
        run(experiment, "local", FakeServer(script=[empty]), limit=1)
        first = read_records(experiment.output_dir, "local", "news")[0]
        assert first["status"] != "ok"

    def test_a_body_that_is_not_json_is_kept_as_text(self, experiment: Experiment) -> None:
        broken = httpx2.Response(502, content=b"<html>bad gateway</html>")
        run(experiment, "local", FakeServer(script=[broken] * 3), limit=1)
        first = read_records(experiment.output_dir, "local", "news")[0]
        assert first["status"] == "api_error"
        assert first["response"] == {"unparsed": "<html>bad gateway</html>"}

    def test_limit_sends_the_first_samples_in_order(self, experiment: Experiment) -> None:
        server = FakeServer()
        run(experiment, "local", server, limit=3)
        first_three = [e.state for e in read_samples(experiment.output_dir, "news")[:3]]
        assert [json.loads(body)["state"] for body in server.bodies] == first_three

    def test_a_stop_request_ends_the_run_after_the_request_in_flight(
        self, experiment: Experiment
    ) -> None:
        stop = threading.Event()
        server = FakeServer(on_request=lambda _: stop.set())
        summary = run(experiment, "local", server, stop=stop)
        assert len(server.bodies) == 1
        assert summary.sent == 1
        assert summary.interrupted is True

    def test_an_unknown_target_is_refused(self, experiment: Experiment) -> None:
        with pytest.raises(ExperimentError, match="nowhere"):
            run(experiment, "nowhere", FakeServer())

    def test_an_unknown_task_is_refused(self, experiment: Experiment) -> None:
        with pytest.raises(ExperimentError, match="nosuch"):
            run(experiment, "local", FakeServer(), task_ids=["nosuch"])


class TestConditions:
    """A record counts only under the sample and the question it was made with."""

    def test_changing_the_wording_resends_everything(self, experiment: Experiment) -> None:
        run(experiment, "local", FakeServer())
        task = experiment.tasks[0]
        reworded = experiment.model_copy(
            update={"tasks": [task.model_copy(update={"instructions": "Other?"})]}
        )
        server = FakeServer()
        summary = run(reworded, "local", server)
        assert summary.sent == 6
        assert json.loads(server.bodies[0])["questions"]["label"]["instructions"] == "Other?"

    def test_changing_a_label_description_needs_no_prepare(self, experiment: Experiment) -> None:
        run(experiment, "local", FakeServer())
        task = experiment.tasks[0]
        relabelled = experiment.model_copy(
            update={
                "tasks": [
                    task.model_copy(update={"labels": {"world": "World", "sports": "Sport"}})
                ]
            }
        )
        assert run(relabelled, "local", FakeServer()).sent == 6

    def test_a_changed_sample_size_refuses_until_prepare(self, experiment: Experiment) -> None:
        task = experiment.tasks[0]
        smaller = task.sample.model_copy(update={"per_label": 2})
        resized = experiment.model_copy(
            update={"tasks": [task.model_copy(update={"sample": smaller})]}
        )
        server = FakeServer()
        with pytest.raises(SampleError, match="prepare"):
            run(resized, "local", server)
        assert server.bodies == []

    def test_records_of_an_older_sample_are_ignored(self, experiment: Experiment) -> None:
        run(experiment, "local", FakeServer())
        path = experiment.output_dir / "runs" / "local" / "news.jsonl"
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        for record in records:
            record["sample_sha256"] = "0" * 64
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        assert run(experiment, "local", FakeServer()).sent == 6

    def test_a_truncated_line_is_named_not_a_traceback(self, experiment: Experiment) -> None:
        run(experiment, "local", FakeServer(), limit=1)
        path = experiment.output_dir / "runs" / "local" / "news.jsonl"
        path.write_text(path.read_text("utf-8") + '{"sample_id": "x", "sta', encoding="utf-8")
        with pytest.raises(SampleError, match="news.jsonl:2"):
            run(experiment, "local", FakeServer())
