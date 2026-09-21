"""The comparison is computed on the samples both targets answered (BENCHMARK 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from bench.config import Experiment, load_experiment
from bench.datasets import Example, write_samples
from bench.report import build_report, render_markdown

REVISION = "0" * 40


def experiment(tmp_path: Path) -> Experiment:
    data = {
        "name": "t",
        "seed": 1,
        "output_dir": "out",
        "targets": {
            "local": {"base_url": "http://local.test", "api_key_env": "L"},
            "jev": {"base_url": "https://jev.test", "api_key_env": "J", "billable": True},
        },
        "tasks": [
            {
                "id": "arxiv",
                "source": {
                    "kind": "arxiv_snapshot",
                    "repo": "librarian-bots/arxiv-metadata-snapshot",
                    "revision": REVISION,
                    "files": ["x.parquet"],
                    "min_yymm": "2606",
                },
                "state": {"title": {"column": "title"}},
                "instructions": "Primary?",
                "labels": {"cs.CL": "CL", "cs.CV": "CV", "cs.LG": "LG"},
                "sample": {"per_label": 1},
            }
        ],
    }
    path = tmp_path / "e.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_experiment(path)


def record(sample_id: str, choice: str | None, *, status: str = "ok", sha: str = "s",
           model: str = "m", latency: float = 10.0) -> dict[str, Any]:
    probabilities = None
    if choice is not None:
        probabilities = {"cs.CL": 0.1, "cs.CV": 0.1, "cs.LG": 0.1}
        probabilities[choice] = 0.8
    return {
        "sample_id": sample_id,
        "status": status,
        "choice": choice,
        "probabilities": probabilities,
        "response_model": model if status == "ok" else None,
        "request_sha256": sha + sample_id,
        "latency_ms": latency,
        "attempts": 1,
    }


def write_run(exp: Experiment, target: str, records: list[dict[str, Any]]) -> None:
    path = exp.output_dir / "runs" / target / "arxiv.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def setup(tmp_path: Path) -> Experiment:
    exp = experiment(tmp_path)
    examples = [
        Example("p1", "cs.CL", {"title": "a"}, {"categories": ["cs.CL"]}),
        Example("p2", "cs.CV", {"title": "b"}, {"categories": ["cs.CV", "cs.LG"]}),
        Example("p3", "cs.LG", {"title": "c"}, {"categories": ["cs.LG"]}),
        Example("p4", "cs.CL", {"title": "d"}, {"categories": ["cs.CL"]}),
    ]
    write_samples(exp.output_dir, exp.tasks[0], examples, seed=1, source_files=[])
    return exp


class TestReport:
    def test_per_target_numbers_and_the_paired_intersection(self, tmp_path: Path) -> None:
        exp = setup(tmp_path)
        write_run(exp, "local", [
            record("p1", "cs.CL"), record("p2", "cs.LG"), record("p3", "cs.LG"),
            record("p4", None, status="api_error"),
        ])
        write_run(exp, "jev", [
            record("p1", "cs.CL", model="jev-1"), record("p2", "cs.CV", model="jev-1"),
            record("p3", "cs.CV", model="jev-2"), record("p4", "cs.CL", model="jev-2"),
        ])
        report = build_report(exp)
        task = report["tasks"]["arxiv"]

        local = task["targets"]["local"]
        assert local["samples"] == 4 and local["ok"] == 3
        assert local["coverage"] == 0.75
        assert local["accuracy"] == 2 / 3
        # p2 predicted cs.LG, which is cross-listed: a hit on any category.
        assert local["any_category_hit"] == 1.0
        assert task["targets"]["jev"]["response_models"] == {"jev-1": 2, "jev-2": 2}

        pair = task["pairs"]["local vs jev"]
        assert pair["n"] == 3  # p4 failed on local
        assert pair["accuracy_a"] == 2 / 3
        assert pair["accuracy_b"] == 2 / 3
        assert pair["identical_requests"] == 3

    def test_a_request_mismatch_is_counted(self, tmp_path: Path) -> None:
        exp = setup(tmp_path)
        write_run(exp, "local", [record(s, "cs.CL") for s in ("p1", "p2", "p3", "p4")])
        write_run(exp, "jev", [record(s, "cs.CL", sha="other") for s in ("p1", "p2", "p3", "p4")])
        pair = build_report(exp)["tasks"]["arxiv"]["pairs"]["local vs jev"]
        assert pair["identical_requests"] == 0

    def test_the_latest_record_of_a_sample_wins(self, tmp_path: Path) -> None:
        exp = setup(tmp_path)
        write_run(exp, "local", [
            record("p1", None, status="connection_error"), record("p1", "cs.CL"),
        ])
        local = build_report(exp)["tasks"]["arxiv"]["targets"]["local"]
        assert local["ok"] == 1

    def test_one_target_alone_reports_without_a_pair(self, tmp_path: Path) -> None:
        exp = setup(tmp_path)
        write_run(exp, "local", [record("p1", "cs.CL")])
        report = build_report(exp)
        assert report["tasks"]["arxiv"]["pairs"] == {}
        assert "jev" not in report["tasks"]["arxiv"]["targets"]

    def test_markdown_carries_the_caveats_pointer_and_the_table(self, tmp_path: Path) -> None:
        exp = setup(tmp_path)
        write_run(exp, "local", [record("p1", "cs.CL")])
        text = render_markdown(build_report(exp))
        assert "BENCHMARK.md" in text
        assert "| local |" in text
