"""The ``python -m bench`` entry point: exit codes a script can rely on."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bench.__main__ import main
from bench.config import load_experiment
from bench.datasets import prepare_task

REVISION = "0" * 40


def experiment(tmp_path: Path) -> Path:
    data = {
        "name": "t",
        "seed": 1,
        "output_dir": "out",
        "targets": {
            "jev": {"base_url": "https://jev.test", "api_key_env": "J", "billable": True},
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
                "sample": {"per_label": 1},
            }
        ],
    }
    path = tmp_path / "e.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_a_broken_experiment_file_exits_2(tmp_path: Path, capsys) -> None:
    path = tmp_path / "e.yaml"
    path.write_text("name: t\n", encoding="utf-8")
    assert main(["plan", str(path)]) == 2
    assert "e.yaml" in capsys.readouterr().err


def test_running_before_prepare_exits_2_and_says_prepare(tmp_path: Path, capsys) -> None:
    assert main(["run", str(experiment(tmp_path)), "--target", "jev", "--allow-billable"]) == 2
    assert "prepare" in capsys.readouterr().err


def test_plan_before_prepare_exits_2(tmp_path: Path, capsys) -> None:
    assert main(["plan", str(experiment(tmp_path))]) == 2


def prepared(tmp_path: Path, billable: bool = True) -> Path:
    path = experiment(tmp_path)
    data = yaml.safe_load(path.read_text("utf-8"))
    if not billable:
        data["targets"] = {"local": {"base_url": "http://127.0.0.1:1", "api_key_env": "J"}}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"text": f"t{i}", "label": i % 2} for i in range(4)]),
        source / "a.parquet",
    )
    loaded = load_experiment(path)
    prepare_task(loaded.tasks[0], loaded.seed, loaded.output_dir,
                 fetch=lambda r, v, f: source / f, log=lambda line: None)
    return path


def test_a_billable_target_without_permission_exits_3(tmp_path: Path, capsys) -> None:
    assert main(["run", str(prepared(tmp_path)), "--target", "jev"]) == 3
    assert "--allow-billable" in capsys.readouterr().err


def test_a_missing_key_exits_2_before_sending(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("J", raising=False)
    path = prepared(tmp_path)
    assert main(["run", str(path), "--target", "jev", "--allow-billable"]) == 2
    assert "J" in capsys.readouterr().err


def test_failed_samples_exit_1(tmp_path: Path, monkeypatch) -> None:
    # Port 1 on loopback refuses the connection: every sample is a connection_error.
    monkeypatch.setenv("J", "k")
    path = prepared(tmp_path, billable=False)
    data = yaml.safe_load(path.read_text("utf-8"))
    data["client"] = {"max_retries": 0, "timeout_seconds": 2}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert main(["run", str(path), "--target", "local"]) == 1


def test_an_unknown_task_exits_2(tmp_path: Path, capsys) -> None:
    path = prepared(tmp_path)
    assert main(["run", str(path), "--target", "jev", "--task", "nosuch"]) == 2
    assert main(["prepare", str(path), "--task", "nosuch"]) == 2
    assert "nosuch" in capsys.readouterr().err


def test_a_limit_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["run", str(prepared(tmp_path)), "--target", "jev", "--limit", "0"])


def test_plan_prints_the_first_body_and_sends_nothing(tmp_path: Path, capsys) -> None:
    assert main(["plan", str(prepared(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert '"instructions": "Which section?"' in out
    assert "news -> jev (billable): 2 to send" in out


def test_report_writes_json_and_markdown(tmp_path: Path) -> None:
    path = prepared(tmp_path)
    assert main(["report", str(path)]) == 0
    out = tmp_path / "out"
    assert json.loads((out / "report.json").read_text("utf-8"))["experiment"] == "t"
    assert (out / "report.md").is_file()
