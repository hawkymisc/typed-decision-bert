"""The ``python -m bench`` entry point: exit codes a script can rely on."""

from __future__ import annotations

from pathlib import Path

import yaml

from bench.__main__ import main

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
