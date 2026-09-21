"""Experiment files: one YAML / JSON / TOML file is the whole experiment (BENCHMARK 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bench.config import ExperimentError, load_experiment, resolve_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = sorted((REPO_ROOT / "bench" / "experiments").glob("*.yaml"))

REVISION = "0" * 40


def minimal() -> dict:
    return {
        "name": "t",
        "seed": 1,
        "targets": {
            "local": {"base_url": "http://127.0.0.1:8765", "api_key_env": "LOCAL_KEY"},
            "jev": {
                "base_url": "https://api.typesafe.ai",
                "api_key_env": "JEV_KEY",
                "billable": True,
            },
        },
        "tasks": [
            {
                "id": "news",
                "source": {
                    "kind": "hf_parquet",
                    "repo": "fancyzhx/ag_news",
                    "revision": REVISION,
                    "files": ["data/test-00000-of-00001.parquet"],
                    "label_column": "label",
                    "label_values": {"0": "world", "1": "sports"},
                },
                "state": {"text": {"column": "text", "max_chars": 100}},
                "instructions": "Which section?",
                "labels": {"world": "World news", "sports": "Sports"},
                "sample": {"per_label": 3},
            }
        ],
    }


def write(tmp_path: Path, data: dict, suffix: str = ".yaml") -> Path:
    path = tmp_path / f"experiment{suffix}"
    if suffix in (".yaml", ".yml"):
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        raise AssertionError(suffix)
    return path


class TestLoading:
    def test_defaults_are_the_shared_conditions(self, tmp_path: Path) -> None:
        experiment = load_experiment(write(tmp_path, minimal()))
        assert experiment.model == "jev-latest"
        assert experiment.question_id == "label"
        assert experiment.client.max_retries == 2
        assert experiment.client.concurrency == 1
        assert experiment.client.timeout_seconds == 60.0
        assert experiment.targets["local"].billable is False
        assert experiment.targets["jev"].billable is True

    def test_output_dir_is_relative_to_the_file_and_defaults_by_name(self, tmp_path: Path) -> None:
        experiment = load_experiment(write(tmp_path, minimal()))
        assert experiment.output_dir == (tmp_path / "bench_runs" / "t").resolve()

        data = minimal() | {"output_dir": "out/here"}
        experiment = load_experiment(write(tmp_path, data))
        assert experiment.output_dir == (tmp_path / "out" / "here").resolve()

    def test_json_is_accepted(self, tmp_path: Path) -> None:
        experiment = load_experiment(write(tmp_path, minimal(), ".json"))
        assert experiment.tasks[0].id == "news"

    def test_toml_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.toml"
        path.write_text(
            f"""
name = "t"
seed = 1

[targets.local]
base_url = "http://127.0.0.1:8765"
api_key_env = "LOCAL_KEY"

[[tasks]]
id = "news"
instructions = "Which section?"
labels = {{ world = "World news", sports = "Sports" }}
sample = {{ per_label = 3 }}
state = {{ text = {{ column = "text" }} }}

[tasks.source]
kind = "hf_parquet"
repo = "fancyzhx/ag_news"
revision = "{REVISION}"
files = ["data/test-00000-of-00001.parquet"]
label_column = "label"
label_values = {{ "0" = "world", "1" = "sports" }}
""",
            encoding="utf-8",
        )
        experiment = load_experiment(path)
        assert experiment.tasks[0].labels == {"world": "World news", "sports": "Sports"}

    def test_an_unknown_extension_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.ini"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ExperimentError, match="extension"):
            load_experiment(path)


class TestValidation:
    def test_a_misspelt_key_is_refused_not_ignored(self, tmp_path: Path) -> None:
        data = minimal()
        data["client"] = {"max_retry": 5}
        with pytest.raises(ExperimentError, match="max_retry"):
            load_experiment(write(tmp_path, data))

    def test_a_revision_must_be_a_commit_sha(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"][0]["source"]["revision"] = "main"
        with pytest.raises(ExperimentError, match="revision"):
            load_experiment(write(tmp_path, data))

    def test_a_label_value_must_name_a_declared_label(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"][0]["source"]["label_values"]["2"] = "business"
        with pytest.raises(ExperimentError, match="business"):
            load_experiment(write(tmp_path, data))

    def test_a_choice_needs_two_to_255_labels(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"][0]["labels"] = {"world": "World news"}
        data["tasks"][0]["source"]["label_values"] = {"0": "world"}
        with pytest.raises(ExperimentError, match="labels"):
            load_experiment(write(tmp_path, data))

    def test_task_ids_are_unique(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"].append(data["tasks"][0])
        with pytest.raises(ExperimentError, match="news"):
            load_experiment(write(tmp_path, data))

    def test_an_invalid_regex_is_refused_at_load(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"][0]["state"]["text"]["remove_patterns"] = ["["]
        with pytest.raises(ExperimentError, match="remove_patterns"):
            load_experiment(write(tmp_path, data))

    def test_the_arxiv_source_takes_a_month_floor(self, tmp_path: Path) -> None:
        data = minimal()
        data["tasks"][0]["source"] = {
            "kind": "arxiv_snapshot",
            "repo": "librarian-bots/arxiv-metadata-snapshot",
            "revision": REVISION,
            "files": ["data/train-00000-of-00010.parquet"],
            "min_yymm": "2606",
        }
        data["tasks"][0]["labels"] = {"cs.CL": "Computation and Language", "cs.CV": "Vision"}
        experiment = load_experiment(write(tmp_path, data))
        assert experiment.tasks[0].source.min_yymm == "2606"

        data["tasks"][0]["source"]["min_yymm"] = "26-06"
        with pytest.raises(ExperimentError, match="min_yymm"):
            load_experiment(write(tmp_path, data))


class TestShippedExperiments:
    @pytest.mark.parametrize("path", SHIPPED, ids=[path.name for path in SHIPPED])
    def test_every_shipped_experiment_loads(self, path: Path) -> None:
        experiment = load_experiment(path)
        assert experiment.tasks

    def test_the_main_experiment_has_the_three_datasets_and_both_targets(self) -> None:
        experiment = load_experiment(REPO_ROOT / "bench" / "experiments" / "jev-vs-local.yaml")
        assert [task.id for task in experiment.tasks] == ["arxiv-primary", "ag-news", "livedoor"]
        assert set(experiment.targets) == {"local", "jev"}
        assert experiment.targets["jev"].billable is True
        assert experiment.targets["local"].billable is False


class TestApiKey:
    def test_the_environment_wins(self, tmp_path: Path) -> None:
        experiment = load_experiment(write(tmp_path, minimal()))
        key = resolve_api_key(experiment.targets["local"], {"LOCAL_KEY": " k1 , k2 "})
        assert key == "k1"

    def test_the_dotenv_file_is_read_when_the_environment_has_nothing(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / ".env").write_text("# comment\nLOCAL_KEY=from-file\n", encoding="utf-8")
        data = minimal()
        data["targets"]["local"]["dotenv"] = ".env"
        experiment = load_experiment(write(tmp_path, data))
        assert resolve_api_key(experiment.targets["local"], {}) == "from-file"

    def test_a_missing_key_names_the_variable_and_nothing_else(self, tmp_path: Path) -> None:
        experiment = load_experiment(write(tmp_path, minimal()))
        with pytest.raises(ExperimentError, match="JEV_KEY"):
            resolve_api_key(experiment.targets["jev"], {"LOCAL_KEY": "secret-value"})
