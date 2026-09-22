"""From a pinned parquet file to the sample both targets are sent (BENCHMARK 2)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench.config import TaskSpec
from bench.datasets import (
    Example,
    SampleError,
    load_examples,
    prepare_task,
    read_samples,
    render_field,
    stratified_sample,
)

REVISION = "0" * 40


def news_task(**overrides: Any) -> TaskSpec:
    data: dict[str, Any] = {
        "id": "news",
        "source": {
            "kind": "hf_parquet",
            "repo": "example/news",
            "revision": REVISION,
            "files": ["a.parquet", "b.parquet"],
            "label_column": "label",
            "label_values": {"0": "world", "1": "sports"},
        },
        "state": {"text": {"column": "text", "max_chars": 10}},
        "instructions": "Which section?",
        "labels": {"world": "World news", "sports": "Sports"},
        "sample": {"per_label": 2},
    }
    data.update(overrides)
    return TaskSpec.model_validate(data)


def arxiv_task(**overrides: Any) -> TaskSpec:
    data: dict[str, Any] = {
        "id": "arxiv",
        "source": {
            "kind": "arxiv_snapshot",
            "repo": "librarian-bots/arxiv-metadata-snapshot",
            "revision": REVISION,
            "files": ["x.parquet"],
            "min_yymm": "2606",
        },
        "state": {"title": {"column": "title"}, "abstract": {"column": "abstract"}},
        "instructions": "Primary category?",
        "labels": {"cs.CL": "Computation and Language", "cs.CV": "Computer Vision"},
        "sample": {"per_label": 1},
    }
    data.update(overrides)
    return TaskSpec.model_validate(data)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def many(label_counts: dict[str, int]) -> list[Example]:
    examples = []
    for label, count in label_counts.items():
        for index in range(count):
            examples.append(Example(f"{label}-{index}", label, {"text": str(index)}, {}))
    return examples


class TestRenderField:
    def test_whitespace_is_collapsed_and_trimmed(self) -> None:
        field = news_task().state["text"]
        assert render_field("  a \n\n b\t", field.model_copy(update={"max_chars": None})) == "a b"

    def test_truncation_counts_characters_not_bytes(self) -> None:
        field = news_task().state["text"].model_copy(update={"max_chars": 3})
        assert render_field("日本語の文章", field) == "日本語"

    def test_patterns_are_removed_before_truncation(self) -> None:
        field = news_task().state["text"].model_copy(
            update={"max_chars": 5, "remove_patterns": ["【[^】]*】"]}
        )
        assert render_field("【Sports Watch】ボルト失格", field) == "ボルト失格"

    def test_a_missing_value_renders_empty(self) -> None:
        assert render_field(None, news_task().state["text"]) == ""


class TestHfParquet:
    def test_labels_are_mapped_and_unmapped_rows_are_dropped(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet",
            [
                {"text": "first  row", "label": 0},
                {"text": "second", "label": 2},  # not in label_values
            ],
        )
        b = write_parquet(tmp_path / "b.parquet", [{"text": "third row is long", "label": 1}])
        examples = load_examples(news_task(), [a, b])

        assert [(e.sample_id, e.label, e.state) for e in examples] == [
            ("a.parquet#0", "world", {"text": "first row"}),
            ("b.parquet#0", "sports", {"text": "third row"}),
        ]

    def test_an_empty_state_field_drops_the_row(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", [{"text": "   ", "label": 0}])
        b = write_parquet(tmp_path / "b.parquet", [{"text": "ok", "label": 1}])
        assert [e.label for e in load_examples(news_task(), [a, b])] == ["sports"]


class TestArxivSnapshot:
    def test_recent_new_style_papers_with_a_primary_in_the_label_set(
        self, tmp_path: Path
    ) -> None:
        rows = [
            {"id": "2606.00001", "categories": "cs.CL cs.AI", "title": "T1", "abstract": "A1"},
            {"id": "2605.99999", "categories": "cs.CL", "title": "old", "abstract": "old"},
            {"id": "hep-th/9901001", "categories": "cs.CL", "title": "x", "abstract": "x"},
            {"id": "2609.12345", "categories": "cs.AI cs.CV", "title": "T2", "abstract": "A2"},
            {"id": "2607.00002", "categories": "cs.CV cs.LG", "title": "T3", "abstract": "A3"},
        ]
        path = write_parquet(tmp_path / "x.parquet", rows)
        examples = load_examples(arxiv_task(), [path])

        assert [(e.sample_id, e.label) for e in examples] == [
            ("2606.00001", "cs.CL"),
            ("2607.00002", "cs.CV"),
        ]
        assert examples[0].state == {"title": "T1", "abstract": "A1"}
        assert examples[0].meta["categories"] == ["cs.CL", "cs.AI"]


class TestStratifiedSample:
    def test_takes_per_label_from_each_label(self) -> None:
        sample = stratified_sample(many({"world": 5, "sports": 7}), news_task(), seed=1)
        assert Counter(e.label for e in sample) == {"world": 2, "sports": 2}

    def test_is_deterministic_and_independent_of_input_order(self) -> None:
        examples = many({"world": 20, "sports": 20})
        first = stratified_sample(examples, news_task(), seed=3)
        second = stratified_sample(list(reversed(examples)), news_task(), seed=3)
        assert [e.sample_id for e in first] == [e.sample_id for e in second]

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        examples = many({"world": 50, "sports": 50})
        first = {e.sample_id for e in stratified_sample(examples, news_task(), seed=1)}
        second = {e.sample_id for e in stratified_sample(examples, news_task(), seed=2)}
        assert first != second

    def test_a_label_without_enough_examples_is_an_error_naming_it(self) -> None:
        with pytest.raises(SampleError, match="sports"):
            stratified_sample(many({"world": 5, "sports": 1}), news_task(), seed=1)

    def test_the_order_interleaves_labels_so_a_prefix_is_not_one_label(self) -> None:
        task = news_task(sample={"per_label": 10})
        sample = stratified_sample(many({"world": 30, "sports": 30}), task, seed=5)
        assert len({e.label for e in sample[:6]}) == 2


class TestPrepare:
    def test_writes_samples_and_a_manifest_that_read_verifies(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        write_parquet(
            source / "a.parquet",
            [{"text": f"w{i}", "label": 0} for i in range(3)]
            + [{"text": f"s{i}", "label": 1} for i in range(3)],
        )
        write_parquet(source / "b.parquet", [{"text": "s9", "label": 1}])

        def fetch(repo: str, revision: str, filename: str) -> Path:
            assert (repo, revision) == ("example/news", REVISION)
            return source / filename

        out = tmp_path / "out"
        prepared = prepare_task(news_task(), seed=11, output_dir=out, fetch=fetch)
        assert prepared.count == 4

        manifest = json.loads((out / "samples" / "news.manifest.json").read_text("utf-8"))
        assert manifest["source"]["revision"] == REVISION
        assert manifest["seed"] == 11
        assert manifest["counts"] == {"world": 2, "sports": 2}
        assert len(manifest["sha256"]) == 64

        samples = read_samples(out, "news")
        assert [e.sample_id for e in samples] == [e.sample_id for e in prepared.examples]

    def test_a_sample_file_edited_after_prepare_is_refused(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        write_parquet(source / "a.parquet", [{"text": f"w{i}", "label": i % 2} for i in range(8)])
        write_parquet(source / "b.parquet", [{"text": "x", "label": 0}])
        out = tmp_path / "out"
        prepare_task(news_task(), seed=1, output_dir=out, fetch=lambda r, v, f: source / f)

        path = out / "samples" / "news.jsonl"
        path.write_text(path.read_text("utf-8").replace("w", "W"), encoding="utf-8")
        with pytest.raises(SampleError, match="SHA-256"):
            read_samples(out, "news")

    def test_reading_before_prepare_says_to_prepare(self, tmp_path: Path) -> None:
        with pytest.raises(SampleError, match="prepare"):
            read_samples(tmp_path, "news")
