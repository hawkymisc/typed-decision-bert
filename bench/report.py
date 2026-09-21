"""The comparison report (BENCHMARK 4, 7).

Per target, each task is scored on the samples that target answered; the head-to-head
numbers are computed only on the samples **both** targets answered, so a sample one
side refused cannot tilt the comparison. Failures show up as coverage instead.

Only records made under the current sample and question count (``current_records``);
the rest are reported as ``stale_records`` and left out.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from bench.config import Experiment, TaskSpec
from bench.datasets import Example, read_samples
from bench.metrics import (
    classification_metrics,
    paired_comparison,
    percentile,
    probabilistic_metrics,
)
from bench.runner import current_records

CAVEATS = "docs/BENCHMARK.md §5"


def build_report(experiment: Experiment) -> dict[str, Any]:
    return {
        "experiment": experiment.name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_model": experiment.model,
        "client": experiment.client.model_dump(),
        "caveats": CAVEATS,
        "tasks": {task.id: _task_report(experiment, task) for task in experiment.tasks},
    }


def _task_report(experiment: Experiment, task: TaskSpec) -> dict[str, Any]:
    examples = read_samples(experiment.output_dir, task.id)
    labels = list(task.labels)
    records: dict[str, dict[str, dict[str, Any]]] = {}
    stale: dict[str, int] = {}
    for target in experiment.targets:
        records[target], stale[target] = current_records(experiment, target, task)
    present = [target for target in experiment.targets if records[target] or stale[target]]

    targets = {
        target: _target_report(examples, labels, records[target])
        | {"stale_records": stale[target]}
        for target in present
    }
    pairs = {
        f"{a} vs {b}": _pair_report(examples, labels, records[a], records[b], experiment.seed)
        for a, b in combinations(present, 2)
    }
    return {"labels": labels, "samples": len(examples), "targets": targets, "pairs": pairs}


def _ok(record: dict[str, Any] | None) -> bool:
    return record is not None and record.get("status") == "ok" and record.get("choice") is not None


def _target_report(
    examples: list[Example], labels: list[str], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    answered = [(e, records[e.sample_id]) for e in examples if _ok(records.get(e.sample_id))]
    gold = [e.label for e, _ in answered]
    pred = [r["choice"] for _, r in answered]
    classification = classification_metrics(gold, pred, labels)
    with_distribution = [(e, r) for e, r in answered if r.get("probabilities")]
    probabilistic = probabilistic_metrics(
        [e.label for e, _ in with_distribution],
        [r["probabilities"] for _, r in with_distribution],
        labels,
    )
    latencies = [r["latency_ms"] for _, r in answered if r.get("latency_ms") is not None]
    statuses = Counter(
        records[e.sample_id]["status"] if e.sample_id in records else "not_sent" for e in examples
    )
    cross_listed = [(e, r) for e, r in answered if e.meta.get("categories")]
    return {
        "samples": len(examples),
        "ok": len(answered),
        "coverage": len(answered) / len(examples) if examples else None,
        "statuses": dict(statuses),
        "accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "per_label": classification["per_label"],
        "confusion": classification["confusion"],
        "nll": probabilistic["nll"],
        "brier": probabilistic["brier"],
        "ece": probabilistic["ece"],
        "any_category_hit": (
            sum(1 for e, r in cross_listed if r["choice"] in e.meta["categories"])
            / len(cross_listed)
            if cross_listed
            else None
        ),
        "response_models": dict(Counter(r.get("response_model") for _, r in answered)),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "mean_attempts": (
            sum(r.get("attempts", 1) for _, r in answered) / len(answered) if answered else None
        ),
    }


def _pair_report(
    examples: list[Example],
    labels: list[str],
    records_a: dict[str, dict[str, Any]],
    records_b: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    both = [
        (e, records_a[e.sample_id], records_b[e.sample_id])
        for e in examples
        if _ok(records_a.get(e.sample_id)) and _ok(records_b.get(e.sample_id))
    ]
    gold = [e.label for e, _, _ in both]
    pred_a = [a["choice"] for _, a, _ in both]
    pred_b = [b["choice"] for _, _, b in both]
    result = paired_comparison(gold, pred_a, pred_b, seed=seed)
    result["macro_f1_a"] = classification_metrics(gold, pred_a, labels)["macro_f1"]
    result["macro_f1_b"] = classification_metrics(gold, pred_b, labels)["macro_f1"]
    result["identical_requests"] = sum(
        1
        for _, a, b in both
        if a.get("request_sha256") is not None
        and a.get("request_sha256") == b.get("request_sha256")
    )
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(text: Any) -> str:
    """Text from a server, made safe for one Markdown table cell."""
    value = str(text)
    for character in "|`<>\r\n":
        value = value.replace(character, " ")
    return value


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark report: {report['experiment']}",
        "",
        f"- generated: {report['generated_at']}",
        f"- requested model: `{report['requested_model']}` (answered models are listed per target)",
        f"- client: {report['client']}",
        "",
        f"> **Read the caveats before quoting any number here: {report['caveats']}.** "
        "Accuracy and macro-F1 are the headline metrics; NLL / Brier / ECE read each "
        "target's own probabilities, which are not on a common scale (C3); latency is "
        "infrastructure, not model (C4).",
        "",
    ]
    for task_id, task in report["tasks"].items():
        summary = f"{task['samples']} samples, {len(task['labels'])} labels."
        lines += [f"## {task_id}", "", summary, ""]
        if not task["targets"]:
            lines += ["No run yet.", ""]
            continue
        lines += [
            "| target | ok / samples | coverage | accuracy | macro-F1 | any-category hit | "
            "NLL | Brier | ECE | p50 ms | p95 ms | answered model(s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, t in task["targets"].items():
            models = ", ".join(f"`{_cell(m)}`×{n}" for m, n in t["response_models"].items())
            lines.append(
                f"| {name} | {t['ok']} / {t['samples']} | {_fmt(t['coverage'])} | "
                f"{_fmt(t['accuracy'])} | {_fmt(t['macro_f1'])} | {_fmt(t['any_category_hit'])} | "
                f"{_fmt(t['nll'])} | {_fmt(t['brier'])} | {_fmt(t['ece'])} | "
                f"{_fmt(t['latency_ms_p50'], 1)} | {_fmt(t['latency_ms_p95'], 1)} | {models} |"
            )
        lines.append("")
        lines += [
            f"- {name}: statuses {t['statuses']}"
            + (f"; {t['stale_records']} stale record(s) ignored" if t["stale_records"] else "")
            for name, t in task["targets"].items()
        ]
        lines.append("")
        if task["pairs"]:
            lines += [
                "Head to head, on the samples both answered:",
                "",
                "| pair (a vs b) | n | acc a | acc b | a − b [95% CI] | macro-F1 a | macro-F1 b | "
                "only a right | only b right | McNemar p | agreement | identical request bytes |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for name, p in task["pairs"].items():
                ci = p["difference_ci95"]
                interval = f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else ""
                lines.append(
                    f"| {name} | {p['n']} | {_fmt(p['accuracy_a'])} | {_fmt(p['accuracy_b'])} | "
                    f"{_fmt(p['difference'])}{interval} | {_fmt(p['macro_f1_a'])} | "
                    f"{_fmt(p['macro_f1_b'])} | {p['only_a_correct']} | {p['only_b_correct']} | "
                    f"{_fmt(p['mcnemar_p'], 4)} | {_fmt(p['agreement'])} | "
                    f"{p['identical_requests']} / {p['n']} |"
                )
            lines.append("")
    return "\n".join(lines)
