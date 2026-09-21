"""``python -m bench prepare|plan|run|report <experiment file>`` (docs/BENCHMARK.md 1).

Exit codes: 0 done, 1 a run finished with failed samples, 2 the experiment, the sample
or the key cannot be used, 3 a billable target was not allowed to send.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from bench.config import Experiment, ExperimentError, load_experiment
from bench.datasets import SampleError, prepare_task
from bench.report import build_report, render_markdown
from bench.runner import BillableTargetRefused, build_body, pending_samples, run_target


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="fetch the pinned data and draw the sample")
    prepare.add_argument("experiment")
    prepare.add_argument("--task", action="append", help="only this task (repeatable)")

    plan = commands.add_parser("plan", help="show what `run` would send, without sending")
    plan.add_argument("experiment")
    plan.add_argument("--limit", type=int, default=None, help="first N samples of each task")

    run = commands.add_parser("run", help="send the sample to one target")
    run.add_argument("experiment")
    run.add_argument("--target", required=True)
    run.add_argument("--task", action="append", help="only this task (repeatable)")
    run.add_argument("--limit", type=int, default=None, help="first N samples of each task")
    run.add_argument(
        "--allow-billable",
        action="store_true",
        help="required to send to a target marked billable (the real Jev API)",
    )

    report = commands.add_parser("report", help="write report.md and report.json")
    report.add_argument("experiment")

    args = parser.parse_args(argv)
    try:
        experiment = load_experiment(args.experiment)
        if args.command == "prepare":
            return _prepare(experiment, args.task)
        if args.command == "plan":
            return _plan(experiment, args.limit)
        if args.command == "run":
            return _run(experiment, args)
        return _report(experiment)
    except (ExperimentError, SampleError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BillableTargetRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3


def _prepare(experiment: Experiment, task_ids: list[str] | None) -> int:
    for task in experiment.tasks:
        if task_ids and task.id not in task_ids:
            continue
        prepared = prepare_task(task, experiment.seed, experiment.output_dir)
        counts = prepared.manifest["counts"]
        print(f"[{task.id}] wrote {prepared.count} samples to {prepared.path} {counts}")
    return 0


def _plan(experiment: Experiment, limit: int | None) -> int:
    print(f"experiment {experiment.name}: model={experiment.model} client={experiment.client}")
    for task in experiment.tasks:
        for name, target in experiment.targets.items():
            todo, done = pending_samples(experiment, name, task, limit)
            billable = " (billable)" if target.billable else ""
            print(f"  {task.id} -> {name}{billable}: {len(todo)} to send, {done} already ok")
        todo, _ = pending_samples(experiment, next(iter(experiment.targets)), task, 1)
        first = todo[0] if todo else None
        if first is not None:
            print(f"  first request body of {task.id}:")
            body = json.dumps(build_body(experiment, task, first), ensure_ascii=False, indent=2)
            print("    " + body.replace("\n", "\n    "))
    return 0


def _run(experiment: Experiment, args: argparse.Namespace) -> int:
    if args.target not in experiment.targets:
        known = ", ".join(experiment.targets)
        raise ExperimentError(f"unknown target {args.target!r}; the file declares {known}")
    summary = run_target(
        experiment,
        args.target,
        task_ids=args.task,
        limit=args.limit,
        allow_billable=args.allow_billable,
    )
    print(f"{args.target}: sent {summary.sent}, skipped {summary.skipped}, {summary.by_status}")
    failed = sum(count for status, count in summary.by_status.items() if status != "ok")
    return 1 if failed else 0


def _report(experiment: Experiment) -> int:
    report = build_report(experiment)
    text = render_markdown(report)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    (experiment.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (experiment.output_dir / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwritten to {experiment.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
