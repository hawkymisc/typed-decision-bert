"""K4: compare the hypothesis templates on the smoke fixtures (POC_DESIGN 5.3).

    uv run python scripts/compare_templates.py [--json out.json]

The template decides every compiled model input, so it cannot be chosen by taste after
the fact. This runs the same fixtures through each candidate against the real model and
prints one table per template, so the decision recorded in ``docs/POC_RESULTS.md`` has
numbers behind it.

The numbers are a trend on a few dozen hand-written examples, not a quality claim
(see ``jevbert.evaluation.smoke``). They are good enough to prefer one template over
another and not good enough to say the bundle is any good.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jevbert.compiler.compiled import TokenBudget
from jevbert.compiler.serializer_nli import TEMPLATES
from jevbert.config import Limits, Settings
from jevbert.evaluation.smoke import load_cases, pipeline_answerer, run_smoke
from jevbert.inference.registry import backend_context, build_backend, read_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "smoke"
MANIFEST = PROJECT_ROOT / "manifests" / "jevbert-poc-nli-ja-en-0.2.0.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="compare nli hypothesis templates")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary as JSON")
    args = parser.parse_args(argv)

    manifest, digest = read_manifest(MANIFEST)
    settings = Settings(models_dir=PROJECT_ROOT / "models", limits=Limits())
    backend = build_backend(manifest, backend_context(settings))
    print(f"loading {manifest.public_id} ({digest}) ...", flush=True)
    backend.load()

    cases = load_cases(FIXTURES)
    print(f"{len(cases)} smoke cases from {FIXTURES}\n", flush=True)

    budget = TokenBudget(
        max_sequence_tokens=manifest.limits.max_sequence_tokens,
        max_request_tokens=manifest.limits.max_request_tokens,
    )
    summaries: dict[str, object] = {}
    for template_id, template in TEMPLATES.items():
        answerer = pipeline_answerer(
            backend,
            model_id=manifest.public_id,
            budget=budget,
            temperatures=dict(manifest.calibration.temperature),
            limits=settings.limits,
            template=template,
        )
        started = time.perf_counter()
        report = run_smoke(cases, answerer)
        elapsed = time.perf_counter() - started
        print(f"=== {template_id} ===  ({elapsed:.1f}s)")
        print(report.format_report())
        print()
        summaries[template_id] = report.summary()

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {"bundle": manifest.public_id, "digest": digest, "templates": summaries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
