"""AC6: end-to-end latency at the spec 14.3 benchmark point.

    uv run python scripts/bench_latency.py

The point spec 14.3 fixes as the provisional engineering target is: warm, Q=4,
Choice K=8, every sequence at or below 512 tokens, concurrency 1, end-to-end p95
<= 250 ms. This measures exactly that and prints the answer, met or not.

Whether the target is met is not adjusted for, argued around or re-measured under
friendlier conditions. An A0 backend runs Q x K = 32 sequences for this request where
the A1 the target was written for would run 4, so a miss is an expected property of the
backend, and saying so is the useful part.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import httpx

from jevbert.config import API_KEYS_ENV, load_api_keys

#: spec 14.3: Q=4, Choice K=8, concurrency 1, warm.
QUESTION_COUNT = 4
OPTION_COUNT = 8
TARGET_P95_MS = 250.0

#: Kept well inside 512 tokens per sequence; the actual count is reported and checked.
STATE = (
    "同じ利用料金が二重に引き落とされました。今日中に確認して、重複分を返金してください。"
    "先月の請求書も確認しましたが、同じ金額が二回記載されています。"
)
MAX_SEQUENCE_TOKENS = 512

_OPTIONS = {
    "billing": "請求、支払い、返金の問い合わせ",
    "technical": "ソフトウェアの不具合や接続障害",
    "shipping": "配送や受け取りに関する問い合わせ",
    "returns": "返品や交換に関する問い合わせ",
    "account": "ログインやアカウントに関する問い合わせ",
    "sales": "新規契約や見積もりに関する問い合わせ",
    "legal": "契約条件や法務に関する問い合わせ",
    "other": "上記に該当しない問い合わせ",
}


def build_body(model: str) -> dict[str, Any]:
    assert len(_OPTIONS) == OPTION_COUNT
    question = {
        "type": "choice",
        "instructions": "この問い合わせを最初に担当すべき部署を選んでください。",
        "criteria": _OPTIONS,
    }
    return {
        "model": model,
        "state": STATE,
        "questions": {f"q{index}": dict(question) for index in range(QUESTION_COUNT)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure end-to-end latency (spec 14.3)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    args = parser.parse_args(argv)

    keys = load_api_keys(
        {API_KEYS_ENV: os.environ[API_KEYS_ENV]} if API_KEYS_ENV in os.environ else None
    )
    if not keys:
        print(f"No API key. Set {API_KEYS_ENV} or run `init-env`.", file=sys.stderr)
        return 2

    body = json.dumps(build_body(args.model), ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"}

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        ready = client.get("/readyz")
        if ready.status_code != 200:
            print(f"/readyz -> {ready.status_code}; the bundle is not ready.", file=sys.stderr)
            return 3

        first = client.post("/v1/systemone", content=body, headers=headers)
        first.raise_for_status()
        payload = first.json()
        sequences = QUESTION_COUNT * OPTION_COUNT
        per_sequence = payload["usage"]["input_tokens"] / sequences
        print(f"warm: {args.warmup} requests; measuring {args.iterations}, concurrency 1")
        print(
            f"Q={QUESTION_COUNT} K={OPTION_COUNT} -> {sequences} sequences per request, "
            f"{payload['usage']['input_tokens']} input tokens "
            f"({per_sequence:.0f} per sequence, limit {MAX_SEQUENCE_TOKENS})"
        )
        if per_sequence > MAX_SEQUENCE_TOKENS:
            print(
                f"the benchmark point requires <= {MAX_SEQUENCE_TOKENS} tokens per sequence",
                file=sys.stderr,
            )
            return 4

        for _ in range(args.warmup):
            client.post("/v1/systemone", content=body, headers=headers).raise_for_status()

        samples: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            response = client.post("/v1/systemone", content=body, headers=headers)
            samples.append((time.perf_counter() - started) * 1000.0)
            response.raise_for_status()

    samples.sort()
    result = {
        "iterations": len(samples),
        "questions": QUESTION_COUNT,
        "options": OPTION_COUNT,
        "sequences_per_request": sequences,
        "input_tokens": payload["usage"]["input_tokens"],
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": statistics.fmean(samples),
        "target_p95_ms": TARGET_P95_MS,
    }
    result["target_met"] = result["p95_ms"] <= TARGET_P95_MS

    print()
    for name in ("p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "mean_ms"):
        print(f"  {name:<10} {result[name]:8.1f}")
    verdict = "MET" if result["target_met"] else "NOT MET"
    print(f"\nspec 14.3 target p95 <= {TARGET_P95_MS:.0f} ms: {verdict} "
          f"(measured p95 {result['p95_ms']:.1f} ms)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    return 0


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    """Nearest-rank percentile; with 100 samples p95 is simply the 95th smallest."""
    if not sorted_samples:
        raise ValueError("no samples")
    rank = max(1, min(len(sorted_samples), round(percentile / 100.0 * len(sorted_samples))))
    return sorted_samples[rank - 1]


if __name__ == "__main__":
    raise SystemExit(main())
