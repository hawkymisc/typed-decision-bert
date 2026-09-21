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


def build_premise_probe(model: str) -> dict[str, Any]:
    """A request whose two candidates are empty, so ``usage`` measures the state alone.

    Both Noul criteria render to the empty string and there are no instructions, so
    each sequence is ``[bos] + premise + [eos, eos] + [eos]``: exactly ``P + 4`` tokens.
    """
    return {
        "model": model,
        "state": STATE,
        "questions": {"probe": {"type": "noul", "criteria": {"true": "", "false": ""}}},
    }


def premise_sequence_tokens(probe_tokens: int) -> int:
    """``P + 4`` from the probe's two identical sequences."""
    if probe_tokens <= 0 or probe_tokens % 2:
        raise ValueError(
            f"the probe should report an even, positive token count, got {probe_tokens}"
        )
    return probe_tokens // 2


def longest_sequence_upper_bound(
    total_tokens: int, questions: int, options: int, premise_cost: int
) -> int:
    """An upper bound on the longest single sequence in the benchmark request.

    ``usage.input_tokens`` is a *sum*, and the check used to divide it by the number of
    sequences, which gives the mean. The benchmark point of spec 14.3 is stated per
    sequence, and a mean under 512 says nothing about the longest one (Q-M4).

    Every sequence here carries the same premise and differs only in its candidate, so
    with ``C`` for the candidate token counts::

        per_question = options x premise_cost + sum(C)
        longest      = premise_cost + max(C)  <=  premise_cost + sum(C)

    The bound needs one extra request rather than a tokenizer in this process, and a
    bound that holds is what the condition asks for - not an average that does not.
    """
    if questions <= 0 or options <= 0:
        raise ValueError("the benchmark point has at least one question and one option")
    if total_tokens % questions:
        raise ValueError(
            f"{total_tokens} tokens do not divide evenly over {questions} identical questions"
        )
    per_question = total_tokens // questions
    candidate_tokens = per_question - options * premise_cost
    if candidate_tokens < 0:
        raise ValueError(
            f"the premise probe reported {premise_cost} tokens per sequence, which is more "
            f"than the {per_question} the whole question costs"
        )
    return premise_cost + candidate_tokens


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

        probe = client.post(
            "/v1/systemone",
            content=json.dumps(build_premise_probe(args.model), ensure_ascii=False).encode(),
            headers=headers,
        )
        probe.raise_for_status()
        premise_cost = premise_sequence_tokens(probe.json()["usage"]["input_tokens"])

        first = client.post("/v1/systemone", content=body, headers=headers)
        first.raise_for_status()
        payload = first.json()
        sequences = QUESTION_COUNT * OPTION_COUNT
        total_tokens = payload["usage"]["input_tokens"]
        longest = longest_sequence_upper_bound(
            total_tokens, QUESTION_COUNT, OPTION_COUNT, premise_cost
        )
        print(f"warm: {args.warmup} requests; measuring {args.iterations}, concurrency 1")
        print(
            f"Q={QUESTION_COUNT} K={OPTION_COUNT} -> {sequences} sequences per request, "
            f"{total_tokens} input tokens ({total_tokens / sequences:.0f} per sequence on "
            f"average, longest at most {longest}, limit {MAX_SEQUENCE_TOKENS})"
        )
        if longest > MAX_SEQUENCE_TOKENS:
            print(
                f"the benchmark point requires <= {MAX_SEQUENCE_TOKENS} tokens in the "
                f"longest sequence; the bound is {longest}",
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
        "input_tokens": total_tokens,
        "longest_sequence_tokens_at_most": longest,
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
