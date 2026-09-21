"""AC2: the spec 5.6 example through the official SDK, against a running server.

    uv run python scripts/run_server.ps1        # in another terminal
    uv run python scripts/sdk_demo.py

Two things are demonstrated, in this order:

1. **Drop in** (C2, spec 18.3): a ``TypeSafeClient`` built with nothing but a base URL
   and an API key. The model is not named, so the SDK sends its own default,
   ``jev-latest``, and the PoC configuration resolves it. This is the case a caller
   written against Jev actually hits.
2. **The spec 5.6 request**: three question types at once, in Japanese, restored into
   the SDK's typed answers, with every invariant I01-I09 checked against the wire body.

No ``curl``: the whole point is that the *official client* restores the response, which
a shell cannot tell you.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeClient,
)

from jevbert.config import API_KEYS_ENV, load_api_keys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATE = {
    "message": (
        "同じ利用料金が二重に引き落とされました。"
        "今日中に確認して、重複分を返金してください。"
    )
}

QUESTIONS = {
    "refund_requested": Noul(instructions="顧客は明示的に返金を求めていますか。"),
    "department": Choice(
        instructions="この問い合わせを最初に担当すべき部署を選んでください。",
        criteria={
            "billing": "請求、支払い、返金の問い合わせ",
            "technical": "ソフトウェアの不具合や接続障害",
            "other": "上記に該当しない問い合わせ",
        },
    ),
    "urgency": Score(
        instructions="顧客が表明している対応期限の切迫度を評価してください。",
        criteria=[
            "対応期限の指定がない",
            "数日以内の対応を求めている",
            "当日中または直ちに対応することを求めている",
        ],
    ),
}

#: The SDK retries 5xx by default; a demo wants the first answer, not the third.
NO_RETRY = RetryPolicy(max_retries=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="drive the PoC server from the official SDK")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--model",
        default=None,
        help="bundle ID or alias; omitted means the SDK's own default (jev-latest)",
    )
    args = parser.parse_args(argv)

    api_key = _api_key()
    if api_key is None:
        print(
            f"No API key. Set {API_KEYS_ENV} or run `uv run python -m jevbert init-env`.",
            file=sys.stderr,
        )
        return 2

    if not _server_is_ready(args.base_url):
        return 3

    print("=" * 78)
    print("1. drop-in: base_url and api_key only, the SDK's default model (jev-latest)")
    print("=" * 78)
    with TypeSafeClient(base_url=args.base_url, api_key=api_key, retry=NO_RETRY) as client:
        default_model_response = client.system_one(state="返金してほしい", questions={"q": Noul()})
    print("  requested model : (not set -> the SDK sends its default, 'jev-latest')")
    print(f"  answered model  : {default_model_response.model}")
    print(f"  answer          : noul={default_model_response.answers['q'].noul!r}")
    print("  -> the alias resolved and the answer names the immutable bundle (D01).\n")

    print("=" * 78)
    print("2. the spec 5.6 example: three question types, Japanese, one request")
    print("=" * 78)
    raw = _raw_request(args.base_url, api_key, args.model)
    print(json.dumps(raw, ensure_ascii=False, indent=2))
    print()

    kwargs: dict[str, Any] = {"base_url": args.base_url, "api_key": api_key, "retry": NO_RETRY}
    if args.model is not None:
        kwargs["model"] = args.model
    with TypeSafeClient(**kwargs) as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    print("restored by the SDK:")
    refund = response.answers["refund_requested"]
    department = response.answers["department"]
    urgency = response.answers["urgency"]
    assert isinstance(refund, NoulAnswer)
    assert isinstance(department, ChoiceAnswer)
    assert isinstance(urgency, ScoreAnswer)
    print(f"  model            : {response.model}")
    print(f"  refund_requested : noul={refund.noul}")
    print(f"  department       : choice={department.choice!r} "
          f"confidence={department.confidence}")
    print(f"    probabilities  : {department.probabilities}")
    print(f"  urgency          : score={urgency.score} confidence={urgency.confidence}")
    print(f"    probabilities  : {urgency.probabilities}")
    print(f"    legend         : {urgency.legend}")
    print(f"  usage            : input_tokens={response.usage.input_tokens} "
          f"output_tokens={response.usage.output_tokens}")
    print(f"  request_id       : {response.request_id}")
    print()

    failures = check_invariants(raw)
    print("invariants (spec 5.5), checked against the wire body:")
    for line in failures:
        print(f"  {line}")
    return 0 if all(line.startswith("I0") and "OK" in line for line in failures) else 1


def _api_key() -> str | None:
    keys = load_api_keys({API_KEYS_ENV: os.environ[API_KEYS_ENV]}) if (
        API_KEYS_ENV in os.environ
    ) else load_api_keys()
    return keys[0] if keys else None


def _server_is_ready(base_url: str) -> bool:
    """``/readyz`` before anything else, so a cold bundle is not read as a bad answer."""
    try:
        response = httpx.get(f"{base_url}/readyz", timeout=5.0)
    except httpx.HTTPError as exc:
        print(
            f"Cannot reach {base_url}: {type(exc).__name__}. "
            "Start the server with ./scripts/run_server.ps1",
            file=sys.stderr,
        )
        return False
    if response.status_code != 200:
        print(
            f"{base_url}/readyz returned {response.status_code} {response.text!r}. "
            "The bundle is still loading, or it failed to load.",
            file=sys.stderr,
        )
        return False
    print(f"{base_url}/readyz -> 200 {response.text}\n")
    return True


def _raw_request(base_url: str, api_key: str, model: str | None) -> dict[str, Any]:
    """Fetch the same answer as a plain body, so the invariants can be checked on it.

    The SDK's typed answers are the point of the demo, but an invariant like "no
    ``confidence`` on a Noul" is about the *wire* body: a model that drops unknown
    fields cannot tell you whether the server sent one.
    """
    body = {
        "model": model or "jev-latest",
        "state": STATE,
        "questions": {
            "refund_requested": {
                "type": "noul",
                "instructions": "顧客は明示的に返金を求めていますか。",
            },
            "department": {
                "type": "choice",
                "instructions": "この問い合わせを最初に担当すべき部署を選んでください。",
                "criteria": {
                    "billing": "請求、支払い、返金の問い合わせ",
                    "technical": "ソフトウェアの不具合や接続障害",
                    "other": "上記に該当しない問い合わせ",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "顧客が表明している対応期限の切迫度を評価してください。",
                "criteria": [
                    "対応期限の指定がない",
                    "数日以内の対応を求めている",
                    "当日中または直ちに対応することを求めている",
                ],
            },
        },
    }
    response = httpx.post(
        f"{base_url}/v1/systemone",
        content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


#: The rubric the demo sends, which I07 requires the answer to give back unchanged.
RUBRIC = [
    "対応期限の指定がない",
    "数日以内の対応を求めている",
    "当日中または直ちに対応することを求めている",
]


def check_invariants(payload: dict[str, Any]) -> list[str]:
    """I01-I09 of spec 5.5, reported one line each.

    Every check is a callable rather than an eagerly evaluated expression: a body that
    is missing an answer fails I01, and the checks after it would then raise a KeyError
    and take the whole report with them. A broken body has to produce a report saying
    which invariant broke, not a traceback (Q-M4).
    """
    answers = payload.get("answers", {})

    def numbers_are_finite() -> bool:
        return (
            all(value == value and abs(value) != float("inf") for value in _numbers(payload))
            and 0.0 <= answers["refund_requested"]["noul"] <= 1.0
            and all(
                0.0 <= value <= 1.0
                for key in ("department", "urgency")
                for value in answers[key]["probabilities"].values()
            )
            and all(0.0 <= answers[key]["confidence"] <= 1.0 for key in ("department", "urgency"))
        )

    def choice_is_the_argmax() -> bool:
        probabilities = answers["department"]["probabilities"]
        best = max(probabilities.values())
        return probabilities[answers["department"]["choice"]] == best and answers["department"][
            "choice"
        ] == min(key for key, value in probabilities.items() if value == best)

    def score_is_the_expected_value() -> bool:
        levels = answers["urgency"]["probabilities"]
        expected = sum(int(key) * value for key, value in levels.items())
        return abs(answers["urgency"]["score"] - expected) <= 1e-6

    return [
        _check("I01 question IDs round-trip", lambda: set(answers) == set(QUESTIONS)),
        _check(
            "I02 answer types match the questions",
            lambda: answers["refund_requested"]["type"] == "noul"
            and answers["department"]["type"] == "choice"
            and answers["urgency"]["type"] == "score",
        ),
        _check(
            "I03 every number finite, probabilities and confidence in [0,1]",
            numbers_are_finite,
        ),
        _check(
            "I04 distributions sum to 1 within 1e-6",
            lambda: all(
                abs(sum(answers[key]["probabilities"].values()) - 1.0) <= 1e-6
                for key in ("department", "urgency")
            ),
        ),
        _check("I05 choice is the argmax (ties: smallest key)", choice_is_the_argmax),
        _check("I06 score equals the expected value within 1e-6", score_is_the_expected_value),
        _check(
            "I07 legend preserves the rubric and its types",
            lambda: answers["urgency"]["legend"]
            == {str(index): text for index, text in enumerate(RUBRIC)},
        ),
        _check(
            "I08 no confidence on a Noul",
            lambda: "confidence" not in answers["refund_requested"],
        ),
        _check(
            "I09 no unspecified fields",
            lambda: set(payload) == {"model", "answers", "usage"}
            and set(answers["refund_requested"]) == {"type", "noul"}
            and set(answers["department"]) == {"type", "choice", "probabilities", "confidence"}
            and set(answers["urgency"])
            == {"type", "score", "legend", "probabilities", "confidence"},
        ),
    ]


def _numbers(value: Any) -> Any:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def _check(label: str, predicate: Callable[[], bool]) -> str:
    """One invariant, as a line. A check that cannot even run is a FAILED, named.

    The exception is not swallowed: its class is in the line, and the line is what the
    demo prints and what its exit code is computed from.
    """
    try:
        ok = bool(predicate())
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        return f"{label:<58} FAILED ({type(exc).__name__})"
    return f"{label:<58} {'OK' if ok else 'FAILED'}"


if __name__ == "__main__":
    raise SystemExit(main())
