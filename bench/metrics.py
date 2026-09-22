"""Classification metrics and the paired comparison (docs/BENCHMARK.md 4).

Pure functions over plain lists, so the report and its tests share one definition.

Conventions:

* ``macro_f1`` averages over the labels that occur in the gold labels **or** in the
  predictions, as scikit-learn does (a label nobody predicted and nobody needed does
  not dilute it; a prediction outside ``labels`` is scored, with F1 = 0); an undefined
  precision or recall counts as 0.
* The probabilistic metrics read the returned *distribution*, never the ``confidence``
  field, because the two targets define ``confidence`` differently (BENCHMARK C3).
  ECE bins answers by the top probability of that distribution.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

#: Floor for ``p(gold)`` in the NLL, so one confident miss is large but finite.
PROBABILITY_FLOOR = 1e-12


def classification_metrics(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    confusion = {g: dict.fromkeys(labels, 0) for g in labels}
    for g, p in zip(gold, pred, strict=True):
        confusion.setdefault(g, {}).setdefault(p, 0)
        confusion[g][p] += 1

    extra = sorted(set(pred) - set(labels))
    per_label: dict[str, dict[str, float]] = {}
    for label in [*labels, *extra]:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        predicted = sum(1 for p in pred if p == label)
        support = sum(1 for g in gold if g == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    if not gold:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_label": per_label,
            "confusion": confusion,
        }
    present = [label for label in per_label if label in set(gold) | set(pred)]
    return {
        "n": len(gold),
        "accuracy": sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / len(gold),
        "macro_f1": sum(per_label[label]["f1"] for label in present) / len(present),
        "per_label": per_label,
        "confusion": confusion,
    }


def probabilistic_metrics(
    gold: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    bins: int = 15,
) -> dict[str, Any]:
    if len(gold) != len(probabilities):
        raise ValueError(f"{len(gold)} gold labels but {len(probabilities)} distributions")
    if not gold:
        return {"n": 0, "nll": None, "brier": None, "ece": None}

    nll = 0.0
    brier = 0.0
    binned: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for g, distribution in zip(gold, probabilities, strict=True):
        nll -= math.log(max(float(distribution.get(g, 0.0)), PROBABILITY_FLOOR))
        brier += sum(
            (float(distribution.get(label, 0.0)) - (1.0 if label == g else 0.0)) ** 2
            for label in labels
        )
        top_label = top_choice(distribution)
        top = float(distribution[top_label])
        index = min(bins - 1, max(0, math.ceil(top * bins) - 1))
        binned[index].append((top, top_label == g))

    n = len(gold)
    ece = 0.0
    for entries in binned:
        if entries:
            accuracy = sum(ok for _, ok in entries) / len(entries)
            confidence = sum(top for top, _ in entries) / len(entries)
            ece += len(entries) / n * abs(accuracy - confidence)
    return {"n": n, "nll": nll / n, "brier": brier / n, "ece": ece}


def top_choice(distribution: Mapping[str, float]) -> str:
    """The argmax; a tie goes to the smallest key, as the API's argmax does (spec I05)."""
    return min(distribution, key=lambda key: (-float(distribution[key]), key))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(
    gold: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Two targets on the same samples: counts, McNemar, and a paired bootstrap CI."""
    if not len(gold) == len(pred_a) == len(pred_b):
        raise ValueError("gold and both prediction lists must have the same length")
    n = len(gold)
    if n == 0:
        return {
            "n": 0,
            "accuracy_a": None,
            "accuracy_b": None,
            "difference": None,
            "difference_ci95": None,
            "both_correct": 0,
            "only_a_correct": 0,
            "only_b_correct": 0,
            "both_wrong": 0,
            "agreement": None,
            "mcnemar_p": None,
        }
    right_a = [g == p for g, p in zip(gold, pred_a, strict=True)]
    right_b = [g == p for g, p in zip(gold, pred_b, strict=True)]
    only_a = sum(1 for a, b in zip(right_a, right_b, strict=True) if a and not b)
    only_b = sum(1 for a, b in zip(right_a, right_b, strict=True) if b and not a)
    both = sum(1 for a, b in zip(right_a, right_b, strict=True) if a and b)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(right_a[i] - right_b[i] for i in indices) / n)
    diffs.sort()

    accuracy_a = sum(right_a) / n
    accuracy_b = sum(right_b) / n
    return {
        "n": n,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "difference": accuracy_a - accuracy_b,
        "difference_ci95": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
        "both_correct": both,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": n - both - only_a - only_b,
        "agreement": sum(1 for a, b in zip(pred_a, pred_b, strict=True) if a == b) / n,
        "mcnemar_p": mcnemar_exact_p(only_a, only_b),
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (NumPy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
