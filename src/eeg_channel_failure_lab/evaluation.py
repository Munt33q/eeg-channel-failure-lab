"""Paired evaluation of clean vs corrupted held-out predictions.

Rules enforced here:
- Exactly the same eligible trial IDs for the clean and corrupted paths
  (eligibility frozen from clean-data checks).
- Corrupted trials are never dropped silently; model-invalid inputs and
  prediction failures are counted and treated as INCORRECT in the
  all-eligible accuracy, with coverage reported alongside.
- Metrics never use evaluation labels for anything except scoring.
- Accuracy among successfully predicted trials is reported separately and
  clearly labeled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEFT, RIGHT = 0, 1


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(np.sum((y_true == LEFT) & (y_pred == LEFT))),
        "fp": int(np.sum((y_true == LEFT) & (y_pred == RIGHT))),
        "fn": int(np.sum((y_true == RIGHT) & (y_pred == LEFT))),
        "tp": int(np.sum((y_true == RIGHT) & (y_pred == RIGHT))),
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not len(y_true):
        return float("nan")
    accs = []
    for cls in (LEFT, RIGHT):
        m = y_true == cls
        accs.append(float(np.mean(y_pred[m] == cls)) if m.any() else float("nan"))
    return float(np.nanmean(accs))


def majority_baseline(y_train: np.ndarray) -> float:
    """Majority-class accuracy derived from TRAINING labels only."""
    if not len(y_train):
        return float("nan")
    counts = np.bincount(y_train, minlength=2)
    return float(counts.max() / counts.sum())


def paired_metrics(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    corrupt_pred: np.ndarray | None,
    prediction_ok: np.ndarray,
) -> dict[str, Any]:
    """Metrics for one paired clean/corrupted comparison on identical trials.

    ``prediction_ok`` marks trials whose corrupted predictions succeeded
    (model-valid input and finite prediction). Failed trials count as
    incorrect for the primary accuracy and are reported explicitly.
    """
    n = len(y_true)
    assert len(clean_pred) == n
    ok = np.asarray(prediction_ok, dtype=bool)

    clean_acc = accuracy(y_true, clean_pred)
    clean_bacc = balanced_accuracy(y_true, clean_pred)

    if corrupt_pred is None:
        corrupt_pred = np.full(n, -1, dtype=np.int64)

    # Primary metric: all eligible trials, failures counted as incorrect.
    filled = np.where(ok, corrupt_pred, -1)  # -1 never equals 0/1 -> incorrect
    corrupt_acc_all = accuracy(y_true, filled)
    corrupt_bacc_all = balanced_accuracy(y_true, filled)

    # Secondary metric, clearly labeled: accuracy among predicted trials only.
    if ok.any():
        corrupt_acc_pred = accuracy(y_true[ok], corrupt_pred[ok])
    else:
        corrupt_acc_pred = float("nan")

    flips = clean_pred != corrupt_pred
    flip_rate = float(np.mean(flips)) if n else float("nan")
    c2i = int(np.sum((clean_pred == y_true) & (filled != y_true)))
    i2c = int(np.sum((clean_pred != y_true) & (filled == y_true)))
    n_fail = int(np.sum(~ok))
    coverage = float(np.mean(ok)) if n else float("nan")

    return {
        "n_trials": n,
        "clean_accuracy": clean_acc,
        "clean_balanced_accuracy": clean_bacc,
        "corrupt_accuracy_all_eligible": corrupt_acc_all,
        "corrupt_balanced_accuracy_all_eligible": corrupt_bacc_all,
        "accuracy_drop_pp": (clean_acc - corrupt_acc_all) * 100.0,
        "corrupt_accuracy_among_predicted": corrupt_acc_pred,
        "prediction_failures": n_fail,
        "coverage": coverage,
        "flip_rate": flip_rate,
        "n_flips": int(np.sum(flips)),
        "correct_to_incorrect": c2i,
        "incorrect_to_correct": i2c,
        "clean_confusion": confusion(y_true, clean_pred),
        "corrupt_confusion": confusion(y_true, filled),
    }


def summarize_subjects(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Unweighted mean across subjects for selected numeric fields."""
    if not rows:
        return {}
    out: dict[str, float] = {}
    keys = [
        "clean_accuracy",
        "corrupt_accuracy_all_eligible",
        "accuracy_drop_pp",
        "flip_rate",
        "coverage",
    ]
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]  # skip NaN
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out
