"""Core experiment runner shared by the CLI and the Streamlit app.

Applies a corruption spec to raw data BEFORE preprocessing, then runs the
identical preprocessing pipeline on clean and corrupted copies, and scores
both with the subject's trained model. Eligibility is frozen from clean
checks; corrupted trials keep their clean eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .corruptions import CorruptionSpec, apply_corruption
from .evaluation import majority_baseline, paired_metrics
from .preprocessing import (
    CHANNEL_ORDER,
    RunData,
    epochs_from_trials,
    preprocess_run,
)
from . import models as models_mod


@dataclass
class TrialFrame:
    """Per-trial clean/corrupted outputs for one run + corruption spec."""

    trials: pd.DataFrame       # one row per eligible trial, with predictions
    metrics: dict[str, Any]


def run_single_experiment(
    run: RunData,
    model: Any,
    spec: CorruptionSpec | None,
    train_labels: np.ndarray,
) -> TrialFrame:
    """Paired clean-vs-corrupted evaluation on one held-out run.

    ``spec=None`` returns the clean baseline. The corrupted raw array is
    created by apply_corruption (pure); clean and corrupted copies are then
    processed independently through preprocess_run. Y_true is used only to
    score predictions at the end.
    """
    clean_trials = preprocess_run(run, corrupted=None)
    elig = clean_trials["eligible"].to_numpy()
    X_clean, y_true, trial_ids = epochs_from_trials(clean_trials)

    clean_pred = np.full(len(y_true), -1, dtype=np.int64)
    clean_p1 = np.full(len(y_true), np.nan)
    if len(y_true):
        clean_pred, clean_p1 = models_mod.predict(model, X_clean)

    metrics: dict[str, Any] = {
        "subject_id": run.subject_id,
        "run_id": run.run_id,
        "n_train_majority_baseline": majority_baseline(train_labels),
    }

    if spec is None:
        trials = clean_trials.loc[elig].copy()
        trials["clean_pred"] = clean_pred
        trials["clean_p_right"] = clean_p1
        trials["corrupt_pred"] = clean_pred
        trials["corrupt_p_right"] = clean_p1
        trials["prediction_ok"] = True
        for k, v in paired_metrics(y_true, clean_pred, clean_pred, np.ones(len(y_true), bool)).items():
            metrics[k] = v
        metrics["failure"] = "none"
        return TrialFrame(trials=trials.reset_index(drop=True), metrics=metrics)

    # Corrupted path: corruption applied to RAW data before preprocessing.
    data_uv = run.data_v * 1e6
    corrupt_uv = apply_corruption(data_uv, run.channel_names, spec, run.sfreq_hz, units="µV")
    corrupt_v = corrupt_uv * 1e-6
    corrupt_trials = preprocess_run(run, corrupted=corrupt_v)

    # Identical eligible trial IDs as clean (frozen eligibility).
    if not np.array_equal(
        clean_trials.loc[elig, "trial_id"].to_numpy(),
        corrupt_trials.loc[corrupt_trials["eligible"].to_numpy(), "trial_id"].to_numpy(),
    ):
        # Eligibility is frozen from clean checks, so corrupted-path reasons
        # are recorded but never used to drop trials.
        pass
    X_cor, _, ids_cor = epochs_from_trials(corrupt_trials)
    if not np.array_equal(trial_ids, ids_cor):
        raise RuntimeError("trial ID mismatch between clean and corrupted epochs")

    corrupt_pred = np.full(len(y_true), -1, dtype=np.int64)
    corrupt_p1 = np.full(len(y_true), np.nan)
    ok = np.zeros(len(y_true), dtype=bool)
    if len(y_true):
        model_valid = corrupt_trials.loc[corrupt_trials["eligible"].to_numpy(), "model_valid"].to_numpy()
        finite_X = np.isfinite(X_cor).all(axis=(1, 2))
        ok = model_valid & finite_X
        if ok.any():
            p, pr = models_mod.predict(model, X_cor[ok])
            corrupt_pred[ok] = p
            corrupt_p1[ok] = pr

    m = paired_metrics(y_true, clean_pred, corrupt_pred, ok)
    m.update(metrics)
    m["failure"] = spec.describe()
    m["spec"] = spec.to_dict()

    trials = clean_trials.loc[elig].copy()
    trials["clean_pred"] = clean_pred
    trials["clean_p_right"] = clean_p1
    trials["corrupt_pred"] = corrupt_pred
    trials["corrupt_p_right"] = corrupt_p1
    trials["prediction_ok"] = ok
    return TrialFrame(trials=trials.reset_index(drop=True), metrics=m)


def sensitivity_sweep(
    run: RunData,
    model: Any,
    specs: list[CorruptionSpec],
    train_labels: np.ndarray,
) -> pd.DataFrame:
    """Run many specs (e.g., channel x severity grid); one row per spec."""
    rows = []
    for spec in specs:
        tf = run_single_experiment(run, model, spec, train_labels)
        rows.append(tf.metrics)
    return pd.DataFrame(rows)
