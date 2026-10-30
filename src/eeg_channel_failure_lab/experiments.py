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

    corrupt_pred = np.full(len(y_true), -1, dtype=np.int64)
    corrupt_p1 = np.full(len(y_true), np.nan)
    ok = np.zeros(len(y_true), dtype=bool)
    if len(y_true):
        # Freeze eligibility from clean data. Never use corrupted-path
        # eligibility to drop a paired trial. Missing/invalid corrupted
        # epochs become explicit prediction failures instead.
        by_id = {int(row.trial_id): row for row in corrupt_trials.itertuples()}
        cor_epochs = []
        model_valid = []
        for trial_id in trial_ids:
            row = by_id.get(int(trial_id))
            epoch = None if row is None else row.epoch
            cor_epochs.append(epoch)
            model_valid.append(bool(row is not None and row.model_valid and epoch is not None))
        finite_rows = [e is not None and np.isfinite(e).all() for e in cor_epochs]
        ok = np.asarray(model_valid, dtype=bool) & np.asarray(finite_rows, dtype=bool)
        if ok.any():
            X_cor_valid = np.stack([cor_epochs[i] for i in np.flatnonzero(ok)]).astype(np.float64)
            p, pr = models_mod.predict(model, X_cor_valid)
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
        row = dict(tf.metrics)
        row["failure_type"] = spec.failure_type
        row["seed"] = spec.seed
        rows.append(row)
    return pd.DataFrame(rows)
