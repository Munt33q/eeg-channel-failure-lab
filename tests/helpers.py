"""Shared test helpers (predict2, synth_model_for, fit_synth_model)."""

from __future__ import annotations

import numpy as np

from eeg_channel_failure_lab import models as models_mod
from eeg_channel_failure_lab.preprocessing import epochs_from_trials, preprocess_run


def fit_synth_model(X: np.ndarray, y: np.ndarray):
    return models_mod.fit_pipeline(X, y, seed=42)


def synth_model_for(runs: dict):
    """Fit CSP+LDA on dev runs (keys 4 and 8) only."""
    X_parts, y_parts = [], []
    for r in (4, 8):
        trials = preprocess_run(runs[r])
        X, y, _ = epochs_from_trials(trials)
        X_parts.append(X)
        y_parts.append(y)
    return fit_synth_model(np.concatenate(X_parts), np.concatenate(y_parts))


def predict2(model, X):
    """(preds, proba1) for any fitted/restored model."""
    return models_mod.predict(model, X)
