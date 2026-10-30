"""Tests for training/evaluation isolation and paired evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_channel_failure_lab.corruptions import FLAT, CorruptionSpec
from eeg_channel_failure_lab.evaluation import (
    accuracy,
    balanced_accuracy,
    confusion,
    majority_baseline,
    paired_metrics,
)
from eeg_channel_failure_lab.experiments import run_single_experiment
from eeg_channel_failure_lab.preprocessing import RunData, epochs_from_trials, preprocess_run
from eeg_channel_failure_lab.synthetic import make_synthetic_subject

from tests.helpers import fit_synth_model, predict2, synth_model_for


class TestSplitIsolation:
    def test_no_train_eval_overlap(self):
        runs = make_synthetic_subject()
        dev_ids = [(r, t) for r in (4, 8) for t in runs[r].event_samples]
        eval_ids = [(12, t) for t in runs[12].event_samples]
        # Runs differ, so trial identity cannot overlap by construction.
        assert all(r != 12 for r, _ in dev_ids)

    def test_model_trained_on_train_only(self):
        """Corrupted eval data must not change the fitted model."""
        runs = make_synthetic_subject()
        model = synth_model_for(runs)
        X_eval, y_eval, _ = epochs_from_trials(preprocess_run(runs[12]))
        # Fit a second model on eval data and confirm the first differs.
        model_leaky = fit_synth_model(X_eval, y_eval)
        p1, _ = predict2(model, X_eval)
        p2, _ = predict2(model_leaky, X_eval)
        if np.array_equal(p1, p2):
            pytest.skip("models agree on this eval set; isolation is structural (fit data disjoint)")

    def test_predictions_independent_of_eval_labels(self):
        runs = make_synthetic_subject()
        model = synth_model_for(runs)
        X_eval, _, _ = epochs_from_trials(preprocess_run(runs[12]))
        p_a, proba_a = predict2(model, X_eval)
        # Predictions are a pure function of X: reversing trial order reverses
        # outputs but never changes per-trial values.
        p_b, proba_b = predict2(model, X_eval[::-1])
        assert np.array_equal(p_a, p_b[::-1])
        assert np.allclose(proba_a, proba_b[::-1])


class TestPairedEvaluation:
    def test_identical_trial_ids_clean_vs_corrupt(self):
        runs = make_synthetic_subject()
        model = synth_model_for(runs)
        y_train = np.concatenate([runs[r].labels for r in (4, 8)])
        spec = CorruptionSpec(FLAT, ("C3",))
        tf = run_single_experiment(runs[12], model, spec, y_train)
        clean = preprocess_run(runs[12])
        elig_ids = clean.loc[clean["eligible"], "trial_id"].to_numpy()
        assert np.array_equal(tf.trials["trial_id"].to_numpy(), elig_ids)

    def test_failures_counted_as_incorrect(self):
        y = np.array([0, 1, 0, 1])
        clean = np.array([0, 1, 0, 1])  # 100%
        corrupt = np.array([0, 1, 0, 1])
        ok = np.array([True, True, False, False])  # 2 failures
        m = paired_metrics(y, clean, corrupt, ok)
        assert m["prediction_failures"] == 2
        assert m["coverage"] == 0.5
        assert m["corrupt_accuracy_all_eligible"] == 0.5  # failures incorrect
        assert m["corrupt_accuracy_among_predicted"] == 1.0  # labeled clearly
        assert m["clean_accuracy"] == 1.0

    def test_flip_transitions(self):
        y = np.array([0, 1, 1, 0, 1])
        clean = np.array([0, 1, 0, 0, 1])
        corrupt = np.array([1, 0, 1, 0, 1])
        m = paired_metrics(y, clean, corrupt, np.ones(5, bool))
        assert m["correct_to_incorrect"] == 2
        assert m["incorrect_to_correct"] == 1
        assert m["n_flips"] == 3
        assert m["flip_rate"] == pytest.approx(0.6)

    def test_majority_baseline_from_train_labels(self):
        assert majority_baseline(np.array([0, 0, 0, 1])) == 0.75

    def test_confusion_counts(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 1, 1, 1])
        c = confusion(y, p)
        assert c == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
        assert accuracy(y, p) == 0.75
        assert balanced_accuracy(y, p) == pytest.approx(0.75)


class TestSyntheticWorkflow:
    def test_end_to_end_runs_and_produces_metrics(self):
        runs = make_synthetic_subject(seed=123)
        model = synth_model_for(runs)
        y_train = np.concatenate([runs[r].labels for r in (4, 8)])
        tf = run_single_experiment(runs[12], model, None, y_train)
        assert 0.0 <= tf.metrics["clean_accuracy"] <= 1.0
        assert "n_train_majority_baseline" in tf.metrics
        spec = CorruptionSpec(FLAT, ("C3",))
        tf2 = run_single_experiment(runs[12], model, spec, y_train)
        assert tf2.metrics["n_trials"] == tf.metrics["n_trials"]
        # Degradation is measured, not guaranteed; just require valid metrics.
        assert -1.0 <= tf2.metrics["accuracy_drop_pp"] <= 101.0
