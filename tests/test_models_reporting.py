"""Tests for artifact validation and reporting security."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from eeg_channel_failure_lab import models as models_mod
from eeg_channel_failure_lab.preprocessing import CHANNEL_ORDER, SFREQ_HZ
from eeg_channel_failure_lab.reporting import config_json, esc, metrics_csv, standalone_html
from eeg_channel_failure_lab.synthetic import make_synthetic_subject

from tests.helpers import synth_model_for


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    runs = make_synthetic_subject(seed=77)
    model = synth_model_for(runs)
    from eeg_channel_failure_lab.evaluation import majority_baseline

    meta = models_mod.provenance_metadata(
        subject_id=0,
        dev_runs=[4, 8],
        seed=42,
        train_trial_ids={"4": [0, 1, 2], "8": [0, 1, 2]},
        package_versions=models_mod.package_versions(),
    )
    npz, mj = models_mod.save_model(model, meta, out, 0)
    return out, runs, model


class TestArtifacts:
    def test_roundtrip_predictions_match_pipeline(self, trained):
        out, runs, model = trained
        restored = models_mod.load_model(out, 0)
        X, _, _ = __import__(
            "eeg_channel_failure_lab.preprocessing", fromlist=["epochs_from_trials"]
        ).epochs_from_trials(__import__(
            "eeg_channel_failure_lab.preprocessing", fromlist=["preprocess_run"]
        ).preprocess_run(runs[12]))
        p_pipe, pr_pipe = models_mod.predict(model, X)
        p_res = restored.predict(X)
        pr_res = restored.predict_proba1(X)
        # Numeric artifact restoration must preserve both labels and the
        # sklearn LDA probability output exactly.
        assert np.array_equal(p_pipe, p_res)
        assert np.allclose(pr_pipe, pr_res, rtol=1e-10, atol=1e-12)

    def test_contract_mismatch_rejected(self, trained):
        out, _, _ = trained
        restored = models_mod.load_model(out, 0)
        restored.metadata["contract"]["channels"] = ["C3", "Cz", "C4"]
        with pytest.raises(ValueError, match="contract mismatch"):
            restored.validate_contract()

    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            models_mod.load_model(tmp_path, 99)

    def test_artifacts_are_not_pickles(self, trained):
        out, _, _ = trained
        npz = out / "model_sub0.npz"
        with open(npz, "rb") as fh:
            head = fh.read(4)
        # npz is a zip; pickle files start with \x80 (protocol marker)
        assert head[:2] == b"PK"
        meta = json.loads((out / "model_sub0_metadata.json").read_text())
        assert meta["contract"]["channels"] == list(CHANNEL_ORDER)
        assert meta["contract"]["sfreq_hz"] == SFREQ_HZ


class TestReportingSecurity:
    def test_escapes_channel_names(self):
        malicious = "<script>alert('x')</script>"
        assert "<script>" not in esc(malicious)
        assert "&lt;script&gt;" in esc(malicious)

    def test_html_report_escapes_input_text(self):
        evil_channel = 'C3<img src=x onerror=alert(1)>'
        metadata = {"channel": evil_channel, "montage": CHANNEL_ORDER}
        html_out = standalone_html("t", [("s", "<b>safe</b>")], metadata)
        assert "<img src=x" not in html_out
        assert esc(evil_channel) in html_out
        # No external resources
        assert "http://" not in html_out and "https://" not in html_out

    def test_csv_and_json_roundtrip(self):
        rows = [{"acc": 0.5, "cm": {"tn": 1}}]
        csv_text = metrics_csv(rows)
        assert "0.5" in csv_text
        cfg = config_json({"a": 1})
        assert json.loads(cfg)["a"] == 1
