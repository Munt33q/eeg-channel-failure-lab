"""Model: CSP + LDA pipeline; fit on training epochs only; artifact save/load.

- CSP spatial filters are learned from training data exclusively.
- Model artifacts are stored as numeric parameter files (.npz + JSON), NOT
  pickles: loading never executes code, and uploaded pickle/joblib files are
  not accepted. Artifacts produced by other projects are rejected by
  metadata validation (channel order, sfreq, epoch length, labels).
- Predictions never depend on evaluation labels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from mne.decoding import CSP
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .preprocessing import (
    BANDPASS_HZ,
    CHANNEL_ORDER,
    EPOCH_END_S,
    EPOCH_START_S,
    SAMPLES_PER_EPOCH,
    SFREQ_HZ,
    FILTER_ORDER,
)

ARTIFACT_FORMAT_VERSION = 1


def build_pipeline(seed: int = 42) -> "Pipeline":  # type: ignore[name-defined]
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def fit_pipeline(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Fit CSP+LDA on TRAINING epochs only. X: [n, channels, samples] volts."""
    if X_train.ndim != 3 or X_train.shape[2] != SAMPLES_PER_EPOCH:
        raise ValueError(f"training epochs must be [n, {len(CHANNEL_ORDER)}, {SAMPLES_PER_EPOCH}]")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("training labels must contain both classes 0 and 1")
    model = build_pipeline(seed)
    model.fit(X_train, y_train)
    return model


def predict(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, probability_of_class_1). Never uses y_true."""
    preds = model.predict(X).astype(np.int64)
    if hasattr(model, "predict_proba1"):
        return preds, model.predict_proba1(X)
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    col1 = int(np.where(classes == 1)[0][0])
    return preds, proba[:, col1]


# ---------------------------------------------------------------------------
# Artifact persistence (numeric, non-executable format)


def save_model(model, metadata: dict, out_dir: str | Path, subject_id: int) -> tuple[Path, Path]:
    """Save CSP spatial filters + LDA parameters as .npz, plus provenance JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csp: CSP = model.named_steps["csp"]
    lda = model.named_steps["lda"]
    np.savez(
        out_dir / f"model_sub{subject_id}.npz",
        filters_=csp.filters_,
        patterns_=csp.patterns_,
        classes_=np.asarray(model.classes_),
        lda_coef=np.asarray(lda.coef_, dtype=np.float64),
        lda_intercept=np.asarray(lda.intercept_, dtype=np.float64),
        lda_means=np.asarray(lda.means_, dtype=np.float64),
        lda_covariances=np.atleast_2d(np.asarray(lda.covariance_, dtype=np.float64)),
        lda_priors=np.asarray(lda.priors_, dtype=np.float64),
    )
    meta_path = out_dir / f"model_sub{subject_id}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return out_dir / f"model_sub{subject_id}.npz", meta_path


def load_model(models_dir: str | Path, subject_id: int) -> "RestoredModel":
    """Load a project-produced artifact. Rejects pickles by design."""
    models_dir = Path(models_dir)
    npz_path = models_dir / f"model_sub{subject_id}.npz"
    meta_path = models_dir / f"model_sub{subject_id}_metadata.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"model artifact not found for subject {subject_id}; run scripts/train.py first"
        )
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return RestoredModel(npz_path, metadata)


class RestoredModel:
    """Lightweight predictor reconstructed from numeric arrays.

    Reimplements CSP log-variance features + LDA decision exactly as fitted.
    """

    def __init__(self, npz_path: Path, metadata: dict) -> None:
        with np.load(npz_path) as z:
            self.filters_ = z["filters_"][: self._n_components_from_meta(metadata)]
            self.classes_ = z["classes_"]
            self.lda_coef = z["lda_coef"]          # [1, n_components]
            self.lda_intercept = z["lda_intercept"]
            self.lda_means = z["lda_means"]
            self.lda_covariances = z["lda_covariances"]
            self.lda_priors = z["lda_priors"]
        self.metadata = metadata
        self.n_components = self.lda_coef.shape[1]
        self._inv_covs = np.linalg.inv(self.lda_covariances)

    @staticmethod
    def _n_components_from_meta(metadata: dict) -> int:
        return int(metadata.get("contract", {}).get("csp", {}).get("n_components", 4))

    def validate_contract(self) -> None:
        m = self.metadata.get("contract", {})
        expected = {
            "channels": list(CHANNEL_ORDER),
            "sfreq_hz": SFREQ_HZ,
            "epoch_start_s": EPOCH_START_S,
            "epoch_end_s": EPOCH_END_S,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
            "bandpass_hz": list(BANDPASS_HZ),
            "filter_order": FILTER_ORDER,
            "labels": [0, 1],
        }
        for key, want in expected.items():
            got = m.get(key)
            if got != want and not (isinstance(want, float) and isinstance(got, (int, float)) and float(got) == want):
                raise ValueError(
                    f"model artifact contract mismatch on {key!r}: artifact={got!r} expected={want!r}"
                )

    def _features(self, X: np.ndarray) -> np.ndarray:
        # CSP feature extraction identical to mne CSP(log=True):
        # log of the mean band power (squared signal) of spatially filtered
        # signals, per component. Matches mne's _GEDTransformer.transform
        # (filters_[:n_components] @ X) followed by (X**2).mean(axis=2), log.
        if X.ndim != 3 or X.shape[1] != self.filters_.shape[1]:
            raise ValueError("epochs shape does not match model channel count")
        filtered = np.einsum("ij,ejt->eit", self.filters_[: self.n_components], X)
        power = (filtered**2).mean(axis=2)
        return np.log(power)

    def predict(self, X: np.ndarray) -> np.ndarray:
        d = (self.lda_coef @ self._features(X).T + self.lda_intercept).ravel()
        # lsqr LDA decision: sign maps to classes via class ordering
        # sklearn orders decision d < 0 -> classes_[0], d >= 0 -> classes_[1]
        # (verified against fitted pipeline in tests).
        idx = (d >= 0).astype(np.int64)
        return self.classes_[idx].astype(np.int64)

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        d = (self.lda_coef @ self._features(X).T + self.lda_intercept).ravel()
        # sklearn's binary LDA uses sigmoid(decision_function), not a doubled
        # decision value. These are decision probabilities, not calibrated
        # confidence estimates.
        return expit(d)


def provenance_metadata(
    subject_id: int,
    dev_runs: list[int],
    seed: int,
    train_trial_ids: dict[int, list[int]],
    package_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "project": "eeg-channel-failure-lab",
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "subject_id": subject_id,
        "fit_runs": dev_runs,
        "eval_run": 12,
        "seed": seed,
        "train_trial_ids": {str(k): [int(t) for t in v] for k, v in train_trial_ids.items()},
        "package_versions": package_versions,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "channels": list(CHANNEL_ORDER),
            "sfreq_hz": SFREQ_HZ,
            "epoch_start_s": EPOCH_START_S,
            "epoch_end_s": EPOCH_END_S,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
            "bandpass_hz": list(BANDPASS_HZ),
            "filter_order": FILTER_ORDER,
            "labels": [0, 1],
            "referencing": "common average across the nine contract channels",
            "csp": {"n_components": 4, "reg": "ledoit_wolf", "log": True, "norm_trace": False},
            "lda": {"solver": "lsqr", "shrinkage": "auto"},
        },
    }


def package_versions() -> dict[str, str]:
    import numpy, pandas, scipy, sklearn, mne, joblib, matplotlib, streamlit

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "mne": mne.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "streamlit": streamlit.__version__,
    }
