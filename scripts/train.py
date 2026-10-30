"""Train subject-specific CSP+LDA models on runs 4+8 (dev) only.

The held-out run 12 is never touched here. Saves numeric (non-pickle)
artifacts under models/ with full provenance metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_channel_failure_lab import data as data_mod
from eeg_channel_failure_lab import models as models_mod
from eeg_channel_failure_lab.evaluation import majority_baseline
from eeg_channel_failure_lab.preprocessing import epochs_from_trials, preprocess_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    versions = models_mod.package_versions()
    for subj in args.subjects:
        dev_runs = data_mod.load_subject(subj, data_dir=args.data_dir)
        X_parts, y_parts, ids = [], [], []
        for run_id in data_mod.DEV_RUNS:
            run = dev_runs[run_id]
            trials = preprocess_run(run)
            X, y, tids = epochs_from_trials(trials)
            X_parts.append(X)
            y_parts.append(y)
            ids.append((run_id, tids))
        import numpy as np

        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)
        model = models_mod.fit_pipeline(X, y)
        meta = models_mod.provenance_metadata(
            subject_id=subj,
            dev_runs=list(data_mod.DEV_RUNS),
            seed=42,
            train_trial_ids={str(r): t.tolist() for r, t in ids},
            package_versions=versions,
        )
        npz_path, meta_path = models_mod.save_model(model, meta, args.models_dir, subj)
        print(
            f"subject {subj}: trained on {len(y)} trials (runs 4+8), "
            f"majority baseline {majority_baseline(y):.3f}; saved {npz_path.name}"
        )


if __name__ == "__main__":
    main()
