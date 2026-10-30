"""Data loading: PhysioNet EEGMMIDB via MNE eegbci, runs 4/8/12 only.

Event mapping is verified against the PhysioNet documentation before use:
- Runs 4, 8, 12 are Motor IMAGERY: left vs right hand.
- T1 -> left (label 0), T2 -> right (label 1). T0 (rest) is excluded.
- Runs 6/10/14 (hands vs feet) are rejected explicitly and never relabeled.
Real data are downloaded at runtime and never committed to the repository
(license: ODC-By 1.0; attribution required, redistribution of raw files is
not part of this repo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import CHANNEL_ORDER, SFREQ_HZ, RunData, validate_channel_names

IMAGERY_RUNS = {4: "imagery_left_right", 8: "imagery_left_right", 12: "imagery_left_right"}
DEV_RUNS = (4, 8)
EVAL_RUN = 12
FORBIDDEN_RUNS = {6, 10, 14}  # hands vs feet imagery — never mixed in
LEFT, RIGHT = 0, 1


def check_run_allowed(run_id: int) -> None:
    if run_id in FORBIDDEN_RUNS:
        raise ValueError(
            f"run {run_id} is hands-vs-feet imagery and must never be relabeled left/right"
        )
    if run_id not in IMAGERY_RUNS:
        raise ValueError(f"run {run_id} is not a supported left/right imagery run")


def standardize_channels(raw: "object") -> "object":  # mne.io.BaseRaw
    """Rename to standard 10-10 names and keep only the contract channels."""
    mne = _mne()
    raw = raw.copy()
    mne.datasets.eegbci.standardize(raw)
    missing = [c for c in CHANNEL_ORDER if c not in raw.ch_names]
    if missing:
        raise ValueError(f"recording is missing contract channels: {missing}")
    raw.pick(CHANNEL_ORDER)
    raw.reorder_channels(list(CHANNEL_ORDER))
    validate_channel_names(tuple(raw.ch_names))
    return raw


def _mne():
    import mne

    mne.set_log_level("WARNING")
    return mne


def load_run(
    subject_id: int,
    run_id: int,
    data_dir: str | Path | None = None,
) -> RunData:
    """Download (if needed) and load one imagery run as a RunData object.

    Actual sampling rate is read from the recording and verified against the
    contract (160 Hz). Annotations are mapped by DESCRIPTION (T1/T2), never
    by MNE-assigned numeric event ids.
    """
    check_run_allowed(run_id)
    mne = _mne()
    if data_dir is not None:
        # Honor an explicit download/cache directory (must exist for MNE).
        dd = Path(data_dir)
        dd.mkdir(parents=True, exist_ok=True)
        mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(dd.resolve()))
    paths = mne.datasets.eegbci.load_data(subjects=subject_id, runs=[run_id])
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
    raw = standardize_channels(raw)

    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, SFREQ_HZ, rtol=0, atol=1e-6):
        raise ValueError(f"recording sampling rate {sfreq} Hz != contract {SFREQ_HZ} Hz")

    # Annotation -> label mapping by description, verified against dataset docs.
    labels: list[int] = []
    samples: list[int] = []
    descriptions: list[str] = []
    for ann in raw.annotations:
        desc = ann["description"]
        if desc == "T0":
            continue  # rest, excluded
        if desc == "T1":
            label = LEFT
        elif desc == "T2":
            label = RIGHT
        else:
            raise ValueError(f"unexpected annotation {desc!r} in S{subject_id:03d}R{run_id:02d}")
        labels.append(label)
        samples.append(int(round(ann["onset"] * sfreq)))
        descriptions.append(desc)
    if not labels:
        raise ValueError(f"no T1/T2 events found in S{subject_id:03d}R{run_id:02d}")

    data_v = raw.get_data().astype(np.float64)  # volts, [channels, samples]
    return RunData(
        data_v=data_v,
        channel_names=tuple(raw.ch_names),
        sfreq_hz=sfreq,
        subject_id=subject_id,
        run_id=run_id,
        event_samples=np.asarray(samples, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        descriptions=tuple(descriptions),
    )


def load_subject(
    subject_id: int,
    data_dir: str | Path | None = None,
) -> dict[int, RunData]:
    """Load dev runs (4, 8) and the held-out eval run (12) for one subject."""
    return {run: load_run(subject_id, run, data_dir) for run in (*DEV_RUNS, EVAL_RUN)}


def run_signature(run: RunData) -> dict:
    """Identity metadata persisted with every experiment."""
    return {
        "subject_id": run.subject_id,
        "run_id": run.run_id,
        "sfreq_hz": run.sfreq_hz,
        "n_channels": len(run.channel_names),
        "channel_names": list(run.channel_names),
        "n_events": int(len(run.event_samples)),
        "label_counts": {int(k): int(v) for k, v in zip(*np.unique(run.labels, return_counts=True))},
    }
