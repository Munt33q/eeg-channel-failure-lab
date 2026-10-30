"""Preprocessing contract: channel order, referencing, causal filtering, epochs.

Fixed contract (guide):
- Nine channels, in this exact order: FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4.
- Explicit physical units. Internal computation in volts (MNE convention);
  corruption severities are specified and displayed in µV.
- Referencing: per-sample common average across those nine channels.
- Filter: 4th-order Butterworth bandpass 8-30 Hz as SOS applied with
  scipy.signal.sosfilt (forward-only, causal). Never filtfilt. Filter state
  resets at each run boundary.
- Epochs: [1.0, 3.0) s after cue onset -> 320 samples at 160 Hz.
- Corruption is applied to raw data BEFORE referencing and filtering; clean
  and corrupted data flow through the identical pipeline independently.
- Note: common-average referencing spreads one channel's corruption into the
  other channels; this is visible when comparing raw vs processed views.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

CHANNEL_ORDER: tuple[str, ...] = (
    "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4",
)
SFREQ_HZ = 160.0
EPOCH_START_S = 1.0
EPOCH_END_S = 3.0
SAMPLES_PER_EPOCH = int(round((EPOCH_END_S - EPOCH_START_S) * SFREQ_HZ))  # 320
BANDPASS_HZ = (8.0, 30.0)
FILTER_ORDER = 4
SETTLING_S = 1.0  # conservative causal-filter settling window

# Quality checks on clean data determine eligibility. Frozen from clean runs:
# corrupted trials keep the eligibility of their clean counterparts.
STD_EPS_V = 1e-12   # channel SD (V) at or below this is "flat"
ABS_LIMIT_V = 1e-3  # |amplitude| beyond 1 mV after bandpass is out of bounds

REASON_NONFINITE = "nonfinite"
REASON_FLAT = "flat"
REASON_OUT_OF_BOUNDS = "out_of_bounds"
REASON_INCOMPLETE = "incomplete"
REASON_SETTLING = "filter_settling"


def validate_channel_names(channel_names: tuple[str, ...]) -> None:
    if tuple(channel_names) != CHANNEL_ORDER:
        raise ValueError(
            f"channel order mismatch: expected {list(CHANNEL_ORDER)}, got {list(channel_names)}"
        )


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract per-sample mean across the nine channels. Pure function."""
    return data - data.mean(axis=0, keepdims=True)


def design_bandpass(sfreq_hz: float) -> np.ndarray:
    return butter(FILTER_ORDER, BANDPASS_HZ, btype="bandpass", output="sos", fs=sfreq_hz)


def causal_bandpass(data: np.ndarray, sfreq_hz: float, zi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forward-only (causal) filtering with carry-over state.

    ``zi`` carries filter state between chunks/runs; pass zi=None at each run
    boundary to reset. Returns (filtered, zi).
    """
    sos = design_bandpass(sfreq_hz)
    if zi is None:
        zi = np.zeros((sos.shape[0], data.shape[0], 2), dtype=np.float64)
    out, zi = sosfilt(sos, data, axis=-1, zi=zi)
    return out, zi


def check_epoch_quality(epoch_v: np.ndarray) -> str:
    """Return an exclusion reason code, or '' if acceptable. Used on CLEAN data."""
    if not np.isfinite(epoch_v).all():
        return REASON_NONFINITE
    if np.any(epoch_v.std(axis=1) <= STD_EPS_V):
        return REASON_FLAT
    if np.any(np.abs(epoch_v) > ABS_LIMIT_V):
        return REASON_OUT_OF_BOUNDS
    return ""


@dataclass
class RunData:
    """One continuous run of selected-channel raw data plus metadata."""

    data_v: np.ndarray          # [channels, samples] volts, immutable intent
    channel_names: tuple[str, ...]
    sfreq_hz: float
    subject_id: int
    run_id: int
    event_samples: np.ndarray   # cue onsets in samples
    labels: np.ndarray          # 0=left, 1=right
    descriptions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_channel_names(self.channel_names)
        if self.data_v.ndim != 2 or self.data_v.shape[0] != len(self.channel_names):
            raise ValueError("data must be [channels, samples] matching channel_names")
        if len(self.event_samples) != len(self.labels):
            raise ValueError("event_samples and labels must have equal length")


def preprocess_run(
    run: RunData,
    corrupted: np.ndarray | None = None,
) -> pd.DataFrame:
    """Reference, causally filter, and epoch a run.

    ``corrupted`` (volts, same shape) replaces run.data_v when provided. The
    corrupted array must be produced by corruptions.apply_corruption on the
    RAW data BEFORE this function. Clean and corrupted data are processed
    through the identical pipeline here.

    Returns a per-trial DataFrame with epoch arrays (volts), eligibility from
    CLEAN-data checks only, and failure accounting for corrupted paths.
    """
    data = run.data_v if corrupted is None else corrupted
    validate_channel_names(run.channel_names)
    if data.shape != run.data_v.shape:
        raise ValueError("corrupted data shape does not match clean data shape")
    fs = run.sfreq_hz
    start_samp = int(round(EPOCH_START_S * fs))
    end_samp = int(round(EPOCH_END_S * fs))

    # Common-average reference, then causal filter with state reset per run.
    ref = common_average_reference(data)
    filt, _ = causal_bandpass(ref, fs, zi=None)

    rows: list[dict] = []
    settling_samples = int(round(SETTLING_S * fs))
    for i, onset in enumerate(run.event_samples):
        s0, s1 = int(onset) + start_samp, int(onset) + end_samp
        reason = ""
        epoch = None
        if s0 < 0:
            reason = REASON_OUT_OF_BOUNDS
        elif s1 > data.shape[1]:
            reason = REASON_INCOMPLETE
        else:
            epoch = filt[:, s0:s1].astype(np.float64, copy=True)
            if corrupted is None:
                # Eligibility frozen from clean-data checks only.
                reason = check_epoch_quality(epoch)
            else:
                # Corrupted path: report model-level validity separately in
                # evaluation; do not silently drop trials here.
                reason = ""
        if reason == "" and s0 < settling_samples:
            reason = REASON_SETTLING
        rows.append(
            {
                "subject_id": run.subject_id,
                "run_id": run.run_id,
                "trial_id": i,
                "event_sample": int(onset),
                "onset_s": float(onset) / fs,
                "label": int(run.labels[i]),
                "description": run.descriptions[i] if run.descriptions else "",
                "eligible": reason == "",
                "exclusion_reason": reason,
                "epoch": epoch,
                "model_valid": True if epoch is None else bool(np.isfinite(epoch).all()),
            }
        )
    return pd.DataFrame(rows)


def epochs_from_trials(trials: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X [n, ch, samples] volts, y, trial_ids) for eligible trials."""
    ok = trials["eligible"].to_numpy() & trials["epoch"].notna().to_numpy()
    X = np.stack([e for e in trials.loc[ok, "epoch"]]).astype(np.float64)
    y = trials.loc[ok, "label"].to_numpy(dtype=np.int64)
    ids = trials.loc[ok, "trial_id"].to_numpy(dtype=np.int64)
    return X, y, ids
