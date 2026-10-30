"""Deterministic synthetic EEG-like data for the offline demonstration.

Generates multichannel band-limited signals with an 8-30 Hz event-related
oscillation whose lateralization encodes the label, plus noise. Everything is
seeded and fully offline. The demo classifier is the same CSP+LDA pipeline.

Label prominently in UI: "Synthetic demonstration - not evidence of
performance on human EEG." Metrics from synthetic mode are kept separate from
real-data results everywhere.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import CHANNEL_ORDER, RunData, SAMPLES_PER_EPOCH, SFREQ_HZ

SYNTH_BANNER = "Synthetic demonstration — not evidence of performance on human EEG."


def _band_noise(rng: np.random.Generator, n_samples: int, n_channels: int, sfreq: float, sd_uv: float) -> np.ndarray:
    """Gaussian noise, then causal bandpass 8-30 Hz to mimic EEG background."""
    from scipy.signal import butter, sosfilt

    x = rng.normal(0.0, sd_uv, size=(n_channels, n_samples))
    sos = butter(4, (8.0, 30.0), btype="bandpass", output="sos", fs=sfreq)
    return sosfilt(sos, x, axis=-1)


def generate_synthetic_run(
    seed: int,
    n_trials: int = 30,
    sfreq: float = SFREQ_HZ,
    effect_uv: float = 0.6,
    noise_uv: float = 1.2,
    label_bias: float = 0.5,
) -> RunData:
    """Generate one synthetic 'run'.

    Mu-like 10-20 Hz bursts with amplitude modulated over the channels
    contralateral to the imagined hand: label 0 (left) -> mu on C4/FC4/CP4
    side, label 1 (right) -> mu on C3/FC3/CP3 side. Effect size is small
    relative to noise so the classifier is imperfect (measured, not rigged).
    """
    rng = np.random.default_rng(seed)
    # Inter-trial intervals: cue + 4 s per trial; pre-cue baseline of 2 s.
    trial_len_s = 4.0
    n_total = int((2.0 + n_trials * trial_len_s) * sfreq)
    n_ch = len(CHANNEL_ORDER)
    data = _band_noise(rng, n_total, n_ch, sfreq, noise_uv)

    # Lateralization weights (µV scale amplitude modulation per channel).
    lat = np.zeros(n_ch)
    for i, ch in enumerate(CHANNEL_ORDER):
        if ch.endswith("3"):
            lat[i] = 1.0
        elif ch.endswith("4"):
            lat[i] = -1.0
        # midline (z) channels get 0

    t = np.arange(SAMPLES_PER_EPOCH) / sfreq
    envelope = np.exp(-0.5 * ((t - 1.8) / 0.6) ** 2)  # bursts after the cue
    carrier = np.sin(2 * np.pi * 12.0 * t)

    samples: list[int] = []
    labels: list[int] = []
    onset = int(2.0 * sfreq)
    for i in range(n_trials):
        label = int(rng.random() < label_bias)  # ~balanced labels
        lat_sign = -lat if label == 1 else lat  # contralateral mu modulation
        burst = effect_uv * envelope * carrier  # [samples]
        s0 = onset + i * int(trial_len_s * sfreq)
        data[:, s0 : s0 + SAMPLES_PER_EPOCH] += np.outer(lat_sign, burst)
        samples.append(s0)
        labels.append(label)

    return RunData(
        data_v=data * 1e-6,  # store volts, consistent with the real-data path
        channel_names=CHANNEL_ORDER,
        sfreq_hz=sfreq,
        subject_id=0,
        run_id=seed,
        event_samples=np.asarray(samples, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        descriptions=tuple("synthetic" for _ in labels),
    )


def make_synthetic_subject(seed: int = 100) -> dict[int, RunData]:
    """Dev runs (4, 8 -> keys reused) and held-out run (key 12) for demo mode."""
    return {
        4: generate_synthetic_run(seed + 1, n_trials=30),
        8: generate_synthetic_run(seed + 2, n_trials=30),
        12: generate_synthetic_run(seed + 3, n_trials=24),
    }
