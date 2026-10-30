"""Deterministic corruption functions applied to raw channel data.

Failures are applied to raw selected-channel data BEFORE referencing,
filtering, and epoching (guide section on corruption placement). Clean and
corrupted versions must be processed independently through the same
preprocessing pipeline. Corruption functions are pure: they never mutate the
input array and never depend on evaluation labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FLAT = "flat"
NOISE = "noise"
GAIN = "gain"
SWAP = "swap"

FAILURE_TYPES = (FLAT, NOISE, GAIN, SWAP)

# Documented severity grids.
NOISE_GRID_UV = (0.0, 5.0, 10.0, 20.0, 50.0)
GAIN_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# Fixed noise grid for repeated seeds in sensitivity sweeps.
DEFAULT_SEEDS = (7, 19, 42)


@dataclass(frozen=True)
class CorruptionSpec:
    """Full specification of one simulated channel failure.

    Everything needed to reproduce the corrupted signal exactly: type,
    target channels, severity, seed, units, and the affected half-open
    time interval. ``interval`` is (start_s, end_s) in seconds relative to
    the raw segment start; (0.0, None) means the entire segment.
    """

    failure_type: str
    channels: tuple[str, ...]
    severity: float | None = None  # µV SD for noise, gain factor for gain
    seed: int = 42
    units: str = "µV"
    interval: tuple[float, float | None] = (0.0, None)

    def __post_init__(self) -> None:
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"unknown failure type {self.failure_type!r}")
        if self.failure_type == NOISE and (self.severity is None or self.severity < 0):
            raise ValueError("noise failure requires severity = noise SD in µV >= 0")
        if self.failure_type == GAIN and (self.severity is None or self.severity < 0):
            raise ValueError("gain failure requires severity = gain factor >= 0")
        if self.failure_type == FLAT and self.severity is not None and self.severity != 0:
            raise ValueError("flat failure takes no severity")
        if self.failure_type == SWAP and len(self.channels) != 2:
            raise ValueError("swap requires exactly two channel names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "channels": list(self.channels),
            "severity": self.severity,
            "seed": self.seed,
            "units": self.units,
            "interval": list(self.interval),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorruptionSpec":
        return cls(
            failure_type=d["failure_type"],
            channels=tuple(d["channels"]),
            severity=d.get("severity"),
            seed=int(d.get("seed", 42)),
            units=d.get("units", "µV"),
            interval=tuple(d.get("interval", (0.0, None))),
        )

    def describe(self) -> str:
        ch = ", ".join(self.channels)
        if self.failure_type == FLAT:
            return f"Flat channel: {ch} replaced with constant 0 µV (simulated flatline, not physical removal)"
        if self.failure_type == NOISE:
            return f"Additive noise: Gaussian noise, SD {self.severity} µV, seed {self.seed}, channel(s) {ch}"
        if self.failure_type == GAIN:
            return f"Gain error: channels {ch} multiplied by factor {self.severity}"
        return f"Channel swap: signals of {ch[0]} and {ch[1]} exchanged (simulated channel-assignment error)"


def _resolve_indices(channel_names: tuple[str, ...], channels: tuple[str, ...]) -> list[int]:
    if not channels:
        raise ValueError("no channels selected for corruption")
    unknown = [c for c in channels if c not in channel_names]
    if unknown:
        raise ValueError(f"channel(s) {unknown} not in recording channels {list(channel_names)}")
    return [channel_names.index(c) for c in channels]


def _sample_slice(data: np.ndarray, sfreq_hz: float, interval: tuple[float, float | None]) -> slice:
    n = data.shape[-1]
    start = int(round(interval[0] * sfreq_hz))
    end = n if interval[1] is None else min(n, int(round(interval[1] * sfreq_hz)))
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    return slice(start, end)


def apply_corruption(
    data: np.ndarray,
    channel_names: tuple[str, ...],
    spec: CorruptionSpec,
    sfreq_hz: float,
    units: str = "µV",
) -> np.ndarray:
    """Return a corrupted COPY of ``data`` [channels, samples].

    The input array is never mutated. ``units`` states the physical unit of
    the data array so noise severity (µV) and constants are converted
    explicitly. Gain of 1.0 and noise SD of 0 are exact identity operations.
    A swap applied twice (via the same spec) restores the original signal.
    """
    if units not in ("µV", "V"):
        raise ValueError(f"unsupported unit {units!r}; expected 'µV' or 'V'")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be 2-D [channels, samples]")
    out = data.copy()  # original stays immutable
    idx = _resolve_indices(channel_names, spec.channels)
    sl = _sample_slice(out, sfreq_hz, spec.interval)
    scale = 1.0 if units == "µV" else 1e-6  # µV severity -> volts when data unit is V

    if spec.failure_type == FLAT:
        const_v = 0.0
        for i in idx:
            out[i, sl] = const_v
        return out

    if spec.failure_type == NOISE:
        sev_uV = spec.severity
        if sev_uV == 0.0:
            return out  # exact identity
        sd = sev_uV * scale  # µV -> data unit
        rng = np.random.default_rng(spec.seed)
        shape = (len(idx), sl.stop - sl.start)
        noise = rng.normal(loc=0.0, scale=sd, size=shape)
        for k, i in enumerate(idx):
            out[i, sl] = out[i, sl] + noise[k]
        return out

    if spec.failure_type == GAIN:
        if spec.severity == 1.0:
            return out  # exact identity
        for i in idx:
            out[i, sl] = out[i, sl] * spec.severity
        return out

    # SWAP: exchange signals of the two named channels for the interval.
    a, b = idx
    tmp = out[a, sl].copy()
    out[a, sl] = out[b, sl]
    out[b, sl] = tmp
    return out


def default_spec(failure_type: str, channels: tuple[str, ...], severity: float | None = None, seed: int = 42) -> CorruptionSpec:
    return CorruptionSpec(
        failure_type=failure_type,
        channels=tuple(channels),
        severity=severity,
        seed=seed,
    )
