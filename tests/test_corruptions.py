"""Tests for corruption mechanics and preprocessing contracts."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_channel_failure_lab.corruptions import (
    FLAT,
    GAIN,
    NOISE,
    SWAP,
    CorruptionSpec,
    apply_corruption,
)
from eeg_channel_failure_lab.preprocessing import (
    CHANNEL_ORDER,
    SAMPLES_PER_EPOCH,
    SFREQ_HZ,
    RunData,
    causal_bandpass,
    check_epoch_quality,
    common_average_reference,
    preprocess_run,
)

CH = CHANNEL_ORDER
N = 2000


def make_data(seed: int = 0) -> np.ndarray:
    """Random smooth-ish data in µV, [9, N]."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 20, size=(len(CH), N))
    # Keep it band-limited enough to pass quality checks after filtering.
    from scipy.signal import butter, sosfilt

    sos = butter(4, (8.0, 30.0), btype="bandpass", output="sos", fs=SFREQ_HZ)
    return sosfilt(sos, x, axis=-1)


def make_run(data_uv: np.ndarray, n_events: int = 5, seed: int = 1) -> RunData:
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.arange(600, N - SAMPLES_PER_EPOCH - 10), size=n_events, replace=False)
    samples.sort()
    labels = rng.integers(0, 2, size=n_events)
    return RunData(
        data_v=data_uv * 1e-6,
        channel_names=CH,
        sfreq_hz=SFREQ_HZ,
        subject_id=0,
        run_id=1,
        event_samples=samples.astype(np.int64),
        labels=labels.astype(np.int64),
        descriptions=tuple("T1" for _ in labels),
    )


class TestCorruptions:
    def test_flat_affects_only_target_channel(self):
        data = make_data()
        out = apply_corruption(data, CH, CorruptionSpec(FLAT, ("C3",)), SFREQ_HZ, units="µV")
        i = CH.index("C3")
        assert np.all(out[i] == 0.0)
        for j, ch in enumerate(CH):
            if j != i:
                assert np.array_equal(out[j], data[j])

    def test_gain_identity(self):
        data = make_data()
        out = apply_corruption(data, CH, CorruptionSpec(GAIN, ("Cz",), severity=1.0), SFREQ_HZ, units="µV")
        assert np.array_equal(out, data)

    def test_noise_zero_identity(self):
        data = make_data()
        out = apply_corruption(data, CH, CorruptionSpec(NOISE, ("Cz",), severity=0.0), SFREQ_HZ, units="µV")
        assert np.array_equal(out, data)

    def test_swap_twice_restores(self):
        data = make_data()
        spec = CorruptionSpec(SWAP, ("C3", "C4"))
        once = apply_corruption(data, CH, spec, SFREQ_HZ, units="µV")
        twice = apply_corruption(once, CH, spec, SFREQ_HZ, units="µV")
        assert np.array_equal(twice, data)
        i, k = CH.index("C3"), CH.index("C4")
        assert np.array_equal(once[i], data[k])
        assert np.array_equal(once[k], data[i])

    def test_input_not_mutated(self):
        data = make_data()
        before = data.copy()
        for spec in [
            CorruptionSpec(FLAT, ("Cz",)),
            CorruptionSpec(NOISE, ("Cz",), severity=25.0),
            CorruptionSpec(GAIN, ("Cz",), severity=3.0),
            CorruptionSpec(SWAP, ("FC3", "CP4")),
        ]:
            apply_corruption(data, CH, spec, SFREQ_HZ, units="µV")
        assert np.array_equal(data, before)

    def test_identical_seeds_reproduce_noise(self):
        data = make_data()
        s = CorruptionSpec(NOISE, ("C4",), severity=15.0, seed=123)
        a = apply_corruption(data, CH, s, SFREQ_HZ, units="µV")
        b = apply_corruption(data, CH, s, SFREQ_HZ, units="µV")
        assert np.array_equal(a, b)
        s_other = CorruptionSpec(NOISE, ("C4",), severity=15.0, seed=124)
        c = apply_corruption(data, CH, s_other, SFREQ_HZ, units="µV")
        assert not np.array_equal(a, c)

    def test_noise_proportional_to_severity(self):
        data = np.zeros((len(CH), N))
        sd_small = apply_corruption(data, CH, CorruptionSpec(NOISE, ("C3",), severity=5.0), SFREQ_HZ, units="µV")[CH.index("C3")].std()
        sd_big = apply_corruption(data, CH, CorruptionSpec(NOISE, ("C3",), severity=50.0), SFREQ_HZ, units="µV")[CH.index("C3")].std()
        assert sd_big / sd_small == pytest.approx(10.0, rel=0.2)

    def test_unknown_channel_rejected(self):
        data = make_data()
        with pytest.raises(ValueError, match="not in recording"):
            apply_corruption(data, CH, CorruptionSpec(FLAT, ("X9",)), SFREQ_HZ, units="µV")

    def test_swap_requires_two_channels(self):
        with pytest.raises(ValueError):
            CorruptionSpec(SWAP, ("C3",))

    def test_interval_limits_effect(self):
        data = make_data()
        spec = CorruptionSpec(FLAT, ("C3",), interval=(0.0, 0.5))
        out = apply_corruption(data, CH, spec, SFREQ_HZ, units="µV")
        i = CH.index("C3")
        assert np.all(out[i, :80] == 0.0)
        assert not np.any(out[i, 80:] == 0.0)


class TestUnits:
    def test_unit_conversion_v(self):
        """Spec severity in µV converts correctly when data are in volts."""
        data = np.zeros((len(CH), N))  # volts would be tiny; here we pass unit="V"
        spec = CorruptionSpec(NOISE, ("C3",), severity=10.0)
        # With unit="V", severity 10 µV would be interpreted as volts *only if*
        # the caller converted. The function's unit parameter declares the
        # data unit; severity is always µV -> sd in data units = 10 µV.
        out = apply_corruption(data, CH, spec, SFREQ_HZ, units="µV")
        sd = out[CH.index("C3")].std()  # in µV since data unit is µV
        assert sd == pytest.approx(10.0, rel=0.2)


class TestPreprocessing:
    def test_raw_run_data_is_immutable(self):
        run = make_run(make_data())
        with pytest.raises(ValueError):
            run.data_v[0, 0] = 0.0

    def test_channel_order_mismatch_rejected(self):
        data = make_data()
        make_run(data)
        # RunData.__post_init__ rejects a wrong channel order at construction.
        with pytest.raises(ValueError, match="channel order mismatch"):
            RunData(
                data_v=data * 1e-6,
                channel_names=("C3",) + CH[1:],  # wrong: first channel should be FC3
                sfreq_hz=SFREQ_HZ,
                subject_id=0,
                run_id=1,
                event_samples=np.arange(600, 600 + 5 * 640, 640, dtype=np.int64),
                labels=np.zeros(5, dtype=np.int64),
            )

    def test_causal_filter_differs_from_filtfilt(self):
        from scipy.signal import butter, sosfiltfilt

        data = make_data()
        sos = butter(4, (8.0, 30.0), btype="bandpass", output="sos", fs=SFREQ_HZ)
        fwd, _ = causal_bandpass(data, SFREQ_HZ)
        zero = sosfiltfilt(sos, data, axis=-1)
        assert not np.allclose(fwd, zero)

    def test_causal_output_depends_only_on_past(self):
        data = make_data()
        full, _ = causal_bandpass(data, SFREQ_HZ)
        part = data[:, :1000]
        chunk1, zi = causal_bandpass(part, SFREQ_HZ)
        assert np.allclose(chunk1, full[:, :1000])
        chunk2, _ = causal_bandpass(data[:, 1000:], SFREQ_HZ, zi=zi)
        assert np.allclose(chunk2, full[:, 1000:])

    def test_common_average_spreads_flatline(self):
        """Referencing AFTER corruption spreads it; before, it stays local."""
        data = make_data()
        flat = apply_corruption(data, CH, CorruptionSpec(FLAT, ("C3",)), SFREQ_HZ, units="µV")
        ref = common_average_reference(flat)
        # Other channels now differ from their referenced clean version.
        ref_clean = common_average_reference(data)
        j = CH.index("Cz")
        assert not np.allclose(ref[j], ref_clean[j])
        # C3 itself: its flat constant gets referenced too.
        assert not np.allclose(ref[CH.index("C3")], flat[CH.index("C3")])

    def test_quality_flags_flat_epoch(self):
        epoch = make_data()[:, :SAMPLES_PER_EPOCH]
        assert check_epoch_quality(epoch * 1e-6) == ""
        flat_epoch = epoch.copy()
        flat_epoch[3] = 0.0
        assert check_epoch_quality(flat_epoch * 1e-6) == "flat"

    def test_corruption_before_preprocessing_changes_trials(self):
        data = make_data()
        run = make_run(data)
        clean = preprocess_run(run)
        spec = CorruptionSpec(FLAT, ("C3",))
        corrupt_uv = apply_corruption(run.data_v * 1e6, CH, spec, SFREQ_HZ, units="µV")
        corrupt = preprocess_run(run, corrupted=corrupt_uv * 1e-6)
        # Same trial ids/eligibility, different epochs.
        assert np.array_equal(clean["trial_id"].to_numpy(), corrupt["trial_id"].to_numpy())
        assert np.array_equal(clean["eligible"].to_numpy(), corrupt["eligible"].to_numpy())
        e_c = clean.loc[clean["eligible"], "epoch"].iloc[0]
        e_f = corrupt.loc[corrupt["eligible"], "epoch"].iloc[0]
        assert not np.allclose(e_c, e_f)
