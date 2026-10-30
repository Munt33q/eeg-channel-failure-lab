"""Real-data integration tests. Run with: pytest -m integration

Skipped automatically when the PhysioNet recordings are not downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from eeg_channel_failure_lab import data as data_mod  # noqa: E402
from eeg_channel_failure_lab.preprocessing import (  # noqa: E402
    CHANNEL_ORDER,
    SAMPLES_PER_EPOCH,
    SFREQ_HZ,
    preprocess_run,
)


@pytest.fixture(scope="module")
def subj1():
    try:
        return data_mod.load_subject(1, data_dir="data/raw")
    except Exception as exc:  # download failure -> skip, not fail
        pytest.skip(f"real EEGMMIDB data unavailable: {exc}")


def test_channel_contract(subj1):
    run = subj1[12]
    assert tuple(run.channel_names) == CHANNEL_ORDER
    assert run.sfreq_hz == pytest.approx(SFREQ_HZ)


def test_event_mapping_and_epochs(subj1):
    run = subj1[12]
    # PhysioNet docs: imagery runs 4/8/12 contain T1 (left) and T2 (right)
    # events in the annotation channel; T0 rest is excluded.
    assert set(np.unique(run.labels)) <= {0, 1}
    assert len(run.descriptions) == len(run.labels)
    trials = preprocess_run(run)
    ok = trials[trials["eligible"]]
    assert (ok["epoch"].apply(lambda e: e.shape == (9, SAMPLES_PER_EPOCH))).all()
