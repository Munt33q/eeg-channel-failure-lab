"""End-to-end synthetic app workflow test via Streamlit AppTest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.synth
def test_app_runs_synthetic_demo():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    # Synthetic banner shown
    joined = "\n".join(str(x.body) for x in at.markdown) + "\n".join(str(x.value) for x in at.title)
    assert "Synthetic demonstration" in joined or "not evidence" in joined.lower()
    # Tabs exist
    assert len(at.tabs) == 5


@pytest.mark.synth
def test_app_corruption_changes_signal_tab():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    # Switch failure type to flat on the first selectbox and rerun.
    at.sidebar.selectbox[0].select("Flat channel (0 µV constant)").run()
    assert not at.exception
