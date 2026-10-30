"""Capture app screenshots for the README (development helper, offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit.testing.v1 as stt  # noqa: E402

from eeg_channel_failure_lab.corruptions import CorruptionSpec  # noqa: E402


def main() -> None:
    """Render app views headlessly and save PNG screenshots to assets/."""
    at = stt.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception

    # Screenshot 1: the signal comparison figure (flat C3, trial 0).
    import app as appmod

    runs = at.session_state["runs"]
    spec = CorruptionSpec("flat", ("C3",))
    fig = appmod.plot_waveforms(runs, spec, 0)
    out = Path(__file__).resolve().parents[1] / "assets" / "signal_comparison.png"
    fig.savefig(out, dpi=120, facecolor="#0d1b2a", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
