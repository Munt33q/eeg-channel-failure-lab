"""EEG Channel Failure Lab - interactive Streamlit app.

Modes:
- Synthetic (default, offline, no credentials): deterministic EEG-like data
  with a prominent disclaimer banner.
- Real data: PhysioNet EEGMMIDB runs 4/8/12 + models trained by
  scripts/train.py. Never allows arbitrary uploaded model files.

The same core package functions power the CLI and this app.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_channel_failure_lab import models as models_mod  # noqa: E402
from eeg_channel_failure_lab import synthetic as synth  # noqa: E402
from eeg_channel_failure_lab.corruptions import (  # noqa: E402
    DEFAULT_SEEDS,
    FLAT,
    GAIN,
    NOISE,
    NOISE_GRID_UV,
    GAIN_GRID,
    SWAP,
    CorruptionSpec,
)
from eeg_channel_failure_lab.evaluation import majority_baseline  # noqa: E402
from eeg_channel_failure_lab.experiments import run_single_experiment, sensitivity_sweep  # noqa: E402
from eeg_channel_failure_lab.preprocessing import (  # noqa: E402
    CHANNEL_ORDER,
    RunData,
    epochs_from_trials,
    preprocess_run,
)
from eeg_channel_failure_lab.reporting import esc  # noqa: E402

st.set_page_config(
    page_title="EEG Channel Failure Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Theme: dark navy / teal with mint and amber accents ----
st.markdown(
    """
    <style>
      :root {
        --navy: #0d1b2a; --navy2: #13293d; --teal: #5bc0be; --mint: #7fffd4;
        --amber: #f4a259; --text: #e0e6ed;
      }
      .stApp { background: var(--navy); color: var(--text); }
      h1, h2, h3 { color: var(--mint); }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        background: var(--navy2); border-radius: 8px 8px 0 0; padding: 8px 18px;
      }
      .stTabs [aria-selected="true"] { background: var(--teal); color: #062a30; }
      div[data-testid="stSidebar"] { background: var(--navy2); }
      .banner {
        background: #2b1d0e; border: 1px solid var(--amber); color: var(--amber);
        padding: 0.7rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 600;
      }
      .metric-card { background: var(--navy2); border: 1px solid #1b3a4b;
        border-radius: 8px; padding: 0.6rem 1rem; }
      .good { color: var(--mint); } .bad { color: var(--amber); }
    </style>
    """,
    unsafe_allow_html=True,
)

SYNTH_BANNER = synth.SYNTH_BANNER

FAILURE_LABELS = {
    FLAT: "Flat channel (0 µV constant)",
    NOISE: "Additive Gaussian noise",
    GAIN: "Channel gain error",
    SWAP: "Channel swap (assignment error)",
}


# ---------------------------------------------------------------------------
# Cached data/model loading


@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(subject: int, models_dir: str):
    model = models_mod.load_model(models_dir, subject)
    model.validate_contract()
    return model


@st.cache_data(show_spinner="Loading EEG data…")
def load_real_runs(subject: int, data_dir: str) -> dict[int, RunData]:
    from eeg_channel_failure_lab import data as data_mod

    return data_mod.load_subject(subject, data_dir=data_dir)


@st.cache_data(show_spinner="Generating synthetic data…")
def load_synth_runs(seed: int) -> dict[int, RunData]:
    return synth.make_synthetic_subject(seed)


@st.cache_data(show_spinner="Running experiment…")
def cached_experiment(
    runs_key: int,
    subject: int,
    spec_key: str,
    spec_d: dict,
    train_labels_key: int,
) -> tuple:
    spec = CorruptionSpec.from_dict(spec_d) if spec_d else None
    runs = st.session_state["runs"]
    model = st.session_state["model"]
    tf = run_single_experiment(runs[12], model, spec, st.session_state["train_labels"])
    return tf.trials, tf.metrics


@st.cache_data(show_spinner="Training synthetic demo model…")
def fit_synth_model_cached(runs_tuple: tuple, seed: int):
    """Fit the demo CSP+LDA on synthetic dev runs (keys 4, 8) only."""
    runs = {k: v for k, v in runs_tuple}
    X_parts, y_parts = [], []
    for r in (4, 8):
        trials = preprocess_run(runs[r])
        X, y, _ = epochs_from_trials(trials)
        X_parts.append(X)
        y_parts.append(y)
    return models_mod.fit_pipeline(np.concatenate(X_parts), np.concatenate(y_parts))


def get_session(subject: int, mode: str, seed: int) -> None:
    """Populate st.session_state with runs/model/train labels for the mode."""
    if mode == "Synthetic":
        runs = load_synth_runs(seed)
        model = fit_synth_model_cached(tuple(runs.items()), seed)
    else:
        runs = load_real_runs(subject, "data/raw")
        model = load_model_cached(subject, "models")
    st.session_state["runs"] = runs
    st.session_state["model"] = model
    st.session_state["train_labels"] = np.concatenate(
        [runs[r].labels for r in (4, 8)]
    )


# ---------------------------------------------------------------------------
# Sidebar controls


def sidebar_controls() -> dict:
    with st.sidebar:
        st.title("⚙️ Experiment")
        mode = st.radio("Data mode", ["Synthetic", "Real data"], help="Synthetic works fully offline.")
        if mode == "Synthetic":
            st.caption(SYNTH_BANNER)
            seed = st.number_input("Synthetic data seed", 0, 10_000, 100)
            subject = 0
            get_session(0, mode, int(seed))
        else:
            subject = st.selectbox("Subject", [1, 2, 3, 4, 5])
            get_session(int(subject), mode, 100)
            st.caption("PhysioNet EEGMMIDB; models must be trained with scripts/train.py.")

        failure = st.selectbox(
            "Failure type", list(FAILURE_LABELS.keys()), format_func=lambda k: FAILURE_LABELS[k]
        )
        if failure == SWAP:
            c1 = st.selectbox("Swap channel A", CHANNEL_ORDER, index=CHANNEL_ORDER.index("C3"))
            c2 = st.selectbox("Swap channel B", CHANNEL_ORDER, index=CHANNEL_ORDER.index("C4"))
            channels = (c1, c2)
            severity = None
        else:
            ch = st.selectbox("Channel", CHANNEL_ORDER, index=CHANNEL_ORDER.index("C3"))
            channels = (ch,)
            if failure == NOISE:
                severity = st.select_slider("Noise SD (µV)", options=list(NOISE_GRID_UV), value=20.0)
            elif failure == GAIN:
                severity = st.select_slider("Gain factor", options=list(GAIN_GRID), value=0.5)
            else:
                severity = None
        cs = st.select_slider("Noise seed (noise failures only)", options=list(DEFAULT_SEEDS), value=42)
        trial_sel = st.selectbox("Trial selection", ["All eligible trials", "Single trial"])
        trial_id = None
        if trial_sel == "Single trial":
            n_trials = int(st.session_state["runs"][12].labels.shape[0])
            trial_id = st.number_input("Trial ID", 0, max(0, n_trials - 1), 0)

        if st.button("Run paired experiment", type="primary", use_container_width=True):
            st.session_state["run_requested"] = True

    spec = CorruptionSpec(
        failure_type=failure,
        channels=channels,
        severity=severity,
        seed=int(cs),
    )
    return {
        "mode": mode,
        "subject": subject,
        "spec": spec,
        "trial_id": trial_id,
        "trial_sel": trial_sel,
    }


# ---------------------------------------------------------------------------
# Tabs


def tab_overview(controls: dict) -> None:
    st.header("Overview")
    if controls["mode"] == "Synthetic":
        st.markdown(f'<div class="banner">{SYNTH_BANNER}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This lab measures **model sensitivity to simulated channel failure**:
        a trained CSP+LDA motor-imagery classifier is evaluated on identical
        held-out trials, once on clean data and once on data corrupted by a
        controlled channel failure.

        **Corruption placement.** Failures are applied to *raw* selected-channel
        data **before** common-average referencing, causal 8-30 Hz filtering,
        and epoching. Clean and corrupted copies flow through the identical
        preprocessing pipeline independently. Common-average referencing can
        spread one channel's corruption into the other channels.

        **Fixed contract.** Channels (exact order):
        """
        + ", ".join(f"`{c}`" for c in CHANNEL_ORDER)
        + """ · Sampling rate 160 Hz · Epochs [1.0, 3.0) s after cue ·
        Causal 4th-order Butterworth 8-30 Hz bandpass · CSP(4) + LDA fitted on
        training runs 4+8 only · held-out run 12.

        **Honest evaluation.** Trial eligibility is frozen from clean-data
        checks; corrupted trials are never dropped silently. Prediction
        failures are counted and treated as incorrect. An accuracy change does
        **not** mean the channel is physiologically important.
        """
    )
    m = st.session_state.get("last_metrics")
    if m:
        st.subheader("Current experiment")
        cols = st.columns(4)
        cols[0].metric("Clean accuracy", f"{m['clean_accuracy']:.1%}")
        cols[1].metric("Corrupted accuracy (all eligible)", f"{m['corrupt_accuracy_all_eligible']:.1%}")
        cols[2].metric(
            "Drop",
            f"{m['accuracy_drop_pp']:+.1f} pp",
            delta_color="inverse",
        )
        cols[3].metric("Coverage", f"{m['coverage']:.0%}")


def plot_waveforms(runs: dict, spec: CorruptionSpec, trial_id: int) -> "object":
    import matplotlib.pyplot as plt

    from eeg_channel_failure_lab.corruptions import apply_corruption

    run = runs[12]
    onset = int(run.event_samples[trial_id])
    fs = run.sfreq_hz
    s0, s1 = onset, onset + int(3.0 * fs)
    t = (np.arange(s0, s1) - onset) / fs
    data_uv = run.data_v[:, s0:s1] * 1e6
    corrupt_uv = apply_corruption(data_uv, run.channel_names, spec, fs, units="µV")

    affected = [run.channel_names.index(c) for c in spec.channels]
    n_ch = len(CHANNEL_ORDER)
    fig, axes = plt.subplots(
        n_ch, 1, figsize=(10, 1.15 * n_ch), sharex=True, facecolor="#0d1b2a"
    )
    vmax = max(np.abs(data_uv).max(), np.abs(corrupt_uv).max())
    for i, ax in enumerate(axes):
        ax.set_facecolor("#13293d")
        kw = dict(linewidth=0.9)
        if i in affected:
            ax.plot(t, data_uv[i], color="#7fffd4", label="clean", **kw)
            ax.plot(t, corrupt_uv[i], color="#f4a259", label="corrupted", **kw)
            ax.set_ylabel(run.channel_names[i], color="#f4a259", fontweight="bold")
        else:
            ax.plot(t, data_uv[i], color="#5bc0be", alpha=0.9, label="clean", **kw)
            ax.plot(t, corrupt_uv[i], color="#f4a259", alpha=0.55, label="corrupted", **kw)
            ax.set_ylabel(run.channel_names[i], color="#e0e6ed")
        ax.set_ylim(-vmax, vmax)
        ax.tick_params(colors="#e0e6ed", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1b3a4b")
    axes[0].legend(loc="upper right", fontsize=7, facecolor="#13293d", labelcolor="#e0e6ed")
    axes[-1].set_xlabel("Time since cue (s)", color="#e0e6ed")
    fig.suptitle(
        f"Trial {trial_id}: clean vs corrupted raw waveforms (µV, same scale)",
        color="#7fffd4", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_processed_comparison(runs: dict, spec: CorruptionSpec, trial_id: int) -> "object":
    """Processed (referenced + filtered) epochs: shows referencing spread."""
    import matplotlib.pyplot as plt

    from eeg_channel_failure_lab.corruptions import apply_corruption

    run = runs[12]
    clean_trials = preprocess_run(run)
    data_uv = run.data_v * 1e6
    corrupt_uv = apply_corruption(data_uv, run.channel_names, spec, run.sfreq_hz, units="µV")
    corrupt_trials = preprocess_run(run, corrupted=corrupt_uv * 1e-6)

    e_clean = clean_trials.loc[clean_trials["eligible"], "epoch"].iloc[trial_id] * 1e6
    e_corr = corrupt_trials.loc[corrupt_trials["eligible"], "epoch"].iloc[trial_id] * 1e6
    fs = run.sfreq_hz
    t = np.arange(e_clean.shape[1]) / fs
    n_ch = len(CHANNEL_ORDER)
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.15 * n_ch), sharex=True, facecolor="#0d1b2a")
    vmax = max(np.abs(e_clean).max(), np.abs(e_corr).max())
    affected = [run.channel_names.index(c) for c in spec.channels]
    for i, ax in enumerate(axes):
        ax.set_facecolor("#13293d")
        color_clean = "#f4a259" if i in affected else "#5bc0be"
        ax.plot(t, e_clean[i], color=color_clean, linewidth=0.9, label="clean (processed)")
        ax.plot(t, e_corr[i], color="#7fffd4", linewidth=0.9, alpha=0.8, label="corrupted (processed)")
        note = " ← affected" if i in affected else ""
        spread = ""
        if i not in affected and not np.allclose(e_clean[i], e_corr[i], atol=0.05):
            spread = " (corruption spread via common average)"
        ax.set_ylabel(run.channel_names[i] + note, color="#e0e6ed", fontsize=8)
        ax.set_ylim(-vmax, vmax)
        ax.tick_params(colors="#e0e6ed", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#1b3a4b")
    axes[-1].set_xlabel("Time in epoch (s)", color="#e0e6ed")
    fig.suptitle(
        "Processed epochs (common-average referenced, causally filtered, µV)",
        color="#7fffd4", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def tab_signal_comparison(controls: dict) -> None:
    st.header("Signal comparison")
    spec: CorruptionSpec = controls["spec"]
    trial_id = controls["trial_id"] or 0
    st.caption(spec.describe())
    fig = plot_waveforms(st.session_state["runs"], spec, trial_id)
    st.pyplot(fig, use_container_width=True)
    st.info("Waveform display is the raw selected-channel signal; no downsampling is applied for the full epoch view.")
    st.subheader("After preprocessing (shows common-average spread)")
    fig2 = plot_processed_comparison(st.session_state["runs"], spec, trial_id)
    st.pyplot(fig2, use_container_width=True)


def tab_model_impact(controls: dict) -> None:
    st.header("Model impact")
    trials, metrics = cached_experiment(
        0, controls["subject"], controls["spec"].failure_type, controls["spec"].to_dict(), 0
    )
    st.session_state["last_metrics"] = metrics
    st.session_state["last_trials"] = trials

    cols = st.columns(4)
    cols[0].metric("Clean accuracy", f"{metrics['clean_accuracy']:.1%}")
    cols[1].metric("Corrupted (all eligible)", f"{metrics['corrupt_accuracy_all_eligible']:.1%}")
    cols[2].metric("Drop", f"{metrics['accuracy_drop_pp']:+.1f} pp")
    cols[3].metric("Flip rate", f"{metrics['flip_rate']:.1%}")
    cols2 = st.columns(4)
    cols2[0].metric("Balanced acc (clean)", f"{metrics['clean_balanced_accuracy']:.1%}")
    cols2[1].metric("Balanced acc (corrupt)", f"{metrics['corrupt_balanced_accuracy_all_eligible']:.1%}")
    cols2[2].metric("Correct→incorrect", f"{metrics['correct_to_incorrect']}")
    cols2[3].metric("Incorrect→correct", f"{metrics['incorrect_to_correct']}")

    st.caption(
        f"Prediction failures: {metrics['prediction_failures']} · coverage {metrics['coverage']:.0%} · "
        f"accuracy among predicted trials: {metrics['corrupt_accuracy_among_predicted']:.1%} (labeled secondary) · "
        f"majority baseline from training labels: {metrics['n_train_majority_baseline']:.1%}"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor="#0d1b2a")
    for ax, key, title in (
        (axes[0], "clean_confusion", "Clean predictions"),
        (axes[1], "corrupt_confusion", "Corrupted predictions"),
    ):
        cm = metrics[key]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks([0, 1], ["pred left", "pred right"], color="#e0e6ed")
        ax.set_yticks([0, 1], ["true left", "true right"], color="#e0e6ed")
        ax.set_title(title, color="#7fffd4", fontsize=10)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center", color="white", fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Per-trial predictions")
    show = trials[
        ["trial_id", "label", "clean_pred", "corrupt_pred", "prediction_ok", "clean_p_right", "corrupt_p_right"]
    ].rename(columns={"clean_p_right": "p(right) clean*", "corrupt_p_right": "p(right) corrupt*"})
    st.dataframe(show, use_container_width=True, height=280)
    st.caption("*LDA decision values are not calibrated confidence.")


def compute_sensitivity(runs: dict, model, subject: int) -> pd.DataFrame:
    specs = []
    for ch in CHANNEL_ORDER:
        specs.append(CorruptionSpec(FLAT, (ch,), seed=42))
        for sev in NOISE_GRID_UV[1:]:
            specs.append(CorruptionSpec(NOISE, (ch,), severity=sev, seed=42))
        for g in GAIN_GRID:
            if g != 1.0:
                specs.append(CorruptionSpec(GAIN, (ch,), severity=g, seed=42))
    df = sensitivity_sweep(runs[12], model, specs, st.session_state["train_labels"])
    return df


@st.cache_data(show_spinner="Running sensitivity sweep (channel × severity)…")
def cached_sensitivity(subject: int, mode: str, seed: int, runs_key: int) -> pd.DataFrame:
    return compute_sensitivity(st.session_state["runs"], st.session_state["model"], subject)


def tab_sensitivity(controls: dict) -> None:
    st.header("Channel sensitivity")
    st.caption(
        "Each cell: accuracy drop (percentage points) for one channel × severity. "
        "This measures model sensitivity to simulated channel failure — not physiological importance."
    )
    df = cached_sensitivity(controls["subject"], controls["mode"], 100, 0)
    df["channel"] = [s["channels"][0] for s in df["spec"]]
    for_noise = df[df["spec"].apply(lambda s: s["failure_type"] == NOISE)].copy()
    for_noise["severity"] = for_noise["spec"].apply(lambda s: s["severity"])
    pivot = for_noise.pivot_table(index="channel", columns="severity", values="accuracy_drop_pp")
    pivot = pivot.reindex(list(CHANNEL_ORDER))
    st.subheader("Additive noise: accuracy drop (pp) by channel × severity (µV)")
    st.dataframe(pivot.style.format("{:.1f}").background_gradient(cmap="Oranges", axis=None), use_container_width=True)

    fig_heat(pivot)

    flat = df[df["spec"].apply(lambda s: s["failure_type"] == FLAT)].copy()
    flat["channel"] = flat["spec"].apply(lambda s: s["channels"][0])
    st.subheader("Flat channel: drop (pp) per channel")
    flat_plot = flat.set_index("channel")["accuracy_drop_pp"].reindex(list(CHANNEL_ORDER))
    st.bar_chart(flat_plot, height=260, color="#f4a259")


def fig_heat(pivot: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1b2a")
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(c) for c in pivot.columns], color="#e0e6ed")
    ax.set_yticks(range(data.shape[0]), list(pivot.index), color="#e0e6ed")
    ax.set_xlabel("Noise SD (µV)", color="#e0e6ed")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if v == v:
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Accuracy drop (pp): channel × noise severity", color="#7fffd4")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def tab_export(controls: dict) -> None:
    st.header("Export")
    trials = st.session_state.get("last_trials")
    metrics = st.session_state.get("last_metrics")
    if trials is None or metrics is None:
        st.info("Run a paired experiment first (Model impact tab or sidebar button).")
        return
    spec: CorruptionSpec = controls["spec"]

    from eeg_channel_failure_lab.reporting import standalone_html, write_report

    preds_csv = trials.to_csv(index=False)
    st.download_button("Per-trial predictions CSV", preds_csv, "predictions.csv", "text/csv")

    metrics_row = {k: v for k, v in metrics.items() if k != "spec"}
    import json as _json

    metrics_csv_text = pd.DataFrame(
        [{k: (_json.dumps(v) if isinstance(v, dict) else v) for k, v in metrics_row.items()}]
    ).to_csv(index=False)
    st.download_button("Aggregate metrics CSV", metrics_csv_text, "metrics.csv", "text/csv")

    cfg = {
        "mode": controls["mode"],
        "subject": controls["subject"],
        "spec": spec.to_dict(),
        "channel_order": list(CHANNEL_ORDER),
        "preprocessing": {
            "order": ["corruption on raw data", "common-average reference", "causal 8-30 Hz bandpass", "epoch [1.0,3.0) s"],
            "units": "volts internal; µV for severities and display",
        },
        "package_versions": models_mod.package_versions(),
    }
    cfg_json = _json.dumps(cfg, indent=2)
    st.download_button("Experiment configuration JSON", cfg_json, "config.json", "application/json")

    banner = SYNTH_BANNER if controls["mode"] == "Synthetic" else ""
    html_text = standalone_html(
        title="EEG Channel Failure Lab — report",
        sections=[
            ("Experiment", f"<p>{esc(spec.describe())}</p>"),
            ("Metrics", "<pre>" + esc(_json.dumps(metrics_row, indent=2)) + "</pre>"),
            ("Configuration", "<pre>" + esc(cfg_json) + "</pre>"),
        ],
        metadata={
            "mode": controls["mode"],
            "subject": controls["subject"],
            "montage": list(CHANNEL_ORDER),
            "units": "µV for display/severity; volts internal",
            "seed": spec.seed,
            "evaluation_split": "held-out run 12; training runs 4+8",
            "model": "CSP(4, ledoit_wolf, log) + LDA(lsqr, shrinkage auto)",
        },
    )
    if controls["mode"] != "Synthetic":
        html_text = html_text.replace(
            '<div class="banner">Synthetic demonstration — not evidence of performance on human EEG (applies to synthetic-mode reports).</div>',
            "",
        )
    st.download_button("Standalone HTML report", html_text, "report.html", "text/html")


def main() -> None:
    st.title("🧠 EEG Channel Failure Lab")
    controls = sidebar_controls()

    tabs = st.tabs(["Overview", "Signal comparison", "Model impact", "Channel sensitivity", "Export"])
    with tabs[0]:
        tab_overview(controls)
    with tabs[1]:
        tab_signal_comparison(controls)
    with tabs[2]:
        tab_model_impact(controls)
    with tabs[3]:
        tab_sensitivity(controls)
    with tabs[4]:
        tab_export(controls)


if __name__ == "__main__":
    main()
