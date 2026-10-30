"""Run paired held-out corruption experiments on run 12 and save results.

Examples:
  python scripts/evaluate_failures.py --subject 1 --failure flat --channel C3
  python scripts/evaluate_failures.py --subject 1 --failure noise --channel Cz --severity 20 --seeds 7 19 42
  python scripts/evaluate_failures.py --all-subjects --sweep      # full channel x severity grid
  python scripts/evaluate_failures.py --subject 1 --failure swap --channels C3 C4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_channel_failure_lab import data as data_mod
from eeg_channel_failure_lab import models as models_mod
from eeg_channel_failure_lab.corruptions import DEFAULT_SEEDS, FLAT, GAIN, NOISE, SWAP, CorruptionSpec
from eeg_channel_failure_lab.experiments import run_single_experiment, sensitivity_sweep
from eeg_channel_failure_lab.preprocessing import CHANNEL_ORDER, epochs_from_trials, preprocess_run
from eeg_channel_failure_lab.reporting import config_json, metrics_csv, predictions_csv, write_report


def build_spec(args, severity: float | None, seed: int) -> CorruptionSpec:
    if args.failure == SWAP:
        chans = tuple(args.channels) if args.channels else ("C3", "C4")
        return CorruptionSpec(SWAP, chans, seed=seed)
    ch = args.channel or "C3"
    if ch not in CHANNEL_ORDER:
        raise SystemExit(f"unknown channel {ch!r}; choose from {list(CHANNEL_ORDER)}")
    if args.failure == FLAT:
        return CorruptionSpec(FLAT, (ch,), seed=seed)
    if args.failure == NOISE:
        sev = args.severity if severity is None else severity
        if sev is None:
            raise SystemExit("noise failure requires --severity (µV SD)")
        return CorruptionSpec(NOISE, (ch,), severity=float(sev), seed=seed)
    if args.failure == GAIN:
        sev = args.severity if severity is None else severity
        if sev is None:
            raise SystemExit("gain failure requires --severity (gain factor)")
        return CorruptionSpec(GAIN, (ch,), severity=float(sev), seed=seed)
    raise SystemExit(f"unknown failure {args.failure!r}")


def get_train_labels(subj: int, data_dir: str) -> np.ndarray:
    runs = data_mod.load_subject(subj, data_dir=data_dir)
    y = []
    for r in data_mod.DEV_RUNS:
        y.append(runs[r].labels)
    return np.concatenate(y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int)
    ap.add_argument("--all-subjects", action="store_true")
    ap.add_argument("--failure", choices=[FLAT, NOISE, GAIN, SWAP], required=False)
    ap.add_argument("--channel", help="single channel for flat/noise/gain")
    ap.add_argument("--channels", nargs=2, help="two channels for swap")
    ap.add_argument("--severity", type=float, help="noise SD in µV, or gain factor")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--sweep", action="store_true", help="full channel x severity grid")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    subjects = list(range(1, 6)) if args.all_subjects else [args.subject]
    if not subjects or any(s is None for s in subjects):
        raise SystemExit("specify --subject N or --all-subjects")
    if not args.sweep and args.failure is None:
        raise SystemExit("--failure is required unless --sweep is given")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    versions = models_mod.package_versions()
    all_metric_rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    for subj in subjects:
        model = models_mod.load_model(args.models_dir, subj)
        model.validate_contract()
        runs = data_mod.load_subject(subj, data_dir=args.data_dir)
        eval_run = runs[data_mod.EVAL_RUN]
        train_labels = get_train_labels(subj, args.data_dir)

        if args.sweep:
            specs: list[CorruptionSpec] = []
            for ch in CHANNEL_ORDER:
                specs.append(CorruptionSpec(FLAT, (ch,), seed=42))
                for sev in (5.0, 10.0, 20.0, 50.0):
                    specs.append(CorruptionSpec(NOISE, (ch,), severity=sev, seed=42))
                for gain in (0.0, 0.25, 0.5, 2.0, 4.0):
                    specs.append(CorruptionSpec(GAIN, (ch,), severity=gain, seed=42))
            df = sensitivity_sweep(eval_run, model, specs, train_labels)
            if "subject_id" not in df.columns:
                df.insert(0, "subject_id", subj)
            all_metric_rows.extend(df.to_dict("records"))
        else:
            for seed in args.seeds:
                spec = build_spec(args, None, seed)
                tf = run_single_experiment(eval_run, model, spec, train_labels)
                m = dict(tf.metrics)
                m["seed"] = seed
                all_metric_rows.append(m)
                if "subject_id" not in tf.trials.columns:
                    tf.trials.insert(0, "subject_id", subj)
                if "seed" not in tf.trials.columns:
                    tf.trials.insert(1, "seed", seed)
                all_preds.append(tf.trials)

        print(f"subject {subj} done")

    tag = f"{args.failure}" + ("_sweep" if args.sweep else "")
    metrics_path = out / f"metrics_{tag}.csv"
    metrics_path.write_text(metrics_csv(all_metric_rows), encoding="utf-8")
    cfg = {
        "failure": args.failure,
        "channel": args.channel,
        "channels": args.channels,
        "severity": args.severity,
        "seeds": args.seeds,
        "sweep": args.sweep,
        "subjects": subjects,
        "package_versions": versions,
        "channel_order": list(CHANNEL_ORDER),
    }
    (out / f"config_{tag}.json").write_text(config_json(cfg), encoding="utf-8")
    if all_preds:
        pd.concat(all_preds).to_csv(out / f"predictions_{tag}.csv", index=False)
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
