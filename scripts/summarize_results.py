"""Summarize sweep results into README tables (development helper)."""
import json

import numpy as np
import pandas as pd

df = pd.read_csv("results/metrics_flat_noise_gain_sweep.csv")
df["spec"] = df["spec"].apply(json.loads)
df["channel"] = df["spec"].apply(lambda s: s["channels"][0])
df["ftype"] = df["spec"].apply(lambda s: s["failure_type"])
df["sev"] = df["spec"].apply(lambda s: s.get("severity"))

clean = df[df["ftype"] == "flat"].groupby("subject_id")["clean_accuracy"].first()
bal = df[df["ftype"] == "flat"].groupby("subject_id")["clean_balanced_accuracy"].first()
maj = df[df["ftype"] == "flat"].groupby("subject_id")["n_train_majority_baseline"].first()
ntr = df[df["ftype"] == "flat"].groupby("subject_id")["n_trials"].first()

print("=== Clean baseline (held-out run 12) ===")
for s in clean.index:
    print(f"| {s} | {clean[s]:.3f} | {bal[s]:.3f} | {maj[s]:.3f} | {int(ntr[s])} |")
print()

for ftype, sev, title in [
    ("flat", None, "Flat"),
    ("noise", 50.0, "Noise 50 uV"),
    ("gain", 0.0, "Gain 0 (dead)"),
]:
    sub = df[df["ftype"] == ftype]
    if sev is not None:
        sub = sub[sub["sev"] == sev]
    means = sub.groupby("channel")["accuracy_drop_pp"].mean().reindex(
        ["C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"]
    )
    print(f"=== {title}: mean drop (pp), unweighted subject mean ===")
    print(" | ".join(f"{v:+.1f}" for v in means.values))
    print()

print("prediction failures total:", int(df["prediction_failures"].sum()))
print("coverage min:", df["coverage"].min())

# Noise severity curve per channel (subject mean), for the heatmap check
for sev in (5.0, 10.0, 20.0, 50.0):
    sub = df[(df["ftype"] == "noise") & (df["sev"] == sev)]
    m = sub.groupby("channel")["accuracy_drop_pp"].mean()
    print(f"noise {sev:>4.0f} uV: " + " ".join(f"{m[c]:+5.1f}" for c in ["C3", "Cz", "C4"]))
