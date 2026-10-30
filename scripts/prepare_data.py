"""Prepare real EEG data: download PhysioNet EEGMMIDB runs 4/8/12 and cache
per-run processed trial tables for subjects 1-5 (default).

Downloads land under data/raw (never committed). Verifies sampling rate,
channel contract, and T1/T2 event mapping at load time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_channel_failure_lab import data as data_mod
from eeg_channel_failure_lab import models as models_mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"subjects": {}}
    for subj in args.subjects:
        runs = data_mod.load_subject(subj, data_dir=args.data_dir)
        subj_info = {}
        for run_id, run in runs.items():
            sig = data_mod.run_signature(run)
            subj_info[str(run_id)] = sig
            print(f"S{subj:03d} run {run_id}: {sig['n_events']} events, labels {sig['label_counts']}")
        manifest["subjects"][str(subj)] = subj_info  # type: ignore[index]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}; package versions: {models_mod.package_versions()}")


if __name__ == "__main__":
    main()
