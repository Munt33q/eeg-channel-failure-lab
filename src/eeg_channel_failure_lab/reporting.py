"""Reporting: export CSVs, config JSON, and standalone HTML reports.

- All input-derived text (channel names, descriptions, file names) is escaped
  with html.escape before any HTML output.
- Reports embed plots as base64 PNG and use no external network resources.
- Every report records: units, montage, preprocessing order, failure
  placement, seeds, model identity, data identifiers, evaluation split.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def esc(text: Any) -> str:
    """Escape any input-derived text for HTML inclusion."""
    return html.escape(str(text), quote=True)


def predictions_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def metrics_csv(rows: list[dict[str, Any]]) -> str:
    flat: list[dict[str, Any]] = []
    for r in rows:
        flat.append({k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v) for k, v in r.items()})
    return pd.DataFrame(flat).to_csv(index=False)


def config_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True, default=str)


def figure_to_png_b64(fig: "object") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def standalone_html(
    title: str,
    sections: list[tuple[str, str]],  # (heading, html_body)
    metadata: dict[str, Any],
    figures: list[tuple[str, "object"]] | None = None,
    is_synthetic: bool = False,
) -> str:
    """Build a self-contained HTML report (no external dependencies)."""
    figs_html = ""
    for caption, fig in figures or []:
        b64 = figure_to_png_b64(fig)
        figs_html += (
            f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(caption)}"/>'
            f"<figcaption>{esc(caption)}</figcaption></figure>"
        )

    meta_rows = "".join(
        f"<tr><th>{esc(k)}</th><td><pre>{esc(json.dumps(v, sort_keys=True, default=str)) if isinstance(v, (dict, list, tuple)) else esc(v)}</pre></td></tr>"
        for k, v in sorted(metadata.items())
    )
    body_sections = "".join(f"<section><h2>{esc(h)}</h2>{b}</section>" for h, b in sections)

    now = datetime.now(timezone.utc).isoformat()
    banner = '<div class="banner">Synthetic demonstration — not evidence of performance on human EEG.</div>' if is_synthetic else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{esc(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; background:#0d1b2a; color:#e0e6ed;
         max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.5; }}
  h1 {{ color:#7fffd4; }} h2 {{ color:#5bc0be; border-bottom:1px solid #1b3a4b; padding-bottom:.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; }}
  th, td {{ border: 1px solid #1b3a4b; padding: .35rem .6rem; text-align: left; }}
  th {{ background:#13293d; color:#7fffd4; }}
  pre {{ background:#13293d; padding: .4rem; border-radius: 4px; overflow-x: auto; }}
  figure {{ margin: 1rem 0; }} img {{ max-width: 100%; border-radius: 6px; }}
  figcaption {{ color:#f4a259; font-size: .9rem; }}
  .banner {{ background:#2b1d0e; border:1px solid #f4a259; color:#f4a259; padding:.6rem 1rem;
             border-radius:6px; margin:1rem 0; }}
  section {{ margin: 1.6rem 0; }}
</style>
</head>
<body>
<h1>{esc(title)}</h1>
<div class="banner">Synthetic demonstration — not evidence of performance on human EEG (applies to synthetic-mode reports).</div>
<p>Generated {esc(now)} by eeg-channel-failure-lab.</p>
{banner}
{body_sections}
{figs_html}
<section><h2>Run metadata</h2>
<table>{meta_rows}</table></section>
<section><h2>Interpretation cautions</h2>
<ul>
<li>Flatline zero-fill simulates a failed channel; it is not physical channel removal.</li>
<li>Channel swap simulates a channel-assignment error, not physiology.</li>
<li>Common-average referencing can spread one channel's corruption into other channels.</li>
<li>Changes in predictions reflect <em>model sensitivity to simulated channel failure</em>,
not physiological importance of a channel.</li>
<li>LDA decision values are not calibrated confidence.</li>
<li>Repeated noise seeds measure corruption variability, not additional independent subjects.</li>
</ul></section>
</body></html>"""


def write_report(html_text: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
