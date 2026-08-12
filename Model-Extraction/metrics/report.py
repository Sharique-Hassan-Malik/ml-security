"""
HTML report generator for model extraction experiments.

Renders fidelity vs. query budget curves as inline SVG line charts
inside a self-contained HTML file, along with a summary table.
"""

from __future__ import annotations

import json
from typing import Dict, List


def _svg_linechart(
    series:  Dict[str, List[tuple]],
    width:   int   = 500,
    height:  int   = 200,
    x_label: str   = "Oracle queries",
    y_label: str   = "Agreement",
    y_min:   float = 0.0,
    y_max:   float = 1.0,
) -> str:
    COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    pl, pr, pt, pb = 50, 20, 18, 38

    w = width  - pl - pr
    h = height - pt - pb

    all_x = [x for pts in series.values() for x, _ in pts]
    x_min = min(all_x) if all_x else 0
    x_max = max(all_x) if all_x else 1

    def cx(x: float) -> float:
        return pl + (x - x_min) / max(x_max - x_min, 1) * w

    def cy(y: float) -> float:
        return pt + (1 - (y - y_min) / max(y_max - y_min, 1e-9)) * h

    grid = "".join(
        f'<line x1="{pl}" y1="{cy(t):.1f}" x2="{pl+w}" y2="{cy(t):.1f}" '
        f'stroke="#2a2a2a" stroke-width="1"/>'
        f'<text x="{pl-5}" y="{cy(t)+4:.1f}" fill="#555" font-size="9" text-anchor="end">'
        f'{t:.2f}</text>'
        for t in [0.0, 0.25, 0.50, 0.75, 1.0]
    )

    lines  = ""
    legend = ""
    for i, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        color  = COLORS[i % len(COLORS)]
        coords = " ".join(f"{cx(x):.1f},{cy(y):.1f}" for x, y in pts)
        lines += (
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
        lx = pl + i * 130
        legend += (
            f'<line x1="{lx}" y1="{height-10}" x2="{lx+16}" y2="{height-10}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<text x="{lx+20}" y="{height-6}" fill="#aaa" font-size="10">{label}</text>'
        )

    axis_x = (
        f'<line x1="{pl}" y1="{pt+h}" x2="{pl+w}" y2="{pt+h}" stroke="#444"/>'
        f'<text x="{pl+w//2}" y="{height-1}" fill="#555" font-size="10" '
        f'text-anchor="middle">{x_label}</text>'
    )
    axis_y = (
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+h}" stroke="#444"/>'
    )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#111" rx="6"/>'
        f'{grid}{axis_x}{axis_y}{lines}{legend}'
        f'</svg>'
    )


def generate_html_report(
    strategy_results: Dict[str, dict],
    output_path:      str = "report.html",
) -> None:
    """
    Write a self-contained HTML report comparing extraction strategies.

    Parameters
    ----------
    strategy_results : {strategy_name: ExtractionAttack.to_dict(result)}
    output_path      : file path to write
    """
    series: Dict[str, List[tuple]] = {}
    for name, data in strategy_results.items():
        series[name] = [(r["queries_used"], r["agreement"]) for r in data["rounds"]]

    chart = _svg_linechart(series)

    rows = ""
    for name, data in strategy_results.items():
        rounds  = data["rounds"]
        final   = data["final_agreement"]
        total_q = data["total_queries"]
        q90     = next(
            (r["queries_used"] for r in rounds if r["agreement"] >= 0.90), "—"
        )
        rows += (
            f'<tr><td>{name}</td><td>{total_q:,}</td>'
            f'<td>{final:.4f}</td><td>{q90}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Extraction — Fidelity Report</title>
<style>
  body  {{ font-family: "Segoe UI", Arial, sans-serif; background:#0f0f0f;
           color:#e0e0e0; margin:0; padding:24px; }}
  h1    {{ font-size:1.3rem; color:#fff; margin-bottom:4px; }}
  .sub  {{ color:#888; font-size:.85rem; margin-bottom:24px; }}
  .row  {{ display:flex; gap:28px; flex-wrap:wrap; align-items:flex-start; }}
  table {{ border-collapse:collapse; background:#1a1a1a; border-radius:8px;
           overflow:hidden; min-width:400px; }}
  th    {{ padding:10px 14px; text-align:left; font-size:.78rem; color:#888;
           text-transform:uppercase; background:#222; letter-spacing:.05em; }}
  td    {{ padding:9px 14px; font-size:.85rem; border-top:1px solid #2a2a2a; }}
  tr:hover td {{ background:#202020; }}
</style>
</head>
<body>
<h1>Model Extraction Attack — Fidelity vs. Query Budget</h1>
<p class="sub">
  Agreement = fraction of test inputs where the substitute and oracle
  predict the same class.
</p>
<div class="row">
  <div>
    <p style="color:#666;font-size:.78rem;margin-bottom:6px">
      Fidelity vs. oracle queries
    </p>
    {chart}
  </div>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Total queries</th>
        <th>Final agreement</th>
        <th>Queries to 90%</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
<details style="margin-top:28px">
  <summary style="color:#666;cursor:pointer;font-size:.82rem">Raw JSON</summary>
  <pre style="background:#111;padding:14px;border-radius:6px;font-size:.72rem;
              color:#aaa;margin-top:8px;overflow:auto;max-height:400px">
{json.dumps(strategy_results, indent=2)}
  </pre>
</details>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
