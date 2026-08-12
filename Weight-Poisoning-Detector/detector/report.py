"""
Finding and Report types, plus JSON and HTML serialisation.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Finding:
    layer: str
    test: str
    severity: str          # "low" | "medium" | "high"
    score: float           # 0.0 – 1.0
    detail: str
    metadata: Dict = field(default_factory=dict)


class Report:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.findings: List[Finding] = []
        self.layer_count: int = 0
        self.param_count: int = 0
        self.timestamp: str = datetime.datetime.utcnow().isoformat() + "Z"

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @property
    def overall_score(self) -> float:
        if not self.findings:
            return 0.0
        n = max(1, self.layer_count)
        high = [f for f in self.findings if f.severity == "high"]
        med  = [f for f in self.findings if f.severity == "medium"]
        if high:
            return min(1.0, 0.50 + 0.50 * len(high) / n)
        if med:
            return min(0.49, 0.20 + 0.29 * len(med) / n)
        return min(0.19, 0.04 * len(self.findings))

    @property
    def verdict(self) -> str:
        s = self.overall_score
        if s >= 0.60:
            return "HIGH_RISK"
        if s >= 0.25:
            return "SUSPICIOUS"
        return "CLEAN"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_path":    self.model_path,
            "timestamp":     self.timestamp,
            "layer_count":   self.layer_count,
            "param_count":   self.param_count,
            "overall_score": round(self.overall_score, 4),
            "verdict":       self.verdict,
            "findings": [
                {
                    "layer":    f.layer,
                    "test":     f.test,
                    "severity": f.severity,
                    "score":    round(f.score, 4),
                    "detail":   f.detail,
                    "metadata": f.metadata,
                }
                for f in self.findings
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_html(self) -> str:
        d = self.to_dict()
        verdict_color = {"CLEAN": "#2ecc71", "SUSPICIOUS": "#f39c12", "HIGH_RISK": "#e74c3c"}
        vc = verdict_color.get(d["verdict"], "#999")

        rows = ""
        for f in d["findings"]:
            sev_color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}.get(f["severity"], "#999")
            rows += (
                f'<tr>'
                f'<td>{f["layer"]}</td>'
                f'<td>{f["test"]}</td>'
                f'<td style="color:{sev_color};font-weight:600">{f["severity"]}</td>'
                f'<td>{f["score"]:.4f}</td>'
                f'<td>{f["detail"]}</td>'
                f'</tr>\n'
            )

        no_findings = '<tr><td colspan="5" style="text-align:center;color:#aaa">No findings</td></tr>' if not rows else rows

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weight Poisoning Report — {d["model_path"]}</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 24px; }}
  h1   {{ font-size: 1.4rem; color: #fff; margin-bottom: 4px; }}
  .sub {{ color: #888; font-size: .85rem; margin-bottom: 28px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px 24px; min-width: 140px; }}
  .card .label {{ font-size: .75rem; color: #888; text-transform: uppercase; letter-spacing: .06em; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden; }}
  thead tr {{ background: #222; }}
  th {{ padding: 10px 14px; text-align: left; font-size: .8rem; color: #888; text-transform: uppercase; letter-spacing: .06em; }}
  td {{ padding: 10px 14px; font-size: .85rem; border-top: 1px solid #2a2a2a; vertical-align: top; word-break: break-all; }}
  tr:hover td {{ background: #202020; }}
</style>
</head>
<body>
<h1>Neural Network Weight Poisoning Detector</h1>
<p class="sub">Model: {d["model_path"]} &nbsp;|&nbsp; Scanned: {d["timestamp"]}</p>

<div class="cards">
  <div class="card">
    <div class="label">Verdict</div>
    <div class="value" style="color:{vc}">{d["verdict"]}</div>
  </div>
  <div class="card">
    <div class="label">Score</div>
    <div class="value">{d["overall_score"]:.4f}</div>
  </div>
  <div class="card">
    <div class="label">Layers</div>
    <div class="value">{d["layer_count"]}</div>
  </div>
  <div class="card">
    <div class="label">Parameters</div>
    <div class="value">{d["param_count"]:,}</div>
  </div>
  <div class="card">
    <div class="label">Findings</div>
    <div class="value">{len(d["findings"])}</div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>Layer</th><th>Test</th><th>Severity</th><th>Score</th><th>Detail</th>
    </tr>
  </thead>
  <tbody>
    {no_findings}
  </tbody>
</table>
</body>
</html>
"""
