"""One renderer for six modules.

Before this file there were three HTML report generators in this codebase, each
with its own copy of a dark theme, its own severity colours and its own inline
SVG chart code. They disagreed about what "high" looks like. Now a module emits
`Finding`s and, if it has a curve worth showing, a declarative chart spec in
`ModuleResult.metrics["charts"]` — and the drawing happens exactly once, here.

Colour follows a status palette, and severity is never carried by colour alone:
every severity ships with a distinct glyph and its written name, in the terminal
and in HTML. That is the required mitigation for a status scale, and it is also
what makes the report readable in a CI log that has stripped the ANSI codes.
"""

from __future__ import annotations

import base64
import html
import sys
from typing import Any, Iterable, Sequence

from .finding import Finding, ModuleResult, Report, Severity, Verdict

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Status hexes are fixed by the design system and are not re-stepped per theme.
# INFO and LOW are not status states, so they take muted ink and a sequential
# blue step rather than impersonating a status colour.

_SEVERITY_HEX_LIGHT: dict[Severity, str] = {
    Severity.SAFE: "#0ca30c",      # status: good
    Severity.INFO: "#52514e",      # muted ink — informational, not a state
    Severity.LOW: "#1c5cab",       # sequential blue 550
    Severity.MEDIUM: "#fab219",    # status: warning
    Severity.HIGH: "#ec835a",      # status: serious
    Severity.CRITICAL: "#d03b3b",  # status: critical
}

_SEVERITY_HEX_DARK: dict[Severity, str] = {
    Severity.SAFE: "#0ca30c",
    Severity.INFO: "#c3c2b7",
    Severity.LOW: "#6da7ec",       # sequential blue 300, stepped for dark
    Severity.MEDIUM: "#fab219",
    Severity.HIGH: "#ec835a",
    Severity.CRITICAL: "#d03b3b",
}

# Distinct shapes, so severity survives greyscale, colour-blindness and a
# terminal with no colour support.
_SEVERITY_GLYPH: dict[Severity, str] = {
    Severity.SAFE: "✓",
    Severity.INFO: "i",
    Severity.LOW: "▪",
    Severity.MEDIUM: "▲",
    Severity.HIGH: "◆",
    Severity.CRITICAL: "✖",
}

_SEVERITY_ANSI: dict[Severity, str] = {
    Severity.SAFE: "\033[32m",
    Severity.INFO: "\033[2m",
    Severity.LOW: "\033[34m",
    Severity.MEDIUM: "\033[33m",
    Severity.HIGH: "\033[38;5;209m",
    Severity.CRITICAL: "\033[31;1m",
}

_VERDICT_SEVERITY: dict[Verdict, Severity] = {
    Verdict.CLEAN: Severity.SAFE,
    Verdict.SUSPICIOUS: Severity.MEDIUM,
    Verdict.HIGH_RISK: Severity.HIGH,
    Verdict.CRITICAL: Severity.CRITICAL,
}

# Eight categorical slots, in the fixed order that clears every adjacent-pair
# gate in both modes. Line charts compare neighbours, so the full eight are in
# play here; an all-pairs form (scatter, choropleth) would be capped at three.
# A ninth series is never a generated hue — the caller facets instead.
_SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767")
_MAX_SERIES = len(_SERIES_LIGHT)
# Past four lines, right-hand labels start colliding, so identity moves to a
# legend row instead of sitting on the marks.
_DIRECT_LABEL_LIMIT = 4

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def _colour(text: str, severity: Severity, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_SEVERITY_ANSI[severity]}{text}{_RESET}"


def _label(severity: Severity, enabled: bool) -> str:
    """Glyph plus name — the two carry the meaning, the colour only reinforces."""
    return _colour(f"{_SEVERITY_GLYPH[severity]} {severity.value:<8}", severity, enabled)


def render_terminal(
    report: Report,
    *,
    colour: bool | None = None,
    verbose: bool = False,
    stream: Any = None,
) -> None:
    out = stream or sys.stdout
    if colour is None:
        colour = hasattr(out, "isatty") and out.isatty()

    def emit(line: str = "") -> None:
        print(line, file=out)

    emit()
    if report.target:
        emit(f"{_BOLD if colour else ''}── {report.target}{_RESET if colour else ''}")

    for result in report.results:
        _render_result_terminal(result, emit, colour, verbose)

    _render_summary_terminal(report, emit, colour)


def _render_result_terminal(
    result: ModuleResult,
    emit: Any,
    colour: bool,
    verbose: bool,
) -> None:
    emit()
    head = f"  {result.module}  {_DIM if colour else ''}({result.kind.value}){_RESET if colour else ''}"
    emit(head)

    if result.skipped:
        emit(f"    skipped — {result.skipped}")
        return
    if result.error:
        emit(f"    ERROR — {result.error}")
        return

    for key, value in result.metrics.items():
        if key in ("charts", "traceback"):
            continue
        emit(f"    {key:<24} {_fmt_metric(value)}")

    findings = result.findings if verbose else [
        f for f in result.findings if f.severity >= Severity.LOW
    ]
    if not findings:
        note = "no findings" if not result.findings else "no findings at or above LOW"
        emit(f"    {_label(Severity.SAFE, colour)} {note}")
        return

    for finding in sorted(findings, key=lambda f: (-f.severity.rank, f.location)):
        where = f"{finding.location}  " if finding.location else ""
        emit(f"    {_label(finding.severity, colour)} {where}{finding.title}")
        text = finding.summary or finding.detail
        if text:
            for line in _wrap(text, 84):
                emit(f"{'':>15}{line}")
        if verbose and finding.detail and finding.summary:
            for line in _wrap(finding.detail, 84):
                emit(f"{'':>15}{_DIM if colour else ''}{line}{_RESET if colour else ''}")


def _render_summary_terminal(report: Report, emit: Any, colour: bool) -> None:
    counts = {sev: n for sev, n in report.counts().items() if n}
    emit()
    emit(f"  {'─' * 60}")
    verdict_sev = _VERDICT_SEVERITY[report.verdict]
    emit(f"  verdict   {_label(verdict_sev, colour)} {report.verdict.value}")
    if counts:
        parts = " ".join(
            f"{_SEVERITY_GLYPH[sev]} {sev.value.lower()} {n}"
            for sev, n in sorted(counts.items(), key=lambda kv: -kv[0].rank)
        )
        emit(f"  findings  {len(report.findings)}   {parts}")
    else:
        emit("  findings  0")
    if report.errors:
        emit(f"  errors    {len(report.errors)} module(s) failed to run")
    emit()


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt_metric(v)}" for k, v in list(value.items())[:4])
    return str(value)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > width and line:
            lines.append(" ".join(line))
            line = []
        line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines


# ---------------------------------------------------------------------------
# Charts — declarative specs from modules, drawn once here
# ---------------------------------------------------------------------------

def svg_line_chart(
    series: dict[str, Sequence[tuple[float, float]]],
    *,
    width: int = 560,
    height: int = 240,
    x_label: str = "",
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """A line chart as inline SVG, themed through CSS custom properties.

    Up to four series are direct-labelled at their right end; beyond that a
    legend row carries the names. Either way identity is written down, never
    left to colour alone.
    """
    series = _cap_series(series, _MAX_SERIES)
    direct = len(series) <= _DIRECT_LABEL_LIMIT
    left, right, top, bottom = 52, (96 if direct else 24), 16, (40 if direct else 58)
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs or not ys:
        return ""

    x_lo, x_hi = min(xs), max(xs)
    y_lo = min(ys) if y_min is None else y_min
    y_hi = max(ys) if y_max is None else y_max
    if y_hi - y_lo < 1e-9:
        y_hi = y_lo + 1.0

    def px(x: float) -> float:
        return left + (x - x_lo) / max(x_hi - x_lo, 1e-9) * plot_w

    def py(y: float) -> float:
        return top + (1 - (y - y_lo) / max(y_hi - y_lo, 1e-9)) * plot_h

    ticks = [y_lo + (y_hi - y_lo) * t / 4 for t in range(5)]
    grid = "".join(
        f'<line x1="{left}" y1="{py(t):.1f}" x2="{left + plot_w}" y2="{py(t):.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="{left - 8}" y="{py(t) + 4:.1f}" fill="var(--text-muted)" '
        f'font-size="10" text-anchor="end">{_tick(t)}</text>'
        for t in ticks
    )

    body = ""
    legend = ""
    legend_x = left
    for index, (label, points) in enumerate(series.items()):
        if not points:
            continue
        colour = f"var(--series-{index + 1})"
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        body += (
            f'<polyline points="{coords}" fill="none" stroke="{colour}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        last_x, last_y = points[-1]
        body += (
            f'<circle cx="{px(last_x):.1f}" cy="{py(last_y):.1f}" r="4" fill="{colour}" '
            f'stroke="var(--surface-1)" stroke-width="2"/>'
        )
        if direct:
            body += (
                f'<text x="{px(last_x) + 9:.1f}" y="{py(last_y) + 4:.1f}" '
                f'fill="var(--text-secondary)" font-size="11">{html.escape(label)}</text>'
            )
        else:
            legend += (
                f'<line x1="{legend_x}" y1="{height - 26}" x2="{legend_x + 14}" '
                f'y2="{height - 26}" stroke="{colour}" stroke-width="2"/>'
                f'<circle cx="{legend_x + 7}" cy="{height - 26}" r="3.5" fill="{colour}"/>'
                f'<text x="{legend_x + 19}" y="{height - 22}" fill="var(--text-secondary)" '
                f'font-size="10">{html.escape(label)}</text>'
            )
            legend_x += 22 + 7 * len(label)
    body += legend

    axes = (
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        f'stroke="var(--axis)"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="var(--axis)"/>'
    )
    captions = ""
    if x_label:
        captions += (
            f'<text x="{left + plot_w / 2:.0f}" y="{height - 6}" fill="var(--text-muted)" '
            f'font-size="10" text-anchor="middle">{html.escape(x_label)}</text>'
        )
    if y_label:
        captions += (
            f'<text x="12" y="{top + plot_h / 2:.0f}" fill="var(--text-muted)" font-size="10" '
            f'text-anchor="middle" transform="rotate(-90 12 {top + plot_h / 2:.0f})">'
            f'{html.escape(y_label)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'role="img" xmlns="http://www.w3.org/2000/svg">'
        f"{grid}{axes}{body}{captions}</svg>"
    )


def _cap_series(
    series: dict[str, Sequence[tuple[float, float]]], cap: int
) -> dict[str, Sequence[tuple[float, float]]]:
    """Keep the first *cap* series by name; the rest are dropped, not recoloured.

    Silently renaming the ninth series "Other" would be a lie about which line
    the reader is looking at — the caller facets instead.
    """
    if len(series) <= cap:
        return series
    return dict(list(series.items())[:cap])


def _tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def png_data_uri(data: bytes) -> str:
    """Embed an image so a report is one self-contained file."""
    return "data:image/png;base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --surface-0: #f4f3f1;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --border: #dedcd6;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #77756f;
  --grid: #e6e4df;
  --axis: #c7c5bf;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #4a3aa7;
  --series-8: #e34948;
  --sev-safe: #0ca30c;
  --sev-info: #52514e;
  --sev-low: #1c5cab;
  --sev-medium: #fab219;
  --sev-high: #ec835a;
  --sev-critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --surface-2: #222220;
    --border: #333330;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #918f86;
    --grid: #2b2b28;
    --axis: #3d3d39;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
    --sev-info: #c3c2b7;
    --sev-low: #6da7ec;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --surface-2: #222220;
  --border: #333330;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #918f86;
  --grid: #2b2b28;
  --axis: #3d3d39;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
  --series-4: #c98500;
  --series-5: #d55181;
  --series-6: #008300;
  --series-7: #9085e9;
  --series-8: #e66767;
  --sev-info: #c3c2b7;
  --sev-low: #6da7ec;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px;
  background: var(--surface-0); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.35rem; margin: 0 0 4px; }
h2 { font-size: 1rem; margin: 32px 0 10px; font-weight: 600; }
.sub { color: var(--text-secondary); font-size: .85rem; margin: 0 0 26px; }
.cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 20px; min-width: 132px;
}
.card .k { font-size: .7rem; letter-spacing: .07em; text-transform: uppercase;
           color: var(--text-muted); }
.card .v { font-size: 1.5rem; font-weight: 650; margin-top: 3px; }
.panel {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 4px 0; overflow-x: auto;
}
table { width: 100%; border-collapse: collapse; font-size: .85rem; min-width: 620px; }
th {
  text-align: left; padding: 10px 16px; font-size: .7rem; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; color: var(--text-muted);
}
td { padding: 9px 16px; border-top: 1px solid var(--border); vertical-align: top; }
td.loc { color: var(--text-secondary); font-family: ui-monospace, Menlo, monospace;
         font-size: .78rem; white-space: nowrap; }
.sev { font-weight: 600; white-space: nowrap; }
.glyph { display: inline-block; width: 1.1em; }
.detail { color: var(--text-secondary); }
.chart { padding: 14px 16px; }
.meta { display: flex; gap: 22px; flex-wrap: wrap; font-size: .8rem;
        color: var(--text-secondary); padding: 10px 16px; }
.meta b { color: var(--text-primary); font-weight: 600; }
.empty { padding: 14px 16px; color: var(--text-secondary); font-size: .85rem; }
.err { color: var(--sev-critical); }
figure { margin: 0; }
figure img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }
figcaption { color: var(--text-muted); font-size: .75rem; margin-top: 5px; }
.shots { display: flex; gap: 16px; flex-wrap: wrap; padding: 14px 16px; }
"""


def _sev_css(severity: Severity) -> str:
    return f"var(--sev-{severity.value.lower()})"


def _sev_html(severity: Severity) -> str:
    return (
        f'<span class="sev" style="color:{_sev_css(severity)}">'
        f'<span class="glyph">{_SEVERITY_GLYPH[severity]}</span>{severity.value}</span>'
    )


def render_html(report: Report, *, title: str = "AI security report") -> str:
    """One self-contained HTML file: no external fonts, scripts or stylesheets."""
    counts = {sev: n for sev, n in report.counts().items() if n}
    verdict_sev = _VERDICT_SEVERITY[report.verdict]

    cards = [
        ("Verdict", f'<span style="color:{_sev_css(verdict_sev)}">'
                    f"{_SEVERITY_GLYPH[verdict_sev]} {report.verdict.value}</span>"),
        ("Findings", str(len(report.findings))),
        ("Modules", str(len(report.results))),
    ]
    for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM):
        if counts.get(sev):
            cards.append((sev.value.title(), f'<span style="color:{_sev_css(sev)}">'
                                             f"{counts[sev]}</span>"))

    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in cards
    )

    sections = "".join(_render_result_html(result) for result in report.results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(report.target or "multiple targets")}
 &nbsp;·&nbsp; {html.escape(report.timestamp)}</p>
<div class="cards">{card_html}</div>
{sections}
</main>
</body>
</html>
"""


def _render_result_html(result: ModuleResult) -> str:
    head = (
        f"<h2>{html.escape(result.module)} "
        f'<span style="color:var(--text-muted);font-weight:400">'
        f"· {result.kind.value}</span></h2>"
    )

    if result.skipped:
        return head + f'<div class="panel"><div class="empty">Skipped — {html.escape(result.skipped)}</div></div>'
    if result.error:
        return head + f'<div class="panel"><div class="empty err">Error — {html.escape(result.error)}</div></div>'

    blocks: list[str] = []

    scalars = {
        k: v for k, v in result.metrics.items()
        if k not in ("charts", "images", "traceback")
    }
    if scalars:
        blocks.append(
            '<div class="meta">'
            + "".join(
                f"<span>{html.escape(str(k))} <b>{html.escape(_fmt_metric(v))}</b></span>"
                for k, v in scalars.items()
            )
            + "</div>"
        )

    for chart in result.metrics.get("charts", []) or []:
        svg = svg_line_chart(
            {k: v for k, v in chart.get("series", {}).items()},
            x_label=chart.get("x_label", ""),
            y_label=chart.get("y_label", ""),
            y_min=chart.get("y_min"),
            y_max=chart.get("y_max"),
        )
        if svg:
            caption = chart.get("title", "")
            blocks.append(
                '<div class="chart">'
                + (f'<div style="color:var(--text-muted);font-size:.75rem;'
                   f'margin-bottom:6px">{html.escape(caption)}</div>' if caption else "")
                + svg
                + "</div>"
            )

    images = result.metrics.get("images", []) or []
    if images:
        blocks.append(
            '<div class="shots">'
            + "".join(
                f'<figure><img src="{img["uri"]}" alt="{html.escape(img.get("caption", ""))}">'
                f'<figcaption>{html.escape(img.get("caption", ""))}</figcaption></figure>'
                for img in images
                if img.get("uri")
            )
            + "</div>"
        )

    if result.findings:
        rows = "".join(
            "<tr>"
            f"<td>{_sev_html(f.severity)}</td>"
            f'<td class="loc">{html.escape(f.location)}</td>'
            f"<td><b>{html.escape(f.title)}</b></td>"
            f'<td class="detail">{html.escape(f.summary or f.detail)}</td>'
            f"<td>{'' if f.score is None else f'{f.score:.4f}'}</td>"
            "</tr>"
            for f in sorted(result.findings, key=lambda f: (-f.severity.rank, f.location))
        )
        blocks.append(
            '<table><thead><tr><th>Severity</th><th>Location</th><th>Finding</th>'
            "<th>Detail</th><th>Score</th></tr></thead><tbody>"
            f"{rows}</tbody></table>"
        )
    else:
        blocks.append('<div class="empty">No findings.</div>')

    return head + '<div class="panel">' + "".join(blocks) + "</div>"


def findings_table_text(findings: Iterable[Finding]) -> str:
    """Plain-text table — the accessible fallback the colour rules require."""
    lines = [f"{'SEVERITY':<10} {'LOCATION':<28} TITLE"]
    for finding in sorted(findings, key=lambda f: (-f.severity.rank, f.location)):
        lines.append(f"{finding.severity.value:<10} {finding.location[:27]:<28} {finding.title}")
    return "\n".join(lines)
