"""
CSIC 2010 HTTP dataset parser.

The dataset ships as two text files — normalTraffic.txt and anomalousTraffic.txt —
where requests are separated by blank lines and each request begins with the
HTTP method line followed by headers and an optional body.

Download: http://www.isi.csic.es/dataset/
Place files at data/csic_normal.txt and data/csic_anomalous.txt.
"""

from __future__ import annotations

import re
from pathlib import Path


def _parse_block(block: str) -> dict | None:
    """Parse a single request block into a request dict."""
    lines = block.strip().splitlines()
    if not lines:
        return None

    # First line: METHOD /path HTTP/version
    m = re.match(r"(\S+)\s+(\S+)\s+HTTP/[\d.]+", lines[0])
    if not m:
        return None

    method = m.group(1).upper()
    url    = m.group(2)
    path   = url.split("?")[0]
    query  = url.split("?")[1] if "?" in url else ""

    headers: dict[str, str] = {}
    body    = ""
    i       = 1

    while i < len(lines) and lines[i].strip():
        colon = lines[i].find(":")
        if colon != -1:
            k = lines[i][:colon].strip()
            v = lines[i][colon + 1:].strip()
            headers[k] = v
        i += 1

    # skip the blank line
    i += 1
    if i < len(lines):
        body = "\n".join(lines[i:]).strip()

    return {
        "method":  method,
        "url":     url,
        "query":   query,
        "body":    body,
        "headers": headers,
    }


def load(
    normal_path:    str | Path,
    anomalous_path: str | Path,
    max_normal:     int = 10_000,
    max_anomalous:  int = 10_000,
) -> tuple[list[dict], list[int]]:
    """
    Returns (requests, labels) where label 0 = benign and 1 = malicious.
    Caps the number of samples to avoid memory issues with large datasets.
    """
    requests: list[dict] = []
    labels:   list[int]  = []

    def _read(path: Path, label: int, cap: int) -> None:
        if not path.exists():
            return
        text   = path.read_text(errors="replace")
        blocks = re.split(r"\n\s*\n", text)
        count  = 0
        for block in blocks:
            if count >= cap:
                break
            r = _parse_block(block)
            if r:
                requests.append(r)
                labels.append(label)
                count += 1

    _read(Path(normal_path),    0, max_normal)
    _read(Path(anomalous_path), 1, max_anomalous)
    return requests, labels


def is_available(data_dir: str | Path = "data") -> bool:
    d = Path(data_dir)
    return (d / "csic_normal.txt").exists() and (d / "csic_anomalous.txt").exists()
