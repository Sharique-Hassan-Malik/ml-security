from __future__ import annotations

import math
import re
import urllib.parse
from typing import Sequence


# Characters that commonly appear in injection payloads
_SPECIAL = set("'\"<>|;`(){}[]\\&$#@!%^*+=~")

_SQL_KW = re.compile(
    r"(?i)\b(select|insert|update|delete|union|from|where|having|order|group|"
    r"drop|create|alter|exec|execute|cast|convert|char|nchar|varchar|declare|"
    r"sleep|benchmark|information_schema|sysobjects|xp_cmdshell)\b"
)
_XSS_KW  = re.compile(r"(?i)(script|onerror|onload|alert|document\.cookie|eval\s*\()")
_CMDI_KW = re.compile(r"(?i)(;|\||\`|\$\()(ls|cat|id|whoami|bash|sh|curl|wget|nc)\b")
_TRAV_KW = re.compile(r"(\.\.[\\/]|%2e%2e|%252e)")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _special_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if c in _SPECIAL) / len(s)


def _digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if c.isdigit()) / len(s)


def _count_params(query: str) -> int:
    if not query:
        return 0
    return len(urllib.parse.parse_qsl(query, keep_blank_values=True))


def _max_param_value_len(query: str) -> int:
    if not query:
        return 0
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    return max((len(v) for _, v in pairs), default=0)


def extract(request: dict) -> list[float]:
    """
    Extract a fixed-length feature vector from a request dict.

    Keys used: method, url, query, body, headers (dict).

    Returns a list of 38 floats — see FEATURE_NAMES for the index map.
    """
    url     = request.get("url", "") or ""
    query   = request.get("query", "") or ""
    body    = request.get("body", "") or ""
    headers = request.get("headers", {}) or {}
    method  = (request.get("method", "GET") or "GET").upper()

    if not query and "?" in url:
        path, query = url.split("?", 1)
    else:
        path = url.split("?")[0]

    full = " ".join([url, query, body])

    ua      = str(headers.get("User-Agent", ""))
    referer = str(headers.get("Referer", ""))
    cookie  = str(headers.get("Cookie", ""))

    # URL / path features
    f0  = len(url)
    f1  = len(path)
    f2  = path.count("/")
    f3  = len(query)
    f4  = _entropy(query)
    f5  = _special_ratio(query)
    f6  = _digit_ratio(query)
    f7  = _count_params(query)
    f8  = _max_param_value_len(query)
    f9  = url.count("%")                     # percent-encoded chars
    f10 = url.lower().count("../")           # traversal sequences
    f11 = 1.0 if url.lower().startswith("https") else 0.0

    # Body features
    f12 = len(body)
    f13 = _entropy(body)
    f14 = _special_ratio(body)
    f15 = _digit_ratio(body)
    f16 = body.count("%")

    # Combined / cross-field
    f17 = len(full)
    f18 = _entropy(full)
    f19 = _special_ratio(full)

    # Keyword signal counts
    f20 = len(_SQL_KW.findall(full))
    f21 = len(_XSS_KW.findall(full))
    f22 = len(_CMDI_KW.findall(full))
    f23 = len(_TRAV_KW.findall(full))
    f24 = full.lower().count("script")
    f25 = full.lower().count("select")
    f26 = full.lower().count("union")
    f27 = full.count("'")
    f28 = full.count('"')
    f29 = full.count(";")
    f30 = full.count("|")
    f31 = full.count("`")

    # Header-derived
    f32 = len(ua)
    f33 = _entropy(ua)
    f34 = len(referer)
    f35 = len(cookie)
    f36 = _entropy(cookie)

    # Method encoding
    f37 = 1.0 if method == "POST" else (0.5 if method in ("PUT", "PATCH", "DELETE") else 0.0)

    return [
        float(f0), float(f1), float(f2), float(f3), f4, f5, f6,
        float(f7), float(f8), float(f9), float(f10), f11,
        float(f12), f13, f14, f15, float(f16),
        float(f17), f18, f19,
        float(f20), float(f21), float(f22), float(f23),
        float(f24), float(f25), float(f26),
        float(f27), float(f28), float(f29), float(f30), float(f31),
        float(f32), f33, float(f34), float(f35), f36,
        f37,
    ]


FEATURE_NAMES = [
    "url_len", "path_len", "path_depth", "query_len",
    "query_entropy", "query_special_ratio", "query_digit_ratio",
    "param_count", "max_param_value_len", "url_pct_encoded", "url_traversal_seqs",
    "is_https",
    "body_len", "body_entropy", "body_special_ratio", "body_digit_ratio", "body_pct_encoded",
    "full_len", "full_entropy", "full_special_ratio",
    "sql_kw_count", "xss_kw_count", "cmdi_kw_count", "traversal_kw_count",
    "script_count", "select_count", "union_count",
    "single_quote_count", "double_quote_count", "semicolon_count",
    "pipe_count", "backtick_count",
    "ua_len", "ua_entropy", "referer_len", "cookie_len", "cookie_entropy",
    "method_score",
]

assert len(FEATURE_NAMES) == 38
