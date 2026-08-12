from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field

from .patterns import RULES, SEVERITY_SCORE, Rule


@dataclass
class RuleMatch:
    rule:        Rule
    matched_text: str
    field:       str   # which request field triggered it


@dataclass
class RuleVerdict:
    malicious:   bool
    score:       int           # sum of severity scores across all matches
    matches:     list[RuleMatch] = field(default_factory=list)
    categories:  set[str]      = field(default_factory=set)

    @property
    def primary_category(self) -> str | None:
        if not self.matches:
            return None
        # category with highest cumulative severity score
        totals: dict[str, int] = {}
        for m in self.matches:
            totals[m.rule.category] = totals.get(m.rule.category, 0) + SEVERITY_SCORE[m.rule.severity]
        return max(totals, key=totals.__getitem__)


class RuleEngine:
    """
    Evaluates a set of compiled regex rules against URL-decoded request fields.
    Each field is tested independently so the triggering field is reported.
    """

    THRESHOLD = 3   # minimum score to flag as malicious

    def __init__(self, threshold: int = THRESHOLD):
        self.threshold = threshold

    def inspect(self, request: dict) -> RuleVerdict:
        """
        request keys: method, url, query, headers (dict), body (str)
        Returns a RuleVerdict with all matched rules.
        """
        fields = self._extract_fields(request)
        matches: list[RuleMatch] = []

        for field_name, value in fields:
            if not value:
                continue
            decoded = _decode(value)
            for rule in RULES:
                m = rule.match(decoded)
                if m:
                    matches.append(RuleMatch(
                        rule=rule,
                        matched_text=m.group(0)[:120],
                        field=field_name,
                    ))

        score      = sum(SEVERITY_SCORE[m.rule.severity] for m in matches)
        categories = {m.rule.category for m in matches}
        return RuleVerdict(
            malicious=score >= self.threshold,
            score=score,
            matches=matches,
            categories=categories,
        )

    def _extract_fields(self, request: dict) -> list[tuple[str, str]]:
        fields = []

        url = request.get("url", "")
        fields.append(("url", url))

        query = request.get("query", "")
        if not query and "?" in url:
            query = url.split("?", 1)[1]
        if query:
            for key, value in urllib.parse.parse_qsl(query, keep_blank_values=True):
                fields.append((f"param:{key}", value))

        body = request.get("body", "")
        if body:
            fields.append(("body", body))
            ct = str(request.get("headers", {}).get("Content-Type", ""))
            if "application/x-www-form-urlencoded" in ct:
                for key, value in urllib.parse.parse_qsl(body, keep_blank_values=True):
                    fields.append((f"body_param:{key}", value))

        headers = request.get("headers", {})
        for hname in ("User-Agent", "Referer", "X-Forwarded-For", "Cookie"):
            if hname in headers:
                fields.append((f"header:{hname}", headers[hname]))

        return fields


def _decode(value: str) -> str:
    """Multi-pass URL decode to catch double-encoded payloads."""
    prev = None
    while prev != value:
        prev = value
        try:
            value = urllib.parse.unquote(value)
        except Exception:
            break
    return value
