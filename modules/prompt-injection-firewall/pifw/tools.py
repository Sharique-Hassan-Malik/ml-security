"""Tool-call allowlisting: the control that bounds a successful injection.

Detection asks "is this text an attack?", which has no reliable answer. This
asks "is this call permitted?", which has an exact one.

The distinction matters because almost every real prompt-injection incident is
injection *plus* an over-broad tool. If the model can only call
`search(query: str)`, a total detection failure produces a bad search. If it can
call `http_request(url, method, body)`, the same failure exfiltrates the
conversation.

Three layers, cheapest first:

  1. the tool is on the allowlist at all
  2. its arguments match the declared schema
  3. argument *values* pass a policy -- the URL is in-domain, the path is inside
     the sandbox, the amount is under the limit

Layer 3 is the one people skip, and it is where `read_file("../../.ssh/id_rsa")`
lives: a permitted tool, a schema-valid argument, and a catastrophe.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ToolPolicy:
    name: str
    parameters: dict[str, type] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    # Value-level policy, per argument.
    allowed_domains: tuple[str, ...] = ()
    allowed_path_prefixes: tuple[str, ...] = ()
    max_length: int = 4096
    numeric_limits: dict[str, float] = field(default_factory=dict)
    # Tools whose effects cannot be undone need a human, not a better detector.
    requires_confirmation: bool = False


@dataclass
class ToolVerdict:
    allowed: bool
    reason: str = ""
    layer: str = ""
    needs_confirmation: bool = False


class ToolGuard:
    def __init__(self, policies: list[ToolPolicy]):
        self.policies = {policy.name: policy for policy in policies}
        self.checked = 0
        self.denied = 0

    def check(self, call: ToolCall) -> ToolVerdict:
        self.checked += 1
        policy = self.policies.get(call.name)
        if policy is None:
            self.denied += 1
            # Default-deny. An allowlist that falls open is a denylist.
            return ToolVerdict(False, f"tool {call.name!r} is not on the allowlist "
                                      f"(allowed: {sorted(self.policies)})", "allowlist")

        unknown = set(call.arguments) - set(policy.parameters)
        if unknown:
            self.denied += 1
            return ToolVerdict(False, f"unexpected argument(s): {sorted(unknown)}", "schema")
        missing = [name for name in policy.required if name not in call.arguments]
        if missing:
            self.denied += 1
            return ToolVerdict(False, f"missing required argument(s): {missing}", "schema")

        for name, value in call.arguments.items():
            expected = policy.parameters[name]
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, expected):
                self.denied += 1
                return ToolVerdict(False, f"{name} must be {expected.__name__}, "
                                          f"got {type(value).__name__}", "schema")
            if isinstance(value, str) and len(value) > policy.max_length:
                self.denied += 1
                return ToolVerdict(False, f"{name} exceeds {policy.max_length} characters", "schema")

            verdict = self._check_value(policy, name, value)
            if verdict is not None:
                self.denied += 1
                return verdict

        return ToolVerdict(True, "permitted", "policy",
                           needs_confirmation=policy.requires_confirmation)

    def _check_value(self, policy: ToolPolicy, name: str, value) -> ToolVerdict | None:
        if isinstance(value, str) and policy.allowed_domains and _looks_like_url(value):
            host = (urlparse(value).hostname or "").lower()
            if not any(host == d or host.endswith("." + d) for d in policy.allowed_domains):
                return ToolVerdict(False, f"{name}: host {host!r} is not in "
                                          f"{list(policy.allowed_domains)}", "value-policy")

        if isinstance(value, str) and policy.allowed_path_prefixes and _looks_like_path(value):
            # normpath first: the whole point of ../.. is that the raw string
            # passes a prefix check the resolved path would fail.
            resolved = os.path.normpath(os.path.join("/sandbox", value)) if not value.startswith("/") \
                else os.path.normpath(value)
            if not any(resolved == p.rstrip("/") or resolved.startswith(p.rstrip("/") + "/")
                       for p in policy.allowed_path_prefixes):
                return ToolVerdict(False, f"{name}: {resolved!r} escapes "
                                          f"{list(policy.allowed_path_prefixes)}", "value-policy")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            limit = policy.numeric_limits.get(name)
            if limit is not None and value > limit:
                return ToolVerdict(False, f"{name}={value} exceeds the limit {limit}", "value-policy")
        return None


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE))


def _looks_like_path(value: str) -> bool:
    return "/" in value and not _looks_like_url(value)


def default_policies() -> list[ToolPolicy]:
    """A deliberately narrow default set, as a worked example."""
    return [
        ToolPolicy("search", {"query": str}, ("query",), max_length=512),
        ToolPolicy("fetch_url", {"url": str}, ("url",),
                   allowed_domains=("example.com", "docs.internal")),
        ToolPolicy("read_file", {"path": str}, ("path",),
                   allowed_path_prefixes=("/sandbox/data",)),
        ToolPolicy("send_email", {"to": str, "subject": str, "body": str},
                   ("to", "subject", "body"), requires_confirmation=True),
        ToolPolicy("transfer", {"account": str, "amount": float}, ("account", "amount"),
                   numeric_limits={"amount": 100.0}, requires_confirmation=True),
    ]
