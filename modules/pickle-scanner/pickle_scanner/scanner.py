"""Scan a file's pickle payloads without executing a single opcode.

A PyTorch checkpoint is a zip of several pickle streams, so one file yields
one `ModuleResult` per payload rather than one per file — a backdoor hidden in
the third tensor's stream is not the same finding as a clean `data.pkl`.
"""

from __future__ import annotations

from pathlib import Path

from pickle_scanner.analyser import Analyser
from pickle_scanner.extractor import extract_payloads
from pickle_scanner.opcodes import Kind, ModuleResult
from pickle_scanner.parser import ParseError, PickleParser

MODULE_NAME = "pickle-scanner"


def _blank(target: str) -> ModuleResult:
    return ModuleResult(module=MODULE_NAME, kind=Kind.SCANNER, target=target)


def scan_file(path: str | Path, strict: bool = False) -> list[ModuleResult]:
    """One result per pickle payload found in *path*.

    Args:
        path:   file to scan
        strict: raise severity for private C-extension modules
    """
    path = str(path)
    try:
        payloads = extract_payloads(path)
    except (OSError, PermissionError) as exc:
        result = _blank(path)
        result.error = str(exc)
        return [result]

    results: list[ModuleResult] = []
    parser = PickleParser()

    for payload in payloads:
        result = _blank(payload.source)

        if not payload.data:
            result.skipped = "no pickle payload (non-pickle format or empty file)"
            results.append(result)
            continue

        _consume(parser, payload.data, result, strict)
        if result.error and payload.heuristic:
            # Only the opening bytes suggested this member was a pickle, and it
            # did not parse — so it was not one. A PyTorch checkpoint stores its
            # tensors as raw floats, and reporting those as a malformed pickle
            # makes any caller that fails closed on errors reject a genuine
            # model. Recorded as skipped, never dropped: a member the scanner
            # did not read has to stay visible.
            result.skipped = f"not a pickle after all ({result.error})"
            result.error = ""
        results.append(result)

    return results


def scan_bytes(data: bytes, label: str = "<bytes>", strict: bool = False) -> ModuleResult:
    """Scan raw pickle bytes — the entry point for tests and for embedding."""
    result = _blank(label)
    _consume(PickleParser(), data, result, strict)
    return result


def _consume(parser: PickleParser, data: bytes, result: ModuleResult, strict: bool) -> None:
    analyser = Analyser(result, strict=strict)
    try:
        for instruction in parser.parse(data):
            analyser.feed(instruction)
    except ParseError as exc:
        result.error = f"Parse error: {exc}"
    result.metrics["protocol"] = analyser.proto
    result.metrics["opcodes"] = analyser.n_opcodes
