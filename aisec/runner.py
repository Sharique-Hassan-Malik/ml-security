"""Fan one target out across every module that claims it, into one report.

A module that is not installable here is *skipped and said so*, never silently
dropped. "Scanned, clean" and "could not scan" are different answers and a
security tool that blurs them is worse than one that refuses to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .core import registry
from .core.finding import Kind, ModuleResult, Report

_ARCHIVE_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".bin", ".ckpt", ".safetensors"}


def collect_files(targets: Iterable[str], recursive: bool = False) -> list[Path]:
    """Expand files, directories and globs into a deduplicated file list."""
    found: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            walk = path.rglob if recursive else path.glob
            for suffix in sorted(_ARCHIVE_SUFFIXES):
                found.extend(walk(f"*{suffix}"))
        else:
            root = Path(".")
            matches = root.rglob(target) if "**" in target else root.glob(target)
            found.extend(m for m in matches if m.is_file())

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _skipped(spec, target: str, reason: str) -> ModuleResult:
    return ModuleResult(module=spec.name, kind=spec.kind, target=target, skipped=reason)


def _run(spec, target: str, call) -> ModuleResult:
    absent = registry.missing_requirements(spec)
    if absent:
        return _skipped(spec, target, f"needs {', '.join(absent)}")
    try:
        module = registry.load(spec.name)
    except registry.ModuleUnavailable as exc:
        return _skipped(spec, target, str(exc))
    return module.execute(lambda: call(module), target=target)


def scan(
    paths: Iterable[Path],
    *,
    only: list[str] | None = None,
    options: dict[str, Any] | None = None,
) -> Report:
    """Run every scanner that claims each path."""
    options = options or {}
    paths = list(paths)
    report = Report(target=_describe(paths))

    for path in paths:
        for spec in registry.specs(Kind.SCANNER, only):
            if not spec.handles(path):
                continue
            report.add(_run(spec, str(path), lambda m, p=path: m.scan(p, **options)))
    return report


def payload_kind(payload: Any) -> str:
    """Text or tensor. Decided once, here, so guards never have to guess."""
    if hasattr(payload, "shape") and hasattr(payload, "dtype"):
        return "tensor"
    if isinstance(payload, (str, Path)):
        suffix = Path(str(payload)).suffix.lower()
        if suffix in (".pt", ".pth", ".npy", ".npz"):
            return "tensor"
    return "text"


def guard(
    payload: Any,
    *,
    only: list[str] | None = None,
    options: dict[str, Any] | None = None,
    label: str = "<input>",
) -> Report:
    """Run every guard that accepts this kind of payload."""
    options = options or {}
    kind = payload_kind(payload)
    report = Report(target=label)
    for spec in registry.specs(Kind.GUARD, only):
        if not spec.accepts(kind):
            report.add(_skipped(spec, label, f"takes {'/'.join(spec.payloads)}, got {kind}"))
            continue
        report.add(_run(spec, label, lambda m: m.inspect(payload, **options)))
    return report


def probe(name: str, *, options: dict[str, Any] | None = None) -> Report:
    """Run one named probe. Probes are expensive, so never more than asked for."""
    options = options or {}
    spec = registry.spec(name)
    if spec.kind is not Kind.PROBE:
        raise ValueError(f"{name} is a {spec.kind.value}, not a probe")
    report = Report(target=name)
    report.add(_run(spec, name, lambda m: m.run(**options)))
    return report


def _describe(paths: list[Path]) -> str:
    if not paths:
        return ""
    if len(paths) == 1:
        return str(paths[0])
    return f"{len(paths)} files"
