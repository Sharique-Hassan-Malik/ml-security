"""What a module has to implement to join the suite.

Three base classes rather than one, because the three kinds really do have
different signatures. Forcing a `run()` on all of them would mean a scanner
taking a path it ignores, or a guard returning a report nobody awaits. The
common part — provenance, timing, error capture — lives in `Module.execute`
so no module writes it twice.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .finding import Kind, ModuleResult


@dataclass(frozen=True)
class ModuleSpec:
    """The registry entry for one module.

    `name` is both the folder under `modules/` and the CLI handle, so
    `aisec scan --only pickle-scanner` and `cd modules/pickle-scanner` agree.
    """

    name: str
    kind: Kind
    title: str
    summary: str
    requires: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    # What a guard can actually be handed. A text firewall and a tensor
    # detector are both guards, and running one on the other's input would
    # produce a confident answer about nothing.
    payloads: tuple[str, ...] = ("text",)

    @property
    def folder(self) -> str:
        return self.name

    def accepts(self, payload_kind: str) -> bool:
        return payload_kind in self.payloads

    def handles(self, path: Path) -> bool:
        """Whether this scanner claims a file, by extension."""
        if not self.extensions:
            return True
        return path.suffix.lower() in self.extensions


class Module(ABC):
    """Base for every registered module."""

    spec: ModuleSpec

    def __init__(self, spec: ModuleSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def result(self, target: str = "") -> ModuleResult:
        return ModuleResult(module=self.spec.name, kind=self.spec.kind, target=target)

    def execute(self, fn: Callable[[], ModuleResult], target: str = "") -> ModuleResult:
        """Run *fn*, timing it and turning a crash into a reported error.

        One module blowing up must not cost the operator the other five
        results — a scan that dies on file three of forty is worse than
        useless, because it looks like it finished.
        """
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 — deliberate: report, never abort the run
            result = self.result(target)
            result.error = f"{type(exc).__name__}: {exc}"
            result.metrics["traceback"] = traceback.format_exc(limit=6)
        result.elapsed = time.perf_counter() - started
        return result


class Scanner(Module):
    """Offline analysis of an artifact on disk. No model execution."""

    @abstractmethod
    def scan(self, target: Path, **options: Any) -> ModuleResult:
        ...


class Guard(Module):
    """A control in a live path: inspect a payload, return a decision.

    Guards must be cheap and must never raise into the caller's request path,
    which is why `ModuleResult.metrics` carries the allow/flag/block decision
    rather than the guard signalling by exception.
    """

    @abstractmethod
    def inspect(self, payload: Any, **options: Any) -> ModuleResult:
        ...


class Probe(Module):
    """An attack you run against your own model to measure exposure.

    Probes are the expensive ones — they train substitutes, invert gradients,
    search for adversarial perturbations — so nothing runs them implicitly.
    """

    @abstractmethod
    def run(self, **options: Any) -> ModuleResult:
        ...


@dataclass
class Options:
    """Loosely-typed run options, shared so the CLI can pass one bag to any kind."""

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values
