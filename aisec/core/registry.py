"""Which modules exist, and loading them only when they are actually needed.

The manifest is static data, not the result of importing anything. That is the
whole point: `aisec list` and `aisec scan model.pkl` must work on a machine
with no torch installed, and they do, because nothing under `modules/` is
imported until a module is selected to run.

Each module folder is its own source root — `modules/pickle-scanner` contains
the `pickle_scanner` package — so loading one means putting that folder on
`sys.path` and importing its `integration.py` under a unique name. Two modules
can both have an `integration.py` without colliding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .finding import Kind
from .module import Module, ModuleSpec

MODULES_ROOT = Path(__file__).resolve().parents[2] / "modules"


MANIFEST: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        name="pickle-scanner",
        kind=Kind.SCANNER,
        title="Pickle opcode scanner",
        summary=(
            "Walks pickle bytecode without executing it, flagging the opcodes "
            "that can reach arbitrary code — GLOBAL, REDUCE, INST, STACK_GLOBAL."
        ),
        extensions=(".pkl", ".pickle", ".pt", ".pth", ".joblib", ".bin", ".ckpt"),
        tags=("supply-chain", "static-analysis"),
    ),
    ModuleSpec(
        name="weight-poisoning",
        kind=Kind.SCANNER,
        title="Weight poisoning detector",
        summary=(
            "Statistical, spectral and neuron-level analysis of checkpoint "
            "tensors for backdoor and trojan signatures."
        ),
        requires=("torch",),
        extensions=(".pt", ".pth", ".ckpt"),
        tags=("supply-chain", "backdoor"),
    ),
    ModuleSpec(
        name="prompt-injection-firewall",
        kind=Kind.GUARD,
        title="Prompt injection firewall",
        summary=(
            "Channel separation, tool allowlisting and heuristic detection for "
            "untrusted text reaching an LLM."
        ),
        tags=("llm", "runtime"),
    ),
    ModuleSpec(
        name="adversarial-detection",
        kind=Kind.GUARD,
        title="Adversarial input detector",
        summary=(
            "Feature squeezing, input transformation and density estimates "
            "combined into one adversarial probability per input."
        ),
        requires=("torch",),
        payloads=("tensor",),
        tags=("evasion", "runtime"),
    ),
    ModuleSpec(
        name="model-extraction",
        kind=Kind.PROBE,
        title="Model extraction probe",
        summary=(
            "Trains a substitute against your model as a black box to measure "
            "how many queries buy how much fidelity."
        ),
        requires=("torch",),
        tags=("stealing", "query-budget"),
    ),
    ModuleSpec(
        name="gradient-leakage",
        kind=Kind.PROBE,
        title="Gradient leakage probe",
        summary=(
            "Reconstructs training inputs from shared gradients (DLG/iDLG/R-GAP) "
            "and scores how much each defense actually removes."
        ),
        requires=("torch",),
        tags=("privacy", "federated"),
    ),
)


_BY_NAME = {spec.name: spec for spec in MANIFEST}
_LOADED: dict[str, Module] = {}


class ModuleUnavailable(RuntimeError):
    """Raised when a module cannot run here — usually a missing dependency."""


def specs(kind: Kind | None = None, names: list[str] | None = None) -> list[ModuleSpec]:
    """The manifest, optionally narrowed by kind or explicit names."""
    found = list(MANIFEST)
    if kind is not None:
        found = [s for s in found if s.kind is kind]
    if names:
        wanted = {n.strip() for n in names}
        unknown = wanted - set(_BY_NAME)
        if unknown:
            raise KeyError(
                f"unknown module(s): {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(_BY_NAME))}"
            )
        found = [s for s in found if s.name in wanted]
    return found


def spec(name: str) -> ModuleSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown module {name!r}") from exc


def missing_requirements(spec_: ModuleSpec) -> list[str]:
    """Which of a module's third-party imports are not installed here."""
    absent = []
    for requirement in spec_.requires:
        if importlib.util.find_spec(requirement) is None:
            absent.append(requirement)
    return absent


def available(spec_: ModuleSpec) -> bool:
    return not missing_requirements(spec_)


def load(name: str) -> Module:
    """Import a module's `integration.py` and hand back its `MODULE`.

    Cached: probes get asked for their spec and then run, and importing torch
    twice for that would be a visible pause.
    """
    if name in _LOADED:
        return _LOADED[name]

    spec_ = spec(name)
    absent = missing_requirements(spec_)
    if absent:
        raise ModuleUnavailable(
            f"{name} needs {', '.join(absent)} — install with "
            f"`pip install -r modules/{name}/requirements.txt`"
        )

    folder = MODULES_ROOT / spec_.folder
    entry = folder / "integration.py"
    if not entry.is_file():
        raise ModuleUnavailable(f"{name} has no integration.py at {entry}")

    # The module folder is its own source root, so its package imports resolve
    # exactly as they do when the module is run standalone from that directory.
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

    unique = f"aisec._modules.{name.replace('-', '_')}"
    file_spec = importlib.util.spec_from_file_location(unique, entry)
    if file_spec is None or file_spec.loader is None:
        raise ModuleUnavailable(f"could not load {entry}")
    imported = importlib.util.module_from_spec(file_spec)
    sys.modules[unique] = imported
    file_spec.loader.exec_module(imported)

    instance = getattr(imported, "MODULE", None)
    if instance is None:
        raise ModuleUnavailable(f"{entry} defines no MODULE")
    _LOADED[name] = instance
    return instance


def module_path(name: str) -> Path:
    """Where a module lives, so one module can find a sibling without
    hardcoding a relative path that breaks the moment either one moves."""
    return MODULES_ROOT / spec(name).folder


def scanners_for(path: Path) -> list[ModuleSpec]:
    """Scanners that claim this file by extension."""
    return [s for s in specs(Kind.SCANNER) if s.handles(path)]
