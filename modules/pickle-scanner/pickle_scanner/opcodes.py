"""Static severity data for pickle opcodes.

The `Severity` and `Finding` types used here are the suite-wide ones from
`aisec.core` — this module contributes the opcode knowledge, not a private
vocabulary for describing risk. Kept importable on its own: `aisec.core` is
stdlib-only, so `from pickle_scanner.opcodes import Severity` costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

if __package__ in (None, ""):  # running straight from this folder
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aisec.core.finding import Finding, Kind, ModuleResult, Severity  # noqa: E402

__all__ = ["Finding", "Kind", "ModuleResult", "Severity", "DANGEROUS_OPCODES",
           "KNOWN_SAFE_GLOBALS", "DANGEROUS_MODULES"]


def opcode_finding(
    opcode: str,
    offset: int,
    severity: Severity,
    description: str,
    detail: str = "",
) -> Finding:
    """Build a suite Finding from the coordinates a pickle stream speaks in."""
    return Finding(
        title=opcode,
        severity=severity,
        summary=description,
        detail=detail,
        location=f"offset 0x{offset:04x}",
        metadata={"opcode": opcode, "offset": offset},
    )


# ---------------------------------------------------------------------------
# Dangerous opcode registry
# ---------------------------------------------------------------------------

# Each entry: opcode_name → (Severity, short description, detail template)
# Detail template may reference {arg} which is filled in at scan time.
DANGEROUS_OPCODES: dict[str, tuple[Severity, str, str]] = {
    # ── Arbitrary code execution ──────────────────────────────────────────
    "GLOBAL": (
        Severity.CRITICAL,
        "Imports an arbitrary module attribute",
        "Calls __import__('{module}') then getattr(module, '{name}'). "
        "Enables import of os, subprocess, builtins, etc.",
    ),
    "INST": (
        Severity.CRITICAL,
        "Instantiates a class by module/classname string",
        "Equivalent to GLOBAL + REDUCE; fully arbitrary instantiation.",
    ),
    "REDUCE": (
        Severity.HIGH,
        "Calls a callable with a tuple of arguments",
        "Invokes any previously pushed callable. Combined with GLOBAL or "
        "INST this executes arbitrary code.",
    ),
    "BUILD": (
        Severity.HIGH,
        "Calls __setstate__ or updates __dict__",
        "Can trigger custom __setstate__ methods on deserialized objects.",
    ),
    "NEWOBJ": (
        Severity.MEDIUM,
        "Calls cls.__new__(cls, *args)",
        "Constructs an object via __new__; less dangerous than REDUCE but "
        "can still invoke custom __new__ implementations.",
    ),
    "NEWOBJ_EX": (
        Severity.MEDIUM,
        "Calls cls.__new__(cls, *args, **kwargs) — protocol 4+",
        "Extended NEWOBJ with keyword arguments.",
    ),
    "STACK_GLOBAL": (
        Severity.CRITICAL,
        "Imports module attribute from stack strings — protocol 4+",
        "Pushes __import__(module).__getattr__(name); same risk as GLOBAL.",
    ),
    # ── Potentially dangerous depending on target ─────────────────────────
    "OBJ": (
        Severity.HIGH,
        "Instantiates an object using the top of the stack as class",
        "Class is resolved at runtime from the stack.",
    ),
    # ── Protocol / framing ────────────────────────────────────────────────
    "PROTO": (
        Severity.INFO,
        "Declares pickle protocol version",
        "",
    ),
    "FRAME": (
        Severity.INFO,
        "Protocol 4 framing opcode",
        "",
    ),
    # ── Persistent ID (hook for custom object loading) ────────────────────
    "PERSID": (
        Severity.LOW,
        "Loads a persistent object by string ID",
        "Invokes PersistentUnpickler.persistent_load(); safe only if the "
        "unpickler's persistent_load is trusted.",
    ),
    "BINPERSID": (
        Severity.LOW,
        "Loads a persistent object by ID from stack",
        "Same risk as PERSID.",
    ),
    # ── Memo manipulation ─────────────────────────────────────────────────
    "DUP": (
        Severity.INFO,
        "Duplicates top of stack",
        "",
    ),
}

# Opcodes considered safe data primitives — logged only at INFO level when
# verbose mode is active, never flagged as a finding.
SAFE_OPCODES = frozenset({
    "INT", "LONG", "LONG1", "LONG4",
    "FLOAT", "BINFLOAT",
    "STRING", "BINSTRING", "SHORT_BINSTRING",
    "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
    "NONE", "NEWTRUE", "NEWFALSE",
    "EMPTY_LIST", "EMPTY_TUPLE", "EMPTY_DICT", "EMPTY_SET",
    "LIST", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3",
    "DICT", "FROZENSET",
    "APPEND", "APPENDS", "SETITEM", "SETITEMS",
    "ADD_ITEMS",
    "PUT", "BINPUT", "LONG_BINPUT",
    "GET", "BINGET", "LONG_BINGET",
    "MEMOIZE",
    "MARK", "POP", "POP_MARK",
    "STOP",
    "BYTEARRAY8", "NEXT_BUFFER", "READONLY_BUFFER",
})

# Known-safe (module, name) pairs that are commonly used in ML workflows.
# A GLOBAL/STACK_GLOBAL importing one of these is downgraded from CRITICAL
# to LOW as it is a normal serialisation pattern.
KNOWN_SAFE_GLOBALS: frozenset[tuple[str, str]] = frozenset({
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("torch", "Tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._tensor", "_rebuild_from_type_v2"),
    ("torch.storage", "_load_from_bytes"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("_codecs", "encode"),
    ("builtins", "bytearray"),
    ("builtins", "bytes"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "complex"),
    ("builtins", "slice"),
    ("builtins", "range"),
})

# High-risk module prefixes — any GLOBAL targeting these should always remain
# CRITICAL regardless of the exact name.
DANGEROUS_MODULES = frozenset({
    "os", "subprocess", "sys", "socket", "shutil", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "pty", "atexit", "signal", "gc", "tempfile",
    "builtins",   # exec, eval, open, __import__, compile, etc.
    "posix", "nt",
    "pickle", "pickletools",
    "_pickle",
})
