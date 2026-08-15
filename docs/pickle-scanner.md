# Architecture — Memory-Safe Pickle Scanner

## Overview

The scanner performs static analysis of pickle bytecode to detect dangerous
opcodes without executing any of the payload.  It never calls `pickle.loads`,
never imports user-controlled modules and never invokes any callable contained
in the file.  It is safe to run on untrusted inputs.

---

## Why Static Pickle Analysis

Python's `pickle` module is intentionally Turing-complete at deserialisation
time.  The `GLOBAL` opcode can import any module attribute, and `REDUCE` can
call it with arbitrary arguments.  A malicious actor can embed shellcode,
reverse shells or file-exfiltration commands in a `.pkl` or `.pt` file that
fires the moment `torch.load` or `pickle.load` is called.

Existing defences (`pickle.loads` with a restricted `Unpickler`) are still
vulnerable because the restriction is applied after the bytecode runs far
enough to determine the callable.  The only safe approach is to never execute
the bytecode at all.

---

## Pipeline

```
File on disk
    │
    ▼
Extractor (scanner/extractor.py)
    Detects container format:
        raw pickle → single Payload
        PyTorch ZIP → extract data.pkl members
        NumPy .npy  → no pickle payload
        Generic ZIP → scan all .pkl members
    │
    ▼ bytes (one per pickle stream)
    │
Parser (scanner/parser.py)
    Decodes opcode bytes and arguments WITHOUT executing anything.
    Yields Instruction(offset, opcode, arg) objects.
    Handles protocols 0–5.
    │
    ▼ stream of Instructions
    │
Analyser (scanner/analyser.py)
    Maintains a lightweight symbolic string stack.
    Emits Finding objects with Severity and detail.
    Applies KNOWN_SAFE_GLOBALS whitelist.
    Applies DANGEROUS_MODULES blacklist.
    │
    ▼ ScanResult with findings list
    │
Reporter (scanner/reporter.py)
    Renders colour-coded terminal output or JSON.
```

---

## Opcode Classification

### CRITICAL

| Opcode | Risk |
|--------|------|
| `GLOBAL` | Imports an arbitrary module attribute. Combined with `REDUCE` this executes arbitrary code. |
| `STACK_GLOBAL` | Same as GLOBAL but sources module and attribute from the stack (protocol 4+). |
| `INST` | Instantiates a class by module and classname string — equivalent to GLOBAL + REDUCE. |

Any `GLOBAL` targeting a module in `DANGEROUS_MODULES` (os, subprocess,
builtins, sys, socket, ctypes, …) remains CRITICAL.  Targeting a module in
`KNOWN_SAFE_GLOBALS` (torch._utils, numpy.core.multiarray, …) is downgraded
to INFO because this is the normal pattern for ML checkpoint serialisation.

### HIGH

| Opcode | Risk |
|--------|------|
| `REDUCE` | Calls the top-of-stack callable with the next-on-stack argument tuple. |
| `BUILD` | Calls `__setstate__` or updates `__dict__` on the just-constructed object. |
| `OBJ` | Constructs an object from a class resolved at runtime. |

### MEDIUM

| Opcode | Risk |
|--------|------|
| `NEWOBJ` | Calls `cls.__new__(cls, *args)` — less dangerous than REDUCE but still invokes custom `__new__`. |
| `NEWOBJ_EX` | Same with keyword arguments (protocol 4+). |

### LOW

| Opcode | Risk |
|--------|------|
| `PERSID` | Invokes the Unpickler's `persistent_load` hook — safe only if the hook is trusted. |
| `BINPERSID` | Same, with stack-sourced ID. |

### INFO

Everything else — data primitives, memo management, protocol framing.

---

## Symbolic Stack Tracking

The analyser maintains a simplified string stack to resolve `STACK_GLOBAL`
(which sources its module and name from the stack rather than inline bytes)
and to annotate `REDUCE` findings with the callee's name.

The stack tracks string pushes (`SHORT_BINUNICODE`, `BINUNICODE`, `STRING`,
etc.) and clears on `REDUCE`/`NEWOBJ` (which consume the top of stack).
It does not attempt full stack simulation — that would be unnecessary
complexity given that the goal is detection, not exact emulation.

---

## Known-Safe Globals Whitelist

The whitelist covers globals that appear in every normal PyTorch and NumPy
checkpoint:

```
torch._utils._rebuild_tensor_v2
torch._utils._rebuild_parameter
torch.Tensor
numpy.core.multiarray._reconstruct
numpy.core.multiarray.scalar
collections.OrderedDict
...
```

These are downgraded from CRITICAL to INFO so that scanning a normal
`model.pt` does not produce alarming false positives.  The whitelist is
conservative — any name not on it from a non-dangerous module retains its
default severity.

---

## Container Detection (extractor.py)

| Magic bytes | Format | Action |
|-------------|--------|--------|
| `PK\x03\x04` | ZIP | Extract all `.pkl` members; scan non-pkl members heuristically |
| `\x93NUMPY` | NumPy .npy | Report as non-pickle format |
| Anything else | Raw pickle | Scan directly |

PyTorch `.pt` / `.pth` files are ZIP archives.  The tensor data lives in
`data.pkl`; the scanner extracts and analyses each pickle member
independently so that a malicious payload injected into a single member is
still detected.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All files clean (max severity below HIGH) |
| 1 | At least one HIGH or CRITICAL finding, or a file that errored |

`--exit-zero` overrides this to always return 0, for use in pipelines where
the scanner is advisory only.

---

## Files

```
pickle_scanner/
├── scan.py                        — CLI entry point
├── scanner/
│   ├── __init__.py                — public API: scan_file, scan_bytes
│   ├── opcodes.py                 — Severity, Finding, ScanResult, opcode tables
│   ├── parser.py                  — PickleParser: opcode byte decoder
│   ├── analyser.py                — Analyser: emits Findings from Instructions
│   ├── extractor.py               — container detection and payload extraction
│   ├── scanner.py                 — scan_file and scan_bytes entry points
│   └── reporter.py                — terminal and JSON output
├── tests/
│   ├── test_scanner.py            — pytest test suite (30+ tests)
│   └── fixtures/                  — generated by scripts/make_fixtures.py
└── scripts/
    └── make_fixtures.py           — generates known-malicious test pickles
```
