# Memory-Safe Pickle Scanner

> Part of the [AI Security Suite](../../README.md). Runs standalone from this
> folder, or alongside every other module via the platform CLI — see
> [Using it](#using-it) below.

Static analysis of pickle bytecode to detect dangerous opcodes in `.pkl`,
`.pickle`, `.pt` and `.pth` files — without executing any of the payload.

Python's `pickle` format is Turing-complete at deserialisation time.  A
malicious model checkpoint can embed `os.system`, `subprocess.Popen` or
`builtins.eval` calls that fire the moment `torch.load` or `pickle.load` is
called.  This scanner walks the raw opcode stream and flags dangerous
constructs before any code runs.

---

## Features

- Parses pickle protocols 0–5 from first principles, no `pickle` module used
- Never calls `pickle.loads`, never imports or invokes user-controlled code
- Detects GLOBAL, STACK_GLOBAL, INST, REDUCE, BUILD, NEWOBJ, OBJ and more
- Known-safe ML globals (PyTorch, NumPy) whitelisted to suppress false positives
- High-risk modules (os, subprocess, builtins, ctypes, …) blacklisted as CRITICAL
- PyTorch ZIP checkpoint extraction — scans each embedded `data.pkl` stream
- Symbolic stack tracking for STACK_GLOBAL callee attribution
- Colour-coded terminal output or JSON for CI pipelines
- Exit code 1 when HIGH or CRITICAL findings are present
- 30+ pytest tests covering all opcode classes

---

## Requirements

Python 3.11+ — no runtime dependencies beyond the standard library.

```bash
pip install pytest    # for running tests only
```

---

## Usage

Scan a single file:

```bash
python scan.py model.pt
```

Scan all checkpoints in a directory recursively:

```bash
python scan.py checkpoints/ --recursive
```

Scan with verbose INFO-level output:

```bash
python scan.py suspicious.pkl --verbose
```

Raise severity for private C-extension modules:

```bash
python scan.py payload.pkl --strict
```

JSON output for CI pipelines:

```bash
python scan.py model.pt --json | jq '.[] | select(.max_severity == "CRITICAL")'
```

Only report HIGH and above:

```bash
python scan.py checkpoints/ -r --min-severity HIGH
```

---

## Example Output

```
── checkpoints/model.pt::data.pkl
  Protocol : 2
  Opcodes  : 147
  Status   : SAFE — no dangerous opcodes found

── suspicious.pkl
  Protocol : 2
  Opcodes  : 8
  Status   : CRITICAL
  Findings : 3

  [CRITICAL]  0x0002  GLOBAL               Imports an arbitrary module attribute
                                           os.system — high-risk module 'os'
  [HIGH    ]  0x0010  REDUCE               Calls a callable with a tuple of arguments
                                           Invokes os.system
  [HIGH    ]  0x0012  BUILD                Calls __setstate__ or updates __dict__
```

---

## Severity Levels

| Level | Meaning |
|-------|---------|
| CRITICAL | Arbitrary code execution is almost certain if unpickled |
| HIGH | Code execution is likely depending on context |
| MEDIUM | Non-obvious construction that may trigger custom `__new__` |
| LOW | Hooks into persistent ID or other extension points |
| INFO | Normal ML serialisation globals or protocol framing |
| SAFE | No findings |

---

## Running Tests

```bash
# Generate test fixtures first
python scripts/make_fixtures.py

# Run the test suite
python -m pytest tests/ -v
```

---

## Embedding as a Library

```python
from pickle_scanner import scan_file, scan_bytes
from pickle_scanner.opcodes import Severity

# Scan a file
results = scan_file("model.pt")
for result in results:
    if result.max_severity >= Severity.HIGH:
        print(f"DANGEROUS: {result.path}")
        for finding in result.findings:
            print(f"  {finding}")

# Scan raw bytes
import pickle
data   = pickle.dumps({"safe": True}, protocol=2)
result = scan_bytes(data)
print(result.safe)   # True
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full description of the parse
pipeline, opcode classification logic, symbolic stack tracking and container
detection.

---

## Project Structure

```
pickle_scanner/
├── scan.py
├── scanner/
│   ├── __init__.py
│   ├── opcodes.py
│   ├── parser.py
│   ├── analyser.py
│   ├── extractor.py
│   ├── scanner.py
│   └── reporter.py
├── tests/
│   └── test_scanner.py
└── scripts/
    └── make_fixtures.py
```
