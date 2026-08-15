#!/usr/bin/env python3
"""
Generate test payloads for the scanner test suite.

All payloads are constructed by writing raw pickle bytes directly — they
are never unpickled here.  The payloads are written to tests/fixtures/.

Generated files:
    clean.pkl          — safe payload containing only a plain dict
    os_system.pkl      — CRITICAL: os.system("id") via GLOBAL + REDUCE
    subprocess.pkl     — CRITICAL: subprocess.check_output via STACK_GLOBAL
    eval_exec.pkl      — CRITICAL: builtins.eval
    build_attack.pkl   — HIGH: __setstate__ trigger via BUILD
    newobj.pkl         — MEDIUM: __new__ invocation
    persid.pkl         — LOW: persistent ID reference
    torch_safe.pkl     — INFO: legitimate PyTorch tensor globals
    multi_payload.pkl  — multiple payloads concatenated after STOP
"""

import os
import pickle
import struct
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def make_clean() -> bytes:
    """A plain {key: value} dict — no dangerous opcodes."""
    return pickle.dumps({"result": 42, "name": "test"}, protocol=2)


def make_os_system() -> bytes:
    """
    os.system("id") — the classic pickle RCE payload.
    Opcodes: PROTO GLOBAL MARK STRING TUPLE REDUCE STOP
    """
    return (
        b"\x80\x02"
        b"cos\nsystem\n"
        b"("
        b"X\x02\x00\x00\x00id"
        b"\x85"
        b"R"
        b"."
    )


def make_subprocess() -> bytes:
    """subprocess.check_output(['id']) via STACK_GLOBAL (proto 4)."""
    module = b"subprocess"
    name   = b"check_output"
    cmd    = b"id"
    return (
        b"\x80\x04"
        + b"\x8c" + bytes([len(module)]) + module
        + b"\x8c" + bytes([len(name)]) + name
        + b"\x93"
        + b"("
        + b"]"
        + b"\x8c" + bytes([len(cmd)]) + cmd
        + b"\x41"
        + b"\x85"
        + b"R"
        + b"."
    )


def make_eval_exec() -> bytes:
    """builtins.eval with a code string."""
    code = b"__import__('os').system('id')"
    return (
        b"\x80\x02"
        + b"cbuiltins\neval\n"
        + b"("
        + b"X" + struct.pack("<I", len(code)) + code
        + b"\x85"
        + b"R"
        + b"."
    )


def make_build_attack() -> bytes:
    """
    Trigger __setstate__ on an object via BUILD.
    The object class itself is harmless; the danger arrives via BUILD.
    """
    return (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n"
        b")R"
        b"}"
        b"b."
    )


def make_newobj() -> bytes:
    """NEWOBJ with a custom class."""
    return (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n"
        b")\x81."
    )


def make_persid() -> bytes:
    """PERSID reference."""
    return b"\x80\x02Psome_id\n."


def make_torch_safe() -> bytes:
    """Legitimate PyTorch globals used in normal model checkpoints."""
    return (
        b"\x80\x02"
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        b"ctorch\nTensor\n"
        b"."
    )


def make_multi_payload() -> bytes:
    """Two pickle payloads concatenated — scanner should report both."""
    return make_clean() + make_os_system()


PAYLOADS = {
    "clean.pkl":         make_clean,
    "os_system.pkl":     make_os_system,
    "subprocess.pkl":    make_subprocess,
    "eval_exec.pkl":     make_eval_exec,
    "build_attack.pkl":  make_build_attack,
    "newobj.pkl":        make_newobj,
    "persid.pkl":        make_persid,
    "torch_safe.pkl":    make_torch_safe,
    "multi_payload.pkl": make_multi_payload,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in PAYLOADS.items():
        path = OUT_DIR / filename
        data = builder()
        path.write_bytes(data)
        print(f"  wrote {path}  ({len(data)} bytes)")
    print(f"\n{len(PAYLOADS)} fixtures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
