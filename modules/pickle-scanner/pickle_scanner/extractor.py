"""
Detects the container format of a file and extracts raw pickle payloads
without importing or executing any user-controlled code.

Supported containers:
    - Raw pickle (.pkl, .pickle)
    - PyTorch checkpoint (.pt, .pth)  — ZIP archive containing data.pkl
    - NumPy array (.npy, .npz)        — magic-byte detection
    - Joblib dump (.joblib)           — pickle-wrapped
    - Generic ZIP archive             — scans all .pkl members
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Payload:
    source:  str    # human-readable description of where this payload came from
    data:    bytes
    #: True when nothing about the file said "pickle" except its first bytes.
    #: A payload found this way that then fails to parse is a heuristic that
    #: misfired, not a malformed pickle, and callers must be able to tell the
    #: two apart — one is noise, the other is a file refusing to be read.
    heuristic: bool = False


# Pickle protocol 2+ starts with \x80\x02 or higher; protocol 0/1 starts
# with a printable opcode byte.  We accept any byte that is a valid first
# opcode.
_PICKLE_FIRST_BYTES = frozenset(
    b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    b"\x90\x91\x92\x93\x94\x95\x96\x97\x98"    # proto 2–5 opcodes
    b"IFLNPRSTUVG(].abcdefghijklmnopqstuz"      # proto 0/1 printable opcodes
)
_NUMPY_MAGIC = b"\x93NUMPY"
_ZIP_MAGIC   = b"PK\x03\x04"


def _looks_like_pickle(data: bytes) -> bool:
    return len(data) > 0 and data[0:1] in {bytes([b]) for b in _PICKLE_FIRST_BYTES}


def _looks_like_pickle_loose(data: bytes) -> bool:
    """More lenient check — a PROTO opcode near the start of the file.

    PROTO is `0x80` followed by the protocol version, and pickle has only ever
    had versions 0-5. Both bytes are checked, because a lone `0x80` is not a
    signal: a PyTorch checkpoint stores its tensors as raw little-endian floats
    in `data/<n>`, and one in every ~64 of those blobs begins with a `0x80`
    inside the first four bytes by chance. Matching on that alone hands the
    parser a wall of random bytes, which fails and is then reported as an
    unparseable payload — a scanner that calls one in fifty genuine checkpoints
    suspicious is a scanner that gets switched off.
    """
    window = data[:5]
    return any(window[i] == 0x80 and i + 1 < len(window) and window[i + 1] <= 5
               for i in range(min(4, len(window))))


def extract_payloads(path: str) -> list[Payload]:
    """
    Read a file and return a list of Payload objects containing raw pickle
    bytes to be analysed.  Never imports, executes or unpickles anything.
    """
    p    = Path(path)
    data = p.read_bytes()

    # ── PyTorch / generic ZIP ─────────────────────────────────────────────
    if data[:4] == _ZIP_MAGIC:
        return _extract_from_zip(path, data)

    # ── NumPy .npy (single array) ─────────────────────────────────────────
    if data[:6] == _NUMPY_MAGIC:
        return [Payload(source=f"{path} (numpy .npy — no pickle payload)", data=b"")]

    # ── Raw pickle ────────────────────────────────────────────────────────
    return [Payload(source=path, data=data)]


def _extract_from_zip(path: str, data: bytes) -> list[Payload]:
    payloads: list[Payload] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                lower = member.lower()
                # PyTorch checkpoints store the tensor data in data.pkl
                if lower.endswith(".pkl") or lower.endswith(".pickle"):
                    raw = zf.read(member)
                    payloads.append(Payload(
                        source=f"{path}::{member}",
                        data=raw,
                    ))
                # Also check for any member that starts with pickle bytes
                elif not lower.endswith((".jpg", ".png", ".txt", ".json")):
                    try:
                        raw = zf.read(member)
                        if _looks_like_pickle_loose(raw):
                            payloads.append(Payload(
                                source=f"{path}::{member} (heuristic pickle detection)",
                                data=raw,
                                heuristic=True,
                            ))
                    except Exception:
                        pass
    except zipfile.BadZipFile as exc:
        # Not a valid ZIP despite the magic bytes — treat as raw pickle
        payloads.append(Payload(source=f"{path} (bad ZIP: {exc})", data=data))
    return payloads or [Payload(source=path, data=data)]
