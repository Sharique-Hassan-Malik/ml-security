"""
Static pickle opcode parser.

Reads raw bytes and decodes each opcode and its argument without executing
anything.  The parser never constructs Python objects, calls __reduce__,
imports modules or invokes any callable.  It is safe to run on untrusted
payloads.

Supported protocols: 0, 1, 2, 3, 4, 5.
Reference: cpython/Lib/pickle.py and cpython/Lib/pickletools.py.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Iterator


# ---------------------------------------------------------------------------
# Decoded instruction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    offset: int
    opcode: str
    arg:    object          # decoded argument or None


# ---------------------------------------------------------------------------
# Opcode byte → (name, arg_reader) table
# Built from first principles so we have no dependency on pickletools.
# ---------------------------------------------------------------------------

def _read_none(buf: io.RawIOBase) -> None:
    return None

def _read_uint1(buf: io.RawIOBase) -> int:
    return struct.unpack(">B", buf.read(1))[0]

def _read_uint2_le(buf: io.RawIOBase) -> int:
    return struct.unpack("<H", buf.read(2))[0]

def _read_int4_le(buf: io.RawIOBase) -> int:
    return struct.unpack("<i", buf.read(4))[0]

def _read_uint4_le(buf: io.RawIOBase) -> int:
    return struct.unpack("<I", buf.read(4))[0]

def _read_uint8_le(buf: io.RawIOBase) -> int:
    return struct.unpack("<Q", buf.read(8))[0]

def _read_float64_be(buf: io.RawIOBase) -> float:
    return struct.unpack(">d", buf.read(8))[0]

def _read_float64_le(buf: io.RawIOBase) -> float:
    return struct.unpack("<d", buf.read(8))[0]

def _read_newline(buf: io.RawIOBase) -> str:
    """Read until newline (protocol 0 text opcodes)."""
    out = bytearray()
    while True:
        ch = buf.read(1)
        if not ch or ch == b"\n":
            break
        out += ch
    return out.decode("utf-8", errors="replace")

def _read_counted(n_bytes: int):
    """Return a reader that reads a length prefix of n_bytes then that many bytes."""
    def _reader(buf: io.RawIOBase) -> bytes:
        raw = buf.read(n_bytes)
        if len(raw) < n_bytes:
            raise EOFError("Truncated length prefix")
        if n_bytes == 1:
            length = raw[0]
        elif n_bytes == 2:
            length = struct.unpack("<H", raw)[0]
        elif n_bytes == 4:
            length = struct.unpack("<I", raw)[0]
        else:
            length = struct.unpack("<Q", raw)[0]
        data = buf.read(length)
        return data
    return _reader

def _read_long_newline(buf: io.RawIOBase) -> int:
    """Read a decimal integer terminated by 'L\n'."""
    raw = _read_newline(buf)
    return int(raw.rstrip("L") or "0")

def _read_long1(buf: io.RawIOBase) -> int:
    n = buf.read(1)[0]
    raw = buf.read(n)
    return int.from_bytes(raw, "little", signed=True) if n else 0

def _read_long4(buf: io.RawIOBase) -> int:
    n = struct.unpack("<I", buf.read(4))[0]
    raw = buf.read(n)
    return int.from_bytes(raw, "little", signed=True) if n else 0


# Opcode byte → (mnemonic, arg_reader)
_OPCODE_TABLE: dict[int, tuple[str, object]] = {
    # Protocol 0
    0x28: ("MARK",             _read_none),
    0x2e: ("STOP",             _read_none),
    0x30: ("POP",              _read_none),
    0x31: ("POP_MARK",         _read_none),
    0x32: ("DUP",              _read_none),
    0x46: ("FLOAT",            _read_newline),
    0x49: ("INT",              _read_newline),
    0x4a: ("LONG",             _read_long_newline),
    0x4c: ("LIST",             _read_none),
    0x4e: ("NONE",             _read_none),
    0x50: ("PERSID",           _read_newline),
    0x51: ("BINPERSID",        _read_none),
    0x52: ("REDUCE",           _read_none),
    0x53: ("STRING",           _read_newline),
    0x54: ("UNICODE",          _read_newline),
    0x56: ("UNICODE",          _read_newline),     # alias
    0x58: ("SHORT_BINUNICODE", _read_counted(4)),
    0x59: ("SHORT_BINSTRING",  _read_counted(1)),
    0x5d: ("EMPTY_LIST",       _read_none),
    0x61: ("APPEND",          _read_none),
    0x62: ("BUILD",            _read_none),
    0x63: ("GLOBAL",           _read_newline),     # two newline-terms: module\nname\n
    0x64: ("DICT",             _read_none),
    0x65: ("APPENDS",       _read_none),
    0x67: ("GET",              _read_newline),
    0x68: ("BINGET",           _read_uint1),
    0x69: ("INST",             _read_newline),
    0x6a: ("LONG_BINGET",      _read_uint4_le),
    0x6b: ("MARK",             _read_none),
    0x6c: ("LIST",             _read_none),
    0x6e: ("NONE",             _read_none),
    0x6f: ("OBJ",              _read_none),
    0x70: ("PUT",              _read_newline),
    0x71: ("BINPUT",           _read_uint1),
    0x72: ("LONG_BINPUT",      _read_uint4_le),
    0x73: ("SETITEMS",         _read_none),
    0x74: ("TUPLE",            _read_none),
    0x75: ("SETITEM",          _read_none),
    0x7d: ("EMPTY_DICT",       _read_none),
    0x7e: ("FROZENSET",        _read_none),
    0x29: ("EMPTY_TUPLE",      _read_none),
    # Protocol 1
    0x4b: ("BININT1",          _read_uint1),
    0x4d: ("BININT2",          _read_uint2_le),
    0x4f: ("NEWOBJ",           _read_none),
    0x47: ("BINFLOAT",         _read_float64_be),
    0x43: ("SHORT_BINSTRING",  _read_counted(1)),
    0x55: ("SHORT_BINUNICODE", _read_counted(1)),
    0x78: ("BININT",           _read_int4_le),     # 'x'
    0x4b: ("BININT1",          _read_uint1),       # 'K'
    0x4d: ("BININT2",          _read_uint2_le),    # 'M'
    0x4a: ("BININT",           _read_int4_le),     # 'J' override
    0x54: ("BINSTRING",        _read_counted(4)),  # 'T'
    0x55: ("SHORT_BINSTRING",  _read_counted(1)),  # 'U'
    0x58: ("BINUNICODE",       _read_counted(4)),  # 'X'
    0x6d: ("BINUNICODE8",      _read_counted(8)),  # 'm'
    0x5a: ("SHORT_BINUNICODE", _read_counted(1)),  # 'Z'
    0x7a: ("LONG_BINUNICODE",  _read_counted(8)),  # 'z'  proto5
    # Protocol 2
    0x80: ("PROTO",            _read_uint1),
    0x81: ("NEWOBJ",           _read_none),
    0x82: ("EXT1",             _read_uint1),
    0x83: ("EXT2",             _read_uint2_le),
    0x84: ("EXT4",             _read_uint4_le),
    0x85: ("TUPLE1",           _read_none),
    0x86: ("TUPLE2",           _read_none),
    0x87: ("TUPLE3",           _read_none),
    0x88: ("NEWTRUE",          _read_none),
    0x89: ("NEWFALSE",         _read_none),
    0x8a: ("LONG1",            _read_long1),
    0x8b: ("LONG4",            _read_long4),
    # Protocol 4
    0x8c: ("SHORT_BINUNICODE", _read_counted(1)),
    0x8d: ("BINUNICODE8",      _read_counted(8)),
    0x8e: ("BINBYTES8",        _read_counted(8)),
    0x8f: ("EMPTY_SET",        _read_none),
    0x90: ("ADDITEMS",         _read_none),
    0x91: ("FROZENSET",        _read_none),
    0x92: ("NEWOBJ_EX",        _read_none),
    0x93: ("STACK_GLOBAL",     _read_none),
    0x94: ("MEMOIZE",          _read_none),
    0x95: ("FRAME",            _read_uint8_le),
    # Protocol 5
    0x96: ("BYTEARRAY8",       _read_counted(8)),
    0x97: ("NEXT_BUFFER",      _read_none),
    0x98: ("READONLY_BUFFER",  _read_none),
    # Misc
    0x41: ("APPEND",           _read_none),        # 'A'
    0x42: ("BINBYTES",         _read_counted(4)),  # 'B'
    0x43: ("SHORT_BINBYTES",   _read_counted(1)),  # 'C'
    0x4e: ("NONE",             _read_none),        # 'N' override
}

# The GLOBAL opcode has two newline-terminated strings: module then name.
# We handle it specially in the parser below.
_GLOBAL_LIKE = {"GLOBAL", "INST"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PickleParser:
    """
    Stateless static parser.  Call parse() to obtain an instruction stream.
    No Python objects are ever constructed during parsing.
    """

    def parse(self, data: bytes) -> Iterator[Instruction]:
        """
        Yield Instruction objects for every opcode in `data`.

        Raises ParseError on malformed input.
        """
        buf    = io.BytesIO(data)
        offset = 0

        while True:
            offset = buf.tell()
            byte   = buf.read(1)
            if not byte:
                return

            code = byte[0]
            if code not in _OPCODE_TABLE:
                raise ParseError(
                    f"Unknown opcode 0x{code:02x} at offset 0x{offset:04x}"
                )

            mnemonic, reader = _OPCODE_TABLE[code]

            # GLOBAL reads two newline-terminated strings (module then name)
            if mnemonic == "GLOBAL":
                module = _read_newline(buf)
                name   = _read_newline(buf)
                arg    = (module, name)
            elif mnemonic == "INST":
                module = _read_newline(buf)
                name   = _read_newline(buf)
                arg    = (module, name)
            else:
                try:
                    arg = reader(buf)
                except (struct.error, EOFError) as exc:
                    raise ParseError(
                        f"Failed to read argument for {mnemonic} "
                        f"at offset 0x{offset:04x}: {exc}"
                    ) from exc

            yield Instruction(offset=offset, opcode=mnemonic, arg=arg)

            if mnemonic == "STOP":
                # Multiple pickles can be concatenated; continue parsing
                # rather than stopping so we catch payloads after STOP.
                pass


class ParseError(ValueError):
    pass
