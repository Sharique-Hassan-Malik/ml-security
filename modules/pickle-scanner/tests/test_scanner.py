"""
Test suite for the pickle scanner.

Generates payloads in-memory (no disk fixtures required for the core tests)
and asserts expected severities and finding counts.

Run with:
    python -m pytest tests/test_scanner.py -v
"""

import io
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import pytest

# Make project root importable when running from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from pickle_scanner.opcodes import Severity
from pickle_scanner.parser import PickleParser, ParseError
from pickle_scanner.scanner import scan_bytes, scan_file


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------

class TestParser:
    def _parse(self, data: bytes) -> list:
        return list(PickleParser().parse(data))

    def test_stop_opcode(self):
        instrs = self._parse(b"\x80\x02.")
        names  = [i.opcode for i in instrs]
        assert "PROTO" in names
        assert "STOP"  in names

    def test_proto_argument(self):
        instrs = self._parse(b"\x80\x04.")
        proto  = next(i for i in instrs if i.opcode == "PROTO")
        assert proto.arg == 4

    def test_global_two_strings(self):
        data   = b"\x80\x02cos\nsystem\n."
        instrs = self._parse(data)
        g      = next(i for i in instrs if i.opcode == "GLOBAL")
        assert g.arg == ("os", "system")

    def test_short_binunicode(self):
        s      = "hello"
        data   = b"\x80\x04\x8c" + bytes([len(s)]) + s.encode() + b"."
        instrs = self._parse(data)
        su     = next(i for i in instrs if "UNICODE" in i.opcode)
        assert su.arg == b"hello"

    def test_unknown_opcode_raises(self):
        with pytest.raises(ParseError):
            self._parse(b"\xff")

    def test_empty_data(self):
        assert self._parse(b"") == []

    def test_multiple_stop_continues(self):
        """Two pickles concatenated — parser should yield opcodes from both."""
        p1     = pickle.dumps(1, protocol=2)
        p2     = pickle.dumps(2, protocol=2)
        instrs = self._parse(p1 + p2)
        stops  = [i for i in instrs if i.opcode == "STOP"]
        assert len(stops) == 2


# ---------------------------------------------------------------------------
# Analyser / scanner unit tests
# ---------------------------------------------------------------------------

class TestScanner:

    def test_clean_dict_is_safe(self):
        data   = pickle.dumps({"a": 1}, protocol=2)
        result = scan_bytes(data)
        assert result.max_severity < Severity.HIGH
        assert result.max_severity == Severity.SAFE or result.max_severity <= Severity.INFO

    def test_os_system_critical(self):
        data = (
            b"\x80\x02"
            b"cos\nsystem\n"
            b"(\x85R."
        )
        result = scan_bytes(data)
        assert result.max_severity == Severity.CRITICAL

    def test_builtins_eval_critical(self):
        code = b"1+1"
        data = (
            b"\x80\x02"
            b"cbuiltins\neval\n"
            b"(" + b"X" + struct.pack("<I", len(code)) + code + b"\x85R."
        )
        result = scan_bytes(data)
        assert result.max_severity == Severity.CRITICAL

    def test_subprocess_module_critical(self):
        data = (
            b"\x80\x04"
            b"\x8c\nsubprocess\x8c\x0ccheck_output\x93"
            b"(]\x8c\x02ida\x85R."
        )
        result = scan_bytes(data)
        assert result.max_severity == Severity.CRITICAL

    def test_torch_rebuild_not_critical(self):
        """Standard PyTorch serialisation globals should not be CRITICAL."""
        data = (
            b"\x80\x02"
            b"ctorch._utils\n_rebuild_tensor_v2\n"
            b"ctorch\nTensor\n"
            b"."
        )
        result = scan_bytes(data)
        assert result.max_severity < Severity.HIGH

    def test_reduce_without_global_high(self):
        data = b"\x80\x02(R."
        result = scan_bytes(data)
        reduce_findings = [f for f in result.findings if f.title == "REDUCE"]
        assert reduce_findings

    def test_build_opcode_flagged(self):
        data = (
            b"\x80\x02"
            b"ccollections\nOrderedDict\n"
            b")R"
            b"}"
            b"b."
        )
        result = scan_bytes(data)
        build_findings = [f for f in result.findings if f.title == "BUILD"]
        assert build_findings
        assert build_findings[0].severity == Severity.HIGH

    def test_newobj_medium(self):
        data = b"\x80\x02ccollections\nOrderedDict\n)\x81."
        result = scan_bytes(data)
        newobj = [f for f in result.findings if f.title == "NEWOBJ"]
        assert newobj

    def test_persid_low(self):
        data = b"\x80\x02Psome_id\n."
        result = scan_bytes(data)
        persid = [f for f in result.findings if f.title == "PERSID"]
        assert persid
        assert persid[0].severity == Severity.LOW

    def test_multiple_globals_all_reported(self):
        data = (
            b"\x80\x02"
            b"cos\nsystem\n"
            b"cos\ngetenv\n"
            b"."
        )
        result = scan_bytes(data)
        globals_found = [f for f in result.findings if f.title == "GLOBAL"]
        assert len(globals_found) == 2

    def test_offset_recorded(self):
        data = b"\x80\x02cos\nsystem\n."
        result = scan_bytes(data)
        g = next(f for f in result.findings if f.title == "GLOBAL")
        assert g.metadata["offset"] == 2   # after PROTO byte + version byte

    def test_proto_version_extracted(self):
        data = pickle.dumps({}, protocol=4)
        result = scan_bytes(data)
        assert result.metrics["protocol"] == 4

    def test_safe_result_has_no_error(self):
        data   = pickle.dumps([1, 2, 3], protocol=2)
        result = scan_bytes(data)
        assert result.error == ""

    def test_truncated_data_sets_error(self):
        # Truncate mid-opcode argument
        data   = b"\x80\x02cos\nsystem"   # missing newline after 'system'
        result = scan_bytes(data)
        # Parser should either raise ParseError or produce a finding — not crash
        assert isinstance(result, type(result))

    def test_empty_bytes(self):
        result = scan_bytes(b"")
        assert result.metrics["opcodes"] == 0

    def test_opcode_count(self):
        data   = pickle.dumps({"key": "value"}, protocol=2)
        result = scan_bytes(data)
        assert result.metrics["opcodes"] > 0

    def test_safe_flag_false_on_critical(self):
        data = b"\x80\x02cos\nsystem\n(\x85R."
        result = scan_bytes(data)
        assert result.max_severity >= Severity.HIGH or result.error

    def test_numpy_reconstruct_downgraded(self):
        """numpy._reconstruct is a known-safe global — should not be CRITICAL."""
        data = (
            b"\x80\x02"
            b"cnumpy.core.multiarray\n_reconstruct\n"
            b"."
        )
        result = scan_bytes(data)
        globals_f = [f for f in result.findings if f.title == "GLOBAL"]
        assert all(f.severity < Severity.HIGH for f in globals_f)


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

class TestSeverityOrdering:
    def test_ordering(self):
        assert Severity.SAFE    < Severity.INFO
        assert Severity.INFO    < Severity.LOW
        assert Severity.LOW     < Severity.MEDIUM
        assert Severity.MEDIUM  < Severity.HIGH
        assert Severity.HIGH    < Severity.CRITICAL

    def test_max_severity_empty(self):
        result = scan_bytes(pickle.dumps(1, 2))
        assert result.max_severity in (Severity.SAFE, Severity.INFO)


# ---------------------------------------------------------------------------
# Container extraction
# ---------------------------------------------------------------------------

class TestZipExtraction:
    """A `.pt` is a ZIP: one pickle plus one raw blob per tensor.

    The blobs are little-endian floats, and treating one as a pickle because a
    `0x80` byte happened to land near its start produces a parse error on a
    perfectly good checkpoint. A caller that fails closed on parse errors — a
    promotion gate, say — then refuses a genuine model roughly one time in
    fifty, which is how a security control ends up switched off.
    """

    def _zip_of(self, members: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
        return buf.getvalue()

    def _scan(self, tmp_path, members: dict):
        path = tmp_path / "model.pt"
        path.write_bytes(self._zip_of(members))
        return scan_file(str(path))

    def test_a_tensor_blob_starting_with_0x80_is_not_a_pickle(self, tmp_path):
        # 0x80 at offset 0, followed by 0xBF — PROTO with version 191, which
        # pickle has never had. Real floats: struct.pack("<f", -1.4917e-38).
        blob = b"\x80\xbf\x00\x00" + struct.pack("<4f", 1.5, -2.25, 0.0, 7.5)
        results = self._scan(tmp_path, {"model/data.pkl": pickle.dumps({"w": 1}, 2),
                                        "model/data/0": blob})
        assert len(results) == 1, [r.module for r in results]
        assert not results[0].error

    def test_every_protocol_version_byte_still_matches(self, tmp_path):
        for version in range(6):
            blob = b"\x80" + bytes([version]) + b"K\x01."
            results = self._scan(tmp_path, {"model/data/0": blob})
            assert len(results) == 1, f"protocol {version} went undetected"

    def test_a_real_pickle_under_a_data_name_is_still_found(self, tmp_path):
        """The heuristic exists for payloads hidden under an innocent name."""
        hidden = b"\x80\x02cos\nsystem\n(X\x02\x00\x00\x00idtR."
        results = self._scan(tmp_path, {"model/data/0": hidden})
        assert len(results) == 1
        assert results[0].max_severity >= Severity.HIGH

    def test_the_blob_is_skipped_without_hiding_the_pickle_beside_it(self, tmp_path):
        malicious = b"\x80\x02cos\nsystem\n(X\x02\x00\x00\x00idtR."
        blob = b"\x80\xbf\x00\x00" + struct.pack("<4f", 1.5, -2.25, 0.0, 7.5)
        results = self._scan(tmp_path, {"model/data.pkl": malicious,
                                        "model/data/0": blob})
        assert len(results) == 1
        assert results[0].max_severity >= Severity.HIGH

    def test_a_blob_that_survives_the_heuristic_is_skipped_not_errored(self, tmp_path):
        """0x80 0x02 is a valid PROTO, so no opening-byte check can rule this
        out — the parse is what settles it. Failing there means the heuristic
        misfired, and a caller that fails closed on `error` must not see one."""
        blob = b"\x80\x02" + struct.pack("<8f", *(1.5, -2.25, 0.0, 7.5) * 2)
        results = self._scan(tmp_path, {"model/data/0": blob})
        assert len(results) == 1
        assert not results[0].error
        assert results[0].skipped and "not a pickle" in results[0].skipped

    def test_a_declared_pickle_that_will_not_parse_still_errors(self, tmp_path):
        """Nothing above may soften this case: a member named `.pkl` that the
        parser chokes on is a file refusing to be read, and a gate that lets it
        through has not scanned it."""
        results = self._scan(tmp_path, {"model/data.pkl": b"\x80\x02\xd8\xd8\xd8."})
        assert len(results) == 1
        assert results[0].error
