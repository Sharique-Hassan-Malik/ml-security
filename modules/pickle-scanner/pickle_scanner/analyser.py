"""
Analyses a parsed instruction stream and emits Findings without executing
any pickle opcode.

The analyser maintains a lightweight symbolic stack to track what GLOBAL
resolved to, so it can correctly attribute a REDUCE finding to the specific
callable being invoked.
"""

from __future__ import annotations

from pickle_scanner.opcodes import (
    DANGEROUS_OPCODES, KNOWN_SAFE_GLOBALS, DANGEROUS_MODULES,
    ModuleResult, Severity, opcode_finding,
)
from pickle_scanner.parser import Instruction


class Analyser:
    """
    Stateful single-pass analyser over an instruction stream.

    The analyser tracks:
        - The last GLOBAL/STACK_GLOBAL pair seen (module, name)
        - A simplified symbolic value stack for STACK_GLOBAL resolution

    It emits a Finding for every opcode that warrants attention.
    """

    def __init__(self, result: ModuleResult, strict: bool = False):
        self._result = result
        self._strict = strict
        # Counted here and copied into result.metrics by the caller: an
        # opcode tally is a property of the stream, not a finding.
        self.n_opcodes = 0
        self.proto = 0
        # Symbolic stack stores string tokens or None for unknown values
        self._stack: list[object] = []
        # Last resolved global, used to annotate REDUCE
        self._last_global: tuple[str, str] | None = None

    def feed(self, instr: Instruction):
        self.n_opcodes += 1
        name = instr.opcode

        # ── Protocol version ──────────────────────────────────────────────
        if name == "PROTO":
            self.proto = int(instr.arg or 0)
            self._emit_info(instr, f"Protocol {instr.arg}")
            return

        # ── String/bytes pushes onto symbolic stack ───────────────────────
        if name in (
            "STRING", "BINSTRING", "SHORT_BINSTRING",
            "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
        ):
            val = instr.arg
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8")
                except UnicodeDecodeError:
                    val = None
            self._stack.append(val)
            return

        if name in ("NONE", "NEWTRUE", "NEWFALSE"):
            self._stack.append(None)
            return

        if name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"):
            self._stack.append(instr.arg)
            return

        # ── GLOBAL ────────────────────────────────────────────────────────
        if name == "GLOBAL":
            module, attr = instr.arg
            self._last_global = (module, attr)
            self._stack.append(f"{module}.{attr}")
            self._check_global(instr, module, attr)
            return

        # ── STACK_GLOBAL ──────────────────────────────────────────────────
        if name == "STACK_GLOBAL":
            # Pops two strings from the stack: name then module (reversed)
            attr   = self._pop_str()
            module = self._pop_str()
            if module and attr:
                self._last_global = (module, attr)
                self._stack.append(f"{module}.{attr}")
                self._check_global(instr, module, attr)
            else:
                self._last_global = None
                self._stack.append(None)
                sev, desc, _ = DANGEROUS_OPCODES["STACK_GLOBAL"]
                self._result.add(opcode_finding(
                    opcode="STACK_GLOBAL",
                    offset=instr.offset,
                    severity=sev,
                    description=desc,
                    detail="Module or attribute could not be statically resolved",
                ))
            return

        # ── INST ──────────────────────────────────────────────────────────
        if name == "INST":
            module, attr = instr.arg
            self._last_global = (module, attr)
            self._check_global(instr, module, attr, opcode_override="INST")
            return

        # ── REDUCE ────────────────────────────────────────────────────────
        if name == "REDUCE":
            sev, desc, detail = DANGEROUS_OPCODES["REDUCE"]
            callee = self._peek_callable()
            if callee:
                detail = f"Invokes {callee}"
                # If the callee is a known-safe global, downgrade to INFO
                parts = callee.split(".")
                if len(parts) == 2:
                    pair = tuple(parts)
                    if pair in KNOWN_SAFE_GLOBALS:
                        sev = Severity.INFO
            self._result.add(opcode_finding(
                opcode="REDUCE", offset=instr.offset,
                severity=sev, description=desc, detail=detail,
            ))
            # REDUCE consumes callable + args, pushes result
            self._stack = []
            self._last_global = None
            return

        # ── BUILD ─────────────────────────────────────────────────────────
        if name == "BUILD":
            sev, desc, detail = DANGEROUS_OPCODES["BUILD"]
            self._result.add(opcode_finding(
                opcode="BUILD", offset=instr.offset,
                severity=sev, description=desc, detail=detail,
            ))
            return

        # ── NEWOBJ / NEWOBJ_EX ────────────────────────────────────────────
        if name in ("NEWOBJ", "NEWOBJ_EX"):
            sev, desc, detail = DANGEROUS_OPCODES[name]
            callee = self._peek_callable()
            if callee:
                parts = callee.split(".")
                if len(parts) == 2 and tuple(parts) in KNOWN_SAFE_GLOBALS:
                    sev = Severity.INFO
                detail = f"Constructs {callee}"
            self._result.add(opcode_finding(
                opcode=name, offset=instr.offset,
                severity=sev, description=desc, detail=detail,
            ))
            self._stack = []
            return

        # ── OBJ ───────────────────────────────────────────────────────────
        if name == "OBJ":
            sev, desc, detail = DANGEROUS_OPCODES["OBJ"]
            self._result.add(opcode_finding(
                opcode="OBJ", offset=instr.offset,
                severity=sev, description=desc, detail=detail,
            ))
            return

        # ── Persistent ID ─────────────────────────────────────────────────
        if name in ("PERSID", "BINPERSID"):
            sev, desc, detail = DANGEROUS_OPCODES[name]
            self._result.add(opcode_finding(
                opcode=name, offset=instr.offset,
                severity=sev, description=desc, detail=detail,
            ))
            return

        # ── Stack management (symbolic) ───────────────────────────────────
        if name == "POP":
            if self._stack:
                self._stack.pop()
        elif name in ("MARK", "POP_MARK"):
            self._stack.append(None)
        elif name == "DUP":
            if self._stack:
                self._stack.append(self._stack[-1])
        elif name == "MEMOIZE":
            pass  # memo write, stack unchanged

    # ── Helpers ──────────────────────────────────────────────────────────

    def _check_global(
        self,
        instr: Instruction,
        module: str,
        attr: str,
        opcode_override: str | None = None,
    ):
        opcode_name = opcode_override or "GLOBAL"
        base_module = module.split(".")[0]

        pair = (module, attr)
        sev, desc, template = DANGEROUS_OPCODES[opcode_name if opcode_name in DANGEROUS_OPCODES else "GLOBAL"]

        if pair in KNOWN_SAFE_GLOBALS:
            sev = Severity.INFO
            detail = f"{module}.{attr} — known safe ML serialisation global"
        elif base_module in DANGEROUS_MODULES:
            sev = Severity.CRITICAL
            detail = (
                f"Imports from high-risk module '{module}'. "
                f"Attribute '{attr}' may allow command execution or file access."
            )
        elif self._strict and module.startswith("_"):
            sev = max(sev, Severity.HIGH)
            detail = f"{module}.{attr} — private C-extension module"
        else:
            detail = f"{module}.{attr}"

        self._result.add(opcode_finding(
            opcode=opcode_name, offset=instr.offset,
            severity=sev, description=desc, detail=detail,
        ))

    def _emit_info(self, instr: Instruction, detail: str = ""):
        pass   # Protocol/framing info suppressed unless verbose

    def _pop_str(self) -> str | None:
        if not self._stack:
            return None
        val = self._stack.pop()
        return val if isinstance(val, str) else None

    def _peek_callable(self) -> str | None:
        """Return the most recent string on the stack that looks like a dotted name."""
        for item in reversed(self._stack):
            if isinstance(item, str) and "." in item:
                return item
        return None
