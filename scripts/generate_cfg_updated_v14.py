"""
Compact textual CFG generation for compiler-generated assembly and objdump output.

Validated input families:
  * compiler-generated .s
  * objdump of relocatable objects (.o)
  * objdump of shared libraries (.so/.dylib)
  * objdump of fully linked programs

Validated ISA families:
  * x86-64 Linux
  * AArch64 Linux
  * AArch64 macOS
  * RISC-V Linux

The CFG is intraprocedural: calls stay inside the current basic block and are
not followed into the callee. Calls are conservatively treated as potentially
returning, even for known noreturn library/runtime functions, because call-target
symbol information is not uniformly available across compiler assembly,
relocatable-object disassembly, shared-library disassembly, and linked-program
disassembly. Direct branches get explicit edges. Indirect branches that cannot
be resolved statically get an UNKNOWN edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


class BasicBlock:
    """A straight-line sequence of instructions with CFG successor edges."""

    def __init__(self, key):
        self.key = key
        self.instructions = []
        self.jump_edge = None
        self.no_jump_edge = None

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def add_jump_edge(self, target_key):
        self.jump_edge = target_key.key if isinstance(target_key, BasicBlock) else target_key

    def add_no_jump_edge(self, target_key):
        self.no_jump_edge = target_key.key if isinstance(target_key, BasicBlock) else target_key


class Instruction:
    def __init__(self, text, pc, opcode, ops, target_info, *, real_address=False):
        self.text = text
        self.pc = pc
        self.opcode = opcode
        self.ops = ops
        self.target_pc = None
        # Optional symbolic target known to leave the current function. This
        # is especially useful for relocatable-object tail branches whose
        # encoded displacement is only a linker placeholder.
        self.external_target = None
        self.info = target_info
        self.real_address = real_address

    def is_call(self):
        return self.info.is_call(self.opcode, self.ops)

    def is_jump(self):
        return self.info.is_jump(self.opcode, self.ops)

    def is_sink(self):
        return self.info.is_sink(self.opcode, self.ops)

    def is_unconditional_jump(self):
        return self.info.is_unconditional_jump(self.opcode, self.ops)

    def is_indirect_jump(self):
        return self.info.is_indirect_jump(self.opcode, self.ops)


class X86TargetInfo:
    name = "x86"
    platform = "linux"
    target = "x86_linux"

    def comment_chars(self):
        return ["#", "//"]

    def is_call(self, opcode, ops=None):
        return opcode.startswith("call")

    def is_jump(self, opcode, ops=None):
        # jcc/jmp plus the loop-family conditional transfers.
        return opcode.startswith("j") or opcode in {
            "loop", "loope", "loopz", "loopne", "loopnz"
        }

    def is_unconditional_jump(self, opcode, ops=None):
        return opcode.startswith("jmp")

    def is_indirect_jump(self, opcode, ops=None):
        if not opcode.startswith("jmp") or not ops:
            return False
        operand = ops[0].strip()
        return operand.startswith("*") or operand.startswith("%") or operand.startswith("(")

    def is_sink(self, opcode, ops=None):
        return opcode.startswith("ret") or opcode in {"ud2", "hlt"}

    def target_op_index(self):
        return 0


class ARMTargetInfo:
    name = "arm"

    def __init__(self, platform="linux"):
        if platform not in {"linux", "macos"}:
            raise ValueError("ARM platform must be 'linux' or 'macos'")
        self.platform = platform
        self.target = "arm_linux" if platform == "linux" else "arm_mac"

    _COND = {
        "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl",
        "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
        "al", "nv",
    }

    def comment_chars(self):
        # '#' is an AArch64 immediate marker, not a comment marker.
        return ["//", ";"]

    def is_call(self, opcode, ops=None):
        return opcode in {"bl", "blr"}

    def is_jump(self, opcode, ops=None):
        # Support both LLVM spelling (b.eq) and GNU assembler spelling (beq).
        if opcode in {"b", "br", "cbz", "cbnz", "tbz", "tbnz"}:
            return True
        if opcode.startswith("b.") and opcode[2:] in self._COND:
            return True
        if opcode.startswith("b") and opcode[1:] in self._COND:
            return True
        return False

    def is_unconditional_jump(self, opcode, ops=None):
        return opcode in {"b", "br"}

    def is_indirect_jump(self, opcode, ops=None):
        return opcode == "br"

    def is_sink(self, opcode, ops=None):
        return opcode in {"ret", "eret", "brk", "hlt"}

    def target_op_index(self):
        return -1


class RISCVTargetInfo:
    name = "riscv"
    platform = "linux"
    target = "riscv_linux"

    _BRANCHES = {
        # Base conditional branches.
        "beq", "bne", "blt", "bge", "bltu", "bgeu",

        # GNU assembler branch pseudo-instructions.  These appear in GCC
        # compiler-generated .s, while objdump normally prints their canonical
        # base-instruction forms (for example, `ble a0,a1,L` becomes
        # `bge a1,a0,L`).  Treating the pseudo forms as branches is essential
        # for compiler-assembly CFGs to match CFGs recovered from the object,
        # shared-library, and linked-program disassemblies.
        "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
        "bgt", "ble", "bgtu", "bleu",

        # Compressed conditional branches.
        "c.beqz", "c.bnez",
    }

    @staticmethod
    def _first_operand(ops):
        if not ops:
            return ""
        return ops[0].strip().split()[0].lower()

    @staticmethod
    def _is_zero_reg(reg):
        return reg in {"zero", "x0"}

    @staticmethod
    def _is_ra_reg(reg):
        return reg in {"ra", "x1"}

    def comment_chars(self):
        return ["#", "//"]

    def is_call(self, opcode, ops=None):
        if opcode in {"call", "c.jal", "c.jalr"}:
            return True
        if opcode == "jal":
            # `jal target` implies rd=ra; `jal x0,target` is a jump.
            if ops and len(ops) >= 2:
                return not self._is_zero_reg(self._first_operand(ops))
            return True
        if opcode == "jalr":
            # GNU/objdump often prints a call as `jalr ra`.
            if ops:
                return not self._is_zero_reg(self._first_operand(ops))
            return True
        return False

    def is_jump(self, opcode, ops=None):
        return opcode in self._BRANCHES or self.is_unconditional_jump(opcode, ops)

    def is_unconditional_jump(self, opcode, ops=None):
        if opcode in {"j", "tail", "c.j"}:
            return True
        if opcode in {"jr", "c.jr"}:
            # `jr ra` is a return, not an ordinary jump.
            return not self._is_ra_reg(self._first_operand(ops))
        if opcode == "jal" and ops and len(ops) >= 2:
            return self._is_zero_reg(self._first_operand(ops))
        if opcode == "jalr" and ops:
            return self._is_zero_reg(self._first_operand(ops))
        return False

    def is_indirect_jump(self, opcode, ops=None):
        if opcode in {"jr", "c.jr"}:
            return not self._is_ra_reg(self._first_operand(ops))
        if opcode == "jalr" and ops:
            return self._is_zero_reg(self._first_operand(ops))
        return False

    def is_sink(self, opcode, ops=None):
        if opcode in {"ret", "ebreak", "ecall", "unimp", "c.ebreak"}:
            return True
        if opcode in {"jr", "c.jr"} and self._is_ra_reg(self._first_operand(ops)):
            return True
        # Canonical unaliased return: jalr x0, ra, 0
        if opcode == "jalr" and ops and len(ops) >= 2:
            return (
                self._is_zero_reg(self._first_operand(ops))
                and self._is_ra_reg(ops[1].strip().split()[0].lower())
            )
        return False

    def target_op_index(self):
        return -1


# Compiler-assembly labels. We intentionally do not classify numeric assembler
# labels like `0:` as objdump evidence.
LABEL_RE = re.compile(r"^\s*([.$A-Za-z_][.$A-Za-z0-9_]*)\s*:")
TYPE_FUNCTION_RE = re.compile(r"^\s*\.type\s+([^,\s]+)\s*,\s*[@%]function\b")
GLOBAL_RE = re.compile(r"^\s*\.(?:globl|global)\s+([^\s,]+)")
CFI_ENDPROC_RE = re.compile(r"^\s*\.cfi_endproc\b")

# Objdump forms in the supplied Linux ELF and Apple Mach-O examples.
OBJDUMP_FUNCTION_RE = re.compile(r"^\s*([0-9A-Fa-f]+)\s+<([^>]+)>:\s*$")
OBJDUMP_INSTRUCTION_RE = re.compile(r"^\s*([0-9A-Fa-f]+):\s*(.*?)\s*$")
OBJDUMP_SECTION_RE = re.compile(r"^\s*Disassembly of section\b")
OBJDUMP_FILE_FORMAT_RE = re.compile(r".*\bfile format\b", re.IGNORECASE)
RELOCATION_TOKEN_RE = re.compile(r"^(?:R_[A-Za-z0-9_]+|[A-Za-z0-9_]*RELOC[A-Za-z0-9_]*)$")
OBJDUMP_RELOCATION_RE = re.compile(
    r"^\s*([0-9A-Fa-f]+):\s*"
    r"(R_[A-Za-z0-9_]+|[A-Za-z0-9_]*RELOC[A-Za-z0-9_]*)"
    r"(?:\s+(.+?))?\s*$"
)


def _strip_instruction_comments(line, target_info):
    code = line
    for comment_char in target_info.comment_chars():
        code = code.split(comment_char, 1)[0]
    return code.strip()


def _split_operands(ops_str):
    # CFG needs the final branch target. Commas inside x86 address syntax are
    # harmless because x86 branch targets are a single operand, and AArch64 /
    # RISC-V branch targets are the final operand.
    return [op.strip() for op in ops_str.split(",")] if ops_str else []


def _extract_numeric_target(target_text):
    """Extract a direct objdump target like `401180 <foo+0x10>` or `0xc0 <...>`."""
    if not target_text:
        return None

    text = target_text.strip()

    # Common indirect forms.
    if text.startswith("*") or text.startswith("[") or text.startswith("("):
        return None

    match = re.match(r"^(?:0x)?([0-9A-Fa-f]+)\b", text)
    if not match:
        return None

    return int(match.group(1), 16)


def _remove_objdump_encoding(rest, target_info):
    """Remove raw machine bytes/words while keeping mnemonic + operands."""
    tokens = rest.split()
    if not tokens:
        return ""

    if target_info.name == "x86":
        i = 0
        while i < len(tokens) and re.fullmatch(r"[0-9A-Fa-f]{2}", tokens[i]):
            i += 1
        if i:
            tokens = tokens[i:]
    else:
        # AArch64 and RISC-V objdump use one encoded word/halfword before the
        # mnemonic in the supplied datasets.
        if re.fullmatch(r"[0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}|[0-9A-Fa-f]{16}", tokens[0]):
            tokens = tokens[1:]
        else:
            # Also tolerate byte-separated encodings.
            i = 0
            while i < len(tokens) and re.fullmatch(r"[0-9A-Fa-f]{2}", tokens[i]):
                i += 1
            if i:
                tokens = tokens[i:]

    return " ".join(tokens)


def _looks_like_relocation(code):
    if not code:
        return False
    first_token = code.split(None, 1)[0]
    return bool(RELOCATION_TOKEN_RE.fullmatch(first_token))


def _is_padding_noop(code, target_info):
    """Recognize no-ops materialized by binary alignment.

    Compiler `.s` usually represents the same alignment with directives such
    as `.p2align`, so keeping these no-op bytes as CFG nodes creates
    representation-only blocks with no useful semantics for Harbor.
    """
    text = code.strip().lower()

    if target_info.name == "x86":
        if re.match(r"^(?:(?:data16|cs)\s+)*(?:nop|nopl|nopw)\b", text):
            return True
        if re.fullmatch(r"xchg\s+%?(?:e?ax),\s*%?(?:e?ax)", text):
            return True
        return False

    if target_info.name == "arm":
        return text == "nop"

    if target_info.name == "riscv":
        return text in {"nop", "c.nop"}

    return False


def _relocation_symbol(reloc_text):
    """Extract the referenced symbol from an objdump relocation payload."""
    if not reloc_text:
        return None
    token = reloc_text.strip().split()[0]
    if not token or token in {"*ABS*", "*UND*"}:
        return None

    # x86 PC-relative relocations commonly render `symbol-0x4`.
    token = re.sub(r"[-+]0x[0-9A-Fa-f]+$", "", token)
    return token or None


def _relocation_is_external_control_transfer(reloc_type, symbol):
    """Return True when a relocation overrides a direct branch placeholder."""
    if not reloc_type or not symbol:
        return False
    if symbol.startswith(".L") or symbol.startswith("LBB"):
        return False

    upper = reloc_type.upper()

    if upper.startswith("R_X86_64_") and ("PLT32" in upper or "PC32" in upper):
        return True
    if upper.startswith("R_AARCH64_") and ("CALL26" in upper or "JUMP26" in upper):
        return True
    if "ARM64_RELOC_BRANCH26" in upper:
        return True

    # RISC-V CALL relocations normally sit on AUIPC in a multi-instruction
    # sequence, not on the final JR/JALR tail transfer, so the existing
    # conservative UNKNOWN handling remains appropriate there.
    return False


def _next_retained_address(address, retained_addresses):
    for candidate in retained_addresses:
        if candidate > address:
            return candidate
    return None


def _detect_input_format(lines):
    """Detect compiler `.s` versus objdump without confusing numeric labels with addresses."""
    if any(OBJDUMP_SECTION_RE.match(line) for line in lines):
        return "objdump"
    if any(OBJDUMP_FILE_FORMAT_RE.match(line) for line in lines):
        return "objdump"

    # Fallback for an objdump excerpt that omitted the banner but retained a
    # symbol header plus encoded instruction lines.
    has_function_header = any(OBJDUMP_FUNCTION_RE.match(line) for line in lines)
    has_encoded_instruction = False
    for line in lines:
        m = OBJDUMP_INSTRUCTION_RE.match(line)
        if not m:
            continue
        rest = m.group(2).strip()
        if re.match(r"^(?:[0-9A-Fa-f]{2}(?:\s+|$)|[0-9A-Fa-f]{4,16}(?:\s+|$))", rest):
            has_encoded_instruction = True
            break

    return "objdump" if (has_function_header and has_encoded_instruction) else "asm"


def _find_compiled_function_labels(lines):
    """Identify real function entry labels in compiler-generated assembly.

    ELF/GNU assembly normally marks every function with ``.type ..., @function``.
    Apple/Clang assembly does not use ``.type`` and only emits ``.globl`` for
    externally visible functions; file-local/static helpers therefore need a
    second signal.  Clang emits an ``Lfunc_beginN:`` marker immediately after
    each real function entry label, so recognize those labels as function
    entries too.

    This keeps the previous behavior for HumanEval/MC-Eval while allowing a
    multi-function Mach-O translation unit (such as Bringup-Bench) to retain
    static helpers like ``_phi`` and ``_phiphi``.
    """
    typed_functions = set()
    globals_ = set()
    labels_present = set()
    apple_function_labels = set()

    for raw_line in lines:
        m = TYPE_FUNCTION_RE.match(raw_line)
        if m:
            typed_functions.add(m.group(1))

        m = GLOBAL_RE.match(raw_line)
        if m:
            globals_.add(m.group(1))

        m = LABEL_RE.match(raw_line)
        if m:
            labels_present.add(m.group(1))

    # Apple/Clang: a real function label is followed (allowing blank/comment
    # lines) by an Lfunc_beginN label.  Do not treat LBB/Ltmp/data labels as
    # functions merely because they are non-local symbols.
    for i, raw_line in enumerate(lines):
        m = LABEL_RE.match(raw_line)
        if not m:
            continue
        name = m.group(1)
        if name.startswith(("L", ".")):
            continue

        for j in range(i + 1, min(i + 6, len(lines))):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith(";") or stripped.startswith("//"):
                continue
            next_label = LABEL_RE.match(lines[j])
            if next_label and re.fullmatch(r"Lfunc_begin\d+", next_label.group(1)):
                apple_function_labels.add(name)
            break

    # `.type ..., @/%function` is strongest.  When it is absent (Apple), use
    # both global symbols and Clang's Lfunc_begin markers so static helpers are
    # not silently dropped.
    if typed_functions:
        candidates = typed_functions
    elif apple_function_labels:
        # On Mach-O, .globl also marks global data objects.  When Clang's
        # Lfunc_begin markers are available, they are the reliable function
        # discriminator; using all globals here misclassifies data such as
        # `_infile` as code.
        candidates = apple_function_labels
    else:
        # Conservative fallback for unusual Apple assembly lacking Lfunc markers.
        candidates = globals_

    return {name for name in candidates if name in labels_present}


def _split_compiled_functions(lines):
    """Split compiler `.s` into real function bodies, stopping at `.cfi_endproc`."""
    function_labels = _find_compiled_function_labels(lines)

    starts = []
    for index, raw_line in enumerate(lines):
        m = LABEL_RE.match(raw_line)
        if m and m.group(1) in function_labels:
            starts.append((index, m.group(1)))

    if not starts:
        # Conservative fallback for small function-only assembly snippets.
        for index, raw_line in enumerate(lines):
            m = LABEL_RE.match(raw_line)
            if m:
                name = m.group(1)
                if not name.startswith(".") and not name.startswith("LBB"):
                    starts.append((index, name))
                    break

    if not starts:
        return [("unknown_function", lines)]

    functions = []
    for i, (start_index, function_name) in enumerate(starts):
        next_start = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        end_index = next_start

        # Prefer the compiler's explicit function end marker. This prevents
        # trailing numeric assembler labels / metadata from becoming fake
        # instructions in a one-function source file.
        for j in range(start_index + 1, next_start):
            if CFI_ENDPROC_RE.match(lines[j]):
                end_index = j + 1
                break

        functions.append((function_name, lines[start_index:end_index]))

    return functions


def _parse_compiled_function(lines, function_name, target_info):
    """Parse one compiler-generated function; `pc` is an instruction index."""
    instructions = []
    labels_to_pc = {}
    pending_labels = []
    pc = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        label_match = LABEL_RE.match(raw_line)
        if label_match:
            pending_labels.append(label_match.group(1))
            continue

        if stripped.startswith("."):
            continue

        code = _strip_instruction_comments(raw_line, target_info)
        if not code:
            continue

        parts = code.split(None, 1)
        opcode = parts[0].lower()
        ops_str = parts[1] if len(parts) > 1 else ""
        ops = _split_operands(ops_str)

        for label in pending_labels:
            labels_to_pc[label] = pc
        pending_labels = []

        instructions.append(
            Instruction(code, pc, opcode, ops, target_info, real_address=False)
        )
        pc += 1

    # Resolve symbolic direct branch/call labels. Calls are resolved too for
    # completeness, although intraprocedural CFG construction does not follow them.
    for inst in instructions:
        if not (inst.is_jump() or inst.is_call()) or not inst.ops:
            continue

        # Register-indirect branches (AArch64 `br xN`, RISC-V `jr aN`,
        # x86 `jmp *%reg`, etc.) are unresolved control flow, not symbolic
        # external direct branches.  Leave both target_pc and external_target
        # unset so CFG construction emits UNKNOWN.
        if inst.is_jump() and inst.is_indirect_jump():
            continue

        target_text = inst.ops[target_info.target_op_index()].strip()
        if target_text in labels_to_pc:
            inst.target_pc = labels_to_pc[target_text]
        elif inst.is_jump() and _extract_numeric_target(target_text) is None:
            # Symbolic direct jump not defined in the current function:
            # e.g. `jmp free@PLT`, `b _free`, or `tail free@plt`.
            inst.external_target = target_text

    # Apply the same target-aware NOP normalization used for objdump input.
    # GCC/Clang can leave an untargeted `nop` immediately before a labeled
    # epilogue at O0.  In the encoded binary that no-op may be folded into the
    # following block, so retaining it as its own compiler-only block creates a
    # representation-only topology difference.  A NOP that is itself the
    # target of a direct branch is still preserved.
    if instructions:
        candidate_noop_pcs = {
            inst.pc for inst in instructions if _is_padding_noop(inst.text, target_info)
        }
        direct_jump_targets = {
            inst.target_pc
            for inst in instructions
            if inst.is_jump() and inst.target_pc is not None
        }
        first_pc = instructions[0].pc
        removable_noops = {
            pc
            for pc in candidate_noop_pcs
            if pc not in direct_jump_targets and pc != first_pc
        }
        if removable_noops:
            instructions = [
                inst for inst in instructions if inst.pc not in removable_noops
            ]

    return instructions


def _is_objdump_local_label(name):
    # GNU objdump emits `.L2`, `.L3`, ... as symbol headers inside the enclosing
    # function.  Mach-O llvm-objdump can similarly surface jump-table labels such
    # as `lJTI3_0` inside __text.  These are basic-block/data labels, not
    # standalone functions.  Keep ltmpN separate because llvm-objdump sometimes
    # uses it as the only visible symbol for a real Mach-O object function.
    if name.startswith(".L"):
        return True
    if re.fullmatch(r"[lL]JTI\d+_\d+", name):
        return True
    return False


def _split_objdump_functions(lines):
    """Split objdump into functions while retaining local `.L*` headers inside their parent."""
    functions = []
    current_name = None
    current_lines = []

    for raw_line in lines:
        m = OBJDUMP_FUNCTION_RE.match(raw_line)
        if m:
            name = m.group(2)
            if _is_objdump_local_label(name) and current_name is not None:
                # Keep the marker in the current chunk; the instruction parser
                # simply ignores it, while following instruction addresses remain.
                current_lines.append(raw_line)
                continue

            if current_name is not None:
                functions.append((current_name, current_lines))

            current_name = name
            current_lines = []
            continue

        if current_name is not None:
            current_lines.append(raw_line)

    if current_name is not None:
        functions.append((current_name, current_lines))

    if not functions:
        return [("unknown_function", lines)]

    return functions


def _parse_objdump_function(lines, function_name, target_info):
    """Parse one objdump function using real disassembly addresses.

    Relocatable/binary disassembly requires two normalizations:
      * alignment-only no-ops may be omitted, but a no-op that is itself the
        target of a direct branch must be preserved as a real basic-block
        leader;
      * external branch relocations override unlinked placeholder displacements.

    The target-aware no-op rule is important at O0: GCC sometimes emits a
    labeled ``nop`` as the body of an otherwise-empty control-flow block.  At
    O2, by contrast, assemblers frequently materialize ``.p2align`` directives
    as untargeted NOP padding.  Dropping every NOP breaks O0 topology, while
    keeping every NOP creates spurious O2 blocks.
    """
    instructions = []
    candidate_noop_addresses = set()
    last_instruction = None
    pending_riscv_external_symbol = None

    for raw_line in lines:
        reloc_match = OBJDUMP_RELOCATION_RE.match(raw_line)
        if reloc_match:
            _, reloc_type, reloc_payload = reloc_match.groups()
            symbol = _relocation_symbol(reloc_payload)

            if (
                last_instruction is not None
                and last_instruction.is_jump()
                and _relocation_is_external_control_transfer(reloc_type, symbol)
            ):
                last_instruction.target_pc = None
                last_instruction.external_target = symbol

            # RISC-V external tail calls are commonly encoded as
            #   auipc t1, ...
            #   jr    t1
            # with R_RISCV_CALL(_PLT) attached to the AUIPC. Carry that
            # symbol forward one instruction so the final JR can be labeled
            # as an external edge rather than UNKNOWN.
            upper = reloc_type.upper()
            if (
                target_info.name == "riscv"
                and symbol
                and not symbol.startswith(".L")
                and last_instruction is not None
                and last_instruction.opcode == "auipc"
                and upper in {"R_RISCV_CALL", "R_RISCV_CALL_PLT"}
            ):
                pending_riscv_external_symbol = symbol
            continue

        match = OBJDUMP_INSTRUCTION_RE.match(raw_line)
        if not match:
            continue

        address = int(match.group(1), 16)
        rest = match.group(2).strip()
        if not rest:
            continue

        code = _remove_objdump_encoding(rest, target_info)
        if not code or _looks_like_relocation(code):
            continue

        code = _strip_instruction_comments(code, target_info)
        if not code or _looks_like_relocation(code):
            continue

        if _is_padding_noop(code, target_info):
            candidate_noop_addresses.add(address)

        parts = code.split(None, 1)
        opcode = parts[0].lower()
        ops_str = parts[1] if len(parts) > 1 else ""
        ops = _split_operands(ops_str)

        inst = Instruction(
            code, address, opcode, ops, target_info, real_address=True
        )

        if (inst.is_jump() or inst.is_call()) and inst.ops:
            # Do not parse register names such as RISC-V `a5` as hexadecimal
            # addresses, and do not turn AArch64 `br x8` into a fake external
            # symbol.  Relocation-backed RISC-V tail calls are handled below.
            if not (inst.is_jump() and inst.is_indirect_jump()):
                target_text = inst.ops[target_info.target_op_index()]
                inst.target_pc = _extract_numeric_target(target_text)

                symbol_match = re.search(r"<([^>]+)>", target_text)
                if symbol_match:
                    inst.external_target = symbol_match.group(1)

        if pending_riscv_external_symbol is not None:
            if inst.is_unconditional_jump():
                inst.target_pc = None
                inst.external_target = pending_riscv_external_symbol
            pending_riscv_external_symbol = None

        instructions.append(inst)
        last_instruction = inst

    # Keep NOPs that are direct branch targets: at O0 they can represent a
    # genuine labeled empty block (for example ``.L11: nop``).  Remove only
    # untargeted NOPs, which are overwhelmingly binary alignment/padding and
    # otherwise create representation-only blocks at O2.
    if candidate_noop_addresses and instructions:
        direct_jump_targets = {
            inst.target_pc
            for inst in instructions
            if inst.is_jump() and inst.target_pc is not None
        }
        first_pc = instructions[0].pc
        removable_noops = {
            pc
            for pc in candidate_noop_addresses
            if pc not in direct_jump_targets and pc != first_pc
        }
        if removable_noops:
            instructions = [
                inst for inst in instructions if inst.pc not in removable_noops
            ]

    return instructions


def parse_assembly(source, target_info, input_format="auto"):
    """
    Parse one source string into [(function_name, [Instruction, ...]), ...].

    input_format: 'auto', 'asm', or 'objdump'.
    """
    lines = source.splitlines() if isinstance(source, str) else list(source)

    if input_format not in {"auto", "asm", "objdump"}:
        raise ValueError("input_format must be 'auto', 'asm', or 'objdump'")

    if input_format == "auto":
        input_format = _detect_input_format(lines)

    parsed_functions = []

    if input_format == "asm":
        chunks = _split_compiled_functions(lines)
        for function_name, function_lines in chunks:
            instructions = _parse_compiled_function(function_lines, function_name, target_info)
            if instructions:
                parsed_functions.append((function_name, instructions))
    else:
        chunks = _split_objdump_functions(lines)
        for function_name, function_lines in chunks:
            instructions = _parse_objdump_function(function_lines, function_name, target_info)
            if instructions:
                parsed_functions.append((function_name, instructions))

    return parsed_functions


def build_basic_blocks(instructions, target_info):
    """
    Build basic blocks and intraprocedural direct CFG edges using the leader method.

    * direct in-function targets -> block edge
    * unresolved indirect targets -> UNKNOWN
    * direct target outside current function -> EXTERNAL_0x...
    """
    if not instructions:
        return {}

    pc_to_inst = {inst.pc: inst for inst in instructions}
    leaders = {instructions[0].pc}

    for i, inst in enumerate(instructions):
        next_inst = instructions[i + 1] if i + 1 < len(instructions) else None

        if inst.is_jump():
            if inst.target_pc in pc_to_inst:
                leaders.add(inst.target_pc)
            if next_inst is not None:
                leaders.add(next_inst.pc)
        elif inst.is_sink() and next_inst is not None:
            leaders.add(next_inst.pc)

    basic_blocks = {}
    current_block = None

    for inst in instructions:
        if current_block is None or inst.pc in leaders:
            current_block = BasicBlock(inst.pc)
            basic_blocks[current_block.key] = current_block
        current_block.add_instruction(inst)

    block_keys = list(basic_blocks.keys())
    pc_to_block = {}
    for block in basic_blocks.values():
        for inst in block.instructions:
            pc_to_block[inst.pc] = block.key

    blocks = list(basic_blocks.values())
    for block_index, block in enumerate(blocks):
        last_inst = block.instructions[-1]
        next_block_key = block_keys[block_index + 1] if block_index + 1 < len(block_keys) else None

        if last_inst.is_sink():
            continue

        if last_inst.is_jump():
            if last_inst.external_target is not None and last_inst.target_pc not in pc_to_block:
                block.add_jump_edge(f"EXTERNAL_{last_inst.external_target}")
            elif last_inst.target_pc is None:
                block.add_jump_edge("UNKNOWN")
            elif last_inst.target_pc in pc_to_block:
                block.add_jump_edge(pc_to_block[last_inst.target_pc])
            else:
                block.add_jump_edge(f"EXTERNAL_0x{last_inst.target_pc:x}")

            if not last_inst.is_unconditional_jump() and next_block_key is not None:
                block.add_no_jump_edge(next_block_key)
            continue

        if next_block_key is not None:
            block.add_no_jump_edge(next_block_key)

    return basic_blocks


def trim_unreachable_trailing_suffix(instructions, target_info):
    """Conservatively remove a trailing objdump suffix unreachable from entry.

    This is a standalone, file-local boundary recovery rule.  It uses only the
    instructions already parsed from the current artifact.  Starting at the
    function entry, it follows resolved intraprocedural JUMP/NO_JUMP edges.  A
    suffix is removable only when every block after the last reachable block is
    unreachable.

    If any *reachable* block has an UNKNOWN jump edge, no trimming is performed:
    an unresolved indirect branch could target one of the apparently unreachable
    later blocks.  This deliberately prefers retaining ambiguous bytes over
    deleting potentially reachable code.

    Returns ``(instructions, trimmed_instruction_count, reason)``.
    """
    if not instructions:
        return instructions, 0, ""

    blocks = build_basic_blocks(instructions, target_info)
    ordered = list(blocks.values())
    if len(ordered) < 2:
        return instructions, 0, ""

    block_keys = set(blocks)
    entry = ordered[0].key
    reachable = set()
    worklist = [entry]

    while worklist:
        key = worklist.pop()
        if key in reachable or key not in blocks:
            continue
        reachable.add(key)
        block = blocks[key]

        # UNKNOWN means a reachable indirect jump may enter code whose target
        # is not statically recoverable from this artifact.  Do not prune.
        if block.jump_edge == "UNKNOWN":
            return instructions, 0, ""

        for target in (block.no_jump_edge, block.jump_edge):
            if target in block_keys and target not in reachable:
                worklist.append(target)

    reachable_indices = [
        i for i, block in enumerate(ordered) if block.key in reachable
    ]
    if not reachable_indices:
        return instructions, 0, ""

    last_reachable_index = max(reachable_indices)
    if last_reachable_index == len(ordered) - 1:
        return instructions, 0, ""

    # Because this is specifically a *suffix* rule, do not delete holes inside
    # the retained prefix.  We cut only immediately before the first block after
    # the last reachable block.
    cut_key = ordered[last_reachable_index + 1].key
    cut_index = next(
        (i for i, inst in enumerate(instructions) if inst.pc == cut_key),
        None,
    )
    if cut_index is None or cut_index <= 0:
        return instructions, 0, ""

    trimmed = len(instructions) - cut_index
    if trimmed <= 0:
        return instructions, 0, ""

    return (
        instructions[:cut_index],
        trimmed,
        "trailing blocks unreachable from function entry",
    )


def _cfg_uses_real_addresses(basic_blocks):
    return any(
        inst.real_address
        for block in basic_blocks.values()
        for inst in block.instructions
    )


def _format_key(key, real_addresses):
    if isinstance(key, int):
        return f"0x{key:x}" if real_addresses else f"I{key}"
    return str(key)


def cfg_to_text(function_name, basic_blocks):
    """Render a compact textual CFG suitable for inspection or agent prompting."""
    real_addresses = _cfg_uses_real_addresses(basic_blocks)
    lines = [
        f"Function: {function_name}",
        f"Basic blocks: {len(basic_blocks)}",
        "",
    ]

    for key, block in basic_blocks.items():
        lines.extend([
            "====",
            f"BLOCK {_format_key(key, real_addresses)}",
            "- Instructions:",
        ])

        for inst in block.instructions:
            if real_addresses:
                lines.append(f"  0x{inst.pc:x}: {inst.text}")
            else:
                lines.append(f"  {inst.text}")

        edges = []
        if block.no_jump_edge is not None:
            edges.append(("NO_JUMP", block.no_jump_edge))
        if block.jump_edge is not None:
            edges.append(("JUMP", block.jump_edge))

        lines.append("")
        if edges:
            lines.append("- Edges:")
            for edge_kind, target in edges:
                lines.append(f"  {edge_kind} -> {_format_key(target, real_addresses)}")
        else:
            lines.append("- Edges: (none)")
        lines.append("")

    return "\n".join(lines).rstrip()


def canonical_symbol(name, target_info):
    """Return the target-appropriate canonical spelling of a function symbol.

    Mach-O AArch64 prefixes C symbols with ``_``.  ELF/Linux does not, and a
    leading underscore can be part of the real Linux symbol name, so stripping
    it globally would conflate distinct symbols.
    """
    name = name.strip()
    if (
        target_info.name == "arm"
        and getattr(target_info, "platform", None) == "macos"
        and name.startswith("_")
    ):
        return name[1:]
    return name


def select_function(parsed_functions, preferred="func0", *, target_info,
                    allow_largest_fallback=False):
    """Select one function from parse_assembly() output for this target."""
    preferred_canonical = canonical_symbol(preferred, target_info)
    exact = [
        item
        for item in parsed_functions
        if canonical_symbol(item[0], target_info) == preferred_canonical
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Multiple matches for {preferred!r}: {[n for n, _ in exact]}")

    if allow_largest_fallback and parsed_functions:
        return max(parsed_functions, key=lambda item: len(item[1]))

    raise ValueError(
        f"Could not find function {preferred!r}; parsed: {[n for n, _ in parsed_functions]}"
    )


def generate_cfg_text(source, target_info, input_format="auto", function_name=None,
                      allow_largest_fallback=False):
    """
    Generate CFG text.

    If function_name is None, render every parsed function. If supplied, render
    only that function (target-specific symbol normalization is applied). For Mach-O
    relocatable objects that expose a temporary symbol such as ltmp0, set
    allow_largest_fallback=True when the source is known to contain one target
    function.
    """
    if input_format == "auto":
        source_lines = source.splitlines() if isinstance(source, str) else list(source)
        resolved_input_format = _detect_input_format(source_lines)
    else:
        resolved_input_format = input_format

    parsed_functions = parse_assembly(
        source, target_info, input_format=resolved_input_format
    )

    if function_name is not None:
        parsed_functions = [
            select_function(
                parsed_functions,
                preferred=function_name,
                target_info=target_info,
                allow_largest_fallback=allow_largest_fallback,
            )
        ]

    outputs = []
    for parsed_name, instructions in parsed_functions:
        if resolved_input_format == "objdump":
            instructions, _, _ = trim_unreachable_trailing_suffix(
                instructions, target_info
            )
        basic_blocks = build_basic_blocks(instructions, target_info)
        outputs.append(cfg_to_text(parsed_name, basic_blocks))

    return "\n\n".join(outputs)


if __name__ == "__main__":
    # Small local CLI for inspecting one file. Dataset-wide upload logic is
    # intentionally kept separate from this analysis module.
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument(
        "--target",
        choices=["x86_linux", "arm_linux", "arm_mac", "riscv_linux"],
        required=True,
    )
    parser.add_argument("--format", choices=["auto", "asm", "objdump"], default="auto")
    parser.add_argument("--function", default=None)
    parser.add_argument("--largest-fallback", action="store_true")
    args = parser.parse_args()

    target_map = {
        "x86_linux": X86TargetInfo(),
        "arm_linux": ARMTargetInfo("linux"),
        "arm_mac": ARMTargetInfo("macos"),
        "riscv_linux": RISCVTargetInfo(),
    }

    source = Path(args.source).read_text(encoding="utf-8")
    print(
        generate_cfg_text(
            source,
            target_map[args.target],
            input_format=args.format,
            function_name=args.function,
            allow_largest_fallback=args.largest_fallback,
        )
    )
