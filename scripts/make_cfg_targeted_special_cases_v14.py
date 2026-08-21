#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
CFG_ROOT = SCRIPT_DIR.parent
VALIDATION = CFG_ROOT / "validation" / "v14"
OUT = VALIDATION / "CFG_TARGETED_SPECIAL_CASES_V14.txt"

sys.path.insert(0, str(SCRIPT_DIR))

from generate_cfg_updated_v14 import (  # noqa: E402
    parse_assembly,
    trim_unreachable_trailing_suffix,
)
from run_cfg_bringup_v14 import (  # noqa: E402
    REPOS,
    rows_by_task,
    reference_functions,
    map_functions_to_reference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def extract_function_section(path: Path, function_name: str) -> str:
    """Extract exactly one Function: ... section from a Bringup CFG bundle."""
    text = path.read_text(encoding="utf-8")

    pattern = re.compile(
        rf"(?ms)^Function:\s*{re.escape(function_name)}\s*$"
        rf".*?(?=^Function:\s|\Z)"
    )

    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            f"Could not find function {function_name!r} in {path}"
        )

    return match.group(0).rstrip()


def emit_function(out, title: str, path: Path, function_name: str):
    out.write("\n")
    out.write("-" * 78 + "\n")
    out.write(title + "\n")
    out.write(f"FILE: {path}\n")
    out.write("-" * 78 + "\n")
    out.write(extract_function_section(path, function_name))
    out.write("\n")


def emit_case_header(out, number: int, title: str, explanation: str):
    out.write("\n\n")
    out.write("#" * 78 + "\n")
    out.write(f"TARGETED CASE {number}: {title}\n")
    out.write("#" * 78 + "\n")
    out.write(explanation.strip() + "\n")


def first_bringup_row(predicate, split_order=("O0", "O2")):
    """
    Return the first Bringup TSV row satisfying predicate(split, row).

    Prefer a non-reference representation because targeted cases are most useful
    when they show a compiler reference beside a binary representation.
    """
    fallback = None

    for split in split_order:
        tsv = VALIDATION / f"bringup_{split}" / "summary.tsv"

        for row in read_tsv(tsv):
            if not predicate(split, row):
                continue

            candidate = (split, row)

            if row["asm_column"] != row["reference_column"]:
                return candidate

            if fallback is None:
                fallback = candidate

    if fallback is not None:
        return fallback

    raise RuntimeError("No matching Bringup TSV row found")


def emit_reference_and_representation(out, split: str, row: dict):
    task = row["task"]
    arch = row["arch"]
    function = row["function"]
    reference = row["reference_column"]
    representation = row["asm_column"]

    out.write(f"\nSplit:          {split}\n")
    out.write(f"Task:           {task}\n")
    out.write(f"Architecture:   {arch}\n")
    out.write(f"Family:         {row['family']}\n")
    out.write(f"Function:       {function}\n")
    out.write(f"Reference:      {reference}\n")
    out.write(f"Representation: {representation}\n")
    out.write(f"UNKNOWN count:  {row['unknown']}\n")
    out.write(f"EXTERNAL count: {row['external']}\n")
    out.write(f"Topology:       {row['topology_vs_reference']}\n")

    base = VALIDATION / f"bringup_{split}" / task / arch

    emit_function(
        out,
        f"SAME-PROVENANCE REFERENCE: {reference}",
        base / f"{reference}.cfg.txt",
        function,
    )

    if representation != reference:
        emit_function(
            out,
            f"REPRESENTATION: {representation}",
            base / f"{representation}.cfg.txt",
            function,
        )


def format_instruction(inst) -> str:
    if isinstance(inst.pc, int):
        pc = f"0x{inst.pc:x}"
    else:
        pc = str(inst.pc)

    return f"{pc}: {inst.text}"


def first_provenance_function_set_difference():
    """
    Find one normal-vs-PIC compiler function-set difference from a V14 summary.

    This is informational only: normal and PIC compiler invocations are separate
    compilation provenances and are not required to emit identical function sets.
    """
    for split in ("O2", "O0"):
        summary = VALIDATION / f"bringup_{split}" / "summary.txt"
        current_task = None

        for line in summary.read_text(encoding="utf-8").splitlines():
            task_match = re.fullmatch(r"=== (.+) ===", line)
            if task_match:
                current_task = task_match.group(1)
                continue

            match = re.match(
                r"\s*(\S+)\s+PROVENANCE_FUNCTION_SET_DIFF\s+"
                r"normal_only=(\[[^\]]*\])\s+pic_only=(\[[^\]]*\])\s*$",
                line,
            )
            if not match:
                continue

            arch = match.group(1)
            normal_only = ast.literal_eval(match.group(2))
            pic_only = ast.literal_eval(match.group(3))

            if current_task is None:
                raise RuntimeError(
                    f"Found provenance difference without task context in {summary}"
                )

            return {
                "split": split,
                "task": current_task,
                "arch": arch,
                "normal_only": normal_only,
                "pic_only": pic_only,
            }

    raise RuntimeError("No normal-vs-PIC function-set difference found")


# ---------------------------------------------------------------------------
# Case 1: UNKNOWN edge
# ---------------------------------------------------------------------------

unknown_split, unknown_row = first_bringup_row(
    lambda split, row: int(row["unknown"]) > 0
)


# ---------------------------------------------------------------------------
# Case 2: generic EXTERNAL edge
#
# Prefer O2 and exclude the special terminal-success validation case so this
# case demonstrates ordinary direct out-of-function control flow.
# ---------------------------------------------------------------------------

external_split, external_row = first_bringup_row(
    lambda split, row:
        int(row["external"]) > 0
        and row["topology_vs_reference"]
        != "MATCH_ARM_SHARED_TERMINAL_SUCCESS_BRANCH",
    split_order=("O2", "O0"),
)


# ---------------------------------------------------------------------------
# Case 3: recognized RISC-V long-branch normalization
# ---------------------------------------------------------------------------

long_split, long_row = first_bringup_row(
    lambda split, row:
        row["arch"] == "riscv_linux"
        and row["topology_vs_reference"] == "MATCH_LONG_BRANCH_NORMALIZED"
)


# ---------------------------------------------------------------------------
# Case 4: AArch64 Linux standalone reachability boundary recovery
#
# The V14 summary reports the total number of trimmed instructions but does not
# record a per-function trim field in summary.tsv. Re-run the exact standalone
# V14 recovery rule on shared_asm functions until the first real case is found.
#
# The compiler reference is used here only to identify/map the corresponding
# source function for this audit report. The trim decision itself receives only
# the current shared-object instruction list plus the AArch64 target semantics.
# ---------------------------------------------------------------------------

repo_id, target_info = REPOS["arm_linux"]
dataset = load_dataset(repo_id, split="O0")
rows = rows_by_task(dataset)

boundary_case = None

for task, row in rows.items():
    pic_compiler_source = row.get("compiler_pic_asm")
    shared_source = row.get("shared_asm")

    if not pic_compiler_source or not shared_source:
        continue

    compiler_parsed = parse_assembly(
        pic_compiler_source,
        target_info,
        input_format="auto",
    )
    shared_parsed = parse_assembly(
        shared_source,
        target_info,
        input_format="auto",
    )

    refs = reference_functions(
        compiler_parsed,
        "arm_linux",
    )
    mapped = map_functions_to_reference(
        refs,
        shared_parsed,
        "arm_linux",
        "shared_asm",
    )

    for ref in refs:
        canonical = ref["canonical"]
        representation_symbol, raw_instructions = mapped[canonical]

        (
            trimmed_instructions,
            trimmed_count,
            trim_reason,
        ) = trim_unreachable_trailing_suffix(
            raw_instructions,
            target_info,
        )

        if trimmed_count > 0:
            removed = raw_instructions[len(trimmed_instructions):]

            boundary_case = {
                "task": task,
                "function": canonical,
                "representation_symbol": representation_symbol,
                "trimmed_count": trimmed_count,
                "trim_reason": trim_reason,
                "removed": removed,
            }
            break

    if boundary_case is not None:
        break


if boundary_case is None:
    raise RuntimeError(
        "Expected an O0 AArch64 Linux shared-object standalone boundary-recovery "
        "case, but none was found"
    )


# ---------------------------------------------------------------------------
# Case 5: linked-only terminal success branch, validation only
# ---------------------------------------------------------------------------

terminal_split, terminal_row = first_bringup_row(
    lambda split, row:
        row["arch"] == "arm_linux"
        and row["asm_column"] == "shared_asm"
        and row["topology_vs_reference"]
        == "MATCH_ARM_SHARED_TERMINAL_SUCCESS_BRANCH"
)


# ---------------------------------------------------------------------------
# Case 6: legitimate normal-vs-PIC function-set difference
# ---------------------------------------------------------------------------

provenance_case = first_provenance_function_set_difference()


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

VALIDATION.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as out:

    out.write("CFG V14 TARGETED SPECIAL-CASE MANUAL AUDIT\n")
    out.write("=" * 78 + "\n")
    out.write("\n")
    out.write(
        "Purpose: supplement the broad 72-check / 432-representation manual "
        "spot check with cases that deliberately exercise unusual CFG behavior.\n"
    )
    out.write("\n")
    out.write(
        "V14 design rule: production CFG extraction is standalone. Each CFG is "
        "constructed from the current assembly/disassembly artifact alone. "
        "Cross-representation information is used only by this validation/audit "
        "layer after CFG generation.\n"
    )
    out.write("\n")
    out.write("Cases included:\n")
    out.write("  1. unresolved indirect jump -> UNKNOWN\n")
    out.write("  2. known direct out-of-function jump -> EXTERNAL\n")
    out.write("  3. RISC-V assembler long-branch expansion\n")
    out.write(
        "  4. AArch64 Linux standalone reachability-based trailing-suffix recovery\n"
    )
    out.write(
        "  5. AArch64 Linux linked-only terminal success branch retained in the "
        "real CFG and recognized only during validation\n"
    )
    out.write("  6. legitimate normal-vs-PIC compiler function-set difference\n")


    # CASE 1
    emit_case_header(
        out,
        1,
        "UNRESOLVED INDIRECT JUMP (UNKNOWN)",
        """
An indirect jump uses a runtime-computed destination, for example a register
value. If the target cannot be determined statically from the current artifact,
the intraprocedural CFG must not invent a destination. V14 records the taken
edge as UNKNOWN.
        """,
    )

    emit_reference_and_representation(
        out,
        unknown_split,
        unknown_row,
    )


    # CASE 2
    emit_case_header(
        out,
        2,
        "DIRECT OUT-OF-FUNCTION CONTROL FLOW (EXTERNAL)",
        """
A direct branch/tail transfer may have a statically known target that lies
outside the current function. Because V14 is intraprocedural, the target is
represented as EXTERNAL_* rather than as an internal basic block. This case
deliberately excludes the Bringup-specific linked-only terminal-success case
shown separately below.
        """,
    )

    emit_reference_and_representation(
        out,
        external_split,
        external_row,
    )


    # CASE 3
    emit_case_header(
        out,
        3,
        "RISC-V LONG-BRANCH ASSEMBLER EXPANSION",
        """
A source-level RISC-V conditional branch can be too far away for the encoded
conditional-branch displacement. The assembler may invert the condition and
insert a one-instruction unconditional-jump trampoline.

The binary CFG intentionally keeps that real trampoline. Validation alone
recognizes the strict equivalent pattern, producing
MATCH_LONG_BRANCH_NORMALIZED. The emitted CFG is never rewritten to match the
compiler assembly.
        """,
    )

    emit_reference_and_representation(
        out,
        long_split,
        long_row,
    )


    # CASE 4
    emit_case_header(
        out,
        4,
        "AARCH64 LINUX STANDALONE REACHABILITY-BASED BOUNDARY RECOVERY",
        """
Some AArch64 ELF shared-object disassemblies attach trailing unnamed instructions
to the preceding source-function chunk. V14 no longer uses compiler_pic_asm as
a function-end anchor.

Instead, for shared_asm only, V14 tentatively builds the CFG from that function's
own instructions, computes reachability from the entry, and removes only a
trailing suffix whose blocks are unreachable. If a reachable UNKNOWN indirect
jump exists, V14 refuses to trim because that jump might target the apparent
suffix.

The compiler reference below is shown only for manual validation after the
standalone trim; it did not determine the trim boundary.
        """,
    )

    task = boundary_case["task"]
    function = boundary_case["function"]

    out.write(f"\nSplit:                 O0\n")
    out.write(f"Task:                  {task}\n")
    out.write("Architecture:          arm_linux\n")
    out.write(f"Function:              {function}\n")
    out.write(
        f"Representation symbol: {boundary_case['representation_symbol']}\n"
    )
    out.write(
        f"Instructions trimmed:  {boundary_case['trimmed_count']}\n"
    )
    out.write(
        f"Trim reason:           {boundary_case['trim_reason']}\n"
    )

    out.write("\nRAW SHARED-OBJECT SUFFIX REMOVED BY V14 STANDALONE RECOVERY\n")
    out.write("-" * 78 + "\n")

    for inst in boundary_case["removed"]:
        out.write(format_instruction(inst) + "\n")

    boundary_base = (
        VALIDATION
        / "bringup_O0"
        / task
        / "arm_linux"
    )

    emit_function(
        out,
        "SAME-PROVENANCE PIC COMPILER REFERENCE (VALIDATION ONLY)",
        boundary_base / "compiler_pic_asm.cfg.txt",
        function,
    )

    emit_function(
        out,
        "V14 STANDALONE-RECOVERED REPRESENTATION: shared_asm",
        boundary_base / "shared_asm.cfg.txt",
        function,
    )


    # CASE 5
    emit_case_header(
        out,
        5,
        "AARCH64 LINUX LINKED-ONLY TERMINAL SUCCESS BRANCH (VALIDATION ONLY)",
        """
In two Bringup O0 AArch64 Linux shared-object cases, the linked disassembly
contains a final direct branch to libtarg_success immediately after a call to
libmin_success. The same-provenance compiler assembly and PIC relocatable object
do not contain that final branch.

The standalone generator conservatively RETAINS the branch because it is a real
instruction and is locally reachable under the calls-return policy. V14 does not
trim or rewrite it. Only after the real CFG has been generated does the Bringup
validation harness recognize the exact linked-only success-path pattern and
compare an alternate topology signature against the independently generated PIC
compiler reference.

Therefore the emitted shared-object CFG below must still contain the
EXTERNAL_libtarg_success edge.
        """,
    )

    emit_reference_and_representation(
        out,
        terminal_split,
        terminal_row,
    )


    # CASE 6
    emit_case_header(
        out,
        6,
        "LEGITIMATE NORMAL-vs-PIC COMPILER FUNCTION-SET DIFFERENCE",
        """
The normal and PIC compiler invocations are separate compilation provenances.
At O2 they are not required to emit identical optimized function sets. V14
therefore validates object/program representations only against compiler_asm,
and pic_object/shared representations only against compiler_pic_asm.

This case shows one function that exists in only one compiler provenance. Such a
cross-family difference is informational and is not a CFG validation failure.
        """,
    )

    split = provenance_case["split"]
    task = provenance_case["task"]
    arch = provenance_case["arch"]
    normal_only = provenance_case["normal_only"]
    pic_only = provenance_case["pic_only"]

    out.write(f"\nSplit:                     {split}\n")
    out.write(f"Task:                      {task}\n")
    out.write(f"Architecture:              {arch}\n")
    out.write(f"Functions only in normal:  {normal_only}\n")
    out.write(f"Functions only in PIC:     {pic_only}\n")

    if normal_only:
        chosen_function = normal_only[0]
        family = "NORMAL"
        reps = ("compiler_asm", "object_asm", "program_asm")
    elif pic_only:
        chosen_function = pic_only[0]
        family = "PIC"
        reps = ("compiler_pic_asm", "pic_object_asm", "shared_asm")
    else:
        raise RuntimeError(
            "Provenance function-set difference contained no differing functions"
        )

    out.write(f"Selected function:         {chosen_function}\n")
    out.write(f"Selected provenance:       {family}\n")

    provenance_base = (
        VALIDATION
        / f"bringup_{split}"
        / task
        / arch
    )

    for rep in reps:
        emit_function(
            out,
            f"{family} REPRESENTATION: {rep}",
            provenance_base / f"{rep}.cfg.txt",
            chosen_function,
        )


    out.write("\n\n")
    out.write("=" * 78 + "\n")
    out.write("TARGETED SPECIAL-CASE AUDIT GENERATION COMPLETE\n")
    out.write("=" * 78 + "\n")
    out.write("Cases generated: 6 / 6\n")


print()
print("PASS: generated 6 targeted CFG special cases")
print()
print("Output:")
print(f"  {OUT}")
print()
print("Size:")
if OUT.exists():
    size = OUT.stat().st_size
    print(f"  {size:,} bytes")
