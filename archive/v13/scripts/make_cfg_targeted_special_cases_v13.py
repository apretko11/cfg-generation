#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from datasets import load_dataset


HOME = Path.home()
CFG_ROOT = HOME / "CFG_Analysis"
SCRIPTS = CFG_ROOT / "scripts"
VALIDATION = CFG_ROOT / "validation" / "v13"
OUT = VALIDATION / "CFG_TARGETED_SPECIAL_CASES_V13.txt"

sys.path.insert(0, str(SCRIPTS))

from generate_cfg_updated_v13 import parse_assembly
from run_cfg_bringup_v13 import (
    REPOS,
    rows_by_task,
    reference_functions,
    map_functions_to_reference,
    trim_arm_linux_shared_suffix_to_reference_end,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def extract_function_section(path: Path, function_name: str) -> str:
    """
    Extract exactly one Function: ... section from a Bringup CFG bundle.
    """
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


def first_bringup_row(predicate):
    """
    Search Bringup O0 first, then O2, returning the first TSV row satisfying
    predicate(split, row).

    Prefer a non-reference representation because that lets us inspect both
    the compiler reference and another binary representation.
    """
    fallback = None

    for split in ("O0", "O2"):
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


# ---------------------------------------------------------------------------
# Case 1: UNKNOWN edge
# ---------------------------------------------------------------------------

unknown_split, unknown_row = first_bringup_row(
    lambda split, row: int(row["unknown"]) > 0
)


# ---------------------------------------------------------------------------
# Case 2: EXTERNAL edge
# ---------------------------------------------------------------------------

external_split, external_row = first_bringup_row(
    lambda split, row: int(row["external"]) > 0
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
# Case 4: AArch64 Linux shared-object boundary trim
#
# summary.tsv does not contain the per-function boundary_trimmed field.
# Re-run only the exact V13 boundary-detection logic against the already
# existing HF input rows until the first real trimmed function is found.
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
        ) = trim_arm_linux_shared_suffix_to_reference_end(
            ref["instructions"],
            raw_instructions,
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
        "Expected an O0 AArch64 Linux shared-object boundary-trim case, "
        "but none was found"
    )


# ---------------------------------------------------------------------------
# Case 5: known normal-vs-PIC provenance difference
#
# Broad spot checking already showed Bringup O2 / mersenne / genrand differs
# structurally between the normal and PIC compiler outputs.
# Show all six forms together.
# ---------------------------------------------------------------------------

PROVENANCE_TASK = "mersenne"
PROVENANCE_ARCH = "x86_linux"
PROVENANCE_FUNCTION = "genrand"
PROVENANCE_SPLIT = "O2"

PROVENANCE_REPS = (
    "compiler_asm",
    "object_asm",
    "program_asm",
    "compiler_pic_asm",
    "pic_object_asm",
    "shared_asm",
)


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

with OUT.open("w", encoding="utf-8") as out:

    out.write("CFG V13 TARGETED SPECIAL-CASE MANUAL AUDIT\n")
    out.write("=" * 78 + "\n")
    out.write("\n")
    out.write(
        "Purpose: supplement the broad 432-representation manual spot check "
        "with cases that deliberately exercise unusual CFG behavior.\n"
    )
    out.write("\n")
    out.write("Cases included:\n")
    out.write("  1. unresolved indirect jump -> UNKNOWN\n")
    out.write("  2. known direct out-of-function jump -> EXTERNAL\n")
    out.write("  3. RISC-V assembler long-branch expansion\n")
    out.write("  4. AArch64 Linux shared-object function-boundary trimming\n")
    out.write("  5. legitimate normal-vs-PIC CFG difference\n")


    # CASE 1
    emit_case_header(
        out,
        1,
        "UNRESOLVED INDIRECT JUMP (UNKNOWN)",
        """
An indirect jump uses a runtime-computed destination, for example a register
value. If the target cannot be determined statically, the intraprocedural CFG
must not invent a destination. V13 records the taken edge as UNKNOWN.
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
outside the current function. Because V13 is intraprocedural, the target is
represented as EXTERNAL_* rather than as an internal basic block.
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
MATCH_LONG_BRANCH_NORMALIZED.
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
        "AARCH64 LINUX SHARED-OBJECT FUNCTION-BOUNDARY TRIM",
        """
Some AArch64 ELF shared-object disassemblies attach unnamed linker-generated
instructions after the real source-function end. V13 uses compiler_pic_asm,
which belongs to the same PIC provenance family as shared_asm, as the narrow
function-end anchor.
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

    out.write("\nRAW SHARED-OBJECT SUFFIX REMOVED BY V13\n")
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
        "PIC COMPILER REFERENCE: compiler_pic_asm",
        boundary_base / "compiler_pic_asm.cfg.txt",
        function,
    )

    emit_function(
        out,
        "VALIDATED/TRIMMED REPRESENTATION: shared_asm",
        boundary_base / "shared_asm.cfg.txt",
        function,
    )


    # CASE 5
    emit_case_header(
        out,
        5,
        "LEGITIMATE NORMAL-vs-PIC CFG DIFFERENCE",
        """
The normal and PIC compiler invocations are separate compilation provenances.
They are not required to produce identical optimized CFGs.

For this known Bringup O2 example, inspect the three normal representations
together and the three PIC representations together. The correct requirement
is agreement WITHIN each provenance family, not equality BETWEEN the two
families.
        """,
    )

    out.write(f"\nSplit:        {PROVENANCE_SPLIT}\n")
    out.write(f"Task:         {PROVENANCE_TASK}\n")
    out.write(f"Architecture: {PROVENANCE_ARCH}\n")
    out.write(f"Function:     {PROVENANCE_FUNCTION}\n")

    provenance_base = (
        VALIDATION
        / f"bringup_{PROVENANCE_SPLIT}"
        / PROVENANCE_TASK
        / PROVENANCE_ARCH
    )

    for rep in PROVENANCE_REPS:

        family = (
            "NORMAL"
            if rep in {
                "compiler_asm",
                "object_asm",
                "program_asm",
            }
            else "PIC"
        )

        emit_function(
            out,
            f"{family} REPRESENTATION: {rep}",
            provenance_base / f"{rep}.cfg.txt",
            PROVENANCE_FUNCTION,
        )


    out.write("\n\n")
    out.write("=" * 78 + "\n")
    out.write("TARGETED SPECIAL-CASE AUDIT GENERATION COMPLETE\n")
    out.write("=" * 78 + "\n")
    out.write("Cases generated: 5 / 5\n")


print()
print("PASS: generated 5 targeted CFG special cases")
print()
print("Output:")
print(f"  {OUT}")
print()
print("Size:")
