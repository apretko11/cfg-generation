#!/usr/bin/env python3
"""
Generate and validate multi-function CFG bundles for Bringup-Bench.

Bringup-Bench differs from HumanEval/MC-Eval: one benchmark row is a complete
translation unit containing multiple source-defined functions.

V13 validates two independent compilation-provenance families:

    normal: compiler_asm     -> object_asm, program_asm
    PIC:    compiler_pic_asm -> pic_object_asm, shared_asm

Each compiler-assembly column independently defines the source-function set for
its own family.  A function may legitimately exist in only one family (for
example, because optimization under -fPIC created or removed a clone).

Linked runtime/PLT/startup functions are ignored unless they are present in the
corresponding compiler-assembly reference for that provenance family.

Outputs:
    <out>/<task>/<arch>/<column>.cfg.txt   # all source-function CFGs
    <out>/summary.txt
    <out>/summary.tsv                     # one row per function CFG
    <out>/problems.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset

from generate_cfg_updated_v13 import (
    X86TargetInfo,
    ARMTargetInfo,
    RISCVTargetInfo,
    parse_assembly,
    build_basic_blocks,
    cfg_to_text,
)


REPOS = {
    "x86_linux": (
        "adpretko/bringup_x86_linux_reloc",
        X86TargetInfo(),
    ),
    "arm_linux": (
        "adpretko/bringup_arm_linux_reloc",
        ARMTargetInfo(),
    ),
    "arm_mac": (
        "adpretko/bringup_arm_mac_reloc",
        ARMTargetInfo(),
    ),
    "riscv_linux": (
        "adpretko/bringup_riscv_linux_reloc",
        RISCVTargetInfo(),
    ),
}

PROVENANCE_FAMILIES = {
    "normal": {
        "reference": "compiler_asm",
        "columns": [
            "compiler_asm",
            "object_asm",
            "program_asm",
        ],
    },
    "pic": {
        "reference": "compiler_pic_asm",
        "columns": [
            "compiler_pic_asm",
            "pic_object_asm",
            "shared_asm",
        ],
    },
}

ASM_COLUMNS = [
    column
    for family in PROVENANCE_FAMILIES.values()
    for column in family["columns"]
]

REFERENCE_COLUMNS = {
    family["reference"]
    for family in PROVENANCE_FAMILIES.values()
}

_TEMP_MACHO_SYMBOL_RE = re.compile(r"(?:l?tmp\d+|Ltmp\d+)$", re.IGNORECASE)


def block_topology_signature(basic_blocks):
    """Canonicalize CFG connectivity while ignoring addresses/instruction spelling."""
    blocks = list(basic_blocks.values())
    key_to_index = {block.key: i for i, block in enumerate(blocks)}

    def normalize_target(target):
        if target is None:
            return None
        if target in key_to_index:
            return f"B{key_to_index[target]}"
        target_text = str(target)
        if target_text.startswith("EXTERNAL_"):
            return "OUTSIDE"
        return target_text

    return tuple(
        (
            normalize_target(block.no_jump_edge),
            normalize_target(block.jump_edge),
        )
        for block in blocks
    )



def comparison_topology_signature(basic_blocks, arch_name, asm_column):
    """
    Return a topology signature used only for cross-representation validation.

    RISC-V assemblers must expand an out-of-range conditional branch because the
    B-type branch encoding has a limited displacement.  A source-level branch

        b<cond> target

    can therefore become

        b<!cond> .+8
        j        target

    in the encoded binary.  That introduces one real trampoline basic block,
    even though the source-level control-flow decision is unchanged.

    For RISC-V objdump representations only, recognize the exact structural
    expansion above and collapse the one-instruction `j` trampoline for
    *comparison*.  The emitted Harbor CFG is NOT changed: it still describes
    the actual binary instructions and therefore retains the extra block.
    """
    raw = block_topology_signature(basic_blocks)

    if arch_name != "riscv_linux" or asm_column in REFERENCE_COLUMNS:
        return raw, 0

    blocks = list(basic_blocks.values())
    if len(blocks) < 3:
        return raw, 0

    key_set = {b.key for b in blocks}
    incoming = {b.key: 0 for b in blocks}
    for b in blocks:
        for target in (b.no_jump_edge, b.jump_edge):
            if target in key_set:
                incoming[target] += 1

    removed = set()
    override_edges = {}
    collapsed = 0

    # Canonical GNU/LLVM RISC-V long-conditional expansion:
    #
    #   B_i:   inverted conditional
    #            NO_JUMP -> B_{i+1}  (one-instruction `j` trampoline)
    #            JUMP    -> B_{i+2}  (skip trampoline)
    #   B_i+1: j original_target
    #
    # Restore the source-level polarity for comparison:
    #   NO_JUMP -> skip
    #   JUMP    -> original_target
    for i in range(len(blocks) - 2):
        cond = blocks[i]
        tramp = blocks[i + 1]
        skip = blocks[i + 2]

        if cond.key in removed or tramp.key in removed:
            continue
        if not cond.instructions:
            continue

        cond_last = cond.instructions[-1]
        if not cond_last.is_jump() or cond_last.is_unconditional_jump():
            continue

        if cond.no_jump_edge != tramp.key or cond.jump_edge != skip.key:
            continue

        if incoming.get(tramp.key, 0) != 1:
            continue

        if len(tramp.instructions) != 1:
            continue

        tramp_inst = tramp.instructions[0]
        if not tramp_inst.is_jump() or not tramp_inst.is_unconditional_jump():
            continue
        if tramp.no_jump_edge is not None or tramp.jump_edge is None:
            continue

        # Do not collapse unresolved/external control flow.  More importantly,
        # require the trampoline target to be outside the architectural
        # conditional-branch reach (RISC-V B-type: roughly +/-4 KiB).  This
        # sharply distinguishes genuine assembler long-branch relaxation from
        # ordinary source CFGs that happen to contain a conditional followed by
        # a one-instruction jump block.
        if tramp.jump_edge not in key_set:
            continue
        if abs(int(tramp.jump_edge) - int(cond_last.pc)) <= 4094:
            continue

        override_edges[cond.key] = (skip.key, tramp.jump_edge)
        removed.add(tramp.key)
        collapsed += 1

    if not removed:
        return raw, 0

    retained = [b for b in blocks if b.key not in removed]
    key_to_index = {b.key: i for i, b in enumerate(retained)}

    def normalize_target(target):
        if target is None:
            return None
        if target in key_to_index:
            return f"B{key_to_index[target]}"
        text = str(target)
        if text.startswith("EXTERNAL_"):
            return "OUTSIDE"
        return text

    signature = []
    for b in retained:
        no_jump, jump = override_edges.get(
            b.key, (b.no_jump_edge, b.jump_edge)
        )
        signature.append(
            (normalize_target(no_jump), normalize_target(jump))
        )

    return tuple(signature), collapsed


def validate_basic_blocks(basic_blocks):
    """Return obvious structural CFG problems."""
    problems = []
    block_keys = set(basic_blocks)

    for block in basic_blocks.values():
        if not block.instructions:
            problems.append(f"empty block {block.key}")
            continue

        last = block.instructions[-1]

        if last.is_sink() and (
            block.jump_edge is not None or block.no_jump_edge is not None
        ):
            problems.append(f"sink block {block.key} unexpectedly has successor")

        if last.is_unconditional_jump() and block.no_jump_edge is not None:
            problems.append(
                f"unconditional jump block {block.key} has fallthrough edge"
            )

        if last.is_jump() and block.jump_edge is None:
            problems.append(f"jump block {block.key} has no jump edge")

        for edge_name, target in (
            ("NO_JUMP", block.no_jump_edge),
            ("JUMP", block.jump_edge),
        ):
            if target is None:
                continue
            if isinstance(target, int) and target not in block_keys:
                problems.append(
                    f"{edge_name} from {block.key} points to missing block {target}"
                )

    return problems


def count_edges(basic_blocks):
    jump = sum(b.jump_edge is not None for b in basic_blocks.values())
    fallthrough = sum(b.no_jump_edge is not None for b in basic_blocks.values())
    unknown = sum(b.jump_edge == "UNKNOWN" for b in basic_blocks.values())
    external = sum(
        isinstance(b.jump_edge, str) and b.jump_edge.startswith("EXTERNAL_")
        for b in basic_blocks.values()
    )
    return jump, fallthrough, unknown, external


def task_id_from_row(row):
    for key in ("problem_name", "task_name", "task_id", "name", "id", "problem_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(
        "Could not find a task identifier; tried "
        "problem_name/task_name/task_id/name/id/problem_id. "
        f"Available columns: {list(row.keys())}"
    )


def rows_by_task(dataset):
    rows = {}
    for row in dataset:
        task_id = task_id_from_row(row)
        if task_id in rows:
            raise ValueError(f"Duplicate task identifier: {task_id}")
        rows[task_id] = row
    return rows


def choose_tasks(datasets, num_tasks, start):
    x86_order = list(datasets["x86_linux"].keys())
    common = [
        task
        for task in x86_order
        if all(task in datasets[arch] for arch in REPOS)
    ]

    if num_tasks is None:
        return common[start:]

    chosen = common[start:start + num_tasks]
    if len(chosen) != num_tasks:
        raise RuntimeError(
            f"Requested {num_tasks} common tasks starting at {start}, "
            f"but only found {len(chosen)}"
        )
    return chosen


def canonical_symbol_for_arch(name, arch_name):
    """Normalize C symbols only where the object format requires it.

    Mach-O prefixes C symbols with `_`; ELF does not.  Stripping a leading
    underscore globally conflates distinct Linux symbols such as `init` and
    `_init`, so normalization is deliberately architecture-specific here.
    """
    name = name.strip()
    if arch_name == "arm_mac" and name.startswith("_"):
        return name[1:]
    return name


def reference_functions(parsed_functions, arch_name):
    """
    Return compiler-emitted functions in program order, keyed by canonical C name.

    compiler_asm is the authority for which functions belong to the benchmark
    translation unit.  Static helpers are included; linked runtime functions are not.
    """
    refs = []
    seen = set()
    for symbol, instructions in parsed_functions:
        canonical = canonical_symbol_for_arch(symbol, arch_name)
        if canonical in seen:
            raise ValueError(
                f"Duplicate canonical compiler function {canonical!r}; "
                f"parsed symbols: {[name for name, _ in parsed_functions]}"
            )
        seen.add(canonical)
        refs.append(
            {
                "canonical": canonical,
                "compiler_symbol": symbol,
                "instructions": instructions,
            }
        )
    return refs



def _normalized_call_target(inst):
    """Return a comparable symbolic call target when one is visible."""
    if not inst.is_call():
        return None

    target = inst.external_target
    if not target and inst.ops:
        try:
            target = inst.ops[inst.info.target_op_index()].strip()
        except (IndexError, AttributeError):
            target = inst.ops[-1].strip()

    if not target:
        return None

    m = re.search(r"<([^>]+)>", target)
    if m:
        target = m.group(1)

    # Remove representation decorations while preserving the actual ELF symbol
    # spelling (in particular, do NOT globally strip a leading underscore).
    target = re.sub(r"@(?:plt|plt\.sec)$", "", target, flags=re.IGNORECASE)
    target = re.sub(r"[+-]0x[0-9A-Fa-f]+$", "", target)
    return target.strip() or None


def trim_arm_linux_shared_suffix_to_reference_end(reference_instructions, repr_instructions):
    """Trim AArch64 ELF shared-object bytes that objdump attached to a function.

    In a few Bringup-Bench AArch64 ``.so`` files, the disassembly has unnamed
    linker stubs/islands immediately after the true end of a source function.
    Because there is no intervening function-symbol header, a text-only objdump
    splitter assigns those bytes to the preceding function.

    Use the same-provenance compiler reference only as a *function-boundary anchor*:
      * if the compiler function ends in a sink (normally ``ret``), cut the
        representation after the last matching sink;
      * if the compiler function ends in a call, cut only when the same final
        symbolic call is visible in the representation.  This covers the O0
        harness exits such as ``libmin_success`` without changing the global
        calls-return CFG policy.

    The rule is deliberately applied by the Bringup runner only to
    arm_linux/shared_asm.  It does not alter object/program/Mach-O parsing and
    it does not mark calls as CFG sinks.
    """
    if not reference_instructions or not repr_instructions:
        return repr_instructions, 0, ""

    reference_last = reference_instructions[-1]
    cut_index = None
    reason = ""

    if reference_last.is_sink():
        # Match the same terminal opcode where possible.  Search from the end so
        # earlier return blocks inside a multi-exit function are left intact.
        for i in range(len(repr_instructions) - 1, -1, -1):
            inst = repr_instructions[i]
            if inst.is_sink() and inst.opcode == reference_last.opcode:
                cut_index = i + 1
                reason = f"reference-final sink {reference_last.opcode}"
                break

    elif reference_last.is_call():
        compiler_target = _normalized_call_target(reference_last)
        if compiler_target:
            for i in range(len(repr_instructions) - 1, -1, -1):
                inst = repr_instructions[i]
                if not inst.is_call():
                    continue
                if _normalized_call_target(inst) == compiler_target:
                    cut_index = i + 1
                    reason = f"reference-final call {compiler_target}"
                    break

    if cut_index is None or cut_index >= len(repr_instructions):
        return repr_instructions, 0, ""

    trimmed = len(repr_instructions) - cut_index
    return repr_instructions[:cut_index], trimmed, reason


def map_functions_to_reference(refs, parsed_functions, arch_name, asm_column):
    """
    Map a representation's parsed functions to its same-provenance compiler reference.

    Normal ELF/Mach-O linked representations are matched by canonical symbol name.

    Mach-O relocatable objects have one known quirk: llvm-objdump may expose an
    otherwise named source function as an ``ltmpN`` symbol.  Exact symbol matches
    are anchored first; any remaining compiler functions may then map one-to-one,
    in program order, only to remaining ltmpN entries.  We never use a largest-
    function heuristic.
    """
    mapped = {}
    used_indices = set()

    by_canonical = {}
    for i, (symbol, instructions) in enumerate(parsed_functions):
        by_canonical.setdefault(canonical_symbol_for_arch(symbol, arch_name), []).append(i)

    # Exact canonical matches first.
    for ref in refs:
        canonical = ref["canonical"]
        indices = [i for i in by_canonical.get(canonical, []) if i not in used_indices]
        if len(indices) == 1:
            i = indices[0]
            symbol, instructions = parsed_functions[i]
            mapped[canonical] = (symbol, instructions)
            used_indices.add(i)
        elif len(indices) > 1:
            raise ValueError(
                f"Multiple matches for source function {canonical!r}: "
                f"{[parsed_functions[i][0] for i in indices]}"
            )

    missing_refs = [ref for ref in refs if ref["canonical"] not in mapped]

    if missing_refs and arch_name == "arm_mac" and asm_column in {"object_asm", "pic_object_asm"}:
        remaining = [
            (i, symbol, instructions)
            for i, (symbol, instructions) in enumerate(parsed_functions)
            if i not in used_indices
        ]
        temp_remaining = [item for item in remaining if _TEMP_MACHO_SYMBOL_RE.fullmatch(item[1])]

        # Safe positional fallback only when every unmatched parsed function is an
        # anonymous Mach-O temporary and the cardinalities agree.
        if len(temp_remaining) == len(remaining) == len(missing_refs):
            # Preserve compiler/object program order.  Exact matches on either side
            # act as anchors; for the common case this maps compiler main -> ltmp0.
            ref_order = {ref["canonical"]: i for i, ref in enumerate(refs)}
            obj_order = {i: i for i in range(len(parsed_functions))}
            missing_refs = sorted(missing_refs, key=lambda r: ref_order[r["canonical"]])
            temp_remaining = sorted(temp_remaining, key=lambda item: obj_order[item[0]])
            for ref, (i, symbol, instructions) in zip(missing_refs, temp_remaining):
                mapped[ref["canonical"]] = (symbol, instructions)
                used_indices.add(i)

    still_missing = [ref["canonical"] for ref in refs if ref["canonical"] not in mapped]
    if still_missing:
        raise ValueError(
            f"Missing source functions {still_missing}; parsed representation functions: "
            f"{[name for name, _ in parsed_functions]}"
        )

    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="O0")
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of common tasks; default is all tasks in the split",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset within the common Bringup-Bench task order",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"cfg_bringup_{args.split}_v13"
    if args.num_tasks is not None and args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0 when supplied")
    if args.start < 0:
        raise ValueError("--start must be >= 0")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading split {args.split!r} from all four repositories...")
    datasets = {}
    for arch_name, (repo_id, _target_info) in REPOS.items():
        print(f"  {arch_name:12} <- {repo_id}")
        ds = load_dataset(repo_id, split=args.split)
        datasets[arch_name] = rows_by_task(ds)

    tasks = choose_tasks(datasets, args.num_tasks, args.start)
    expected_bundles = len(tasks) * len(REPOS) * len(ASM_COLUMNS)

    print("\nTasks selected:")
    for i, task in enumerate(tasks, start=1):
        print(f"  {i:3d}. {task}")
    print(
        f"\nExpected bundle files: {len(tasks)} x {len(REPOS)} x "
        f"{len(ASM_COLUMNS)} = {expected_bundles}\n"
    )

    detail_lines = []
    tsv_rows = [
        "task\tarch\tfamily\treference_column\tasm_column\tfunction\t"
        "compiler_symbol\trepresentation_symbol\tinsns\tblocks\tjump\t"
        "fallthrough\tunknown\texternal\tvalidation\ttopology_vs_reference\terror"
    ]
    problem_lines = []

    generated_bundles = 0
    expected_function_cfgs = 0
    generated_function_cfgs = 0
    family_expected_function_cfgs = {
        family_name: 0 for family_name in PROVENANCE_FAMILIES
    }
    family_generated_function_cfgs = {
        family_name: 0 for family_name in PROVENANCE_FAMILIES
    }

    parse_failures = 0
    validation_failures = 0
    raw_topology_diffs = 0
    topology_diffs = 0
    riscv_long_branch_expansions = 0
    total_unknown = 0
    total_external = 0
    arm_shared_boundary_trimmed = 0

    cross_family_function_set_differences = 0
    cross_family_normal_only_functions = 0
    cross_family_pic_only_functions = 0

    for task_index, task_name in enumerate(tasks, start=1):
        print(f"=== [{task_index}/{len(tasks)}] {task_name} ===")
        detail_lines.append(f"=== {task_name} ===")

        for arch_name, (_repo_id, target_info) in REPOS.items():
            row = datasets[arch_name][task_name]
            family_ref_sets = {}

            for family_name, family_spec in PROVENANCE_FAMILIES.items():
                reference_column = family_spec["reference"]
                family_columns = family_spec["columns"]
                reference_source = row.get(reference_column)

                try:
                    if not reference_source:
                        raise RuntimeError(
                            f"missing/empty {reference_column}"
                        )

                    reference_parsed = parse_assembly(
                        reference_source,
                        target_info,
                        input_format="auto",
                    )
                    if not reference_parsed:
                        raise RuntimeError(
                            f"no functions parsed from {reference_column}"
                        )

                    refs = reference_functions(
                        reference_parsed,
                        arch_name,
                    )
                except Exception as exc:
                    # A family without its compiler reference cannot define its
                    # own source-function set, but it must not prevent the other
                    # provenance family from being validated independently.
                    err = f"{type(exc).__name__}: {exc}"
                    parse_failures += len(family_columns)
                    problem_lines.append(
                        f"{task_name}\t{arch_name}\t{family_name}\t"
                        f"{reference_column}\tERROR\t{err}"
                    )
                    msg = (
                        f"  {arch_name:12} {family_name:6} ALL_COLUMNS "
                        f"functions=? valid=ERROR topology=ERROR ERROR={err}"
                    )
                    print(msg)
                    detail_lines.append(msg)
                    continue

                family_ref_sets[family_name] = {
                    ref["canonical"] for ref in refs
                }

                family_expected = len(refs) * len(family_columns)
                expected_function_cfgs += family_expected
                family_expected_function_cfgs[family_name] += family_expected

                compiler_symbols = {
                    ref["canonical"]: ref["compiler_symbol"]
                    for ref in refs
                }

                per_column = {}

                for asm_column in family_columns:
                    source = row.get(asm_column)
                    column_results = {}
                    column_texts = []
                    column_error = ""

                    try:
                        if not source:
                            raise RuntimeError(
                                "missing/empty assembly column"
                            )

                        parsed = parse_assembly(
                            source,
                            target_info,
                            input_format="auto",
                        )
                        if not parsed:
                            raise RuntimeError("no functions parsed")

                        if asm_column == reference_column:
                            mapped = {
                                ref["canonical"]: (
                                    ref["compiler_symbol"],
                                    ref["instructions"],
                                )
                                for ref in refs
                            }
                        else:
                            mapped = map_functions_to_reference(
                                refs,
                                parsed,
                                arch_name,
                                asm_column,
                            )

                        for ref in refs:
                            canonical = ref["canonical"]
                            symbol, instructions = mapped[canonical]

                            boundary_trimmed = 0
                            boundary_trim_reason = ""

                            # shared_asm belongs to the PIC family.  Therefore
                            # this V12 boundary-recovery mechanism is retained,
                            # but its anchor is now compiler_pic_asm rather than
                            # the unrelated non-PIC compiler_asm.
                            if (
                                arch_name == "arm_linux"
                                and asm_column == "shared_asm"
                            ):
                                (
                                    instructions,
                                    boundary_trimmed,
                                    boundary_trim_reason,
                                ) = trim_arm_linux_shared_suffix_to_reference_end(
                                    ref["instructions"],
                                    instructions,
                                )
                                arm_shared_boundary_trimmed += boundary_trimmed

                            blocks = build_basic_blocks(
                                instructions,
                                target_info,
                            )
                            problems = validate_basic_blocks(blocks)
                            jump, fall, unknown, external = count_edges(blocks)
                            (
                                comparison_signature,
                                long_branch_expansions,
                            ) = comparison_topology_signature(
                                blocks,
                                arch_name,
                                asm_column,
                            )

                            result = {
                                "canonical": canonical,
                                "compiler_symbol": compiler_symbols[canonical],
                                "representation_symbol": symbol,
                                "insns": len(instructions),
                                "blocks": len(blocks),
                                "jump": jump,
                                "fall": fall,
                                "unknown": unknown,
                                "external": external,
                                "validation": (
                                    "OK" if not problems else "FAIL"
                                ),
                                "error": "",
                                "signature": block_topology_signature(blocks),
                                "comparison_signature": comparison_signature,
                                "long_branch_expansions": long_branch_expansions,
                                "boundary_trimmed": boundary_trimmed,
                                "boundary_trim_reason": boundary_trim_reason,
                            }
                            column_results[canonical] = result

                            # Render the canonical family-reference name so a
                            # Mach-O temporary object symbol does not obscure
                            # function identity in the emitted CFG bundle.
                            column_texts.append(
                                cfg_to_text(canonical, blocks)
                            )

                            generated_function_cfgs += 1
                            family_generated_function_cfgs[family_name] += 1
                            total_unknown += unknown
                            total_external += external
                            riscv_long_branch_expansions += (
                                long_branch_expansions
                            )

                            if problems:
                                validation_failures += 1
                                for p in problems:
                                    problem_lines.append(
                                        f"{task_name}\t{arch_name}\t"
                                        f"{family_name}\t{asm_column}\t"
                                        f"{canonical}\tVALIDATION\t{p}"
                                    )

                        task_dir = out_root / task_name / arch_name
                        task_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        cfg_path = (
                            task_dir / f"{asm_column}.cfg.txt"
                        )
                        cfg_path.write_text(
                            "\n\n".join(column_texts) + "\n",
                            encoding="utf-8",
                        )
                        generated_bundles += 1

                    except Exception as exc:
                        parse_failures += 1
                        column_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        problem_lines.append(
                            f"{task_name}\t{arch_name}\t"
                            f"{family_name}\t{asm_column}\t"
                            f"ERROR\t{column_error}"
                        )

                    per_column[asm_column] = {
                        "results": column_results,
                        "error": column_error,
                    }

                # Compare each representation only to the compiler assembly
                # from the SAME provenance family.
                ref_results = (
                    per_column
                    .get(reference_column, {})
                    .get("results", {})
                )

                for asm_column in family_columns:
                    column = per_column[asm_column]
                    results = column["results"]
                    column_error = column["error"]
                    column_topologies = []

                    if column_error:
                        bundle_topology = "ERROR"
                    else:
                        for ref in refs:
                            canonical = ref["canonical"]
                            result = results[canonical]
                            ref_result = ref_results.get(canonical)

                            if asm_column == reference_column:
                                topology = "MATCH"
                            elif ref_result is None:
                                topology = "NO_REFERENCE"
                            else:
                                raw_match = (
                                    result["signature"]
                                    == ref_result["signature"]
                                )
                                normalized_match = (
                                    result["comparison_signature"]
                                    == ref_result["comparison_signature"]
                                )

                                if not raw_match:
                                    raw_topology_diffs += 1

                                if raw_match:
                                    topology = "MATCH"
                                elif normalized_match:
                                    topology = (
                                        "MATCH_LONG_BRANCH_NORMALIZED"
                                    )
                                else:
                                    topology = "DIFF"
                                    topology_diffs += 1
                                    problem_lines.append(
                                        f"{task_name}\t{arch_name}\t"
                                        f"{family_name}\t{asm_column}\t"
                                        f"{canonical}\tTOPOLOGY\t"
                                        f"differs from {reference_column}"
                                    )

                            column_topologies.append(topology)

                            tsv_rows.append(
                                "\t".join(
                                    [
                                        task_name,
                                        arch_name,
                                        family_name,
                                        reference_column,
                                        asm_column,
                                        canonical,
                                        result["compiler_symbol"],
                                        result["representation_symbol"],
                                        str(result["insns"]),
                                        str(result["blocks"]),
                                        str(result["jump"]),
                                        str(result["fall"]),
                                        str(result["unknown"]),
                                        str(result["external"]),
                                        result["validation"],
                                        topology,
                                        result["error"]
                                        .replace("\t", " ")
                                        .replace("\n", " "),
                                    ]
                                )
                            )

                        bundle_topology = (
                            "MATCH"
                            if column_topologies
                            and all(
                                t in {
                                    "MATCH",
                                    "MATCH_LONG_BRANCH_NORMALIZED",
                                }
                                for t in column_topologies
                            )
                            else "DIFF"
                        )

                    if results:
                        total_insns = sum(
                            r["insns"] for r in results.values()
                        )
                        total_blocks = sum(
                            r["blocks"] for r in results.values()
                        )
                        total_jump = sum(
                            r["jump"] for r in results.values()
                        )
                        total_fall = sum(
                            r["fall"] for r in results.values()
                        )
                        total_u = sum(
                            r["unknown"] for r in results.values()
                        )
                        total_e = sum(
                            r["external"] for r in results.values()
                        )
                        valid = (
                            "OK"
                            if all(
                                r["validation"] == "OK"
                                for r in results.values()
                            )
                            else "FAIL"
                        )
                        msg = (
                            f"  {arch_name:12} {family_name:6} "
                            f"{asm_column:16} "
                            f"functions={len(results):2d} "
                            f"insns={total_insns:4d} "
                            f"blocks={total_blocks:3d} "
                            f"jump={total_jump:3d} "
                            f"fall={total_fall:3d} "
                            f"unknown={total_u:2d} "
                            f"external={total_e:2d} "
                            f"valid={valid:5} "
                            f"topology={bundle_topology}"
                        )
                    else:
                        msg = (
                            f"  {arch_name:12} {family_name:6} "
                            f"{asm_column:16} functions= 0 "
                            f"valid=ERROR topology={bundle_topology}"
                        )

                    if column_error:
                        msg += f" ERROR={column_error}"

                    print(msg)
                    detail_lines.append(msg)

            # Cross-family function-set differences are now informational.
            # They are expected to be possible because normal and PIC code were
            # separately compiled.
            normal_set = family_ref_sets.get("normal")
            pic_set = family_ref_sets.get("pic")

            if (
                normal_set is not None
                and pic_set is not None
                and normal_set != pic_set
            ):
                normal_only = sorted(normal_set - pic_set)
                pic_only = sorted(pic_set - normal_set)

                cross_family_function_set_differences += 1
                cross_family_normal_only_functions += len(normal_only)
                cross_family_pic_only_functions += len(pic_only)

                info = (
                    f"  {arch_name:12} PROVENANCE_FUNCTION_SET_DIFF "
                    f"normal_only={normal_only} pic_only={pic_only}"
                )
                print(info)
                detail_lines.append(info)

            detail_lines.append("")

        print()

    overall_ok = (
        generated_bundles == expected_bundles
        and generated_function_cfgs == expected_function_cfgs
        and parse_failures == 0
        and validation_failures == 0
        and topology_diffs == 0
    )

    header = [
        "CFG BRINGUP-BENCH MULTI-FUNCTION VALIDATION SUMMARY",
        "===================================================",
        f"Split: {args.split}",
        f"Tasks: {len(tasks)} ({', '.join(tasks)})",
        (
            "Provenance families: "
            "normal[compiler_asm -> object_asm, program_asm]; "
            "pic[compiler_pic_asm -> pic_object_asm, shared_asm]"
        ),
        f"Expected bundle files: {expected_bundles}",
        f"Generated bundle files: {generated_bundles}",
        f"Expected function CFGs: {expected_function_cfgs}",
        f"Generated function CFGs: {generated_function_cfgs}",
        (
            "Normal-family expected/generated function CFGs: "
            f"{family_expected_function_cfgs['normal']}/"
            f"{family_generated_function_cfgs['normal']}"
        ),
        (
            "PIC-family expected/generated function CFGs: "
            f"{family_expected_function_cfgs['pic']}/"
            f"{family_generated_function_cfgs['pic']}"
        ),
        f"Parse/generation failures: {parse_failures}",
        f"Structural validation failures: {validation_failures}",
        (
            "Raw function topology differences vs same-provenance "
            f"compiler reference: {raw_topology_diffs}"
        ),
        (
            "Function topology differences vs same-provenance compiler "
            "reference after recognized RISC-V long-branch normalization: "
            f"{topology_diffs}"
        ),
        (
            "Recognized RISC-V long-branch expansion trampolines: "
            f"{riscv_long_branch_expansions}"
        ),
        (
            "AArch64 Linux shared-object trailing instructions trimmed "
            "by compiler_pic_asm boundary anchors: "
            f"{arm_shared_boundary_trimmed}"
        ),
        (
            "Task/ISA pairs with informational normal-vs-PIC function-set "
            f"differences: {cross_family_function_set_differences}"
        ),
        (
            "Functions present only in normal compiler references: "
            f"{cross_family_normal_only_functions}"
        ),
        (
            "Functions present only in PIC compiler references: "
            f"{cross_family_pic_only_functions}"
        ),
        f"Unresolved indirect jump edges: {total_unknown}",
        f"Direct external jump edges: {total_external}",
        f"OVERALL: {'PASS' if overall_ok else 'REVIEW NEEDED'}",
        "",
    ]

    summary_path = out_root / "summary.txt"
    summary_path.write_text(
        "\n".join(header + detail_lines) + "\n",
        encoding="utf-8",
    )

    tsv_path = out_root / "summary.tsv"
    tsv_path.write_text(
        "\n".join(tsv_rows) + "\n",
        encoding="utf-8",
    )

    problems_path = out_root / "problems.txt"
    if problem_lines:
        problems_path.write_text(
            "\n".join(problem_lines) + "\n",
            encoding="utf-8",
        )
    else:
        problems_path.write_text(
            "No problems detected.\n",
            encoding="utf-8",
        )

    print("=" * 72)
    for line in header[:-1]:
        print(line)
    print(f"\nOutputs: {out_root}")
    print(f"  {summary_path}")
    print(f"  {tsv_path}")
    print(f"  {problems_path}")


if __name__ == "__main__":
    main()
