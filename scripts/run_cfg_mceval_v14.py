#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_dataset
from generate_cfg_updated_v14 import (
    X86TargetInfo, ARMTargetInfo, RISCVTargetInfo,
    parse_assembly, build_basic_blocks, cfg_to_text,
    trim_unreachable_trailing_suffix,
)

REPOS = {
    "x86_linux": ("adpretko/mceval_x86_linux_reloc", X86TargetInfo()),
    "arm_linux": ("adpretko/mceval_arm_linux_reloc", ARMTargetInfo("linux")),
    "arm_mac": ("adpretko/mceval_arm_mac_reloc", ARMTargetInfo("macos")),
    "riscv_linux": ("adpretko/mceval_riscv_linux_reloc", RISCVTargetInfo()),
}
PROVENANCE_FAMILIES = {
    "normal": {"reference": "compiler_asm", "columns": ["compiler_asm", "object_asm", "program_asm"]},
    "pic": {"reference": "compiler_pic_asm", "columns": ["compiler_pic_asm", "pic_object_asm", "shared_asm"]},
}
ASM_COLUMNS = [c for f in PROVENANCE_FAMILIES.values() for c in f["columns"]]
REFERENCE_COLUMNS = {f["reference"] for f in PROVENANCE_FAMILIES.values()}
DEFAULT_OUT_PREFIX = "cfg_mceval"
SUMMARY_TITLE = "CFG MC-EVAL PROVENANCE-AWARE VALIDATION SUMMARY"

def task_id_from_row(row):
    for key in ("task_name", "task_id", "name", "id", "problem_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"Could not find task identifier; columns={list(row.keys())}")

def choose_reference_function(parsed_functions, row, arch_name, asm_column):
    available = {}
    for name, insns in parsed_functions:
        canonical = canonical_symbol_for_arch(name, arch_name)
        if canonical in available:
            raise ValueError(f"Duplicate canonical function {canonical!r}")
        available[canonical] = (name, insns)

    for key in ("entry_point", "function_name", "target_function", "func_name", "method_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            wanted = canonical_symbol_for_arch(value.strip(), arch_name)
            if wanted in available:
                return available[wanted]

    if "func0" in available:
        return available["func0"]

    if len(parsed_functions) == 1:
        return parsed_functions[0]

    non_main = [
        item for item in parsed_functions
        if canonical_symbol_for_arch(item[0], arch_name) not in {"main", "_start"}
    ]
    if len(non_main) == 1:
        return non_main[0]

    raise ValueError(
        f"Ambiguous MC-Eval target function in {asm_column}; "
        f"parsed={[name for name, _ in parsed_functions]}"
    )


def canonical_symbol_for_arch(name, arch_name):
    name = name.strip()
    if arch_name == "arm_mac" and name.startswith("_"):
        return name[1:]
    return name


def block_topology_signature(basic_blocks):
    blocks = list(basic_blocks.values())
    key_to_index = {block.key: i for i, block in enumerate(blocks)}

    def normalize_target(target):
        if target is None:
            return None
        if target in key_to_index:
            return f"B{key_to_index[target]}"
        text = str(target)
        if text.startswith("EXTERNAL_"):
            return "OUTSIDE"
        return text

    return tuple(
        (normalize_target(b.no_jump_edge), normalize_target(b.jump_edge))
        for b in blocks
    )


def comparison_topology_signature(basic_blocks, arch_name, asm_column):
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

    for i in range(len(blocks) - 2):
        cond = blocks[i]
        tramp = blocks[i + 1]
        skip = blocks[i + 2]

        if cond.key in removed or tramp.key in removed or not cond.instructions:
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
    fall = sum(b.no_jump_edge is not None for b in basic_blocks.values())
    unknown = sum(b.jump_edge == "UNKNOWN" for b in basic_blocks.values())
    external = sum(
        isinstance(b.jump_edge, str) and b.jump_edge.startswith("EXTERNAL_")
        for b in basic_blocks.values()
    )
    return jump, fall, unknown, external


def rows_by_task(dataset):
    rows = {}
    for row in dataset:
        task = task_id_from_row(row)
        if task in rows:
            raise ValueError(f"Duplicate task identifier: {task}")
        rows[task] = row
    return rows


def choose_tasks(datasets, num_tasks, start):
    x86_order = list(datasets["x86_linux"].keys())
    common = [
        task for task in x86_order
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


def select_named_function(parsed, preferred, arch_name, asm_column):
    matches = [
        item for item in parsed
        if canonical_symbol_for_arch(item[0], arch_name) == preferred
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches for {preferred!r}: {[x[0] for x in matches]}"
        )

    if arch_name == "arm_mac" and asm_column in {"object_asm", "pic_object_asm"}:
        # Preserve the previously validated narrow Mach-O relocatable-object
        # fallback for ltmp-style symbol loss.
        if parsed:
            return max(parsed, key=lambda item: len(item[1]))

    raise ValueError(
        f"Could not find function {preferred!r}; parsed: {[x[0] for x in parsed]}"
    )


def run_validation(args):
    if args.out is None:
        args.out = f"{DEFAULT_OUT_PREFIX}_{args.split}_v14"

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
    expected_cfgs = len(tasks) * len(REPOS) * len(ASM_COLUMNS)

    print(
        f"\nExpected CFGs: {len(tasks)} x 4 ISAs x 6 representations "
        f"= {expected_cfgs}\n"
    )

    detail_lines = []
    tsv_rows = [
        "task\tarch\tfamily\tasm_column\tfunction\treference_function\t"
        "insns\tblocks\tjump\tfallthrough\tunknown\texternal\tvalidation\t"
        "topology_vs_same_provenance_reference\terror"
    ]
    problem_lines = []

    generated = 0
    parse_failures = 0
    validation_failures = 0
    raw_topology_diffs = 0
    topology_diffs = 0
    long_branch_expansions = 0
    standalone_boundary_trimmed = 0
    total_unknown = 0
    total_external = 0
    family_generated = {"normal": 0, "pic": 0}
    reference_name_diffs = 0

    for task_index, task_name in enumerate(tasks, 1):
        print(f"=== [{task_index}/{len(tasks)}] {task_name} ===")
        detail_lines.append(f"=== {task_name} ===")

        for arch_name, (_repo_id, target_info) in REPOS.items():
            row = datasets[arch_name][task_name]
            family_results = {}

            for family_name, family in PROVENANCE_FAMILIES.items():
                ref_column = family["reference"]
                ref_source = row.get(ref_column)

                try:
                    if not ref_source:
                        raise RuntimeError(f"missing/empty {ref_column}")
                    ref_parsed = parse_assembly(
                        ref_source, target_info, input_format="auto"
                    )
                    if not ref_parsed:
                        raise RuntimeError(f"no functions parsed from {ref_column}")

                    ref_name, ref_insns = choose_reference_function(
                        ref_parsed, row, arch_name, ref_column
                    )
                    preferred = canonical_symbol_for_arch(ref_name, arch_name)
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    parse_failures += len(family["columns"])
                    problem_lines.append(
                        f"{task_name}\t{arch_name}\t{family_name}\t"
                        f"{ref_column}\tERROR\t{err}"
                    )
                    family_results[family_name] = {
                        "reference_name": None,
                        "columns": {},
                        "error": err,
                    }
                    continue

                columns = {}

                for asm_column in family["columns"]:
                    source = row.get(asm_column)
                    result = {
                        "function": "",
                        "reference_function": preferred,
                        "insns": 0,
                        "blocks": 0,
                        "jump": 0,
                        "fall": 0,
                        "unknown": 0,
                        "external": 0,
                        "validation": "ERROR",
                        "error": "",
                    }

                    try:
                        if not source:
                            raise RuntimeError("missing/empty assembly column")

                        parsed = parse_assembly(
                            source, target_info, input_format="auto"
                        )
                        if not parsed:
                            raise RuntimeError("no functions parsed")

                        if asm_column == ref_column:
                            function_name, instructions = ref_name, ref_insns
                        else:
                            function_name, instructions = select_named_function(
                                parsed, preferred, arch_name, asm_column
                            )

                        # Mirror the standalone V14 generator's file-local
                        # boundary recovery for objdump representations.  This
                        # uses only the current representation; no paired
                        # compiler/object/program file participates in trimming.
                        boundary_trimmed = 0
                        if asm_column not in REFERENCE_COLUMNS:
                            (
                                instructions,
                                boundary_trimmed,
                                _boundary_trim_reason,
                            ) = trim_unreachable_trailing_suffix(
                                instructions, target_info
                            )
                            standalone_boundary_trimmed += boundary_trimmed

                        blocks = build_basic_blocks(instructions, target_info)
                        problems = validate_basic_blocks(blocks)
                        jump, fall, unknown, external = count_edges(blocks)
                        comp_sig, collapsed = comparison_topology_signature(
                            blocks, arch_name, asm_column
                        )

                        result.update(
                            function=function_name,
                            insns=len(instructions),
                            blocks=len(blocks),
                            jump=jump,
                            fall=fall,
                            unknown=unknown,
                            external=external,
                            validation="OK" if not problems else "FAIL",
                            signature=block_topology_signature(blocks),
                            comparison_signature=comp_sig,
                            long_branch_expansions=collapsed,
                        )

                        task_dir = out_root / task_name / arch_name
                        task_dir.mkdir(parents=True, exist_ok=True)
                        (task_dir / f"{asm_column}.cfg.txt").write_text(
                            cfg_to_text(preferred, blocks) + "\n",
                            encoding="utf-8",
                        )

                        generated += 1
                        family_generated[family_name] += 1
                        total_unknown += unknown
                        total_external += external
                        long_branch_expansions += collapsed

                        if problems:
                            validation_failures += 1
                            for p in problems:
                                problem_lines.append(
                                    f"{task_name}\t{arch_name}\t{family_name}\t"
                                    f"{asm_column}\t{preferred}\tVALIDATION\t{p}"
                                )

                    except Exception as exc:
                        parse_failures += 1
                        result["error"] = f"{type(exc).__name__}: {exc}"
                        problem_lines.append(
                            f"{task_name}\t{arch_name}\t{family_name}\t"
                            f"{asm_column}\t{preferred}\tERROR\t{result['error']}"
                        )

                    columns[asm_column] = result

                family_results[family_name] = {
                    "reference_name": preferred,
                    "columns": columns,
                    "error": "",
                }

            normal_ref = family_results.get("normal", {}).get("reference_name")
            pic_ref = family_results.get("pic", {}).get("reference_name")
            if normal_ref and pic_ref and normal_ref != pic_ref:
                reference_name_diffs += 1
                problem_lines.append(
                    f"{task_name}\t{arch_name}\tINFO\tREFERENCE_NAME_DIFF\t"
                    f"normal={normal_ref}\tpic={pic_ref}"
                )

            for family_name, family in PROVENANCE_FAMILIES.items():
                family_data = family_results.get(family_name, {})
                columns = family_data.get("columns", {})
                ref_column = family["reference"]
                ref_result = columns.get(ref_column)
                reference = (
                    ref_result.get("comparison_signature")
                    if ref_result and not ref_result.get("error")
                    else None
                )

                for asm_column in family["columns"]:
                    result = columns.get(asm_column)
                    if result is None:
                        continue

                    if result["error"]:
                        topology = "ERROR"
                    elif asm_column == ref_column:
                        topology = "MATCH"
                    elif reference is None:
                        topology = "NO_REFERENCE"
                    else:
                        raw_match = (
                            result["signature"] == ref_result["signature"]
                        )
                        normalized_match = (
                            result["comparison_signature"] == reference
                        )

                        if not raw_match:
                            raw_topology_diffs += 1

                        if raw_match:
                            topology = "MATCH"
                        elif normalized_match:
                            topology = "MATCH_LONG_BRANCH_NORMALIZED"
                        else:
                            topology = "DIFF"
                            topology_diffs += 1
                            problem_lines.append(
                                f"{task_name}\t{arch_name}\t{family_name}\t"
                                f"{asm_column}\t{result['reference_function']}\t"
                                f"TOPOLOGY\tdiffers from {ref_column}"
                            )

                    msg = (
                        f"  {arch_name:12} {family_name:6} {asm_column:16} "
                        f"func={result['function']!r:18} "
                        f"insns={result['insns']:3d} blocks={result['blocks']:2d} "
                        f"jump={result['jump']:2d} fall={result['fall']:2d} "
                        f"unknown={result['unknown']:2d} external={result['external']:2d} "
                        f"valid={result['validation']:5} topology={topology}"
                    )
                    if result["error"]:
                        msg += f" ERROR={result['error']}"
                    print(msg)
                    detail_lines.append(msg)

                    tsv_rows.append(
                        "\t".join(
                            [
                                task_name,
                                arch_name,
                                family_name,
                                asm_column,
                                result["function"],
                                result["reference_function"],
                                str(result["insns"]),
                                str(result["blocks"]),
                                str(result["jump"]),
                                str(result["fall"]),
                                str(result["unknown"]),
                                str(result["external"]),
                                result["validation"],
                                topology,
                                result["error"].replace("\t", " ").replace("\n", " "),
                            ]
                        )
                    )

            detail_lines.append("")

        print()

    family_expected = len(tasks) * len(REPOS) * 3
    overall_ok = (
        generated == expected_cfgs
        and parse_failures == 0
        and validation_failures == 0
        and topology_diffs == 0
    )

    header = [
        SUMMARY_TITLE,
        "=" * len(SUMMARY_TITLE),
        f"Split: {args.split}",
        f"Tasks: {len(tasks)}",
        "Provenance families: normal[compiler_asm -> object_asm, program_asm]; "
        "pic[compiler_pic_asm -> pic_object_asm, shared_asm]",
        f"Expected CFGs: {expected_cfgs}",
        f"Generated CFGs: {generated}",
        f"Normal-family expected/generated CFGs: "
        f"{family_expected}/{family_generated['normal']}",
        f"PIC-family expected/generated CFGs: "
        f"{family_expected}/{family_generated['pic']}",
        f"Parse/generation failures: {parse_failures}",
        f"Structural validation failures: {validation_failures}",
        f"Raw topology differences vs same-provenance compiler reference: "
        f"{raw_topology_diffs}",
        "Topology differences vs same-provenance compiler reference after "
        f"recognized validation-only normalizations: {topology_diffs}",
        f"Recognized RISC-V long-branch expansion trampolines: "
        f"{long_branch_expansions}",
        f"Objdump trailing instructions trimmed by standalone reachability "
        f"boundary recovery: {standalone_boundary_trimmed}",
        f"Informational normal-vs-PIC selected-reference name differences: "
        f"{reference_name_diffs}",
        f"Unresolved indirect jump edges: {total_unknown}",
        f"Direct external jump edges: {total_external}",
        f"OVERALL: {'PASS' if overall_ok else 'REVIEW NEEDED'}",
        "",
    ]

    (out_root / "summary.txt").write_text(
        "\n".join(header + detail_lines) + "\n", encoding="utf-8"
    )
    (out_root / "summary.tsv").write_text(
        "\n".join(tsv_rows) + "\n", encoding="utf-8"
    )
    (out_root / "problems.txt").write_text(
        ("\n".join(problem_lines) + "\n")
        if problem_lines else "No problems detected.\n",
        encoding="utf-8",
    )

    print("=" * 72)
    for line in header[:-1]:
        print(line)
    print(f"\nOutputs: {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="O0")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_validation(args)

if __name__ == "__main__":
    main()
