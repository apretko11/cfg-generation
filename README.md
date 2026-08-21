# CFG Generation

Standalone intraprocedural control-flow graph (CFG) extraction for compiler assembly and binary disassembly across multiple ISAs and object formats.

**Current version: V14**

Validated targets:

- x86-64 Linux
- AArch64 Linux
- AArch64 macOS
- RISC-V Linux

Validated input forms:

- compiler-generated assembly (`.s`)
- relocatable-object disassembly (`.o`)
- shared-library disassembly (`.so` / `.dylib`)
- linked-program disassembly

## Core design principle

Production CFG generation is **standalone**:

```text
one assembly/disassembly artifact
        |
        v
artifact-local parsing/recovery
        |
        v
basic blocks + edges
        |
        v
     real CFG
```

The CFG for one artifact must not depend on a paired compiler assembly, object file, shared library, executable, or reference CFG.

Artifact-local metadata such as relocation records may be used.

Cross-representation information is used **only after independent CFG generation**, for validation and comparison.

## CFG scope

The extractor builds a conservative intraprocedural CFG.

- direct in-function branch → internal CFG edge
- direct out-of-function branch → `EXTERNAL_*`
- unresolved indirect jump → `UNKNOWN`
- return/sink → no outgoing edge
- ordinary calls are conservatively treated as returning and do not normally terminate a basic block

The tool intentionally does not attempt full interprocedural call-graph recovery, general jump-table recovery, symbolic execution, or runtime path-feasibility analysis.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── scripts/
│   ├── generate_cfg_updated_v14.py
│   ├── run_cfg_humaneval_v14.py
│   ├── run_cfg_mceval_v14.py
│   ├── run_cfg_bringup_v14.py
│   ├── run_all_cfg_v14.sh
│   ├── make_cfg_spotcheck_v14.sh
│   └── make_cfg_targeted_special_cases_v14.py
├── docs/
│   ├── cfg_validation_methodology.txt
│   └── cfg_v14_design_rationale_and_history.txt
├── validation/
│   ├── v13/
│   └── v14/
└── archive/
    └── v13/
        └── scripts/
```

The repository intentionally does **not** include the benchmark datasets or the large generated per-task CFG trees. Dataset generation is maintained separately.

Under `validation/`, only the **top-level per-version** human-readable reports (`.txt`) and console logs (`.log`) are intended to be tracked. The generated `bringup_O0/`, `bringup_O2/`, `humaneval_*`, and `mceval_*` trees are intentionally excluded.

## Standalone usage

The generator accepts one assembly/disassembly text file at a time.

```bash
python scripts/generate_cfg_updated_v14.py \
    input.s \
    --target x86_linux
```

Explicit targets:

```text
x86_linux
arm_linux
arm_mac
riscv_linux
```

Input format can be detected automatically or specified explicitly:

```bash
python scripts/generate_cfg_updated_v14.py \
    input.objdump.txt \
    --target riscv_linux \
    --format objdump
```

To select one function:

```bash
python scripts/generate_cfg_updated_v14.py \
    input.objdump.txt \
    --target arm_mac \
    --function func0
```

The generated CFG is written to standard output, so it can be redirected:

```bash
python scripts/generate_cfg_updated_v14.py \
    input.objdump.txt \
    --target arm_linux \
    --format objdump \
    > output.cfg.txt
```

For all CLI options:

```bash
python scripts/generate_cfg_updated_v14.py --help
```

## Validation provenance

The six dataset representations form two separate compilation-provenance families.

### Normal

```text
compiler_asm
    ├── object_asm
    └── program_asm
```

### PIC

```text
compiler_pic_asm
    ├── pic_object_asm
    └── shared_asm
```

Validation compares representations only within the correct provenance family.

This matters because the shared library is linked from separately compiled PIC objects and may legitimately differ from the normal compilation, especially under optimization.

## Run one full dataset split

The benchmark runners process an entire dataset split, rather than one standalone assembly file.

### HumanEval

```bash
python scripts/run_cfg_humaneval_v14.py \
    --split O0 \
    --out validation/v14/humaneval_O0
```

Use `--split O2` and a corresponding output directory for the optimized split:

```bash
python scripts/run_cfg_humaneval_v14.py \
    --split O2 \
    --out validation/v14/humaneval_O2
```

### MC-Eval

```bash
python scripts/run_cfg_mceval_v14.py \
    --split O0 \
    --out validation/v14/mceval_O0
```

or:

```bash
python scripts/run_cfg_mceval_v14.py \
    --split O2 \
    --out validation/v14/mceval_O2
```

### Bringup-Bench

```bash
python scripts/run_cfg_bringup_v14.py \
    --split O0 \
    --out validation/v14/bringup_O0
```

or:

```bash
python scripts/run_cfg_bringup_v14.py \
    --split O2 \
    --out validation/v14/bringup_O2
```

These runners load the corresponding benchmark data from Hugging Face, generate CFGs for all four target platforms and all six representations in the selected split, and write the validation outputs under the requested `--out` directory.

## Full validation

Run all six benchmark/split validations with:

```bash
./scripts/run_all_cfg_v14.sh
```

This runs:

- HumanEval O0
- HumanEval O2
- MC-Eval O0
- MC-Eval O2
- Bringup-Bench O0
- Bringup-Bench O2

Default outputs are written under:

```text
validation/v14/
```

The dataset-wide validation runners require the Hugging Face `datasets` Python package and access to the corresponding benchmark repositories.

## Manual audit reports

Generate the deterministic broad spot check with:

```bash
./scripts/make_cfg_spotcheck_v14.sh
```

This produces 72 task/ISA checks covering:

```text
3 datasets
x 2 optimization levels
x 3 deterministic tasks
x 4 targets
= 72 checks
```

with all six representations shown for every check.

Generate targeted unusual cases with:

```bash
python scripts/make_cfg_targeted_special_cases_v14.py
```

The targeted report covers:

1. unresolved indirect jump → `UNKNOWN`
2. direct out-of-function jump → `EXTERNAL`
3. RISC-V long-branch assembler expansion
4. AArch64 standalone reachability-based boundary recovery
5. AArch64 linked-only terminal-success branch retained in the real CFG and handled only during validation
6. legitimate normal-vs-PIC function-set difference

## V14 validation results

All six full V14 validation runs pass.

| Benchmark | O0 | O2 |
|---|---:|---:|
| HumanEval | PASS | PASS |
| MC-Eval | PASS | PASS |
| Bringup-Bench | PASS | PASS |

Aggregate V14 evidence:

- 15,456 representation/bundle outputs
- 36,675 individual function CFGs
- 0 parse/generation failures
- 0 structural validation failures
- 0 unexplained topology differences after approved validation-only normalization
- 484 recognized RISC-V long-branch expansion trampolines
- 11 AArch64 trailing instructions removed by standalone reachability recovery
- 2 AArch64 linked-only terminal-success cases recognized during validation only
- 162 unresolved indirect-jump (`UNKNOWN`) edges
- 2,186 direct external-jump edges
- 6 / 6 full validation runs PASS

## Important V14 behavior

### RISC-V long branches

An assembler may expand a source conditional branch into an inverted conditional branch plus an unconditional-jump trampoline.

The real binary CFG keeps that trampoline. The validation layer recognizes only the strict known transformation when comparing against compiler assembly.

### AArch64 shared-object boundary recovery

V14 no longer uses another representation to determine the end of an AArch64 shared-object function.

For objdump-style input, V14 uses only the current artifact:

1. build a tentative CFG
2. compute entry reachability
3. remove only a trailing suffix proven unreachable
4. refuse to trim if a reachable unresolved indirect jump could target the suffix

### Linked-only terminal-success branches

Two Bringup-Bench O0 AArch64 shared-object cases contain a real linked-only branch to `libtarg_success`.

The emitted CFG **retains**:

```text
JUMP -> EXTERNAL_libtarg_success
```

A narrow Bringup-specific equivalence is applied only during validation; the production CFG is not rewritten.

## Documentation

For the final validation methodology, see:

```text
docs/cfg_validation_methodology.txt
```

For the development history, major fixes, rejected approaches, and the V13 → V14 rationale, see:

```text
docs/cfg_v14_design_rationale_and_history.txt
```

## Version history

### V14 — current

- standalone production CFG generation
- explicit Linux/macOS target distinction
- artifact-local AArch64 shared-object boundary recovery
- validation-only handling of linked-only terminal-success branches
- provenance-aware validation retained
- all six full benchmark/split validations PASS

### V13 — archived

V13 introduced the correct normal/PIC provenance-aware validation model and produced clean validation results, but one AArch64 shared-object boundary-recovery path still used another representation before CFG generation. It is retained under `archive/v13/` for development history and reproducibility.
