#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ROOT="${1:-$PROJECT_ROOT/validation/v14}"
mkdir -p "$ROOT"

run_one() {
    local label="$1"
    shift

    echo
    echo "========================================================================"
    echo "RUNNING: $label"
    echo "========================================================================"

    "$@" 2>&1 | tee "$ROOT/${label}.console.log"
}

run_one humaneval_O0 \
    python "$SCRIPT_DIR/run_cfg_humaneval_v14.py" \
        --split O0 \
        --out "$ROOT/humaneval_O0"

run_one humaneval_O2 \
    python "$SCRIPT_DIR/run_cfg_humaneval_v14.py" \
        --split O2 \
        --out "$ROOT/humaneval_O2"

run_one mceval_O0 \
    python "$SCRIPT_DIR/run_cfg_mceval_v14.py" \
        --split O0 \
        --out "$ROOT/mceval_O0"

run_one mceval_O2 \
    python "$SCRIPT_DIR/run_cfg_mceval_v14.py" \
        --split O2 \
        --out "$ROOT/mceval_O2"

run_one bringup_O0 \
    python "$SCRIPT_DIR/run_cfg_bringup_v14.py" \
        --split O0 \
        --out "$ROOT/bringup_O0"

run_one bringup_O2 \
    python "$SCRIPT_DIR/run_cfg_bringup_v14.py" \
        --split O2 \
        --out "$ROOT/bringup_O2"

echo
echo "========================================================================"
echo "FINAL SUMMARIES"
echo "========================================================================"

for summary in \
    "$ROOT/humaneval_O0/summary.txt" \
    "$ROOT/humaneval_O2/summary.txt" \
    "$ROOT/mceval_O0/summary.txt" \
    "$ROOT/mceval_O2/summary.txt" \
    "$ROOT/bringup_O0/summary.txt" \
    "$ROOT/bringup_O2/summary.txt"
do
    echo
    echo "----- $summary -----"
    grep -E \
        '^(Split:|Tasks:|Provenance families:|Expected (CFGs|bundle files|function CFGs):|Generated (CFGs|bundle files|function CFGs):|Normal-family expected/generated|PIC-family expected/generated|Parse/generation failures:|Structural validation failures:|Raw .*topology differences|Topology differences|Function topology differences|Recognized RISC-V|Objdump trailing instructions|AArch64 Linux shared-object trailing instructions|Recognized AArch64 Linux shared-object linked-only terminal success branches|Task/ISA pairs|Functions present only|Informational normal-vs-PIC|Unresolved indirect|Direct external|OVERALL:)' \
        "$summary" || true
done
