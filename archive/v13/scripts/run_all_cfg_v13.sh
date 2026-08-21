#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-cfg_validation_v13_all}"
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
    python run_cfg_humaneval_v13.py \
        --split O0 \
        --out "$ROOT/humaneval_O0"

run_one humaneval_O2 \
    python run_cfg_humaneval_v13.py \
        --split O2 \
        --out "$ROOT/humaneval_O2"

run_one mceval_O0 \
    python run_cfg_mceval_v13.py \
        --split O0 \
        --out "$ROOT/mceval_O0"

run_one mceval_O2 \
    python run_cfg_mceval_v13.py \
        --split O2 \
        --out "$ROOT/mceval_O2"

run_one bringup_O0 \
    python run_cfg_bringup_v13.py \
        --split O0 \
        --out "$ROOT/bringup_O0"

run_one bringup_O2 \
    python run_cfg_bringup_v13.py \
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
        '^(Split:|Tasks:|Expected (CFGs|bundle files|function CFGs):|Generated (CFGs|bundle files|function CFGs):|Parse/generation failures:|Structural validation failures:|Raw .*topology differences|Topology differences|Function topology differences|Recognized RISC-V|AArch64 Linux shared-object|Task/ISA pairs|Functions present only|Informational normal-vs-PIC|Unresolved indirect|Direct external|OVERALL:)' \
        "$summary" || true
done
