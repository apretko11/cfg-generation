#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

ROOT="${1:-$PROJECT_ROOT/validation/v14}"
OUT="$ROOT/CFG_SPOTCHECK_72_V14.txt"

ARCHES=(
    x86_linux
    arm_linux
    arm_mac
    riscv_linux
)

# Keep representations in provenance order.
REPS=(
    compiler_asm
    object_asm
    program_asm
    compiler_pic_asm
    pic_object_asm
    shared_asm
)

# Pick three tasks deterministically from each dataset:
# alphabetically first, middle, last.
select_three_tasks() {
    local split_dir="$1"

    mapfile -t tasks < <(
        find "$split_dir" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -printf '%f\n' |
        sort
    )

    local n="${#tasks[@]}"

    if (( n < 3 )); then
        echo "ERROR: fewer than 3 task directories in $split_dir" >&2
        exit 1
    fi

    local middle=$(( n / 2 ))

    printf '%s\n' \
        "${tasks[0]}" \
        "${tasks[$middle]}" \
        "${tasks[$((n - 1))]}"
}

# Start fresh.
: > "$OUT"

{
    echo "CFG V14 MANUAL SPOT CHECK"
    echo "========================="
    echo
    echo "Validation root:"
    echo "  $ROOT"
    echo
    echo "Sampling policy:"
    echo "  3 tasks per dataset: alphabetically first, middle, last"
    echo "  2 optimization levels: O0, O2"
    echo "  4 ISAs/platforms"
    echo "  all 6 representations shown for every check"
    echo
    echo "Expected spot checks:"
    echo "  3 datasets x 2 optimization levels x 3 tasks x 4 ISAs = 72"
    echo
    echo "Representation order:"
    echo "  NORMAL: compiler_asm -> object_asm -> program_asm"
    echo "  PIC:    compiler_pic_asm -> pic_object_asm -> shared_asm"
    echo
} >> "$OUT"

spotcheck_count=0

for dataset in humaneval mceval bringup; do

    # Select tasks from O0, then require those exact same task names in O2.
    mapfile -t selected_tasks < <(
        select_three_tasks "$ROOT/${dataset}_O0"
    )

    {
        echo
        echo
        echo "######################################################################"
        echo "DATASET: $dataset"
        echo "SELECTED TASKS:"
        printf '  %s\n' "${selected_tasks[@]}"
        echo "######################################################################"
    } >> "$OUT"

    for opt in O0 O2; do
        split_dir="$ROOT/${dataset}_${opt}"

        {
            echo
            echo
            echo "======================================================================"
            echo "DATASET: $dataset"
            echo "OPTIMIZATION: $opt"
            echo "======================================================================"
        } >> "$OUT"

        for task in "${selected_tasks[@]}"; do

            if [[ ! -d "$split_dir/$task" ]]; then
                echo "ERROR: missing task directory: $split_dir/$task" >&2
                exit 1
            fi

            for arch in "${ARCHES[@]}"; do
                arch_dir="$split_dir/$task/$arch"

                if [[ ! -d "$arch_dir" ]]; then
                    echo "ERROR: missing arch directory: $arch_dir" >&2
                    exit 1
                fi

                spotcheck_count=$((spotcheck_count + 1))

                {
                    echo
                    echo
                    echo "######################################################################"
                    printf 'SPOT CHECK %02d / 72\n' "$spotcheck_count"
                    echo "Dataset:      $dataset"
                    echo "Optimization: $opt"
                    echo "Task:         $task"
                    echo "Architecture: $arch"
                    echo "######################################################################"
                } >> "$OUT"

                for rep in "${REPS[@]}"; do
                    cfg="$arch_dir/${rep}.cfg.txt"

                    if [[ ! -f "$cfg" ]]; then
                        echo "ERROR: missing CFG file: $cfg" >&2
                        exit 1
                    fi

                    {
                        echo
                        echo "----------------------------------------------------------------------"
                        echo "REPRESENTATION: $rep"
                        echo "FILE: $cfg"
                        echo "----------------------------------------------------------------------"
                        cat "$cfg"
                        echo
                    } >> "$OUT"
                done
            done
        done
    done
done

{
    echo
    echo
    echo "======================================================================"
    echo "SPOT CHECK GENERATION COMPLETE"
    echo "======================================================================"
    echo "Spot checks generated: $spotcheck_count"
    echo "Expected:              72"
} >> "$OUT"

if (( spotcheck_count != 72 )); then
    echo "ERROR: expected 72 spot checks, generated $spotcheck_count" >&2
    exit 1
fi

echo
echo "PASS: generated $spotcheck_count spot checks"
echo "Output:"
echo "  $OUT"
echo
echo "Size:"
ls -lh "$OUT"
echo
echo "Selected task summary:"
grep -A3 '^SELECTED TASKS:' "$OUT"
