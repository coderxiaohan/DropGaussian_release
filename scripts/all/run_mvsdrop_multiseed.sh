#!/usr/bin/env bash
set -euo pipefail

# Paired evaluation with fresh setting outputs. Run one seed per GPU/process:
#   SEEDS=1 GPU_ID=1 bash scripts/all/run_mvsdrop_multiseed.sh
#   SEEDS=2 GPU_ID=2 bash scripts/all/run_mvsdrop_multiseed.sh
# For one run per method/scene, use scripts/all/run_all.sh instead.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
dry_run=${DRY_RUN:-0}
if [[ "$dry_run" != 0 && "$dry_run" != 1 ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 2
fi
if [[ "$dry_run" == 0 && -n $(git -C "$repo_root" status --porcelain --untracked-files=no) ]]; then
    echo "Formal paired evaluation requires a clean committed checkout." >&2
    echo "Commit or stash tracked changes before starting the benchmark." >&2
    exit 2
fi
protocol_commit=$(git -C "$repo_root" rev-parse HEAD)
echo "Paired-protocol commit: $protocol_commit"

output_root=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/multiseed}
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a matrix <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
read -r -a modes <<< "${MODES:-dropgaussian mvs}"

for mode in "${modes[@]}"; do
    if [[ "$mode" != dropgaussian && "$mode" != mvs ]]; then
        echo "MODES accepts only dropgaussian and mvs, got: $mode" >&2
        exit 2
    fi
done

for seed in "${seeds[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "SEEDS must contain non-negative integers, got: $seed" >&2
        exit 2
    fi
    for entry in "${matrix[@]}"; do
        dataset=${entry%%:*}
        n_views=${entry##*:}
        tag=$dataset
        [[ "$dataset" == "mipnerf360" ]] && tag=mipnerf360
        for mode in "${modes[@]}"; do
            run_root="$output_root/seed_${seed}/${mode}/${tag}${n_views}"
            if [[ "$dry_run" == 1 ]]; then
                printf 'DRY-RUN seed=%s setting=%s mode=%s output=%s\n' \
                    "$seed" "$entry" "$mode" "$run_root"
                continue
            fi
            echo "Paired matrix: seed=$seed setting=$entry mode=$mode"
            DATASET="$dataset" N_VIEWS="$n_views" MODE="$mode" \
                SEED="$seed" MASK_SEED="$seed" GPU_ID="${GPU_ID:-0}" \
                OUTPUT_ROOT="$run_root" \
                bash "$script_dir/train_mvsdrop_benchmark.sh" "$@"
        done
    done
done

if [[ "$dry_run" == 1 ]]; then
    echo "Dry run only: no training, evaluation, or output files were created."
    exit 0
fi

"${PYTHON_BIN:-python}" "$script_dir/summarize_mvsdrop_multiseed.py" \
    --root "$output_root" --seeds "${seeds[@]}"
