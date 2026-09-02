#!/usr/bin/env bash
set -euo pipefail

# Resumable paired evaluation. Run one seed per GPU/process, for example:
#   SEEDS=1 GPU_ID=1 bash scripts/run_mvsdrop_multiseed.sh
#   SEEDS=2 GPU_ID=2 bash scripts/run_mvsdrop_multiseed.sh

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=${OUTPUT_ROOT:-$repo_root/output/mvsdrop/multiseed}
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a matrix <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
read -r -a modes <<< "${MODES:-dropgaussian mvsdrop}"

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
            echo "Paired matrix: seed=$seed setting=$entry mode=$mode"
            DATASET="$dataset" N_VIEWS="$n_views" MODE="$mode" \
                SEED="$seed" MASK_SEED="$seed" GPU_ID="${GPU_ID:-0}" \
                OUTPUT_ROOT="$run_root" \
                bash "$repo_root/scripts/train_mvsdrop_benchmark.sh" "$@"
        done
    done
done

"${PYTHON_BIN:-python}" "$repo_root/scripts/summarize_mvsdrop_multiseed.py" \
    --root "$output_root" --seeds "${seeds[@]}"
