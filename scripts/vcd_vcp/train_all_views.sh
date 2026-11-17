#!/usr/bin/env bash
set -euo pipefail

# Nine existing sparse-view settings, serial dispatch with explicit GPU routing.
# This script schedules settings, not distributed or simultaneous training.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            printf '%s\n' \
                'Usage: bash scripts/vcd_vcp/train_all_views.sh [--dry-run] [train.py arguments...]' \
                '' \
                'Runs all configured scenes in nine sparse-view settings, sequentially:' \
                'LLFF 3/6/9; DTU 3/6/9; Mip-NeRF360 12/24; Blender 8.' \
                'Default: direct MVS initialization + VCD/VCP, one run per scene/setting.' \
                '' \
                'GPU priority: per-setting > per-dataset > GPU_ID > 0.' \
                'Per-setting: GPU_LLFF_3, GPU_LLFF_6, GPU_LLFF_9,' \
                '             GPU_DTU_3, GPU_DTU_6, GPU_DTU_9,' \
                '             GPU_MIPNERF360_12, GPU_MIPNERF360_24, GPU_BLENDER_8.' \
                'Per-dataset: GPU_LLFF, GPU_DTU, GPU_MIPNERF360, GPU_BLENDER.' \
                'Each GPU value must be one non-negative integer; this is not multi-GPU DDP.' \
                '' \
                'OUTPUT_ROOT is an optional parent: each setting adds <dataset><views>.' \
                'When unset, existing wrappers keep output/mvs_vcd_vcp/<dataset><views>.' \
                'MODE/DENSITY_CONTROL, PYTHON_BIN, ITERATIONS and other wrapper options propagate.' \
                'Inherited N_VIEWS is replaced by each scheduled setting.' \
                'CLI --n_views overrides are rejected; use individual scripts for custom view counts.' \
                'Do not set one DATA_ROOT/SCENES/MVS_INIT_PATH for a mixed-dataset batch.' \
                'Use the individual dataset scripts for dataset-specific overrides.' \
                '--dry-run prints commands only: no training, output writes, or GPU allocation.' \
                'A failed setting stops subsequent settings. No jobs are backgrounded.' \
                '' \
                'Example: GPU_LLFF=0 GPU_DTU=1 GPU_MIPNERF360=2 GPU_BLENDER=3 bash scripts/vcd_vcp/train_all_views.sh --dry-run'
            exit 0
            ;;
        --) shift; break ;;
        *) break ;;
    esac
done

for argument in "$@"; do
    if [[ "$argument" == "--n_views" || "$argument" == --n_views=* ]]; then
        echo "--n_views cannot override the nine-setting matrix; use N_VIEWS with an individual dataset script." >&2
        exit 2
    fi
done

settings=(llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8)
gpu_ids=()

# Validate the entire GPU mapping before the first potentially expensive run.
for setting in "${settings[@]}"; do
    dataset=${setting%:*}
    n_views=${setting#*:}
    setting_gpu_variable="GPU_${dataset^^}_${n_views}"
    dataset_gpu_variable="GPU_${dataset^^}"
    gpu_id=${!setting_gpu_variable:-${!dataset_gpu_variable:-${GPU_ID:-0}}}
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU for $setting: '$gpu_id'; use one non-negative integer GPU ID." >&2
        exit 2
    fi
    gpu_ids+=("$gpu_id")
done

for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    dataset=${setting%:*}
    n_views=${setting#*:}
    gpu_id=${gpu_ids[$index]}
    command=(env "N_VIEWS=$n_views" "GPU_ID=$gpu_id")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
        command+=("OUTPUT_ROOT=${OUTPUT_ROOT%/}/${dataset}${n_views}")
    fi
    command+=(bash "$script_dir/train_${dataset}.sh" "$@")
    if [[ "$dry_run" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf 'Training %s: N_VIEWS=%s GPU_ID=%s\n' "$dataset" "$n_views" "$gpu_id"
        "${command[@]}"
    fi
done
