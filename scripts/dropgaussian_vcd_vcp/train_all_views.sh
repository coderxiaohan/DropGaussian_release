#!/usr/bin/env bash
set -euo pipefail

# Original initialization + VCD/VCP, all nine settings without MVSAnywhere.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            printf '%s\n' \
                'Usage: bash scripts/dropgaussian_vcd_vcp/train_all_views.sh [--dry-run] [train.py arguments...]' \
                '' \
                'Sequential original DropGaussian initialization + existing FastGS-inspired VCD/VCP.' \
                'No MVSAnywhere: forces MODE=dropgaussian and init_mode=colmap_dense.' \
                'Original COLMAP-based clouds or Blender random points; no learned MVS caches required.' \
                'LLFF 3/6/9; DTU 3/6/9; Mip-NeRF360 12/24; Blender 8.' \
                'All configured scenes, one run per scene/setting (91 runs by default).' \
                '' \
                'GPU priority: per-setting > per-dataset > GPU_ID > 0.' \
                'Per-setting: GPU_LLFF_3/6/9, GPU_DTU_3/6/9,' \
                '             GPU_MIPNERF360_12/24, GPU_BLENDER_8.' \
                'Per-dataset: GPU_LLFF, GPU_DTU, GPU_MIPNERF360, GPU_BLENDER.' \
                'Each GPU value must be one non-negative integer.' \
                'DENSITY_CONTROL: none | vcd | vcp | vcd_vcp (default: vcd_vcp).' \
                'OUTPUT_ROOT is a parent; each setting adds <dataset><views>.' \
                'Default output: output/dropgaussian_<DENSITY_CONTROL>/<dataset><views>.' \
                'Inherited N_VIEWS is replaced; CLI --n_views overrides are rejected.' \
                'Use individual scripts for custom views, DATA_ROOT or SCENES.' \
                '--dry-run prints commands only and writes no outputs.' \
                'Execution is serial; a failed setting stops the batch. No jobs are backgrounded.'
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
    command=(env MODE=dropgaussian "N_VIEWS=$n_views" "GPU_ID=$gpu_id")
    if [[ -n "${OUTPUT_ROOT:-}" ]]; then
        command+=("OUTPUT_ROOT=${OUTPUT_ROOT%/}/${dataset}${n_views}")
    fi
    command+=(bash "$script_dir/train_${dataset}.sh" "$@")
    if [[ "$dry_run" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        printf 'Training DropGaussian + VCD/VCP %s: N_VIEWS=%s GPU_ID=%s\n' "$dataset" "$n_views" "$gpu_id"
        "${command[@]}"
    fi
done
