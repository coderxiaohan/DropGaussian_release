#!/usr/bin/env bash
set -euo pipefail

# Three initialization ablations with identical downstream training.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dry_run=${DRY_RUN:-0}
train_args=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        -h|--help)
            printf '%s\n' \
                'Usage: bash scripts/twings/run_comparison.sh [--dry-run] [train.py arguments...]' \
                '' \
                'Sequential none / similarity / tps comparison; DropGaussian and VCD/VCP fixed.' \
                'Default MATRIX: llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8.' \
                'All configured scenes: 273 runs total (91 per geometry mode), no multi-seed repetitions.' \
                'CORRECTIONS selects modes, e.g. "none tps_consistent tps_local" for the correspondence/support ablation.' \
                'GPU priority: GPU_<DATASET>_<VIEWS> > GPU_<DATASET> > GPU_ID > 0.' \
                'OUTPUT_ROOT is a parent; each run adds <correction>/<dataset><views>.' \
                'MATRIX replaces inherited N_VIEWS. Use N_VIEWS with individual dataset scripts.' \
                'DATA_ROOT, SCENES and explicit cache paths should only be used for a compatible subset.' \
                '--dry-run or DRY_RUN=1 prints all dispatch commands, writes nothing, and needs no GPU/data.' \
                'A failed run stops the comparison; no processes are backgrounded.'
            exit 0
            ;;
        --n_views|--n_views=*)
            echo "Use MATRIX for comparison views, not --n_views." >&2
            exit 2
            ;;
        *) train_args+=("$argument") ;;
    esac
done
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

read -r -a settings <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
read -r -a corrections <<< "${CORRECTIONS:-none similarity tps}"
if (( ${#settings[@]} == 0 || ${#corrections[@]} == 0 )); then
    echo "MATRIX and CORRECTIONS must each contain at least one entry." >&2
    exit 2
fi
declare -A seen_settings=() seen_corrections=()
gpu_ids=()
for setting in "${settings[@]}"; do
    if ! [[ "$setting" =~ ^(llff|dtu|mipnerf360|blender):([1-9][0-9]*)$ ]]; then
        echo "Invalid MATRIX entry: '$setting'; expected dataset:positive_views." >&2
        exit 2
    fi
    if [[ -n "${seen_settings[$setting]:-}" ]]; then
        echo "Duplicate MATRIX entry: $setting" >&2
        exit 2
    fi
    seen_settings[$setting]=1
    dataset=${setting%:*}
    n_views=${setting#*:}
    setting_gpu_variable="GPU_${dataset^^}_${n_views}"
    dataset_gpu_variable="GPU_${dataset^^}"
    gpu_id=${!setting_gpu_variable:-${!dataset_gpu_variable:-${GPU_ID:-0}}}
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU for $setting: '$gpu_id'; use one non-negative integer." >&2
        exit 2
    fi
    gpu_ids+=("$gpu_id")
done
for correction in "${corrections[@]}"; do
    case "$correction" in
        none|similarity|tps|tps_consistent|tps_local) ;;
        *) echo "Invalid CORRECTIONS entry: '$correction'." >&2; exit 2 ;;
    esac
    if [[ -n "${seen_corrections[$correction]:-}" ]]; then
        echo "Duplicate CORRECTIONS entry: $correction" >&2
        exit 2
    fi
    seen_corrections[$correction]=1
done

for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    dataset=${setting%:*}
    n_views=${setting#*:}
    for correction in "${corrections[@]}"; do
        DRY_RUN="$dry_run" MVS_GEOMETRY_CORRECTION="$correction" \
        N_VIEWS="$n_views" GPU_ID="${gpu_ids[$index]}" \
            bash "$script_dir/train_${dataset}.sh" "${train_args[@]}"
    done
done
