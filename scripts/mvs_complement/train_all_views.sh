#!/usr/bin/env bash
set -euo pipefail

# Run one selected initialization mode over all nine configured view settings.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dry_run=${DRY_RUN:-0}
train_args=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        -h|--help)
            printf '%s\n' \
                'Usage: bash scripts/mvs_complement/train_all_views.sh [--dry-run] [train.py arguments...]' \
                '' \
                'Real datasets: geometry-preserving MVS initialization + unchanged DropGaussian and VCD/VCP (EGD/TBP).' \
                'Blender main method: original DropGaussian random initialization (10000 points), without MVS.' \
                'MVS_COMPLEMENT_MODE selects coverage (default), concat, balanced, or none.' \
                'Blender coverage/concat/balanced ignore MVS budgets/selection settings and need no MVS cache or depth preflight.' \
                'Mode none is the explicit direct-MVS ablation, including on Blender.' \
                'Default MATRIX: llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8.' \
                'All configured scenes: 91 runs total, one per scene/setting; no multi-seed repetitions.' \
                'Real datasets: MVS_COMPLEMENT_TOTAL_POINTS=0 keeps all original geometry; positive budgets may subsample it.' \
                'Real-dataset coverage and balanced need training-view depth caches; random clouds are not geometric anchors.' \
                'Balanced uses coverage source quotas and eligible candidates, but uniform sampling.' \
                'None with a positive total budget applies that budget to the direct-MVS loader.' \
                'Baseline geometry training-view-only provenance is not independently verified.' \
                'GPU priority: GPU_<DATASET>_<VIEWS> > GPU_<DATASET> > GPU_ID > 0.' \
                'OUTPUT_ROOT is a parent; Blender random adds blender_random/blender<views>.' \
                'MVS settings add <mode>/<keep_base or budgetN>/<dataset><views>.' \
                'MATRIX replaces inherited N_VIEWS. Use N_VIEWS with individual dataset scripts.' \
                'DATA_ROOT, SCENES and explicit cache paths should only be used for a compatible subset.' \
                '--dry-run or DRY_RUN=1 prints dispatch only, writes nothing, needs no GPU/data.' \
                'A failed run stops the batch; no processes are backgrounded.'
            exit 0
            ;;
        *) train_args+=("$argument") ;;
    esac
done
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
read -r -a settings <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
if (( ${#settings[@]} == 0 )); then
    echo "MATRIX must contain at least one entry." >&2; exit 2
fi
declare -A seen_settings=()
gpu_ids=()
for setting in "${settings[@]}"; do
    if ! [[ "$setting" =~ ^(llff|dtu|mipnerf360|blender):([1-9][0-9]*)$ ]]; then
        echo "Invalid MATRIX entry: '$setting'; expected dataset:positive_views." >&2; exit 2
    fi
    if [[ -n "${seen_settings[$setting]:-}" ]]; then
        echo "Duplicate MATRIX entry: $setting" >&2; exit 2
    fi
    seen_settings[$setting]=1
    dataset=${setting%:*}
    n_views=${setting#*:}
    setting_gpu_variable="GPU_${dataset^^}_${n_views}"
    dataset_gpu_variable="GPU_${dataset^^}"
    gpu_id=${!setting_gpu_variable:-${!dataset_gpu_variable:-${GPU_ID:-0}}}
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU for $setting: '$gpu_id'; use one non-negative integer." >&2; exit 2
    fi
    gpu_ids+=("$gpu_id")
done
# Preflight every setting before dispatching any training. This validates method
# parameters and owned CLI options, without consulting data or creating outputs.
for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    dataset=${setting%:*}
    n_views=${setting#*:}
    DRY_RUN=1 N_VIEWS="$n_views" GPU_ID="${gpu_ids[$index]}" \
        bash "$script_dir/train_${dataset}.sh" "${train_args[@]}" >/dev/null
done
for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    dataset=${setting%:*}
    n_views=${setting#*:}
    DRY_RUN="$dry_run" N_VIEWS="$n_views" GPU_ID="${gpu_ids[$index]}" \
        bash "$script_dir/train_${dataset}.sh" "${train_args[@]}"
done
