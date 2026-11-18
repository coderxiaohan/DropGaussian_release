#!/usr/bin/env bash
set -euo pipefail

# A single ablation variant across the same nine settings as the full method.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
variant=${1:?An ablation variant is required}
shift
case "$variant" in
    baseline|complement_only|egd_only|tbp_only|complement_egd|complement_tbp|egd_tbp) ;;
    *) echo "Unknown ablation variant: $variant" >&2; exit 2 ;;
esac
dry_run=${DRY_RUN:-0}
train_args=()
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/ablation/${variant}/train_all_views.sh [--dry-run] [train.py arguments...]" \
                '' \
                "Fixed module ablation: ${variant}. The full method remains scripts/mvs_complement/." \
                'Default MATRIX: llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8.' \
                'All configured scenes: 91 runs total, one per scene/setting; no multi-seed repetitions.' \
                'Blender uses original random initialization; complementary initialization is inactive there.' \
                'OUTPUT_ROOT is always a common parent: <root>/<variant>/<dataset><views>/<scene>.' \
                'Default parent: <repo>/output/ablation. Nonempty settings are rejected, never reused.' \
                'GPU priority: GPU_<DATASET>_<VIEWS> > GPU_<DATASET> > GPU_ID > 0.' \
                'MATRIX selects a subset and replaces inherited N_VIEWS; individual scripts accept N_VIEWS.' \
                'Use DATA_ROOT, SCENES and explicit MVS cache paths only with compatible settings.' \
                'Execution is serial even when GPU IDs differ; a failed setting stops the batch.' \
                '--dry-run or DRY_RUN=1 prints dispatch commands only, writes nothing, and needs no GPU/data.' \
                "Use train_<dataset>.sh --help for fixed module choices and additional environment settings."
            exit 0
            ;;
        *) train_args+=("$argument") ;;
    esac
done
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
read -r -a settings <<< "${MATRIX-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
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

# Check method parameters and all owned CLI options before any training starts.
for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    DRY_RUN=1 N_VIEWS="${setting#*:}" GPU_ID="${gpu_ids[$index]}" \
        bash "$script_dir/_run.sh" "$variant" "${setting%:*}" "${train_args[@]}" >/dev/null
done
for index in "${!settings[@]}"; do
    setting=${settings[$index]}
    DRY_RUN="$dry_run" N_VIEWS="${setting#*:}" GPU_ID="${gpu_ids[$index]}" \
        bash "$script_dir/_run.sh" "$variant" "${setting%:*}" "${train_args[@]}"
done
