#!/usr/bin/env bash
set -euo pipefail

# Only change the MVS point-cloud geometry before the existing initializer.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
dataset=${1:?A dataset is required}
shift

case "$dataset" in
    llff|dtu) default_views=3 ;;
    mipnerf360) default_views=12 ;;
    blender) default_views=8 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

geometry_mode=${MVS_GEOMETRY_CORRECTION:-tps}
case "$geometry_mode" in
    none|similarity|tps|tps_consistent|tps_local) ;;
    *) echo "MVS_GEOMETRY_CORRECTION must be none, similarity, tps, tps_consistent, or tps_local." >&2; exit 2 ;;
esac

dry_run=${DRY_RUN:-0}
train_args=()
while (( $# )); do
    argument_name=${1%%=*}
    case "$argument_name" in
        -m*) echo "Use OUTPUT_ROOT, not $argument_name; this wrapper owns per-mode model paths." >&2; exit 2 ;;
        -s*) echo "Use DATA_ROOT and SCENES, not $argument_name; this wrapper owns source paths." >&2; exit 2 ;;
        -r*) echo "Use RESOLUTION, not $argument_name; training and evaluation must agree." >&2; exit 2 ;;
        --*)
            # argparse permits long-option abbreviations, so checking only
            # --model_path would leave --model able to bypass output isolation.
            for controlled in \
                --model_path:OUTPUT_ROOT --source_path:DATA_ROOT/SCENES \
                --iterations:ITERATIONS --n_views:N_VIEWS \
                --dataset_name:dataset-specific-script --resolution:RESOLUTION; do
                reserved=${controlled%%:*}
                replacement=${controlled#*:}
                if [[ "$reserved" == "$argument_name"* && "$argument_name" != "--n_views" ]]; then
                    echo "Use $replacement, not $argument_name; this wrapper owns dataset, output and evaluation settings." >&2
                    exit 2
                fi
            done
            ;;
    esac
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/twings/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
                '' \
                'TWINGS-inspired MVS geometry correction; existing DropGaussian and VCD/VCP unchanged.' \
                'MVS_GEOMETRY_CORRECTION: none | similarity | tps | tps_consistent | tps_local (default: tps).' \
                'Forces direct MVS replacement and density_control=vcd_vcp.' \
                'MVS_GEOMETRY_CORRECTION overrides any --mvs_geometry_correction argument.' \
                'Overrides: N_VIEWS, GPU_ID, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION,' \
                'PYTHON_BIN, OUTPUT_ROOT, MVS_INIT_PATH, MVS_INIT_MAX_POINTS, DTU_MASK_ROOT.' \
                'Optional: MVS_GEOMETRY_MAX_CONTROLS, MVS_GEOMETRY_SMOOTHING.' \
                'Use environment overrides, not CLI source/model/dataset/iteration/resolution options.' \
                'OUTPUT_ROOT is a parent: <root>/<geometry_mode>/<dataset><views>.' \
                'Default parent: <repo>/output/twings. Change it when changing other settings.' \
                'Actual runs require an empty/new setting directory to avoid reusing incompatible results.' \
                'DRY_RUN=1 or --dry-run prints dispatch only, without data checks or output writes.' \
                'Train -> render -> evaluate -> summarize uses the existing common runner.'
            exit 0
            ;;
        --mvs_geometry_correction|--density_control|--n_views)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == "--n_views" ]]; then
                echo "Use N_VIEWS, not --n_views, so the output setting matches the training views." >&2
                exit 2
            fi
            shift 2
            ;;
        --mvs_geometry_correction=*|--density_control=*) shift ;;
        --n_views=*)
            echo "Use N_VIEWS, not --n_views, so the output setting matches the training views." >&2
            exit 2
            ;;
        *) train_args+=("$1"); shift ;;
    esac
done

n_views=${N_VIEWS:-$default_views}
gpu_id=${GPU_ID:-0}
if ! [[ "$n_views" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_VIEWS must be one positive integer." >&2
    exit 2
fi
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one non-negative integer." >&2
    exit 2
fi
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

# Explicit control options remain optional; core defaults own the method.
if [[ -n "${MVS_GEOMETRY_MAX_CONTROLS:-}" ]]; then
    train_args+=(--mvs_geometry_max_controls "$MVS_GEOMETRY_MAX_CONTROLS")
fi
if [[ -n "${MVS_GEOMETRY_SMOOTHING:-}" ]]; then
    train_args+=(--mvs_geometry_smoothing "$MVS_GEOMETRY_SMOOTHING")
fi
output_parent=${OUTPUT_ROOT:-$repo_root/output/twings}
setting_output="${output_parent%/}/${geometry_mode}/${dataset}${n_views}"
if [[ "$dry_run" == 0 && ( -e "$setting_output" || -L "$setting_output" ) ]]; then
    if [[ ! -d "$setting_output" ]]; then
        echo "Output setting path already exists and is not a directory: $setting_output" >&2
        exit 2
    fi
    shopt -s nullglob dotglob
    output_entries=("$setting_output"/*)
    if (( ${#output_entries[@]} )); then
        echo "Output setting directory is nonempty: $setting_output" >&2
        echo "Use a fresh OUTPUT_ROOT to avoid silently reusing results from different geometry settings." >&2
        exit 2
    fi
fi
command=(env MODE=mvs MVS_INIT_STRATEGY=replace DENSITY_CONTROL=vcd_vcp
    "N_VIEWS=$n_views" "GPU_ID=$gpu_id"
    "OUTPUT_ROOT=$setting_output"
    bash "$script_dir/../mvs_vcd_vcp/train_${dataset}.sh"
    "${train_args[@]}" --mvs_geometry_correction "$geometry_mode")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
