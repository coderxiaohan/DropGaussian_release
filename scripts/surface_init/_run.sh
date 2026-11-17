#!/usr/bin/env bash
set -euo pipefail

# Only change Gaussian covariance initialization; do not combine with TPS.
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
surface_mode=${MVS_SURFACE_INIT:-oas}
case "$surface_mode" in
    none|oas) ;;
    *) echo "MVS_SURFACE_INIT must be none or oas." >&2; exit 2 ;;
esac
neighbors=${MVS_SURFACE_NEIGHBORS:-32}
if ! [[ "$neighbors" =~ ^[1-9][0-9]*$ ]] || (( neighbors < 3 )); then
    echo "MVS_SURFACE_NEIGHBORS must be an integer >= 3." >&2
    exit 2
fi

dry_run=${DRY_RUN:-0}
train_args=()
while (( $# )); do
    argument_name=${1%%=*}
    case "$argument_name" in
        -m*) echo "Use OUTPUT_ROOT, not $argument_name; this wrapper owns model paths." >&2; exit 2 ;;
        -s*) echo "Use DATA_ROOT and SCENES, not $argument_name; this wrapper owns source paths." >&2; exit 2 ;;
        -r*) echo "Use RESOLUTION, not $argument_name; training and evaluation must agree." >&2; exit 2 ;;
        --mvs_geometry*)
            echo "Geometry correction is fixed to none; TPS cannot be combined with this comparison." >&2
            exit 2
            ;;
        --*)
            # Reject abbreviations too: argparse accepts --model, --iter, etc.
            for controlled in \
                --model_path:OUTPUT_ROOT --source_path:DATA_ROOT/SCENES \
                --iterations:ITERATIONS --n_views:N_VIEWS \
                --dataset_name:dataset-specific-script --resolution:RESOLUTION \
                --mvs_surface_init:MVS_SURFACE_INIT --mvs_surface_neighbors:MVS_SURFACE_NEIGHBORS \
                --mvs_geometry_correction:fixed-none --density_control:fixed-vcd_vcp \
                --init_mode:fixed-mvs_anywhere --mvs_init_strategy:fixed-replace; do
                reserved=${controlled%%:*}
                replacement=${controlled#*:}
                if [[ "$reserved" == "$argument_name"* ]]; then
                    echo "Use $replacement, not $argument_name; this wrapper owns initialization and evaluation settings." >&2
                    exit 2
                fi
            done
            ;;
    esac
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        -h|--help)
            printf '%s\n' \
                "Usage: bash scripts/surface_init/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
                '' \
                'MVS surface covariance initialization; DropGaussian and VCD/VCP unchanged.' \
                'MVS_SURFACE_INIT: none | oas (default: oas).' \
                'MVS_SURFACE_NEIGHBORS: integer >= 3 (default: 32).' \
                'Forces direct MVS replacement, geometry correction=none, density_control=vcd_vcp.' \
                'Overrides: N_VIEWS, GPU_ID, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION,' \
                'PYTHON_BIN, OUTPUT_ROOT, MVS_INIT_PATH, MVS_INIT_MAX_POINTS, DTU_MASK_ROOT.' \
                'Use environment settings, not CLI source/model/dataset/iteration/resolution/method options.' \
                'OUTPUT_ROOT is a parent: <root>/<surface_mode>/<dataset><views>.' \
                'Default parent: <repo>/output/surface_init. Use a fresh parent for changed settings.' \
                'Actual runs require an empty/new setting directory; existing results are never reused.' \
                'DRY_RUN=1 or --dry-run prints dispatch only, without data checks or output writes.' \
                'Train -> render -> evaluate -> summarize uses the existing common runner.'
            exit 0
            ;;
        *) train_args+=("$1"); shift ;;
    esac
done
n_views=${N_VIEWS:-$default_views}
gpu_id=${GPU_ID:-0}
if ! [[ "$n_views" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_VIEWS must be one positive integer." >&2; exit 2
fi
if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one non-negative integer." >&2; exit 2
fi
case "$dry_run" in
    0|1) ;;
    *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
output_parent=${OUTPUT_ROOT:-$repo_root/output/surface_init}
setting_output="${output_parent%/}/${surface_mode}/${dataset}${n_views}"
if [[ "$dry_run" == 0 && ( -e "$setting_output" || -L "$setting_output" ) ]]; then
    if [[ ! -d "$setting_output" ]]; then
        echo "Output setting path already exists and is not a directory: $setting_output" >&2
        exit 2
    fi
    shopt -s nullglob dotglob
    output_entries=("$setting_output"/*)
    if (( ${#output_entries[@]} )); then
        echo "Output setting directory is nonempty: $setting_output" >&2
        echo "Use a fresh OUTPUT_ROOT to avoid reusing results from different initialization settings." >&2
        exit 2
    fi
fi
command=(env MODE=mvs MVS_INIT_STRATEGY=replace DENSITY_CONTROL=vcd_vcp
    "N_VIEWS=$n_views" "GPU_ID=$gpu_id" "OUTPUT_ROOT=$setting_output"
    bash "$script_dir/../mvs_vcd_vcp/train_${dataset}.sh"
    "${train_args[@]}" --mvs_geometry_correction none
    --mvs_surface_init "$surface_mode" --mvs_surface_neighbors "$neighbors")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
