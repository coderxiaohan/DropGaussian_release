#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the original DropGaussian initializer on the StereoGS DTU
# sparse-view protocol. This script always uses COLMAP and never enables MVSInit.
#
# Examples:
#   bash scripts/train_dtu.sh
#   N_VIEWS=3 SCENES="scan21 scan30" GPU_ID=1 bash scripts/train_dtu.sh

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/DTU_colmap}
mask_root=${DTU_MASK_ROOT:-$data_root/idrmasks}
n_views=${N_VIEWS:-3}
resolution=${RESOLUTION:-4}
iterations=${ITERATIONS:-10000}
seed=${SEED:-0}
gpu_id=${GPU_ID:-0}
port=${PORT:-7100}
output_root=${OUTPUT_ROOT:-$repo_root/output/raw/dtu${n_views}}

default_scenes=(
    scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41
    scan45 scan55 scan63 scan82 scan103 scan110 scan114
)
if [[ -n ${SCENES:-} ]]; then
    read -r -a scenes <<< "$SCENES"
else
    scenes=("${default_scenes[@]}")
fi

case "$n_views" in
    3|6|9) ;;
    *)
        echo "N_VIEWS must be one of 3, 6, or 9; got: $n_views" >&2
        exit 2
        ;;
esac

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi

# Validate all scenes before starting a long training batch. DTU uses the
# COLMAP sparse cloud as the DropGaussian baseline and IDR masks for metrics.
preflight_failed=0
for requested_scene in "${scenes[@]}"; do
    scene=${requested_scene%_colmap}
    source_dir="$data_root/${scene}_colmap"
    if [[ ! -d "$source_dir/sparse/0" ]]; then
        echo "Missing COLMAP camera directory: $source_dir/sparse/0" >&2
        preflight_failed=1
    fi
    if [[ ! -d "$source_dir/images" ]]; then
        echo "Missing DTU image directory: $source_dir/images" >&2
        preflight_failed=1
    fi
    if [[ ! -f "$source_dir/poses_bounds.npy" ]]; then
        echo "Missing DTU camera bounds: $source_dir/poses_bounds.npy" >&2
        preflight_failed=1
    fi
    if [[ ! -f "$source_dir/sparse/0/points3D.ply" \
          && ! -f "$source_dir/sparse/0/points3D.bin" ]]; then
        echo "Missing COLMAP point cloud below: $source_dir/sparse/0" >&2
        preflight_failed=1
    fi
    if [[ ! -d "$mask_root/$scene" ]]; then
        echo "Missing DTU IDR masks: $mask_root/$scene" >&2
        preflight_failed=1
    fi
done
if (( preflight_failed )); then
    echo "DTU preflight failed; no training was started." >&2
    exit 2
fi

mkdir -p "$output_root"
for requested_scene in "${scenes[@]}"; do
    scene=${requested_scene%_colmap}
    source_dir="$data_root/${scene}_colmap"
    run_dir="$output_root/$scene"
    mkdir -p "$run_dir"

    echo "Training DTU/$scene (${n_views} views)..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/train.py" \
        -s "$source_dir" \
        -m "$run_dir" \
        --dataset_name dtu \
        --eval -r "$resolution" \
        --n_views "$n_views" \
        --iterations "$iterations" \
        --test_iterations "$iterations" \
        --save_iterations "$iterations" \
        --seed "$seed" \
        --port "$port" \
        "$@" \
        --init_mode colmap_dense \
        2>&1 | tee "$run_dir/train.log"

    echo "Rendering DTU/$scene..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/render.py" \
        -m "$run_dir" \
        -r "$resolution" \
        --iteration "$iterations" \
        --skip_train 2>&1 | tee "$run_dir/render_${iterations}.log"

    echo "Evaluating DTU/$scene with IDR foreground masks..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" \
        "$repo_root/scripts/evaluate_dtu.py" \
        --model-path "$run_dir" \
        --mask-root "$mask_root" \
        --scene "$scene" \
        --iteration "$iterations" \
        2>&1 | tee "$run_dir/evaluate_${iterations}.log"
done

"$python_bin" "$repo_root/scripts/summarize_mvsdrop_setting.py" \
    --experiment-root "$output_root" \
    --iteration "$iterations"
