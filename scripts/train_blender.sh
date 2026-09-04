#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the original DropGaussian initializer on NeRF-Synthetic.
# This script always uses the random baseline cloud and never enables MVSInit.
#
# Examples:
#   bash scripts/train_blender.sh
#   SCENES="lego chair" GPU_ID=1 bash scripts/train_blender.sh

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/nerf_synthetic}
n_views=${N_VIEWS:-8}
resolution=${RESOLUTION:-1}
iterations=${ITERATIONS:-10000}
seed=${SEED:-0}
gpu_id=${GPU_ID:-0}
port=${PORT:-7200}
output_root=${OUTPUT_ROOT:-$repo_root/output/raw/blender${n_views}}

default_scenes=(chair drums ficus hotdog lego materials mic ship)
if [[ -n ${SCENES:-} ]]; then
    read -r -a scenes <<< "$SCENES"
else
    scenes=("${default_scenes[@]}")
fi

if (( n_views < 1 || n_views > 8 )); then
    echo "N_VIEWS must be in [1, 8]; got: $n_views" >&2
    exit 2
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi

# Validate all scenes before starting a long training batch. The baseline uses
# the original random point cloud and the renderer's white background mode.
preflight_failed=0
for scene in "${scenes[@]}"; do
    source_dir="$data_root/$scene"
    for split_file in transforms_train.json transforms_test.json; do
        if [[ ! -f "$source_dir/$split_file" ]]; then
            echo "Missing Blender split file: $source_dir/$split_file" >&2
            preflight_failed=1
        fi
    done
done
if (( preflight_failed )); then
    echo "Blender preflight failed; no training was started." >&2
    exit 2
fi

mkdir -p "$output_root"
for scene in "${scenes[@]}"; do
    source_dir="$data_root/$scene"
    run_dir="$output_root/$scene"
    mkdir -p "$run_dir"

    echo "Training Blender/$scene (${n_views} views)..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/train.py" \
        -s "$source_dir" \
        -m "$run_dir" \
        --dataset_name blender \
        --eval -r "$resolution" \
        --n_views "$n_views" \
        --rand_pcd \
        --white_background \
        --iterations "$iterations" \
        --test_iterations "$iterations" \
        --save_iterations "$iterations" \
        --seed "$seed" \
        --port "$port" \
        "$@" \
        --init_mode colmap_dense \
        2>&1 | tee "$run_dir/train.log"

    echo "Rendering Blender/$scene..."
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/render.py" \
        -m "$run_dir" \
        -r "$resolution" \
        --iteration "$iterations" \
        --skip_train 2>&1 | tee "$run_dir/render_${iterations}.log"
done

"$python_bin" "$repo_root/scripts/summarize_mvsdrop_setting.py" \
    --experiment-root "$output_root" \
    --iteration "$iterations"
