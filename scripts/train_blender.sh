#!/usr/bin/env bash
set -euo pipefail

# Original DropGaussian training on the fixed NeRF-Synthetic sparse-view split.
# The original trainer uses seed=0; it does not accept --seed or --init_mode.
# Examples:
#   bash scripts/train_blender.sh
#   SCENES="lego chair" GPU_ID=1 bash scripts/train_blender.sh
#   DRY_RUN=1 bash scripts/train_blender.sh

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/nerf_synthetic}
n_views=${N_VIEWS:-8}
resolution=${RESOLUTION:-1}
iterations=${ITERATIONS:-10000}
gpu_id=${GPU_ID:-0}
port=${PORT:-7200}
output_root=${OUTPUT_ROOT:-$repo_root/output/raw/blender${n_views}}
dry_run=${DRY_RUN:-0}

fail() { echo "$*" >&2; exit 2; }
if (( $# )); then
    fail "Configure this script with environment variables; extra arguments are not supported."
fi
[[ ${SEED:-0} == 0 ]] || fail "The original trainer fixes seed=0; SEED=${SEED} is unsupported."
[[ $n_views =~ ^[1-8]$ ]] || fail "N_VIEWS must be in [1, 8]; got: $n_views"
[[ $resolution =~ ^[1-9][0-9]*$ ]] || fail "RESOLUTION must be a positive integer."
# The unchanged renderer uses dropout=0.2*iteration/10000, independently of
# ITERATIONS. At 50000 every Gaussian is dropped; larger values are invalid.
[[ $iterations =~ ^[1-9][0-9]{0,4}$ ]] && (( iterations < 50000 )) \
    || fail "ITERATIONS must be in [1, 49999]; the original default is 10000."
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) \
    || fail "PORT must be in [1, 65535]."
[[ $dry_run == 0 || $dry_run == 1 ]] || fail "DRY_RUN must be 0 or 1."
command -v "$python_bin" >/dev/null 2>&1 || fail "Python executable not found: $python_bin"

default_scenes=(chair drums ficus hotdog lego materials mic ship)
if [[ -n ${SCENES:-} ]]; then
    read -r -a scenes <<< "$SCENES"
else
    scenes=("${default_scenes[@]}")
fi
(( ${#scenes[@]} )) || fail "SCENES must contain at least one scene."

# Check every scene before starting the training batch. The loader selects
# the fixed training views and every eighth test frame from these JSON files.
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
(( ! preflight_failed )) || fail "Blender preflight failed; no training was started."

run_logged() {
    local log_path=$1
    shift
    if (( dry_run )); then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@" 2>&1 | tee "$log_path"
    fi
}

echo "Blender: ${n_views} views, ${iterations} iterations, original seed=0."
for scene in "${scenes[@]}"; do
    source_dir="$data_root/$scene"
    run_dir="$output_root/$scene"
    if (( ! dry_run )); then
        mkdir -p "$run_dir"
    fi

    echo "Training Blender/$scene..."
    run_logged "$run_dir/train.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/train.py" \
        -s "$source_dir" -m "$run_dir" \
        --dataset_name blender --eval -r "$resolution" \
        --n_views "$n_views" --rand_pcd --white_background \
        --iterations "$iterations" --test_iterations "$iterations" \
        --save_iterations "$iterations" --port "$port"

    echo "Rendering Blender/$scene..."
    run_logged "$run_dir/render_${iterations}.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/render.py" \
        -m "$run_dir" -r "$resolution" --iteration "$iterations" --skip_train
done

# metric.py averages all scene directories in OUTPUT_ROOT. Use a separate
# OUTPUT_ROOT for each setting so that prior experiments are not mixed in.
if (( dry_run )); then
    printf '%q ' "$python_bin" "$repo_root/metric.py" --path "$output_root" --iteration "$iterations"
    printf '\n'
else
    "$python_bin" "$repo_root/metric.py" --path "$output_root" --iteration "$iterations"
fi
