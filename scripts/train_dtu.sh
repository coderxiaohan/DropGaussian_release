#!/usr/bin/env bash
set -euo pipefail

# Original DropGaussian training with the StereoGS DTU sparse-view split,
# COLMAP point cloud initialization, and IDR-mask foreground evaluation.
# The original trainer uses seed=0; it does not accept --seed or --init_mode.
# Examples:
#   bash scripts/train_dtu.sh
#   N_VIEWS=3 SCENES="scan21 scan30" GPU_ID=1 bash scripts/train_dtu.sh
#   DRY_RUN=1 bash scripts/train_dtu.sh

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/home/wuxiaohan/wxh/data/DTU_colmap}
mask_root=${DTU_MASK_ROOT:-$data_root/idrmasks}
n_views=${N_VIEWS:-3}
resolution=${RESOLUTION:-4}
iterations=${ITERATIONS:-10000}
gpu_id=${GPU_ID:-0}
port=${PORT:-7100}
output_root=${OUTPUT_ROOT:-$repo_root/output/raw/dtu${n_views}}
dry_run=${DRY_RUN:-0}

fail() { echo "$*" >&2; exit 2; }
if (( $# )); then
    fail "Configure this script with environment variables; extra arguments are not supported."
fi
[[ ${SEED:-0} == 0 ]] || fail "The original trainer fixes seed=0; SEED=${SEED} is unsupported."
case "$n_views" in
    3|6|9) ;;
    *) fail "N_VIEWS must be one of 3, 6, or 9; got: $n_views" ;;
esac
[[ $resolution =~ ^[1-9][0-9]*$ ]] || fail "RESOLUTION must be a positive integer."
# The unchanged renderer uses dropout=0.2*iteration/10000, independently of
# ITERATIONS. At 50000 every Gaussian is dropped; larger values are invalid.
[[ $iterations =~ ^[1-9][0-9]{0,4}$ ]] && (( iterations < 50000 )) \
    || fail "ITERATIONS must be in [1, 49999]; the original default is 10000."
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) \
    || fail "PORT must be in [1, 65535]."
[[ $dry_run == 0 || $dry_run == 1 ]] || fail "DRY_RUN must be 0 or 1."
command -v "$python_bin" >/dev/null 2>&1 || fail "Python executable not found: $python_bin"

default_scenes=(
    scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41
    scan45 scan55 scan63 scan82 scan103 scan110 scan114
)
if [[ -n ${SCENES:-} ]]; then
    read -r -a scenes <<< "$SCENES"
else
    scenes=("${default_scenes[@]}")
fi
(( ${#scenes[@]} )) || fail "SCENES must contain at least one scene."

# Validate all scenes before starting a long batch. The loader accepts either
# binary or text COLMAP cameras, with the original images and camera bounds.
preflight_failed=0
for requested_scene in "${scenes[@]}"; do
    scene=${requested_scene%_colmap}
    source_dir="$data_root/${scene}_colmap"
    if [[ ! -f "$source_dir/sparse/0/cameras.bin" || ! -f "$source_dir/sparse/0/images.bin" ]] \
        && [[ ! -f "$source_dir/sparse/0/cameras.txt" || ! -f "$source_dir/sparse/0/images.txt" ]]; then
        echo "Missing COLMAP cameras/images pair (.bin or .txt): $source_dir/sparse/0" >&2
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
    if [[ ! -f "$source_dir/sparse/0/points3D.ply" && ! -f "$source_dir/sparse/0/points3D.bin" ]]; then
        echo "Missing COLMAP point cloud: $source_dir/sparse/0/points3D.{ply,bin}" >&2
        preflight_failed=1
    fi
    scene_mask_dir="$mask_root/$scene"
    case "$scene" in
        scan40|scan55|scan63|scan110|scan114) scene_mask_dir="$scene_mask_dir/mask" ;;
    esac
    if [[ ! -d "$scene_mask_dir" ]]; then
        echo "Missing DTU IDR masks: $scene_mask_dir" >&2
        preflight_failed=1
    fi
done
(( ! preflight_failed )) || fail "DTU preflight failed; no training was started."

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

echo "DTU: ${n_views} views, ${iterations} iterations, original seed=0."
for requested_scene in "${scenes[@]}"; do
    scene=${requested_scene%_colmap}
    source_dir="$data_root/${scene}_colmap"
    run_dir="$output_root/$scene"
    if (( ! dry_run )); then
        mkdir -p "$run_dir"
    fi

    echo "Training DTU/$scene..."
    run_logged "$run_dir/train.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/train.py" \
        -s "$source_dir" -m "$run_dir" \
        --dataset_name dtu --eval -r "$resolution" --n_views "$n_views" \
        --iterations "$iterations" --test_iterations "$iterations" \
        --save_iterations "$iterations" --port "$port"

    echo "Rendering DTU/$scene..."
    run_logged "$run_dir/render_${iterations}.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/render.py" \
        -m "$run_dir" -r "$resolution" --iteration "$iterations" --skip_train

    echo "Evaluating DTU/$scene with IDR foreground masks..."
    run_logged "$run_dir/evaluate_${iterations}.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$repo_root/scripts/evaluate_dtu.py" \
        --model-path "$run_dir" --mask-root "$mask_root" \
        --scene "$scene" --iteration "$iterations"
done

# metric.py averages all scene directories in OUTPUT_ROOT. Use a separate
# OUTPUT_ROOT for each setting so that prior experiments are not mixed in.
if (( dry_run )); then
    printf '%q ' "$python_bin" "$repo_root/metric.py" --path "$output_root" --iteration "$iterations"
    printf '\n'
else
    "$python_bin" "$repo_root/metric.py" --path "$output_root" --iteration "$iterations"
fi
