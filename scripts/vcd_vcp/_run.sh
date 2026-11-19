#!/usr/bin/env bash
set -euo pipefail

# Keep density-control experiment dispatch isolated from the original runners.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dataset=${1:?A dataset is required}
shift

case "$dataset" in
    llff) default_views=3; default_resolution=8 ;;
    dtu) default_views=3; default_resolution=4 ;;
    mipnerf360) default_views=12; default_resolution=8 ;;
    blender) default_views=8; default_resolution=1 ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        printf '%s\n' \
            "Usage: bash scripts/vcd_vcp/train_${dataset}.sh [extra train.py arguments]" \
            "" \
            "Runs the existing train -> render -> evaluate -> summarize workflow." \
            "Defaults: direct MVS initialization, VCD + VCP density control," \
            "all configured ${dataset} scenes, ${default_views} views, resolution ${default_resolution}, 10000 iterations." \
            "" \
            "DENSITY_CONTROL: none | vcd | vcp | vcd_vcp (default: vcd_vcp)." \
            "MODE: mvs (default) | dropgaussian. Adaptive initialization is unavailable." \
            "Overrides: N_VIEWS, ITERATIONS, RESOLUTION, DATA_ROOT, SCENES, GPU_ID," \
            "PYTHON_BIN, OUTPUT_ROOT, SEED, PORT, DTU_MASK_ROOT and MVS_INIT_* options." \
            "Default output: output/mvs_<DENSITY_CONTROL>/${dataset}<N_VIEWS>." \
            "MODE=dropgaussian uses the dropgaussian_ prefix instead." \
            "Every run requires a fresh or empty OUTPUT_ROOT; nonempty settings are rejected before any writes." \
            "DENSITY_CONTROL is authoritative over any extra --density_control argument." \
            "" \
            "Example: SCENES=fern DENSITY_CONTROL=vcd bash scripts/vcd_vcp/train_llff.sh"
        exit 0
    fi
done

density_control=${DENSITY_CONTROL:-vcd_vcp}
case "$density_control" in
    none|vcd|vcp|vcd_vcp) ;;
    *) echo "DENSITY_CONTROL must be none, vcd, vcp, or vcd_vcp, got: $density_control" >&2; exit 2 ;;
esac

benchmark_mode=${MODE:-mvs}
case "$benchmark_mode" in
    dropgaussian) method_prefix=dropgaussian ;;
    mvs) method_prefix=mvs ;;
    *) echo "MODE must be mvs or dropgaussian; adaptive mvsdrop is unavailable (got: $benchmark_mode)." >&2; exit 2 ;;
esac

n_views=${N_VIEWS:-$default_views}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-$default_resolution}
output_root=${OUTPUT_ROOT:-$repo_root/output/${method_prefix}_${density_control}/${dataset}${n_views}}

MODE="$benchmark_mode" DATASET="$dataset" \
N_VIEWS="$n_views" ITERATIONS="$iterations" RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
exec bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh" \
    "$@" --density_control "$density_control"
