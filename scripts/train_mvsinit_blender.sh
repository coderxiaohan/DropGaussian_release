#!/usr/bin/env bash
set -euo pipefail

# Scene-adaptive MVSInit-DG on NeRF-Synthetic (Blender). The adaptive gate may
# select the exact DropGaussian fallback when the MVS prior is unreliable.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
n_views=${N_VIEWS:-8}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-1}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvsinit/blender${n_views}}

MODE=mvsdrop \
DATASET=blender \
N_VIEWS="$n_views" \
ITERATIONS="$iterations" \
RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
bash "$repo_root/scripts/train_mvsdrop_benchmark.sh" "$@" \
    --init_mode mvs_anywhere \
    --mvs_init_strategy auto
