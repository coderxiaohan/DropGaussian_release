#!/usr/bin/env bash
set -euo pipefail

# Scene-adaptive MVSInit-DG on Mip-NeRF360.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
n_views=${N_VIEWS:-24}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-8}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvsinit/mipnerf360${n_views}}

MODE=mvsdrop \
DATASET=mipnerf360 \
N_VIEWS="$n_views" \
ITERATIONS="$iterations" \
RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
bash "$repo_root/scripts/train_mvsdrop_benchmark.sh" "$@" \
    --init_mode mvs_anywhere \
    --mvs_init_strategy auto
