#!/usr/bin/env bash
set -euo pipefail

# Direct MVS + VCD/VCP on NeRF-Synthetic (Blender), exactly as on real data.
# No reliability gate or fallback initializer is available in this branch.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
n_views=${N_VIEWS:-8}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-1}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/blender${n_views}}

MODE=mvs \
DATASET=blender \
N_VIEWS="$n_views" \
ITERATIONS="$iterations" \
RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh" "$@" \
    --init_mode mvs_anywhere \
    --mvs_init_strategy replace
