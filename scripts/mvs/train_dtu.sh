#!/usr/bin/env bash
set -euo pipefail

# Direct MVS-only initialization on all DTU scenes, without adaptive gating.
# Training is followed by rendering, IDR-mask evaluation, and aggregation.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
n_views=${N_VIEWS:-3}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-4}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs/dtu${n_views}}

MODE=mvs DATASET=dtu \
DENSITY_CONTROL="${DENSITY_CONTROL:-none}" \
N_VIEWS="$n_views" ITERATIONS="$iterations" RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
exec bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh" "$@"
