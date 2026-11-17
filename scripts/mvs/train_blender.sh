#!/usr/bin/env bash
set -euo pipefail

# Direct MVS-only initialization on all Blender scenes, without adaptive gating.
# MVS replaces the loaded baseline cloud; missing MVS caches are errors.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
n_views=${N_VIEWS:-8}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-1}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs/blender${n_views}}

MODE=mvs DATASET=blender \
DENSITY_CONTROL="${DENSITY_CONTROL:-none}" \
N_VIEWS="$n_views" ITERATIONS="$iterations" RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
exec bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh" "$@"
