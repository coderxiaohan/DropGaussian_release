#!/usr/bin/env bash
set -euo pipefail

# Direct MVS-only initialization on all configured Mip-NeRF360 scenes.
# No adaptive reliability gate or confidence-conditioned attributes are used.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
n_views=${N_VIEWS:-12}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-8}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs/mipnerf360${n_views}}

MODE=mvs DATASET=mipnerf360 \
DENSITY_CONTROL="${DENSITY_CONTROL:-none}" \
N_VIEWS="$n_views" ITERATIONS="$iterations" RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
exec bash "$repo_root/scripts/all/train_mvsdrop_benchmark.sh" "$@"
