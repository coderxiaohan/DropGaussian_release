#!/usr/bin/env bash
set -euo pipefail

# Scene-adaptive MVSInit-DG on LLFF. Environment variables and extra CLI
# arguments are forwarded to the common four-dataset benchmark runner.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
n_views=${N_VIEWS:-9}
iterations=${ITERATIONS:-10000}
resolution=${RESOLUTION:-8}
output_root=${OUTPUT_ROOT:-$repo_root/output/mvsinit/llff${n_views}}

MODE=mvsdrop \
DATASET=llff \
N_VIEWS="$n_views" \
ITERATIONS="$iterations" \
RESOLUTION="$resolution" \
OUTPUT_ROOT="$output_root" \
bash "$repo_root/scripts/train_mvsdrop_benchmark.sh" "$@" \
    --init_mode mvs_anywhere \
    --mvs_init_strategy auto
