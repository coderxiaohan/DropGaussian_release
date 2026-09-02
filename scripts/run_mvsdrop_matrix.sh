#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
read -r -a modes <<< "${MODES:-dropgaussian mvsdrop}"
read -r -a matrix <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"

for mode in "${modes[@]}"; do
    for entry in "${matrix[@]}"; do
        dataset=${entry%%:*}
        n_views=${entry##*:}
        DATASET="$dataset" N_VIEWS="$n_views" MODE="$mode" \
            bash "$repo_root/scripts/train_mvsdrop_benchmark.sh" "$@"
    done
done

"${PYTHON_BIN:-python}" "$repo_root/scripts/summarize_mvsdrop_matrix.py" \
    --root "$repo_root/output/mvsdrop/final"

