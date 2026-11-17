#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=no) ]]; then
    echo "Paired evaluation requires a clean committed checkout." >&2
    exit 2
fi
read -r -a modes <<< "${MODES:-dropgaussian mvs}"
read -r -a matrix <<< "${MATRIX:-llff:3 llff:6 llff:9 dtu:3 dtu:6 dtu:9 mipnerf360:12 mipnerf360:24 blender:8}"
output_root=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/final}

for mode in "${modes[@]}"; do
    if [[ "$mode" != dropgaussian && "$mode" != mvs ]]; then
        echo "MODES accepts only dropgaussian and mvs, got: $mode" >&2
        exit 2
    fi
done

for mode in "${modes[@]}"; do
    for entry in "${matrix[@]}"; do
        dataset=${entry%%:*}
        n_views=${entry##*:}
        DATASET="$dataset" N_VIEWS="$n_views" MODE="$mode" \
            OUTPUT_ROOT="$output_root/$mode/${dataset}${n_views}" \
            bash "$script_dir/train_mvsdrop_benchmark.sh" "$@"
    done
done

"${PYTHON_BIN:-python}" "$script_dir/summarize_mvsdrop_matrix.py" \
    --root "$output_root"
