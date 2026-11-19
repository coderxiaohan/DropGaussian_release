#!/usr/bin/env bash
set -euo pipefail

# Original DropGaussian initialization + existing FastGS-inspired VCD/VCP.
# Both the mode and final init flag are pinned: extra arguments cannot load
# MVSAnywhere, even if a cache/strategy or a different MODE is inherited.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset=${1:?A dataset is required}
shift
case "$dataset" in
    llff|dtu|mipnerf360|blender) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        printf '%s\n' \
            "Usage: bash scripts/dropgaussian_vcd_vcp/train_${dataset}.sh [extra train.py arguments]" \
            '' \
            'Original DropGaussian initialization + existing FastGS-inspired VCD/VCP.' \
            'No MVSAnywhere: no learned MVS cache, adaptive gate, or residual initialization.' \
            'Not an exact reproduction of upstream FastGS VCD/VCP.' \
            'Forces MODE=dropgaussian and --init_mode colmap_dense after user arguments.' \
            'Uses the original COLMAP-based point cloud, or random points for Blender.' \
            'Original dataset/SfM assets are still required; MVSAnywhere caches are not.' \
            '' \
            'Full configured scene lists; default views LLFF=3, DTU=3, Mip-NeRF360=12, Blender=8.' \
            'Same resolutions, 10000 iterations, and train -> render -> evaluate -> summarize workflow.' \
            'DENSITY_CONTROL: none | vcd | vcp | vcd_vcp (default: vcd_vcp).' \
            'Overrides: N_VIEWS, GPU_ID, DATA_ROOT, SCENES, ITERATIONS, RESOLUTION,' \
            'PYTHON_BIN, OUTPUT_ROOT, SEED, PORT, DTU_MASK_ROOT.' \
            'Default output: output/dropgaussian_<DENSITY_CONTROL>/<dataset><views>.' \
            'Every run requires a fresh or empty OUTPUT_ROOT; nonempty settings are rejected before any writes.'
        exit 0
    fi
done

MODE=dropgaussian \
exec bash "$script_dir/../vcd_vcp/_run.sh" "$dataset" "$@" --init_mode colmap_dense
