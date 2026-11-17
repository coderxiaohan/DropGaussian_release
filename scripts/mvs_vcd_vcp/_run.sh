#!/usr/bin/env bash
set -euo pipefail

# Direct MVS initialization + the existing FastGS-inspired VCD/VCP controller.
# Pin MODE=mvs so neither the caller's environment nor extra CLI options can
# re-enable adaptive initialization; the common runner forces replace last.
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
            "Usage: bash scripts/mvs_vcd_vcp/train_${dataset}.sh [extra train.py arguments]" \
            '' \
            'Direct MVS-only initialization + existing FastGS-inspired VCD/VCP.' \
            'MVS replaces the baseline cloud: no adaptive MVS scoring, gate, or residual fusion.' \
            'Not an exact reproduction of upstream FastGS VCD/VCP.' \
            'Forces --init_mode mvs_anywhere --mvs_init_strategy replace.' \
            'MODE and MVS_INIT_STRATEGY cannot change this initialization policy.' \
            'Missing/incompatible MVS caches are errors; no baseline fallback.' \
            '' \
            'Full configured scene lists; default views LLFF=3, DTU=3, Mip-NeRF360=12, Blender=8.' \
            'Same resolutions, 10000 iterations, and train -> render -> evaluate -> summarize workflow.' \
            'DENSITY_CONTROL: none | vcd | vcp | vcd_vcp (default: vcd_vcp).' \
            'Overrides: N_VIEWS, GPU_ID, DATA_ROOT, SCENES, ITERATIONS, RESOLUTION,' \
            'PYTHON_BIN, OUTPUT_ROOT, MVS_INIT_PATH, MVS_INIT_MAX_POINTS, DTU_MASK_ROOT.' \
            'Default output: output/mvs_<DENSITY_CONTROL>/<dataset><views>.' \
            'Use a fresh OUTPUT_ROOT when changing settings beyond dataset/views/density mode.'
        exit 0
    fi
done

MODE=mvs MVS_INIT_STRATEGY=replace \
exec bash "$script_dir/../vcd_vcp/_run.sh" "$dataset" "$@"
