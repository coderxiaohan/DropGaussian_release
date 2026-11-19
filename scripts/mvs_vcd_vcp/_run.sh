#!/usr/bin/env bash
set -euo pipefail

# Direct MVS initialization + the existing FastGS-inspired VCD/VCP controller.
# Pin MODE=mvs so neither the caller's environment nor extra CLI options can
# re-enable adaptive initialization; the common runner forces replace last.
# Geometry-warp and surface-covariance experiments have been removed; reject
# their old options instead of silently treating them as direct-MVS runs.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset=${1:?A dataset is required}
shift
case "$dataset" in
    llff|dtu|mipnerf360|blender) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

for argument in "$@"; do
    argument_name=${argument%%=*}
    case "$argument_name" in
        --mvs_geometry*|--mvs_surface*)
            echo "Removed geometry/surface experiment option: $argument_name." >&2
            exit 2
            ;;
        --?*)
            # Reject the former argparse abbreviations as well. The retained
            # --mvs_init_* and --mvs_complement_* options are unaffected.
            for removed_prefix in --mvs_geometry --mvs_surface; do
                if [[ "$removed_prefix" == "$argument_name"* ]]; then
                    echo "Removed geometry/surface experiment option abbreviation: $argument_name." >&2
                    exit 2
                fi
            done
            ;;
    esac
done

for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        printf '%s\n' \
            "Usage: bash scripts/mvs_vcd_vcp/train_${dataset}.sh [extra train.py arguments]" \
            '' \
            'Direct MVS-only initialization + existing FastGS-inspired VCD/VCP.' \
            'MVS replaces the baseline cloud: no adaptive MVS scoring, gate, or residual fusion.' \
            'Not an exact reproduction of upstream FastGS VCD/VCP.' \
            'Forces --init_mode mvs_anywhere --mvs_init_strategy replace.' \
            'Geometry warping and surface-covariance initialization are unavailable.' \
            'MODE and MVS_INIT_STRATEGY cannot change this initialization policy.' \
            'Missing/incompatible MVS caches are errors; no baseline fallback.' \
            '' \
            'Full configured scene lists; default views LLFF=3, DTU=3, Mip-NeRF360=12, Blender=8.' \
            'Same resolutions, 10000 iterations, and train -> render -> evaluate -> summarize workflow.' \
            'DENSITY_CONTROL: none | vcd | vcp | vcd_vcp (default: vcd_vcp).' \
            'Overrides: N_VIEWS, GPU_ID, DATA_ROOT, SCENES, ITERATIONS, RESOLUTION,' \
            'PYTHON_BIN, OUTPUT_ROOT, MVS_INIT_PATH, MVS_INIT_MAX_POINTS, DTU_MASK_ROOT.' \
            'Default output: output/mvs_<DENSITY_CONTROL>/<dataset><views>.' \
            'Every run requires a fresh or empty OUTPUT_ROOT; nonempty settings are rejected before any writes.'
        exit 0
    fi
done

MODE=mvs MVS_INIT_STRATEGY=replace \
exec bash "$script_dir/../vcd_vcp/_run.sh" "$dataset" "$@"
