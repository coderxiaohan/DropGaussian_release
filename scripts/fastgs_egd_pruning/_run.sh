#!/usr/bin/env bash
set -euo pipefail

# Thin, fixed-policy wrapper: keep all scheduling/validation in the shared runner.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
dataset=${1:?A dataset is required}
shift
case "$dataset" in
    llff|dtu|mipnerf360|blender) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        printf '%s\n' \
            "Usage: bash scripts/fastgs_egd_pruning/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
            '' \
            'Geometry-complement + unchanged EGD + FastGS pruning with same-event EGD protection.' \
            'Fixes FASTGS_PROTECT_EGD=1; conflicting values and direct method CLI overrides are rejected.' \
            'Only protects the clone parent/new point or split children selected by complete EGD densification criteria.' \
            'Protection applies to that FastGS pruning event only; native opacity/size cleanup is unchanged.' \
            'FastGS scores, thresholds, interval, and all-training-view evaluation are unchanged.' \
            'Defaults: interval 8000, pixel loss threshold 0.1, prune threshold 0.9, opacity threshold 0.1.' \
            'Overrides: GPU_ID, N_VIEWS, SCENES, DATA_ROOT, ITERATIONS, RESOLUTION, PYTHON_BIN,' \
            'OUTPUT_ROOT, DTU_MASK_ROOT, SEED, PORT, numeric FASTGS_* and MVS_* settings from the shared runner.' \
            'Blender keeps original random 10000-point initialization and does not use MVS caches.' \
            'Default OUTPUT_ROOT: <repo>/output/fastgs_egd_pruning (a common parent).' \
            'Real output: <root>/coverage/<keep_base or budgetN>/<dataset><views>/<scene>.' \
            'Blender output: <root>/blender_random/blender<views>/<scene>.' \
            'Existing nonempty setting directories are rejected; use a new OUTPUT_ROOT for repeat runs.' \
            '--dry-run or DRY_RUN=1 writes nothing, does not compile CUDA, and starts no training.' \
            'The old scripts/fastgs_pruning/ default remains unprotected and its outputs are not changed.' \
            'This is an unverified experiment, not a claim of improved quality or speed.'
        exit 0
    fi
done
if [[ ${FASTGS_PROTECT_EGD+x} && "$FASTGS_PROTECT_EGD" != 1 ]]; then
    echo "This experiment fixes FASTGS_PROTECT_EGD=1; unset the conflicting value." >&2
    exit 2
fi
output_parent=${OUTPUT_ROOT:-$repo_root/output/fastgs_egd_pruning}
exec env FASTGS_PROTECT_EGD=1 OUTPUT_ROOT="$output_parent" \
    bash "$script_dir/../fastgs_pruning/_run.sh" "$dataset" "$@"
