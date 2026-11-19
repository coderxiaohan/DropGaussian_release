#!/usr/bin/env bash
set -euo pipefail

# Descriptive entry point for the existing combined method. Reuse its policy,
# validation and dataset protocols; isolate results under a separate parent.
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
            "Usage: bash scripts/mvs_complement_egd_fastgs/train_${dataset}.sh [--dry-run] [train.py arguments...]" \
            '' \
            'MVS geometry complement + EGD densification with same-event protection + FastGS pruning.' \
            'Real datasets: preserve base geometry and fill coverage gaps with MVS points.' \
            'Blender exception: existing random 10000-point initialization, no MVS; protected EGD + FastGS remain active.' \
            'Same algorithm and parameters as scripts/fastgs_egd_pruning/; only the default output parent changes.' \
            'Fixes FASTGS_PROTECT_EGD=1; conflicting policy environment values or method CLI overrides are rejected.' \
            'Defaults: all configured scenes, 10000 iterations, GPU 0; views 3 (LLFF/DTU), 12 (Mip-NeRF360), 8 (Blender).' \
            'Configure GPU_ID, N_VIEWS, SCENES, DATA_ROOT, PYTHON_BIN and OUTPUT_ROOT as in the shared launcher.' \
            'Default OUTPUT_ROOT: <repo>/output/mvs_complement_egd_fastgs (a common parent).' \
            'Real output: <root>/coverage/<keep_base or budgetN>/<dataset><views>/<scene>.' \
            'Blender output: <root>/blender_random/blender<views>/<scene>.' \
            'Nonempty result directories are rejected; existing runs are not overwritten.' \
            '--dry-run or DRY_RUN=1 prints dispatch only; no writes, CUDA compilation or training.' \
            'See scripts/mvs_complement_egd_fastgs/README.md for commands and policy details.'
        exit 0
    fi
done
output_parent=${OUTPUT_ROOT:-$repo_root/output/mvs_complement_egd_fastgs}
exec env OUTPUT_ROOT="$output_parent" \
    bash "$script_dir/../fastgs_egd_pruning/_run.sh" "$dataset" "$@"
