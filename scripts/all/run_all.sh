#!/usr/bin/env bash
set -euo pipefail

# Single-run, all-scene paired workflow: train -> render -> evaluate -> audit.
# The shared runner still records a fixed seed for reproducibility, but this
# entry point always schedules exactly one run per method/scene/setting.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Usage: bash scripts/all/run_all.sh [--dry-run] [train.py arguments...]' \
            '' \
            'Defaults: original baseline + direct MVS/VCD/VCP, every scene in all 9 settings,' \
            'one run each (182 training runs), followed by rendering/evaluation/audit.' \
            'LLFF: 3/6/9 views; DTU: 3/6/9; Mip-NeRF360: 12/24; Blender: 8.' \
            '' \
            'Environment: GPU_ID, PYTHON_BIN, OUTPUT_ROOT, MATRIX, MODES.' \
            'Every output setting must be fresh or empty; existing experiment results are never reused.' \
            'Default output: <repo>/output/mvs_vcd_vcp/all (single run under seed_0).' \
            'SEEDS is intentionally ignored; this entry point never repeats runs.' \
            'Use --dry-run to preview without training or writing outputs.' \
            'Actual runs require a clean committed checkout and prepared data/MVS caches.' \
            'This workflow does not generate MVS caches or implement new ablations.'
        exit 0
        ;;
    --dry-run)
        export DRY_RUN=1
        shift
        ;;
esac

run_seed=${SEED:-0}
if ! [[ "$run_seed" =~ ^[0-9]+$ ]]; then
    echo "SEED must be one non-negative integer; run_all.sh never repeats runs." >&2
    exit 2
fi
export OUTPUT_ROOT=${OUTPUT_ROOT:-$repo_root/output/mvs_vcd_vcp/all}
export SEEDS=$run_seed
exec bash "$script_dir/run_mvsdrop_multiseed.sh" "$@"
