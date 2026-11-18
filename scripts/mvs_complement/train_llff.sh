#!/usr/bin/env bash
set -euo pipefail

# LLFF: retain reconstructed geometry and complement it with MVS; no surface/TPS.
# Defaults: all 8 scenes, N_VIEWS=3, resolution=8, 10000 steps, GPU_ID=0.
# Views: N_VIEWS=3 / 6 / 9. Subset example: SCENES=horns GPU_ID=2 N_VIEWS=9.
# Use --dry-run for a read-only preview; --help documents budgets and ablations.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" llff "$@"
