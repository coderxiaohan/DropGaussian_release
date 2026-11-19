#!/usr/bin/env bash
set -euo pipefail

# Blender protocol is unchanged: 8 scenes, N_VIEWS=8, resolution=1, 10000 steps.
# The main method uses original DropGaussian random initialization (10000
# points), without MVS caches, with EGD and protected DPP.
# MVS_COMPLEMENT_MODE=none explicitly selects the unchanged direct-MVS ablation.
# Use --dry-run for a read-only preview; --help documents budgets and ablations.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
