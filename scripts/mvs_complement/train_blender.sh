#!/usr/bin/env bash
set -euo pipefail

# Blender protocol is unchanged: 8 scenes, N_VIEWS=8, resolution=1, 10000 steps.
# Random base points are NOT reliable reconstructed geometry. Core initialization
# falls back to direct MVS when no valid geometry exists, and records the reason.
# No dataset-specific algorithm switch is introduced by this launcher.
# Use --dry-run for a read-only preview; --help documents budgets and ablations.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
