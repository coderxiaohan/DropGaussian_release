#!/usr/bin/env bash
set -euo pipefail

# Mip-NeRF360: reconstructed geometry + complementary MVS, unchanged VCD/VCP.
# Defaults: all 7 configured scenes, N_VIEWS=12, resolution=8, 10000 steps, GPU 0.
# Views: N_VIEWS=12 / 24. GPU example: GPU_ID=3 N_VIEWS=24.
# Use --dry-run for a read-only preview; --help documents budgets and ablations.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" mipnerf360 "$@"
