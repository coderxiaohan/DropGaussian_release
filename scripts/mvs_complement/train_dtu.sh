#!/usr/bin/env bash
set -euo pipefail

# DTU: retain reconstructed geometry and complement it with MVS; no surface/TPS.
# Defaults: all 15 scenes, N_VIEWS=3, resolution=4, 10000 steps, GPU_ID=0.
# Views: N_VIEWS=3 / 6 / 9. DTU evaluation uses the existing IDR-mask protocol.
# Use --dry-run for a read-only preview; --help documents budgets and ablations.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" dtu "$@"
