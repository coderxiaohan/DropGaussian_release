#!/usr/bin/env bash
set -euo pipefail

# Geometry-complementary initialization only; no added EGD or TBP.
# N_VIEWS / GPU_ID select the setting; --dry-run previews without training.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_run.sh" complement_only llff "$@"
