#!/usr/bin/env bash
set -euo pipefail

# Base initialization + native ADC + DPP with same-event multiview protection.
# N_VIEWS / GPU_ID select the setting; --dry-run previews without training.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_run.sh" dpp_only llff "$@"
