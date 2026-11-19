#!/usr/bin/env bash
set -euo pipefail

# Original random initialization + native ADC + protected DPP; no MVS.
# N_VIEWS defaults to 8; larger values are rejected before dispatch.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_run.sh" dpp_only blender "$@"
