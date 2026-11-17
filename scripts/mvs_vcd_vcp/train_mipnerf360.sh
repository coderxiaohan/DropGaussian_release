#!/usr/bin/env bash
set -euo pipefail

# Direct MVS + VCD/VCP on Mip-NeRF360.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" mipnerf360 "$@"
