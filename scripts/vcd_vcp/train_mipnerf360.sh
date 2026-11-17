#!/usr/bin/env bash
set -euo pipefail

# Adaptive MVS initialization + optional VCD/VCP on configured Mip-NeRF360 scenes.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" mipnerf360 "$@"
