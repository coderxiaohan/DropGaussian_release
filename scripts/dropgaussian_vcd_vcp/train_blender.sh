#!/usr/bin/env bash
set -euo pipefail

# Original Blender random initialization + VCD/VCP; no MVSAnywhere.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
