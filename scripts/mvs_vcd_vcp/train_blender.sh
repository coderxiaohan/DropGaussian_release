#!/usr/bin/env bash
set -euo pipefail

# Direct MVS + VCD/VCP on Blender, without dataset-specific MVS bypasses.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
