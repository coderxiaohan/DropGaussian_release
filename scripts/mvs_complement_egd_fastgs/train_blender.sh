#!/usr/bin/env bash
set -euo pipefail

# Existing Blender protocol: random initialization (no MVS), protected EGD + FastGS.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
