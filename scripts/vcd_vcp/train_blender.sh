#!/usr/bin/env bash
set -euo pipefail

# Preserve the shared Blender white-background / in-memory random-cloud protocol.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" blender "$@"
