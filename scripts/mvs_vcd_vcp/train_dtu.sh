#!/usr/bin/env bash
set -euo pipefail

# Direct MVS + VCD/VCP on DTU; preserves the existing masked evaluation.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" dtu "$@"
