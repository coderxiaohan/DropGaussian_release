#!/usr/bin/env bash
set -euo pipefail

# Direct MVS + VCD/VCP on LLFF; N_VIEWS and GPU_ID remain configurable.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" llff "$@"
