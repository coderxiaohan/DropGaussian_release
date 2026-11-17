#!/usr/bin/env bash
set -euo pipefail

# Original LLFF baseline initialization + VCD/VCP; no MVSAnywhere.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" llff "$@"
