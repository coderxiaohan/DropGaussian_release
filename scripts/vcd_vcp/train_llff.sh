#!/usr/bin/env bash
set -euo pipefail

# Adaptive MVS initialization + optional VCD/VCP on all configured LLFF scenes.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" llff "$@"
