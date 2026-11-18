#!/usr/bin/env bash
set -euo pipefail

# Original initialization; no added EGD or TBP.
# Serial execution over the nine default settings; MATRIX selects a subset.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_matrix.sh" baseline "$@"
