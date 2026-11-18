#!/usr/bin/env bash
set -euo pipefail

# Original initialization plus TBP only; no learned MVS or EGD.
# Serial execution over the nine default settings; MATRIX selects a subset.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_matrix.sh" tbp_only "$@"
