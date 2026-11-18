#!/usr/bin/env bash
set -euo pipefail

# Geometry-complementary initialization plus TBP; no EGD.
# Serial execution over the nine default settings; MATRIX selects a subset.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_matrix.sh" complement_tbp "$@"
