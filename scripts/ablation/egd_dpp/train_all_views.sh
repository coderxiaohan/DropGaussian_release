#!/usr/bin/env bash
set -euo pipefail

# EGD + protected DPP over the nine default settings; MATRIX selects a subset.
# All settings are checked before serial execution begins.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_matrix.sh" egd_dpp "$@"
