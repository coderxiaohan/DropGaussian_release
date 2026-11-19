#!/usr/bin/env bash
set -euo pipefail

# MVS complement + EGD densification/protection + FastGS pruning.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" llff "$@"
