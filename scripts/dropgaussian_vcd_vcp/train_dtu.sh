#!/usr/bin/env bash
set -euo pipefail

# Original DTU baseline initialization + VCD/VCP and masked evaluation.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" dtu "$@"
