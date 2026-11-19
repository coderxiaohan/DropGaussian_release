#!/usr/bin/env bash
set -euo pipefail

# Unchanged full initialization/EGD with same-event EGD-protected FastGS pruning.
# Configure N_VIEWS / GPU_ID / SCENES; --dry-run prints dispatch only.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" dtu "$@"
