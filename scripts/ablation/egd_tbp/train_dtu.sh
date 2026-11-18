#!/usr/bin/env bash
set -euo pipefail

# Original initialization plus EGD and TBP; no learned MVS.
# N_VIEWS / GPU_ID select the setting; --dry-run previews without training.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../_run.sh" egd_tbp dtu "$@"
