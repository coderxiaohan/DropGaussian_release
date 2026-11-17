#!/usr/bin/env bash
set -euo pipefail

# Adaptive MVS initialization + optional VCD/VCP, with the existing IDR evaluator.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/_run.sh" dtu "$@"
