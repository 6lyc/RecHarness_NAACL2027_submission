#!/usr/bin/env bash
# Amazon entrypoint. Arguments are identical to python -m recharness.
set -euo pipefail
umask 077
RECHARNESS_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${RECHARNESS_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RECHARNESS_PYTHON:-python3}" -m recharness "$@"
