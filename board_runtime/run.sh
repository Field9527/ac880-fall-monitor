#!/bin/bash
set -e
cd "$(dirname "$0")"
export XILINX_XRT=/usr PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
action=run
case "${1:-}" in status|stop) action="$1"; shift;; esac
py=/usr/local/share/pynq-venv/bin/python3
if [ "$(id -u)" -eq 0 ]; then exec "$py" app/manage.py "$action" "$@"; fi
exec sudo -E env XILINX_XRT=/usr PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" app/manage.py "$action" "$@"
