#!/usr/bin/env bash
set -euo pipefail

extra=""

if command -v nvidia-smi >/dev/null 2>&1; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | \
    head -n1)"
  major="${driver%%.*}"
  if [ "${major}" -ge 580 ]; then
    extra="cuda13"
  elif [ "${major}" -ge 525 ]; then
    extra="cuda12"
  fi
fi

if [ -n "${extra}" ]; then
  echo "Installing with --extra ${extra}"
  uv sync --extra "${extra}"
else
  echo "Installing CPU-only"
  uv sync
fi
