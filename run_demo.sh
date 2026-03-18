#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT_DIR/habitat-env/bin:$PATH"

if [[ -d "$ROOT_DIR/third_party/habitat-sim/build" ]]; then
  HABITAT_SIM_BUILD_DIR="$(find "$ROOT_DIR/third_party/habitat-sim/build" -maxdepth 1 -type d -name 'cp*' | head -n 1)"
  if [[ -n "${HABITAT_SIM_BUILD_DIR:-}" ]]; then
    if [[ -n "${SKBUILD_EDITABLE_SKIP:-}" ]]; then
      export SKBUILD_EDITABLE_SKIP="$SKBUILD_EDITABLE_SKIP:$HABITAT_SIM_BUILD_DIR"
    else
      export SKBUILD_EDITABLE_SKIP="$HABITAT_SIM_BUILD_DIR"
    fi
  fi
fi

exec "$ROOT_DIR/habitat-env/bin/python" "$ROOT_DIR/soundspaces_mp3d_demo.py" "$@"
