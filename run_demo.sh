#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT_DIR/habitat-env/bin:$PATH"
exec "$ROOT_DIR/habitat-env/bin/python" "$ROOT_DIR/soundspaces_mp3d_demo.py" "$@"
