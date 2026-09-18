#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${1:-$PROJECT_DIR/source_faces}"

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Missing .venv. Create it with Python 3.14 and install requirements.txt first." >&2
  exit 1
fi
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source image directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

export DLC_SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export TMPDIR="$PROJECT_DIR/.tmp/"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
if ! "$PROJECT_DIR/.venv/bin/python" -c 'import onnxruntime as ort; assert "CoreMLExecutionProvider" in ort.get_available_providers()'; then
  echo "CoreMLExecutionProvider is unavailable in .venv." >&2
  exit 1
fi
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" run.py \
  --execution-provider coreml \
  --frame-processor face_swapper \
  --live-resizable
