#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON" >&2
  exit 1
fi

export DLC_PROFILE=mac-smooth
export DLC_SOURCE_DIR="${DLC_SOURCE_DIR:-$PROJECT_DIR/source_faces}"
export DLC_ENHANCER="${DLC_ENHANCER:-None}"
export DLC_ENHANCER_INTERVAL="${DLC_ENHANCER_INTERVAL:-3}"
export DLC_MASK_BLUR="${DLC_MASK_BLUR:-1.5}"
export DLC_MASK_EROSION="${DLC_MASK_EROSION:-4}"
export DLC_DETAIL_STRENGTH="${DLC_DETAIL_STRENGTH:-0.35}"
export DLC_FILM_GRAIN="${DLC_FILM_GRAIN:-0.35}"
export DLC_HAIRLINE_GUARD="${DLC_HAIRLINE_GUARD:-0.16}"
export DLC_COLOR_MATCH="${DLC_COLOR_MATCH:-1}"
export DLC_BLEND_MODE="${DLC_BLEND_MODE:-alpha}"
export DLC_CAPTURE_WIDTH="${DLC_CAPTURE_WIDTH:-640}"
export DLC_CAPTURE_HEIGHT="${DLC_CAPTURE_HEIGHT:-360}"
export DLC_CAPTURE_FPS="${DLC_CAPTURE_FPS:-60}"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export TMPDIR="$PROJECT_DIR/.tmp/"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

cd "$PROJECT_DIR"
exec "$PYTHON" run.py --execution-provider auto --frame-processor face_swapper --live-resizable "$@"
