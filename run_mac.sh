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
export DLC_ENHANCER="${DLC_ENHANCER:-GPEN-256}"
export DLC_ENHANCER_INTERVAL="${DLC_ENHANCER_INTERVAL:-1}"
export DLC_MASK_BLUR="${DLC_MASK_BLUR:-1.5}"
export DLC_MASK_EROSION="${DLC_MASK_EROSION:-4}"
export DLC_DETAIL_STRENGTH="${DLC_DETAIL_STRENGTH:-0.58}"
export DLC_FILM_GRAIN="${DLC_FILM_GRAIN:-0.35}"
export DLC_HAIRLINE_GUARD="${DLC_HAIRLINE_GUARD:-0.16}"
export DLC_HAIR_PARSING="${DLC_HAIR_PARSING:-1}"
if [[ "$DLC_HAIR_PARSING" != "0" && "$DLC_HAIR_PARSING" != "off" ]]; then
  PARSER_MODEL="${DLC_HAIR_PARSER_MODEL:-$PROJECT_DIR/models/face_parsing_resnet18.onnx}"
  if [[ ! -s "$PARSER_MODEL" ]]; then
    if [[ -n "${DLC_HAIR_PARSER_MODEL:-}" ]]; then
      echo "Missing face parser: $PARSER_MODEL" >&2
      exit 1
    fi
    mkdir -p "$PROJECT_DIR/models"
    echo "Downloading face parsing model (about 51 MB)..."
    if ! curl -fL --retry 2 \
      -o "$PARSER_MODEL.part" \
      https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx; then
      rm -f "$PARSER_MODEL.part"
      exit 1
    fi
    mv "$PARSER_MODEL.part" "$PARSER_MODEL"
  fi
fi
export DLC_COLOR_MATCH="${DLC_COLOR_MATCH:-1}"
export DLC_DET_THRESHOLD="${DLC_DET_THRESHOLD:-0.40}"
export DLC_BLEND_MODE=alpha
export DLC_DISABLE_POISSON=1
export DLC_CAPTURE_WIDTH="${DLC_CAPTURE_WIDTH:-640}"
export DLC_CAPTURE_HEIGHT="${DLC_CAPTURE_HEIGHT:-360}"
export DLC_CAPTURE_FPS="${DLC_CAPTURE_FPS:-60}"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export TMPDIR="$PROJECT_DIR/.tmp/"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

cd "$PROJECT_DIR"
exec "$PYTHON" run.py --execution-provider auto --frame-processor face_swapper --live-resizable "$@"
