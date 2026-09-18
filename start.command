#!/bin/zsh
cd "${0:A:h}" || exit 1
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.tmp/"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"
exec "$PWD/.venv/bin/python" "$PWD/launcher.py"
