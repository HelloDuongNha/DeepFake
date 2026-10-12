#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
# Prefer Command Line Tools when full Xcode is installed but its license has
# not yet been accepted; these tools are sufficient for an unsigned build.
if [[ -d /Library/Developer/CommandLineTools ]]; then
  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
fi
PYTHON="${PYTHON:-.venv/bin/python}"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$PWD/.cache/pyinstaller}"
mkdir -p "$PYINSTALLER_CONFIG_DIR"
"$PYTHON" packaging/make_icons.py
"$PYTHON" -m PyInstaller --noconfirm --clean --distpath dist/desktop-macos --workpath build/desktop-macos packaging/DeepLiveCam.spec
if [[ "${DLC_SKIP_DMG:-0}" == "1" ]]; then
  echo "Built dist/desktop-macos/DeepLiveCam.app (DMG skipped)"
  exit 0
fi
rm -rf dist/desktop-macos/dmg-root
mkdir -p dist/desktop-macos/dmg-root
cp -R dist/desktop-macos/DeepLiveCam.app dist/desktop-macos/dmg-root/
ln -s /Applications dist/desktop-macos/dmg-root/Applications
hdiutil create -volname "Deep Live Cam" -srcfolder dist/desktop-macos/dmg-root -ov -format UDZO dist/desktop-macos/DeepLiveCam-macOS.dmg
echo "Built dist/desktop-macos/DeepLiveCam.app and DeepLiveCam-macOS.dmg"
