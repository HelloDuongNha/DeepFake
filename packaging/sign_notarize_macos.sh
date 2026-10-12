#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${APPLE_IDENTITY:?Set APPLE_IDENTITY to a Developer ID Application identity}"
: "${APPLE_NOTARY_PROFILE:?Set APPLE_NOTARY_PROFILE to a notarytool keychain profile}"
APP="dist/desktop-macos/DeepLiveCam.app"
DMG="dist/desktop-macos/DeepLiveCam-macOS.dmg"
ENTITLEMENTS="packaging/macos-entitlements.plist"
test -d "$APP"
codesign --force --deep --options runtime --timestamp --entitlements "$ENTITLEMENTS" --sign "$APPLE_IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
spctl --assess --type execute --verbose "$APP" || true # Notarization has not run yet.
rm -rf dist/desktop-macos/dmg-root
mkdir -p dist/desktop-macos/dmg-root
cp -R "$APP" dist/desktop-macos/dmg-root/
ln -s /Applications dist/desktop-macos/dmg-root/Applications
hdiutil create -volname "Deep Live Cam" -srcfolder dist/desktop-macos/dmg-root -ov -format UDZO "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$APPLE_NOTARY_PROFILE" --wait
xcrun stapler staple "$DMG"
xcrun stapler validate "$DMG"
