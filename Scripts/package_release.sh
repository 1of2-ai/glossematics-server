#!/bin/bash
# Build, sign, notarize, and staple a gloss-server release disk image.
#
# usage:
#   Scripts/package_release.sh [--version V] [--identity NAME|-] [--notarize] [--output DIR]
#
#   --identity   codesigning identity: a "Developer ID Application: …" name or SHA-1 from the
#                keychain, or "-" for an ad-hoc signature (local checks only; not distributable).
#                Default: $CODESIGN_IDENTITY, else "-".
#   --notarize   submit the disk image to Apple's notary service and staple the ticket. Needs an
#                App Store Connect API key: NOTARY_KEY_PATH (.p8), NOTARY_KEY_ID, NOTARY_ISSUER_ID.
#
# Output (DIR, default dist/): gloss-server-<version>-macos-arm64.dmg and its .sha256. The image
# holds gloss-server, its resource bundles (which must stay next to the executable), and the
# README and launch-agent script. The executable is signed with the hardened runtime and a secure
# timestamp; it needs no entitlements. A bare Mach-O cannot carry a stapled ticket, so the disk
# image is what gets stapled; Gatekeeper checks the executable's notarization online.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=""
IDENTITY="${CODESIGN_IDENTITY:--}"
NOTARIZE=0
OUTPUT="$ROOT/dist"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)  VERSION="${2:?missing --version value}"; shift 2 ;;
        --identity) IDENTITY="${2:?missing --identity value}"; shift 2 ;;
        --notarize) NOTARIZE=1; shift ;;
        --output)   OUTPUT="${2:?missing --output value}"; shift 2 ;;
        -h|--help)  sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    VERSION="$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)"
fi
VERSION="${VERSION#v}"
[[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "error: invalid version '$VERSION'" >&2; exit 2; }
if [[ "$NOTARIZE" == "1" && "$IDENTITY" == "-" ]]; then
    echo "error: --notarize needs a Developer ID Application identity, not an ad-hoc signature" >&2
    exit 2
fi
if [[ "$NOTARIZE" == "1" ]]; then
    : "${NOTARY_KEY_PATH:?set NOTARY_KEY_PATH to the App Store Connect API key (.p8)}"
    : "${NOTARY_KEY_ID:?set NOTARY_KEY_ID}"
    : "${NOTARY_ISSUER_ID:?set NOTARY_ISSUER_ID}"
fi

NAME="gloss-server-$VERSION-macos-arm64"
BUILD_PATH="$ROOT/.build/package"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gloss-package.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
STAGE="$WORK/$NAME"
mkdir -p "$STAGE" "$OUTPUT"

echo "==> building release (arm64) into $BUILD_PATH"
swift build -c release --arch arm64 --build-path "$BUILD_PATH" --package-path "$ROOT"
BIN_DIR="$(swift build -c release --arch arm64 --build-path "$BUILD_PATH" --package-path "$ROOT" --show-bin-path)"

cp "$BIN_DIR/gloss-server" "$STAGE/"
# SwiftPM resource bundles are looked up next to the executable at runtime.
shopt -s nullglob
bundles=("$BIN_DIR"/*.bundle)
[[ ${#bundles[@]} -gt 0 ]] || { echo "error: no resource bundles in $BIN_DIR" >&2; exit 1; }
for bundle in "${bundles[@]}"; do cp -R "$bundle" "$STAGE/"; done
[[ -d "$STAGE/GlossematicsServer_gloss-server.bundle" ]] || {
    echo "error: GlossematicsServer_gloss-server.bundle is missing from the build" >&2; exit 1; }
cp "$ROOT/README.md" "$STAGE/"
cp "$ROOT/Scripts/gloss-server-launchagent.sh" "$STAGE/"

echo "==> signing gloss-server ($([[ "$IDENTITY" == "-" ]] && echo ad-hoc || echo "$IDENTITY"))"
sign_args=(--force --options runtime --sign "$IDENTITY")
[[ "$IDENTITY" != "-" ]] && sign_args+=(--timestamp)
codesign "${sign_args[@]}" "$STAGE/gloss-server"
codesign --verify --strict --verbose=2 "$STAGE/gloss-server"
"$STAGE/gloss-server" --version

DMG="$OUTPUT/$NAME.dmg"
rm -f "$DMG"
echo "==> creating $DMG"
hdiutil create -quiet -volname "gloss-server $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO "$DMG"
dmg_args=(--force --sign "$IDENTITY")
[[ "$IDENTITY" != "-" ]] && dmg_args+=(--timestamp)
codesign "${dmg_args[@]}" "$DMG"

if [[ "$NOTARIZE" == "1" ]]; then
    echo "==> notarizing (this usually takes a few minutes)"
    result="$WORK/notary.json"
    xcrun notarytool submit "$DMG" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" \
        --issuer "$NOTARY_ISSUER_ID" --wait --timeout 60m --output-format json > "$result" || true
    status="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$result" 2>/dev/null || true)"
    submission="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("id",""))' "$result" 2>/dev/null || true)"
    if [[ "$status" != "Accepted" ]]; then
        echo "error: notarization status '${status:-unknown}'" >&2
        cat "$result" >&2 || true
        if [[ -n "$submission" ]]; then
            xcrun notarytool log "$submission" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" \
                --issuer "$NOTARY_ISSUER_ID" >&2 || true
        fi
        exit 1
    fi
    echo "==> stapling"
    xcrun stapler staple "$DMG"
    xcrun stapler validate "$DMG"
    spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
fi

(cd "$OUTPUT" && shasum -a 256 "$NAME.dmg" > "$NAME.dmg.sha256")
echo "==> $DMG"
cat "$OUTPUT/$NAME.dmg.sha256"
