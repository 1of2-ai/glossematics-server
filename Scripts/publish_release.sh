#!/bin/bash
# Publish a complete, notarized package to Helmet's fixed download repository.
# GH_TOKEN must have Contents: write on 1of2-ai/glossematics-server.
set -euo pipefail
DIST="${1:?usage: publish_release.sh DIST vVERSION}"
TAG="${2:?usage: publish_release.sh DIST vVERSION}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=1of2-ai/glossematics-server
asset_list="$(python3 "$ROOT/validate_release.py" "$DIST" "$TAG")"
assets=()
while IFS= read -r name; do assets+=("$DIST/$name"); done <<< "$asset_list"
# Preflight authentication separately so a network/auth failure cannot mean 'release absent'.
gh api "repos/$REPO" --silent
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    echo "Release $TAG already exists. Inspect it explicitly; this script never replaces releases." >&2
    exit 1
fi
# Draft first: /releases/latest cannot see a partially uploaded build.
gh release create "$TAG" --repo "$REPO" --draft --title "Glossematics $TAG" \
    --notes "Signed and notarized Glossematics for Apple silicon. Helmet installs release.json and the matching tar.gz. Model weights are downloaded separately."
gh release upload "$TAG" --repo "$REPO" "${assets[@]}"
# Confirm every expected asset arrived before making the release visible.
gh release view "$TAG" --repo "$REPO" --json assets > "$DIST/uploaded-assets.json"
python3 - "$DIST/uploaded-assets.json" "${assets[@]}" <<'PY'
import json, pathlib, sys
actual = {a['name']: a['size'] for a in json.load(open(sys.argv[1]))['assets']}
for name in sys.argv[2:]:
    path = pathlib.Path(name)
    assert actual.get(path.name) == path.stat().st_size, f'Missing or incomplete upload: {path.name}'
PY
if [[ "$TAG" == *-* ]]; then
    gh release edit "$TAG" --repo "$REPO" --draft=false --prerelease --latest=false
else
    gh release edit "$TAG" --repo "$REPO" --draft=false --latest
fi
gh release view "$TAG" --repo "$REPO" --json url --jq .url
