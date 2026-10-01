# Glossematics releases

This public repository is the download destination for **Glossematics**, the local inference daemon managed by Helmet. Active development lives in the private `1of2-ai/glossematics` repository. The current branch contains only release tooling and documentation. Earlier embedding-server source remains in Git history.

Helmet checks [the latest release](https://github.com/1of2-ai/glossematics-server/releases/latest) through GitHub's releases API. No Helmet URL change is needed.

Each stable release contains:

| Asset | Purpose |
| --- | --- |
| `release.json` | Version, archive SHA-256, signing team, server description, and pinned embedding-model catalog |
| `gloss-server-VERSION-macos-arm64.tar.gz` | Signed daemon and SwiftPM resource bundles, installed by Helmet |
| `gloss-server-VERSION-macos-arm64.dmg` | Signed, notarized disk image with a stapled ticket |
| Both `.sha256` files | Checksums for the archive and image |

The archive contains `glossematicsd` plus a `gloss-server` symlink for Helmet's existing installation check. Resource bundles remain beside the executable. Model weights are not included. The current catalog covers embedding models; speech and vision keep their existing setup flows.

## Publishing

Build and notarize from the active source repository, then run:

```sh
bash Scripts/publish_release.sh dist v0.4.0
```

The publisher validates the manifest, checksums and archive layout, creates a draft, uploads all assets, verifies upload sizes, then publishes. Stable versions become latest; prereleases do not. Existing releases are never overwritten. If an upload fails, its draft remains for inspection and cleanup before retrying.

`GH_TOKEN` needs **Contents: write** on this repository. In the private source repository, store it as `RELEASES_TOKEN`; its own `GITHUB_TOKEN` cannot publish across repositories. The five Apple signing/notarization secrets stay in the private source repository. Never put credentials or private source code in this public repository.

The publisher requires a notarized build signed by team `KA589LJT76`. Its metadata checks do not replace Apple's signature/notarization checks during packaging. The public workflow validates published assets without executing downloaded binaries.

## Initial activation

The release destination is configured, but Helmet cannot download a server until the first stable release is published. The source release pipeline is configured in `1of2-ai/glossematics`. Add its signing credentials and `RELEASES_TOKEN`, then push a tag matching `BuildInfo.version`.
