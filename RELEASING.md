# Publishing Glossematics binaries

Builds originate in the private `1of2-ai/glossematics` repository. This public repository only distributes compiled releases and validates their assets. Do not upload application source or signing credentials here.

## Credentials in the private source repository

Add these repository Actions secrets at https://github.com/1of2-ai/glossematics/settings/secrets/actions:

| Secret | Value |
| --- | --- |
| `RELEASES_TOKEN` | Fine-grained GitHub token for `1of2-ai/glossematics-server`, Contents: read and write |
| `DEVELOPER_ID_CERT_P12_BASE64` | Base64 Developer ID Application certificate and private key, exported as a password-protected .p12 |
| `DEVELOPER_ID_CERT_PASSWORD` | The .p12 export password |
| `NOTARY_API_KEY_P8_BASE64` | Base64 App Store Connect notarization API key |
| `NOTARY_API_KEY_ID` | Key ID for that API key |
| `NOTARY_API_ISSUER_ID` | Issuer ID for that API key |

The signing team is `KA589LJT76`. The source workflow tests, signs with the hardened runtime, submits the disk image for notarization, staples it, and creates the matching archive and manifest. The destination token is separate because a repository's automatic GITHUB_TOKEN cannot publish to another repository.

## Cut a release

In the private source checkout, update `BuildInfo.version` in `Sources/glossematicsd/Runtime.swift`, commit, and push the matching `vVERSION` tag. For the current version:

```sh
git tag v0.4.0
git push origin main v0.4.0
```

After tests and notarization succeed, CI invokes `bash Scripts/publish_release.sh dist v0.4.0`. It validates assets, uploads to a draft, confirms asset sizes, then publishes. Stable releases become latest; prereleases remain outside Helmet's stable feed. Existing releases are not overwritten. Inspect and remove a failed draft before retrying.

Helmet reads `release.json` and the matching tar.gz from this repository's latest release. The archive contains `glossematicsd`, a compatibility `gloss-server` symlink, and required runtime bundles. Embedding model weights remain separate downloads.

No first release has been published as part of repository setup. Signing, notarization, and a fresh Helmet installation must still be verified with a real release.
