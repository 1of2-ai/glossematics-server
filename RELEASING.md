# Releasing gloss-server

`.github/workflows/ci.yml` has two jobs:

- **Build and test** runs on every push and pull request: `make test`, which runs the unit
  tests and the HTTP smoke of both Core ML golden fixtures (BidirLM and Jina) against the debug
  and release binaries.
- **Sign and notarize** runs on pushes to `main`, on `v*` tags, and on manual runs. It builds
  an arm64 release, signs `gloss-server` with a Developer ID Application certificate (hardened
  runtime, secure timestamp, no entitlements), packs it with its resource bundles into a disk
  image, notarizes the image with `notarytool`, and staples the ticket
  (`Scripts/package_release.sh`). The image is uploaded as a workflow artifact; a tag also
  publishes it as a GitHub Release.

Both jobs run on GitHub's `xcode-27` image (Xcode 27 / Swift 6.4 on Apple silicon). That image is
a public preview; when a GA macOS image ships Xcode 27, change `runs-on`.

## One-time setup: five repository secrets

No GitHub App or third-party action is needed. The workflow's own `GITHUB_TOKEN` publishes
releases, and the Apple credentials live in repository secrets
(GitHub → Settings → Secrets and variables → Actions → New repository secret).

### 1. Developer ID Application certificate

This needs an Apple Developer Program membership. Developer ID certificates are created by the
Account Holder, or by an Admin who has been given access.

1. In Xcode → Settings → Accounts, select the team, then Manage Certificates → + →
   **Developer ID Application**. You can also create it at developer.apple.com → Certificates.
2. In Keychain Access → My Certificates, select the certificate together with its private key.
   Choose Export → `.p12` and set a password.
3. Add two secrets:

   | Secret | Value |
   | --- | --- |
   | `DEVELOPER_ID_CERT_P12_BASE64` | `base64 -i DeveloperID.p12 \| pbcopy`, then paste |
   | `DEVELOPER_ID_CERT_PASSWORD` | the export password |

### 2. App Store Connect API key for notarization

1. Go to App Store Connect → Users and Access → Integrations → App Store Connect API → **Team Keys**.
   Generate a key with **Developer** access.
2. Download `AuthKey_<KEYID>.p8`. It can be downloaded only once. Note the **Key ID** and the
   **Issuer ID** shown above the key list.
3. Add three secrets:

   | Secret | Value |
   | --- | --- |
   | `NOTARY_API_KEY_P8_BASE64` | `base64 -i AuthKey_<KEYID>.p8 \| pbcopy`, then paste |
   | `NOTARY_API_KEY_ID` | the Key ID |
   | `NOTARY_API_ISSUER_ID` | the Issuer ID (a UUID) |

With the GitHub CLI, the equivalent is `gh secret set NAME --repo 1of2-ai/glossematics-server`,
which reads the value from stdin.

Until all five secrets exist, the signing job on `main` skips with a notice, and a tag fails with
an error that points here. To limit who can use the certificate, you can move the secrets to a
GitHub environment (for example `release`, with required reviewers) and add
`environment: release` to the `release` job.

## Cutting a release

1. Bump `BuildInfo.version` in `Sources/gloss-server/Runtime.swift`. The tag must be
   `v<version>`, and the workflow checks this.
2. Commit, then run `git tag v0.3.0 && git push origin main v0.3.0`.
3. The release job publishes `gloss-server-0.3.0-macos-arm64.dmg` and its `.sha256`.

## Signing locally

```bash
# ad-hoc signature with the hardened runtime: checks packaging, not distributable
make package
# Developer ID signature plus notarization, with the same credentials as CI
CODESIGN_IDENTITY="Developer ID Application: <Team> (<TEAMID>)" \
NOTARY_KEY_PATH=~/keys/AuthKey_<KEYID>.p8 NOTARY_KEY_ID=<KEYID> NOTARY_ISSUER_ID=<issuer> \
  make package NOTARIZE=1
```

The disk image holds `gloss-server`, its resource bundles (they must stay next to the
executable), `README.md`, and `gloss-server-launchagent.sh`. The image carries the stapled
ticket. A bare executable cannot hold one, so Gatekeeper verifies `gloss-server`'s
notarization online the first time it runs from a downloaded copy.
