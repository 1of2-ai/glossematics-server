# GlossematicsServer

OpenAI-compatible `/v1/embeddings` localhost daemon for Glossematics Core ML model bundles.
Serves contract-compatible bundles (e.g. `jina-embeddings-v5-omni-small`) on-device through the
`Glossematics` inference library, bound to 127.0.0.1 only — requests never leave the machine.

```
POST /v1/embeddings   OpenAI contract (+ task/role conditioning, image_path/audio_path items)
GET  /v1/models       served model list
GET  /health          readiness + loaded Matryoshka dimensions
GET  /docs            self-contained API documentation (also at /)
```

## Dependencies

One path dependency, into the sibling SDK checkout (see `Package.swift`):

- `Glossematics` inference library — `../../Models/GlossematicsSDK/SwiftPackages/Glossematics`

Model bundles are **passed by path, not imported**: the server consumes whatever bundle you
give it (`BUNDLE=`), wherever it was produced. For tests, this repo generates its own no-op
dummy bundle — no conversion pipeline needs to exist on the machine.

## Build, run, install

```bash
make dummy-bundle SOURCE_BUNDLE=/path/to/JinaV5OmniSmall.w8a16.bundle
                  # -> TestBundles/…dummy.bundle: passes the full production load contract,
                  #    constant unit-vector embeddings (see the script docstring)

make run                  # serves BUNDLE (defaults to the generated dummy)
make test                 # docs packaging + templating tests

BUNDLE=/path/to/JinaV5OmniSmall.w8a16.bundle make install
                          # release build -> launchd agent (RunAtLoad, KeepAlive)
make status | restart | uninstall | logs
```

`install` refuses a `*dummy*` bundle unless `ALLOW_DUMMY=1` — the dummy emits constant
vectors, which is exactly what you want in tests and never in a running agent.
`install` copies the binary and resource bundles to
`~/Library/Application Support/Glossematics/bin/` and bootstraps
`~/Library/LaunchAgents/com.meridian.glossematics.embeddings.plist`; logs land in
`~/Library/Logs/Glossematics/`.

`dummy-bundle` needs an interpreter with `coremltools`, `torch`, and `numpy`
(`PYTHON=/path/to/venv/bin/python` to point at one) and a real bundle to mirror for the
tokenizer, vision assets, and compiled I/O schemas.

## Client quick start

```bash
curl -s http://127.0.0.1:11435/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"jinaai/jina-embeddings-v5-omni-small",
       "input":["How do I cool an overheating laptop?"],
       "task":"retrieval.query","dimensions":512}'
```

The official OpenAI clients work unchanged against `base_url="http://127.0.0.1:11435/v1"`.
Full endpoint documentation is served at `http://127.0.0.1:11435/docs`.

## Layout

- `Sources/gloss-server/` — HTTP server (Network.framework), OpenAI schema, embeddings
  service, `/docs` page (`Resources/docs.html` packaged as a SwiftPM resource)
- `Tests/GlossServerTests/` — docs packaging + templating tests
- `Scripts/gloss-server-launchagent.sh` — launchd agent lifecycle
- `Scripts/make_dummy_bundle.py` — self-contained no-op test-bundle generator
- `TestBundles/` — generated dummies (gitignored)
