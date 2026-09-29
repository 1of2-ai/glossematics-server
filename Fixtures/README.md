# Core ML test fixtures

`BidirLMOmni.dummy.bundle` is the checked-in, compiled Core ML golden fixture for the BidirLM
family: the production manifest contract, the real tokenizer, every function signature, and
checksums, with no-op functions that return the unit vector e0. Server tests use it without
loading the full model. `make fixture` writes its 622 MB all-zero token table (gitignored; the
checksum is fixed in the manifest). The server requires `--allow-dummy` to load it.

The Jina fixture (`JinaV5OmniSmall.w8a16.dummy.bundle`, constant outputs, `converter.name`
`dummy-noop`) is not checked in; it lives in `GlossematicsCoreML/artifacts/`.
