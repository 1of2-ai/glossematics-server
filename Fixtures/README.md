# Core ML test fixtures

Both fixtures are real compiled Core ML bundles with their family's production contract (manifest,
tokenizer, every function signature, checksums) and no-op functions, so tests exercise bundle
validation, tokenization, scheduling, media decoding, the Core ML execution path, and HTTP without
model weights or model quality. The server loads them only with `--allow-dummy` and labels every
response `X-Glossematics-Dummy: true`.

- `BidirLMOmni.dummy.bundle`: every function returns the unit vector e0. `make fixture` writes
  its 622 MB all-zero token table (gitignored; the checksum is fixed in the manifest).
  Regenerate with `make dummy-bundle TOKENIZER=<BidirLM bundle>/tokenizer`.
- `JinaV5OmniSmall.w8a16.dummy.bundle` (63 MB, `converter.name` `dummy-noop`): every function
  returns `1/32 * ones(1024)`, which stays unit-norm under Matryoshka truncation. Regenerate with
  `make jina-fixture JINA_SOURCE=<JinaV5OmniSmall.w8a16.bundle>` (`Scripts/make_jina_dummy_bundle.py`,
  restored from the pre-BidirLM daemon); it cross-checks every function's I/O schema against the
  source bundle.

`golden-video.mp4` is a one-second H.264 clip with eight distinct 64×64 frames (the file and
uploaded-byte video paths). `golden-long-video.mp4` is a 20-second, 10 fps, 256×256 clip whose
2 fps sampling reaches 32 frames and the video tower's largest (f2048) patch bucket:

```bash
ffmpeg -f lavfi -i 'testsrc=size=64x64:rate=8:duration=1' \
  -c:v libx264 -preset ultrafast -crf 25 -pix_fmt yuv420p golden-video.mp4
ffmpeg -f lavfi -i 'testsrc=size=256x256:rate=10:duration=20' \
  -an -c:v libx264 -preset veryfast -crf 34 -pix_fmt yuv420p -movflags +faststart golden-long-video.mp4
```

`ffmpeg` is not required to build or test the daemon.
