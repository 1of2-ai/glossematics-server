# GlossematicsServer

`gloss-server` is a localhost-only embeddings daemon with an OpenAI-compatible `/v1/embeddings`
endpoint plus explicit media extensions. It binds IPv4 loopback only and never sends requests,
model data, or media anywhere. One process serves one Core ML bundle, and the model family is
detected from the bundle's `manifest.json`:

| Family | Bundle | Output | Inputs | Retrieval roles |
| --- | --- | --- | --- | --- |
| [`BidirLM/BidirLM-Omni-2.5B-Embedding`](https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding) @ `447a6e31` | `format` `bidirlm-omni-ane-v2` or `bidirlm-omni-streamed-v2` (GlossematicsCoreML/bidirlm) | 2048-d | text, token IDs, image, audio, interleaved messages | none (rejected) |
| [`jinaai/jina-embeddings-v5-omni-small`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) @ `87f7a45d` | `formatVersion` 2 (GlossematicsCoreML `artifacts/JinaV5OmniSmall.w8a16.bundle`) | Matryoshka 32-1024-d (default 1024, `--dimensions`) | text, token IDs, image, audio, video | `query` / `document` (default) |

Other bundles (including jina-embeddings-v5-omni-nano) are refused at startup. The Jina
checkpoint is licensed CC BY-NC 4.0 (non-commercial); BidirLM is Apache-2.0. See
[jina-embeddings-v5-omni-small](#jina-embeddings-v5-omni-small) for that family; the rest of this
page up to it describes BidirLM.

## BidirLM Omni

Every BidirLM input produces one 2048-d, L2-normalized vector in a single shared space. The model
has no query or document prefixes (`task`, `role`, and `input_type` are rejected) and no
Matryoshka truncation.

The model runs from a sealed `bidirlm-omni-ane-v2` or `bidirlm-omni-streamed-v2` Core ML bundle
produced by `GlossematicsCoreML/bidirlm`. The bundle is W8A16 (int8 per-channel weights, float16
activations); the server refuses bundles with any other precision. The operator picks one compute
mode, and the server never falls back between modes:

| `--compute` | Core ML units | Notes |
| --- | --- | --- |
| `ane` (default) | CPU + Neural Engine | Startup refuses to become ready unless every function places entirely on the Neural Engine. |
| `gpu` | CPU + GPU | Same bundle and functions; placement is reported, not enforced. |
| `cpu` | CPU only | Text only. Core ML's CPU backend runs the media towers with float16 accumulation (image parity 0.96–0.98 against the FP32 source), so image, audio, and message inputs are rejected with `unsupported_modality`. |

The mode is reported in `/health`, in `/metrics` (`gloss_compute_mode`), and on every response
(`X-Glossematics-Compute`).

## Endpoints

| Route | Purpose |
| --- | --- |
| `POST /v1/embeddings` | Embeddings for text, token IDs, images, audio, and interleaved messages |
| `GET /v1/models` | The served model |
| `GET /live` | Listener liveness |
| `GET /ready`, `GET /health` | Readiness, placement, queues, program residency, and keep-warm state |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Self-contained API and operations documentation |

### Text

`input` accepts the OpenAI shapes: a string, an array of strings, one token-ID array, or an
array of token-ID arrays. Text is wrapped in the model's chat template
(`<|im_start|>user\n…<|im_end|>\n`) and mean-pooled over every token, template included, exactly
as Sentence Transformers does at the pinned revision. Each input may use the model's full
**32,768-token** context (template included). A request may carry up to 2,048 items and
`--max-request-tokens` tokens in total (default 131,072). `dimensions` must be omitted or 2048.

### Media extensions

OpenAI does not define media inputs for embeddings. The server accepts Responses-style content
objects with base64 data URLs; remote URLs and local paths are rejected.

```json
{"model": "BidirLM/BidirLM-Omni-2.5B-Embedding",
 "input": [
   {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo..."},
   {"type": "input_audio", "audio_url": "data:audio/wav;base64,UklGR..."},
   {"type": "message", "role": "user", "content": [
     {"type": "input_text", "text": "A photo of the harbor: "},
     {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQ..."},
     {"type": "input_text", "text": " and the foghorn: "},
     {"type": "input_audio", "audio_url": "data:audio/wav;base64,UklGR..."}]}]}
```

- **Images:** PNG, JPEG, or WebP, up to 20 MiB, 40 megapixels, and a 200:1 aspect ratio.
  Preprocessing reproduces the pinned Qwen2-VL processor bit for bit, as tested against the
  processor's `pixel_values`: RGB conversion with alpha composited on white and no color
  management, then `smart_resize` to the 32-pixel grid (64K to 1M pixels, so 256 to 4,096
  patches), then the processor's antialiased bicubic resample, then normalization. An image
  costs `patches / 4` tokens.
- **Audio:** WAV, up to 20 MiB. Channels are averaged. 16 kHz input is exact; other rates are
  resampled with AVAudioConverter, which differs slightly from the reference's librosa. The whole
  clip is used, with no 30-second padding or truncation: Whisper log-mel over
  `samples / 160` frames, then one token per 8 frames. That comes to about 12.5 tokens per
  second, so the 32K context holds roughly 43 minutes.
- **Message:** one user turn. Parts are concatenated with no separators, as the model's chat
  template does. Images get 3-D (MRoPE) positions and DeepStack features, following the source
  model.
- Video is rejected with `unsupported_modality`; send frames as images. Text parts must not
  contain the media control tokens (`<|image_pad|>` and similar).

`usage.prompt_tokens` counts every language-model token, including the template and media
placeholders.

## How the model runs

**Short inputs (up to 512 tokens).** Short inputs from any number of concurrent requests are
packed into one execution: up to 64 sequences in a 64- or 512-token chunk, with block-diagonal
attention and no padding waste. The language model runs as four resident *stack* programs of
seven whole layers each.

**Long inputs (over 512 tokens, up to 32K).** These run layer by layer over 512-token chunks.
Each `mid` program finishes layer *i* and projects layer *i+1*. Attention against every key
runs in a weightless bucket program between them; keys sit on the channel axis, because the
Neural Engine rejects 32K-long contractions on the innermost axis. A long input advances one
chunk operation per accelerator turn, so short requests keep flowing during a multi-minute
32K document.

**Media.** The vision and audio towers use the same layout and are stepped the same way before
their features enter the language model.

**Neural Engine program budget.** The Neural Engine holds about 124 loaded programs for the
*whole Mac*, shared by every process. Past that limit, Core ML either fails a load or silently
runs it on the CPU. The server stays inside `--ane-program-budget` (default 64):

- The eight stack programs are always resident.
- The long-input set (29 programs), attention buckets, and media towers (about 26 programs each)
  load on demand and are evicted least-recently-used.

In `ane` mode, every load is checked against the process's own Core ML log. A Neural Engine
program-creation failure becomes an error instead of a quiet CPU fallback. Startup loads every
on-demand set once, so later loads come from Core ML's cache in milliseconds. `/health` reports
`programs_resident` and `program_sets_resident`.

## jina-embeddings-v5-omni-small

The Jina runtime lives in `Sources/gloss-server/Jina/` (restored from the pre-BidirLM daemon,
branch `archive/jina-omnismall-wip`, with no dependency on the Glossematics SDK). It validates the
pinned contract, compiled function shapes, and every declared checksum before the listener opens.

- **Dimensions:** 32, 64, 128, 256, 512, or 1024 per request (`dimensions`); `--dimensions` sets
  the default. Each size has its own space ID (`X-Glossematics-Space`, `/health` `spaces`). The
  ID formula is the SDK-based daemon's, but the source revision is part of it and was re-pinned to
  `87f7a45d`, so these IDs differ from earlier builds' and their vectors must be re-embedded
  (see [Source revision re-pin](#source-revision-re-pin-87f7a45d)).
- **Retrieval roles:** texts get the model's `Query: ` or `Document: ` prompt and media the
  matching wrapper. Set `"task": "retrieval.query"` (or `retrieval.passage`), `"role"`, or
  `"input_type"` (`query`, `document`, `passage`); the default is document. Token-ID inputs are
  embedded as given.
- **Text:** up to 32,768 tokens including the prompt, never truncated. Short rows from concurrent
  requests are collected for `--batch-window-ms` into the native batch functions.
- **Images:** PNG, JPEG, WebP up to 20 MiB, resized to 0.26-1.3 megapixels.
- **Audio:** WAV up to 20 MiB and 30 seconds. A clip shorter than 480 samples at 16 kHz (30 ms)
  yields no audio token and is rejected with 400.
- **Video:** MP4 up to 32 MiB and 60 seconds (`input_video` with a `data:video/mp4;base64,`
  URL). Each source frame may be at most 4096 pixels on either edge and 4096 x 4096
  (16,777,216) pixels, checked before any frame is decoded. Frames are decoded in order with
  AVAssetReader and converted from Y'CbCr with the stream's tagged matrix (BT.601 when untagged),
  sampled like the source processor (2 fps, 4-32 frames, rounded linspace over the decoded
  frames), sized with its `smart_resize` within the tower's 2048-patch budget, and resampled with
  antialiased bicubic. `/health` reports the recipe (`video_recipe`), and a response whose request
  had a video input carries it as `X-Glossematics-Video-Recipe`. Vectors from different recipes
  share a space ID but are not comparable, so store the header with them and re-embed video when
  it changes. Text-only requests do not carry it.
- **Media preparation runs off the accelerator.** Decode, resize, mel, and prompt assembly run
  on a bounded CPU pool (about half the cores); only the Core ML predictions hold the accelerator
  lane, released on every path including errors and cancellation. A multi-second video decode
  blocks neither text nor other media. A bad input returns 400 naming its position in `input`.
- **Temporary files.** Audio and video uploads are staged in per-request
  `gloss-media-<pid>-<uuid>` directories of the temporary directory and removed when preparation
  ends. Before the listener opens, startup removes such directories a crashed or killed process
  left behind (dead owner, or older than a day) and logs one line (`examined`, `removed`,
  `bytes`). BidirLM does not use this helper and is not swept.
- Interleaved `message` inputs are not supported by this model (`unsupported_modality`).
- Placement is per function (short text and short media sequences on the Neural Engine, the rest
  on the GPU); `--compute` and `--ane-program-budget` apply to BidirLM only.

### Startup verification

`--startup-verification full|basic` (default `full`) sets what runs before `/ready` turns green.

- `full` loads and runs every Core ML function once with the accelerator lane held: all text
  buckets and batch rungs, every image, video, and audio encoder bucket, and every embed and
  decoder bucket (41 functions in the production bundle). Each output must be finite, correctly
  shaped, and unit-norm, and functions that must agree are cross-checked for self-consistency
  (no goldens): one text through every text bucket, each batch rung against the single-row
  function, and one procedural image, video clip, and audio clip through every encoder bucket.
  Each function is logged as it finishes (`verify #N text.bucket_32 ok ...`). The functions stay
  resident, so serving starts warm. **Readiness takes longer than `basic`**: the fixture verifies
  in a few seconds; a production bundle loads every function, and its first start compiles them
  all.
- `basic` runs one warm text embedding and nothing else.
- A failed check keeps the server **not ready**, as a failed BidirLM placement audit does: the
  run completes so the log lists every failing function, `startup FAILED: ... failed at
  <model.function>: <reason>` is logged, `/ready` stays 503 with `status: failed` and the same
  text in `error`, and the process stays up for inspection.
- `/health` carries a compact `startup_verification` (`mode`, `functions`, `passed`, `failed`,
  `first_failure`, `min_consistency_cosine`, `wall_ms`, `footprint_delta_bytes`); the per-function
  list is in the log.
- The keep-warm loop runs the single warm embedding; it never repeats full verification.

### Failure isolation in batched text

Rows from unrelated requests share a native text batch (up to 64). If a multi-row batch fails,
the scheduler re-runs each row alone on the single-row functions while it still holds the
accelerator, completes the rows that succeed, and fails only the requests whose own rows fail
(with that row's error). A request cancelled mid-wave is dropped without re-running its rows.
`gloss_text_isolation_retries_total` counts batches that fell back;
`gloss_text_isolation_failed_rows_total` counts rows that still failed alone.

### Source revision re-pin (87f7a45d)

**This release changes every Jina space ID (all Matryoshka sizes).** The source revision is part of
the space identity, and the old pin, `41a20a1e`, no longer resolves on the Hub, so it was re-pinned
to the snapshot actually converted, `87f7a45d1ae0265843f8569c47fb53847cb193c3`. Model weights are
unchanged, and so is the manifest `spaceID` string (`...:w8a16:native-v1`) and the identity formula;
only the revision value changed. Existing Jina indexes must be re-embedded.

This cleanly absorbs the preprocessing fixes made in the same release, found by the FP32 gate
(`make verify-jina`), so old and new vectors never share a space:

- **Image `smart_resize`** rounds half to even, like the source processor's Python `round()`. It
  affects images with a side of `k * 32 + 16` pixels, such as 720p (720 / 32 = 22.5).
- **Audio** uses `ceil(samples / 160)` mel frames, like the source. It affects clips whose length
  is not a multiple of 160 samples.
- **Stereo and multichannel audio** is averaged to mono instead of taking the left channel.

Served vectors move closer to the source model: an input affected by these fixes sits about 0.99
cosine from its earlier vector, and stereo audio about 0.94.

**Cutover.** This binary refuses bundles pinned to `41a20a1e`, and older binaries refuse re-pinned
bundles, so install the new binary together with a re-pinned bundle:

```bash
make install BUNDLE=<re-pinned bundle>
```

Re-pin an existing bundle with GlossematicsCoreML `python/converter/repin_bundle_source.py`.
Expect the roughly 7-minute startup verification before `/ready` turns green.

### Deploying on a different Apple Silicon chip

The batch thresholds (`NativeTextBatchPolicy` in `GlossTextEmbedder.swift`) and a W8 numeric fix
were calibrated on an Apple M4 Max. The startup banner and `/health` (`chip`, `macos`) report the
machine the server runs on. On other silicon, `--startup-verification full` catches numeric
faults; rerun `make bench` against the running server to recheck the batch thresholds.

Verified on `JinaV5OmniSmall.w8a16.bundle` (`make verify-jina`, `Scripts/smoke_jina.py`):

| Check | Result |
| --- | --- |
| Text vs the SDK-based daemon, both roles, 1024/256/32 dims | cosine >= 0.999998 (measured before the re-pin; the space IDs now differ by design) |
| Images (5) and audio (0.5 s, 3 s) vs the SDK-based daemon, both roles | cosine >= 0.99997; 45 s audio rejected by both |
| Video golden frames (4x256, 4x384) vs the FP32 source | 0.9972 / 0.9976 (the W8A16 tower floor) |
| MP4 files (3 clips, 4:3, 16:9 at 24 fps, portrait with an odd sample count) vs the FP32 source | 0.9964 / 0.9952 / 0.9961 |
| Mixed text + media request order | exact |

With `--startup-verification basic`, startup took about 6 s; images embed in about 0.3 s, audio
in under 0.1 s, and video in 0.4-1.6 s once warm. (Those figures predate full verification.)

## Build and run

The only external Swift package is `swift-transformers`, used for offline tokenization.

```bash
cd GlossematicsServer
make check-config BUNDLE=/path/to/BidirLMOmni.w8a16.bundle
make run BUNDLE=/path/to/BidirLMOmni.w8a16.bundle COMPUTE=ane
make run BUNDLE=/path/to/JinaV5OmniSmall.w8a16.bundle DIMENSIONS=1024
```

The first start compiles every function for the Neural Engine, which can take several
minutes; later starts use Core ML's cache. If an audit reports no placement at all for a
function, the server discards its own compiled-model cache once and retries. The first start
also runs the placement audit for all functions and one real inference before `/ready` turns
green. A Jina bundle instead runs startup verification (see above), which loads and runs every
function before `/ready` turns green.

## Tests

Two checked-in Core ML golden fixtures (`Fixtures/`, see its README) prove the serving path, not
model quality. Each is a real compiled bundle with its family's production contract, real
tokenizer, every function signature, and checksums, and each function is a no-op:

- `BidirLMOmni.dummy.bundle` returns the unit vector e0.
- `JinaV5OmniSmall.w8a16.dummy.bundle` returns `1/32 * ones(1024)`; it is regenerated with
  `make jina-fixture` (`Scripts/make_jina_dummy_bundle.py`, restored from the pre-BidirLM daemon).

The server requires `--allow-dummy` to load either one and labels every response.

```bash
make test           # unit tests + both fixtures' HTTP smoke, debug and release binaries
make test-release
make verify-model BUNDLE=/path/to/BidirLMOmni.w8a16.bundle COMPUTE=ane   # also gpu, cpu
```

`verify-model` checks the real bundle against independent FP32 evaluations of the untouched
source model:

- tokenizer and text parity through 32K (`reference/bidirlm/text_reference.json`);
- Swift media preprocessing against the processor's tensors, and image, audio, and message parity
  (`reference/bidirlm/media_reference.json`);
- an adversarial retrieval corpus;
- HTTP smoke and SIGTERM shutdown against the release binary.

Set `MAX_TOKENS=8192` to skip the slowest long cases on CPU.

```bash
make verify-jina BUNDLE=/path/to/JinaV5OmniSmall.w8a16.bundle [ORACLE=http://127.0.0.1:18800]
```

`verify-jina` builds the release binary, runs `--check-config`, then:

- the Swift video golden-frame, MP4, and retrieval end-to-end tests (`GLOSS_JINA_BUNDLE`,
  `GLOSS_JINA_REFERENCE` = GlossematicsCoreML `reference/`);
- `Scripts/smoke_jina.py`, which starts its own server and always gates text, image, and audio
  against independent FP32 references captured from the pinned Hugging Face model, with no other
  daemon needed. Text covers 14 short rows plus prose up to about 32,000 tokens, both roles, at
  every Matryoshka size; images and audio cover exact, resampled, and rescaled inputs. Each
  category has one empirical cosine floor (`FP32_FLOORS` in the script); video is gated at the
  media floor (0.995) against MP4 references; mixed-request order is checked. The references are
  `reference/jina/fp32_reference.json`, `text_reference.json`, and `video_reference.json`.
  `JINA_READY_TIMEOUT` (default 900 s) bounds the wait for startup verification.

`ORACLE=<url>` is now an optional extra: it also checks text, image, and audio parity against an
SDK-based daemon. The FP32 references come from
`GlossematicsCoreML/python/parity/export_jina_fp32_http_refs.py` and the MP4 references from
`GlossematicsCoreML/python/parity/export_video_http_refs.py`.

## CI and releases

GitHub Actions runs `make test` on every push and pull request. On `main`, on `v*` tags, and on
manual runs it also builds a Developer ID signed, notarized, and stapled disk image; tags publish
it as a GitHub Release. Setup takes five repository secrets (a Developer ID certificate and an App
Store Connect API key) and no GitHub App; see [RELEASING.md](RELEASING.md). `make package`
builds the same image locally.

## Operations

The server enforces per-request and aggregate body limits, connection limits, request
admission, loopback `Host` validation, Content-Length framing, read/write timeouts, and graceful
shutdown. Invalid bundles (checksums, contract, compiled signatures) fail before the listener
starts.

```bash
make health && make metrics
BUNDLE=/path/to/BidirLMOmni.w8a16.bundle COMPUTE=ane make install   # launchd agent
BUNDLE=/path/to/JinaV5OmniSmall.w8a16.bundle DIMENSIONS=1024 make install
make status && make logs
```

If `ane` startup fails with "no free program slots", another process is holding Neural Engine
models. Stop it, or lower `--ane-program-budget`, which costs more on-demand reloads.

## Layout

- `Sources/gloss-server/BidirLM/`: bundle validation, manifest contract, text and media
  encoders, program residency, placement audit, tokenizer, and multimodal sequence assembly
  (MRoPE, DeepStack).
- `Sources/gloss-server/Jina/`: the jina-embeddings-v5-omni-small runtime (bundle validation,
  text batching, image, audio, and video pipelines, startup verification, temporary-file sweep)
  and its HTTP service.
- `Sources/gloss-server/ModelFamily.swift`: manifest-based family detection.
- `Sources/gloss-server/Media/`: image decode and resample, WAV decode, and Whisper log-mel.
- `Sources/gloss-server/`: HTTP transport, scheduler (packing and stepped long/media work),
  service, API schema, and docs.
- `Fixtures/`: the no-op golden fixtures for both families and two test videos (`make fixture`
  writes the BidirLM fixture's 622 MB zero token table).
- `reference/`: FP32 text and media goldens and the retrieval corpus (BidirLM), and MP4 video
  references (Jina).
- `Scripts/`: launchd agent, fixture generators, HTTP smokes, release packaging
  (`package_release.sh`), and benchmarks.
- `.github/workflows/ci.yml`: tests on every push; signed, notarized disk images on `main` and
  tags ([RELEASING.md](RELEASING.md)).
