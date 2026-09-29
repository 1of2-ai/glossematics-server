"""HTTP smoke and parity check of a jina-embeddings-v5-omni-small bundle served by gloss-server.

Starts its own server on a free loopback port and stops only that child. Checks:

* the Jina contract: family detection, Matryoshka sizes and per-size spaces, retrieval roles,
  clear rejections (bad dimensions or role, interleaved messages, out-of-vocabulary token IDs);
* text, image, audio and video against INDEPENDENT FP32 source-model references, always (no
  oracle needed). The references are captured from the pinned Hugging Face model, not from any
  Core ML bundle or daemon, so they measure conversion fidelity:
    - text: the 14 short rows of ``--text-reference`` (``reference/jina/text_reference.json``) plus
      seeded prose of ~750, ~5000, ~14000 and ~32000 tokens (buckets 1024, 8192, 16384, 32768) from
      ``--fp32-reference`` (``reference/jina/fp32_reference.json``), both roles, Matryoshka 1024..32;
    - images: 5 PNGs (three where the server does not resample, one downscaled, one upscaled to the
      minimum pixel budget; the largest encoder bucket of 5120 patches is covered), both roles;
    - audio: 16 kHz mono WAV at 0.5 s, 3 s and 29.5 s, and a 44.1 kHz stereo clip that takes the
      server's resample path (AVAudioConverter, not bit-identical to librosa), both roles;
    - video: HF references from ``--video-reference``, gated at the media floor.
  Regenerate the FP32 references with
  ``GlossematicsCoreML/python/parity/export_jina_fp32_http_refs.py`` (see its docstring); the
  floors are documented next to ``FP32_FLOORS`` below;
* optionally (``--oracle``, e.g. the SDK-based server on 127.0.0.1:11435), text, image and audio
  parity against a second daemon that takes local file paths (``image_path`` / ``audio_path``)
  and ``task``. Same bundle, same space identity: cosine must be >= 0.999 and the space equal.
  This validates the Swift port against the older daemon and is an extra, not the quality gate;
* request order in a mixed text/media request.

    python Scripts/smoke_jina.py --server-bin BIN --bundle JinaV5OmniSmall.w8a16.bundle \\
        --output DIR --video-reference reference/jina/video_reference.json [--oracle http://127.0.0.1:11435]
"""
import argparse
import base64
import hashlib
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import test_server_http as http

MODEL = "jinaai/jina-embeddings-v5-omni-small"
TEXTS = [
    "How do I cool an overheating laptop?",
    "Bonjour ! 你好。 A mixed-language sentence with numbers 12345 and symbols #@!",
    "The mitochondria is the powerhouse of the cell.",
    " ".join(["Retrieval quality depends on the whole text-to-vector contract."] * 60),
]
PARITY = 0.999          # oracle daemon: same bundle lineage, so near-exact agreement is expected
MEDIA_FLOOR = 0.995     # video vs the HF FP32 references (the W8A16 vision tower's floor)

# ----- FP32 source-model gate -----------------------------------------------------------------
# Cosine of the served embedding against the pinned FP32 source model at every Matryoshka size
# (truncate + renormalize on both sides), judged against one floor per category. FLOORS ARE
# EMPIRICAL: they were set from repeated runs of the released JinaV5OmniSmall.w8a16.bundle on the
# HEAD server (M4 Max, macOS 27), as the observed minimum over every case, both roles and all six
# sizes of the category, minus a margin, rounded down to 0.0005. The margin covers run-to-run and
# chip-to-chip wobble of the Neural Engine/GPU numerics; it is 0.0015 for the categories whose
# observed minimum is >= 0.997, 0.002 for images (the vision tower is the least accurate part of
# the W8A16 bundle, cf. MEDIA_FLOOR), and larger where the server resamples with its own kernel.
# A floor is "no worse than the shipped W8A16 quality"; a conversion that lands below one is a
# regression to investigate, not a floor to lower.
#
#   category           cases (both roles, all 6 sizes)                   observed min  margin  floor
#   text_short         14 short rows (text_reference.json)                     0.999481   0.0015  0.9975
#   text_medium        prose_750tok (bucket 1024)                              0.998486   0.0015  0.9965
#   text_long          prose_5000 / 14000 / 32000tok (buckets 8192..32768)     0.997783   0.0015  0.9960
#   image_exact        512x512, 768x1024, 1280x1024 (5120 patches)             0.995863   0.0020  0.9935
#   image_downscaled   1920x1080 -> 1504x832                                  0.996248   0.0020  0.9940
#   image_upscaled     200x150 -> 608x448                                      0.977934   0.0080  0.9695
#   audio_exact        16 kHz mono 0.5 s, 3 s, 29.5 s                          0.998492   0.0015  0.9965
#   audio_resampled    44.1 kHz mono 3 s (AVAudioConverter vs soxr_hq)         0.971539   0.0100  0.9615
#
# Repeated runs of the same binary and bundle agreed to 6 decimals, so the margins are for other
# chips and OS releases, not for run-to-run noise. Cosines of the vectors truncated to 512..32 are
# typically higher than the full-size one (the lowest observed for a category is at 1024 except for
# audio, where it is at 512), so one floor per category serves all sizes.
#
# "exact" media inputs need no resampling on the server (the smart-resized size equals the source
# size, or the audio is already 16 kHz mono), so only host preprocessing and the converted model
# differ from the source. "downscaled" / "upscaled" images and "resampled" audio add the server's
# own resampler (CoreGraphics, AVAudioConverter), which is not bit-identical to the reference's
# (PIL/torchvision bicubic, librosa soxr_hq), hence looser floors; CoreGraphics `.high` in
# particular differs noticeably from bicubic when enlarging (the video path carries its own cubic
# kernel for that reason, the image path does not yet).
MATRYOSHKA_DIMS = (1024, 512, 256, 128, 64, 32)
FP32_FLOORS = {
    "text_short": 0.9975,       # the 14 rows of text_reference.json, batched per role, server-side sizes
    "text_medium": 0.9965,      # ~750 tokens, the single-row 1024 bucket
    "text_long": 0.996,         # ~5000, ~14000 and ~32000 tokens: buckets 8192, 16384, 32768
    "image_exact": 0.9935,
    "image_downscaled": 0.994,
    "image_upscaled": 0.9695,
    "audio_exact": 0.9965,
    "audio_resampled": 0.9615,
}
ROLE_GAP_MIN = 0.005    # cases whose two role references differ by at least this (1 - cosine) must also
                        # be closer to their own role's reference than to the other role's
ROLE_TASK = {"query": "retrieval.query", "document": "retrieval.passage"}

# Server divergences from the source model that are KNOWN and NOT yet fixed. Each is a real
# preprocessing difference (not model quality), kept as a reference case so the moment the server is
# fixed the gate says so: a listed case is judged as "must still diverge" (below its category floor
# but above KNOWN_DIVERGENCE_SANITY); once it meets the floor the run FAILS with an instruction to
# delete the entry, and the case becomes a normal hard gate. Every run prints them, and the report
# carries their cosines under checks.fp32.known_divergences.
KNOWN_DIVERGENCE_SANITY = 0.90
KNOWN_SERVER_DIVERGENCES = {}
FP32_REFERENCE_FORMAT = "glossematics-jina-fp32-reference-v1"


def embed(base, inputs, **extra):
    return http.call(f"{base}/v1/embeddings", {"model": MODEL, "input": inputs, **extra}, timeout=900)


def vectors(status, body, context):
    assert status == 200, (context, status, body)
    return [d["embedding"] for d in body["data"]]


def unit_prefix(vector, dims):
    """Matryoshka: the first `dims` values, renormalized."""
    head = vector[:dims]
    scale = http.norm(head)
    return [x / scale for x in head]


def fp32_gate(base, args, checks):
    """Gate text, image and audio against the independent FP32 source-model references.

    Every case is judged against the floor of its category at every Matryoshka size. All cosines
    are collected into the report before failing, so one run shows the whole picture.
    """
    ref = json.loads(args.fp32_reference.read_text())
    assert ref.get("format") == FP32_REFERENCE_FORMAT, (args.fp32_reference, ref.get("format"))
    short_rows = json.loads(args.text_reference.read_text())
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    bundle_revision = (manifest.get("source") or {}).get("revision")
    assert bundle_revision == ref["source"]["pinned_revision"], (
        f"bundle converted from source revision {bundle_revision}, but the FP32 references were captured "
        f"from {ref['source']['pinned_revision']}: regenerate them (export_jina_fp32_http_refs.py)")
    result = {"reference": {k: ref[k] for k in ("source", "environment", "generator", "captured_at")},
              "floors": FP32_FLOORS, "text_short": {}, "text": {}, "image": {}, "audio": {}}
    below, minima = {}, {}          # category -> {case: worst cosine} / the lowest cosine seen and where

    def judge(category, case, cosines):
        """Record `cosines` (dims -> cosine) of one case against its category floor."""
        worst = min(cosines.values())
        seen = minima.setdefault(category, {"cosine": 2.0})
        if worst < seen["cosine"]:
            minima[category] = {"cosine": worst, "case": case, "dims": min(cosines, key=cosines.get)}
        if worst < FP32_FLOORS[category]:
            below.setdefault(category, {})[case] = worst

    def cosines_of(served, expected):
        return {dims: http.cosine(unit_prefix(served, dims), unit_prefix(expected, dims)) for dims in MATRYOSHKA_DIMS}

    problems = []

    def check_role(case, role, entry, references):
        """The served vector must be nearer its own role's reference than the other role's, wherever
        the two references are far enough apart for that to be meaningful."""
        gap = 1 - http.cosine(references["query"]["embedding"], references["document"]["embedding"])
        entry["role_gap"] = gap
        if gap >= ROLE_GAP_MIN and entry["role_margin"] <= 0:
            problems.append(f"{case}/{role}: closer to the {'document' if role == 'query' else 'query'} reference "
                            f"than its own (margin {entry['role_margin']:.5f}, role gap {gap:.4f})")

    # Short text: the FP32 capture holds every Matryoshka size, so the server truncates here.
    for role in ROLE_TASK:
        rows = [r for r in short_rows if r["prompt_name"] == role]
        for dims in MATRYOSHKA_DIMS:
            status, body, _ = embed(base, [r["text"] for r in rows], task=ROLE_TASK[role], dimensions=dims)
            ours = vectors(status, body, f"short text {role} {dims}")
            assert len(ours) == len(rows) and all(len(v) == dims for v in ours), (role, dims)
            cosines = [http.cosine(v, r["embedding"][str(dims)]) for v, r in zip(ours, rows)]
            for r, c in zip(rows, cosines):
                judge("text_short", f"short[{r['idx']}]/{role} @{dims}", {dims: c})
            if body["usage"]["prompt_tokens"] != sum(r["n_tokens"] for r in rows):
                problems.append(f"short/{role}/{dims}: the server counted {body['usage']['prompt_tokens']} prompt "
                                f"tokens, the source tokenizer {sum(r['n_tokens'] for r in rows)}")
            result["text_short"][f"{role}/{dims}"] = {"cosines": cosines, "rows": [r["idx"] for r in rows]}
    http.expect(True, f"FP32 short text ({len(short_rows)} rows, both roles, 6 sizes): min cosine "
                      f"{minima['text_short']['cosine']:.6f}")

    # Long text: one request per role at 1024 (the vector is truncated here for smaller sizes); the
    # single-row 1024 bucket also gets server-side dimensions, which the short rows cover per size.
    for case in ref["text"]:
        category = "text_medium" if case["bucket"] <= 1024 else "text_long"
        for role, data in case["roles"].items():
            t0 = time.monotonic()
            status, body, _ = embed(base, [case["text"]], task=ROLE_TASK[role], dimensions=1024)
            seconds = time.monotonic() - t0
            served = vectors(status, body, f"{case['name']} {role}")[0]
            other = "document" if role == "query" else "query"
            entry = {"n_tokens": data["n_tokens"], "bucket": case["bucket"], "seconds": seconds,
                     "prompt_tokens": body["usage"]["prompt_tokens"], "cosine": cosines_of(served, data["embedding"]),
                     "role_margin": (http.cosine(served, data["embedding"])
                                     - http.cosine(served, case["roles"][other]["embedding"]))}
            if entry["prompt_tokens"] != data["n_tokens"]:
                problems.append(f"{case['name']}/{role}: the server counted {entry['prompt_tokens']} prompt tokens, "
                                f"the source tokenizer {data['n_tokens']}")
            check_role(case["name"], role, entry, case["roles"])
            cosines = dict(entry["cosine"])
            if category == "text_medium":
                for dims in (256, 32):
                    status, body, _ = embed(base, [case["text"]], task=ROLE_TASK[role], dimensions=dims)
                    theirs = vectors(status, body, f"{case['name']} {role} {dims}")[0]
                    assert len(theirs) == dims
                    entry.setdefault("server_dimensions", {})[dims] = http.cosine(theirs, unit_prefix(data["embedding"], dims))
                    cosines[f"server {dims}"] = entry["server_dimensions"][dims]
            judge(category, f"{case['name']}/{role}", cosines)
            result["text"][f"{case['name']}/{role}"] = entry
    http.expect(True, "FP32 long text, cosine @1024 (seconds): " + ", ".join(
        f"{k.split('/')[0].removeprefix('prose_')}/{k.split('/')[1][0]} {v['cosine'][1024]:.6f} ({v['seconds']:.0f}s)"
        for k, v in result["text"].items()))

    # Image and audio: one file per case, both roles. Besides the cosine to the same-role reference,
    # `role_margin` records how much closer the served vector is to it than to the other role's.
    known = {}
    for kind, key, mime in (("image", "input_image", "image/png"), ("audio", "input_audio", "audio/wav")):
        for case in ref[kind]:
            data = (args.fp32_reference.parent / case["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == case["sha256"], f"{case['file']} differs from the reference's hash"
            item = {"type": key, f"{kind}_url": f"data:{mime};base64," + base64.b64encode(data).decode()}
            served, per_role = {}, {}
            for role in case["roles"]:
                t0 = time.monotonic()
                status, body, _ = embed(base, [item], task=ROLE_TASK[role])
                seconds = time.monotonic() - t0
                served[role] = vectors(status, body, f"{case['name']} {role}")[0]
                assert len(served[role]) == 1024, len(served[role])
                per_role[role] = {"file": case["file"], "path": case["path"], "seconds": seconds,
                                  "cosine": cosines_of(served[role], case["roles"][role]["embedding"])}
            for role, other in (("query", "document"), ("document", "query")):
                per_role[role]["role_margin"] = (http.cosine(served[role], case["roles"][role]["embedding"])
                                                 - http.cosine(served[role], case["roles"][other]["embedding"]))
            name, category = f"{kind}:{case['name']}", f"{kind}_{case['path']}"
            if name in KNOWN_SERVER_DIVERGENCES:
                worst = min(min(e["cosine"].values()) for e in per_role.values())
                known[name] = {"category": category, "min_cosine": worst, "reason": KNOWN_SERVER_DIVERGENCES[name],
                               "roles": {r: e["cosine"] for r, e in per_role.items()}}
                print(f"  KNOWN DIVERGENCE (not gated) {name}: min cosine {worst:.4f} < {category} floor "
                      f"{FP32_FLOORS[category]}; {KNOWN_SERVER_DIVERGENCES[name].split(';')[0]}", flush=True)
                if worst >= FP32_FLOORS[category]:
                    problems.append(f"{name} now meets the {category} floor ({worst:.6f}): delete it from "
                                    f"KNOWN_SERVER_DIVERGENCES so that it is gated")
                elif worst < KNOWN_DIVERGENCE_SANITY:
                    problems.append(f"{name} is worse than its documented divergence (min cosine {worst:.4f} < "
                                    f"{KNOWN_DIVERGENCE_SANITY})")
            else:
                for role, entry in per_role.items():
                    judge(category, f"{name}/{role}", entry["cosine"])
                    check_role(name, role, entry, case["roles"])
            for role, entry in per_role.items():
                result[kind][f"{case['name']}/{role}"] = entry
        http.expect(True, f"FP32 {kind}, cosine @1024: " + ", ".join(
            f"{k.split('/')[0]}/{k.split('/')[1][0]} {v['cosine'][1024]:.6f}" for k, v in result[kind].items()))
    result["known_divergences"] = known
    stale = sorted(set(KNOWN_SERVER_DIVERGENCES) - set(known))
    if stale:
        problems.append(f"KNOWN_SERVER_DIVERGENCES names cases absent from the reference: {stale}")

    result["observed_min"] = minima
    checks["fp32"] = result
    for category, cases in below.items():
        worst = sorted(cases.items(), key=lambda kv: kv[1])[:6]
        problems.append(f"{category}: {len(cases)} case(s) below the floor {FP32_FLOORS[category]}: " +
                        ", ".join(f"{c} {v:.6f}" for c, v in worst) + (" ..." if len(cases) > len(worst) else ""))
    assert not problems, "FP32 parity gate failed:\n  " + "\n  ".join(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", help="base URL of an independent Jina daemon for parity")
    parser.add_argument("--video-reference", type=Path)
    parser.add_argument("--fp32-reference", type=Path, default=http.ROOT / "reference" / "jina" / "fp32_reference.json",
                        help="FP32 source-model references for text (long), image and audio, with their media files "
                             "(export_jina_fp32_http_refs.py)")
    parser.add_argument("--ready-timeout", type=float, default=900,
                        help="seconds to wait for /ready (startup may verify every Core ML function)")
    parser.add_argument("--text-reference", type=Path, default=http.ROOT / "reference" / "jina" / "text_reference.json",
                        help="FP32 source-model references for the 14 short text rows (capture_reference.py)")
    parser.add_argument("--media-reference", type=Path,
                        default=http.ROOT / "reference" / "bidirlm" / "media_reference.json",
                        help="PNG/WAV inputs (the embeddings in it are BidirLM's and are not used)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(server_bin=str(args.server_bin.resolve()), bundle=args.bundle.resolve(),
                               compute=None, fixture=False, extra_args=[])
    port = http.free_port()
    base = f"http://127.0.0.1:{port}"
    report = {"bundle": str(args.bundle.resolve()), "oracle": args.oracle, "checks": {}, "pass": False}
    started = time.monotonic()
    process = http.start(settings, port, args.output / "server.log")
    try:
        health = http.wait_ready(base, process, args.ready_timeout)
        report["ready_seconds"] = time.monotonic() - started
        report["health"] = health
        assert health["family"] == "jina-embeddings-v5-omni-small", health
        assert health["supported_dimensions"] == [32, 64, 128, 256, 512, 1024], health
        assert health["modalities"] == ["text", "image", "audio", "video"], health
        http.expect(True, f"ready in {report['ready_seconds']:.1f}s; modalities {health['modalities']}")

        # Contract rejections.
        for body, status, what in [
            ({"input": "x", "dimensions": 100}, 400, "unsupported Matryoshka size"),
            ({"input": "x", "task": "text-matching"}, 400, "unsupported task"),
            ({"input": [{"type": "message", "content": [{"type": "input_text", "text": "x"}]}]}, 400, "message input"),
            ({"input": [[151672]]}, 400, "out-of-vocabulary token ID"),
            ({"input": ["  "]}, 400, "blank text"),
        ]:
            got, payload, _ = http.call(f"{base}/v1/embeddings", {"model": MODEL, **body})
            http.expect(got == status, f"rejects {what} ({got}: {payload.get('error', {}).get('message', '')[:90]})")

        # Text: roles and Matryoshka against the oracle.
        checks = report["checks"]
        text = {}
        for role in ("retrieval.query", "retrieval.passage"):
            oracle_full = None
            if args.oracle:
                # The oracle loads one model per requested size, so it is asked for 1024 only;
                # smaller sizes are its 1024 vector truncated and re-normalized (its own recipe).
                ostatus, obody, oheaders = http.call(
                    f"{args.oracle}/v1/embeddings",
                    {"model": MODEL, "input": TEXTS, "task": role, "dimensions": 1024}, timeout=900)
                oracle_full = vectors(ostatus, obody, "oracle text")
                oracle_space = oheaders.get("X-Glossematics-Space")
            for dims in (1024, 256, 32):
                status, body, headers = embed(base, TEXTS, task=role, dimensions=dims)
                ours = vectors(status, body, f"text {role} {dims}")
                assert all(len(v) == dims and abs(http.norm(v) - 1) < 2e-3 for v in ours)
                entry = {"space": headers.get("X-Glossematics-Space"), "role": headers.get("X-Glossematics-Role"),
                         "prompt_tokens": body["usage"]["prompt_tokens"]}
                if oracle_full is not None:
                    theirs = [v[:dims] for v in oracle_full]
                    entry["cosines"] = [http.cosine(a, b) for a, b in zip(ours, theirs)]
                    if dims == 1024:
                        entry["oracle_space"] = oracle_space
                        assert entry["space"] == oracle_space, entry
                    assert min(entry["cosines"]) >= PARITY, entry
                text[f"{role}/{dims}"] = entry
                if dims == 1024:
                    text.setdefault("_vectors", {})[role] = ours
        q, d = text["_vectors"]["retrieval.query"], text["_vectors"]["retrieval.passage"]
        role_gap = [http.cosine(a, b) for a, b in zip(q, d)]
        assert max(role_gap) < 0.999, role_gap     # the role changes the encoding
        text.pop("_vectors")
        checks["text"] = text
        http.expect(True, "text roles x dims" + (f" match the oracle (min cosine {min(min(v['cosines']) for v in text.values()):.6f})" if args.oracle else ""))

        # Image and audio.
        media = json.loads(args.media_reference.read_text())
        tmp = Path(tempfile.mkdtemp(prefix="jina-smoke-"))
        media_checks = {}
        for case in media["cases"]:
            if len(case["parts"]) != 1 or case["parts"][0]["type"] == "text":
                continue
            part = case["parts"][0]
            kind = part["type"]
            payload = part["png"] if kind == "image" else part["wav"]
            item = ({"type": "input_image", "image_url": "data:image/png;base64," + payload} if kind == "image"
                    else {"type": "input_audio", "audio_url": "data:audio/wav;base64," + payload})
            path = tmp / (case["name"] + (".png" if kind == "image" else ".wav"))
            path.write_bytes(base64.b64decode(payload))
            for role in ("retrieval.query", "retrieval.passage"):
                t0 = time.monotonic()
                status, body, headers = embed(base, [item], task=role)
                seconds = time.monotonic() - t0
                entry = {"status": status, "seconds": seconds}
                if status == 200:
                    entry["norm"] = http.norm(body["data"][0]["embedding"])
                else:
                    entry["error"] = body.get("error", {}).get("message")
                if args.oracle:
                    key = "image_path" if kind == "image" else "audio_path"
                    ostatus, obody, _ = http.call(f"{args.oracle}/v1/embeddings",
                                                  {"model": MODEL, "input": [{key: str(path)}], "task": role}, timeout=900)
                    entry["oracle_status"] = ostatus
                    assert (status == 200) == (ostatus == 200), (case["name"], entry, obody)
                    if status == 200:
                        entry["cosine"] = http.cosine(body["data"][0]["embedding"], obody["data"][0]["embedding"])
                        assert entry["cosine"] >= PARITY, (case["name"], entry)
                media_checks[f"{case['name']}/{role}"] = entry
        checks["media"] = media_checks
        cosines = [v["cosine"] for v in media_checks.values() if "cosine" in v]
        rejected = sorted({k.split("/")[0] for k, v in media_checks.items() if v["status"] != 200})
        http.expect(True, f"image/audio: {len(cosines)} embedded" + (f", min cosine vs oracle {min(cosines):.6f}" if cosines else "")
                    + (f"; rejected like the oracle: {rejected}" if rejected else ""))

        # Text (short and long), image and audio against the independent FP32 source references.
        fp32_gate(base, args, checks)

        # Video against HF references.
        if args.video_reference:
            ref = json.loads(args.video_reference.read_text())
            video_checks = {}
            for case in ref["cases"]:
                item = {"type": "input_video", "video_url": "data:video/mp4;base64," + case["mp4"]}
                t0 = time.monotonic()
                status, body, _ = embed(base, [item], task=case.get("task", "retrieval.passage"))
                seconds = time.monotonic() - t0
                vec = vectors(status, body, case["name"])[0]
                status2, body2, _ = embed(base, [item], task=case.get("task", "retrieval.passage"))
                cos = http.cosine(vec, case["embedding"])
                video_checks[case["name"]] = {"cosine_vs_hf_fp32": cos, "seconds": seconds,
                                              "repeat_cosine": http.cosine(vec, vectors(status2, body2, "repeat")[0]),
                                              "reference": {k: case[k] for k in ("sampled_indices", "grid_thw") if k in case}}
                assert cos >= MEDIA_FLOOR, (case["name"], cos)
            checks["video"] = video_checks
            http.expect(True, "video vs HF FP32: " + ", ".join(f"{k} {v['cosine_vs_hf_fp32']:.6f}" for k, v in video_checks.items()))

        # Mixed request keeps order and equals single-item results.
        img = next(c for c in media["cases"] if c["name"] == "image_noise_300x420")["parts"][0]["png"]
        wav = next(c for c in media["cases"] if c["name"] == "audio_0.5s")["parts"][0]["wav"]
        mixed = [TEXTS[0], {"type": "input_image", "image_url": "data:image/png;base64," + img}, TEXTS[2],
                 {"type": "input_audio", "audio_url": "data:audio/wav;base64," + wav}]
        together = vectors(*embed(base, mixed)[:2], "mixed")
        alone = [vectors(*embed(base, [x])[:2], "single")[0] for x in mixed]
        order = [http.cosine(a, b) for a, b in zip(together, alone)]
        assert min(order) >= 0.9999, order
        checks["mixed_order"] = order
        http.expect(True, f"mixed request keeps order (min cosine vs single {min(order):.6f})")
        report["pass"] = True
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        process.terminate()
        try:
            process.wait(timeout=30)
        except http.subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print(json.dumps({"pass": report["pass"]}), flush=True)


if __name__ == "__main__":
    main()
