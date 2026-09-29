"""HTTP smoke and parity check of a jina-embeddings-v5-omni-small bundle served by gloss-server.

Starts its own server on a free loopback port and stops only that child. Checks:

* the Jina contract: family detection, Matryoshka sizes and per-size spaces, retrieval roles,
  clear rejections (bad dimensions or role, interleaved messages, out-of-vocabulary token IDs);
* text, image, and audio parity against an ORACLE daemon (``--oracle``, e.g. the SDK-based
  server on 127.0.0.1:11435), which takes local file paths (``image_path`` / ``audio_path``)
  and ``task``. Same bundle, same space identity: cosine must be >= 0.999 and the space equal;
* video against HF references (``--video-reference``, from
  ``GlossematicsCoreML/python/parity/export_video_http_refs.py``): FP32 source recipe vs the
  W8A16 bundle, gated at the media floor cosine >= 0.995;
* request order in a mixed text/media request.

    python Scripts/smoke_jina.py --server-bin BIN --bundle JinaV5OmniSmall.w8a16.bundle \\
        --output DIR --oracle http://127.0.0.1:11435 --video-reference reference/jina/video_reference.json
"""
import argparse
import base64
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
PARITY = 0.999
MEDIA_FLOOR = 0.995


def embed(base, inputs, **extra):
    return http.call(f"{base}/v1/embeddings", {"model": MODEL, "input": inputs, **extra}, timeout=900)


def vectors(status, body, context):
    assert status == 200, (context, status, body)
    return [d["embedding"] for d in body["data"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", help="base URL of an independent Jina daemon for parity")
    parser.add_argument("--video-reference", type=Path)
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
        health = http.wait_ready(base, process, 900)
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
