#!/usr/bin/env python3
"""End-to-end HTTP checks for gloss-server with a real Core ML bundle.

Default: the no-op golden fixture (fast; exercises validation, tokenization, packing, the long
path, Core ML execution, and HTTP). With --full-model on a sealed BidirLM bundle it also gates
numerical parity against independent FP32 goldens and adversarial retrieval ranking.

    python3 Scripts/test_server_http.py --server-bin .build/release/gloss-server
    python3 Scripts/test_server_http.py --server-bin ... --bundle BUNDLE --compute gpu --full-model
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "Fixtures" / "BidirLMOmni.dummy.bundle"
MODEL = "BidirLM/BidirLM-Omni-2.5B-Embedding"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url: str, body=None, *, timeout: float = 60, headers=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    hdrs = {"Content-Type": "application/json"} if data is not None else {}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        payload = response.read()
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = payload.decode(errors="replace")
        return response.status, parsed, response.headers


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}", flush=True)


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


def embed(base: str, inputs, timeout: float = 900, **extra):
    return call(f"{base}/v1/embeddings", {"model": MODEL, "input": inputs, **extra}, timeout=timeout)


def start(args, port: int, log_path: Path) -> subprocess.Popen:
    cmd = [args.server_bin, "--bundle", str(args.bundle), "--port", str(port),
           "--access-log", "all", "--keep-warm-seconds", "0", "--batch-window-ms", "3"]
    if getattr(args, "compute", None):      # BidirLM only; Jina bundles place functions themselves
        cmd += ["--compute", args.compute]
    cmd += list(getattr(args, "extra_args", []) or [])
    if args.fixture:
        cmd.append("--allow-dummy")
    log = log_path.open("w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def wait_ready(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited early with {process.returncode}")
        try:
            status, body, _ = call(f"{base}/ready", timeout=5)
            if status == 200:
                return body
            if isinstance(body, dict) and body.get("status") == "failed":
                raise AssertionError(f"startup failed: {body.get('error')}")
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    raise AssertionError("server did not become ready")


def metric(base: str, name: str) -> float:
    _, text, _ = call(f"{base}/metrics")
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def contract_checks(base: str, args, health: dict) -> None:
    expect(health["ready"] and health["dimensions"] == 2048, "ready with 2048-d output")
    expect(health["compute"] == args.compute, f"compute mode reported as {args.compute}")
    expect(health["fixture"] == args.fixture, "fixture flag matches the bundle")
    expect(health["space"].endswith(":2048:w8a16:ane-chunked-mean-v1"), "space identity is BidirLM's")
    expect(health["placement_verified"], "placement was audited at startup")
    if args.compute == "ane" and not args.fixture:
        expect(health["placement_strict_ane"] is True, "every function is placed on the Neural Engine")

    status, body, _ = call(f"{base}/v1/models")
    expect(status == 200 and body["data"][0]["id"] == MODEL, "/v1/models lists the served model")

    status, body, headers = embed(base, "How do I cool an overheating laptop?")
    vector = body["data"][0]["embedding"]
    expect(status == 200 and len(vector) == 2048 and abs(norm(vector) - 1) < 1e-3, "single string embeds to a unit 2048-d vector")
    expect(headers["X-Glossematics-Dimensions"] == "2048" and headers["X-Glossematics-Compute"] == args.compute,
           "dimension and compute headers")
    expect(body["usage"]["prompt_tokens"] == 14, "usage counts the chat-templated tokens")
    if args.fixture:
        expect(headers.get("X-Glossematics-Dummy") == "true", "fixture responses are labeled")

    status, body, _ = embed(base, ["one", "two", "three"], encoding_format="base64", dimensions=2048)
    decoded = [list(struct.unpack("<2048f", base64.b64decode(item["embedding"]))) for item in body["data"]]
    expect(status == 200 and [d["index"] for d in body["data"]] == [0, 1, 2]
           and all(abs(norm(v) - 1) < 1e-3 for v in decoded), "array input, base64 float32, explicit 2048")

    status, body, _ = embed(base, [[4616], [4340, 653]])
    expect(status == 200 and len(body["data"]) == 2, "token-ID inputs are templated and embedded")

    status, body, _ = embed(base, "x", dimensions=1024)
    expect(status == 400 and body["error"]["param"] == "dimensions", "Matryoshka dimensions are rejected")
    status, body, _ = embed(base, "x", task="retrieval.query")
    expect(status == 400 and body["error"]["param"] == "task", "retrieval roles are rejected (BidirLM has no prompts)")
    status, body, _ = embed(base, "   ")
    expect(status == 400, "blank text is rejected")
    status, body, _ = embed(base, [[151936]])
    expect(status == 400, "out-of-vocabulary token IDs are rejected")
    status, body, _ = embed(base, " word" * 40000)
    expect(status == 400 and "32768" in body["error"]["message"], "inputs over 32768 templated tokens are rejected")
    video = "data:video/mp4;base64," + base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()
    status, body, _ = embed(base, {"type": "input_video", "video_url": video})
    expect(status == 400 and body["error"]["code"] == "unsupported_modality", "video is rejected explicitly")
    if "image" not in health["modalities"]:
        # Text-only bundles, and cpu mode (media towers are qualified on ane and gpu only).
        image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        status, body, _ = embed(base, {"type": "input_image", "image_url": image})
        expect(status == 400 and body["error"]["code"] == "unsupported_modality",
               "media is rejected when the bundle has no media towers")
    status, body, _ = call(f"{base}/v1/embeddings", {"model": "other", "input": "x"})
    expect(status == 404, "unknown model is 404")
    status, _, _ = call(f"{base}/v1/embeddings", raw=b"{}", headers={"Content-Type": "text/plain"})
    expect(status == 415, "non-JSON content type is 415")
    with socket.create_connection(("127.0.0.1", int(base.rsplit(":", 1)[1])), timeout=5) as sock:
        sock.sendall(b"GET /live HTTP/1.1\r\nHost: evil.example\r\n\r\n")
        line = sock.recv(128).split(b"\r\n", 1)[0]
        expect(b" 400 " in line + b" ", f"non-loopback Host is refused ({line.decode(errors='replace')})")


def batching_checks(base: str) -> None:
    before_waves = metric(base, "gloss_text_waves_total")
    texts = [f"short query number {i} about topic {i % 7}" for i in range(48)]

    def one(text):
        status, body, _ = embed(base, text)
        return status, body["data"][0]["embedding"] if status == 200 else None

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        results = list(pool.map(one, texts))
    expect(all(s == 200 and abs(norm(v) - 1) < 1e-3 for s, v in results), "48 concurrent requests succeed")
    waves = metric(base, "gloss_text_waves_total") - before_waves
    expect(waves < len(texts), f"concurrent requests were packed ({int(waves)} executions for {len(texts)} inputs)")
    expect(metric(base, "gloss_coalesced_waves_total") >= 1, "at least one execution served several requests")

    long_text = " ".join(f"sentence {i} discusses retrieval, attention, and chunking." for i in range(260))
    started = time.time()
    status, body, _ = embed(base, long_text)
    expect(status == 200 and body["usage"]["prompt_tokens"] > 512, f"long document embeds via the chunked path ({body['usage']['prompt_tokens']} tokens, {time.time() - started:.1f}s)")
    expect(metric(base, "gloss_long_documents_total") >= 1, "long-document metric recorded")


def synthetic_png(h: int, w: int) -> bytes:
    """A small valid RGB PNG built with zlib (no image libraries needed)."""
    import struct
    import zlib
    rows = []
    for y in range(h):
        rows.append(b"\x00" + b"".join(bytes(((x * 7 + y * 3) % 256, (x * 5) % 256, (y * 11) % 256)) for x in range(w)))
    raw = b"".join(rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def synthetic_wav(seconds: float, rate: int = 16000) -> bytes:
    import io
    import math
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()


def media_checks(base: str, health: dict) -> None:
    if "image" not in health["modalities"]:
        return
    image = "data:image/png;base64," + base64.b64encode(synthetic_png(300, 420)).decode()
    audio = "data:audio/wav;base64," + base64.b64encode(synthetic_wav(3.0)).decode()
    status, body, _ = embed(base, {"type": "input_image", "image_url": image})
    expect(status == 200 and len(body["data"][0]["embedding"]) == 2048, "image input embeds")
    expect(body["usage"]["prompt_tokens"] == 124, f"300x420 image uses 124 tokens ({body['usage']['prompt_tokens']})")
    status, body, _ = embed(base, {"type": "input_audio", "audio_url": audio})
    expect(status == 200 and body["usage"]["prompt_tokens"] == 45, f"3 s clip uses 45 tokens ({body.get('usage')})")
    message = {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "A picture: "}, {"type": "input_image", "image_url": image},
        {"type": "input_text", "text": " and a sound "}, {"type": "input_audio", "audio_url": audio}]}
    status, body, _ = embed(base, [message, "plain text alongside media"])
    expect(status == 200 and len(body["data"]) == 2, "interleaved message and text embed in one request")
    bad = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<|image_pad|>"},
                                                          {"type": "input_image", "image_url": image}]}
    status, body, _ = embed(base, bad)
    expect(status == 400 and body["error"]["code"] == "invalid_media", "media control tokens in text are rejected")
    broken = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nnot really").decode()
    status, body, _ = embed(base, {"type": "input_image", "image_url": broken})
    expect(status == 400 and body["error"]["code"] == "invalid_media", "undecodable images are rejected")
    expect(metric(base, 'gloss_media_items_total{kind="message"}') >= 1, "media metrics recorded")


def full_media_checks(base: str, reference_path: Path) -> None:
    if not reference_path.exists():
        print(f"skip media parity: {reference_path} not found")
        return
    for case in json.loads(reference_path.read_text())["cases"]:
        content = []
        for part in case["parts"]:
            if part["type"] == "text":
                content.append({"type": "input_text", "text": part["text"]})
            elif part["type"] == "image":
                content.append({"type": "input_image", "image_url": "data:image/png;base64," + part["png"]})
            else:
                content.append({"type": "input_audio", "audio_url": "data:audio/wav;base64," + part["wav"]})
        status, body, _ = embed(base, {"type": "message", "role": "user", "content": content}, timeout=3600)
        expect(status == 200, f"{case['name']} embeds ({body if status != 200 else ''})")
        tokens = body["usage"]["prompt_tokens"]
        c = cosine(body["data"][0]["embedding"], case["embedding"])
        expect(tokens == len(case["input_ids"]), f"{case['name']} token count {tokens} == {len(case['input_ids'])}")
        # WAV clips are 16-bit (the reference used float samples) and parity crosses FP16 towers.
        expect(c >= 0.99, f"{case['name']} FP32 parity cosine {c:.5f}")


def full_model_checks(base: str, reference_path: Path, corpus_path: Path, max_tokens: int) -> None:
    reference = json.loads(reference_path.read_text())
    for case in reference["cases"]:
        tokens = case["tokens"]
        if len(tokens) > max_tokens:
            continue
        content = tokens[3:-2]
        status, body, _ = embed(base, [content], timeout=3600)
        vector = body["data"][0]["embedding"]
        c = cosine(vector, case["embedding"])
        expect(status == 200 and c >= 0.995, f"{case['name']} ({len(tokens)} tokens) FP32 parity cosine {c:.5f}")
        if case.get("text"):
            status, body, _ = embed(base, case["text"])
            c2 = cosine(body["data"][0]["embedding"], case["embedding"])
            expect(c2 >= 0.995, f"{case['name']} text input parity {c2:.5f}")

    corpus = json.loads(corpus_path.read_text())
    pool = [(t["name"], d) for t in corpus["topics"] for d in t["docs"]]
    pool += [(b["topic"], b["text"]) for t in corpus["topics"] for b in t["baits"]]
    _, body, _ = embed(base, [text for _, text in pool])
    doc_vectors = [item["embedding"] for item in body["data"]]
    # The corpus's baits beat the right topic for some queries even in the FP32 source model, so
    # the gate is agreement with the source's own ranking (reference/bidirlm/retrieval_expected.json).
    expected = json.loads((reference_path.parent / "retrieval_expected.json").read_text())["queries"]
    for topic in corpus["topics"]:
        _, body, _ = embed(base, topic["query"])
        q = body["data"][0]["embedding"]
        scores = [cosine(q, v) for v in doc_vectors]
        ranked = sorted(range(len(pool)), key=lambda i: -scores[i])
        source = expected[topic["name"]]
        top = source[0]["index"]
        # Accept a swap only when the source's top two are within 0.002 (FP16 noise).
        tied = len(source) > 1 and source[0]["score"] - source[1]["score"] < 0.002
        ok = ranked[0] == top or (tied and ranked[0] == source[1]["index"])
        expect(ok, f"retrieval: '{topic['name']}' top-1 matches the FP32 source "
                   f"(server {pool[ranked[0]][0]} {scores[ranked[0]]:.4f}, source {source[0]['topic']} {source[0]['score']:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", required=True)
    parser.add_argument("--bundle", type=Path, default=FIXTURE)
    parser.add_argument("--compute", default="ane", choices=["ane", "gpu", "cpu"])
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "32768")))
    parser.add_argument("--ready-timeout", type=float, default=1800)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    args.fixture = manifest.get("fixture") == "dummy-noop"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.mkstemp(prefix="gloss-server-", suffix=".log")[1])
    process = start(args, port, log_path)
    try:
        started = time.time()
        health = wait_ready(base, process, args.ready_timeout)
        print(f"ready in {time.time() - started:.1f}s (compute={args.compute}, fixture={args.fixture})")
        contract_checks(base, args, health)
        batching_checks(base)
        media_checks(base, health)
        if args.full_model:
            if "image" in health["modalities"]:
                full_media_checks(base, ROOT / "reference/bidirlm/media_reference.json")
            full_model_checks(base, ROOT / "reference/bidirlm/text_reference.json",
                              ROOT / "reference/retrieval_corpus.json", args.max_tokens)
        process.send_signal(signal.SIGTERM)
        expect(process.wait(timeout=30) == 0, "SIGTERM shuts down cleanly")
        print("PASS")
    except BaseException:
        print(f"---- server log {log_path} ----")
        print(log_path.read_text()[-6000:])
        raise
    finally:
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    main()
