#!/usr/bin/env python3
"""Exercise the daemon over HTTP with a jina-embeddings-v5-omni-small Core ML bundle.

Defaults to the golden fixture (Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle, regenerated with
Scripts/make_jina_dummy_bundle.py): a real compiled Core ML bundle with the production contract
whose functions all return 1/32 * ones(1024). It exercises bundle validation, tokenization,
dynamic batching, image/audio/video decoding, the real Core ML execution path, and the HTTP
transport without model weights. Restored from the pre-BidirLM daemon (archive/jina-omnismall-wip).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import math
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "Fixtures" / "JinaV5OmniSmall.w8a16.dummy.bundle"
VIDEO_FIXTURE = ROOT / "Fixtures" / "golden-video.mp4"


def request(url: str, body: dict | None = None, *, timeout: int = 30) -> tuple[int, dict, object]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response), response.headers


def raw_status(port: int, payload: bytes) -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(payload)
        response = connection.recv(256)
    return int(response.split(b" ", 2)[1])


def png_data_url() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", zlib.crc32(kind + data))

    width = height = 32
    rows = b"".join(
        b"\x00" + b"".join(bytes((x * 7 % 256, y * 7 % 256, 128)) for x in range(width))
        for y in range(height))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def wav_data_url() -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        samples = [int(6_000 * math.sin(2 * math.pi * 220 * n / 16_000))
                   for n in range(3_200)]
        writer.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return "data:audio/wav;base64," + base64.b64encode(output.getvalue()).decode()


def stereo_48k_wav_data_url() -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        frames = bytearray()
        for index in range(48_000):
            sample = int(6_000 * math.sin(2 * math.pi * 220 * index / 48_000))
            frames.extend(struct.pack("<hh", sample, sample))
        writer.writeframes(frames)
    return "data:audio/wav;base64," + base64.b64encode(output.getvalue()).decode()


def video_data_url() -> str:
    return "data:video/mp4;base64," + base64.b64encode(VIDEO_FIXTURE.read_bytes()).decode()


def assert_golden(vector: list[float], dimensions: int) -> None:
    assert len(vector) == dimensions, len(vector)
    expected = 1 / math.sqrt(dimensions)
    assert all(math.isclose(value, expected, abs_tol=2e-4) for value in vector), vector[:8]
    assert math.isclose(sum(value * value for value in vector), 1, abs_tol=2e-4)


def assert_embedding(vector: list[float], dimensions: int, *, dummy: bool) -> None:
    if dummy:
        assert_golden(vector, dimensions)
        return
    assert len(vector) == dimensions, len(vector)
    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1, abs_tol=5e-3)
    assert max(vector) - min(vector) > 0.01, "production embedding appears constant"


def metric(base: str, name: str) -> float:
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(name + " "):
                return float(line.split()[-1])
    raise AssertionError(f"missing metric {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--bundle", "--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args()
    server = args.server_bin.resolve()
    fixture = args.bundle.resolve()
    if not server.is_file():
        parser.error(f"server binary is missing: {server}")
    if not fixture.is_dir():
        parser.error(f"Core ML bundle is missing: {fixture}")
    manifest = json.loads((fixture / "manifest.json").read_text())
    is_dummy = manifest["converter"]["name"] == "dummy-noop"
    model = manifest["modelID"]
    for tower in ("text", "image", "audio", "video", "decoder"):
        assert tower in manifest, f"golden fixture has no {tower} tower"

    with tempfile.TemporaryDirectory(prefix="gloss-http-test-") as temp:
        incomplete = Path(temp) / "incomplete.bundle"
        incomplete.mkdir()
        (incomplete / "manifest.json").write_text(json.dumps({
            "formatVersion": 2, "converter": {"name": "dummy-noop"},
        }))
        invalid = subprocess.run(
            [str(server), "--bundle", str(incomplete), "--allow-dummy", "--check-config"],
            capture_output=True, text=True)
        assert invalid.returncode != 0, "invalid dummy unexpectedly loaded"

        check_command = [str(server), "--bundle", str(fixture), "--check-config"]
        if is_dummy:
            rejected = subprocess.run(check_command, capture_output=True, text=True)
            assert rejected.returncode == 2 and "--allow-dummy" in rejected.stderr, rejected.stderr
            check_command.append("--allow-dummy")
        checked = subprocess.run(
            check_command,
            capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        assert ("Core ML dummy fixture" in checked.stdout) == is_dummy, checked.stdout

        for _ in range(5):
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1", 0))
                port = reserved.getsockname()[1]
            command = [str(server), "--bundle", str(fixture), "--port", str(port),
                       "--keep-warm-seconds", "0", "--max-body-mb", "1",
                       "--access-log", "off"]
            if is_dummy:
                command.append("--allow-dummy")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.1)
            if process.poll() is None:
                break
            stdout, stderr = process.communicate()
            if "Address already in use" not in stderr:
                raise AssertionError(f"server exited during startup: {stdout}\n{stderr}")
        else:
            raise AssertionError("could not reserve a free loopback port after five attempts")
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"server exited during startup: {process.returncode}")
                try:
                    status, health, ready_headers = request(base + "/ready")
                    if status == 200 and health["ready"]:
                        assert ready_headers.get("X-Glossematics-Dummy") == (
                            "true" if is_dummy else None)
                        break
                    assert health["status"] != "failed", health
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError("Core ML model did not become ready")

            assert health["model"] == model, health
            assert health["family"] == "jina-embeddings-v5-omni-small", health
            assert health["supported_dimensions"] == [32, 64, 128, 256, 512, 1024], health
            assert health["modalities"] == ["text", "image", "audio", "video"], health
            assert health["space"].startswith("glossematics:omni-small:sha256:"), health
            inference_timeout = 30 if is_dummy else 180
            status, models, _ = request(base + "/v1/models")
            assert status == 200 and models["data"][0]["id"] == model
            status, result, headers = request(base + "/v1/embeddings", {
                "model": model, "input": ["one", "two"], "dimensions": 32,
            }, timeout=inference_timeout)
            assert status == 200 and len(result["data"]) == 2
            for item in result["data"]:
                assert_embedding(item["embedding"], 32, dummy=is_dummy)
            assert headers["X-Glossematics-Space"].startswith("glossematics:omni-small:sha256:")
            assert headers.get("X-Glossematics-Dummy") == ("true" if is_dummy else None)
            assert result["usage"]["prompt_tokens"] > 0
            single_calls = 'gloss_text_waves_total{kind="single"}'
            batch_calls = 'gloss_text_waves_total{kind="native_b64"}'
            assert metric(base, single_calls) >= 2, "sparse text request used a padded batch"
            assert metric(base, batch_calls) == 0, "sparse text request loaded b64@32"

            status, dense, _ = request(base + "/v1/embeddings", {
                "model": model, "input": ["one"] * 64, "dimensions": 32,
            }, timeout=inference_timeout)
            assert status == 200 and len(dense["data"]) == 64
            assert_embedding(dense["data"][0]["embedding"], 32, dummy=is_dummy)
            assert metric(base, batch_calls) >= 1, "dense text request missed b64@32"

            status, encoded, _ = request(base + "/v1/embeddings", {
                "model": model, "input": "one", "dimensions": 64,
                "encoding_format": "base64",
            }, timeout=inference_timeout)
            assert status == 200
            values = struct.unpack("<64f", base64.b64decode(encoded["data"][0]["embedding"]))
            assert_embedding(values, 64, dummy=is_dummy)

            status, query, query_headers = request(base + "/v1/embeddings", {
                "model": model, "input": "one", "task": "retrieval.query",
            }, timeout=inference_timeout)
            assert status == 200 and query_headers["X-Glossematics-Role"] == "query", query
            assert_embedding(query["data"][0]["embedding"], 1024, dummy=is_dummy)
            status, rejected, _ = request(base + "/v1/embeddings", {
                "model": model, "input": "one", "task": "text-matching"})
            assert status == 400 and rejected["error"]["param"] == "input", rejected
            status, rejected, _ = request(base + "/v1/embeddings", {
                "model": model, "input": {"type": "message", "content": [{"type": "input_text", "text": "x"}]}})
            assert status == 400 and rejected["error"]["code"] == "unsupported_modality", rejected

            mixed_text_calls_before = metric(base, single_calls)
            status, mixed, _ = request(base + "/v1/embeddings", {
                "model": model,
                "input": ["one", {"type": "input_image", "image_url": png_data_url()}],
                "dimensions": 128,
            }, timeout=inference_timeout)
            assert status == 200 and len(mixed["data"]) == 2, mixed
            for item in mixed["data"]:
                assert_embedding(item["embedding"], 128, dummy=is_dummy)
            assert mixed["usage"]["prompt_tokens"] > 0
            assert metric(base, single_calls) >= mixed_text_calls_before + 1, (
                "text in a mixed request bypassed dynamic batching")
            status, media_first, _ = request(base + "/v1/embeddings", {
                "model": model,
                "input": [{"type": "input_image", "image_url": png_data_url()}, "one", "two"],
                "dimensions": 128,
            }, timeout=inference_timeout)
            assert status == 200 and [item["index"] for item in media_first["data"]] == [0, 1, 2]
            assert metric(base, single_calls) >= mixed_text_calls_before + 3, (
                "text after media bypassed dynamic batching")
            status, media, media_headers = request(base + "/v1/embeddings", {
                "model": model,
                "input": [
                    {"type": "input_audio", "audio_url": wav_data_url()},
                    {"type": "input_video", "video_url": video_data_url()},
                ],
                "dimensions": 64,
            }, timeout=inference_timeout)
            assert status == 200 and len(media["data"]) == 2, media
            for item in media["data"]:
                assert_embedding(item["embedding"], 64, dummy=is_dummy)
            assert media_headers["X-Glossematics-Usage-Scope"] == "text-only"
            assert media["usage"]["prompt_tokens"] == 0
            status, resampled, _ = request(base + "/v1/embeddings", {
                "model": model,
                "input": {"type": "input_audio", "audio_url": stereo_48k_wav_data_url()},
                "dimensions": 64,
            }, timeout=inference_timeout)
            assert status == 200 and len(resampled["data"]) == 1, resampled
            assert_embedding(resampled["data"][0]["embedding"], 64, dummy=is_dummy)
            concurrent_media = [
                {"type": "input_image", "image_url": png_data_url()},
                {"type": "input_audio", "audio_url": wav_data_url()},
                {"type": "input_video", "video_url": video_data_url()},
            ]
            failures_before = metric(base, "gloss_failed_requests_total")

            def one_media(index: int) -> tuple[int, dict]:
                status, payload, _ = request(base + "/v1/embeddings", {
                    "model": model, "input": concurrent_media[index % len(concurrent_media)],
                    "dimensions": 32,
                }, timeout=inference_timeout)
                return status, payload

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                concurrent_results = list(pool.map(one_media, range(12)))
            for media_status, payload in concurrent_results:
                assert media_status == 200 and len(payload["data"]) == 1, payload
                assert_embedding(payload["data"][0]["embedding"], 32, dummy=is_dummy)
            assert metric(base, "gloss_failed_requests_total") == failures_before
            status, invalid_media, _ = request(base + "/v1/embeddings", {
                "model": model,
                "input": {"type": "input_video", "video_url": "data:video/mp4;base64,AQID"},
            }, timeout=inference_timeout)
            assert status == 400 and invalid_media["error"]["param"] == "input", invalid_media
            status, invalid_second, _ = request(base + "/v1/embeddings", {
                "model": model,
                "input": ["valid first", {"type": "input_video", "video_url": "data:video/mp4;base64,AQID"}],
            }, timeout=inference_timeout)
            assert status == 400 and "batch input 1" in invalid_second["error"]["message"], invalid_second
            status, live, _ = request(base + "/live")
            assert status == 200 and live["status"] == "ok", live
            status, _, _ = request(base + "/healthz")
            assert status == 200, "the SDK daemon's /healthz alias is served"
            for old_route in (b"/livez", b"/readyz"):
                assert raw_status(port, b"GET " + old_route + b" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n") == 404
            assert raw_status(port, b"GET /live HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n") == 400
            assert raw_status(port, b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1\r\nContent-Length: 1\r\nConnection: close\r\n\r\n") == 400
            assert raw_status(port, b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1048577\r\nConnection: close\r\n\r\n") == 413
        finally:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise AssertionError(f"server did not shut down: {stdout}\n{stderr}")
            assert process.returncode == 0, f"server exited {process.returncode}: {stdout}\n{stderr}"
            assert "text inference warmed" in stdout, stdout

    print(f"Core ML {'golden fixture' if is_dummy else 'production bundle'} and multimodal HTTP transport smoke passed")


if __name__ == "__main__":
    main()
