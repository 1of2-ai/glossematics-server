"""Post-qualification HTTP smoke: packed text, the fixed long-input case, and media.

Starts its own server on a free loopback port and stops only that child. The stress vectors
come from probe_adversarial_text.py --reference-cache, tying Swift parity to the candidate run.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import test_server_http as http


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stress-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.loads((http.ROOT / "reference/bidirlm/text_reference.json").read_text())
    media = json.loads((http.ROOT / "reference/bidirlm/media_reference.json").read_text())
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if not manifest["qualification"]["sealed"]:
        raise ValueError("HTTP smoke requires a sealed candidate")
    stress = np.load(args.stress_reference, allow_pickle=False)
    if str(stress["revision"]) != manifest["revision"]:
        raise ValueError("stress reference revision differs from the candidate")
    settings = SimpleNamespace(server_bin=str(args.server_bin.resolve()), bundle=args.bundle.resolve(),
                               compute="ane", fixture=False)
    port = http.free_port()
    base = f"http://127.0.0.1:{port}"
    process = http.start(settings, port, args.output / "server.log")
    report = {"bundle": str(args.bundle.resolve()), "pass": False, "cases": {}}
    try:
        health = http.wait_ready(base, process, 600)
        if not health["placement_verified"] or not health["placement_strict_ane"]:
            raise AssertionError(f"server placement failed: {health}")
        report["health"] = health
        short = [c for c in text["cases"] if c["name"].startswith("short_")]
        status, body, _ = http.embed(base, [c["text"] for c in short])
        assert status == 200, body
        cosines = [http.cosine(v["embedding"], c["embedding"]) for v, c in zip(body["data"], short)]
        assert len(cosines) == len(short) and min(cosines) >= 0.995
        report["cases"]["packed_text"] = {"min_cosine": min(cosines), "pass": True}
        status, body, _ = http.embed(base, [stress["ids"][3:-2].tolist()])
        assert status == 200, body
        vector = body["data"][0]["embedding"]
        expected = http.cosine(vector, stress["expected"])
        python = http.cosine(vector, stress["actual"])
        assert expected >= 0.99 and python >= 0.9999, (expected, python)
        report["cases"]["pad_8192"] = {"cosine": expected, "cosine_vs_python": python, "pass": True}
        for case in media["cases"]:
            if case["name"] not in {"image_noise_300x420", "audio_0.5s", "message_text_image_audio"}:
                continue
            content = []
            for p in case["parts"]:
                if p["type"] == "text":
                    content.append({"type": "input_text", "text": p["text"]})
                elif p["type"] == "image":
                    content.append({"type": "input_image", "image_url": "data:image/png;base64," + p["png"]})
                else:
                    content.append({"type": "input_audio", "audio_url": "data:audio/wav;base64," + p["wav"]})
            status, body, _ = http.embed(base, {"type": "message", "role": "user", "content": content})
            assert status == 200, body
            cosine = http.cosine(body["data"][0]["embedding"], case["embedding"])
            assert body["usage"]["prompt_tokens"] == len(case["input_ids"])
            assert cosine >= 0.99, (case["name"], cosine)
            report["cases"][case["name"]] = {"cosine": cosine, "pass": True}
        assert len(report["cases"]) == 5
        report["pass"] = True
        print(json.dumps(report, indent=2), flush=True)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        process.terminate()
        try:
            process.wait(timeout=30)
        except http.subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
