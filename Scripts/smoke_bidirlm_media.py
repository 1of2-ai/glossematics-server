"""HTTP smoke of a sealed release with staged media: every reference image, audio, and mixed message.

Starts its own server per compute mode on a free loopback port and stops only that child. Each
media case goes through the full native path (upload decoding, Swift preprocessing, towers,
language encoder) and is compared with the FP32 source embedding in
``reference/bidirlm/media_reference.json``. The ANE result must meet the W8A16 release gate
(cosine >= 0.995). With ``--modes ane gpu`` the report also records ANE-vs-GPU agreement, since
both modes share the stored weights.

    python Scripts/smoke_bidirlm_media.py --server-bin BIN --bundle RELEASE --output DIR --modes ane gpu
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import test_server_http as http

RELEASE_COSINE = 0.995


def content(case):
    parts = []
    for p in case["parts"]:
        if p["type"] == "text":
            parts.append({"type": "input_text", "text": p["text"]})
        elif p["type"] == "image":
            parts.append({"type": "input_image", "image_url": "data:image/png;base64," + p["png"]})
        else:
            parts.append({"type": "input_audio", "audio_url": "data:audio/wav;base64," + p["wav"]})
    return {"type": "message", "role": "user", "content": parts}


def run_mode(args, mode, text, media):
    settings = SimpleNamespace(server_bin=str(args.server_bin.resolve()), bundle=args.bundle.resolve(),
                               compute=mode, fixture=False)
    port = http.free_port()
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    process = http.start(settings, port, args.output / f"server-{mode}.log")
    result = {"cases": {}, "vectors": {}}
    try:
        health = http.wait_ready(base, process, 900)
        result["ready_seconds"] = time.monotonic() - started
        result["health"] = health
        if mode == "ane" and not (health["placement_verified"] and health["placement_strict_ane"]):
            raise AssertionError(f"server placement failed: {health}")
        short = [c for c in text["cases"] if c["name"].startswith("short_")]
        status, body, _ = http.embed(base, [c["text"] for c in short])
        assert status == 200, body
        cosines = [http.cosine(v["embedding"], c["embedding"]) for v, c in zip(body["data"], short)]
        assert len(cosines) == len(short)
        result["packed_text_min_cosine"] = min(cosines)
        for case in media["cases"]:
            timings = []
            for _ in range(2):                       # cold, then warm: the result must not change
                t0 = time.monotonic()
                status, body, _ = http.embed(base, content(case))
                timings.append(time.monotonic() - t0)
                assert status == 200, (case["name"], body)
                vector = body["data"][0]["embedding"]
                if len(timings) == 1:
                    first = vector
            result["vectors"][case["name"]] = vector
            result["cases"][case["name"]] = {
                "cosine_vs_fp32_source": http.cosine(vector, case["embedding"]),
                "repeat_cosine": http.cosine(vector, first),
                "norm": http.norm(vector),
                "prompt_tokens": body["usage"]["prompt_tokens"],
                "expected_tokens": len(case["input_ids"]),
                "seconds_cold": timings[0], "seconds_warm": timings[1]}
            print(json.dumps({"mode": mode, "case": case["name"], **{k: round(v, 6) if isinstance(v, float) else v
                                                                     for k, v in result["cases"][case["name"]].items()}}),
                  flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except http.subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["ane"], choices=["ane", "gpu", "cpu"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.loads((http.ROOT / "reference/bidirlm/text_reference.json").read_text())
    media = json.loads((http.ROOT / "reference/bidirlm/media_reference.json").read_text())
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if not manifest["qualification"]["sealed"]:
        raise ValueError("HTTP smoke requires a sealed release")
    if not (args.bundle / "streamed_media.json").exists():
        raise ValueError("release has no staged media")
    report = {"bundle": str(args.bundle.resolve()), "revision": manifest["revision"],
              "gate": {"ane_cosine_vs_fp32_source_min": RELEASE_COSINE, "packed_text_cosine_min": RELEASE_COSINE},
              "modes": {}, "pass": False}
    try:
        for mode in args.modes:
            report["modes"][mode] = run_mode(args, mode, text, media)
        ane = report["modes"]["ane"]
        failures = [n for n, c in ane["cases"].items()
                    if c["cosine_vs_fp32_source"] < RELEASE_COSINE or c["prompt_tokens"] != c["expected_tokens"]
                    or c["repeat_cosine"] < 0.99999 or abs(c["norm"] - 1) > 0.03]
        if ane["packed_text_min_cosine"] < RELEASE_COSINE:
            failures.append("packed_text")
        if len(ane["cases"]) != len(media["cases"]):
            failures.append("missing media cases")
        for other in set(args.modes) - {"ane"}:
            report[f"ane_vs_{other}"] = {n: http.cosine(v, report["modes"][other]["vectors"][n])
                                         for n, v in ane["vectors"].items()}
        report["failures"] = failures
        report["pass"] = not failures
    finally:
        vectors = {m: r.pop("vectors", {}) for m, r in report["modes"].items()}
        (args.output / "vectors.json").write_text(json.dumps(vectors))
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "failures": report.get("failures")}), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
