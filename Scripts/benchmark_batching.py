#!/usr/bin/env python3
"""Measure text request latency and native Core ML call occupancy on a running daemon."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request


LENGTHS = (6, 36, 76, 150, 300)
CAPACITIES = (64, 32, 16, 8, 4, 1)


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def metrics(base_url):
    with urllib.request.urlopen(base_url + "/metrics", timeout=5) as response:
        lines = response.read().decode("utf-8").splitlines()
    result = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name, value = line.split(None, 1)
        result[name] = float(value)
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--items", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.items) <= 0 or args.warmup_requests < 0:
        parser.error("concurrency, requests, and items must be positive; warmup must be nonnegative")

    base = args.base_url.rstrip("/")
    if not get_json(base + "/ready").get("ready"):
        parser.error("server is not ready")

    def one(index):
        words = LENGTHS[index % len(LENGTHS)]
        input_text = " ".join(["benchmark"] * words)
        body = json.dumps({
            "model": args.model,
            "input": [input_text] * args.items,
            "dimensions": args.dimensions,
        }).encode("utf-8")
        request = urllib.request.Request(
            base + "/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
                if len(payload.get("data", [])) != args.items:
                    return (0, time.perf_counter() - started, "wrong result count")
                return (response.status, time.perf_counter() - started, "")
        except urllib.error.HTTPError as error:
            return (error.code, time.perf_counter() - started, error.reason)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            return (0, time.perf_counter() - started, str(error))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.warmup_requests:
            warmup = list(pool.map(one, range(args.warmup_requests)))
            failed = [result for result in warmup if result[0] != 200]
            if failed:
                print(f"warmup failed: {len(failed)} of {len(warmup)} requests; first={failed[0]}")
                return 1

        before = metrics(base)
        started = time.perf_counter()
        results = list(pool.map(one, range(args.requests)))
        wall = time.perf_counter() - started
        after = metrics(base)

    ok = [result for result in results if result[0] == 200]
    failed = [result for result in results if result[0] != 200]
    latencies = [result[1] * 1000 for result in ok]
    delta = lambda name: after.get(name, 0) - before.get(name, 0)
    waves = delta("gloss_execution_waves_total")
    rows = delta("gloss_execution_rows_total")
    coalesced = delta("gloss_coalesced_waves_total")

    print(f"warmup       {args.warmup_requests} requests")
    print(f"requests     {len(ok)}/{args.requests} ok ({len(failed)} failed)")
    print(f"wall         {wall:.3f} s")
    print(f"throughput   {len(ok) / wall:.1f} req/s  {len(ok) * args.items / wall:.1f} items/s")
    if latencies:
        print("latency ms   "
              f"p50 {percentile(latencies, .5):.1f} "
              f"p95 {percentile(latencies, .95):.1f} "
              f"p99 {percentile(latencies, .99):.1f} "
              f"mean {statistics.fmean(latencies):.1f}")
    if waves:
        print(f"model calls  {int(waves)} rows {int(rows)} rows/call {rows / waves:.2f} "
              f"coalesced {int(coalesced)}/{int(waves)} ({100 * coalesced / waves:.1f}%)")
        for capacity in CAPACITIES:
            label = f'{{capacity="{capacity}"}}'
            calls = delta("gloss_native_execution_waves_total" + label)
            if calls:
                native_rows = delta("gloss_native_execution_rows_total" + label)
                print(f"native b{capacity:<2}   {int(calls)} calls, {int(native_rows)} real rows, "
                      f"fill {100 * native_rows / (calls * capacity):.1f}%")
    if failed:
        print(f"first error  status={failed[0][0]} detail={failed[0][2]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
