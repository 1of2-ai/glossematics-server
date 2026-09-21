#!/usr/bin/env python3
"""Generate a no-op dummy native-v1 bundle for SDK and server tests.

The output bundle passes the full production load contract (`OmniSmall.load`): identical
manifest pins, identical per-function I/O names / dtypes / shapes in every compiled Core ML
package (verified against the source bundle's own compiled metadata), real tokenizer and
vision-swift assets, and matching SHA-256 artifact checksums. Only the compute differs —
every function consumes its inputs and emits the constant vector `1/32 * ones(D)`, which is
exactly L2-normalized in float32 and stays normalized under Matryoshka truncation. SDK and
server tests therefore exercise the whole load/validate/embed path deterministically, with no
model weights and near-zero inference cost.

The bundle is ~63 MB (tokenizer + vision tables), not gigabytes. The script is
self-contained: it needs coremltools (+ torch, numpy) importable by the interpreter you run
it with, and nothing else from any other project.

    python3 Scripts/make_dummy_bundle.py \
        --source /path/to/JinaV5OmniSmall.w8a16.bundle \
        [--output TestBundles/JinaV5OmniSmall.w8a16.dummy.bundle] [--force]

The contract tables below mirror `OmniSmallBundleValidator` (SwiftPackages/Glossematics/
Sources/Glossematics/OmniSmall.swift); the script cross-checks the generated functions against
the source bundle so drift on either side fails loudly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Checksum declaration, ported from the conversion pipeline's integrity module so this
# script stays self-contained. Semantics must match the Swift validator exactly: every
# regular file beneath the manifest's artifact roots is declared, symlinks are rejected,
# .DS_Store is skipped, and a file covered by two roots is an error.
# ---------------------------------------------------------------------------

def artifact_roots(manifest: dict) -> list[str]:
    roots: list[str] = []
    if "text" in manifest:
        roots += [manifest["text"]["model"], manifest["text"]["tokenizer"]]
    if "colbert" in manifest:
        roots += [manifest["colbert"]["encoder"], manifest["colbert"]["tokenizer"]]
    for section in ("image", "audio", "video"):
        if section in manifest:
            roots.append(manifest[section]["encoder"])
    if "image" in manifest:
        roots.append(manifest["image"]["resources"])
    if "decoder" in manifest:
        roots += [manifest["decoder"]["embed"], manifest["decoder"]["model"]]
    return roots


def _validated_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"artifact path must be a normalized bundle-relative path: {value!r}")
    return path


def build_artifact_checksums(bundle: Path, manifest: dict) -> dict:
    bundle = bundle.resolve()
    files: dict[str, str] = {}
    for root_value in artifact_roots(manifest):
        relative_root = _validated_relative(root_value)
        root = bundle.joinpath(*relative_root.parts)
        if root.is_symlink():
            raise ValueError(f"artifact root is a symlink: {relative_root}")
        if not root.exists():
            raise ValueError(f"artifact root is missing: {relative_root}")

        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            if candidate.name == ".DS_Store":
                continue
            if candidate.is_symlink():
                relative = candidate.relative_to(bundle).as_posix()
                raise ValueError(f"artifact file is a symlink: {relative}")
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(bundle).as_posix()
            if relative in files:
                raise ValueError(f"artifact file is covered by multiple roots: {relative}")
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[relative] = digest.hexdigest()

    return {"algorithm": "sha256", "files": dict(sorted(files.items()))}

DIM = 1024
# 1/32 is a power of two: 1024 * (1/32)^2 == 1.0 exactly in float32.
UNIT_VALUE = 1.0 / 32.0

TEXT_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TEXT_BATCH_PAIRS = [(64, 32), (32, 64), (16, 128), (8, 256), (4, 512)]  # (size, bucket)
IMAGE_BUCKETS = [1024, 1600, 2304, 3072, 4032, 5120]
AUDIO_BUCKETS = [200, 400, 800, 1600, 3200]
DECODER_BUCKETS = [128, 256, 512, 1024, 2048]
VIDEO_BUCKETS = [256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# No-op function bodies. Each input is consumed (float sum) so the graph depends on the
# runtime inputs; the output is the constant unit vector times that sum*0+1. Mirrors the
# repo's converters: torch trace -> ct.convert with named TensorTypes -> multifunction merge.
#
# The constant vector is a registered buffer created in __init__, NOT torch.full() inside
# forward(): the trace sanity checker re-traces with the profiling executor, which constant-
# folds an in-forward aten::full into prim::Constant, so the two traced graphs differ
# structurally (aten::full vs folded Tensor constant) and the check fails. A buffer traces
# to the same prim::Constant in both invocations, so the check runs and passes for real.
# ---------------------------------------------------------------------------


def _convert(module: torch.nn.Module, specs: list[tuple[str, tuple[int, ...], type]],
             output_name: str) -> ct.models.MLModel:
    dtype_map = {np.int32: torch.int32, np.float32: torch.float32}
    inputs = [ct.TensorType(name=name, shape=shape, dtype=dtype)
              for name, shape, dtype in specs]
    example = [torch.zeros(shape, dtype=dtype_map[dtype]) for _, shape, dtype in specs]
    with torch.no_grad():
        traced = torch.jit.trace(module.eval(), example)
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
    )


def make_text_function(rows: int, seq: int) -> ct.models.MLModel:
    class NoOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "base", torch.full((rows, DIM), UNIT_VALUE, dtype=torch.float32))

        def forward(self, input_ids, position_ids, selector):
            total = input_ids.float().sum() + position_ids.float().sum() + selector.sum()
            return self.base * (total * 0.0 + 1.0)

    return _convert(NoOp(), [
        ("input_ids", (rows, seq), np.int32),
        ("position_ids", (3, rows, seq), np.int32),
        ("selector", (rows, seq), np.float32),
    ], "embedding")


def make_embed_function(seq: int) -> ct.models.MLModel:
    class NoOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "base", torch.full((1, seq, DIM), UNIT_VALUE, dtype=torch.float32))

        def forward(self, input_ids):
            return self.base * (input_ids.float().sum() * 0.0 + 1.0)

    return _convert(NoOp(), [
        ("input_ids", (1, seq), np.int32),
    ], "out")


def make_decoder_function(seq: int) -> ct.models.MLModel:
    class NoOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "base", torch.full((1, DIM), UNIT_VALUE, dtype=torch.float32))

        def forward(self, inputs_embeds, position_ids, selector):
            total = (inputs_embeds.sum() + position_ids.float().sum() + selector.sum())
            return self.base * (total * 0.0 + 1.0)

    return _convert(NoOp(), [
        ("inputs_embeds", (1, seq, DIM), np.float32),
        ("position_ids", (3, 1, seq), np.int32),
        ("selector", (1, seq), np.float32),
    ], "embedding")


def make_vision_function(patches: int, square_attention: bool) -> ct.models.MLModel:
    attn = (1, 1, patches, patches) if square_attention else (1, 1, 1, patches)

    class NoOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "base", torch.full((patches // 4, DIM), UNIT_VALUE, dtype=torch.float32))

        def forward(self, pixel_values, pos_embeds, rope_cos, rope_sin, attn_bias):
            total = (pixel_values.sum() + pos_embeds.sum() + rope_cos.sum()
                     + rope_sin.sum() + attn_bias.sum())
            return self.base * (total * 0.0 + 1.0)

    return _convert(NoOp(), [
        ("pixel_values", (patches, 1536), np.float32),
        ("pos_embeds", (patches, DIM), np.float32),
        ("rope_cos", (patches, 64), np.float32),
        ("rope_sin", (patches, 64), np.float32),
        ("attn_bias", attn, np.float32),
    ], "vision_features")


def make_audio_function(frames: int) -> ct.models.MLModel:
    chunks = frames // 200

    class NoOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "base", torch.full((chunks * 50, DIM), UNIT_VALUE, dtype=torch.float32))

        def forward(self, packed_mel, conv_mask, attn_bias):
            total = packed_mel.sum() + conv_mask.sum() + attn_bias.sum()
            return self.base * (total * 0.0 + 1.0)

    return _convert(NoOp(), [
        ("packed_mel", (128, frames), np.float32),
        ("conv_mask", (chunks, 1, 200), np.float32),
        ("attn_bias", (1, 1, chunks * 100, chunks * 100), np.float32),
    ], "audio_features")


# ---------------------------------------------------------------------------
# Package assembly: per-function mlpackage -> multi-function package -> .mlmodelc
# ---------------------------------------------------------------------------

def build_multifunction_model(functions: dict[str, ct.models.MLModel], default: str, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dummy-fn-") as tmp:
        descriptor = MultiFunctionDescriptor()
        for name, model in functions.items():
            package = Path(tmp) / f"{name}.mlpackage"
            model.save(str(package))
            descriptor.add_function(str(package), src_function_name="main",
                                    target_function_name=name)
        descriptor.default_function_name = default
        assert default in functions
        staged = out.parent / f".{out.stem}-staging.mlpackage"
        if staged.exists():
            shutil.rmtree(staged)
        save_multifunction(descriptor, str(staged))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(staged), str(out))


def compile_package(package: Path, out: Path, deployment_target: str) -> None:
    with tempfile.TemporaryDirectory(prefix="dummy-compile-") as tmp:
        cmd = ["xcrun", "coremlcompiler", "compile", str(package), tmp,
               "--platform", "macOS", "--deployment-target", deployment_target]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"coremlcompiler failed for {package.name}: {detail}")
        produced = [d for d in Path(tmp).iterdir() if d.is_dir()]
        if len(produced) != 1:
            raise RuntimeError(f"coremlcompiler produced {produced}, expected one")
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(produced[0]), str(out))


# ---------------------------------------------------------------------------
# Schema cross-check against the real bundle: the mechanical "identical I/O shapes" proof.
# ---------------------------------------------------------------------------

def load_schemas(mlmodelc: Path) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    roots = json.loads((mlmodelc / "metadata.json").read_text())
    schemas: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
    for root in roots:
        for function in root["functions"]:
            schemas[function["name"]] = {
                "inputs": sorted((f["name"], f["dataType"], f["shape"])
                                 for f in function["inputSchema"]),
                "outputs": sorted((f["name"], f["dataType"], f["shape"])
                                  for f in function["outputSchema"]),
            }
    return schemas


def cross_check(dummy: Path, source: Path, label: str) -> None:
    got, want = load_schemas(dummy), load_schemas(source)
    if set(got) != set(want):
        raise SystemExit(
            f"{label}: function mismatch — dummy {sorted(got)} vs source {sorted(want)}")
    for name in want:
        if got[name] != want[name]:
            raise SystemExit(f"{label}.{name}: schema mismatch\n  dummy : {got[name]}\n"
                             f"  source: {want[name]}")


# ---------------------------------------------------------------------------
# Bundle generation
# ---------------------------------------------------------------------------

def compile_tower(name: str, functions: dict[str, Any], default: str,
                  work: Path, out: Path, deployment_target: str) -> None:
    package = work / f"{name}.mlpackage"
    build_multifunction_model(functions, default, package)
    compile_package(package, out, deployment_target)
    print(f"  {out.name}: {len(functions)} functions "
          f"({', '.join(sorted(functions))})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="real native-v1 bundle to mirror (tokenizer + vision assets + schemas)")
    parser.add_argument("--output", type=Path, default=None,
                        help="dummy bundle destination (default: TestBundles/<source-stem>.dummy.bundle)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output bundle")
    args = parser.parse_args()

    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    deployment_target = (manifest.get("minimumDeployment") or {}).get("macOS", "15.0")

    output = args.output or ROOT / "TestBundles" / (source.stem + ".dummy.bundle")
    output = output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"{output} exists; pass --force to overwrite")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    print(f"source: {source}")
    print(f"output: {output}")
    towers = [
        ("text", manifest["text"]["model"],
         {**{f"bucket_{b}": make_text_function(1, b) for b in TEXT_BUCKETS},
          **{f"bucket_{b}_b{s}": make_text_function(s, b) for s, b in TEXT_BATCH_PAIRS}},
         "bucket_128"),
        ("image", manifest["image"]["encoder"],
         {f"f{p}": make_vision_function(p, square_attention=False) for p in IMAGE_BUCKETS},
         f"f{IMAGE_BUCKETS[0]}"),
        ("audio", manifest["audio"]["encoder"],
         {f"f{f}": make_audio_function(f) for f in AUDIO_BUCKETS},
         f"f{AUDIO_BUCKETS[0]}"),
        ("embed", manifest["decoder"]["embed"],
         {f"f{s}": make_embed_function(s) for s in DECODER_BUCKETS},
         f"f{DECODER_BUCKETS[0]}"),
        ("decoder", manifest["decoder"]["model"],
         {f"f{s}": make_decoder_function(s) for s in DECODER_BUCKETS},
         f"f{DECODER_BUCKETS[0]}"),
    ]
    if "video" in manifest:
        towers.append(("video", manifest["video"]["encoder"],
                       {f"f{p}": make_vision_function(p, square_attention=True)
                        for p in VIDEO_BUCKETS},
                       f"f{VIDEO_BUCKETS[0]}"))

    with tempfile.TemporaryDirectory(prefix="dummy-bundle-") as tmp:
        work = Path(tmp)
        print("building no-op towers (coremltools %s)..." % ct.__version__)
        for name, rel, functions, default in towers:
            compile_tower(name, functions, default, work, output / rel, deployment_target)

    print("copying real assets (tokenizer, vision-swift tables)...")
    for rel in {manifest["text"]["tokenizer"], manifest["image"]["resources"]}:
        shutil.copytree(source / rel, output / rel)

    print("cross-checking I/O schemas against the source bundle...")
    for name, rel, _, _ in towers:
        cross_check(output / rel, source / rel, name)
    print(f"  {len(towers)} towers: function names, dtypes, and shapes are identical")

    manifest["converter"] = {"name": "dummy-noop", "version": "1.0.0", "commit": None}
    manifest["compiled"] = {
        "format": "mlmodelc",
        "tool": "coremlcompiler",
        "platform": "macOS",
        "deploymentTarget": deployment_target,
        "compilerVersion": manifest.get("compiled", {}).get("compilerVersion"),
        "commit": None,
    }
    manifest["artifactChecksums"] = build_artifact_checksums(output, manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) / (1024 * 1024)
    files = len(list(output.rglob("*")))
    print(f"dummy bundle ready: {output}")
    print(f"  {size:.1f} MB on disk, every embedding = {UNIT_VALUE} * ones({DIM})")
    print("  serve it:  gloss-server --bundle " + str(output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
