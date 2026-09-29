#!/usr/bin/env python3
"""Generate the no-op BidirLM Core ML fixture used by the server's fast tests.

The fixture follows the production `bidirlm-omni-ane-v2` contract exactly: the same manifest
pins and geometry, the same compiled function names and FP16 input/output signatures in the
four text stack-group packages, the attention buckets, and the vision/audio tower packages,
the real tokenizer, and SHA-256 checksums over every file. Only the compute differs: every function consumes all of its inputs and emits
zeros (hidden states, attention) or the constant unit vector e0 (embeddings). The daemon
recognizes `"fixture": "dummy-noop"`, requires `--allow-dummy`, skips the Neural Engine
placement gate (no-op graphs are not ANE-shaped), and labels every response.

The 622 MB all-zero token table is not checked in. `make fixture` writes it with dd; its
checksum is fixed and recorded in the manifest.

    python Scripts/make_dummy_bundle.py --tokenizer /path/to/BidirLMOmni.ane.bundle/tokenizer \\
        [--output Fixtures/BidirLMOmni.dummy.bundle] [--force]

Needs coremltools and numpy (for example GlossematicsCoreML/bidirlm/.venv). Self-contained:
the contract tables below mirror BidirLMContract / BidirLMBundle in the Swift sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BidirLM/BidirLM-Omni-2.5B-Embedding"
REVISION = "447a6e31be61b84443144afda21374339ce408e6"
H, KV, D, P = 2048, 8, 128, 64
VOCAB = 151936
LAYERS = 28
DEEPSTACK = 2
LONG_KEYS = [1024] + [2048 * n for n in range(1, 17)]
TABLE = "token_embeddings.f16"
f16 = types.fp16


def spec(*shape):
    return mb.TensorSpec(shape=shape, dtype=f16)


def touch(values):
    """A zero scalar that depends on every input so none is pruned from the graph."""
    total = None
    for v in values:
        # Zero each input before reducing: a sum of large inputs (the -30000 attention bias)
        # would overflow FP16, and 0 * inf is NaN.
        s = mb.reduce_sum(x=mb.mul(x=v, y=np.float16(0.0)), keep_dims=False)
        total = s if total is None else mb.add(x=total, y=s)
    return total


def program(specs: dict[str, tuple], outputs: dict[str, tuple]):
    names = list(specs)
    e0 = np.zeros((1, H, P), np.float16)
    e0[0, 0, :] = 1.0

    def body(*inputs):
        zero = touch(inputs)
        by_name = dict(zip(names, inputs))

        def zeros_like(shape):
            # Derive zeros from an input (constants would be stored in the weight file).
            for key, value in by_name.items():
                if tuple(specs[key]) == tuple(shape):
                    return mb.mul(x=value, y=np.float16(0.0))
            count = int(np.prod(shape))
            source = by_name.get("hidden", inputs[0])
            size = int(np.prod(source.shape))
            flat = mb.reshape(x=source, shape=[-1])
            if size < count:
                flat = mb.concat(values=[flat] * (-(-count // size)), axis=0)
            part = mb.slice_by_index(x=flat, begin=[0], end=[count])
            return mb.mul(x=mb.reshape(x=part, shape=list(shape)), y=np.float16(0.0))

        outs = []
        for name, shape in outputs.items():
            base = e0 if name == "embedding" else zeros_like(shape)
            outs.append(mb.add(x=base, y=zero, name=name))
        return tuple(outs) if len(outs) > 1 else outs[0]

    src = f"def prog({', '.join(names)}):\n    return body({', '.join(names)})\n"
    scope = {"body": body}
    exec(src, scope)  # noqa: S102 - the MIL builder needs an explicit parameter list
    return mb.program(input_specs=[spec(*s) for s in specs.values()],
                      opset_version=ct.target.iOS18)(scope["prog"])


STACK = 7


def group_signatures(group: int) -> dict[str, tuple[dict, dict]]:
    """Functions of text stack group `group` (mirrors BidirLMContract.groupFunctions)."""
    C = 512
    q, kv = (1, KV, D, 2 * C), (1, KV, D, C)
    first, last = group * STACK, (group + 1) * STACK
    sigs = {}
    for W in (64, 512):
        inputs = {"hidden": (1, H, 1, W), "cos": (1, 1, D, W), "sin": (1, 1, D, W), "bias": (1, 1, W, 2 * W)}
        for i in range(first, last):
            if i < DEEPSTACK:
                inputs[f"deepstack_{i}"] = (1, H, 1, W)
        if last >= LAYERS:
            sigs[f"stack_c{W}"] = ({**inputs, "pool": (1, W, P), "carry": (1, H, P)},
                                   {"carry_out": (1, H, P), "embedding": (1, H, P)})
        else:
            sigs[f"stack_c{W}"] = (inputs, {"hidden_out": (1, H, 1, W)})
    if group == 0:
        sigs["head_c512"] = ({"hidden": (1, H, 1, C), "cos": (1, 1, D, C), "sin": (1, 1, D, C)},
                             {"query": q, "key": kv, "value": kv})
    for i in range(first, min(last, LAYERS - 1)):
        inputs = {"attention": q, "hidden": (1, H, 1, C), "cos": (1, 1, D, C), "sin": (1, 1, D, C)}
        if i < DEEPSTACK:
            inputs["deepstack_in"] = (1, H, 1, C)
        sigs[f"mid_c512_l{i}"] = (inputs, {"hidden_out": (1, H, 1, C), "query": q, "key": kv, "value": kv})
    if last >= LAYERS:
        sigs["tail_c512"] = ({"attention": q, "hidden": (1, H, 1, C), "pool": (1, C, P), "carry": (1, H, P)},
                             {"carry_out": (1, H, P), "embedding": (1, H, P)})
    return sigs


def attention_signatures() -> dict[str, tuple[dict, dict]]:
    C = 512
    q = (1, KV, D, 2 * C)
    sigs = {}
    for keys, packed in [(k, False) for k in LONG_KEYS]:
        block = min(keys, 2048)
        inputs = {"query": q}
        for b in range(keys // block):
            inputs[f"key_{b}"] = (1, KV, D, block)
            inputs[f"value_{b}"] = (1, KV, D, block)
            inputs[f"bias_{b}"] = (1, 1, block, 2 * C) if packed else (1, 1, block, 1)
        sigs[f"attn_c{C}_s{keys}_{'packed' if packed else 'keys'}"] = (inputs, {"attention": q})
    return sigs


# ---- media towers (mirror BidirLMContract media geometry) ----
EH, EHEADS, EDIM, EC = 1024, 16, 64, 512
VISION_KEYS = [512 * n for n in range(1, 9)]
AUDIO_KEYS = [512, 1024] + [2048 * n for n in range(1, 17)]
MEDIA_TOKENS = {"imagePad": 151655, "visionStart": 151652, "visionEnd": 151653,
                "audioPad": 151676, "audioStart": 151669, "audioEnd": 151670}


def tower_signatures(vision: bool) -> dict[str, tuple[dict, dict]]:
    """head / mid_l{i} / tail of a media tower (mirrors BidirLMContract.towerFunctions)."""
    heads, hidden = (1, EHEADS, EDIM, EC), (1, EH, 1, EC)
    rope = {"cos": (1, 1, EDIM, EC), "sin": (1, 1, EDIM, EC)} if vision else {}
    qkv = {"query": heads, "key": heads, "value": heads}
    if vision:
        sigs = {"head": ({"pixels": (1, 1536, 1, EC), "positions": hidden, **rope}, {"hidden_out": hidden, **qkv})}
    else:
        sigs = {"head": ({"hidden": hidden}, qkv)}
    for i in range(23):
        outputs = {"hidden_out": hidden, **qkv}
        if vision and i in (8, 16):
            outputs["deepstack"] = (1, H, 1, EC // 4)
        sigs[f"mid_l{i}"] = ({"attention": heads, "hidden": hidden, **rope}, outputs)
    sigs["tail"] = ({"attention": heads, "hidden": hidden}, {"features": (1, H, 1, EC // 4 if vision else EC)})
    return sigs


def encoder_attention_signature(keys: int) -> tuple[dict, dict]:
    block = keys if keys <= 4096 else 2048
    heads = (1, EHEADS, EDIM, EC)
    inputs = {"query": heads}
    for b in range(keys // block):
        inputs[f"key_{b}"] = (1, EHEADS, EDIM, block)
        inputs[f"value_{b}"] = (1, EHEADS, EDIM, block)
        inputs[f"bias_{b}"] = (1, 1, block, 1)
    return inputs, {"attention": heads}


def build_media(output: Path, work: Path) -> tuple[dict, dict]:
    vision = {
        "chunk": EC, "hidden": EH, "heads": EHEADS, "headDim": EDIM, "layers": 24, "patchSize": 16,
        "temporalPatchSize": 2, "mergeSize": 2, "patchFeatures": 1536, "minPixels": 65536, "maxPixels": 1048576,
        "imageMean": [0.5, 0.5, 0.5], "imageStd": [0.5, 0.5, 0.5], "ropeTheta": 10000.0, "keyBlock": 2048,
        "singleBlockMaxKeys": 4096, "positionTable": "vision/pos_embed_table.f32", "positionTableShape": [2304, 1024],
        "keys": VISION_KEYS, "deepstackBlocks": [8, 16], "towerModel": "vision/tower.mlmodelc",
    }
    build_package(tower_signatures(vision=True), work, output / "vision" / "tower.mlmodelc")
    vision["attentionModels"] = {}
    for keys in VISION_KEYS:
        inputs, outputs = encoder_attention_signature(keys)
        build_single(inputs, outputs, work, output / "vision" / "attention" / f"attn_s{keys}.mlmodelc")
        vision["attentionModels"][str(keys)] = f"vision/attention/attn_s{keys}.mlmodelc"
    (output / "vision" / "pos_embed_table.f32").write_bytes(bytes(2304 * 1024 * 4))
    print("vision tower", flush=True)

    audio = {
        "chunk": EC, "hidden": EH, "heads": EHEADS, "headDim": EDIM, "layers": 24, "sampleRate": 16000,
        "melBins": 128, "nFFT": 400, "hopLength": 160, "chunkFrames": 200, "tokensPerChunk": 25, "frontBatch": 20,
        "keys": AUDIO_KEYS, "keyBlock": 2048, "singleBlockMaxKeys": 4096,
        "frontModel": "audio/front.mlmodelc", "towerModel": "audio/tower.mlmodelc",
    }
    build_single({"mel": (20, 1, 128, 200), "mask1": (20, 1, 1, 100), "mask2": (20, 1, 1, 50)},
                 {"hidden_out": (1, EH, 1, 500)}, work, output / "audio" / "front.mlmodelc")
    build_package(tower_signatures(vision=False), work, output / "audio" / "tower.mlmodelc")
    audio["attentionModels"] = {}
    for keys in AUDIO_KEYS:
        inputs, outputs = encoder_attention_signature(keys)
        build_single(inputs, outputs, work, output / "audio" / "attention" / f"attn_s{keys}.mlmodelc")
        audio["attentionModels"][str(keys)] = f"audio/attention/attn_s{keys}.mlmodelc"
    print("audio tower", flush=True)
    return vision, audio


def build_single(inputs: dict, outputs: dict, work: Path, out: Path) -> Path:
    pkg = work / f"{out.parent.name}_{out.stem}" / f"{out.stem}.mlpackage"
    pkg.parent.mkdir(parents=True, exist_ok=True)
    model = ct.convert(program(inputs, outputs), minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    model.save(str(pkg))
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out.parent)],
                   check=True, capture_output=True)
    return out


def build_package(sigs: dict[str, tuple[dict, dict]], work: Path, out: Path) -> Path:
    descriptor = ct.utils.MultiFunctionDescriptor()
    for name, (inputs, outputs) in sigs.items():
        pkg = work / f"fn_{name}.mlpackage"
        shutil.rmtree(pkg, ignore_errors=True)
        model = ct.convert(program(inputs, outputs), minimum_deployment_target=ct.target.iOS18,
                           compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        model.save(str(pkg))
        descriptor.add_function(str(pkg), src_function_name="main", target_function_name=name)
    descriptor.default_function_name = next(iter(sigs))
    package = work / f"{out.parent.name}_{out.stem}" / f"{out.stem}.mlpackage"
    package.parent.mkdir(parents=True, exist_ok=True)
    ct.utils.save_multifunction(descriptor, str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(out.parent)],
                   check=True, capture_output=True)
    return out


def zero_table_sha256() -> str:
    digest = hashlib.sha256()
    chunk = bytes(1 << 24)
    remaining = VOCAB * H * 2
    while remaining:
        n = min(remaining, len(chunk))
        digest.update(chunk[:n])
        remaining -= n
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "Fixtures" / "BidirLMOmni.dummy.bundle")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="omit the vision and audio towers")
    args = parser.parse_args()
    if args.output.exists():
        if not args.force:
            raise SystemExit(f"{args.output} exists (use --force)")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="bidirlm-fixture-") as td:
        work = Path(td)
        group_models = []
        for group in range(LAYERS // STACK):
            out = args.output / "text" / f"group_{group}.mlmodelc"
            build_package(group_signatures(group), work, out)
            group_models.append(f"text/{out.name}")
            print("group", group, flush=True)
        attention_models = {}
        for name, (inputs, outputs) in attention_signatures().items():
            out = args.output / "text" / "attention" / f"{name}.mlmodelc"
            build_single(inputs, outputs, work, out)
            attention_models[name] = f"text/attention/{name}.mlmodelc"
        media = None if args.text_only else build_media(args.output, work)
    shutil.copytree(args.tokenizer, args.output / "tokenizer",
                    ignore=shutil.ignore_patterns(".DS_Store"))

    files = {str(p.relative_to(args.output)): sha256(p)
             for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != ".DS_Store"}
    files[TABLE] = zero_table_sha256()
    manifest = {
        "format": "bidirlm-omni-ane-v2",
        "fixture": "dummy-noop",
        "modelID": MODEL_ID,
        "revision": REVISION,
        "embeddingDimension": H,
        "pooling": "masked_mean_l2",
        "padTokenID": 151643,
        "tokenizer": "tokenizer",
        "tokenEmbeddings": {"file": TABLE, "shape": [VOCAB, H], "dtype": "float16", "compute": "cpu"},
        "text": {
            "maxTokens": 32768, "chunks": [64, 512], "stackLayers": STACK, "keyBlock": 2048,
            "longKeys": LONG_KEYS, "poolWidth": P, "layers": LAYERS, "hidden": H, "heads": 16,
            "kvHeads": KV, "headDim": D, "ropeTheta": 5000000.0, "mropeSection": [24, 20, 20],
            "maskNegative": -30000.0, "deepstackLayers": DEEPSTACK,
            "userPrefixIDs": [151644, 872, 198], "userSuffixIDs": [151645, 198],
            "groupModels": group_models, "attentionModels": attention_models,
        },
        **({"vision": media[0], "audio": media[1], "mediaTokens": MEDIA_TOKENS} if media else {}),
        "precision": {"weights": "int8", "activations": "float16"},
        "neuralCompute": "fixture",
        "spaceID": f"{MODEL_ID}:{REVISION}:2048:w8a16:ane-chunked-mean-v1",
        "qualification": {"aneStrictFunctions": 0, "aneStrictAll": False, "sealed": False, "parity": {}},
        "checksums": {"algorithm": "sha256", "files": dict(sorted(files.items()))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output, "(write the zero token table with: make fixture)")


if __name__ == "__main__":
    main()
