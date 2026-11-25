#!/usr/bin/env python3
"""
Build native CoreML ANE full-MLP programs (L{0..63}_mlp_p0_hf8.mlmodelc + pieces_inv1_hf8_L{0..63}.npy)
for any Qwen 3.5 / 3.8 27B MLX 4-bit g64 checkpoint (e.g. ~/models/qwen38-pi-adapted-mlx4).

Recipe (100% NeuralEngine placement on Apple M4 Max, 26.7 TOPS):
  - Fixed 256-row packed input [256, 5184] FP16 = [x'/r (5120) | r (1) | 0 (63)]
  - SmoothQuant (alpha=0.5) folded into inv1 (saved to pieces_inv1_hf8_L{L}.npy) and up_proj rows
  - Per-channel symmetric int8 weights (exact for dequantized g64 4-bit weights, 0% CPU fallback)
  - Per-token int8 activations (x_n quantized at fixed 1/127 scale; down_proj split into 2 K-groups)
  - 4 x 4352-wide intermediate sub-blocks so every op's output tile stays <= 4.2 MB in ANE SRAM!

Usage:
  python3 scripts/build_ane_mlps.py \
      --model ~/models/qwen38-pi-adapted-mlx4 \
      --out   ~/models/qwen38-pi-adapted-mlx4-ane \
      --bits 4 --qmode asym --alpha 0.25 --mse-clip \
      --layers 0-63 --workers 4
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def parse_layers(spec: str, n_layers: int = 64):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(x for x in out if 0 <= x < n_layers))


def _worker_build_layers(args_tuple):
    model_dir, out_dir, calib_path, layer_ids, tag, rows, verify_first, bits, qmode, alpha, mse_clip = args_tuple
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"mi_ane_{os.getpid()}_", dir=out_dir)
    tempfile.tempdir = tmp_dir
    os.environ["TMPDIR"] = tmp_dir

    def wipe_tmp():
        for e in os.listdir(tmp_dir):
            p = os.path.join(tmp_dir, e)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass

    try:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        import mlx.core as mx

        T = ct.target.iOS26
        PW = 5184
        HID = 5120
        SPLITS = [4352, 4352, 4352, 4352]
        OFF = [0, 4352, 8704, 13056]

        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        weight_map = json.load(open(idx_path))["weight_map"]
        # Detect key prefix
        pfx = "language_model.model."
        if f"{pfx}layers.0.mlp.gate_proj.weight" not in weight_map:
            if "model.language_model.layers.0.mlp.gate_proj.weight" in weight_map:
                pfx = "model.language_model."
            else:
                pfx = "model."

        stats = np.load(calib_path) if (calib_path and os.path.isfile(calib_path)) else None
        shard_cache = {}

        def load_shard(key):
            f = weight_map[key]
            if f not in shard_cache:
                shard_cache.clear()
                shard_cache[f] = mx.load(os.path.join(model_dir, f))
            return shard_cache[f]

        def dequant_w(key):
            d = load_shard(key + ".weight")
            w = mx.dequantize(
                d[key + ".weight"], d[key + ".scales"], d[key + ".biases"],
                group_size=64, bits=4
            ).astype(mx.float16)
            return np.array(w)

        def smooth(stat_key, ws, a=0.5):
            aw = np.abs(np.concatenate(ws, 0)).max(0).astype(np.float64) + 1e-5
            if stats is not None and stat_key in stats:
                ax = stats[stat_key].astype(np.float64) + 1e-5
            else:
                ax = np.ones_like(aw)
            return (ax ** a / aw ** (1.0 - a)).astype(np.float32)

        def mse_clip_i4(W_in):
            # Vectorized per-row MSE-optimal clipping for asymmetric 4-bit (16 levels)
            W = W_in.astype(np.float32)
            wmin0 = W.min(axis=1, keepdims=True)
            wmax0 = W.max(axis=1, keepdims=True)
            mid = 0.5 * (wmin0 + wmax0)
            half = 0.5 * (wmax0 - wmin0)
            best_err = np.full((W.shape[0], 1), np.inf, dtype=np.float32)
            best_lo = wmin0.copy()
            best_hi = wmax0.copy()
            for p in np.linspace(0.65, 1.0, 15):
                lo = mid - half * p
                hi = mid + half * p
                sc = np.maximum((hi - lo) / 15.0, 1e-8)
                q = np.clip(np.round((W - lo) / sc), 0.0, 15.0)
                rec = q * sc + lo
                err = np.mean((W - rec) ** 2, axis=1, keepdims=True)
                better = err < best_err
                best_err = np.where(better, err, best_err)
                best_lo = np.where(better, lo, best_lo)
                best_hi = np.where(better, hi, best_hi)
            return np.clip(W, best_lo, best_hi).astype(np.float16)

        def q8(x, s=1.0 / 127.0):
            return mb.dequantize(
                input=mb.quantize(input=x, scale=np.float16(s), zero_point=np.int8(0), output_dtype="int8"),
                scale=np.float16(s), zero_point=np.int8(0)
            )

        def pt(p):
            r = mb.add(x=mb.reduce_max(x=mb.abs(x=p), axes=[-1], keep_dims=True), y=np.float16(1e-4))
            return q8(mb.real_div(x=p, y=r)), r

        def lin_kgroups(p, W, groups=2):
            K = W.shape[1]
            g = K // groups
            parts = mb.split(x=p, split_sizes=[g] * groups, axis=-1)
            y = None
            for i, pi in enumerate(parts):
                xq, r = pt(pi)
                w_sl = np.ascontiguousarray(W[:, i * g:(i + 1) * g])
                if mse_clip and bits == 4:
                    w_sl = np.ascontiguousarray(mse_clip_i4(w_sl))
                yi = mb.mul(x=mb.linear(x=xq, weight=w_sl), y=r)
                y = yi if y is None else mb.add(x=y, y=yi)
            return y

        ct_mode = "linear" if qmode == "asym" else "linear_symmetric"
        ct_dtype = "int4" if bits == 4 else "int8"
        qcfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode=ct_mode, dtype=ct_dtype, granularity="per_channel"
            )
        )

        built = []
        for L in layer_ids:
            name = f"L{L}_mlp_p0_{tag}"
            mlc_path = os.path.join(out_dir, f"{name}.mlmodelc")
            inv1_path = os.path.join(out_dir, f"pieces_inv1_{tag}_L{L}.npy")
            if os.path.isdir(mlc_path) and os.path.isfile(inv1_path):
                built.append((L, 0.0, "cached"))
                continue

            t0 = time.time()
            pre = f"{pfx}layers.{L}.mlp."
            Wg = dequant_w(pre + "gate_proj")
            Wu = dequant_w(pre + "up_proj")
            Wd = dequant_w(pre + "down_proj")

            s1 = smooth(f"L{L}.mlp.gate_proj", [Wg, Wu], a=alpha)
            inv1 = (1.0 / s1).astype(np.float16)
            Wg = (Wg.astype(np.float32) * s1[None]).astype(np.float16)
            Wu = (Wu.astype(np.float32) * s1[None]).astype(np.float16)

            s2 = smooth(f"L{L}.mlp.down_proj", [Wd], a=alpha)
            Wd = (Wd.astype(np.float32) * s2[None]).astype(np.float16)
            Wu = (Wu.astype(np.float32) / s2[:, None]).astype(np.float16)

            if mse_clip and bits == 4:
                Wg = mse_clip_i4(Wg)
                Wu = mse_clip_i4(Wu)

            np.save(inv1_path, inv1)

            @mb.program(input_specs=[mb.TensorSpec(shape=(rows, PW), dtype=types.fp16)], opset_version=T)
            def prog(x):
                sx = mb.slice_by_size(x=x, begin=[0, 0], size=[rows, HID])
                r = mb.slice_by_size(x=x, begin=[0, HID], size=[rows, 1])
                xq = q8(sx)
                acc = None
                for o, w in zip(OFF, SPLITS):
                    g = mb.mul(x=mb.linear(x=xq, weight=np.ascontiguousarray(Wg[o:o + w])), y=r)
                    u = mb.mul(x=mb.linear(x=xq, weight=np.ascontiguousarray(Wu[o:o + w])), y=r)
                    a_ = mb.mul(x=mb.silu(x=g), y=u)
                    yk = lin_kgroups(a_, np.ascontiguousarray(Wd[:, o:o + w]), groups=2)
                    acc = yk if acc is None else mb.add(x=acc, y=yk)
                return mb.identity(x=acc, name="down_proj")

            m = ct.convert(
                prog,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=T,
                skip_model_load=True,
                inputs=[ct.TensorType(name="x", shape=(rows, PW))],
            )
            m = cto.linear_quantize_weights(m, qcfg)

            pkg_path = os.path.join(tmp_dir, f"{name}.mlpackage")
            shutil.rmtree(pkg_path, ignore_errors=True)
            m.save(pkg_path)
            compiled = ct.models.utils.compile_model(pkg_path)
            shutil.rmtree(mlc_path, ignore_errors=True)
            shutil.move(compiled, mlc_path)
            shutil.rmtree(pkg_path, ignore_errors=True)

            note = "ok"
            if verify_first and L == layer_ids[0]:
                from coremltools.models.compute_plan import MLComputePlan
                plan = MLComputePlan.load_from_path(mlc_path, ct.ComputeUnit.CPU_AND_NE)
                devs = {}
                for op in plan.model_structure.program.functions["main"].block.operations:
                    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if u is not None:
                        dn = type(u.preferred_compute_device).__name__.replace("ComputeDevice", "").replace("ML", "")
                        devs[dn] = devs.get(dn, 0) + 1
                note = f"placement={devs}"

            del m, Wg, Wu, Wd
            gc.collect()
            wipe_tmp()
            dt = time.time() - t0
            print(f"[build_ane] L{L:02d} -> {name}.mlmodelc in {dt:.1f}s ({note})", flush=True)
            built.append((L, dt, note))
        return built
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--layers", default="0-63")
    ap.add_argument("--tag", default="hf4")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--mode", default="asym", choices=["sym", "asym"])
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--mse-clip", action="store_true")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-first", action="store_true")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)
    out_dir = os.path.expanduser(args.out)
    calib_path = os.path.expanduser(args.calib) if args.calib else ""
    layers = parse_layers(args.layers)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    n_workers = max(1, min(args.workers, len(layers)))
    if n_workers == 1:
        _worker_build_layers((model_dir, out_dir, calib_path, layers, args.tag, args.rows,
                              args.verify_first, args.bits, args.mode, args.alpha, args.mse_clip))
    else:
        # Interleave layers by shard locality (0..15, 16..31, 32..47, 48..63) so each worker caches 1 shard!
        buckets = [[] for _ in range(n_workers)]
        chunk = (len(layers) + n_workers - 1) // n_workers
        for i, L in enumerate(layers):
            buckets[min(n_workers - 1, i // chunk)].append(L)
        jobs = [(model_dir, out_dir, calib_path, b, args.tag, args.rows, args.verify_first and idx == 0,
                 args.bits, args.mode, args.alpha, args.mse_clip)
                for idx, b in enumerate(buckets) if b]
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [ex.submit(_worker_build_layers, j) for j in jobs]
            for f in as_completed(futs):
                f.result()

    print(f"[build_ane] Done: {len(layers)} layers in {time.time() - t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
