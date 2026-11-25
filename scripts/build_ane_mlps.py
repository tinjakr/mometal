#!/usr/bin/env python3
"""
Build single-copy ANE-Share 4-bit full-MLP programs (L{0..N-1}_mlp_p0_<tag>.mlmodelc +
pieces_inv1_<tag>_L{0..N-1}.npy) from any MLX 4-bit g64 dense SwiGLU checkpoint.

Formats (all single-copy 0.50 B/weight inside model.hwx):
  - lut4 (universal M1/M2/M3/M4/M5):
      Explicit two's-complement 16-entry LUT [0..7, -8..-1] + per-channel post-scale + int8
      activations (iOS18 opset). Stays native 4-bit (0.50 B/wt) with 16 single-family 8-row
      chunks across ALL Apple Silicon generations (h13g M1 -> h17g M5).
  - hf4 (M4/M5 fastest native W4A8):
      Per-channel linear int4 (constexpr_blockwise_shift_scale) + int8 activations (iOS18 opset).
      26.7 TOPS on M4 Max. Note: ANECompiler promotes linear int4 to INT8 (2x RAM) on M1-M3,
      so `--format auto` automatically selects `lut4` on M1/M2/M3 and `hf4` on M4/M5+.

Usage:
  python3 scripts/build_ane_mlps.py \\
      --model ~/models/qwen38-pi-adapted-mlx4 \\
      --out   ~/models/qwen38-pi-adapted-mlx4-ane \\
      --format auto --mse-clip --workers 4
"""
import argparse
import ctypes
import gc
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def detect_ane_arch() -> str:
    """Query AppleNeuralEngine _ANEDeviceInfo.aneArchitectureType() (e.g. 'h13g'..'h17g')."""
    try:
        ctypes.CDLL("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        msg = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        msg_utf8 = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        cls = objc.objc_getClass(b"_ANEDeviceInfo")
        if not cls:
            return ""
        ns_arch = msg(cls, objc.sel_registerName(b"aneArchitectureType"))
        if not ns_arch:
            return ""
        s = msg_utf8(ns_arch, objc.sel_registerName(b"UTF8String"))
        return s.decode("utf-8", "ignore") if s else ""
    except Exception:
        return ""


def resolve_format(fmt: str) -> tuple[str, str]:
    arch = detect_ane_arch()
    if fmt != "auto":
        return fmt, arch
    if arch.startswith(("h13", "h14", "h15")):
        return "lut4", arch
    return "hf4", arch


def choose_subblocks(inter: int) -> int:
    for n in (4, 2, 1):
        if inter % n == 0:
            sub = inter // n
            if sub % 256 == 0 and (sub // 2) % 128 == 0:
                return n
    raise ValueError(
        f"intermediate_size={inter} cannot be split into 1, 2, or 4 sub-blocks of multiples of 256"
    )


def parse_layers(spec: str, n_layers: int):
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
    (
        model_dir,
        out_dir,
        calib_path,
        layer_ids,
        tag,
        rows,
        verify_first,
        fmt,
        target_name,
        qmode,
        alpha,
        mse_clip,
        hid,
        inter,
        n_subblocks,
    ) = args_tuple

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

        T = ct.target.iOS26 if target_name == "ios26" else ct.target.iOS18
        HID = hid
        PW = ((HID + 64) // 64) * 64
        SUB = inter // n_subblocks
        DK = SUB // 2
        SPLITS = [SUB] * n_subblocks
        OFF = [i * SUB for i in range(n_subblocks)]
        LUT_I4 = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], dtype=np.float16
        ).reshape(1, 1, 16, 1)

        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            weight_map = json.load(open(idx_path))["weight_map"]
        else:
            # Single-shard model.safetensors
            sf = "model.safetensors"
            d0 = mx.load(os.path.join(model_dir, sf))
            weight_map = {k: sf for k in d0.keys()}
            del d0

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

        def mse_clip_i4_asym(W_in):
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

        def quant_lut4_sym(W_in, do_mse_clip: bool):
            W = W_in.astype(np.float32)
            amax0 = np.maximum(np.abs(W).max(axis=1, keepdims=True), 1e-6)
            if do_mse_clip:
                best_err = np.full((W.shape[0], 1), np.inf, dtype=np.float32)
                best_sc = amax0 / 7.5
                for p in np.linspace(0.65, 1.0, 15):
                    sc = np.maximum((amax0 * p) / 7.5, 1e-7)
                    q = np.clip(np.round(W / sc), -8.0, 7.0)
                    err = np.mean((W - q * sc) ** 2, axis=1, keepdims=True)
                    better = err < best_err
                    best_err = np.where(better, err, best_err)
                    best_sc = np.where(better, sc, best_sc)
                sc_f16 = np.maximum(best_sc, 1e-6).astype(np.float16)
            else:
                sc_f16 = np.maximum(amax0 / 7.5, 1e-6).astype(np.float16)
            q = np.clip(np.round(W / sc_f16.astype(np.float32)), -8.0, 7.0).astype(np.int8)
            idx_u4 = (q.astype(np.int16) & 0xF).astype(types.np_uint4_dtype)
            return idx_u4, sc_f16.reshape(1, W.shape[0])

        def q8(x, s=1.0 / 127.0):
            return mb.dequantize(
                input=mb.quantize(input=x, scale=np.float16(s), zero_point=np.int8(0), output_dtype="int8"),
                scale=np.float16(s), zero_point=np.int8(0)
            )

        def pt(p):
            r = mb.add(x=mb.reduce_max(x=mb.abs(x=p), axes=[-1], keep_dims=True), y=np.float16(1e-4))
            return q8(mb.real_div(x=p, y=r)), r

        ct_mode = "linear" if qmode == "asym" else "linear_symmetric"
        qcfg_hf4 = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode=ct_mode, dtype="int4", granularity="per_channel"
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

            np.save(inv1_path, inv1)

            if fmt == "lut4":
                # Pre-quantize all 4*n_subblocks linears to two's-complement uint4 LUT indices + fp16 scales
                q_gate, s_gate = [], []
                q_up, s_up = [], []
                q_down, s_down = [], []
                for o, w in zip(OFF, SPLITS):
                    ig, sg = quant_lut4_sym(np.ascontiguousarray(Wg[o:o + w]), mse_clip)
                    iu, su = quant_lut4_sym(np.ascontiguousarray(Wu[o:o + w]), mse_clip)
                    q_gate.append(ig); s_gate.append(sg)
                    q_up.append(iu);   s_up.append(su)
                    for k in range(2):
                        id_, sd_ = quant_lut4_sym(
                            np.ascontiguousarray(Wd[:, o + k * DK:o + (k + 1) * DK]), mse_clip
                        )
                        q_down.append(id_); s_down.append(sd_)

                def lut_lin(x_in, idx_u4, sc_1o, li_idx):
                    w_lut = mb.constexpr_lut_to_dense(
                        indices=idx_u4, lut=LUT_I4, name=f"linear_{li_idx}_weight_lut4"
                    )
                    sc_c = mb.const(val=sc_1o, name=f"linear_{li_idx}_scale")
                    return mb.mul(x=mb.linear(x=x_in, weight=w_lut, name=f"linear_{li_idx}"), y=sc_c)

                @mb.program(input_specs=[mb.TensorSpec(shape=(rows, PW), dtype=types.fp16)], opset_version=T)
                def prog(x):
                    sx = mb.slice_by_size(x=x, begin=[0, 0], size=[rows, HID])
                    r = mb.slice_by_size(x=x, begin=[0, HID], size=[rows, 1])
                    xq = q8(sx)
                    acc = None
                    for sb in range(n_subblocks):
                        g = mb.mul(x=lut_lin(xq, q_gate[sb], s_gate[sb], sb * 4 + 0), y=r)
                        u = mb.mul(x=lut_lin(xq, q_up[sb], s_up[sb], sb * 4 + 1), y=r)
                        a_ = mb.mul(x=mb.silu(x=g), y=u)
                        parts = mb.split(x=a_, split_sizes=[DK, DK], axis=-1)
                        for k in range(2):
                            pk, rk = pt(parts[k])
                            yk = mb.mul(x=lut_lin(pk, q_down[sb * 2 + k], s_down[sb * 2 + k], sb * 4 + 2 + k), y=rk)
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
            else:
                # Native M4/M5 W4A8 linear int4 (hf4)
                if mse_clip:
                    Wg = mse_clip_i4_asym(Wg)
                    Wu = mse_clip_i4_asym(Wu)

                def lin_kgroups(p, W, groups=2):
                    K = W.shape[1]
                    g = K // groups
                    parts = mb.split(x=p, split_sizes=[g] * groups, axis=-1)
                    y = None
                    for i, pi in enumerate(parts):
                        xq, r = pt(pi)
                        w_sl = np.ascontiguousarray(W[:, i * g:(i + 1) * g])
                        if mse_clip:
                            w_sl = np.ascontiguousarray(mse_clip_i4_asym(w_sl))
                        yi = mb.mul(x=mb.linear(x=xq, weight=w_sl), y=r)
                        y = yi if y is None else mb.add(x=y, y=yi)
                    return y

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
                m = cto.linear_quantize_weights(m, qcfg_hf4)

            pkg_path = os.path.join(tmp_dir, f"{name}.mlpackage")
            shutil.rmtree(pkg_path, ignore_errors=True)
            m.save(pkg_path)
            compiled = ct.models.utils.compile_model(pkg_path)
            shutil.rmtree(mlc_path, ignore_errors=True)
            shutil.move(compiled, mlc_path)
            shutil.rmtree(pkg_path, ignore_errors=True)

            note = f"fmt={fmt},target={target_name}"
            if verify_first and L == layer_ids[0]:
                from coremltools.models.compute_plan import MLComputePlan
                plan = MLComputePlan.load_from_path(mlc_path, ct.ComputeUnit.CPU_AND_NE)
                devs = {}
                for op in plan.model_structure.program.functions["main"].block.operations:
                    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if u is not None:
                        dn = type(u.preferred_compute_device).__name__.replace("ComputeDevice", "").replace("ML", "")
                        devs[dn] = devs.get(dn, 0) + 1
                note += f",placement={devs}"

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
    ap.add_argument("--layers", default="all")
    ap.add_argument("--format", default="auto", choices=["auto", "hf4", "lut4"],
                    help="4-bit ANE weight encoding: auto (lut4 on M1-M3, hf4 on M4/M5+), lut4 (universal M1-M5), hf4 (M4/M5+ fastest)")
    ap.add_argument("--target", default="ios18", choices=["ios18", "ios26"],
                    help="Minimum deployment target (default: ios18 -> macOS 15+ compatible)")
    ap.add_argument("--tag", default="auto", help="Model filename tag (default: auto -> hf4 or lut4)")
    ap.add_argument("--mode", default="asym", choices=["sym", "asym"], help="hf4 quantization mode (lut4 is always symmetric)")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--mse-clip", action="store_true")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-first", action="store_true")
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)
    out_dir = os.path.expanduser(args.out)
    calib_path = os.path.expanduser(args.calib) if args.calib else ""

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    hid = int(tc.get("hidden_size", 5120))
    inter = int(tc.get("intermediate_size", 17408))
    n_layers = int(tc.get("num_hidden_layers", 64))
    n_subblocks = choose_subblocks(inter)

    fmt, arch = resolve_format(args.format)
    tag = fmt if args.tag == "auto" else args.tag
    layers = list(range(n_layers)) if args.layers == "all" else parse_layers(args.layers, n_layers)

    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[build_ane] ANE={arch or 'unknown'} | format={fmt} | target={args.target} | tag={tag} | "
        f"dims=({hid}x{inter}, {n_subblocks} sub-blocks) | layers={len(layers)}/{n_layers}",
        flush=True,
    )

    t0 = time.time()
    n_workers = max(1, min(args.workers, len(layers)))
    if n_workers == 1:
        _worker_build_layers((
            model_dir, out_dir, calib_path, layers, tag, args.rows,
            args.verify_first, fmt, args.target, args.mode, args.alpha,
            args.mse_clip, hid, inter, n_subblocks,
        ))
    else:
        buckets = [[] for _ in range(n_workers)]
        chunk = (len(layers) + n_workers - 1) // n_workers
        for i, L in enumerate(layers):
            buckets[min(n_workers - 1, i // chunk)].append(L)
        jobs = [
            (
                model_dir, out_dir, calib_path, b, tag, args.rows,
                args.verify_first and idx == 0, fmt, args.target, args.mode,
                args.alpha, args.mse_clip, hid, inter, n_subblocks,
            )
            for idx, b in enumerate(buckets) if b
        ]
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = [ex.submit(_worker_build_layers, j) for j in jobs]
            for f in as_completed(futs):
                f.result()

    print(f"[build_ane] Done: {len(layers)} layers in {time.time() - t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
