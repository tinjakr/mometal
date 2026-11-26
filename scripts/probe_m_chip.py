#!/usr/bin/env python3
"""
30-second M1-M5 hardware & OS compatibility probe for mometal single-copy ANE-Share.

Checks on the local Apple Silicon Mac:
  1. SoC, macOS version, Metal GPU family, and ANE architecture (h13g M1 .. h17g M5)
  2. Metal 3.0 compilation of all 5 mometal shaders (common, qwen38, dflash, inkling2, vision)
  3. /Library/Caches/com.apple.aned readability & MAP_SHARED mmap permissions
  4. (If coremltools is installed) Compiles & executes universal `lut4` (M1-M5) and
     native `hf4` (M4/M5+) 4-bit linears on ANE, reports NeuralEngine placement,
     real TOPS, model.hwx bytes/weight, and bit-exact 1-copy __KERN weight matching.

Usage:
  python3 scripts/probe_m_chip.py
"""
import ctypes
import glob
import mmap
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detect_ane_info():
    info = {"arch": "unknown", "cores": 0, "num_anes": 0}
    try:
        ctypes.CDLL("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        msg_obj = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        msg_int = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        msg_str = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        cls = objc.objc_getClass(b"_ANEDeviceInfo")
        if cls:
            ns = msg_obj(cls, objc.sel_registerName(b"aneArchitectureType"))
            if ns:
                s = msg_str(ns, objc.sel_registerName(b"UTF8String"))
                if s:
                    info["arch"] = s.decode("utf-8", "ignore")
            info["cores"] = int(msg_int(cls, objc.sel_registerName(b"numANECores")))
            info["num_anes"] = int(msg_int(cls, objc.sel_registerName(b"numANEs")))
    except Exception:
        pass
    return info


def check_metal_shaders():
    tmp = tempfile.mkdtemp(prefix="mometal_metal_probe_")
    src = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <cstdio>
int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("NO_METAL_DEVICE\n"); return 1; }
        bool apple9 = [dev supportsFamily:MTLGPUFamilyApple9];
        bool apple8 = [dev supportsFamily:MTLGPUFamilyApple8];
        bool apple7 = [dev supportsFamily:MTLGPUFamilyApple7];
        printf("GPU: %s | Apple7=%d Apple8=%d Apple9=%d | maxWorkingSet=%.1f GiB\n",
               [[dev name] UTF8String], apple7, apple8, apple9,
               [dev recommendedMaxWorkingSetSize] / (1024.0 * 1024.0 * 1024.0));
        int failed = 0;
        for (int i = 1; i < argc; ++i) {
            NSString* path = @(argv[i]);
            NSString* code = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (!code) { printf("  [FAIL] read %s\n", argv[i]); failed++; continue; }
            for (int v = 0; v < 2; ++v) {
                if (v == 1 && !apple9) continue;
                MTLCompileOptions* opts = [MTLCompileOptions new];
                opts.languageVersion = (v == 0) ? MTLLanguageVersion3_0 : MTLLanguageVersion3_1;
                NSError* err = nil;
                id<MTLLibrary> lib = [dev newLibraryWithSource:code options:opts error:&err];
                const char* vstr = (v == 0) ? "Metal 3.0" : "Metal 3.1";
                if (lib) {
                    printf("  [PASS] %-16s (%s)\n", [[path lastPathComponent] UTF8String], vstr);
                } else {
                    printf("  [FAIL] %-16s (%s): %s\n", [[path lastPathComponent] UTF8String], vstr,
                           err ? [[err localizedDescription] UTF8String] : "unknown");
                    if (v == 0) failed++;
                }
            }
        }
        return failed ? 2 : 0;
    }
}
'''
    try:
        mm = os.path.join(tmp, "probe.mm")
        exe = os.path.join(tmp, "probe")
        open(mm, "w").write(src)
        subprocess.check_call(
            ["clang++", "-std=c++17", "-fobjc-arc", "-framework", "Foundation", "-framework", "Metal", mm, "-o", exe],
            stderr=subprocess.DEVNULL,
        )
        shaders = [
            os.path.join(REPO_ROOT, "shaders", f)
            for f in ("common.metal", "qwen38.metal", "dflash.metal", "inkling2.metal", "vision.metal")
        ]
        r = subprocess.run([exe] + shaders, capture_output=True, text=True)
        print(r.stdout.strip())
        return r.returncode == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_aned_cache():
    root = "/Library/Caches/com.apple.aned"
    if not os.path.isdir(root):
        print(f"  [FAIL] {root} does not exist")
        return False
    if not os.access(root, os.R_OK | os.X_OK):
        print(f"  [FAIL] Cannot read {root} (EACCES — requires SIP disabled + read ACL)")
        return False
    hwxs = glob.glob(f"{root}/*/ModelAssetsCache/**/model.hwx", recursive=True)
    print(f"  aned cache root: {root} | readable model.hwx count: {len(hwxs)}")
    if not hwxs:
        print("  [WARN] No cached model.hwx yet (will re-check after ANE compile test)")
        return True
    sample = max(hwxs, key=os.path.getmtime)
    st = os.stat(sample)
    mode = stat.filemode(st.st_mode)
    if st.st_size < 4:
        print(f"  [FAIL] {sample} too small ({st.st_size} bytes)")
        return False
    try:
        fd = os.open(sample, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, min(st.st_size, 4096), mmap.MAP_SHARED, mmap.PROT_READ)
            magic = struct.unpack_from("<I", mm, 0)[0]
            mm.close()
        finally:
            os.close(fd)
        print(f"  [PASS] MAP_SHARED mmap succeeded ({mode}, magic=0x{magic:08x}, sample={os.path.basename(os.path.dirname(sample))}/model.hwx)")
        return magic == 0xBEEFFACE
    except OSError as e:
        print(f"  [FAIL] Cannot MAP_SHARED {sample} ({mode}): {e}")
        return False


def parse_hwx_chunks(path):
    buf = open(path, "rb").read()
    if len(buf) < 32 or struct.unpack_from("<I", buf, 0)[0] != 0xBEEFFACE:
        return buf, []
    ncmds, = struct.unpack_from("<I", buf, 16)
    sects, syms = [], []
    off = 32
    for _ in range(ncmds):
        cmd, cs = struct.unpack_from("<II", buf, off)
        if cmd == 0x19:
            seg = buf[off + 8:off + 24].split(b"\0", 1)[0].decode("ascii", "ignore")
            nsects, = struct.unpack_from("<I", buf, off + 64)
            so = off + 72
            for _ in range(nsects):
                sname = buf[so:so + 16].split(b"\0", 1)[0].decode("ascii", "ignore")
                addr, size, foff = struct.unpack_from("<QQI", buf, so + 32)
                sects.append((seg, sname, addr, size, foff))
                so += 80
        elif cmd == 0x2:
            symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", buf, off + 8)
            strs = buf[stroff:stroff + strsize]
            for i in range(nsyms):
                strx, nt, ns, nd, nv = struct.unpack_from("<IBBHQ", buf, symoff + i * 16)
                end = strs.find(b"\0", strx)
                syms.append((strs[strx:end].decode("ascii", "ignore"), ns, nv))
        off += cs
    kts = []
    for idx, (seg, sn, addr, sz, fo) in enumerate(sects):
        if not seg.startswith("__KERN"):
            continue
        addrs = sorted(set([nv for _, ns, nv in syms if ns == idx + 1] + [addr + sz]))
        nxt = {addrs[i]: addrs[i + 1] for i in range(len(addrs) - 1)}
        fam = {}
        for nm, ns, nv in syms:
            if ns == idx + 1 and nm.startswith("K") and "_ne_" in nm:
                parts = nm[1:].split("_ne_")
                if len(parts) == 2 and parts[1].isdigit():
                    ci = int(parts[1])
                    if 0 <= ci < 16:
                        fam.setdefault(parts[0], {})[ci] = nv
        for h, d in fam.items():
            if len(d) == 16:
                cb = nxt[d[0]] - d[0]
                offs = [(d[c] - addr) + fo for c in range(16)]
                if all(offs[c] == offs[0] + c * cb for c in range(16)):
                    kts.append((offs, cb))
    return buf, kts


def pack_group0_words(wf, data_off, K, n_words=8):
    rb = K // 2
    out = []
    for k in range(n_words):
        w = 0
        for j in range(8):
            b = wf[data_off + j * rb + (k // 2)]
            nib = (b & 0xF) if (k & 1) == 0 else ((b >> 4) & 0xF)
            w |= nib << (4 * j)
        out.append(w)
    return struct.pack(f"<{n_words}I", *out)


def check_ane_4bit_share():
    try:
        import numpy as np
        import coremltools as ct
        import coremltools.optimize.coreml as cto
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
        from coremltools.models.compute_plan import MLComputePlan
    except ImportError as e:
        print(f"  [SKIP] coremltools/numpy not installed in {sys.executable} ({e})")
        return

    O, K, R = 4352, 5120, 256
    rng = np.random.default_rng(42)
    W = (rng.standard_normal((O, K)) * 0.25).astype(np.float16)
    sc = np.maximum(np.abs(W).max(axis=1, keepdims=True).astype(np.float32) / 7.5, 1e-6).astype(np.float16)
    q = np.clip(np.round(W.astype(np.float32) / sc.astype(np.float32)), -8.0, 7.0).astype(np.int8)
    idx_u4 = (q.astype(np.int16) & 0xF).astype(types.np_uint4_dtype)
    LUT_I4 = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], dtype=np.float16).reshape(1, 1, 16, 1)

    def q8(x):
        return mb.dequantize(
            input=mb.quantize(input=x, scale=np.float16(1.0 / 127.0), zero_point=np.int8(0), output_dtype="int8"),
            scale=np.float16(1.0 / 127.0), zero_point=np.int8(0),
        )

    tmp = tempfile.mkdtemp(prefix="mometal_ane_probe_")
    compiled_mlcs = []
    aned_readable = os.access("/Library/Caches/com.apple.aned", os.R_OK | os.X_OK)
    try:
        for mode in ("lut4_a8", "lut4_a16", "hf4_w4a8"):
            if mode.startswith("lut4"):
                use_a8 = mode == "lut4_a8"
                @mb.program(input_specs=[mb.TensorSpec(shape=(R, K), dtype=types.fp16)], opset_version=ct.target.iOS18)
                def prog(x):
                    xi = q8(x) if use_a8 else x
                    w = mb.constexpr_lut_to_dense(indices=idx_u4, lut=LUT_I4, name="linear_0_weight_lut4")
                    s = mb.const(val=sc.reshape(1, O), name="linear_0_scale")
                    return mb.mul(x=mb.linear(x=xi, weight=w, name="linear_0"), y=s)
                m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                               minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
                               inputs=[ct.TensorType(name="x", shape=(R, K))])
            else:
                @mb.program(input_specs=[mb.TensorSpec(shape=(R, K), dtype=types.fp16)], opset_version=ct.target.iOS18)
                def prog(x):
                    return mb.linear(x=q8(x), weight=W)
                m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                               minimum_deployment_target=ct.target.iOS18, skip_model_load=True,
                               inputs=[ct.TensorType(name="x", shape=(R, K))])
                qcfg = cto.OptimizationConfig(
                    global_config=cto.OpLinearQuantizerConfig(mode="linear", dtype="int4", granularity="per_channel")
                )
                m = cto.linear_quantize_weights(m, qcfg)

            pkg = os.path.join(tmp, f"{mode}.mlpackage")
            m.save(pkg)
            mlc = ct.models.utils.compile_model(pkg)
            compiled_mlcs.append(mlc)

            plan = MLComputePlan.load_from_path(mlc, ct.ComputeUnit.CPU_AND_NE)
            devs = {}
            for op in plan.model_structure.program.functions["main"].block.operations:
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if u is not None:
                    dn = type(u.preferred_compute_device).__name__.replace("ComputeDevice", "").replace("ML", "")
                    devs[dn] = devs.get(dn, 0) + 1

            before = (
                {p: os.path.getmtime(p) for p in glob.glob("/Library/Caches/com.apple.aned/*/ModelAssetsCache/**/model.hwx", recursive=True)}
                if aned_readable else {}
            )
            cm = ct.models.CompiledMLModel(mlc, compute_units=ct.ComputeUnit.CPU_AND_NE)
            x_in = np.zeros((R, K), dtype=np.float16)
            for _ in range(3):
                cm.predict({"x": x_in})
            ts = []
            for _ in range(15):
                t0 = time.perf_counter()
                cm.predict({"x": x_in})
                ts.append((time.perf_counter() - t0) * 1000.0)
            med_ms = sorted(ts)[len(ts) // 2]
            tops = (2 * R * O * K) / (med_ms * 1e-3) / 1e12

            after = (
                {p: os.path.getmtime(p) for p in glob.glob("/Library/Caches/com.apple.aned/*/ModelAssetsCache/**/model.hwx", recursive=True)}
                if aned_readable else {}
            )
            changed = [p for p, mt in after.items() if p not in before or mt > before[p]]
            hwx = sorted(changed, key=lambda p: after[p], reverse=True)[0] if changed else None

            bit_exact = False
            b_per_wt = 0.0
            hdr_found = None
            if hwx:
                buf, kts = parse_hwx_chunks(hwx)
                b_per_wt = len(buf) / (O * K)
                mil = open(os.path.join(mlc, "model.mil")).read()
                wf = open(os.path.join(mlc, "weights/weight.bin"), "rb").read()
                mm = re.search(r"(?:data|indices) = tensor<(?:int4|uint4), \[\d+, \d+\]>\(BLOBFILE\(path = string\(.@model_path/weights/weight\.bin.\), offset = uint64\((\d+)\)\)\)", mil)
                if mm:
                    data_off, = struct.unpack_from("<Q", wf, int(mm.group(1)) + 16)
                    fp0 = pack_group0_words(wf, data_off, K, 8)
                    for offs, cb in kts:
                        for hb in (24, 16, 0):
                            exp = ((64 + (O // 128) * (hb + 4 * K) + 63) // 64) * 64
                            if cb == exp and buf[offs[0] + 64 + hb:offs[0] + 64 + hb + 32] == fp0:
                                bit_exact = True
                                hdr_found = hb
                                break
                        if bit_exact:
                            break

            if not aned_readable:
                status = "aned cache unreadable (SIP/ACL)"
            elif bit_exact:
                status = "PASS (1-copy ready!)"
            elif b_per_wt > 0.9:
                status = "2x INT8 promoted"
            else:
                status = "layout mismatch"
            print(
                f"  [{mode:8s}] {med_ms:5.2f} ms ({tops:5.2f} TOPS) | {b_per_wt:.2f} B/wt | "
                f"hdr={str(hdr_found):>4s} | placement={devs} | {status}"
            )
    finally:
        for mlc in compiled_mlcs:
            shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=== 1. System & ANE Info ===")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    ane = detect_ane_info()
    print(f"  Chip: {chip} | macOS {platform.mac_ver()[0]} ({platform.machine()})")
    print(f"  ANE:  arch={ane['arch']} | numANEs={ane['num_anes']} | cores={ane['cores']}")
    rec = "lut4" if ane["arch"].startswith(("h13", "h14", "h15")) else "hf4"
    print(f"  Recommended mometal single-copy format: --format {rec}")

    print("\n=== 2. Metal 3.0 Shader Compilation ===")
    check_metal_shaders()

    print("\n=== 3. ANE Cache Read & MAP_SHARED Permissions ===")
    check_aned_cache()

    print("\n=== 4. Live ANE 4-Bit Single-Copy Verification (4352x5120, iOS18 target) ===")
    check_ane_4bit_share()


if __name__ == "__main__":
    main()
