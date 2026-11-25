#!/usr/bin/env python3
"""Compare mometal top logits & argmax directly against MLX (Qwen3.8) or MTPLX (Inkling 2 Mini)."""
import argparse
import json
import subprocess
import sys

def run_mometal(binary, model_dir, tokens, extra_args=None):
    cmd = [binary, "verify", "--model", model_dir, "--tokens", ",".join(map(str, tokens)), "--top", "10"]
    if extra_args:
        cmd.extend(extra_args)
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip().splitlines()[-1])

def run_qwen_ref(model_dir, tokens):
    import mlx.core as mx
    from mlx_vlm import load
    model, _ = load(model_dir)
    x = mx.array([tokens], dtype=mx.int32)
    out = model.language_model(x)
    logits = (out.logits if hasattr(out, "logits") else out)[0, -1].astype(mx.float32)
    mx.eval(logits)
    arr = logits.tolist()
    top = sorted(enumerate(arr), key=lambda kv: kv[1], reverse=True)[:10]
    return {"top": [[i, round(v, 6)] for i, v in top], "argmax": top[0][0]}

def run_inkling_ref(model_dir, tokens):
    import mlx.core as mx
    from mtplx.models.inkling.model import load_target
    model, _, _ = load_target(model_dir)
    x = mx.array([tokens], dtype=mx.int32)
    logits = model(x)[0, -1].astype(mx.float32)
    mx.eval(logits)
    arr = logits.tolist()
    top = sorted(enumerate(arr), key=lambda kv: kv[1], reverse=True)[:10]
    return {"top": [[i, round(v, 6)] for i, v in top], "argmax": top[0][0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/mometal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", choices=["qwen38", "inkling2"], required=True)
    ap.add_argument("--tokens", default="9707,11,1268,527,499,30")
    args = ap.parse_args()

    toks = [int(x) for x in args.tokens.split(",")]
    mi = run_mometal(args.binary, args.model, toks)
    ref = run_qwen_ref(args.model, toks) if args.arch == "qwen38" else run_inkling_ref(args.model, toks)

    print("mometal   argmax:", mi["gpu_argmax"], "top5:", mi["top"][:5])
    print("reference   argmax:", ref["argmax"], "top5:", ref["top"][:5])
    mi_map = dict(mi["top"])
    diffs = [abs(mi_map[i] - v) for i, v in ref["top"][:5] if i in mi_map]
    print("top-1 match:", mi["gpu_argmax"] == ref["argmax"], "max diff on shared top5:", max(diffs) if diffs else "n/a")
    sys.exit(0 if mi["gpu_argmax"] == ref["argmax"] else 1)

if __name__ == "__main__":
    main()
