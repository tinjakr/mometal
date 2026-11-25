#!/usr/bin/env python3
"""
Complete Honest Verification for #6 Quest Query-Aware Block-Sparse Attention:
  1. Exact 4k Dense re-check (after gating dual_probe on pre_pos >= 8192) -> updates hardening_results.json 4k row
  2. 32k-prefix WikiText-2 PPL + Full-Vocab KL + Top-1/Top-5 (Dense vs Quest-8192 [4x sparse] vs Quest-12288)
  3. Clean Isolated Fresh-Server Sweep at 32k / 64k / 128k for Quest(8192) with BOTH top + 50%-depth mid-haystack needles!
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harden_suite as hs


def run_single_ctx(label, n_tok, corpus_ids, tok, sparse_budget, mid_needle=True):
    cmd = [
        hs.BIN, "serve", "--model", hs.MODEL,
        "--ane-tag", "hf4", "--ane-share",
        "--max-seq", "131072", "--kv-bits", "8",
        "--host", "127.0.0.1", "--port", str(hs.PORT),
    ]
    if sparse_budget > 0:
        cmd += ["--sparse-kv-budget", str(sparse_budget)]
    tag = f"dense_{label}" if sparse_budget == 0 else f"quest{sparse_budget}_{label}"
    log_path = os.path.join(hs.OUT_DIR, f"long_ctx_{tag}.log")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            mid, load_s = hs.wait_ready(proc)
            if not mid:
                raise RuntimeError(f"Failed to start server for {tag}")
            hs.stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)

            needle_top = f"NEEDLE_CODE_{label}_7391"
            needle_mid = f"MID_VAULT_{label}_4928"
            if mid_needle:
                half = n_tok // 2
                body = (
                    tok.decode(corpus_ids[:half], skip_special_tokens=True)
                    + f"\n\n[ARCHIVAL MID-DOCUMENT NOTE: The mid-document vault code is {needle_mid}.]\n\n"
                    + tok.decode(corpus_ids[half:n_tok], skip_special_tokens=True)
                )
                prompt = (
                    f"[ctx_sweep_{label}_{int(time.time())}] Reference code: {needle_top}.\n\n"
                    f"{body}\n\n"
                    f"First state both the exact Reference code from the top and the exact mid-document vault code from the ARCHIVAL MID-DOCUMENT NOTE, "
                    f"then write a concise 200-word technical summary of the themes above."
                )
            else:
                body = tok.decode(corpus_ids[:n_tok], skip_special_tokens=True)
                prompt = (
                    f"[ctx_sweep_{label}_{int(time.time())}] Reference code: {needle_top}.\n\n"
                    f"{body}\n\n"
                    f"First state the exact Reference code from the top of this message, then write a concise 200-word technical summary of the themes above."
                )
            r = hs.stream_chat(mid, [{"role": "user", "content": prompt}], 256, temp=0.0, timeout=1500)
            fp_gb, rss_gb = hs.measure_mem(proc.pid)
            hit_top = needle_top in r["text"]
            hit_mid = (needle_mid in r["text"]) if mid_needle else True
        finally:
            hs.kill_tree(proc)

    acc = dft = pld = 0
    with open(log_path) as lf_r:
        req_lines = [l for l in lf_r if "[req]" in l and "prompt=15" not in l]
    if req_lines:
        m = re.search(r"mtp=(\d+)/(\d+)\(\d+%,pld=(\d+)\)", req_lines[-1])
        if m:
            acc, dft, pld = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rounds = max(1, r["ctok"] - acc)
    tok_per_round = round(r["ctok"] / rounds, 2)
    dec_ms_total = (r["ctok"] / max(r["dec_tps"], 1e-6)) * 1000.0
    ms_per_round = round(dec_ms_total / rounds, 1)
    avg_m = round((dft / rounds) + 1.0, 2)

    row = {
        "label": label,
        "sparse_kv_budget": sparse_budget,
        "prompt_tokens": r["ptok"],
        "gen_tokens": r["ctok"],
        "ttft_s": round(r["ttft"], 2),
        "prefill_tps": round(r["pf_tps"], 1),
        "decode_tps": round(r["dec_tps"], 1),
        "rounds": rounds,
        "ms_per_round": ms_per_round,
        "tok_per_round": tok_per_round,
        "avg_m": avg_m,
        "mtp_accepted": acc,
        "mtp_drafted": dft,
        "pld_hits": pld,
        "needle_top_recalled": hit_top,
        "needle_mid_recalled": hit_mid,
        "needle_recalled": hit_top and hit_mid,
        "private_fp_gb": fp_gb,
        "rss_gb": rss_gb,
        "total_sys_gb": round(fp_gb + hs.SHARED_HWX_GB, 2),
        "sample_head": r["text"].strip()[:160],
    }
    print(
        f"  [{tag:<14}] ptok={row['prompt_tokens']:>6} | PF={row['prefill_tps']:>5.1f} tok/s ({row['ttft_s']:>6.2f}s)"
        f" | Dec={row['decode_tps']:>5.1f} tok/s ({ms_per_round:>5.1f} ms/rnd, {tok_per_round:.2f} tok/rnd, M={avg_m:.2f}, pld={pld})"
        f" | top={hit_top} mid={hit_mid} | FP={fp_gb:.2f} GB | TotRAM={row['total_sys_gb']:.2f} GB",
        flush=True,
    )
    return row


def main():
    tok = hs.AutoTokenizer.from_pretrained(hs.MODEL)
    ds = hs.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    raw_text = "\n\n".join(x["text"] for x in ds if len(x["text"].strip()) > 60)
    corpus_ids = tok.encode(raw_text, add_special_tokens=False)
    while len(corpus_ids) < 135000:
        corpus_ids = corpus_ids + corpus_ids

    out_path = os.path.join(hs.OUT_DIR, "quest_full_results.json")
    full = json.load(open(out_path)) if os.path.exists(out_path) else {}

    # 1. Re-verify exact 4k Dense row after gating dual_probe on pre_pos >= 8192
    print("\n=== [1] EXACT 4k DENSE RE-CHECK (pre_pos >= 8192 gate) ===", flush=True)
    r4k = run_single_ctx("4k", 4000, corpus_ids, tok, sparse_budget=0, mid_needle=False)
    full["dense_4k_recheck"] = r4k
    hr_path = os.path.join(hs.OUT_DIR, "hardening_results.json")
    if os.path.exists(hr_path):
        hr = json.load(open(hr_path))
        if "long_context_sweep" in hr:
            for idx, row in enumerate(hr["long_context_sweep"]):
                if row.get("label") == "4k":
                    hr["long_context_sweep"][idx] = r4k
                    break
            json.dump(hr, open(hr_path, "w"), indent=2)
    json.dump(full, open(out_path, "w"), indent=2)

    # 2. 32k-Prefix WikiText-2 PPL + Exact KL + Top-1/Top-5 (4x sparsity check!)
    print("\n=== [2] 32k-PREFIX WIKITEXT-2 PPL + KL (Dense vs Quest 8192 [4x] vs Quest 12288) ===", flush=True)
    wt32_path = os.path.join(hs.OUT_DIR, "wt2_32k.txt")
    ref32_path = os.path.join(hs.OUT_DIR, "wt2_32k_dense.ref")
    with open(wt32_path, "w") as f:
        f.write(",".join(str(x) for x in corpus_ids[:33025]) + "\n")
    ppl32 = {}
    for budget in [0, 8192, 12288]:
        cmd = [
            hs.BIN, "eval-ppl", "--model", hs.MODEL, "--ane-share", "--kv-bits", "8",
            "--max-seq", "36864", "--tokens-file", wt32_path, "--prefix-len", "32768",
        ]
        if budget == 0:
            cmd += ["--save-ref", ref32_path]
        else:
            cmd += ["--sparse-kv-budget", str(budget), "--eval-ref", ref32_path]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        js_line = [l for l in out.splitlines() if l.strip().startswith("{")][-1]
        res = json.loads(js_line)
        ppl32[str(budget)] = res
        lbl = "Dense (Exact)" if budget == 0 else f"Quest({budget})"
        extra = ""
        if budget > 0:
            dppl = res["ppl"] - ppl32["0"]["ppl"]
            extra = (
                f" | dPPL={dppl:+.4f} | meanKL={res.get('mean_kl',0):.5f}"
                f" | Top1={res.get('top1_agree_pct',0):.2f}% | Top5={res.get('ref_top1_in_top5_pct',0):.2f}%"
            )
        print(f"  {lbl:14s}: PPL={res['ppl']:.4f}{extra} | elapsed={res['elapsed_s']:.1f}s", flush=True)
    full["ppl_32k"] = ppl32
    json.dump(full, open(out_path, "w"), indent=2)
    if os.path.exists(ref32_path):
        os.remove(ref32_path)

    # 3. Clean Isolated Quest(8192) Sweep (32k / 64k / 128k) with BOTH top + 50%-depth mid-haystack needles!
    print("\n=== [3] QUEST(8192) CLEAN ISOLATED SWEEP (32k / 64k / 128k, Top + 50% Mid Needles) ===", flush=True)
    sweep_rows = []
    for label, n_tok in [("32k", 32500), ("64k", 65000), ("128k", 126500)]:
        sweep_rows.append(run_single_ctx(label, n_tok, corpus_ids, tok, sparse_budget=8192, mid_needle=True))
        full["quest_8192_sweep"] = sweep_rows
        json.dump(full, open(out_path, "w"), indent=2)

    print(f"\nSaved complete Quest results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
