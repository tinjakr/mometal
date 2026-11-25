#!/usr/bin/env python3
"""
Hardening Verification Suite for metal-infer (1-copy 64/64 ANE-Share)
Covers:
  1. WikiText-2 Perplexity, Full-Vocab KL Divergence, Top-1 & Top-5 Agreement
     (Exact g64 BF16-KV vs 1-Copy ANE-Share BF16-KV vs 1-Copy ANE-Share KV8)
  2. Full HumanEval (164 problems, T=0.0) pass@1: Exact g64 vs 1-Copy ANE-Share KV8
  3. Matched-KV control (BF16 KV vs KV8) + 3x repeated variance trials on 1-Copy ANE-Share
  4. Long-context sweep (4k / 16k / 32k / 64k / 128k) on 1-Copy ANE-Share KV8
"""
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MOMETAL_MODEL", os.path.expanduser("~/models/qwen38-pi-adapted-mlx4"))
BIN = os.environ.get("MOMETAL_BIN", os.path.join(REPO_DIR, "build", "mometal"))
OUT_DIR = os.environ.get("MOMETAL_OUT_DIR", os.path.join(REPO_DIR, "benchmarks", "out"))
os.makedirs(OUT_DIR, exist_ok=True)
PORT = 8098
URL = f"http://127.0.0.1:{PORT}"
GIB = float(1024 ** 3)


def measure_mem(pid):
    """Return byte-exact (phys_footprint_gib, neural_gib, total_phys_gib, wired_gib, swapped_gib, rss_gib)."""
    rss_gib = 0.0
    phys_gib = neural_gib = wired_gib = swapped_gib = 0.0
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        if out:
            rss_gib = int(out) * 1024.0 / GIB
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["footprint", "-f", "bytes", "--wired", "--swapped", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        m_fp = re.search(r"phys_footprint:\s*(\d+)\s*B", out)
        m_neu = re.search(r"neural:\s*(\d+)\s*B", out)
        m_tot = re.search(
            r"^\s*(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+\d+\s+TOTAL\s*$",
            out,
            re.M,
        )
        if m_fp:
            phys_gib = int(m_fp.group(1)) / GIB
        if m_neu:
            neural_gib = int(m_neu.group(1)) / GIB
        if m_tot:
            swapped_gib = int(m_tot.group(2)) / GIB
            wired_gib = int(m_tot.group(5)) / GIB
    except Exception:
        pass
    total_gib = phys_gib + neural_gib
    return {
        "phys_footprint_gib": round(phys_gib, 3),
        "neural_gib": round(neural_gib, 3),
        "total_phys_gib": round(total_gib, 3),
        "wired_gib": round(wired_gib, 3),
        "swapped_gib": round(swapped_gib, 3),
        "rss_gib": round(rss_gib, 3),
    }


def kill_tree(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=5)
    time.sleep(2.5)


def wait_ready(proc, timeout_s=180):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            return None, time.perf_counter() - t0
        try:
            with urllib.request.urlopen(URL + "/v1/models", timeout=2) as r:
                d = json.loads(r.read().decode("utf-8"))
                ms = d.get("data") or []
                if ms:
                    return ms[0].get("id"), time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.4)
    return None, time.perf_counter() - t0


def stream_chat(model_id, messages, max_tokens, temp=0.0, timeout=900):
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    chunks = 0
    parts = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                if c:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
                    parts.append(c)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ptok = (usage or {}).get("prompt_tokens") or 0
    ctok = (usage or {}).get("completion_tokens") or chunks
    ttft = t_first - t0
    gen_s = max(t_end - t_first, 1e-6)
    dec_tps = max(ctok - 1, 1) / gen_s if ctok > 1 else ctok / max(t_end - t0, 1e-6)
    pf_tps = ptok / max(ttft, 1e-6) if ptok > 0 else 0.0
    return {
        "ptok": ptok,
        "ctok": ctok,
        "ttft": ttft,
        "pf_tps": pf_tps,
        "dec_tps": dec_tps,
        "text": "".join(parts),
    }


def run_wikitext_ppl():
    print("\n================ [1A] WIKITEXT-2 PPL + KL DIVERGENCE ================", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(x["text"] for x in ds if x["text"].strip())
    all_ids = tok.encode(full_text, add_special_tokens=False)
    print(f"WikiText-2 test total tokens: {len(all_ids)}", flush=True)

    # 8 evenly spaced 512-token passages (256 prefix + 256 scored = 2,048 scored tokens)
    n_passages = 8
    chunk_len = 512
    stride = (len(all_ids) - chunk_len) // n_passages
    tok_file = os.path.join(OUT_DIR, "wt2_passages.txt")
    with open(tok_file, "w") as f:
        for i in range(n_passages):
            s = i * stride
            f.write(",".join(str(x) for x in all_ids[s:s + chunk_len]) + "\n")

    ref_bin = os.path.join(OUT_DIR, "wt2_ref_g64.bin")
    configs = [
        ("exact_g64_bf16", ["--ane-models", "none", "--kv-bits", "16", "--save-ref", ref_bin]),
        ("ane_share_hf4_bf16", ["--ane-tag", "hf4", "--ane-share", "--kv-bits", "16", "--eval-ref", ref_bin]),
        ("ane_share_hf4_kv8", ["--ane-tag", "hf4", "--ane-share", "--kv-bits", "8", "--eval-ref", ref_bin]),
    ]
    res = {}
    for cid, extra in configs:
        cmd = [BIN, "eval-ppl", "--model", MODEL, "--tokens-file", tok_file, "--prefix-len", "256", "--max-seq", "1024"] + extra
        print(f"  Running {cid}...", flush=True)
        out = subprocess.check_output(cmd, text=True)
        line = [l for l in out.strip().splitlines() if l.startswith("{")][-1]
        d = json.loads(line)
        res[cid] = d
        print(f"  -> {cid}: {d}", flush=True)
    if os.path.exists(ref_bin):
        os.remove(ref_bin)
    return res


def extract_python_code(prompt, text, entry_point):
    # Extract fenced python block if present, otherwise use raw text
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    # If the model output includes the function signature `def entry_point`, use code directly (preserving imports from prompt)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\b", code, re.M):
        # Extract any imports/helpers from prompt before the target def
        p_lines = prompt.splitlines()
        prefix_lines = []
        for l in p_lines:
            if re.match(rf"^\s*def\s+{re.escape(entry_point)}\b", l):
                break
            prefix_lines.append(l)
        return "\n".join(prefix_lines) + "\n\n" + code
    # Otherwise assume completion of the function body
    return prompt + "\n" + code


def check_humaneval_solution(prompt, text, test_code, entry_point):
    full_code = extract_python_code(prompt, text, entry_point) + f"\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full_code)
        tpath = tf.name
    try:
        p = subprocess.run([sys.executable, tpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.remove(tpath)
        except OSError:
            pass


def run_humaneval():
    print("\n================ [1B] HUMANEVAL (164 PROBLEMS, GREEDY T=0.0) ================", flush=True)
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = list(ds)

    configs = [
        ("exact_g64_bf16", [BIN, "serve", "--model", MODEL, "--ane-models", "none", "--max-seq", "8192", "--kv-bits", "16", "--host", "127.0.0.1", "--port", str(PORT)]),
        ("ane_share_hf4_kv8", [BIN, "serve", "--model", MODEL, "--ane-tag", "hf4", "--ane-share", "--max-seq", "8192", "--kv-bits", "8", "--host", "127.0.0.1", "--port", str(PORT)]),
    ]
    results = {}
    for cid, cmd in configs:
        print(f"\n  Evaluating {cid} on {len(problems)} HumanEval problems...", flush=True)
        log_path = os.path.join(OUT_DIR, f"he_{cid}.log")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                mid, _ = wait_ready(proc)
                if not mid:
                    raise RuntimeError(f"Failed to start {cid}")
                stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)
                passed = 0
                tps_list = []
                outcomes = []
                t0 = time.perf_counter()
                for idx, row in enumerate(problems):
                    msg = (
                        "Complete the following Python function. Return only the complete, working Python code "
                        "inside a ```python``` code block, with no explanation.\n\n```python\n"
                        + row["prompt"]
                        + "\n```"
                    )
                    r = stream_chat(mid, [{"role": "user", "content": msg}], 512, temp=0.0)
                    ok = check_humaneval_solution(row["prompt"], r["text"], row["test"], row["entry_point"])
                    if ok:
                        passed += 1
                    tps_list.append(r["dec_tps"])
                    outcomes.append({"task_id": row["task_id"], "passed": ok, "ctok": r["ctok"], "tps": round(r["dec_tps"], 1)})
                    if (idx + 1) % 20 == 0 or idx + 1 == len(problems):
                        print(f"    [{cid}] {idx+1}/{len(problems)} | pass={passed}/{idx+1} ({100*passed/(idx+1):.1f}%) | med_tps={statistics.median(tps_list):.1f}", flush=True)
                elapsed = time.perf_counter() - t0
                results[cid] = {
                    "passed": passed,
                    "total": len(problems),
                    "pass_at_1_pct": round(100.0 * passed / len(problems), 2),
                    "median_tps": round(statistics.median(tps_list), 1),
                    "elapsed_s": round(elapsed, 1),
                    "outcomes": outcomes,
                }
                print(f"  -> {cid}: {passed}/{len(problems)} ({results[cid]['pass_at_1_pct']}%) in {elapsed:.1f}s", flush=True)
            finally:
                kill_tree(proc)

    # Compare per-problem agreement
    o_ref = {x["task_id"]: x["passed"] for x in results["exact_g64_bf16"]["outcomes"]}
    o_sh = {x["task_id"]: x["passed"] for x in results["ane_share_hf4_kv8"]["outcomes"]}
    both_pass = sum(1 for k in o_ref if o_ref[k] and o_sh[k])
    only_ref = sum(1 for k in o_ref if o_ref[k] and not o_sh[k])
    only_sh = sum(1 for k in o_ref if not o_ref[k] and o_sh[k])
    both_fail = sum(1 for k in o_ref if not o_ref[k] and not o_sh[k])
    results["agreement"] = {
        "both_pass": both_pass,
        "only_g64_pass": only_ref,
        "only_hf4_pass": only_sh,
        "both_fail": both_fail,
        "exact_outcome_match_pct": round(100.0 * (both_pass + both_fail) / len(problems), 2),
    }
    print(f"  Agreement: {results['agreement']}", flush=True)
    return results


GREEDY_PROMPTS = [
    ("cpp_spsc", "Write a complete, production-quality C++20 lock-free single-producer single-consumer (SPSC) ring buffer template class with cache-line padding, power-of-two capacity, acquire/release memory ordering, try_push, try_pop, and a short main() benchmark.", 384),
    ("py_lru", "Write a clean Python 3.12 async LRU cache class with per-entry TTL expiration, asyncio.Lock safety, hit/miss statistics, max_size eviction, and 4 unit tests using asyncio.run().", 384),
    ("cpp_refactor", """Refactor the following C++ class to add std::shared_mutex thread safety, RAII lock guards, explicit error handling with std::expected, and a batch_insert() method while preserving the existing API style:

```cpp
class MetricsRegistry {
public:
    void increment(const std::string& name, int64_t delta = 1) {
        counters_[name] += delta;
    }
    int64_t get(const std::string& name) const {
        auto it = counters_.find(name);
        return it == counters_.end() ? 0 : it->second;
    }
    void set_gauge(const std::string& name, double value) {
        gauges_[name] = value;
    }
    double get_gauge(const std::string& name) const {
        auto it = gauges_.find(name);
        return it == gauges_.end() ? 0.0 : it->second;
    }
    void reset() {
        counters_.clear();
        gauges_.clear();
    }
private:
    std::unordered_map<std::string, int64_t> counters_;
    std::unordered_map<std::string, double> gauges_;
};
```""", 384),
    ("prose", "Explain in clear technical detail how Apple Silicon unified memory, Metal simdgroup_matrix AMX tiles, and KV-cache quantization interact to determine LLM prefill versus decode bottlenecks.", 384),
]


def run_kv_control_and_3x():
    print("\n================ [3A + 4] MATCHED-KV CONTROL & 3x VARIANCE (1-COPY ANE-SHARE) ================", flush=True)
    cache_dir = os.path.join(OUT_DIR, "cache_3x")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    out = {}
    for cid, kv_bits, max_seq, trials in [
        ("ane_share_bf16_kv_ctrl", 16, 32768, 1),
        ("ane_share_kv8_3x", 8, 131072, 3),
    ]:
        cmd = [
            BIN, "serve", "--model", MODEL,
            "--ane-tag", "hf4", "--ane-share",
            "--max-seq", str(max_seq), "--kv-bits", str(kv_bits),
            "--cache-dir", os.path.join(cache_dir, cid),
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        log_path = os.path.join(OUT_DIR, f"ctrl_{cid}.log")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                mid, load_s = wait_ready(proc)
                if not mid:
                    raise RuntimeError(f"Failed to start {cid}")
                stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)
                stream_chat(mid, [{"role": "user", "content": "Write a 3-line Python hello function."}], 32, temp=0.0)

                trial_data = []
                for t in range(trials):
                    salt = f"[var_{cid}_t{t}_{int(time.time())}] "
                    pad = salt + "The quick brown fox jumps over the lazy dog near the riverbank at dawn while compilers optimize memory layouts. " * 175
                    pf_cold = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
                    pf_warm = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
                    g = {}
                    for pname, prompt, max_tok in GREEDY_PROMPTS:
                        r = stream_chat(mid, [{"role": "user", "content": prompt}], max_tok, temp=0.0)
                        g[pname] = round(r["dec_tps"], 1)
                    g_mean = round(sum(g.values()) / len(g), 1)
                    td = {
                        "trial": t + 1,
                        "pf_cold_tok": pf_cold["ptok"],
                        "pf_cold_ttft": round(pf_cold["ttft"], 2),
                        "pf_cold_tps": round(pf_cold["pf_tps"], 1),
                        "pf_warm_ms": round(pf_warm["ttft"] * 1000.0, 0),
                        "greedy": g,
                        "g_mean": g_mean,
                    }
                    trial_data.append(td)
                    print(f"  [{cid}] trial {t+1}/{trials}: PF={td['pf_cold_tps']} tok/s ({td['pf_cold_ttft']}s) | Warm={td['pf_warm_ms']:.0f}ms | GMean={g_mean} {g}", flush=True)

                mem = measure_mem(proc.pid)
                summary = {
                    "kv_bits": kv_bits,
                    "max_seq": max_seq,
                    "load_s": round(load_s, 2),
                    "phys_footprint_gib": mem["phys_footprint_gib"],
                    "neural_gib": mem["neural_gib"],
                    "total_phys_gib": mem["total_phys_gib"],
                    "wired_gib": mem["wired_gib"],
                    "swapped_gib": mem["swapped_gib"],
                    "rss_gib": mem["rss_gib"],
                    "trials": trial_data,
                    "pf_cold_tps_median": round(statistics.median(x["pf_cold_tps"] for x in trial_data), 1),
                    "pf_cold_tps_min": min(x["pf_cold_tps"] for x in trial_data),
                    "pf_cold_tps_max": max(x["pf_cold_tps"] for x in trial_data),
                    "pf_warm_ms_median": round(statistics.median(x["pf_warm_ms"] for x in trial_data), 0),
                    "g_mean_median": round(statistics.median(x["g_mean"] for x in trial_data), 1),
                    "g_mean_min": min(x["g_mean"] for x in trial_data),
                    "g_mean_max": max(x["g_mean"] for x in trial_data),
                }
                out[cid] = summary
                print(f"  -> {cid} summary: PF_med={summary['pf_cold_tps_median']} [{summary['pf_cold_tps_min']}-{summary['pf_cold_tps_max']}] | GMean_med={summary['g_mean_median']} [{summary['g_mean_min']}-{summary['g_mean_max']}] | TotRAM={summary['total_phys_gib']} GiB", flush=True)
            finally:
                kill_tree(proc)
    return out


def run_long_context_sweep():
    print("\n================ [3B] LONG-CONTEXT SWEEP (4k / 16k / 32k / 64k / 128k) ================", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Build varied technical prose corpus so KV/GDN activations are realistic
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    corpus_text = "\n\n".join(x["text"] for x in ds if len(x["text"].strip()) > 40)
    corpus_ids = tok.encode(corpus_text[:900000], add_special_tokens=False)
    print(f"Long-context corpus ready: {len(corpus_ids)} tokens", flush=True)

    targets = [
        ("4k", 4000),
        ("16k", 16200),
        ("32k", 32500),
        ("64k", 65000),
        ("128k", 126500),  # leaves clean room for chat template + 256 gen inside 131,072
    ]

    cmd = [
        BIN, "serve", "--model", MODEL,
        "--ane-tag", "hf4", "--ane-share",
        "--max-seq", "131072", "--kv-bits", "8",
        "--host", "127.0.0.1", "--port", str(PORT),
    ]
    rows = []
    for label, n_tok in targets:
        # Fresh server per context length -> zero cross-turn session suffix cache or KV watermark carry-over!
        log_path = os.path.join(OUT_DIR, f"long_ctx_{label}.log")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                mid, load_s = wait_ready(proc)
                if not mid:
                    raise RuntimeError(f"Failed to start long-context server for {label}")
                stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)

                needle = f"NEEDLE_CODE_{label}_7391"
                text_body = tok.decode(corpus_ids[:n_tok], skip_special_tokens=True)
                prompt = (
                    f"[ctx_sweep_{label}_{int(time.time())}] Reference code: {needle}.\n\n"
                    f"{text_body}\n\n"
                    f"First state the exact Reference code from the top of this message, then write a concise 200-word technical summary of the themes above."
                )
                r = stream_chat(mid, [{"role": "user", "content": prompt}], 256, temp=0.0, timeout=1200)
                mem = measure_mem(proc.pid)
                needle_hit = needle in r["text"]
            finally:
                kill_tree(proc)

        # Parse exact MTP round stats from server log
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
            "needle_recalled": needle_hit,
            "phys_footprint_gib": mem["phys_footprint_gib"],
            "neural_gib": mem["neural_gib"],
            "total_phys_gib": mem["total_phys_gib"],
            "wired_gib": mem["wired_gib"],
            "swapped_gib": mem["swapped_gib"],
            "rss_gib": mem["rss_gib"],
            "sample_head": r["text"].strip()[:120],
        }
        rows.append(row)
        print(
            f"  [{label:<4}] ptok={row['prompt_tokens']:>6} | PF={row['prefill_tps']:>5.1f} tok/s ({row['ttft_s']:>6.2f}s)"
            f" | Dec={row['decode_tps']:>5.1f} tok/s ({ms_per_round:>5.1f} ms/rnd, {tok_per_round:.2f} tok/rnd, M={avg_m:.2f}, pld={pld})"
            f" | needle={needle_hit} | FP={mem['phys_footprint_gib']:.2f} + Neu={mem['neural_gib']:.2f} = {mem['total_phys_gib']:.2f} GiB",
            flush=True,
        )
    return rows


def run_cold_3x_variance():
    print("\n================ [4] CLEAN COLD 3x VARIANCE (FRESH SERVER PER TRIAL) ================", flush=True)
    cache_dir = os.path.join(OUT_DIR, "cache_cold3x")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    trial_data = []
    loads, fps, neus, tots = [], [], [], []
    for t in range(3):
        tcache = os.path.join(cache_dir, f"t{t}")
        cmd = [
            BIN, "serve", "--model", MODEL,
            "--ane-tag", "hf4", "--ane-share",
            "--max-seq", "131072", "--kv-bits", "8",
            "--cache-dir", tcache,
            "--host", "127.0.0.1", "--port", str(PORT),
        ]
        log_path = os.path.join(OUT_DIR, f"cold3x_t{t+1}.log")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                mid, load_s = wait_ready(proc)
                if not mid:
                    raise RuntimeError(f"Failed to start cold3x trial {t+1}")
                loads.append(round(load_s, 2))
                stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)
                stream_chat(mid, [{"role": "user", "content": "Write a 3-line Python hello function."}], 32, temp=0.0)

                salt = f"[cold3x_t{t}_{int(time.time())}] "
                pad = salt + "The quick brown fox jumps over the lazy dog near the riverbank at dawn while compilers optimize memory layouts. " * 175
                pf_cold = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
                pf_warm = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)

                g = {}
                for pname, prompt, max_tok in GREEDY_PROMPTS:
                    r = stream_chat(mid, [{"role": "user", "content": prompt}], max_tok, temp=0.0)
                    g[pname] = round(r["dec_tps"], 1)
                g_mean = round(sum(g.values()) / len(g), 1)

                mem = measure_mem(proc.pid)
                fps.append(mem["phys_footprint_gib"])
                neus.append(mem["neural_gib"])
                tots.append(mem["total_phys_gib"])
                td = {
                    "trial": t + 1,
                    "load_s": round(load_s, 2),
                    "pf_cold_tok": pf_cold["ptok"],
                    "pf_cold_ttft": round(pf_cold["ttft"], 2),
                    "pf_cold_tps": round(pf_cold["pf_tps"], 1),
                    "pf_warm_ms": round(pf_warm["ttft"] * 1000.0, 0),
                    "greedy": g,
                    "g_mean": g_mean,
                    "phys_footprint_gib": mem["phys_footprint_gib"],
                    "neural_gib": mem["neural_gib"],
                    "total_phys_gib": mem["total_phys_gib"],
                    "wired_gib": mem["wired_gib"],
                    "swapped_gib": mem["swapped_gib"],
                }
                trial_data.append(td)
                print(
                    f"  [cold3x] trial {t+1}/3: load={load_s:.2f}s | PF={td['pf_cold_tps']} tok/s ({td['pf_cold_ttft']}s)"
                    f" | Warm={td['pf_warm_ms']:.0f}ms | GMean={g_mean} {g} | TotRAM={td['total_phys_gib']} GiB",
                    flush=True,
                )
            finally:
                kill_tree(proc)

    def stats_of(vals):
        return {"median": round(statistics.median(vals), 1), "min": min(vals), "max": max(vals)}

    summary = {
        "trials": trial_data,
        "load_s": stats_of(loads),
        "pf_cold_tps": stats_of([x["pf_cold_tps"] for x in trial_data]),
        "pf_warm_ms": stats_of([x["pf_warm_ms"] for x in trial_data]),
        "g_mean": stats_of([x["g_mean"] for x in trial_data]),
        "per_task": {k: stats_of([x["greedy"][k] for x in trial_data]) for k in ("cpp_spsc", "py_lru", "cpp_refactor", "prose")},
        "phys_footprint_gib_median": round(statistics.median(fps), 3),
        "neural_gib_median": round(statistics.median(neus), 3),
        "total_phys_gib_median": round(statistics.median(tots), 3),
    }
    print(f"  -> cold3x summary: PF={summary['pf_cold_tps']} | Warm={summary['pf_warm_ms']} | GMean={summary['g_mean']} | TotRAM={summary['total_phys_gib_median']} GiB", flush=True)
    return summary


def main():
    steps = set(sys.argv[1:]) if len(sys.argv) > 1 else {"ppl", "kv3x", "cold3x", "longctx", "humaneval"}
    out_path = os.path.join(OUT_DIR, "hardening_results.json")
    all_res = json.load(open(out_path)) if os.path.exists(out_path) else {}

    if "ppl" in steps:
        all_res["wikitext2_ppl"] = run_wikitext_ppl()
        json.dump(all_res, open(out_path, "w"), indent=2)
    if "kv3x" in steps:
        all_res["kv_control_and_3x"] = run_kv_control_and_3x()
        json.dump(all_res, open(out_path, "w"), indent=2)
    if "cold3x" in steps:
        all_res["cold_3x_variance"] = run_cold_3x_variance()
        json.dump(all_res, open(out_path, "w"), indent=2)
    if "longctx" in steps:
        all_res["long_context_sweep"] = run_long_context_sweep()
        json.dump(all_res, open(out_path, "w"), indent=2)
    if "humaneval" in steps:
        all_res["humaneval_164"] = run_humaneval()
        json.dump(all_res, open(out_path, "w"), indent=2)

    print(f"\nSaved all hardening results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
