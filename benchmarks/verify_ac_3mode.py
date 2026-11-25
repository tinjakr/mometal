#!/usr/bin/env python3
"""
Clean AC-Power Interleaved Verification Suite for metal-infer
- Records environment, power/thermal status, binary SHA-256, idle vm_stat, and top background processes.
- Runs 3 metal-infer modes (1-copy ANE-Share hf4, GPU-only g64, 2-copy ANE+g64) in balanced
  interleaved order (n=3 fresh restarts each) on AC power so speed and byte-exact memory come
  from the exact same AC session.
- Runs matched controls: 1-copy no-cache-dir cold prefill and 1-copy BF16-KV control.
- Runs fresh-server 66k (1-copy dense vs 2-copy dense) while logging vm_stat pageins/swapins/
  compressions/decompressions separately during prefill and during decode.
- Persists raw footprint and vm_stat dumps in benchmarks/raw_ac_proof/.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MOMETAL_BIN", os.path.join(REPO_DIR, "build", "mometal"))
MODEL = os.environ.get("MOMETAL_MODEL", os.path.expanduser("~/models/qwen38-pi-adapted-mlx4"))
BENCH_DIR = os.path.join(REPO_DIR, "benchmarks")
RAW_DIR = os.environ.get("MOMETAL_RAW_DIR", os.path.join(BENCH_DIR, "raw_ac_proof"))
CACHE_BASE = "/tmp/mometal_ac_verify_cache"
PORT = 8098
URL = f"http://127.0.0.1:{PORT}"
GIB = float(1024 ** 3)
MIB = float(1024 ** 2)

GREEDY_PROMPTS = [
    (
        "cpp_spsc",
        "Write a complete, production-quality C++20 lock-free single-producer single-consumer (SPSC) ring buffer template class with cache-line padding, power-of-two capacity, acquire/release memory ordering, try_push, try_pop, and a short main() benchmark.",
        384,
    ),
    (
        "py_lru",
        "Write a clean Python 3.12 async LRU cache class with per-entry TTL expiration, asyncio.Lock safety, hit/miss statistics, max_size eviction, and 4 unit tests using asyncio.run().",
        384,
    ),
    (
        "cpp_refactor",
        """Refactor the following C++ class to add std::shared_mutex thread safety, RAII lock guards, explicit error handling with std::expected, and a batch_insert() method while preserving the existing API style:

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
```""",
        384,
    ),
    (
        "prose",
        "Explain in clear technical detail how Apple Silicon unified memory, Metal simdgroup_matrix AMX tiles, and KV-cache quantization interact to determine LLM prefill versus decode bottlenecks.",
        384,
    ),
]

SAMPLED_PROMPTS = [
    ("py_lru_s1", GREEDY_PROMPTS[1][1], 384),
    ("py_lru_s2", GREEDY_PROMPTS[1][1], 384),
    ("prose_s1", GREEDY_PROMPTS[3][1], 384),
    ("prose_s2", GREEDY_PROMPTS[3][1], 384),
]


def run_cmd(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def parse_vm_stat(raw=None):
    if raw is None:
        raw = run_cmd(["vm_stat"])
    page_size = 16384
    m = re.search(r"page size of (\d+) bytes", raw)
    if m:
        page_size = int(m.group(1))
    vals = {"page_size": page_size}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        if v.isdigit():
            vals[k.strip()] = int(v)
    return vals, raw


def vm_gib(vm, key):
    return round(vm.get(key, 0) * vm.get("page_size", 16384) / GIB, 3)


def vm_delta(after, before):
    ps = after.get("page_size", 16384)
    keys = [
        "Pages wired down",
        "Pages occupied by compressor",
        "Pages stored in compressor",
        "Pageins",
        "Pageouts",
        "Swapins",
        "Swapouts",
        "Compressions",
        "Decompressions",
    ]
    out = {}
    for k in keys:
        d = after.get(k, 0) - before.get(k, 0)
        out[k] = d
        out[k + " (GiB)"] = round(d * ps / GIB, 3)
    return out


def get_aned_rss_mib():
    try:
        out = run_cmd(["ps", "-axo", "rss,comm"])
        total_kb = 0
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1].endswith("/aned"):
                total_kb += int(parts[0])
        return round(total_kb * 1024.0 / MIB, 2)
    except Exception:
        return 0.0


def count_hwx_lsof(pid):
    try:
        out = subprocess.check_output(["lsof", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL)
        files = []
        total_b = 0
        for line in out.splitlines():
            if "model.hwx" in line:
                parts = line.split()
                if len(parts) >= 9:
                    path = parts[-1]
                    if path not in files and os.path.exists(path):
                        files.append(path)
                        total_b += os.path.getsize(path)
        return len(files), total_b
    except Exception:
        return 0, 0


def parse_footprint(pid, save_path=None):
    raw = subprocess.check_output(
        ["footprint", "-f", "bytes", "--wired", "--swapped", "-p", str(pid)],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    if save_path:
        with open(save_path, "w") as f:
            f.write(raw)

    d = {
        "phys_footprint_gib": 0.0,
        "phys_footprint_peak_gib": 0.0,
        "neural_ledger_gib": 0.0,
        "neural_peak_gib": 0.0,
        "process_wired_gib": 0.0,
        "swapped_gib": 0.0,
        "ioaccelerator_dirty_gib": 0.0,
        "unmapped_graphics_dirty_gib": 0.0,
        "mapped_file_wired_gib": 0.0,
        "unmapped_neural_wired_gib": 0.0,
        "unmapped_neural_regions": 0,
    }
    m = re.search(r"phys_footprint:\s*(\d+)\s*B", raw)
    if m:
        d["phys_footprint_gib"] = round(int(m.group(1)) / GIB, 3)
    m = re.search(r"phys_footprint_peak:\s*(\d+)\s*B", raw)
    if m:
        d["phys_footprint_peak_gib"] = round(int(m.group(1)) / GIB, 3)
    m = re.search(r"neural:\s*(\d+)\s*B", raw)
    if m:
        d["neural_ledger_gib"] = round(int(m.group(1)) / GIB, 3)
    m = re.search(r"neural_peak:\s*(\d+)\s*B", raw)
    if m:
        d["neural_peak_gib"] = round(int(m.group(1)) / GIB, 3)

    for line in raw.splitlines():
        m_row = re.match(
            r"^\s*(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s*B\s+(\d+)\s+(.+?)\s*$",
            line,
        )
        if not m_row:
            continue
        dirty_b = int(m_row.group(1))
        swap_b = int(m_row.group(2))
        recl_b = int(m_row.group(4))
        wired_b = int(m_row.group(5))
        regs = int(m_row.group(6))
        cat = m_row.group(7).strip()
        if cat == "TOTAL":
            d["swapped_gib"] = round(swap_b / GIB, 3)
            d["process_wired_gib"] = round(wired_b / GIB, 3)
            d["reclaimable_gib"] = round(recl_b / GIB, 3)
        elif cat == "IOAccelerator (graphics)":
            d["ioaccelerator_dirty_gib"] = round(dirty_b / GIB, 3)
        elif cat == "Owned physical footprint (unmapped) (graphics)":
            d["unmapped_graphics_dirty_gib"] = round(dirty_b / GIB, 3)
        elif cat == "mapped file":
            d["mapped_file_wired_gib"] = round(wired_b / GIB, 3)
        elif "unmapped) (neural)" in cat:
            d["unmapped_neural_wired_gib"] = round(wired_b / GIB, 3)
            d["unmapped_neural_reclaimable_gib"] = round(recl_b / GIB, 3)
            d["unmapped_neural_regions"] = regs

    d["total_phys_plus_neural_gib"] = round(d["phys_footprint_gib"] + d["neural_ledger_gib"], 3)
    d["total_phys_plus_neural_peak_gib"] = round(d["phys_footprint_gib"] + d["neural_peak_gib"], 3)
    return d


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
    time.sleep(3.0)


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
        time.sleep(0.25)
    return None, time.perf_counter() - t0


def stream_chat(model_id, messages, max_tokens, temp=0.0, timeout=900, on_first_token=None):
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temp > 0:
        body["top_p"] = 0.95
        body["top_k"] = 20
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
                        if on_first_token is not None:
                            on_first_token()
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
    txt = "".join(parts)
    return {
        "ptok": ptok,
        "ctok": ctok,
        "ttft": ttft,
        "pf_tps": pf_tps,
        "dec_tps": dec_tps,
        "sha": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12],
        "text": txt,
    }


def make_cmd(mode, kv_bits=8, max_seq=131072, cache_dir=None):
    cmd = [
        BIN, "serve", "--model", MODEL,
        "--max-seq", str(max_seq),
        "--kv-bits", str(kv_bits),
        "--host", "127.0.0.1", "--port", str(PORT),
    ]
    if mode == "one_copy_ane_share":
        cmd += ["--ane-tag", "hf4", "--ane-share"]
    elif mode == "two_copy_ane":
        cmd += ["--ane-tag", "hf4"]
    elif mode == "gpu_only_g64":
        cmd += ["--ane-models", "none"]
    else:
        raise ValueError(mode)
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    return cmd


def run_one_3mode_trial(mode, trial_idx, kv_bits=8, max_seq=131072, use_cache_dir=True, run_sampled=True):
    tag = f"{mode}_t{trial_idx}" + ("" if use_cache_dir else "_nocache") + (f"_kv{kv_bits}" if kv_bits != 8 else "")
    tcache = os.path.join(CACHE_BASE, tag) if use_cache_dir else None
    if tcache and os.path.exists(tcache):
        shutil.rmtree(tcache)
    cmd = make_cmd(mode, kv_bits=kv_bits, max_seq=max_seq, cache_dir=tcache)

    vm_before, vm_before_raw = parse_vm_stat()
    with open(os.path.join(RAW_DIR, f"vm_{tag}_before.txt"), "w") as f:
        f.write(vm_before_raw)

    log_path = os.path.join(RAW_DIR, f"srv_{tag}.log")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            mid, load_s = wait_ready(proc)
            if not mid:
                raise RuntimeError(f"Server failed to start for {tag}")
            stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)
            stream_chat(mid, [{"role": "user", "content": "Write a 3-line Python hello function."}], 32, temp=0.0)

            salt = f"[ac_verify_{tag}_{int(time.time())}] "
            pad = salt + "The quick brown fox jumps over the lazy dog near the riverbank at dawn while compilers optimize memory layouts. " * 175
            pf_cold = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
            pf_warm = stream_chat(mid, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)

            greedy = {}
            shas = {}
            for pname, prompt, max_tok in GREEDY_PROMPTS:
                r = stream_chat(mid, [{"role": "user", "content": prompt}], max_tok, temp=0.0)
                greedy[pname] = round(r["dec_tps"], 1)
                shas[pname] = r["sha"]
            g_mean = round(sum(greedy.values()) / len(greedy), 1)

            sampled = {}
            if run_sampled:
                s_py, s_pr = [], []
                for pname, prompt, max_tok in SAMPLED_PROMPTS:
                    r = stream_chat(mid, [{"role": "user", "content": prompt}], max_tok, temp=0.7)
                    val = round(r["dec_tps"], 1)
                    sampled[pname] = val
                    if pname.startswith("py_lru"):
                        s_py.append(val)
                    else:
                        s_pr.append(val)
                sampled["s_py_mean"] = round(sum(s_py) / len(s_py), 1)
                sampled["s_prose_mean"] = round(sum(s_pr) / len(s_pr), 1)

            fp = parse_footprint(proc.pid, save_path=os.path.join(RAW_DIR, f"fp_{tag}.txt"))
            vm_after, vm_after_raw = parse_vm_stat()
            with open(os.path.join(RAW_DIR, f"vm_{tag}_after.txt"), "w") as f:
                f.write(vm_after_raw)
            vmd = vm_delta(vm_after, vm_before)
            aned_mib = get_aned_rss_mib()
            hwx_n, hwx_b = count_hwx_lsof(proc.pid)

            res = {
                "tag": tag,
                "mode": mode,
                "trial": trial_idx,
                "kv_bits": kv_bits,
                "use_cache_dir": use_cache_dir,
                "load_s": round(load_s, 2),
                "pf_cold_tok": pf_cold["ptok"],
                "pf_cold_ttft_s": round(pf_cold["ttft"], 2),
                "pf_cold_tps": round(pf_cold["pf_tps"], 1),
                "pf_warm_ms": round(pf_warm["ttft"] * 1000.0, 1),
                "greedy": greedy,
                "shas": shas,
                "g_mean": g_mean,
                "sampled": sampled,
                "footprint": fp,
                "vm_wired_gib": vm_gib(vm_after, "Pages wired down"),
                "vm_wired_delta_gib": vmd["Pages wired down (GiB)"],
                "vm_compressor_delta_gib": vmd["Pages occupied by compressor (GiB)"],
                "vm_swapins_delta": vmd["Swapins"],
                "vm_pageins_delta": vmd["Pageins"],
                "aned_rss_mib": aned_mib,
                "lsof_hwx_count": hwx_n,
                "lsof_hwx_bytes": hwx_b,
            }
            print(
                f"  [{tag:<24}] load={load_s:.2f}s | PF={res['pf_cold_tps']:>5.1f} tok/s ({res['pf_warm_ms']:>5.0f}ms)"
                f" | GMean={g_mean:>5.1f} {greedy}"
                f" | FP={fp['phys_footprint_gib']:.3f}+{fp['neural_ledger_gib']:.3f}={fp['total_phys_plus_neural_gib']:.3f} GiB"
                f" | vmWired+={vmd['Pages wired down (GiB)']:.3f} swap={fp['swapped_gib']:.3f}",
                flush=True,
            )
            return res
        finally:
            kill_tree(proc)


def run_66k_paging_check():
    print("\n================ [66K AC PAGING CHECK: 1-COPY vs 2-COPY] ================", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    corpus_text = "\n\n".join(x["text"] for x in ds if len(x["text"].strip()) > 40)
    corpus_ids = tok.encode(corpus_text[:900000], add_special_tokens=False)
    text_body = tok.decode(corpus_ids[:65780], skip_special_tokens=True)

    out = {}
    for mode in ("one_copy_ane_share", "two_copy_ane"):
        tag = f"66k_{mode}"
        cmd = make_cmd(mode, kv_bits=8, max_seq=131072, cache_dir=None)
        vm_before_launch, raw_bl = parse_vm_stat()
        with open(os.path.join(RAW_DIR, f"vm_{tag}_before_launch.txt"), "w") as f:
            f.write(raw_bl)

        log_path = os.path.join(RAW_DIR, f"srv_{tag}.log")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                mid, load_s = wait_ready(proc)
                if not mid:
                    raise RuntimeError(f"Failed to start {tag}")
                stream_chat(mid, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)

                vm_before_req, raw_br = parse_vm_stat()
                with open(os.path.join(RAW_DIR, f"vm_{tag}_before_req.txt"), "w") as f:
                    f.write(raw_br)

                holder = {}
                def on_first_tok():
                    vm_ft, raw_ft = parse_vm_stat()
                    holder["vm_ft"] = vm_ft
                    with open(os.path.join(RAW_DIR, f"vm_{tag}_first_tok.txt"), "w") as f:
                        f.write(raw_ft)

                needle = f"NEEDLE_CODE_66K_{mode}_7391"
                prompt = (
                    f"[ac_66k_{mode}_{int(time.time())}] Reference code: {needle}.\n\n"
                    f"{text_body}\n\n"
                    f"First state the exact Reference code from the top of this message, then write a concise 200-word technical summary of the themes above."
                )
                r = stream_chat(mid, [{"role": "user", "content": prompt}], 256, temp=0.0, timeout=1200, on_first_token=on_first_tok)
                vm_after_req, raw_ar = parse_vm_stat()
                with open(os.path.join(RAW_DIR, f"vm_{tag}_after_req.txt"), "w") as f:
                    f.write(raw_ar)
                fp = parse_footprint(proc.pid, save_path=os.path.join(RAW_DIR, f"fp_{tag}.txt"))
                aned_mib = get_aned_rss_mib()

                vm_ft = holder.get("vm_ft", vm_before_req)
                pf_vm_d = vm_delta(vm_ft, vm_before_req)
                dec_vm_d = vm_delta(vm_after_req, vm_ft)
                tot_vm_d = vm_delta(vm_after_req, vm_before_launch)

                entry = {
                    "mode": mode,
                    "load_s": round(load_s, 2),
                    "prompt_tokens": r["ptok"],
                    "gen_tokens": r["ctok"],
                    "ttft_s": round(r["ttft"], 2),
                    "prefill_tps": round(r["pf_tps"], 1),
                    "decode_tps": round(r["dec_tps"], 1),
                    "needle_hit": needle in r["text"],
                    "footprint": fp,
                    "aned_rss_mib": aned_mib,
                    "vm_compressor_after_gib": vm_gib(vm_after_req, "Pages occupied by compressor"),
                    "vm_wired_after_gib": vm_gib(vm_after_req, "Pages wired down"),
                    "vm_delta_total": tot_vm_d,
                    "vm_delta_during_prefill": pf_vm_d,
                    "vm_delta_during_decode": dec_vm_d,
                }
                out[mode] = entry
                print(
                    f"  [{tag}] ptok={r['ptok']} PF={entry['prefill_tps']:.1f} tok/s ({entry['ttft_s']}s) "
                    f"Dec={entry['decode_tps']:.1f} tok/s needle={entry['needle_hit']} | "
                    f"RAM={fp['phys_footprint_gib']:.3f}+{fp['neural_ledger_gib']:.3f}={fp['total_phys_plus_neural_gib']:.3f} GiB "
                    f"(wired={fp['process_wired_gib']:.3f}, swap={fp['swapped_gib']:.3f}) | "
                    f"DEC_PAGING: pageins={dec_vm_d['Pageins']} swapins={dec_vm_d['Swapins']} "
                    f"comp={dec_vm_d['Compressions']} decomp={dec_vm_d['Decompressions']}",
                    flush=True,
                )
            finally:
                kill_tree(proc)
    return out


def summarize_mode(trials):
    def med(vals):
        return round(statistics.median(vals), 1)
    def med3(vals):
        return round(statistics.median(vals), 3)

    all_s_py = [t["sampled"][k] for t in trials for k in ("py_lru_s1", "py_lru_s2") if k in t["sampled"]]
    all_s_pr = [t["sampled"][k] for t in trials for k in ("prose_s1", "prose_s2") if k in t["sampled"]]
    return {
        "n_trials": len(trials),
        "load_s_median": round(statistics.median(t["load_s"] for t in trials), 2),
        "pf_cold_tok": trials[0]["pf_cold_tok"],
        "pf_cold_ttft_s_median": round(statistics.median(t["pf_cold_ttft_s"] for t in trials), 2),
        "pf_cold_tps_median": med([t["pf_cold_tps"] for t in trials]),
        "pf_cold_tps_range": [min(t["pf_cold_tps"] for t in trials), max(t["pf_cold_tps"] for t in trials)],
        "pf_warm_ms_median": round(statistics.median(t["pf_warm_ms"] for t in trials), 0),
        "greedy_median": {k: med([t["greedy"][k] for t in trials]) for k in ("cpp_spsc", "py_lru", "cpp_refactor", "prose")},
        "g_mean_median": med([t["g_mean"] for t in trials]),
        "g_mean_range": [min(t["g_mean"] for t in trials), max(t["g_mean"] for t in trials)],
        "shas": trials[0]["shas"],
        "shas_consistent_across_trials": all(t["shas"] == trials[0]["shas"] for t in trials),
        "s_py_all": all_s_py,
        "s_py_mean": round(sum(all_s_py) / len(all_s_py), 1) if all_s_py else 0.0,
        "s_py_range": [min(all_s_py), max(all_s_py)] if all_s_py else [],
        "s_prose_all": all_s_pr,
        "s_prose_mean": round(sum(all_s_pr) / len(all_s_pr), 1) if all_s_pr else 0.0,
        "s_prose_range": [min(all_s_pr), max(all_s_pr)] if all_s_pr else [],
        "phys_footprint_gib_median": med3([t["footprint"]["phys_footprint_gib"] for t in trials]),
        "neural_ledger_gib_median": med3([t["footprint"]["neural_ledger_gib"] for t in trials]),
        "total_phys_plus_neural_gib_median": med3([t["footprint"]["total_phys_plus_neural_gib"] for t in trials]),
        "process_wired_gib_median": med3([t["footprint"]["process_wired_gib"] for t in trials]),
        "swapped_gib_median": med3([t["footprint"]["swapped_gib"] for t in trials]),
        "vm_wired_delta_gib_median": med3([t["vm_wired_delta_gib"] for t in trials]),
        "vm_compressor_delta_gib_median": med3([t["vm_compressor_delta_gib"] for t in trials]),
        "ioaccelerator_dirty_gib_median": med3([t["footprint"]["ioaccelerator_dirty_gib"] for t in trials]),
        "unmapped_graphics_dirty_gib_median": med3([t["footprint"]["unmapped_graphics_dirty_gib"] for t in trials]),
        "mapped_file_wired_gib_median": med3([t["footprint"]["mapped_file_wired_gib"] for t in trials]),
        "unmapped_neural_wired_gib_median": med3([t["footprint"]["unmapped_neural_wired_gib"] for t in trials]),
        "unmapped_neural_regions": trials[0]["footprint"]["unmapped_neural_regions"],
        "aned_rss_mib_median": round(statistics.median(t["aned_rss_mib"] for t in trials), 2),
        "lsof_hwx_count": trials[0]["lsof_hwx_count"],
        "lsof_hwx_bytes": trials[0]["lsof_hwx_bytes"],
    }


def main():
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(CACHE_BASE):
        shutil.rmtree(CACHE_BASE)
    os.makedirs(CACHE_BASE, exist_ok=True)

    # Capture environment snapshot
    env_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sw_vers": run_cmd(["sw_vers"]).strip(),
        "chip": run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "memsize_gib": round(int(run_cmd(["sysctl", "-n", "hw.memsize"]).strip()) / GIB, 2),
        "pmset_batt": run_cmd(["pmset", "-g", "batt"]).strip(),
        "pmset_therm": run_cmd(["pmset", "-g", "therm"]).strip(),
        "binary_sha256": run_cmd(["shasum", "-a", "256", BIN]).split()[0],
    }
    idle_vm, idle_vm_raw = parse_vm_stat()
    top_procs = run_cmd(["sh", "-c", "ps -axo pid,rss,comm -r | head -n 21"]).strip()
    env_info["idle_vm_wired_gib"] = vm_gib(idle_vm, "Pages wired down")
    env_info["idle_vm_compressor_gib"] = vm_gib(idle_vm, "Pages occupied by compressor")
    env_info["idle_vm_free_gib"] = vm_gib(idle_vm, "Pages free")
    env_info["top_background_processes"] = top_procs

    with open(os.path.join(RAW_DIR, "env_snapshot.txt"), "w") as f:
        f.write(json.dumps(env_info, indent=2) + "\n\n--- IDLE vm_stat ---\n" + idle_vm_raw)

    print("================ ENVIRONMENT SNAPSHOT ================", flush=True)
    print(f"Power: {env_info['pmset_batt'].splitlines()[0]}", flush=True)
    print(f"Binary SHA-256: {env_info['binary_sha256']}", flush=True)
    print(f"Idle wired={env_info['idle_vm_wired_gib']} GiB, compressor={env_info['idle_vm_compressor_gib']} GiB", flush=True)

    # Balanced interleaved order across 3 modes (n=3 fresh restarts each)
    schedule = [
        ("one_copy_ane_share", 1),
        ("gpu_only_g64", 1),
        ("two_copy_ane", 1),
        ("two_copy_ane", 2),
        ("gpu_only_g64", 2),
        ("one_copy_ane_share", 2),
        ("gpu_only_g64", 3),
        ("one_copy_ane_share", 3),
        ("two_copy_ane", 3),
    ]
    print("\n================ [INTERLEAVED 3-MODE AC RUN (n=3 FRESH RESTARTS)] ================", flush=True)
    by_mode = {"one_copy_ane_share": [], "gpu_only_g64": [], "two_copy_ane": []}
    all_trials = []
    for mode, tidx in schedule:
        tr = run_one_3mode_trial(mode, tidx, kv_bits=8, max_seq=131072, use_cache_dir=True, run_sampled=True)
        by_mode[mode].append(tr)
        all_trials.append(tr)

    print("\n================ [CONTROLS: NO-CACHE-DIR & BF16-KV] ================", flush=True)
    ctrl_nocache = run_one_3mode_trial("one_copy_ane_share", 1, kv_bits=8, max_seq=131072, use_cache_dir=False, run_sampled=False)
    ctrl_bf16 = run_one_3mode_trial("one_copy_ane_share", 1, kv_bits=16, max_seq=32768, use_cache_dir=True, run_sampled=False)

    paging_66k = run_66k_paging_check()

    summary = {
        "env": env_info,
        "interleaved_order": [f"{m}_t{i}" for m, i in schedule],
        "mode_summaries": {m: summarize_mode(ts) for m, ts in by_mode.items()},
        "all_trials": all_trials,
        "control_1copy_no_cache_dir": ctrl_nocache,
        "control_1copy_bf16_kv": ctrl_bf16,
        "paging_66k": paging_66k,
    }
    out_path = os.path.join(RAW_DIR, "ac_3mode_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved complete AC verification suite to {out_path}", flush=True)


if __name__ == "__main__":
    main()
