#!/usr/bin/env python3
"""
100% Fair, Deterministic, HN-Proof Benchmark Suite on Apple M4 Max (48 GB)
Model: ~/models/qwen38-pi-adapted-mlx4 (Qwen 3.8 27B Q4_AFFINE_G64 + MTP)
- Primary decode metric: ALL 4 prompts at deterministic temperature=0.0 (zero sampling variance!)
- Secondary metric: 2-run sampled (T=0.7) mean on py_lru + prose (identical run count for every engine)
- Cold 3.5k prefill (unique salt per engine) + Warm 3.5k cached prefill
- Honest RAM accounting: process private footprint + RSS + shared clean ANE model.hwx (8.56 GB)
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MOMETAL_BIN", os.path.join(REPO_DIR, "build", "mometal"))
MODEL = os.environ.get("MOMETAL_MODEL", os.path.expanduser("~/models/qwen38-pi-adapted-mlx4"))
BASE = os.environ.get("MOMETAL_BENCH_BASE", os.path.join(REPO_DIR, "benchmarks", "engines"))
MLX_SERVE_MODEL = os.path.join(BASE, "shadow_mlx_serve_model")
OMLX_DIR = os.path.join(BASE, "shadow_omlx_models")
TF_MODEL = os.path.join(BASE, "shadow_tf_model")
PORT = 8099
URL = f"http://127.0.0.1:{PORT}"

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

CACHE_TMP_BASE = os.path.join(BASE, "hn_cache_tmp")
if os.path.exists(CACHE_TMP_BASE):
    shutil.rmtree(CACHE_TMP_BASE)
os.makedirs(CACHE_TMP_BASE, exist_ok=True)

ENGINES = [
    {
        "id": "mi_g64_gpu",
        "name": "metal-infer (exact g64, GPU-only)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            BIN, "serve",
            "--model", MODEL,
            "--ane-models", "none",
            "--max-seq", "32768",
            "--kv-bits", "8",
            "--cache-dir", os.path.join(CACHE_TMP_BASE, "mi_gpu"),
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        "model_id": MODEL,
    },
    {
        "id": "mi_g64_ane",
        "name": "metal-infer (exact g64 + ANE prefill, 2-copy)",
        "shared_hwx_gb": 8.56,
        "cmd": [
            BIN, "serve",
            "--model", MODEL,
            "--ane-tag", "hf4",
            "--max-seq", "32768",
            "--kv-bits", "8",
            "--cache-dir", os.path.join(CACHE_TMP_BASE, "mi_2copy"),
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        "model_id": MODEL,
    },
    {
        "id": "mi_ane_share",
        "name": "metal-infer (1-copy ANE-Share hf4, 128k KV8)",
        "shared_hwx_gb": 8.56,
        "cmd": [
            BIN, "serve",
            "--model", MODEL,
            "--ane-tag", "hf4",
            "--ane-share",
            "--max-seq", "131072",
            "--kv-bits", "8",
            "--cache-dir", os.path.join(CACHE_TMP_BASE, "mi_1copy"),
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        "model_id": MODEL,
    },
    {
        "id": "tensorfold_mtp6",
        "name": "TensorFold v1.3.4 (Zig + Metal + MTP-6)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            f"{BASE}/tensorfold/bin/tensorfold-native", "serve", TF_MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--drafter", "none",
            "--mtp-drafts", "6",
            "--parallel", "1",
            "--context", "32768",
            "--no-thinking",
            "--no-update-check",
        ],
        "model_id": None,
    },
    {
        "id": "mlx_serve_warm",
        "name": "mlx-serve v26.11.3 (Zig + MTP + KV8 + warmup-full)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            f"{BASE}/mlx-serve/mlx-serve-macos-arm64/mlx-serve",
            "--model", MLX_SERVE_MODEL,
            "--serve",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--kv-quant", "8",
            "--mtp",
            "--pld",
            "--warmup-full",
            "--log-file", "off",
        ],
        "model_id": None,
    },
    {
        "id": "omlx",
        "name": "oMLX v0.8.3 (Python MLX + MTP)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            f"{BASE}/omlx/venv/bin/omlx", "serve",
            "--model-dir", OMLX_DIR,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--headless",
        ],
        "model_id": "qwen38-pi-adapted-mlx4",
    },
    {
        "id": "pmetal",
        "name": "pmetal v0.7.1 (Rust Metal/MLX, AR)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            f"{BASE}/pmetal/pmetal-v0.7.1-aarch64-apple-darwin", "serve",
            "--model", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        "model_id": None,
        "cwd": f"{BASE}/pmetal",
    },
    {
        "id": "mlx_lm",
        "name": "mlx-lm v0.31.3 (Apple Reference, AR)",
        "shared_hwx_gb": 0.0,
        "cmd": [
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/mlx_lm.server",
            "--model", MODEL,
            "--host", "127.0.0.1", "--port", str(PORT),
        ],
        "model_id": None,
    },
]


def http_get_json(path, timeout=3):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(proc, timeout_s=240):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            return None, time.perf_counter() - t0
        for ep in ("/v1/models", "/health"):
            try:
                d = http_get_json(ep, timeout=2)
                if ep == "/v1/models":
                    models = d.get("data") or []
                    if models:
                        return models[0].get("id"), time.perf_counter() - t0
                elif isinstance(d, dict) and d.get("status") in ("ok", "healthy", "ready"):
                    try:
                        m = http_get_json("/v1/models", timeout=2)
                        ml = m.get("data") or []
                        if ml:
                            return ml[0].get("id"), time.perf_counter() - t0
                    except Exception:
                        pass
                    return MODEL, time.perf_counter() - t0
            except Exception:
                pass
        time.sleep(0.5)
    return None, time.perf_counter() - t0


def stream_chat(model_id, messages, max_tokens, temp=0.0):
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
    text_parts = []
    usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
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
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
                    text_parts.append(c)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    ptok = (usage or {}).get("prompt_tokens") or 0
    ctok = (usage or {}).get("completion_tokens") or chunks
    ttft = t_first - t0
    gen_s = max(t_end - t_first, 1e-6)
    dec_tps = max(ctok - 1, 1) / gen_s if ctok > 1 else ctok / max(t_end - t0, 1e-6)
    pf_tps = ptok / max(ttft, 1e-6) if ptok > 0 else 0.0
    txt = "".join(text_parts)
    return {
        "ptok": ptok,
        "ctok": ctok,
        "ttft": ttft,
        "pf_tps": pf_tps,
        "dec_tps": dec_tps,
        "sha": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12],
        "text": txt,
    }


def measure_mem(pid):
    gib = float(1024 ** 3)
    rss_gib = phys_gib = neural_gib = wired_gib = swapped_gib = 0.0
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        if out:
            rss_gib = int(out) * 1024.0 / gib
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
            phys_gib = int(m_fp.group(1)) / gib
        if m_neu:
            neural_gib = int(m_neu.group(1)) / gib
        if m_tot:
            swapped_gib = int(m_tot.group(2)) / gib
            wired_gib = int(m_tot.group(5)) / gib
    except Exception:
        pass
    return phys_gib, neural_gib, wired_gib, swapped_gib, rss_gib


def kill_tree(proc):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=5)
    time.sleep(2.5)


def run_engine(eng):
    name = eng["name"]
    eid = eng["id"]
    log_path = os.path.join(BASE, f"hn_log_{eid}.txt")
    print(f"\n================ {name} ================", flush=True)
    res = {"id": eid, "name": name, "status": "ok", "shared_hwx_gb": eng["shared_hwx_gb"]}
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            eng["cmd"],
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=eng.get("cwd"),
            start_new_session=True,
        )
        try:
            disc_id, load_s = wait_ready(proc, timeout_s=240)
            if disc_id is None:
                res["status"] = "failed_start"
                print(f"  FAILED TO START ({load_s:.1f}s). See {log_path}", flush=True)
                return res
            model_id = eng["model_id"] or disc_id
            res["load_s"] = round(load_s, 2)
            print(f"  Ready in {load_s:.2f}s | model_id={model_id}", flush=True)

            # Two warmup calls (short + medium) to ensure Metal shaders are hot
            stream_chat(model_id, [{"role": "user", "content": "Reply OK."}], 8, temp=0.0)
            stream_chat(model_id, [{"role": "user", "content": "Write a 3-line Python hello function."}], 32, temp=0.0)

            # Unique-salted 3.5k cold prefill + exact warm repeat
            salt = f"[hn_bench_{eid}_{int(time.time())}] "
            pad = salt + "The quick brown fox jumps over the lazy dog near the riverbank at dawn while compilers optimize memory layouts. " * 175
            pf_cold = stream_chat(model_id, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
            res["pf_cold_tok"] = pf_cold["ptok"]
            res["pf_cold_ttft"] = round(pf_cold["ttft"], 2)
            res["pf_cold_tps"] = round(pf_cold["pf_tps"], 1)
            print(f"  Cold 3.5k PF: {pf_cold['ptok']} tok, TTFT={pf_cold['ttft']:.2f}s ({pf_cold['pf_tps']:.0f} tok/s)", flush=True)

            pf_warm = stream_chat(model_id, [{"role": "user", "content": pad + "\nReply with the single word OK."}], 2, temp=0.0)
            res["pf_warm_ms"] = round(pf_warm["ttft"] * 1000.0, 0)
            print(f"  Warm 3.5k PF: TTFT={res['pf_warm_ms']:.0f} ms", flush=True)

            # Primary: 4 deterministic greedy (T=0.0) prompts
            g_vals = []
            for pname, prompt, max_tok in GREEDY_PROMPTS:
                r = stream_chat(model_id, [{"role": "user", "content": prompt}], max_tok, temp=0.0)
                res[f"g_{pname}"] = round(r["dec_tps"], 1)
                res[f"g_{pname}_sha"] = r["sha"]
                res[f"g_{pname}_tok"] = r["ctok"]
                g_vals.append(r["dec_tps"])
                head = (r["text"].strip().splitlines()[0][:55]) if r["text"].strip() else ""
                print(f"  Greedy {pname:<12}: {r['ctok']:>3} tok | {r['dec_tps']:>6.1f} tok/s | sha={r['sha']} | '{head}'", flush=True)
                if eid in ("mi_g64_gpu", "mi_ane_share", "tensorfold_mtp6", "mlx_serve_warm") and pname in ("cpp_spsc", "py_lru"):
                    with open(os.path.join(BASE, f"out_{eid}_{pname}.txt"), "w") as of:
                        of.write(r["text"])
            res["g_mean"] = round(sum(g_vals) / len(g_vals), 1)

            # Secondary: 2x sampled (T=0.7) on py_lru + prose
            s_py, s_pr = [], []
            for pname, prompt, max_tok in SAMPLED_PROMPTS:
                r = stream_chat(model_id, [{"role": "user", "content": prompt}], max_tok, temp=0.7)
                if pname.startswith("py_lru"):
                    s_py.append(r["dec_tps"])
                else:
                    s_pr.append(r["dec_tps"])
                print(f"  Sampled {pname:<11}: {r['ctok']:>3} tok | {r['dec_tps']:>6.1f} tok/s", flush=True)
            res["s_py_mean"] = round(sum(s_py) / len(s_py), 1)
            res["s_prose_mean"] = round(sum(s_pr) / len(s_pr), 1)

            phys_gib, neural_gib, wired_gib, swapped_gib, rss_gib = measure_mem(proc.pid)
            res["phys_footprint_gib"] = round(phys_gib, 3)
            res["neural_ledger_gib"] = round(neural_gib, 3)
            res["total_phys_gib"] = round(phys_gib + neural_gib, 3)
            res["wired_gib"] = round(wired_gib, 3)
            res["swapped_gib"] = round(swapped_gib, 3)
            res["rss_gib"] = round(rss_gib, 3)
            res["footprint_gb"] = round(phys_gib, 2)
            res["rss_gb"] = round(rss_gib, 2)
            res["total_sys_gb"] = round(phys_gib + neural_gib, 2)
            print(
                f"  Memory: phys_footprint={phys_gib:.3f} GiB | neural={neural_gib:.3f} GiB | "
                f"total_phys={phys_gib + neural_gib:.3f} GiB | wired={wired_gib:.3f} GiB | "
                f"swapped={swapped_gib:.3f} GiB | RSS={rss_gib:.3f} GiB",
                flush=True,
            )
        except Exception as e:
            res["status"] = f"error: {e}"
            print(f"  ERROR: {e}", flush=True)
        finally:
            kill_tree(proc)
    return res


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    results = []
    for eng in ENGINES:
        if only and eng["id"] not in only:
            continue
        results.append(run_engine(eng))

    out_json = os.path.join(BASE, "hn_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n====================== HN-PROOF DETERMINISTIC SUMMARY ======================", flush=True)
    hdr = f"{'Engine':<44} {'PF3.5k':>6} {'Cache':>7} {'C++':>5} {'Py':>5} {'Ref':>5} {'Prs':>5} {'GMean':>6} {'Py.7':>5} {'Pr.7':>5} {'FP':>5} {'TotRAM':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['name']:<44} {r['status']}", flush=True)
            continue
        print(
            f"{r['name']:<44} {r['pf_cold_tps']:>6.0f} {r['pf_warm_ms']:>6.0f}ms"
            f" {r['g_cpp_spsc']:>5.1f} {r['g_py_lru']:>5.1f} {r['g_cpp_refactor']:>5.1f} {r['g_prose']:>5.1f}"
            f" {r['g_mean']:>6.1f} {r['s_py_mean']:>5.1f} {r['s_prose_mean']:>5.1f}"
            f" {r['footprint_gb']:>5.2f} {r['total_sys_gb']:>6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
