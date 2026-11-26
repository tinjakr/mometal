# mometal

From-scratch C++20, Objective-C++, and Metal inference engine for Apple Silicon featuring **single-copy Apple Neural Engine (ANE) + Metal GPU weight sharing**, **native zero-disk-copy DFlash 2 block-diffusion speculative decoding (`--dflash`)**, a **native zero-Python ARM64 NEON 4-bit ANE builder (`hf4` / `lut4` / `nf4`)**, a **`0.16 ms` BitScreen draft head**, and **zero Python / zero MLX runtime dependencies**.

Full engineering report & open CC BY 4.0 datasets: **[starmo.ai/reports/mometal-ane-share](https://starmo.ai/reports/mometal-ane-share)**

## How single-copy ANE weight sharing works

Offloading 27B dense MLP prefill to Apple's Neural Engine more than doubles prompt speed on Apple Silicon, but pairing ANE prefill with Metal decode normally wires both the 15.4 GiB `.safetensors` checkpoint and 8.7 GiB of compiled CoreML `model.hwx` + I/O buffers (~25 GiB active, pushing 48 GiB Macs into VM compression at long context).

**`mometal`** enforces a strict **single-copy ANE-Share** path (if any layer fails boot-time verification, ANE is unloaded and the engine falls back cleanly to GPU-only). It parses Apple's Mach-O (`0xbeefface`) `__KERN*` 16-chunk 4-bit weight layout inside `/Library/Caches/com.apple.aned/.../model.hwx`, verifies the bytes bit-exact against `weight.bin` (~77–88 ms via `.hwx_manifest`), and maps those files read-only via `MAP_SHARED` so Metal GPU decode reads the **exact same 7.99 GiB wired physical pages** as `com.apple.aned` while leaving duplicate `.safetensors` MLPs unmapped and unwired.

## Verified Apple M4 Max (48 GiB, AC Power) Benchmarks

### Unified Runtime Comparison (`Qwen3.8-27B`, M4 Max 48 GiB, AC Power)

`†` `SPEED-Bench` coding median on identical [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) + [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (`KV8`, 64k ctx; raw JSON: [`benchmarks/v13_splash_head2head.json`](benchmarks/v13_splash_head2head.json)). `‡` 4-task greedy mean on [`netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit`](https://huggingface.co/netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit) using built-in MTP or AR. Memory uses binary GiB from `footprint -f bytes --wired --swapped`.

| Engine | Cold Prefill | Decode | 32k PF / Dec | Total RAM | Ready |
|---|---:|---:|---:|---:|---:|
| **`mometal v1.3` (ANE-Share)** | **658 tok/s** *(707 @ 4k · 646 @ 3.7k)* | **136.2† / 98.2‡** *(154.3 peak · 121.4 T=0.7)* | **515 / 118.3†** | **14.2–17.0 GiB** *(0 swap @ 128k)* | **1.22 s** *(0.82–1.04 s MTP)* |
| `Splash 1.7.3` (ANE + DFlash 2) | 499 tok/s *(506 @ 2k)* | 130.2 tok/s† *(mean 128.9)* | 424 / 88.6† | 18.6 / 23.7 GiB *(idle / 32k)* | 95.8 s* |
| `mlx-serve v26.11.3` (Zig/MLX + MTP) | 263 tok/s | 68.1 tok/s‡ *(70.1 T=0.7)* | — | 16.48 GiB | 25.8 s |
| `oMLX v0.8.3` (Python/MLX + MTP) | 260 tok/s | 64.0 tok/s‡ *(68.4 T=0.7)* | — | 18.44 GiB | 3.57 s |
| `TensorFold v1.3.4` (Zig/Metal MTP-6) | 233 tok/s | 57.7 tok/s‡ *(59.9 T=0.7)* | — | 21.88 GiB *(32k alloc)* | 14.1 s |
| `pmetal v0.7.1` (Rust, AR) | 254 tok/s | 28.5 tok/s‡ | — | 19.69 GiB | 1.52 s |
| `mlx-lm v0.31.3` (Apple ref, AR) | 256 tok/s | 27.3 tok/s‡ | — | 17.35 GiB | 4.28 s |

- **DFlash 2 breakdown (`mometal v1.3` ANE-Share vs `Splash 1.7.3`)**: `SB-code (none)` **`136.2 / 128.8`** vs `130.2 / 128.9`; `SB-code (med)` `118.9 / 116.3` vs **`119.0 / 118.8`**; `Transcripts` `106.3 / 98.6` vs **`112.3 / 103.3`**; cold prefill `2k/8k/16k/32k` **`649 / 658 / 600 / 515`** vs `506 / 499 / 470 / 424` (`+21–32%`, `−13.7 s` 32k TTFT); decode `2k/8k/16k/32k/32k-m` `90.9 / 109.5 / 106.8 / 118.3 / 73.0` vs `105.8 / 92.6 / 97.7 / 88.6 / 83.2`; total RAM **`~15.5 / 17.0 GiB`** (`6.81 / 8.35 GiB` process FP + `~8.7 GiB` shared ANE) vs `18.59 / 23.73 GiB` (`−3.1 to −6.7 GiB`).
- **Scope notes**: `*` Splash 1.7.3's `95.75 s` cold ready includes `61.4 s` of one-time first-start ANE calibration; Splash also supports `4–8` concurrent serving lanes and GGUF/MoE models, and measured higher on `Transcripts`, `2k` decode, and `32k-medium` reasoning.

### Dense `256 → 128k` Context Scaling (`mometal` ANE-Share `hf4` + `KV8`, AC power)

Raw JSON at [`benchmarks/v12_sweep_256_to_128k.json`](benchmarks/v12_sweep_256_to_128k.json) (`10/10` passkey recall, `10/10` 64/64 single-copy ANE-Share, fresh restart per row):

| Context | Prompt tok | Prefill tok/s | TTFT | Decode tok/s | ms/rnd | Phys FP | Wired (+ANE) | Swapped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `256` | 268 | 347.0* | 0.77 s | 71.1 | 43.9 ms | 6.06 GiB | 14.62 (+8.70) | 0.03 GiB |
| `512` | 524 | **644.9** | **0.81 s** | **84.8** | 44.4 ms | **6.05 GiB** | **14.62 (+8.70)** | **0.00 GiB** |
| `1k` | 1,034 | **674.4** | **1.53 s** | **82.0** | 46.6 ms | 6.06 GiB | 14.62 (+8.70) | **0.00 GiB** |
| `2k` | 2,058 | **673.7** | **3.05 s** | 74.9 | 45.0 ms | 6.13 GiB | 14.69 (+8.70) | **0.00 GiB** |
| `4k` | 4,106 | **658.2**† | **6.24 s** | 62.5 | 51.9 ms | 6.20 GiB | 14.76 (+8.70) | 0.02 GiB |
| `8k` | 8,202 | **654.5** | **12.53 s** | 66.9 | 49.7 ms | 6.33 GiB | 14.90 (+8.70) | 0.01 GiB |
| `16k` | 16,396 | **604.7** | **27.11 s** | 66.4 | 53.6 ms | 7.09 GiB | 15.65 (+8.70) | 0.02 GiB |
| `32k` | 32,780 | **514.0** | **63.77 s** | 52.8 | 55.8 ms | 7.64 GiB | 16.20 (+8.70) | 0.04 GiB |
| `64k` | 65,548 | **396.1** | **165.47 s** | **53.4** | 63.9 ms | 8.74 GiB | 17.30 (+8.70) | 0.03 GiB |
| `128k` | 128,078 | **278.9** | **459.18 s** | **37.5** | 71.0 ms | **10.39 GiB** | **18.55 (+8.31)** | **0.00 GiB** |

- `*` Client `serve` TTFT at `268` includes 1st-token decode + HTTP framing; pure `mometal bench` is `414.8 tok/s` (`N=256`). `†` Exact `4096`-token `mometal bench` prefill reaches **`707.2 tok/s`**.
- Opt-in Quest block-sparse decode (`--sparse-kv-budget 12288` at `64k+`) reaches `~57–61 tok/s` (`KL 0.0492`, `100%` Top-5, `3/3` multi-needle); dense exact `KV8` remains the default.

### Cross-Chip M1–M5 Formats & 4-Bit Quality (WikiText-2, 2,048 scored tok, KV8)

| Format | Chips | 64L `build-ane` | WT-2 PPL | Mean KL vs `g64` | Top-1 | Top-5 | M4 Max 4k PF / M=8 |
|---|---|---:|---:|---:|---:|---:|---:|
| `exact_g64` (ref) | M1–M5 | — | **7.787** | `0.0000` | `100.00%` | `100.00%` | — |
| **`--format nf4`** | **M1–M5** | **7.23 s** | **8.070** | **`0.0347` (`−44.3%`)** | **`92.82%`** | **`99.85%`** | `429.6 tok/s` / `41.64 ms` |
| **`--format hf4`** | M4/M5+ (default) | **8.39 s** | 8.249 | `0.0550` | `89.75%` | `99.80%` | **`707.2 tok/s`** / **`38.89 ms`** |
| `--format lut4` | M1–M3 (default) | ~7.5 s | — | — | — | — | probe local* |

*(On M1–M3, `ANECompiler` silently promotes linear `hf4` to 8-bit; `lut4`/`nf4` keep true `0.503 B/wt` across M1–M5. On 164-problem greedy HumanEval, `exact_g64` scored `92.07%` vs `88.41%` on `hf4`. `*`Check local M1–M5 compatibility in 30 s with `python3 scripts/probe_m_chip.py`.)*

## Architecture Highlights

- **Zero-disk-copy DFlash 2 (`src/dflash.{hpp,mm}`, `shaders/dflash.metal`)**: On-the-fly 12-thread ARM64 NEON `BF16 → Q4_AFFINE_G64` quantization (`~963 MiB`, `130–220 ms` warm), zero-Metal-wired CPU `mmap` codebooks (`243 MiB`, `12.4 µs` median selector), single-CB 5-layer bidirectional `2048`-window Metal 3.0 draft (`4.85–5.07 ms`) overlapped with pre-encoded `M=8` verify (`38.89 ms`), verbatim-only PLD, and exact lossless rejection sampling.
- **Zero-Python C++ `build-ane` & memory management (`src/ane_builder.mm`, `src/qwen38.mm`, `src/server.cpp`)**: L1-resident NEON 64-layer `.mlmodelc` builder (`7.23–8.39 s`), lazy Vision Tower (`0 B` on text runs), `MTLHeapTypePlacement` scratch aliasing (`225.8 → 83.3 MiB`), `half2` KV scales, `0.16 ms` `BitScreen` draft head for built-in MTP, lock-free `dispatch_group_t` async ANE pipelining, SSD `MICACHE2` prefix caching, and RFC 3629 UTF-8 SSE carry buffering. Also supports Inkling 2 Mini (`src/inkling2.mm`).

## Build & Quickstart

Requires macOS 15+ on Apple Silicon (M1–M5; full suite verified on M4 Max, macOS 27.0 / `26A428`) and Xcode Command Line Tools + CMake `>= 3.24`.

```bash
# 1. Build (zero runtime dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 2. Optional: check local M1-M5 compatibility (Metal 3.0, aned cache mmap, live 4-bit ANE-Share)
python3 scripts/probe_m_chip.py

# 3. Serve single-copy ANE-Share + DFlash 2 (auto-builds 4-bit ANE MLPs natively in ~8s on first run)
./build/mometal serve \
  --model ~/models/Qwen3.8-27B-4bit \
  --dflash ~/models/Qwen3.8-27B-DFlash2 \
  --ane-share \
  --max-seq 131072 --kv-bits 8 \
  --cache-dir ~/.cache/mometal-qwen38 \
  --host 127.0.0.1 --port 8092

# Universal M1-M5 NF4 quality mode (−44.3% KL, 92.8% top-1):
./build/mometal build-ane --model ~/models/Qwen3.8-27B-4bit --format nf4
./build/mometal serve \
  --model ~/models/Qwen3.8-27B-4bit \
  --dflash ~/models/Qwen3.8-27B-DFlash2 \
  --ane-share --ane-tag nf4 \
  --max-seq 131072 --kv-bits 8 \
  --host 127.0.0.1 --port 8092

# Built-in 1-layer MTP (no external drafter): omit --dflash
# Ultra-low-RAM mode without drafter (~5.05 GiB Phys FP): add --mtp off
# Opt-in Quest block-sparse decode (64k+): add --sparse-kv-budget 12288
```

## Caveats

- Single-copy ANE-Share reads Apple's undocumented Mach-O `model.hwx` `__KERN*` layout inside `/Library/Caches/com.apple.aned/.../ModelAssetsCache`, which macOS protects with SIP (`com.apple.rootless`). Enabling `--ane-share` currently requires disabling SIP and granting your user read access (`sudo chmod -R +a "user:$USER allow list,search,readattr,readextattr,readsecurity,file_inherit,directory_inherit" /Library/Caches/com.apple.aned`). With SIP enabled, `mometal` detects `EACCES` immediately and falls back cleanly to GPU-only mode (GPU-only + `--dflash` works out of the box without touching SIP, and never duplicates weights).
- Cross-chip `lut4` / `nf4` layouts were verified bit-exact for `h13g` (M1) through `h17g` (M5) via `ANECompiler.framework`; live throughput was measured on M4 Max.

## License

Code is licensed under the [MIT License](LICENSE). Benchmark datasets under `benchmarks/` are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
