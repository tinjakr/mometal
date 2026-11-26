# mometal

From-scratch C++20, Objective-C++, and Metal inference engine for Apple Silicon — featuring **single-copy Apple Neural Engine (ANE) + Metal GPU weight sharing**, **native zero-disk-copy DFlash 2 block-diffusion speculative decoding (`--dflash`)**, a **native zero-Python ARM64 NEON 4-bit ANE builder (`hf4` / `lut4` / `nf4`)**, a **`0.16 ms` strided-interleaved BitScreen draft head**, **1-layer causal MTP dead-compute elimination**, and **zero Python / zero MLX runtime dependencies**.

Full engineering report & open CC BY 4.0 dataset: **[starmo.ai/reports/mometal-ane-share](https://starmo.ai/reports/mometal-ane-share)**

## Why single-copy ANE weight sharing?

Offloading 27B dense MLP prefill to Apple's Neural Engine more than doubles prompt speed on Apple Silicon, but standard 2-copy execution wires both the 15.4 GiB `.safetensors` weights for Metal GPU decode *and* 8.7 GiB of compiled CoreML `model.hwx` + I/O buffers (`25.20 GiB` active at 4k, `27.40 GiB` at 66k). On a 48 GiB M4 Max on AC power, 2-copy triggers **343.5 GiB of live VM compressor thrashing** during 66k prefill (`−20%`) and faults **9.06 GiB of decompressions** during decode (`−19%`).

**`mometal`** enforces a strict **single-copy ANE-Share** path (zero 2-copy duplication; if any layer cannot be verified single-copy, ANE is unloaded and the engine falls back cleanly to pure Metal GPU-only). It parses Apple's Mach-O (`0xbeefface`) `__KERN*` 16-chunk 4-bit weight layout inside `/Library/Caches/com.apple.aned/.../model.hwx` (across any readable signed or unsigned owner cache directory), runs a boot-time bit-exact verification against `weight.bin` (32-byte group-0 fingerprint + mid/last-group spot checks per projection, accelerated to `~77–88 ms` via `.hwx_manifest`), and maps those files read-only via `MAP_SHARED` so Metal GPU decode reads the **exact same wired physical pages** as `com.apple.aned` while leaving duplicate `.safetensors` MLPs unmapped and unwired.

### Cross-Chip Apple Silicon Support (M1–M5) & Wave 3 `nf4`
- **Metal 3.0 shaders across M1–M5**: `common.metal`, `qwen38.metal`, `dflash.metal`, `inkling2.metal`, and `vision.metal` all compile under Metal 3.0 (M1/M2+), while automatically enabling Metal 3.1 native `simdgroup_bfloat8x8` in `vision.metal` on Apple9+ (M3/M4/M5).
- **Native zero-Python C++ `build-ane` (`src/ane_builder.mm`)**: Builds all 64 single-copy 4-bit `.mlmodelc` MLPs natively in C++20/ARM64 NEON via L1-resident 2-pass row streaming in **`8.39 s`** (`hf4`, `2.0×` faster) or **`7.23 s`** (`nf4`, `2.3×` faster), supporting `--format auto|hf4|lut4|nf4`.
- **Universal 4-bit `lut4` / `nf4` for M1–M5 + native `hf4` for M4/M5**: On M1–M3 (`h13g`/`h14g`/`h15g`), `ANECompiler` silently promotes linear `int4` (`hf4`) to 8-bit (`2×` RAM). Encoding MLP weights via 4-bit `iOS18` palettization (`--format lut4` or `--format nf4`) produces a **true `0.503 B/weight` 4-bit `__KERN` layout that is bit-exact across M1, M2, M3, M4, and M5**:
  - **`--format hf4`** (M4/M5+ default): linear 4-bit, fastest M4 ANE prefill (`707.2 tok/s` exact 4k) and GPU decode (`38.89 ms` M=8 verify).
  - **`--format lut4`** (M1–M3 default): uniform two's-complement 16-level LUT, identical linear values on M1–M5.
  - **`--format nf4`** (M1–M5 universal quality leader): non-uniform 16-level Gaussian NormalFloat4 per-channel LUT (`0.503 B/wt`) that cuts WikiText-2 KL divergence vs exact `g64` BF16 by **`−44.3%`** (`0.0623 → 0.0347`), raises top-1 token agreement **`+3.90 pp`** (`88.92% → 92.82%`, `99.85%` top-5), and runs GPU decode at **`41.6 ms` M=8** (`2.12×` faster via a 1 KiB byte-pair threadgroup LUT `s_lut2[256]`, `338 GB/s`).
- **30-second local compatibility probe**: run `python3 scripts/probe_m_chip.py` on any M1–M5 Mac to check ANE generation, Metal 3.0 shader compilation, `ModelAssetsCache` `MAP_SHARED` permissions, and live single-copy 4-bit verification.

## Verified Apple M4 Max (48 GiB, AC Power) Benchmarks

### v1.3 Head-to-Head vs Splash 1.7.3 (Same Model `mlx-community/Qwen3.8-27B-4bit`, M4 Max 48 GiB, AC Power, KV8, `SPEED-Bench`)

Apples-to-apples comparison on identical [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) weights + [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) drafter (`KV8`, AC power, 46/46 clean v1.3 runs, 0 errors). Full raw dataset published at [`benchmarks/v13_splash_head2head.json`](benchmarks/v13_splash_head2head.json):

| Engine | Ready | Idle / 32k FP | SB-code (`none`) med / mean | SB-code (`med`) med / mean | Transcripts med / mean | Cold PF `2k / 8k / 16k / 32k` | Dec `2k / 8k / 16k / 32k / 32k-m` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `splash_173` (ANE+DFlash2) | 95.75 s* | 18.59 / 23.73 GiB | 130.2 / 128.9 | 119.0 / 118.8 | **112.3 / 103.3** | 506 / 499 / 470 / 424 | **105.8** / 92.6 / 97.7 / 88.6 / **83.2** |
| `mometal v1.2` ANE (MTP) | **0.82 s** | **6.06 / 7.65 GiB** | 123.4 / 115.6 | 99.4 / 97.1 | 93.5 / 87.8 | **676 / 662 / 604 / 519** | 83.0 / 97.0 / **110.2** / 104.1 / 69.0 |
| `mometal v1.2` GPU (MTP) | 6.10 s | 14.93 / 16.52 GiB | 123.6 / 115.6 | 102.0 / 99.9 | 93.1 / 86.4 | 261 / 256 / 247 / 231 | 78.4 / **114.8** / 85.8 / **120.6** / 58.6 |
| **`mometal v1.3` ANE + DFlash2** | **1.22 s (78.5×)** | **6.81 / 8.35 GiB**† | **136.2** / 128.8 | 118.9 / 116.3 | 106.3 / 98.6 | **649 / 658 / 600 / 515 (+21–32%)** | 90.9 / **109.5 (+18%)** / **106.8 (+9%)** / **118.3 (+34%)** / 73.0 |
| **`mometal v1.3` GPU + DFlash2** | **8.14 s (11.8×)** | **15.65 / 17.19 GiB** | **136.8 / 130.7** | **120.5 / 119.0** | 111.7 / 99.9 | 270 / 264 / 255 / 239 | 94.1 / **112.4 (+21%)** / 96.7 / **110.0 (+24%)** / 70.7 |

- **Exact-weight `mometal v1.3` GPU + DFlash2**: Beats Splash 1.7.3 on **both median and mean** of both `SPEED-Bench` coding suites (`SB-code none`: **`136.8 / 130.7`** vs `130.2 / 128.9`, peaking at `163.6 tok/s` on `sb_code_4`; `SB-code medium`: **`120.5 / 119.0`** vs `119.0 / 118.8`) plus `8k` (`+21%`) and `32k` (`+24%`) decode using identical `g64` weights, while using **`−2.9 to −6.5 GiB` less RAM** and starting `11.8×` faster.
- **`mometal v1.3` ANE + DFlash2**: Hits **`136.2 tok/s`** median coding (`sb_code_0 = 151.1`, `sb_code_2 = 151.3`, `sb_code_4 = 154.4`), **`+9% to +34%` faster** `8k/16k/32k` decode, **`+21% to +32%` faster** `2k–32k` cold prefill (`−13.7 s` 32k TTFT), **`1.22 s`** server ready, and **`−3.1 to −6.7 GiB` less total RAM** (`†` even counting `~8.7 GiB` shared ANE `neural` weights alongside `6.81 / 8.35 GiB` process FP).
- **Honest scope notes**: `*` Splash 1.7.3's `95.75 s` cold ready includes `61.4 s` of one-time first-start ANE calibration; Splash also supports `4–8` concurrent serving lanes and GGUF/MoE models, and led `Transcripts` (`112.3 / 103.3` vs `111.7 / 99.9`), `2k` decode (`105.8` vs `94.1`), and `32k-medium` (`83.2` vs `73.0`).

---

### v1.2 Release & v1.1 `n=3` AC-Proof Benchmarks (`Qwen3.8-27B-pi-adapted-MLX-4bit`)

Checkpoint: [`netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit`](https://huggingface.co/netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit) (`harness-hub/Qwen3.8-27B-pi-adapted` RL LoRA merged into `Qwen/Qwen3.8-27B`, MLX 4-bit `g64` + 228 MiB MTP head). Memory uses binary GiB from `footprint -f bytes --wired --swapped` and `vm_stat`.

#### v1.2 Release Highlights (M4 Max 48 GiB, AC Power, 1-copy ANE-Share `hf4` + `KV8`)
- **4-Task Greedy Decode (`T=0.0`, 384 tok each)**: **`98.2 tok/s`** mean (up from `93.8 tok/s`, `+4.7%`) — `py_lru` **`118.9`**, `cpp_refactor` **`113.4`**, `cpp_spsc` **`86.6`**, `prose` **`73.8 tok/s`**; sampled (`T=0.7`) `py_lru` **`121.4 tok/s`**, `prose` `67.9 tok/s`.
- **Short-to-Long Prefill**: `512` **`644.9 tok/s` (`+80.9%`)**, `1k` **`674.4 tok/s` (`+18.9%`)**, `4k` **`658.2 tok/s`** *(exact `4096` `bench`: **`707.2 tok/s`**)*, `128k` **`278.9 tok/s` (`−16.0 s` TTFT)**; pure `bench N=256`: **`414.8 tok/s` (`+19.5%`)**.
- **Long-Context Dense Decode**: **`66.9 tok/s` @ 8k (`+10.0%`)**, **`53.4 tok/s` @ 64k (`+21.6%`)**, **`37.5 tok/s` @ 128k (`+7.4%`)** — **10/10 contexts (`256 → 128k`) green** with 10/10 needle recall and 10/10 64/64 single-copy verification.
- **Memory & Zero-Swap 128k**: Phys FP drops **`−0.94 to −1.70 GiB`** across all contexts (**`6.05 GiB`** FP / `14.62 GiB` wired at `256–1k`, **`5.05 GiB`** FP / `13.64 GiB` wired with `--mtp off`), eliminating swap at `128k` (**`1.17 → 0.00 GiB`**, keeping **`8.31 GiB`** of ANE weights resident).
- **Fast Boot & Native Builder**: Warm model load **`550–629 ms`** (`2.6–2.8×` faster vs `1,516–1,774 ms`), `FULL M=8` verify **`38.89 ms`** (vs `41.31 ms`), and 64-layer C++ `build-ane` in **`8.39 s` (`hf4`) / `7.23 s` (`nf4`)**.

#### Head-to-head (3.7k cold prefill, 4-task greedy `T=0.0` mean, 384 tok each)

Original balanced interleaved `n=3` 3-mode AC-proof dataset (`v1.1`, raw dumps in [`benchmarks/raw_ac_proof/`](benchmarks/raw_ac_proof/)) shown alongside the verified **v1.2** AC-power measurement on the identical 4-task harness (`benchmarks/bench_hn_proof.py`):

| Engine | MLPs / KV | Cold 3.7k PF | Greedy Mean | Active RAM (GiB) | Process Wired | Swapped |
|---|---|---:|---:|---:|---:|---:|
| **`mometal v1.2` (1-copy ANE-Share)** | `hf4` / `KV8` | **646.0 tok/s** *(707.2 @ 4k)* | **98.2 tok/s** | **14.21 GiB** *(6.22 FP)* | **14.08 GiB** | **0.00 GiB** |
| `mometal v1.1` (1-copy ANE-Share, `n=3`) | `hf4` / `KV8` | 621.9 tok/s *(684.7 @ 4k)* | 93.8 tok/s | 16.24 GiB | 15.05 GiB | 0.00 GiB |
| `mometal v1.1` (GPU-only exact `g64`, `n=3`) | `g64` / `KV8` | 271.5 tok/s | 91.6 tok/s *(+35% vs #2)* | 16.41 GiB | 15.21 GiB | 1.10 GiB |
| `mometal v1.1` (2-copy ANE + `g64`, `n=3`) | `hf4`+`g64` / `KV8` | 464.7 tok/s *(629.8 w/o cache)* | 95.2 tok/s | 25.20 GiB | 23.95 GiB | 1.13 GiB |
| `mlx-serve v26.11.3` (Zig/MLX + MTP) | `g64` / `KV8` | 263.0 tok/s | 68.1 tok/s | 16.48 GiB | — | — |
| `oMLX v0.8.3` (Python/MLX + MTP) | `g64` / `BF16` | 260.3 tok/s | 64.0 tok/s | 18.44 GiB | — | — |
| `TensorFold v1.3.4` (Zig/Metal MTP-6) | `g64` / `BF16` | 233.2 tok/s | 57.7 tok/s | 21.88 GiB | — | — |
| `pmetal v0.7.1` (Rust, AR) | `g64` / `BF16` | 253.7 tok/s | 28.5 tok/s | 19.69 GiB | — | — |
| `mlx-lm v0.31.3` (Apple ref, AR) | `g64` / `BF16` | 255.7 tok/s | 27.3 tok/s | 17.35 GiB | — | — |

*(On v1.2, warm 3.7k prefix cache TTFT is `255 ms`, server ready is `1.04 s`, and `--mtp off` drops Phys FP further to **`5.05 GiB`** / `13.64 GiB` wired.)*

#### v1.2 Full 10-Point `256 → 128k` Sweep (`mometal` 1-copy ANE-Share `hf4` + `KV8`, AC power)

Full raw JSON published at [`benchmarks/v12_sweep_256_to_128k.json`](benchmarks/v12_sweep_256_to_128k.json) (`10/10` passkey needle recall, `10/10` 64/64 single-copy ANE-Share, dense exact KV8 decode; deltas vs v1.1 baseline sweep):

| Context | Prompt tok | Prefill tok/s (Δ) | TTFT | Decode tok/s (Δ) | ms/rnd (tok/rnd) | Phys FP | ANE Res. | Wired | Swapped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `256` | 268 | 347.0* | 0.77 s | **71.1** (+3.2%) | 43.9 (3.12) | **6.06 GiB** | 8.70 GiB | **14.62 GiB** | 0.03 GiB |
| `512` | 524 | **644.9** (+80.9%) | **0.81 s** | **84.8** (+5.9%) | 44.4 (3.76) | **6.05 GiB** | 8.70 GiB | **14.62 GiB** | **0.00 GiB** |
| `1k` | 1,034 | **674.4** (+18.9%) | **1.53 s** | **82.0** (+10.7%) | 46.6 (3.82) | **6.06 GiB** | 8.70 GiB | **14.62 GiB** | **0.00 GiB** |
| `2k` | 2,058 | **673.7** (+13.3%) | **3.05 s** | **74.9** (+10.1%) | 45.0 (3.37) | **6.13 GiB** | 8.70 GiB | **14.69 GiB** | **0.00 GiB** |
| `4k` | 4,106 | **658.2** (+5.9%)† | **6.24 s** | **62.5** (+1.1%) | 51.9 (3.24) | **6.20 GiB** | 8.70 GiB | **14.76 GiB** | 0.02 GiB |
| `8k` | 8,202 | **654.5** (+5.5%) | **12.53 s** | **66.9** (+10.0%) | 49.7 (3.32) | **6.33 GiB** | 8.70 GiB | **14.90 GiB** | 0.01 GiB |
| `16k` | 16,396 | **604.7** (+4.1%) | **27.11 s** | **66.4** (+6.9%) | 53.6 (3.56) | **7.09 GiB** | 8.70 GiB | **15.65 GiB** | 0.02 GiB |
| `32k` | 32,780 | **514.0** (+2.4%) | **63.77 s** | **52.8** (+5.4%) | 55.8 (2.94) | **7.64 GiB** | 8.70 GiB | **16.20 GiB** | 0.04 GiB |
| `64k` | 65,548 | **396.1** (+4.2%) | **165.47 s** | **53.4** (+21.6%) | 63.9 (3.41) | **8.74 GiB** | 8.70 GiB | **17.30 GiB** | **0.03 GiB** |
| `128k` | 128,078 | **278.9** (+3.4%) | **459.18 s** | **37.5** (+7.4%) | 71.0 (2.67) | **10.39 GiB** | **8.31 GiB** | **18.55 GiB** | **0.00 GiB** |

- `*` `serve` client `ptok/ttft` at `268` includes 1st-token decode + HTTP framing; pure `mometal bench` prefill is **`414.8 tok/s`** (`N=256`, `+19.5%`) and **`393.2 tok/s`** (`N=268`, `+10.2%`).
- `†` Exact `4096`-token `mometal bench` prefill hits **`707.2 tok/s`**.
- *Opt-in Quest block-sparse decode*: `--sparse-kv-budget 12288` reaches `60.1 / 61.2 / 56.6 tok/s` at `32k / 64k / 128k` (measured on the v1.1 prose summary sweep; recommended `8192` up to `32k` [`KL <= 0.0169`, `100%` Top-5], `12288` at `64k+` [`KL 0.0492`, `100%` Top-5, `3/3` multi-needle; `128k` KL unmeasured]).

#### Wave 3 4-Bit Quality & Native Builder A/B (WikiText-2 `eval-ppl`, 2,048 scored tok, KV8)

| Format | Chips | 64L `build-ane` | PPL (BF16 / KV8) | Mean KL vs `g64` | Median KL | Top-1 Agree | Top-5 Agree | M=8 Verify |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `exact_g64` (GPU-only ref) | M1–M5 | — | **7.787** / — | `0.0000` | `0.0000` | `100.00%` | `100.00%` | — |
| `hf4` (v1.1 baseline) | M4–M5 | ~16.9 s | 8.010 / 8.016 | `0.0623` | `0.0299` | `88.92%` | `99.56%` | `41.31 ms` |
| `hf4` (v1.2 fixed SQ+zp) | M4–M5 | **8.39 s (2.0×)** | 8.271 / 8.249 | `0.0550` (`−11.7%`) | `0.0253` (`−15.3%`) | `89.75%` (`+0.83 pp`) | `99.61%` | **`38.89 ms`** |
| **`nf4` (v1.2 Gaussian LUT4)** | **M1–M5** | **7.23 s (2.3×)** | **8.073 / 8.070** | **`0.0347` (`−44.3%`)** | **`0.0152` (`−49.3%`)** | **`92.82%` (`+3.90 pp`)** | **`99.85%`** | **`41.64 ms`** |

*(On v1.1, moving the 64 MLPs from 4.5 bpw `g64` to 4.0 bpw `hf4` shifted greedy 164-problem HumanEval pass@1 from `92.07% → 88.41%` (`−3.66 pp`). `nf4` recovers nearly half the KL divergence and `+3.90 pp` top-1 agreement at the exact same `0.503 B/weight` single-copy footprint, with `429.6 tok/s` 4k prefill on M4 Max.)*

## Key Architecture Components

- **Native Zero-Disk-Copy DFlash 2 Drafter (`src/dflash.{hpp,mm}`, `shaders/dflash.metal`, `src/qwen38.mm`)**:
  - **On-the-fly 12-thread ARM64 NEON `BF16 → Q4_AFFINE_G64` loader**: Quantizes all 5 DFlash 2 layers + `fc [5120, 25600]` directly from read-only `model.safetensors` into a single `~963 MiB` Q4 Metal arena (`+3 MiB` FP16 norms/base kernels/`hidden_projection`) in **`130–220 ms` warm** with zero disk copy, then immediately `MADV_DONTNEED`s the quantized BF16 source pages.
  - **`12.4 µs` Zero-Metal-Wiring CPU NEON `CandidateSelector`**: Keeps `predecessor_codebook [248320, 256]` and `successor_codebook [248320, 256]` as pure CPU `mmap` BF16 pointers (`0 B` Metal wired RAM), warmed via overlapped `MADV_WILLNEED` + 12-thread PTE pre-faulting + 128B prefetch (`p50 = 12.4 µs`, `max = 19.7 µs`, zero page-fault spikes).
  - **Single-CB Bidirectional Draft + 100%-Hit Pre-Encoded `M=8` Verify**: Runs 5-layer bidirectional `2048`-window GQA-4 attention (`GroupedDynamicCausalConv`, `dflash_attn_bidir_window_{p1,p2}`) + target `lm_head` `M=7` top-16 + `hidden_proj` in **`4.85–5.07 ms`** (`0.38 ms` context commit), overlaps CPU encoding of the `M=8` target verify CB, pairs with a strict verbatim-only PLD hybrid (`match_len >= 12` in-session / `>= 16` cross-turn), and guarantees 100% lossless greedy & sampled Leviathan rejection sampling.
- **Single-Copy ANE Weight Sharing (`src/ane_share.mm`, `shaders/common.metal`)**: Zero-copy `MAP_SHARED` over compiled `model.hwx` binaries + `.hwx_manifest` instant boot verification (`~77–88 ms`) + `KS=4` vectorized `ane_hf4_down_sk4_bf16` (`379 / 362 GB/s`, `38.89 ms` `FULL M=8`), 1 KiB byte-pair threadgroup LUT `s_lut2[256]` `ane_lut4_*` (`2.12×` speedup, `338 GB/s`, `41.6 ms` M=8), and `BM=32` double-buffered `swiglu_tiled32` + `down_tiled32_sk` for short/mid batches.
- **Zero-Python ARM64 NEON Native ANE Builder (`src/ane_builder.mm`)**: Shared `ShardPool` + 2-pass L1-resident row streaming (`30 KB/thread` working set), pure-NEON MLX `g64` unpack, 8-wide FP16+FMLAL MSE clipping, and register-resident binary-tree `nf4` quantization — builds all 64 layers in `7.23–8.39 s` natively.
- **Memory, Boot & Streaming Reliability (`src/qwen38.mm`, `src/vision.mm`, `src/safetensors.mm`, `src/server.cpp`)**:
  - **Lazy Vision Tower (`QwenVisionTower::create_lazy`)**: `0 B` vision RAM and `0 ms` `vision.metal` compilation until the first image request (`−859 MiB` on text-only runs).
  - **`MTLHeapTypePlacement` Tracked Scratch Arena & Non-Aliasing `swiglu_f16_`**: Packs 13 mutually-disjoint private GPU scratch buffers (`225.75 → 83.25 MiB`, `−142.5 MiB` private RAM on every run) and routes `swiglu_f16_` outputs cleanly through standalone `buf_normed_` to eliminate `M > 256` GPU-only in-place aliasing hazards.
  - **Zero-Wiring `CpuMmapQ4Weight` (`--mtp off`) & Un-Prefetched `embed_store_`**: Saves `−682 MiB` wired `IOAccelerator` RAM when `--mtp off` (`5.05 GiB` total FP) and eliminates startup `MADV_WILLNEED` faulting of unused vocab rows while preserving single-CB GPU MTP/DFlash drafting.
  - **Compact `half2` KV Scales & Full-Context 1-Layer MTP `KV4`**: Halves KV scale memory/bandwidth (`−11% to −18%` 128k GQA-6 attention latency) and keeps full-context `KV4` on the 1-layer MTP drafter (~128 MB at 128k).
  - **RFC 3629 UTF-8 SSE Carry Buffer (`src/server.cpp`)**: Buffers incomplete trailing 2/3/4-byte BPE-split UTF-8 sequences across SSE frames and sanitizes invalid/EOF-truncated bytes to `U+FFFD` (zero `UnicodeDecodeError` across chat reasoning, content, and completions streams).
- **Strided-Interleaved Register-Top2 `BitScreen` Draft Head (`shaders/common.metal`)**: Packs 98,304 `lm_head` rows into 1-bit `sign(W)` (`60.0 MiB`) in stride-2048 bucket order, scores 3-bitplane activations via hardware `popcount`, fuses per-simdgroup register top-2 into a `32 KiB` L2 table, and rescores top-256 in exact `Q4_g64` — **`0.16 ms`** for `B=1/2` (`0.17 ms` for `B=3`).
- **`tri_probe` Trees, `cb2_pre` Overlap & Lazy Fused GDN Replay (`src/qwen38.mm`, `shaders/qwen38.metal`)**: `B=3` parallel branch expansion inside `CB #1`, pre-encoded overlapped `CB #2b`, 16 KiB per-tile threadgroup GDN checkpoint stack (`s_stack[4][4][128]`), and lazy fused GDN replay deleting `bak_conv_state_` & `bak_ssm_state_` (`−65.9 MB`). Exact SpecInfer Theorem 1 verification (`src/sampler.cpp`).
- **1-Layer Causal MTP Dead-Compute Elimination & Balanced Async ANE Pipeline (`src/ane_mlp.mm`, `src/qwen38.mm`)**: Skips dead `q_proj`, 17th `O(L²)` full attention, `o_proj`, and 267M MLP during prefill (`6.65×` faster, `−26.0 s` 128k TTFT); queues both 256-token ANE sub-blocks concurrently via CoreML `completionHandler`, balances short-context `qs={1,0}` ANE/GPU overlap (`512: +80.9%`), and absorbs `<=128`-token tails onto GPU in the same 64-layer ANE pass.
- **2-Level Hierarchical FlashAttention-2, Vision Tower & Shared `MICACHE2`**: Bank-conflict-free `sKV[32][264]` FA-2 prefill (`−10 ms` at `pos=3072`), native M4 `simdgroup_bfloat8x8` Qwen3-VL Vision Tower (`shaders/vision.metal`), lazy `KV8`/`KV4` allocator, and shared Dense/Quest SSD prefix caching (`src/prefix_cache.hpp`). Also includes support for Inkling 2 Mini (`src/inkling2.mm`).

## Build & Quickstart

Requires macOS 15+ on Apple Silicon (M1–M5; full benchmark suite verified on M4 Max, macOS 27.0 / `26A428`) and Xcode Command Line Tools + CMake `>= 3.24`.

```bash
# 1. Build (zero runtime dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Optional: check local M1-M5 compatibility (Metal 3.0, aned cache mmap, live 4-bit ANE-Share)
python3 scripts/probe_m_chip.py

# 3. Serve 1-copy ANE-Share + DFlash 2 block-diffusion drafter (136+ tok/s coding, 650+ tok/s prefill!)
./build/mometal serve \
  --model ~/models/Qwen3.8-27B-4bit \
  --dflash ~/models/Qwen3.8-27B-DFlash2 \
  --ane-share \
  --max-seq 131072 --kv-bits 8 \
  --cache-dir ~/.cache/mometal-pi \
  --host 127.0.0.1 --port 8092

# Exact g64 GPU-only + DFlash 2 (136.8 med / 130.7 mean SPEED-Bench coding, zero ANE conversion):
./build/mometal serve \
  --model ~/models/Qwen3.8-27B-4bit \
  --dflash ~/models/Qwen3.8-27B-DFlash2 \
  --ane-models off \
  --max-seq 131072 --kv-bits 8 \
  --host 127.0.0.1 --port 8092

# Serve 1-copy ANE-Share with built-in 1-layer MTP — ZERO Python needed!
#    `--ane-share` automatically builds single-copy 4-bit .mlmodelc MLPs natively in C++
#    into ~/.cache/mometal/ane/ on first launch (`lut4` on M1-M3, `hf4` on M4/M5+),
#    or pre-build explicitly with:
#      ./build/mometal build-ane --model ~/models/qwen38-pi-adapted-mlx4 --format auto|hf4|lut4|nf4
./build/mometal serve \
  --model ~/models/qwen38-pi-adapted-mlx4 \
  --ane-share \
  --max-seq 131072 --kv-bits 8 \
  --cache-dir ~/.cache/mometal-pi \
  --host 127.0.0.1 --port 8092

# Higher-quality M1-M5 universal NF4 1-copy ANE-Share (-44.3% KL, +3.90 pp top-1):
./build/mometal build-ane --model ~/models/qwen38-pi-adapted-mlx4 --format nf4
./build/mometal serve \
  --model ~/models/qwen38-pi-adapted-mlx4 \
  --ane-share --ane-tag nf4 \
  --max-seq 131072 --kv-bits 8 \
  --host 127.0.0.1 --port 8092

# Ultra-low-RAM mode without MTP/DFlash (~5.05 GiB Phys FP via zero-wiring CpuMmapQ4Weight):
# add: --mtp off

# Optional opt-in Quest block-sparse decode (recommended 64k+ budget):
# add: --sparse-kv-budget 12288
```

## Reproducing Benchmarks

```bash
# AC-power interleaved n=3 3-mode verification + 66k live vm_stat paging proof
MOMETAL_MODEL=~/models/qwen38-pi-adapted-mlx4 python3 benchmarks/verify_ac_3mode.py

# WikiText-2 PPL/KL, HumanEval 164, matched-KV controls, and 4k-128k sweep
MOMETAL_MODEL=~/models/qwen38-pi-adapted-mlx4 python3 benchmarks/harden_suite.py
```

## Caveats

- Single-copy ANE-Share depends on Apple's undocumented Mach-O `model.hwx` `__KERN*` layout and readable `/Library/Caches/com.apple.aned/.../ModelAssetsCache`. If a macOS update or unsupported chip alters the binary layout or permissions, the boot-time `weight.bin` verification catches the mismatch, prints a `stderr` diagnostic, unloads ANE completely, and falls back cleanly to GPU-only mode (never silently duplicating weights in 2-copy mode).
- Cross-chip `lut4` / `nf4` binary layouts were verified bit-exact for `h13g` (M1) through `h17g` (M5) via `ANECompiler.framework`; live M1/M2/M3 ANE throughput and macOS 15 cache permissions can be checked locally with `python3 scripts/probe_m_chip.py`.
- Both 4-bit ANE MLP conversion (`mometal build-ane` / auto-build on `--ane-share`) and inference (including `--dflash` on-the-fly NEON quantization) are 100% native C++20/Obj-C++/Metal with zero Python or MLX runtime dependencies. (`scripts/build_ane_mlps.py` remains available for optional calibration-stats SmoothQuant builds.)

## License

Code is licensed under the [MIT License](LICENSE). Benchmark datasets under `benchmarks/` are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
