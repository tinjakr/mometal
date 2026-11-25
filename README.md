# mometal

From-scratch C++20, Objective-C++, and Metal inference engine for Apple Silicon — featuring **single-copy Apple Neural Engine (ANE) + Metal GPU weight sharing**, a **`0.16 ms` strided-interleaved BitScreen draft head**, **1-layer causal MTP dead-compute elimination**, and **zero Python / zero MLX runtime dependencies**.

Full engineering report & open CC BY 4.0 dataset: **[starmo.ai/reports/mometal-ane-share](https://starmo.ai/reports/mometal-ane-share)**

## Why single-copy ANE weight sharing?

Offloading 27B dense MLP prefill to Apple's Neural Engine more than doubles prompt speed on Apple Silicon, but standard 2-copy execution wires both the 15.4 GiB `.safetensors` weights for Metal GPU decode *and* 8.7 GiB of compiled CoreML `model.hwx` + I/O buffers (`25.20 GiB` active at 4k, `27.40 GiB` at 66k). On a 48 GiB M4 Max on AC power, 2-copy triggers **343.5 GiB of live VM compressor thrashing** during 66k prefill (`−20%`) and faults **9.06 GiB of decompressions** during decode (`−19%`).

**`mometal --ane-share`** parses macOS `26A428`'s Mach-O (`0xbeefface`) `__KERN` 16-chunk asymmetric INT4 weight layout inside `/Library/Caches/com.apple.aned/.../model.hwx`, runs a boot-time spot check against `weight.bin` (32-byte group-0 fingerprint + two 32-byte sampled-group checks per projection; per-layer fallback to 2-copy with `stderr` warning on mismatch), and maps those 64 files read-only via `MAP_SHARED` so Metal GPU decode reads the **exact same 7.99 GiB wired physical pages** as `com.apple.aned` while leaving the 8.93 GiB of duplicate `.safetensors` MLPs unmapped and unwired.

## Verified Apple M4 Max (48 GiB, AC Power) Benchmarks

Checkpoint: [`netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit`](https://huggingface.co/netprojects/Qwen3.8-27B-pi-adapted-MLX-4bit) (`harness-hub/Qwen3.8-27B-pi-adapted` RL LoRA merged into `Qwen/Qwen3.8-27B`, MLX 4-bit `g64` + 228 MiB MTP head). All three `mometal` modes ran in balanced interleaved `n=3` fresh restarts on AC power; memory uses binary GiB from `footprint -f bytes --wired --swapped` and `vm_stat` (raw dumps in [`benchmarks/raw_ac_proof/`](benchmarks/raw_ac_proof/)).

### Head-to-head (3.7k cold prefill, 4-task greedy `T=0.0` mean, 384 tok each)

| Engine | MLPs / KV | Cold 3.7k PF | Greedy Mean | Active RAM (GiB) | Process Wired | Swapped |
|---|---|---:|---:|---:|---:|---:|
| **`mometal` (1-copy ANE-Share)** | `hf4` / `KV8` | **621.9 tok/s** *(684.7 @ 4k)* | **93.8 tok/s** | **16.24 GiB** | **15.05 GiB** | **0.00 GiB** |
| **`mometal` (GPU-only exact `g64`)** | `g64` / `KV8` | 271.5 tok/s | **91.6 tok/s** *(+35% vs #2)* | 16.41 GiB | 15.21 GiB | 1.10 GiB |
| **`mometal` (2-copy ANE + `g64`)** | `hf4`+`g64` / `KV8` | 464.7 tok/s *(629.8 w/o cache)* | 95.2 tok/s | 25.20 GiB | 23.95 GiB | 1.13 GiB |
| `mlx-serve v26.11.3` (Zig/MLX + MTP) | `g64` / `KV8` | 263.0 tok/s | 68.1 tok/s | 16.48 GiB | — | — |
| `oMLX v0.8.3` (Python/MLX + MTP) | `g64` / `BF16` | 260.3 tok/s | 64.0 tok/s | 18.44 GiB | — | — |
| `TensorFold v1.3.4` (Zig/Metal MTP-6) | `g64` / `BF16` | 233.2 tok/s | 57.7 tok/s | 21.88 GiB | — | — |
| `pmetal v0.7.1` (Rust, AR) | `g64` / `BF16` | 253.7 tok/s | 28.5 tok/s | 19.69 GiB | — | — |
| `mlx-lm v0.31.3` (Apple ref, AR) | `g64` / `BF16` | 255.7 tok/s | 27.3 tok/s | 17.35 GiB | — | — |

### Long-context sweep (`mometal` 1-copy ANE-Share `KV8`, byte-exact RAM, prose summary task)

| Context (`kv_cap`) | Prompt | Prefill (TTFT) | Dense Decode | Quest-12k Decode* | Top Passkey | Total GiB |
|---|---:|---:|---:|---:|---:|---:|
| 4k (`8,192`) | 4,077 | **684.7 tok/s** (5.95 s) | **65.2 tok/s** | — | 1/1 | 16.24 |
| 16k (`32,768`) | 16,279 | **600.5 tok/s** (27.11 s) | **49.3 tok/s** | — | 1/1 | 17.14 |
| 32k (`65,536`) | 32,579 | **510.4 tok/s** (63.83 s) | **54.0 tok/s** | **60.1 tok/s** | 1/1 | 18.75 |
| 64k (`65,536`) | 65,079 | **390.4 tok/s** (166.68 s) | **44.4 tok/s** | **61.2 tok/s** | 1/1 | 18.74 |
| 128k (`131,072`) | 126,581 | **273.2 tok/s** (463.25 s) | **39.2 tok/s** | **56.6 tok/s** | 1/1 | 21.18 |

*Quality tradeoffs:* Moving the 64 MLPs from 4.5 bpw `g64` to 4.0 bpw `hf4` shifts WikiText-2 PPL from `7.785 → 8.017` and greedy 164-problem HumanEval pass@1 from **`92.07% → 88.41%`** (`−3.66 pp`). Opt-in Quest `8192` is recommended up to `32k` (`KL <= 0.0169`, `100%` Top-5); at `64k+` use `--sparse-kv-budget 12288` (`KL 0.0492`, `100%` Top-5, `3/3` multi-needle; `128k` KL unmeasured).

## Key Architecture Components

- **Single-Copy ANE Weight Sharing (`src/ane_share.mm`, `shaders/common.metal`)**: Zero-copy `MAP_SHARED` over compiled `model.hwx` binaries + adjacent-slot `KS=8` `TN=2/4` FP16 `simdgroup_matrix` kernels (`360.2 GB/s` across 64 mmap'd `.hwx` files) + dedicated `M=1` `ane_hf4` QMV (`33.1 ms` / `29.2 tok/s` AR).
- **Strided-Interleaved Register-Top2 `BitScreen` Draft Head (`shaders/common.metal`)**: Packs 98,304 `lm_head` rows into 1-bit `sign(W)` (`60.0 MiB`) in stride-2048 bucket order, scores 3-bitplane activations via hardware `popcount`, fuses per-simdgroup register top-2 into a `32 KiB` L2 table (zero DRAM score buffer), and rescores top-256 in exact `Q4_g64` — **`0.16 ms`** for `B=1/2` (`0.17 ms` for `B=3`).
- **`tri_probe` Trees & Lazy Fused GDN Replay (`src/qwen38.mm`, `shaders/qwen38.metal`)**: `B=3` parallel branch expansion inside `CB #1`, zero-wave `M=8` topologies, 16 KiB per-tile threadgroup GDN checkpoint stack (`s_stack[4][4][128]`), and lazy fused GDN replay deleting `bak_conv_state_` & `bak_ssm_state_` (`−65.9 MB`, offsetting `BitScreen`'s `+63.7 MB` for `−2.2 MB` net RAM). Exact SpecInfer Theorem 1 verification (`src/sampler.cpp`).
- **1-Layer Causal MTP Dead-Compute Elimination (`encode_mtp_kv_prime`)**: Skips dead `q_proj`, 17th `O(L²)` full attention, `o_proj`, and 267M MLP during prefill (`32.31 → 4.86 ms` at `M=512`, `6.65×` faster, cutting 128k TTFT by `−26.0 s`); runs `M=1` root-sync attention on row `M−1` only during decode.
- **Async 2-Deep ANE + Tail-Absorption Co-Execution (`src/ane_mlp.mm`, `src/qwen38.mm`)**: Queues both 256-token ANE sub-blocks concurrently via CoreML `completionHandler` and absorbs `<=128`-token tails onto the GPU in the same 64-layer ANE pass (`M=1088: 362 → 595 tok/s`).
- **2-Level Hierarchical FlashAttention-2, Vision Tower & Shared `MICACHE2`**: Bank-conflict-free `sKV[32][264]` FA-2 prefill, native M4 `simdgroup_bfloat8x8` Qwen3-VL Vision Tower (`shaders/vision.metal`), lazy `KV8`/`KV4` allocator, and shared Dense/Quest SSD prefix caching (`src/prefix_cache.hpp`). Also includes support for Inkling 2 Mini (`src/inkling2.mm`).

## Build & Quickstart

Requires macOS on Apple Silicon (tested on M4 Max, macOS 27.0 / `26A428`) and Xcode Command Line Tools + CMake `>= 3.24`.

```bash
# 1. Build (zero runtime dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Optional: one-time offline CoreML hf4 MLP conversion (~4 min, requires Python + coremltools + mlx)
python3 scripts/build_ane_mlps.py \
  --model ~/models/qwen38-pi-adapted-mlx4 \
  --out   ~/models/qwen38-pi-adapted-mlx4-ane \
  --bits 4 --qmode asym --alpha 0.25 --mse-clip \
  --layers 0-63 --workers 4

# 3. Serve 1-copy ANE-Share at 128k KV8 (OpenAI-compatible API)
./build/mometal serve \
  --model ~/models/qwen38-pi-adapted-mlx4 \
  --ane-tag hf4 --ane-share \
  --max-seq 131072 --kv-bits 8 \
  --cache-dir ~/.cache/mometal-pi \
  --host 127.0.0.1 --port 8092

# Exact g64 GPU-only mode (no ANE conversion needed, 91.6 tok/s greedy decode):
./build/mometal serve \
  --model ~/models/qwen38-pi-adapted-mlx4 \
  --ane-models none \
  --max-seq 131072 --kv-bits 8 \
  --host 127.0.0.1 --port 8092

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

- `--ane-share` depends on macOS `26A428`'s undocumented Mach-O `model.hwx` `__KERN` layout. If a macOS update alters the binary layout, the boot-time `weight.bin` spot check catches the mismatch, prints a `stderr` warning, and falls back per-layer to 2-copy mode.
- The 4-bit ANE conversion (`scripts/build_ane_mlps.py`) runs offline once in Python; inference itself is 100% native C++20/Obj-C++/Metal.

## License

Code is licensed under the [MIT License](LICENSE). Benchmark datasets under `benchmarks/` are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
