#pragma once

#include <string>
#include <vector>

namespace mi {

struct AneBuildOptions {
    std::string model_dir;
    std::string out_dir;            // empty -> auto (~/.cache/mometal/ane/<name>-<fmt>)
    std::string format = "auto";    // "auto" (lut4 on M1-M3, hf4 on M4/M5+), "hf4", "lut4", or "nf4"
    std::string tag = "auto";       // "auto" -> resolved format ("hf4", "lut4", or "nf4")
    std::string layers_spec = "all";// "all" or e.g. "0-63" or "0,1,2"
    float alpha = 0.85f;            // Gentle uncalibrated SmoothQuant exponent (clamped to [1/max_sq_scale, max_sq_scale])
    float max_sq_scale = 1.4f;      // Clamp bound for uncalibrated weight-only SmoothQuant (prevents L0 down_proj blowup!)
    bool mse_clip = true;           // Hardware-exact per-row 2-stage coarse-to-fine NEON MSE clipping
    int rows = 256;                 // ANE batch rows per dispatch (256 is M4 SRAM/throughput sweet spot)
    int workers = 0;                // 0 = auto min(12, hardware_concurrency)
};

struct AneBuildResult {
    bool ok = false;
    std::string out_dir;
    std::string format;
    std::string tag;
    int built_layers = 0;
    int cached_layers = 0;
    int total_layers = 0;
    double elapsed_sec = 0.0;
    std::string error;
};

// Native zero-Python C++ CoreML 4-bit ANE MLP builder (NEON-accelerated).
// Reads MLX 4-bit g64 safetensors directly via mmap, applies clamped SmoothQuant + hardware-exact
// MSE clipping (hf4 / lut4 / nf4), and emits ready-to-load .mlmodelc bundles + pieces_inv1_<tag>_L<L>.npy
// targeting iOS18 (macOS 15+, M1-M5 compatible).
AneBuildResult build_ane_mlps_native(const AneBuildOptions& opts);

// Resolve default cache directory for auto-built ANE packs (~/.cache/mometal/ane/<basename>-<fmt>).
std::string default_ane_cache_dir(const std::string& model_dir, const std::string& fmt);

// Resolve "auto" format -> "lut4" on M1/M2/M3 (h13-h15), "hf4" on M4/M5+ (h16+); passes through "hf4"/"lut4"/"nf4".
std::string resolve_ane_build_format(const std::string& format);

} // namespace mi
