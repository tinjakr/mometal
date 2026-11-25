#pragma once

#include "metal_ctx.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mi {

struct AneShareLayer {
    bool ready = false;
    id hwx_buf = nil;          // Zero-copy MTLBuffer over compiled model.hwx (shared physical pages with ANE!)
    void* mmap_ptr = nullptr;
    size_t mapped_size = 0;
    id buf_scales_offs = nil;  // float2[2*INTER + 2*n_subblocks*HID] = {scale, 1032.0f + offset} (0.0f for nf4)
    id buf_folded_norm = nil;  // bf16[HID] = post_attn_norm * inv1 (or nil if inv1 == 1.0)
    id buf_lut16 = nil;        // shared half[16 * 16] (512 B): verified 16-entry fp16 LUT per linear li=0..15
    bool is_lut4 = false;      // true for uniform lut4 or non-uniform nf4_lut4_a8
    bool is_nonuniform_lut = false; // true when LUT != [0..7, -8..-1] (e.g. NF4)
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t sub_size = 0;
    uint32_t dk_size = 0;
    uint32_t n_subblocks = 4u;
    uint32_t lin_base_w[16] = {};
    uint32_t lin_chunk_w[16] = {};
    uint32_t lin_hdr_w[16] = {}; // per-linear group header words: 6 (24B asym), 4 (16B sym/lut4/nf4), 0 (unfused)
};

// Native C++/Obj-C++ single-copy ANE 4-bit weight sharer (supports M4+ hf4, universal M1-M5 lut4, and NF4 lut4).
// Uses O(1) .hwx_manifest cache -> own-process cache dir -> full ModelAssetsCache fallback, always verifying
// every MLP linear tensor bit-exact against .mlmodelc/weights/weight.bin (+ bytes[0..31] LUT check for NF4),
// and mmaps them read-only into zero-copy Metal buffers sharing the exact physical RAM pages as ANE.
class AneShareManager {
public:
    AneShareManager(MetalContext& ctx, const std::string& ane_models_dir,
                    const std::string& tag, int num_layers,
                    uint32_t hidden_size = 5120u, uint32_t intermediate_size = 17408u);
    ~AneShareManager();

    AneShareManager(const AneShareManager&) = delete;
    AneShareManager& operator=(const AneShareManager&) = delete;

    bool has_layer(int L) const {
        return L >= 0 && L < static_cast<int>(layers_.size()) && layers_[L].ready;
    }
    const AneShareLayer& layer(int L) const { return layers_[L]; }
    AneShareLayer& layer_mut(int L) { return layers_[L]; }

    int shared_count() const { return shared_count_; }
    double load_time_ms() const { return load_time_ms_; }
    const std::string& format_name() const { return format_name_; }
    bool is_lut4() const { return is_lut4_; }
    bool is_nonuniform_lut() const { return is_nonuniform_lut_; }

private:
    std::vector<AneShareLayer> layers_;
    int shared_count_ = 0;
    double load_time_ms_ = 0.0;
    std::string format_name_;
    bool is_lut4_ = false;
    bool is_nonuniform_lut_ = false;
};

} // namespace mi
