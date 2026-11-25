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
    id buf_scales_offs = nil;  // float2[17408 + 17408 + 8*5120] = {scale, 136.0f + offset}
    id buf_folded_norm = nil;  // bf16[5120] = post_attn_norm * inv1 (or nil if inv1 == 1.0)
    uint32_t lin_base_w[16] = {};
    uint32_t lin_chunk_w[16] = {};
    uint32_t hdr_words = 6u;   // 6 (24B asymmetric int4) or 4 (16B symmetric int4)
};

// Native C++/Obj-C++ single-copy ANE hf4 weight sharer.
// Locates compiled model.hwx files in /Library/Caches/com.apple.aned, verifies all 16
// MLP linear tensors per layer bit-exact against .mlmodelc/weights/weight.bin, and mmaps
// them read-only into zero-copy Metal buffers sharing the exact physical RAM pages as ANE.
class AneShareManager {
public:
    AneShareManager(MetalContext& ctx, const std::string& ane_models_dir,
                    const std::string& tag, int num_layers, int share_count);
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

private:
    std::vector<AneShareLayer> layers_;
    int shared_count_ = 0;
    double load_time_ms_ = 0.0;
};

} // namespace mi
