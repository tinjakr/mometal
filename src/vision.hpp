#pragma once

#include "metal_ctx.hpp"
#include "model.hpp"
#include "safetensors.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mi {

// Native Zero-Python Qwen 3.5 / 3.8 Vision Tower (27-layer BF16 ViT + 2D RoPE + PatchMerger)
// Uses macOS ImageIO/CoreGraphics for PNG/JPEG/WebP/HEIC decoding + Hardware AMX Metal kernels!
// Supports lazy initialization (init_lazy / create_lazy) so text-only runs incur 0 B vision RAM & 0 ms vision.metal compile!
class QwenVisionTower {
public:
    // Eager constructor (loads weights from an already-mapped SafetensorsStore)
    QwenVisionTower(MetalContext& ctx, const SafetensorsStore& store, const std::string& model_dir);

    // Lazy constructor: call init_lazy(model_dir) to detect vision availability with 0 MB mmap & 0 ms shader compile
    explicit QwenVisionTower(MetalContext& ctx);
    bool init_lazy(const std::string& model_dir);
    static std::unique_ptr<QwenVisionTower> create_lazy(MetalContext& ctx, const std::string& model_dir);

    ~QwenVisionTower() = default;

    QwenVisionTower(const QwenVisionTower&) = delete;
    QwenVisionTower& operator=(const QwenVisionTower&) = delete;

    bool ready() const { return ready_; }
    bool loaded() const { return loaded_; }
    int out_hidden_size() const { return out_hidden_size_; }
    int spatial_merge_size() const { return spatial_merge_size_; }

    // Decode & encode an image from either:
    //   - "data:image/...;base64,..."
    //   - "file:///path/to/img.png" or "/path/to/img.png"
    //   - raw image bytes via encode_bytes()
    // Results are cached in an in-memory LRU feature cache keyed by FNV-1a hash of the raw bytes!
    EncodedImage encode_url(const std::string& url, int max_edge = 1280);
    EncodedImage encode_bytes(const uint8_t* data, size_t len, int max_edge = 1280);

private:
    struct BlockWeights {
        TensorView norm1_w, norm1_b;
        TensorView norm2_w, norm2_b;
        TensorView qkv_w, qkv_b;
        TensorView proj_w, proj_b;
        TensorView fc1_w, fc1_b;
        TensorView fc2_w, fc2_b;
    };

    MetalContext& ctx_;
    bool ready_ = false;
    bool loaded_ = false;
    std::string model_dir_;
    std::string pfx_;
    SafetensorsStore own_store_;
    std::mutex load_mtx_;

    int depth_ = 27;
    int hidden_size_ = 1152;
    int intermediate_size_ = 4304;
    int num_heads_ = 16;
    int head_dim_ = 72;
    int out_hidden_size_ = 5120;
    int patch_size_ = 16;
    int temporal_patch_size_ = 2;
    int spatial_merge_size_ = 2;
    int num_pos_embed_ = 2304;
    int num_grid_per_side_ = 48;
    float eps_ = 1e-6f;

    TensorView patch_w_, patch_b_;
    TensorView pos_embed_w_;
    std::vector<BlockWeights> blocks_;
    TensorView merger_norm_w_, merger_norm_b_;
    TensorView merger_fc1_w_, merger_fc1_b_;
    TensorView merger_fc2_w_, merger_fc2_b_;

    id pso_gemm_ = nil;
    id pso_gemm_gelu_tanh_ = nil;
    id pso_gemm_gelu_exact_ = nil;
    id pso_pos_add_norm_f16_ = nil;
    id pso_norm_f16_ = nil;
    id pso_qkv_rope_ = nil;
    id pso_flash_attn_ = nil;

    // Precomputed u8 -> BF16 LUT + persistent grow-on-demand GPU scratch arena
    uint16_t lut_u8_to_f16_[256] = {};
    int scratch_cap_n_ = 0;
    id b_patches_ = nil;
    id b_pidx_ = nil;
    id b_pwt_ = nil;
    id b_freqs_ = nil;
    id b_hid_ = nil;
    id b_norm_ = nil;
    id b_qkv_ = nil;
    id b_q_ = nil;
    id b_k_ = nil;
    id b_v_ = nil;
    id b_aout_ = nil;
    id b_out_ = nil;

    void parse_vision_config(const std::string& model_dir);
    void init_pipelines_and_weights(const SafetensorsStore& store, const std::string& pfx);
    void ensure_loaded();
    void ensure_scratch(int N);

    // In-memory LRU Vision Feature Cache (up to 16 images -> 0 ms ViT on follow-up turns!)
    static constexpr size_t MAX_CACHED_IMAGES = 16;
    std::list<uint64_t> lru_order_;
    std::unordered_map<uint64_t, std::pair<EncodedImage, std::list<uint64_t>::iterator>> cache_;

    EncodedImage run_vit(const std::vector<uint16_t>& patches_f16, int grid_h, int grid_w, uint64_t hash);
};

} // namespace mi
