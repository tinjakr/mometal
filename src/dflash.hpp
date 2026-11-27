#pragma once

#include "metal_ctx.hpp"
#include "safetensors.hpp"
#include "sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mi {

struct DFlashConfig {
    int num_layers = 5;
    int hidden_size = 5120;
    int intermediate_size = 17408;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int vocab_size = 248320;
    int sliding_window = 2048;
    float rope_theta = 10000000.0f;
    float rms_eps = 1e-6f;

    int block_size = 8;          // 1 anchor + (block_size - 1) MASK tokens
    int mask_token_id = 248070;
    int conv_group_size = 16;
    int conv_kernel_size = 2;
    int selector_rank = 256;
    int selector_top_k = 16;
    std::vector<int> target_layer_ids; // 0-indexed target layer IDs, e.g. {5, 19, 33, 47, 61}
};

// MLX Q4_AFFINE_G64 weight layout — 100% natively compatible with common.metal Q4_g64 kernels:
//   w: DType::U32  [out_dim, in_dim / 8]  (8 x 4-bit unsigned nibbles per uint32)
//   s: DType::BF16 [out_dim, in_dim / 64] (uint16 BF16 per-group scales)
//   b: DType::BF16 [out_dim, in_dim / 64] (uint16 BF16 per-group biases)
struct DFlashQ4Weight {
    TensorView w;
    TensorView s;
    TensorView b;
    uint32_t out_dim = 0;
    uint32_t in_dim = 0;
};

struct DFlashLayerWeights {
    // IEEE FP16 (half, DType::F16) Metal buffers
    TensorView input_layernorm;          // F16 [hidden_size]
    TensorView post_attention_layernorm; // F16 [hidden_size]
    TensorView q_norm;                   // F16 [head_dim]
    TensorView k_norm;                   // F16 [head_dim]
    TensorView attn_base_kernel;         // F16 [2, 2, hidden_size] (stage, tap, ch)
    TensorView mlp_base_kernel;          // F16 [2, 2, hidden_size] (stage, tap, ch)

    // On-the-fly NEON-quantized MLX Q4_AFFINE_G64 Metal buffers (BF16 scales & biases)
    DFlashQ4Weight attn_kernel_proj;     // Q4_g64 [2 * conv_kernel_size * (H / conv_group_size), H] = [1280, 5120]
    DFlashQ4Weight mlp_kernel_proj;      // Q4_g64 [1280, 5120]
    DFlashQ4Weight q_proj;               // Q4_g64 [num_heads * head_dim, H] = [4096, 5120]
    DFlashQ4Weight k_proj;               // Q4_g64 [num_kv_heads * head_dim, H] = [1024, 5120]
    DFlashQ4Weight v_proj;               // Q4_g64 [num_kv_heads * head_dim, H] = [1024, 5120]
    DFlashQ4Weight o_proj;               // Q4_g64 [H, num_heads * head_dim] = [5120, 4096]
    DFlashQ4Weight gate_proj;            // Q4_g64 [intermediate_size, H]
    DFlashQ4Weight up_proj;              // Q4_g64 [intermediate_size, H]
    DFlashQ4Weight down_proj;            // Q4_g64 [H, intermediate_size]
};

class DFlashEngine {
public:
    DFlashEngine(MetalContext& ctx, const std::string& dflash_dir);
    ~DFlashEngine();

    DFlashEngine(const DFlashEngine&) = delete;
    DFlashEngine& operator=(const DFlashEngine&) = delete;

    bool ready() const { return ready_; }
    const DFlashConfig& config() const { return cfg_; }

    // Global weights
    const DFlashQ4Weight& fc() const { return fc_; }                              // Q4_g64 [H, num_target * H]
    const TensorView& hidden_norm() const { return hidden_norm_; }                // F16 [H]
    const TensorView& norm() const { return norm_; }                              // F16 [H]
    const TensorView& selector_hidden_proj() const { return sel_hidden_proj_; }   // F16 [selector_rank, H]
    const DFlashLayerWeights& layer(int i) const { return layers_[static_cast<size_t>(i)]; }

    // Returns 0..(target_layer_ids.size()-1) if target_layer_idx is captured, else -1
    int target_capture_slot(int target_layer_idx) const;

    size_t gpu_weight_bytes() const { return gpu_weight_bytes_; }
    double load_time_ms() const { return load_time_ms_; }
    uint64_t fingerprint() const { return fingerprint_; }

    // CPU ARM64 NEON CandidateSelector (<20 us, 0 B Metal wired RAM for codebooks!).
    //   n_pos = config().block_size - 1 (7 proposal positions), topk = config().selector_top_k (16).
    //   topk_cands: flat [n_pos * topk] LogitCand{val = unary_logit, id = token_id}.
    //   sel_h_f16:  flat [n_pos * selector_rank] IEEE FP16 (uint16_t) from dflash_hidden_proj_f16.
    //   out_draft_ids: [n_pos] selected draft token IDs.
    void select_greedy(int anchor_token,
                       const LogitCand* topk_cands,
                       const uint16_t* sel_h_f16,
                       int* out_draft_ids) const;

    // Sampled CandidateSelector (T > 0):
    //   Samples n_pos draft tokens autoregressively via top_k/top_p-filtered, 0.85x-sharpened
    //   softmax(score / (0.85 * temperature)) over each row's candidates, writing out_draft_ids[0..n_pos-1]
    //   and (if non-null) filling out_draft_dists[0..n_pos-1] with exact q(x) probabilities ready for
    //   Sampler::sample_or_accept_draft_dist().
    void select_sampled(int anchor_token,
                        const LogitCand* topk_cands,
                        const uint16_t* sel_h_f16,
                        float temperature,
                        uint64_t seed,
                        int* out_draft_ids,
                        DraftDist* out_draft_dists = nullptr,
                        int top_k = 0,
                        float top_p = 1.0f) const;

    // Optional FP32 sel_h overloads in case caller outputs float32 from hidden_proj
    void select_greedy_f32(int anchor_token,
                           const LogitCand* topk_cands,
                           const float* sel_h_f32,
                           int* out_draft_ids) const;
    void select_sampled_f32(int anchor_token,
                            const LogitCand* topk_cands,
                            const float* sel_h_f32,
                            float temperature,
                            uint64_t seed,
                            int* out_draft_ids,
                            DraftDist* out_draft_dists = nullptr,
                            int top_k = 0,
                            float top_p = 1.0f) const;

private:
    bool ready_ = false;
    DFlashConfig cfg_;
    std::vector<int> target_slot_lut_; // [max_target_layer + 1] -> capture slot or -1

    // Backing Metal arenas:
    //   - Native MLX Q4 checkpoint: q4_sub_bufs_ (zero-copy MAP_SHARED sub-ranges over mmap_ptr_, 0 B dirty Q4!)
    //   - BF16 checkpoint w/ disk cache: buf_q4_arena_ (zero-copy MAP_SHARED over q4_mmap_ptr_, 0 B dirty Q4!)
    //   - Fallback (unwritable cache dir): buf_q4_arena_ (in-memory alloc_shared)
    id buf_q4_arena_ = nil;
    std::vector<id> q4_sub_bufs_;
    id buf_f16_arena_ = nil;
    size_t gpu_weight_bytes_ = 0;
    double load_time_ms_ = 0.0;

    DFlashQ4Weight fc_;
    TensorView hidden_norm_;
    TensorView norm_;
    TensorView sel_hidden_proj_;
    std::vector<DFlashLayerWeights> layers_;

    // Read-only CPU mmap of model.safetensors (page-aligned length for zero-copy Metal sub-range binding).
    // BF16 projection ranges are MADV_DONTNEED'd when using cached/in-memory Q4; codebooks stay 0 B Metal wired!
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;      // page-aligned mmap length
    size_t mmap_file_size_ = 0; // exact file st.st_size
    // Optional read-only MAP_SHARED mmap of disk-cached 256B-aligned Q4 arena (~/.cache/mometal/dflash/*.q4bin)
    void* q4_mmap_ptr_ = nullptr;
    size_t q4_mmap_size_ = 0;
    const uint8_t* pred_cb_bytes_ = nullptr; // [vocab_size, selector_rank] BF16 (unaligned-safe byte ptr)
    const uint8_t* succ_cb_bytes_ = nullptr; // [vocab_size, selector_rank] BF16 (unaligned-safe byte ptr)
    uint64_t fingerprint_ = 0;
};

} // namespace mi
