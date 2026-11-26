#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38.hpp"
#include "ane_builder.hpp"
#include "json.hpp"
#include "prefix_cache.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mi {

namespace {

struct QmmParams { uint32_t M, N, K, add_residual; };
struct QmmSkParams { uint32_t M, N, K, add_residual, sk; };
struct QmmSkNormParams { uint32_t M, N, K, add_residual, sk; float eps; };
struct EmbedParams { uint32_t M, D; float eps; };
struct NormParams { uint32_t M, D; float eps; };
struct GdnConvParams { uint32_t M, C; };
struct GdnPrepParams { uint32_t M, Hk, Dk, Hv, C; float eps; };
struct GdnRecParams { uint32_t M, Hk, Hv, Dk, Dv, C; };
struct GdnNormParams { uint32_t M, Hv, Dv; float eps; };
struct FullAttnParams {
    uint32_t M, Hq, Hkv, D, rotary_dim, pos, kv_len;
    float rope_theta, eps, scale;
    uint32_t num_splits;
    int rope_offset;
    uint32_t win_sink = 0u;
    uint32_t win_recent = 0u;
};
struct QuestParams {
    uint32_t kv_len, Hq, Hkv, M, start_page, end_page, total_pages, sink_pages;
    uint32_t recent_start_page, budget_pages, max_active_pages, score_stride;
};
struct ScatterEmbedParams { uint32_t count, D; };
struct MtpConcatParams { uint32_t M, D; float eps; };
struct AnePackParams { uint32_t M, block_rows, D, PW, has_inv1; float eps; };
struct AneAddParams { uint32_t M, D, num_pieces; };
struct AneAddNormParams { uint32_t M, D, num_pieces; float eps; };
struct VerifyCompactParams { uint32_t V, n_parts, max_cands; float margin; };
struct DraftSampleParams { uint32_t V, n_parts; float inv_temp, margin; uint32_t seed_lo, seed_hi, top_k; float top_p; };
struct DraftBatchParams {
    uint32_t B, V, n_parts;
    float    inv_temp, margin;
    uint32_t seed_lo, seed_hi, top_k;
    float    top_p;
    uint32_t greedy;
    uint32_t out_tok_idx[8];
    int32_t  out_alt0_idx[8];
    int32_t  out_alt1_idx[8];
    uint32_t out_dist_idx[8];
};
struct MtpGatherParams {
    uint32_t B, D;
    float    eps;
    uint32_t pad;
    uint32_t in_tok_idx[8];
    uint32_t in_hid_idx[8];
    uint32_t out_hid_idx[8];
};
struct PathReplayParams { uint32_t len, pre_pos, Hkv, D, kv_bits, pad0, pad1, pad2; uint32_t path[16]; };
struct DFlashNormParams { uint32_t M, D; float eps; };
struct DFlashConvParams { uint32_t M, D, group_size, stage; };
struct DFlashKvWriteParams { uint32_t M, Hkv, D, pos_start, window; float rope_theta, eps; };
struct DFlashQkvPrepParams { uint32_t M, Hq, Hkv, D, pos_start; float rope_theta, eps; };
struct DFlashAttnParams { uint32_t M, Hq, Hkv, D, pos_start, ctx_len, window, num_splits; float scale; };
struct DFlashTopKParams { uint32_t M, V, K, num_chunks; };
struct DFlashCaptureParams { uint32_t M, D, num_targets, slot_idx, src_row_off, dst_row_off; };
struct DFlashHiddenProjParams { uint32_t M, R, D; };
struct AneShareParams {
    uint32_t M;
    uint32_t HID;
    uint32_t INTER;
    uint32_t SUB;
    uint32_t DK;
    uint32_t n_subblocks;
    uint32_t lin_base_w[16];
    uint32_t lin_chunk_w[16];
    uint32_t lin_hdr_w[16];
};

static inline AneShareParams make_ane_share_params(const AneShareLayer& sl, uint32_t M) {
    AneShareParams ap{};
    ap.M = M;
    ap.HID = sl.hidden_size;
    ap.INTER = sl.intermediate_size;
    ap.SUB = sl.sub_size;
    ap.DK = sl.dk_size;
    ap.n_subblocks = sl.n_subblocks;
    std::memcpy(ap.lin_base_w, sl.lin_base_w, sizeof(ap.lin_base_w));
    std::memcpy(ap.lin_chunk_w, sl.lin_chunk_w, sizeof(ap.lin_chunk_w));
    std::memcpy(ap.lin_hdr_w, sl.lin_hdr_w, sizeof(ap.lin_hdr_w));
    return ap;
}

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

} // namespace

Qwen38Model::Q4Weight Qwen38Model::load_q4(const SafetensorsStore& st, const std::string& prefix) {
    Q4Weight q;
    q.w = st.get(prefix + ".weight");
    q.s = st.get(prefix + ".scales");
    q.b = st.get(prefix + ".biases");
    q.out_dim = static_cast<uint32_t>(q.w.shape[0]);
    q.in_dim = static_cast<uint32_t>(q.w.shape[1] * 8);
    return q;
}

Qwen38Model::Qwen38Model(MetalContext& ctx, const std::string& model_dir, int max_seq_len,
                         const std::string& mtp_dir, int mtp_depth, bool enable_pld,
                         const std::string& ane_models_dir, const std::string& ane_tag,
                         int kv_bits, int ane_share_layers, int sparse_kv_budget,
                         const std::string& dflash_dir)
    : ctx_(ctx), max_seq_len_(max_seq_len), kv_bits_(kv_bits), sparse_kv_budget_(sparse_kv_budget),
      pld_enabled_(enable_pld),
      mtp_depth_(std::clamp(mtp_depth, 1, MAX_VERIFY - 1)),
      cur_mtp_depth_(std::clamp(mtp_depth, 1, MAX_VERIFY - 1)) {
    const bool want_dflash = !dflash_dir.empty() && dflash_dir != "off" && dflash_dir != "none";
    if (kv_bits_ != 16 && kv_bits_ != 8 && kv_bits_ != 4) {
        throw std::runtime_error("Unsupported --kv-bits value (expected 16, 8, or 4): " + std::to_string(kv_bits_));
    }
    auto t0 = std::chrono::steady_clock::now();

    std::string shader_dir = MOMETAL_SHADER_DIR;
    ctx_.load_shader_file("common", shader_dir + "/common.metal");
    ctx_.load_shader_file("qwen38", shader_dir + "/qwen38.metal");

    pso_embed_           = ctx_.pipeline("common", "embed_q4_g64_f32");
    pso_qmm_q4_          = ctx_.pipeline("common", "qmm_q4_g64_f32");
    pso_mma_qmm8_        = ctx_.pipeline("common", "mma_qmm8_q4_g64_f32");
    pso_mma_qmm8_f16x_   = ctx_.pipeline("common", "mma_qmm8_q4_g64_f16x");
    pso_mma_qmm8_f16x_sk_ = ctx_.pipeline("common", "mma_qmm8_q4_g64_f16x_sk");
    pso_mma_sk_reduce_   = ctx_.pipeline("common", "mma_qmm8_sk_reduce");
    pso_mma_sk_reduce_norm_f16_ = ctx_.pipeline("common", "mma_qmm8_sk_reduce_rmsnorm_f16");
    pso_mma_sk_reduce_dual_norm_f16_ = ctx_.pipeline("common", "mma_qmm8_sk_reduce_dual_norm_f16");
    pso_mma_lmhead_max_f16x_ = ctx_.pipeline("common", "mma_qmm8_lmhead_max_f16x");
    pso_mma_lmhead_max_f16x_indexed_ = ctx_.pipeline("common", "mma_qmm8_lmhead_max_f16x_indexed");
    pso_qmv_q4_f16x_indexed_max_ = ctx_.pipeline("common", "qmv_q4_g64_f16x_indexed_max");
    pso_mma_qmm8_swiglu_ = ctx_.pipeline("common", "mma_qmm8_q4_g64_swiglu_f32");
    pso_mma_qmm8_swiglu_f16_ = ctx_.pipeline("common", "mma_qmm8_q4_g64_swiglu_f16");
    pso_mma_qmm16_       = ctx_.pipeline("common", "mma_qmm16_q4_g64_f32");
    pso_qmm_q4_tiled_    = ctx_.pipeline("common", "qmm_q4_g64_tiled");
    pso_qmm_q4_tiled_f16x_  = ctx_.pipeline("common", "qmm_q4_g64_tiled_f16x");
    pso_qmm_q4_smalln_f16x_ = ctx_.pipeline("common", "qmm_q4_g64_smalln_f16x");
    pso_mma_in_proj_ab_f16x_   = ctx_.pipeline("common", "mma_qmm8_in_proj_ab_f16x");
    pso_qmv_q4_swiglu_   = ctx_.pipeline("common", "qmv_q4_g64_swiglu_f32");
    pso_qmv_q4_f16x_     = ctx_.pipeline("common", "qmv_q4_g64_f16x");
    pso_qmv_q4_f16x_sk_  = ctx_.pipeline("common", "qmv_q4_g64_f16x_sk");
    pso_qmv_q4_swiglu_f16x_ = ctx_.pipeline("common", "qmv_q4_g64_swiglu_f16x");
    pso_mtp_embed_concat_f16_ = ctx_.pipeline("qwen38", "qwen_mtp_embed_concat_f16");
    pso_rmsnorm_         = ctx_.pipeline("common", "rmsnorm_f32_bf16w");
    pso_rmsnorm_f16_     = ctx_.pipeline("common", "rmsnorm_f32_bf16w_to_f16");
    pso_rmsnorm_dual_    = ctx_.pipeline("common", "rmsnorm_f32_bf16w_dual");
    pso_swiglu_          = ctx_.pipeline("common", "swiglu_f32");
    pso_swiglu_f16_      = ctx_.pipeline("common", "swiglu_f32_to_f16");
    pso_copy_            = ctx_.pipeline("common", "vec_copy_f32");
    pso_argmax1_         = ctx_.pipeline("common", "argmax_stage1");
    pso_argmax2_         = ctx_.pipeline("common", "argmax_stage2");
    pso_logit_max2_      = ctx_.pipeline("common", "logit_max_stage2");
    pso_logit_compact_   = ctx_.pipeline("common", "logit_compact_scan");
    pso_mtp_sample_draft_ = ctx_.pipeline("common", "mtp_sample_draft_gpu");
    pso_mtp_sample_draft_batch_ = ctx_.pipeline("common", "mtp_sample_draft_batch_gpu");
    pso_mtp_embed_concat_gather_f16_ = ctx_.pipeline("qwen38", "qwen_mtp_embed_concat_gather_f16");
    pso_rmsnorm_dual_scatter_f16_ = ctx_.pipeline("qwen38", "qwen_rmsnorm_dual_scatter_f16");

    pso_gdn_conv_     = ctx_.pipeline("qwen38", "qwen_gdn_conv1d_silu");
    pso_gdn_conv_pf_  = ctx_.pipeline("qwen38", "qwen_gdn_conv1d_silu_prefill");
    pso_gdn_conv_upd_ = ctx_.pipeline("qwen38", "qwen_gdn_conv_state_update");
    pso_gdn_qknorm_   = ctx_.pipeline("qwen38", "qwen_gdn_qknorm");
    pso_gdn_gates_    = ctx_.pipeline("qwen38", "qwen_gdn_gates");
    pso_gdn_prework_fused_ = ctx_.pipeline("qwen38", "qwen_gdn_prework_fused");
    pso_gdn_prework_tree_  = ctx_.pipeline("qwen38", "qwen_gdn_prework_tree");
    pso_gdn_prework_replay_path_ = ctx_.pipeline("qwen38", "qwen_gdn_prework_replay_path");
    pso_gdn_prework_lazy_  = ctx_.pipeline("qwen38", "qwen_gdn_prework_lazy");
    pso_gdn_rec_      = ctx_.pipeline("qwen38", "qwen_gdn_recurrence");
    pso_gdn_rec_tree_ = ctx_.pipeline("qwen38", "qwen_gdn_recurrence_tree");
    pso_gdn_rec_lazy_ = ctx_.pipeline("qwen38", "qwen_gdn_recurrence_lazy");
    pso_gdn_rec_tree_lazy_ = ctx_.pipeline("qwen38", "qwen_gdn_recurrence_tree_lazy");
    pso_gdn_rec_tree_lazy_s1_ = ctx_.pipeline("qwen38", "qwen_gdn_recurrence_tree_lazy_s1");
    pso_gdn_rec_tree_lazy_s2_ = ctx_.pipeline("qwen38", "qwen_gdn_recurrence_tree_lazy_s2");
    pso_gdn_norm_     = ctx_.pipeline("qwen38", "qwen_gdn_rmsnorm_gated");
    pso_gdn_norm_f16_ = ctx_.pipeline("qwen38", "qwen_gdn_rmsnorm_gated_f16");
    pso_fa_q_prep_        = ctx_.pipeline("qwen38", "qwen_full_attn_q_prep");
    pso_fa_q_prep_hier_   = ctx_.pipeline("qwen38", "qwen_full_attn_q_prep_hier");
    pso_fa_q_prep_tree_   = ctx_.pipeline("qwen38", "qwen_full_attn_q_prep_tree");
    pso_fa_kv_prep_       = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep");
    pso_fa_kv_prep_tree_  = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_tree");
    pso_fa_q_prep_mrope_  = ctx_.pipeline("qwen38", "qwen_full_attn_q_prep_mrope");
    pso_fa_q_prep_mrope_hier_ = ctx_.pipeline("qwen38", "qwen_full_attn_q_prep_mrope_hier");
    pso_fa_kv_prep_mrope_ = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_mrope");
    pso_scatter_embeds_   = ctx_.pipeline("qwen38", "qwen_scatter_embeds");
    pso_fa_attn_          = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated");
    pso_fa_attn_f16_      = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_f16");
    pso_fa_attn_tree_f16_ = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_tree_f16");
    pso_fa_prefill_flash_      = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash");
    pso_fa_prefill_flash_f16_  = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_f16");
    pso_fa_prefill_flash_pair_f16_ = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_pair_f16");
    pso_fa_prefill_hier_f16_   = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_hier_f16_f16");
    pso_pack_q_pf_h_           = ctx_.pipeline("qwen38", "qwen_pack_q_pf_h");
    pso_fa_split1_        = ctx_.pipeline("qwen38", "qwen_gqa_decode_splitk_p1");
    pso_fa_split2_        = ctx_.pipeline("qwen38", "qwen_gqa_decode_splitk_p2");
    pso_fa_vsplit1_       = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1");
    pso_fa_vsplit1_tree_  = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1_tree");
    pso_fa_vsplit2_       = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p2");
    pso_fa_vsplit2_f16_   = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p2_f16");
    pso_fa_gqa6_m1_       = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_f16");
    pso_fa_gqa6_m1_tree_  = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_f16_tree");
    pso_fa_gqa6_m2_       = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_f16");
    pso_fa_gqa6_m2_tree_  = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_f16_tree");
    pso_fa_gqa6_m4_       = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_f16");
    pso_fa_gqa6_m4_tree_  = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_f16_tree");
    pso_fa_gqa6_v_        = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_f16");
    pso_fa_gqa6_v_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_f16_tree");
    pso_pack_q_h_         = ctx_.pipeline("qwen38", "qwen_pack_q_h");
    pso_pack_q_p2_        = ctx_.pipeline("qwen38", "qwen_pack_q_p2");
    pso_pack_q_p4_        = ctx_.pipeline("qwen38", "qwen_pack_q_p4");
    pso_kv_compact_f16_   = ctx_.pipeline("qwen38", "qwen_kv_compact_f16");
    pso_kv_compact_q_     = ctx_.pipeline("qwen38", "qwen_kv_compact_q");
    pso_copy_hid_path_    = ctx_.pipeline("qwen38", "qwen_copy_hid_path");
    if (kv_bits_ < 16) {
        std::string qs = (kv_bits_ == 8) ? "q8" : "q4";
        pso_fa_kv_prep_q_        = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_" + qs);
        pso_fa_kv_prep_q_tree_   = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_" + qs + "_tree");
        pso_fa_kv_prep_mrope_q_  = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_" + qs + "_mrope");
        pso_fa_attn_q_           = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_" + qs);
        pso_fa_attn_q_f16_       = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_" + qs + "_f16");
        pso_fa_attn_q_tree_f16_  = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_" + qs + "_tree_f16");
        pso_fa_prefill_flash_q_     = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_" + qs);
        pso_fa_prefill_flash_q_f16_ = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_" + qs + "_f16");
        pso_fa_prefill_flash_pair_q_f16_ = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_pair_" + qs + "_f16");
        pso_fa_prefill_hier_q_f16_  = ctx_.pipeline("qwen38", "qwen_gqa_prefill_flash_hier_" + qs + "_f16");
        pso_fa_split1_q_         = ctx_.pipeline("qwen38", "qwen_gqa_decode_splitk_p1_" + qs);
        pso_fa_vsplit1_q_        = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1_" + qs);
        pso_fa_vsplit1_q_tree_   = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1_" + qs + "_tree");
        pso_fa_gqa6_m1_q_        = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + qs);
        pso_fa_gqa6_m1_q_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + qs + "_tree");
        pso_fa_gqa6_m2_q_        = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + qs);
        pso_fa_gqa6_m2_q_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + qs + "_tree");
        pso_fa_gqa6_m4_q_        = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + qs);
        pso_fa_gqa6_m4_q_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + qs + "_tree");
        pso_fa_gqa6_v_q_         = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + qs);
        pso_fa_gqa6_v_q_tree_    = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + qs + "_tree");
    }
    if (sparse_kv_budget_ > 0) {
        std::string qs = (kv_bits_ == 16) ? "f16" : ((kv_bits_ == 8) ? "q8" : "q4");
        pso_quest_build_bounds_ = ctx_.pipeline("qwen38", "qwen_quest_build_bounds_" + qs);
        pso_quest_score_pages_  = ctx_.pipeline("qwen38", "qwen_quest_score_pages");
        pso_quest_select_topk_  = ctx_.pipeline("qwen38", "qwen_quest_select_topk");
        pso_fa_gqa6_m1_sp_      = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + qs + "_sparse");
        pso_fa_gqa6_m1_sp_tree_ = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + qs + "_tree_sparse");
        pso_fa_gqa6_m2_sp_      = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + qs + "_sparse");
        pso_fa_gqa6_m2_sp_tree_ = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + qs + "_tree_sparse");
        pso_fa_gqa6_m4_sp_      = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + qs + "_sparse");
        pso_fa_gqa6_m4_sp_tree_ = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + qs + "_tree_sparse");
        pso_fa_gqa6_v_sp_       = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + qs + "_sparse");
        pso_fa_gqa6_v_sp_tree_  = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + qs + "_tree_sparse");
    }
    pso_mtp_concat_       = ctx_.pipeline("qwen38", "qwen_mtp_concat_norm");
    pso_ane_pack_         = ctx_.pipeline("qwen38", "qwen_rmsnorm_ane_pack");
    pso_ane_add_          = ctx_.pipeline("qwen38", "qwen_ane_add_output");
    pso_ane_add_norm_     = ctx_.pipeline("qwen38", "qwen_ane_add_and_rmsnorm");
    pso_ane_add_norm_f16_ = ctx_.pipeline("qwen38", "qwen_ane_add_and_rmsnorm_f16");
    // Note: ane_hf4/lut4_* and bitscreen_{build,quant_x,score_*} PSOs are specialized via function constants
    // after reading config.json and initializing AneShareManager below!
    pso_bs_select_cands_      = ctx_.pipeline("common", "bitscreen_select_cands");
    pso_bs_rescore_q4_        = ctx_.pipeline("common", "bitscreen_rescore_q4");

    Json cfg = parse_json(read_file_to_string(model_dir + "/config.json"));
    const Json& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;

    hidden_size_        = static_cast<int>(tc["hidden_size"].as_int(5120));
    intermediate_size_  = static_cast<int>(tc["intermediate_size"].as_int(17408));
    num_layers_         = static_cast<int>(tc["num_hidden_layers"].as_int(64));
    vocab_size_         = static_cast<int>(tc["vocab_size"].as_int(248320));
    full_attn_interval_ = static_cast<int>(tc["full_attention_interval"].as_int(4));
    num_heads_          = static_cast<int>(tc["num_attention_heads"].as_int(24));
    num_kv_heads_       = static_cast<int>(tc["num_key_value_heads"].as_int(4));
    head_dim_           = static_cast<int>(tc["head_dim"].as_int(256));
    rms_eps_            = static_cast<float>(tc["rms_norm_eps"].as_double(1e-6));
    lin_num_k_heads_    = static_cast<int>(tc["linear_num_key_heads"].as_int(16));
    lin_num_v_heads_    = static_cast<int>(tc["linear_num_value_heads"].as_int(48));
    lin_head_k_dim_     = static_cast<int>(tc["linear_key_head_dim"].as_int(128));
    lin_head_v_dim_     = static_cast<int>(tc["linear_value_head_dim"].as_int(128));
    lin_conv_dim_       = 2 * lin_num_k_heads_ * lin_head_k_dim_ + lin_num_v_heads_ * lin_head_v_dim_;

    double partial_rot = tc["partial_rotary_factor"].as_double(0.25);
    if (tc.contains("rope_parameters") && tc["rope_parameters"].is_object()) {
        const auto& rp = tc["rope_parameters"];
        if (rp.contains("partial_rotary_factor")) partial_rot = rp["partial_rotary_factor"].as_double(0.25);
        if (rp.contains("rope_theta")) rope_theta_ = static_cast<float>(rp["rope_theta"].as_double(10000000.0));
    }
    rotary_dim_ = static_cast<int>(std::round(head_dim_ * partial_rot));
    image_token_id_ = static_cast<int>(cfg["image_token_id"].as_int(248056));

    // Hardware-adaptive ANE prefill offload threshold (base M1/M2/M3/M4 -> 64, Pro -> 96-128, Max/Ultra -> 176)
    const int est_gpu_cores = ctx_.estimated_gpu_cores();
    ane_min_ = (est_gpu_cores <= 12) ? 64 : ((est_gpu_cores <= 20) ? 96 : ((est_gpu_cores <= 32) ? 128 : 176));

    // Specialize BitScreen kernels via FC_BS_D (index 10 = hidden_size_) for any hidden_size % 64 == 0, <= 8192!
    if (hidden_size_ > 0 && (hidden_size_ % 64) == 0 && hidden_size_ <= 8192) {
        const std::vector<std::pair<uint32_t, uint32_t>> bs_fc = {{10u, static_cast<uint32_t>(hidden_size_)}};
        const std::string bs_sfx = "_d" + std::to_string(hidden_size_);
        pso_lmh_build_bitscreen_ = ctx_.pipeline_u32_constants("common", "lmh_build_bitscreen", "lmh_build_bitscreen" + bs_sfx, bs_fc);
        pso_bs_quant_x_          = ctx_.pipeline_u32_constants("common", "bitscreen_quant_x",   "bitscreen_quant_x" + bs_sfx,   bs_fc);
        pso_bs_score_b1_         = ctx_.pipeline_u32_constants("common", "bitscreen_score_b1",  "bitscreen_score_b1" + bs_sfx,  bs_fc);
        pso_bs_score_b2_         = ctx_.pipeline_u32_constants("common", "bitscreen_score_b2",  "bitscreen_score_b2" + bs_sfx,  bs_fc);
        pso_bs_score_b3_         = ctx_.pipeline_u32_constants("common", "bitscreen_score_b3",  "bitscreen_score_b3" + bs_sfx,  bs_fc);
    }

    // Single-copy ANE-Share MLP accelerator (1-copy ONLY; falls back cleanly to GPU-only on any failure!)
    // Initialize BEFORE store_.load_dir() so shared MLP .safetensors pages are NEVER faulted or GPU-pinned.
    std::string resolved_ane = ane_models_dir;
    std::string auto_fmt = resolve_ane_build_format("auto");
    std::string effective_tag = ane_tag;
    if (resolved_ane == "off" || resolved_ane == "none") {
        resolved_ane.clear();
    } else if (access("/Library/Caches/com.apple.aned", R_OK | X_OK) != 0) {
        const int err = errno;
        if (ane_share_layers > 0 || resolved_ane != "auto") {
            std::fprintf(stderr,
                         "[ane-share] Cannot read /Library/Caches/com.apple.aned (%s); falling back to GPU-only.\n"
                         "[ane-share] Single-copy ANE-Share currently requires SIP disabled + read ACL:\n"
                         "[ane-share]   sudo chmod -R +a \"user:$USER allow list,search,readattr,readextattr,readsecurity,file_inherit,directory_inherit\" /Library/Caches/com.apple.aned\n"
                         "[ane-share] (GPU-only + DFlash 2 works out of the box with SIP enabled.)\n",
                         std::strerror(err));
        }
        resolved_ane.clear();
    } else if (resolved_ane == "auto") {
        std::string c1 = model_dir + "/ane-models";
        std::string c2 = model_dir + "-ane";
        std::string c3 = default_ane_cache_dir(model_dir, auto_fmt);
        if (std::filesystem::is_directory(c1)) resolved_ane = c1;
        else if (std::filesystem::is_directory(c2)) resolved_ane = c2;
        else if (std::filesystem::is_directory(c3)) resolved_ane = c3;
        else if (ane_share_layers > 0) {
            // Zero-Python on-demand native C++ ANE MLP build into ~/.cache/mometal/ane/!
            AneBuildOptions bopt;
            bopt.model_dir = model_dir;
            bopt.out_dir = c3;
            bopt.format = (ane_tag == "lut4" || ane_tag == "hf4" || ane_tag == "nf4") ? ane_tag : "auto";
            bopt.tag = (ane_tag == "auto" || ane_tag.empty()) ? "auto" : ane_tag;
            auto bres = build_ane_mlps_native(bopt);
            if (bres.ok) {
                resolved_ane = bres.out_dir;
                effective_tag = bres.tag;
            } else {
                std::fprintf(stderr, "[ane-build] Auto-build failed (%s); falling back to GPU-only\n", bres.error.c_str());
                resolved_ane.clear();
            }
        } else {
            resolved_ane.clear();
        }
    } else if (ane_share_layers > 0) {
        // Explicit --ane-models <dir> + --ane-share: auto-build missing layers natively if dir doesn't have L0 yet
        std::string want_fmt = (ane_tag == "lut4" || ane_tag == "hf4" || ane_tag == "nf4") ? ane_tag : auto_fmt;
        std::string want_tag = (ane_tag == "auto" || ane_tag.empty()) ? want_fmt : ane_tag;
        if (!std::filesystem::is_directory(resolved_ane + "/L0_mlp_p0_" + want_tag + ".mlmodelc")) {
            AneBuildOptions bopt;
            bopt.model_dir = model_dir;
            bopt.out_dir = resolved_ane;
            bopt.format = want_fmt;
            bopt.tag = want_tag;
            auto bres = build_ane_mlps_native(bopt);
            if (bres.ok) {
                effective_tag = bres.tag;
            } else {
                std::fprintf(stderr, "[ane-build] Build into %s failed (%s); falling back to GPU-only\n",
                             resolved_ane.c_str(), bres.error.c_str());
                resolved_ane.clear();
            }
        }
    }
    // Always skip Vision weights in main LLM store (Vision tower lazy-loads on first image -> -859 MiB RAM on text-only!)
    // Also skip embed_tokens from main prefetched store when untied: loaded via zero-wiring CpuMmapQ4Weight when !mtp_loaded_
    // (-682 MiB wired RAM!) or dedicated un-prefetched embed_store_ when mtp_loaded_ (zero startup fault, keeps single-CB MTP!).
    const bool tie_word_embeddings = tc.contains("tie_word_embeddings")
        ? tc["tie_word_embeddings"].as_bool(false)
        : (cfg.contains("tie_word_embeddings") ? cfg["tie_word_embeddings"].as_bool(false) : false);
    std::vector<std::string> skip_prefixes = {
        "vision_tower.",
        "model.visual.",
        "visual."
    };
    if (!tie_word_embeddings) {
        skip_prefixes.push_back("language_model.model.embed_tokens.");
        skip_prefixes.push_back("model.language_model.embed_tokens.");
        skip_prefixes.push_back("model.embed_tokens.");
    }
    if (want_dflash || mtp_dir == "off" || mtp_dir == "none") {
        skip_prefixes.push_back("mtp.");
    }
    if (!resolved_ane.empty()) {
        try {
            // Defer ANE warmup so AneShareManager verifies existing model.hwx BEFORE warmup launches (eliminates cold-cache race),
            // then async warmup overlaps with safetensors loading & BitScreen build!
            ane_ = std::make_unique<AneMlpEngine>(ctx_, resolved_ane, num_layers_, hidden_size_, effective_tag, /*defer_warmup=*/true);
            if (ane_ && ane_->ready()) {
                ane_share_ = std::make_unique<AneShareManager>(
                    ctx_, resolved_ane, ane_->tag(), num_layers_,
                    static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_));
                if (ane_share_->shared_count() < num_layers_) {
                    // Cold first-ever build writes model.hwx during CoreML warmup; run warmup to completion once and retry
                    ane_->start_warmup_async();
                    ane_->wait_warmup();
                    ane_share_ = std::make_unique<AneShareManager>(
                        ctx_, resolved_ane, ane_->tag(), num_layers_,
                        static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_));
                } else {
                    ane_->start_warmup_async();
                }
                if (ane_share_->shared_count() == num_layers_) {
                    ane_share_nonuniform_lut_ = ane_share_->is_nonuniform_lut();
                    const AneShareLayer& sl0 = ane_share_->layer(0);
                    uint32_t hdr0 = sl0.lin_hdr_w[0];
                    bool uniform_hdr = true;
                    for (int li_layer = 0; li_layer < num_layers_ && uniform_hdr; ++li_layer) {
                        const AneShareLayer& sl = ane_share_->layer(li_layer);
                        uint32_t n_lin = sl.n_subblocks * 4u;
                        for (uint32_t li = 0; li < n_lin; ++li) {
                            if (sl.lin_hdr_w[li] != hdr0) { uniform_hdr = false; break; }
                        }
                    }
                    std::vector<std::pair<uint32_t, uint32_t>> ane_fc = {
                        {0u, sl0.hidden_size},
                        {1u, sl0.intermediate_size},
                        {2u, sl0.sub_size},
                        {3u, sl0.dk_size},
                        {4u, sl0.n_subblocks}
                    };
                    std::string ane_sfx = "_h" + std::to_string(sl0.hidden_size)
                                        + "_i" + std::to_string(sl0.intermediate_size)
                                        + "_s" + std::to_string(sl0.sub_size)
                                        + "_n" + std::to_string(sl0.n_subblocks);
                    if (uniform_hdr) {
                        ane_fc.push_back({5u, hdr0});
                        ane_sfx += "_w" + std::to_string(hdr0);
                    }
                    if (ane_share_nonuniform_lut_) {
                        pso_ane_hf4_swiglu_bf16_     = ctx_.pipeline_u32_constants("common", "ane_lut4_swiglu_bf16",     "ane_lut4_swiglu_bf16"     + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk4_bf16_   = ctx_.pipeline_u32_constants("common", "ane_lut4_down_sk4_bf16",   "ane_lut4_down_sk4_bf16"   + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk_bf16_    = pso_ane_hf4_down_sk4_bf16_;
                        pso_ane_hf4_swiglu_m4_       = ctx_.pipeline_u32_constants("common", "ane_lut4_swiglu_m4",       "ane_lut4_swiglu_m4"       + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk_m4_      = ctx_.pipeline_u32_constants("common", "ane_lut4_down_sk_m4",      "ane_lut4_down_sk_m4"      + ane_sfx, ane_fc);
                        pso_ane_hf4_swiglu_m1_       = pso_ane_hf4_swiglu_m4_;
                        pso_ane_hf4_down_m1_sk_      = pso_ane_hf4_down_sk_m4_;
                        pso_ane_hf4_tiled_gateup_    = ctx_.pipeline_u32_constants("common", "ane_lut4_tiled_gateup",    "ane_lut4_tiled_gateup"    + ane_sfx, ane_fc);
                        pso_ane_hf4_tiled_down_      = ctx_.pipeline_u32_constants("common", "ane_lut4_tiled_down",      "ane_lut4_tiled_down"      + ane_sfx, ane_fc);
                        pso_ane_hf4_swiglu_tiled32_  = ctx_.pipeline_u32_constants("common", "ane_lut4_swiglu_tiled32",  "ane_lut4_swiglu_tiled32"  + ane_sfx, ane_fc);
                        pso_ane_hf4_down_tiled32_sk_ = ctx_.pipeline_u32_constants("common", "ane_lut4_down_tiled32_sk", "ane_lut4_down_tiled32_sk" + ane_sfx, ane_fc);
                    } else {
                        pso_ane_hf4_swiglu_bf16_     = ctx_.pipeline_u32_constants("common", "ane_hf4_swiglu_bf16",      "ane_hf4_swiglu_bf16"      + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk_bf16_    = ctx_.pipeline_u32_constants("common", "ane_hf4_down_sk_bf16",     "ane_hf4_down_sk_bf16"     + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk4_bf16_   = ctx_.pipeline_u32_constants("common", "ane_hf4_down_sk4_bf16",    "ane_hf4_down_sk4_bf16"    + ane_sfx, ane_fc);
                        pso_ane_hf4_swiglu_m1_       = ctx_.pipeline("common", "ane_hf4_swiglu_m1");
                        pso_ane_hf4_down_m1_sk_      = ctx_.pipeline("common", "ane_hf4_down_m1_sk");
                        pso_ane_hf4_swiglu_m4_       = ctx_.pipeline_u32_constants("common", "ane_hf4_swiglu_m4",        "ane_hf4_swiglu_m4"        + ane_sfx, ane_fc);
                        pso_ane_hf4_down_sk_m4_      = ctx_.pipeline_u32_constants("common", "ane_hf4_down_sk_m4",       "ane_hf4_down_sk_m4"       + ane_sfx, ane_fc);
                        pso_ane_hf4_tiled_gateup_    = ctx_.pipeline_u32_constants("common", "ane_hf4_tiled_gateup",     "ane_hf4_tiled_gateup"     + ane_sfx, ane_fc);
                        pso_ane_hf4_tiled_down_      = ctx_.pipeline_u32_constants("common", "ane_hf4_tiled_down",       "ane_hf4_tiled_down"       + ane_sfx, ane_fc);
                        pso_ane_hf4_swiglu_tiled32_  = ctx_.pipeline_u32_constants("common", "ane_hf4_swiglu_tiled32",   "ane_hf4_swiglu_tiled32"   + ane_sfx, ane_fc);
                        pso_ane_hf4_down_tiled32_sk_ = ctx_.pipeline_u32_constants("common", "ane_hf4_down_tiled32_sk",  "ane_hf4_down_tiled32_sk"  + ane_sfx, ane_fc);
                    }
                    for (int i = 0; i < num_layers_; ++i) {
                        std::string ls = "layers." + std::to_string(i) + ".mlp.";
                        skip_prefixes.push_back("model.language_model." + ls);
                        skip_prefixes.push_back("language_model.model." + ls);
                        skip_prefixes.push_back("model." + ls);
                    }
                } else {
                    std::fprintf(stderr,
                                 "[ane-share] Only %d/%d layers verified single-copy; "
                                 "unloading ANE and falling back to GPU-only (no 2-copy duplication)\n",
                                 ane_share_->shared_count(), num_layers_);
                    ane_share_.reset();
                    ane_.reset();
                }
            } else {
                ane_.reset();
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ane-share] ANE init failed (%s); falling back to GPU-only\n", e.what());
            ane_share_.reset();
            ane_.reset();
        }
    }

    store_.load_dir(ctx_.device(), model_dir, skip_prefixes);

    std::string lm_pfx = store_.has("language_model.model.norm.weight")
        ? "language_model."
        : (store_.has("model.language_model.norm.weight") ? "model.language_model." : "");
    std::string model_pfx = (lm_pfx == "language_model.") ? "language_model.model."
                          : (lm_pfx == "model.language_model.") ? "model.language_model." : "model.";

    final_norm_ = store_.get(model_pfx + "norm.weight");
    std::string head_key = store_.has("language_model.lm_head.weight") ? "language_model.lm_head"
                         : (store_.has("lm_head.weight") ? "lm_head" : "");
    if (!head_key.empty()) {
        lm_head_ = load_q4(store_, head_key);
        if (tie_word_embeddings && store_.has(model_pfx + "embed_tokens.weight")) {
            embed_tokens_ = load_q4(store_, model_pfx + "embed_tokens");
        }
    } else {
        // Weight-tied model fallback (no separate lm_head.weight): map embed_tokens in Metal and share with lm_head!
        if (store_.has(model_pfx + "embed_tokens.weight")) {
            embed_tokens_ = load_q4(store_, model_pfx + "embed_tokens");
        } else {
            embed_store_.load_dir_only(ctx_.device(), model_dir, {model_pfx + "embed_tokens."}, /*prefetch_now=*/true);
            embed_tokens_ = load_q4(embed_store_, model_pfx + "embed_tokens");
        }
        lm_head_ = embed_tokens_;
    }

    layers_.resize(num_layers_);
    states_.resize(num_layers_);
    kv_cap_ = std::min(max_seq_len_, 2048);
    size_t kv_bytes = 0;

    for (int i = 0; i < num_layers_; ++i) {
        std::string lp = model_pfx + "layers." + std::to_string(i) + ".";
        LayerWeights& lw = layers_[i];
        lw.is_linear = ((i + 1) % full_attn_interval_) != 0;
        lw.input_norm     = store_.get(lp + "input_layernorm.weight");
        lw.post_attn_norm = store_.get(lp + "post_attention_layernorm.weight");

        if (lw.is_linear) {
            lw.in_proj_qkv = load_q4(store_, lp + "linear_attn.in_proj_qkv");
            lw.in_proj_z   = load_q4(store_, lp + "linear_attn.in_proj_z");
            lw.in_proj_b   = load_q4(store_, lp + "linear_attn.in_proj_b");
            lw.in_proj_a   = load_q4(store_, lp + "linear_attn.in_proj_a");
            lw.out_proj    = load_q4(store_, lp + "linear_attn.out_proj");
            lw.conv1d_w    = store_.get(lp + "linear_attn.conv1d.weight");
            lw.A_log       = store_.get(lp + "linear_attn.A_log");
            lw.dt_bias     = store_.get(lp + "linear_attn.dt_bias");
            lw.gdn_norm    = store_.get(lp + "linear_attn.norm.weight");

            size_t cs_bytes = 3ull * lin_conv_dim_ * sizeof(float);
            size_t ss_bytes = static_cast<size_t>(lin_num_v_heads_) * lin_head_v_dim_ * lin_head_k_dim_ * sizeof(uint16_t);
            states_[i].conv_state = ctx_.alloc_private(cs_bytes);
            states_[i].ssm_state  = ctx_.alloc_private(ss_bytes);
            kv_bytes += cs_bytes + ss_bytes;
        } else {
            lw.q_proj = load_q4(store_, lp + "self_attn.q_proj");
            lw.k_proj = load_q4(store_, lp + "self_attn.k_proj");
            lw.v_proj = load_q4(store_, lp + "self_attn.v_proj");
            lw.o_proj = load_q4(store_, lp + "self_attn.o_proj");
            lw.q_norm = store_.get(lp + "self_attn.q_norm.weight");
            lw.k_norm = store_.get(lp + "self_attn.k_norm.weight");

            size_t elem_div = (kv_bits_ == 16) ? 1ull : ((kv_bits_ == 8) ? 2ull : 4ull);
            size_t kv_sz = (static_cast<size_t>(kv_cap_) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / elem_div;
            states_[i].k_cache = ctx_.alloc_private(kv_sz);
            states_[i].v_cache = ctx_.alloc_private(kv_sz);
            kv_bytes += 2 * kv_sz;
            if (kv_bits_ < 16) {
                size_t sb_sz = static_cast<size_t>(kv_cap_) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t);
                states_[i].k_sb = ctx_.alloc_private(sb_sz);
                states_[i].v_sb = ctx_.alloc_private(sb_sz);
                kv_bytes += 2 * sb_sz;
            }
            if (sparse_kv_budget_ > 0) {
                size_t kb_sz = static_cast<size_t>(ceil_div(static_cast<uint32_t>(kv_cap_), QUEST_PAGE_SIZE)) * num_kv_heads_ * head_dim_ * 4ull;
                states_[i].k_bounds = ctx_.alloc_private(kb_sz);
                states_[i].quest_built_pages = 0;
                kv_bytes += kb_sz;
            }
        }

        if (ane_share_ && ane_share_->has_layer(i)) {
            lw.ane_share_idx = i;
            id inv1_buf = ane_->inv1_buffer(i);
            if (inv1_buf != nil && lw.post_attn_norm.host_ptr != nullptr) {
                const uint16_t* nw = reinterpret_cast<const uint16_t*>(lw.post_attn_norm.host_ptr);
                const float* iv = reinterpret_cast<const float*>([(id<MTLBuffer>)inv1_buf contents]);
                id<MTLBuffer> fbuf = ctx_.alloc_shared(static_cast<size_t>(hidden_size_) * sizeof(uint16_t));
                uint16_t* dst = reinterpret_cast<uint16_t*>([fbuf contents]);
                for (int j = 0; j < hidden_size_; ++j) {
                    uint32_t u = static_cast<uint32_t>(nw[j]) << 16;
                    float w = 0.0f;
                    std::memcpy(&w, &u, 4);
                    float v = w * iv[j];
                    uint32_t vu = 0;
                    std::memcpy(&vu, &v, 4);
                    uint32_t r = vu + 0x7FFFu + ((vu >> 16) & 1u);
                    dst[j] = static_cast<uint16_t>(r >> 16);
                }
                ane_share_->layer_mut(i).buf_folded_norm = fbuf;
                lw.post_attn_norm_share = TensorView{fbuf, 0, static_cast<size_t>(hidden_size_) * 2u,
                                                     reinterpret_cast<const uint8_t*>(dst), DType::BF16,
                                                     {hidden_size_}};
            } else {
                lw.post_attn_norm_share = lw.post_attn_norm;
            }
        } else {
            lw.gate_proj = load_q4(store_, lp + "mlp.gate_proj");
            lw.up_proj   = load_q4(store_, lp + "mlp.up_proj");
            lw.down_proj = load_q4(store_, lp + "mlp.down_proj");
        }
    }

    // Persistent scratch sizing (computed before MTP scratch so MTP buffers are allocated ONCE at C_attn!)
    // When ANE is active, attention & MTP scratch supports up to 512+64 = 576-tok halves (1152-tok co-exec chunks!)
    // while MLP scratch stays at chunk_size_ (512) to keep peak RAM lean!
    size_t C_mlp  = static_cast<size_t>(chunk_size_);
    size_t C_attn = (ane_ && ane_->ready())
        ? static_cast<size_t>(AneMlpEngine::MAX_SUBS * ane_->block_rows() + 64)
        : C_mlp;
    c_attn_ = C_attn;

    // Optional DFlash 2 block-diffusion drafter (replaces 1-layer MTP & BitScreen -> saves ~288 MiB!)
    if (want_dflash) {
        dflash_ = std::make_unique<DFlashEngine>(ctx_, dflash_dir);
        const DFlashConfig& dcfg = dflash_->config();
        if (!dflash_->ready()) {
            throw std::runtime_error("Failed to initialize DFlash 2 engine from " + dflash_dir);
        }
        if (dcfg.hidden_size != hidden_size_ || dcfg.vocab_size != vocab_size_) {
            throw std::runtime_error("DFlash 2 hidden_size/vocab_size mismatch with target Qwen3.8 model");
        }
        if (dcfg.block_size < 2 || dcfg.block_size > MAX_VERIFY) {
            throw std::runtime_error("Unsupported DFlash 2 block_size (expected 2.." + std::to_string(MAX_VERIFY) + ")");
        }

        ctx_.load_shader_file("dflash", shader_dir + "/dflash.metal");
        pso_df_rmsnorm_f16_     = ctx_.pipeline("dflash", "dflash_rmsnorm_f16");
        pso_df_conv_prep_f16_   = ctx_.pipeline("dflash", "dflash_conv_prepare_f16");
        pso_df_conv_finish_f32_ = ctx_.pipeline("dflash", "dflash_conv_finish_add_f32");
        pso_df_kv_write_        = ctx_.pipeline("dflash", "dflash_context_kv_write");
        pso_df_qkv_prep_        = ctx_.pipeline("dflash", "dflash_qkv_prep");
        pso_df_attn_p1_         = ctx_.pipeline("dflash", "dflash_attn_bidir_window_p1");
        pso_df_attn_p2_         = ctx_.pipeline("dflash", "dflash_attn_bidir_window_p2");
        pso_df_topk16_p1_       = ctx_.pipeline("dflash", "dflash_lmhead_topk16_p1");
        pso_df_topk16_p2_       = ctx_.pipeline("dflash", "dflash_lmhead_topk16_p2");
        pso_df_capture_hidden_  = ctx_.pipeline("dflash", "dflash_capture_hidden");
        pso_df_hidden_proj_f16_ = ctx_.pipeline("dflash", "dflash_hidden_proj_f16");

        // 5-layer FP16 sliding-window ring KV cache ([Hkv, window, D] per layer, ~40 MiB total!)
        df_k_cache_.resize(static_cast<size_t>(dcfg.num_layers), nil);
        df_v_cache_.resize(static_cast<size_t>(dcfg.num_layers), nil);
        const size_t ring_sz = static_cast<size_t>(dcfg.num_kv_heads) * dcfg.sliding_window * dcfg.head_dim * sizeof(uint16_t);
        for (int l = 0; l < dcfg.num_layers; ++l) {
            df_k_cache_[l] = ctx_.alloc_private(ring_sz);
            df_v_cache_[l] = ctx_.alloc_private(ring_sz);
            kv_bytes += 2 * ring_sz;
        }

        // Compact target-hidden capture buffer (per-chunk, NOT 2048 rows!) + block-8 draft scratch (~0.75 MiB)
        const size_t n_tgt = dcfg.target_layer_ids.size();
        const size_t B = static_cast<size_t>(dcfg.block_size);
        const size_t P = B - 1u;
        const size_t H = static_cast<size_t>(dcfg.hidden_size);
        const size_t G_dyn = 2u * static_cast<size_t>(dcfg.conv_kernel_size) * (H / static_cast<size_t>(dcfg.conv_group_size));
        const size_t num_chunks = ceil_div(static_cast<uint32_t>(vocab_size_), 2048u);

        df_cap_rows_      = (ane_ && ane_->ready()) ? (2 * C_attn) : C_attn;
        buf_df_capture_   = ctx_.alloc_private(df_cap_rows_ * n_tgt * H * sizeof(uint16_t));
        buf_df_ids_       = ctx_.alloc_shared(B * sizeof(int));
        buf_df_h_         = ctx_.alloc_private(B * H * sizeof(float));
        buf_df_x_         = ctx_.alloc_private(B * H * sizeof(uint16_t));
        buf_df_x2_        = ctx_.alloc_private(B * H * sizeof(uint16_t));
        buf_df_dyn_       = ctx_.alloc_private(B * G_dyn * sizeof(float));
        buf_df_o_         = ctx_.alloc_private(B * H * sizeof(float));
        buf_df_q_h_       = ctx_.alloc_private(static_cast<size_t>(dcfg.num_heads) * 8u * dcfg.head_dim * sizeof(uint16_t));
        buf_df_kn_        = ctx_.alloc_private(static_cast<size_t>(dcfg.num_kv_heads) * B * dcfg.head_dim * sizeof(uint16_t));
        buf_df_vn_        = ctx_.alloc_private(static_cast<size_t>(dcfg.num_kv_heads) * B * dcfg.head_dim * sizeof(uint16_t));
        buf_df_attn_out_  = ctx_.alloc_private(B * static_cast<size_t>(dcfg.num_heads) * dcfg.head_dim * sizeof(uint16_t));
        buf_df_topk_part_ = ctx_.alloc_private(P * num_chunks * static_cast<size_t>(dcfg.selector_top_k) * sizeof(LogitCand));
        buf_df_topk_out_  = ctx_.alloc_shared(P * static_cast<size_t>(dcfg.selector_top_k) * sizeof(LogitCand));
        buf_df_sel_h_     = ctx_.alloc_shared(P * static_cast<size_t>(dcfg.selector_rank) * sizeof(uint16_t));
        dflash_loaded_ = true;
    }

    // Optional MTP drafter loading (skipped when DFlash 2 is active; check bundled store_ FIRST!)
    std::string resolved_mtp = want_dflash ? std::string() : mtp_dir;
    if (resolved_mtp == "off" || resolved_mtp == "none") {
        resolved_mtp.clear();
    } else if (resolved_mtp == "auto") {
        std::string cand_dir = model_dir + "-mtp";
        std::string cand_file = model_dir + "/model-mtp.safetensors";
        std::string cand_sibling = std::filesystem::path(model_dir).parent_path().string() + "/awq4-mtp";
        if (store_.has("mtp.fc.weight")) resolved_mtp = "__bundled__";
        else if (std::filesystem::is_directory(cand_dir)) resolved_mtp = cand_dir;
        else if (std::filesystem::is_regular_file(cand_file)) resolved_mtp = cand_file;
        else if (model_dir.find("rowq4") != std::string::npos && std::filesystem::is_directory(cand_sibling)) resolved_mtp = cand_sibling;
        else resolved_mtp.clear();
    }

    if (!resolved_mtp.empty()) {
        const SafetensorsStore* ms = &store_;
        std::string mp = "mtp.";
        if (resolved_mtp != "__bundled__") {
            if (std::filesystem::is_directory(resolved_mtp)) {
                for (const auto& entry : std::filesystem::directory_iterator(resolved_mtp)) {
                    if (entry.path().extension() == ".safetensors") {
                        mtp_store_.load_file(ctx_.device(), entry.path().string());
                    }
                }
            } else {
                mtp_store_.load_file(ctx_.device(), resolved_mtp);
            }
            ms = &mtp_store_;
            mp = mtp_store_.has("mtp.fc.weight") ? "mtp." : "";
        }

        mtp_.fc                    = load_q4(*ms, mp + "fc");
        mtp_.pre_fc_norm_embedding = ms->get(mp + "pre_fc_norm_embedding.weight");
        mtp_.pre_fc_norm_hidden    = ms->get(mp + "pre_fc_norm_hidden.weight");
        mtp_.norm                  = ms->get(mp + "norm.weight");

        std::string lp = mp + "layers.0.";
        mtp_.layer.is_linear      = false;
        mtp_.layer.input_norm     = ms->get(lp + "input_layernorm.weight");
        mtp_.layer.post_attn_norm = ms->get(lp + "post_attention_layernorm.weight");
        mtp_.layer.q_proj         = load_q4(*ms, lp + "self_attn.q_proj");
        mtp_.layer.k_proj         = load_q4(*ms, lp + "self_attn.k_proj");
        mtp_.layer.v_proj         = load_q4(*ms, lp + "self_attn.v_proj");
        mtp_.layer.o_proj         = load_q4(*ms, lp + "self_attn.o_proj");
        mtp_.layer.q_norm         = ms->get(lp + "self_attn.q_norm.weight");
        mtp_.layer.k_norm         = ms->get(lp + "self_attn.k_norm.weight");
        mtp_.layer.gate_proj      = load_q4(*ms, lp + "mlp.gate_proj");
        mtp_.layer.up_proj        = load_q4(*ms, lp + "mlp.up_proj");
        mtp_.layer.down_proj      = load_q4(*ms, lp + "mlp.down_proj");

        // Lossless KV4 + 4k windowed attention for the 1-layer MTP drafter ONLY (independent of target kv_bits_!)
        mtp_kv_bits_ = (head_dim_ == 256) ? 4 : kv_bits_;
        if (mtp_kv_bits_ == kv_bits_ && kv_bits_ < 16) {
            pso_mtp_kv_prep_        = pso_fa_kv_prep_q_;
            pso_mtp_kv_prep_tree_   = pso_fa_kv_prep_q_tree_;
            pso_mtp_attn_f16_       = pso_fa_attn_q_f16_;
            pso_mtp_attn_tree_f16_  = pso_fa_attn_q_tree_f16_;
            pso_mtp_vsplit1_        = pso_fa_vsplit1_q_;
            pso_mtp_vsplit1_tree_   = pso_fa_vsplit1_q_tree_;
            pso_mtp_gqa6_m1_        = pso_fa_gqa6_m1_q_;
            pso_mtp_gqa6_m1_tree_   = pso_fa_gqa6_m1_q_tree_;
            pso_mtp_gqa6_m2_        = pso_fa_gqa6_m2_q_;
            pso_mtp_gqa6_m2_tree_   = pso_fa_gqa6_m2_q_tree_;
            pso_mtp_gqa6_m4_        = pso_fa_gqa6_m4_q_;
            pso_mtp_gqa6_m4_tree_   = pso_fa_gqa6_m4_q_tree_;
            pso_mtp_gqa6_v_         = pso_fa_gqa6_v_q_;
            pso_mtp_gqa6_v_tree_    = pso_fa_gqa6_v_q_tree_;
        } else if (mtp_kv_bits_ < 16) {
            std::string mqs = (mtp_kv_bits_ == 8) ? "q8" : "q4";
            pso_mtp_kv_prep_        = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_" + mqs);
            pso_mtp_kv_prep_tree_   = ctx_.pipeline("qwen38", "qwen_full_attn_kv_prep_" + mqs + "_tree");
            pso_mtp_attn_f16_       = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_" + mqs + "_f16");
            pso_mtp_attn_tree_f16_  = ctx_.pipeline("qwen38", "qwen_gqa_attn_gated_" + mqs + "_tree_f16");
            pso_mtp_vsplit1_        = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1_" + mqs);
            pso_mtp_vsplit1_tree_   = ctx_.pipeline("qwen38", "qwen_gqa_verify_splitk_p1_" + mqs + "_tree");
            pso_mtp_gqa6_m1_        = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + mqs);
            pso_mtp_gqa6_m1_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m1_splitk_p1_" + mqs + "_tree");
            pso_mtp_gqa6_m2_        = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + mqs);
            pso_mtp_gqa6_m2_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m2_splitk_p1_" + mqs + "_tree");
            pso_mtp_gqa6_m4_        = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + mqs);
            pso_mtp_gqa6_m4_tree_   = ctx_.pipeline("qwen38", "qwen_gqa6_m4_splitk_p1_" + mqs + "_tree");
            pso_mtp_gqa6_v_         = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + mqs);
            pso_mtp_gqa6_v_tree_    = ctx_.pipeline("qwen38", "qwen_gqa6_verify_splitk_p1_" + mqs + "_tree");
        } else {
            pso_mtp_kv_prep_        = pso_fa_kv_prep_;
            pso_mtp_kv_prep_tree_   = pso_fa_kv_prep_tree_;
            pso_mtp_attn_f16_       = pso_fa_attn_f16_;
            pso_mtp_attn_tree_f16_  = pso_fa_attn_tree_f16_;
            pso_mtp_vsplit1_        = pso_fa_vsplit1_;
            pso_mtp_vsplit1_tree_   = pso_fa_vsplit1_tree_;
            pso_mtp_gqa6_m1_        = pso_fa_gqa6_m1_;
            pso_mtp_gqa6_m1_tree_   = pso_fa_gqa6_m1_tree_;
            pso_mtp_gqa6_m2_        = pso_fa_gqa6_m2_;
            pso_mtp_gqa6_m2_tree_   = pso_fa_gqa6_m2_tree_;
            pso_mtp_gqa6_m4_        = pso_fa_gqa6_m4_;
            pso_mtp_gqa6_m4_tree_   = pso_fa_gqa6_m4_tree_;
            pso_mtp_gqa6_v_         = pso_fa_gqa6_v_;
            pso_mtp_gqa6_v_tree_    = pso_fa_gqa6_v_tree_;
        }

        size_t mtp_elem_div = (mtp_kv_bits_ == 16) ? 1ull : ((mtp_kv_bits_ == 8) ? 2ull : 4ull);
        size_t kv_sz = (static_cast<size_t>(kv_cap_) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / mtp_elem_div;
        mtp_k_cache_ = ctx_.alloc_private(kv_sz);
        mtp_v_cache_ = ctx_.alloc_private(kv_sz);
        kv_bytes += 2 * kv_sz;
        if (mtp_kv_bits_ < 16) {
            size_t sb_sz = static_cast<size_t>(kv_cap_) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t);
            mtp_k_sb_ = ctx_.alloc_private(sb_sz);
            mtp_v_sb_ = ctx_.alloc_private(sb_sz);
            kv_bytes += 2 * sb_sz;
        }
        // MTP uses direct 4k windowed attention (128 sink + 3968 recent) rather than Quest bounds!
        mtp_k_bounds_ = nil;
        mtp_quest_built_pages_ = 0;

        buf_mtp_seed_hid_   = ctx_.alloc_private(16ull * hidden_size_ * sizeof(float));
        buf_mtp_last_hid_   = ctx_.alloc_private(hidden_size_ * sizeof(float));
        buf_mtp_verify_hid_ = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * hidden_size_ * sizeof(float));
        buf_mtp_in_hid_     = ctx_.alloc_private((C_attn + 1) * hidden_size_ * sizeof(float));
        buf_mtp_sync_ids_   = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY + 10) * sizeof(int));
        buf_draft_dists_    = ctx_.alloc_shared(16ull * sizeof(DraftDist));
        mtp_loaded_ = true;
    }

    // Shared speculative verification & Lazy Fused GDN Replay buffers (used by BOTH MTP and DFlash 2!)
    if (mtp_loaded_ || dflash_loaded_) {
        for (int sl = 0; sl < 2; ++sl) {
            rec_qkv_[sl].resize(num_layers_, nil);
            rec_a_[sl].resize(num_layers_, nil);
            rec_b_[sl].resize(num_layers_, nil);
        }
        rec_slot_ = 0;
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) continue;
            for (int sl = 0; sl < 2; ++sl) {
                rec_qkv_[sl][i] = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * lin_conv_dim_ * sizeof(float));
                rec_a_[sl][i]   = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * lin_num_v_heads_ * sizeof(float));
                rec_b_[sl][i]   = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * lin_num_v_heads_ * sizeof(float));
                kv_bytes += static_cast<size_t>(MAX_VERIFY) * (lin_conv_dim_ + 2 * lin_num_v_heads_) * sizeof(float);
            }
        }
        buf_mtp_logits_     = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY) * vocab_size_ * sizeof(float));
        buf_mtp_argmax_     = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY) * sizeof(uint32_t));
        buf_verify_row_max_ = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * sizeof(float));
        buf_verify_cands_   = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY) * MAX_VERIFY_CANDS * sizeof(LogitCand));
        buf_verify_cnt_     = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY) * sizeof(uint32_t));
    }

    // Finalize embed_tokens loading now that mtp_loaded_ / dflash_loaded_ are known:
    // - When neither MTP nor DFlash is active (and untied): zero-Metal-wiring CpuMmapQ4Weight (-682 MiB wired IOAccelerator RAM!)
    // - When MTP or DFlash is active: dedicated un-prefetched embed_store_ (zero MADV_WILLNEED startup fault, keeps single-CB GPU embed + cb2_pre!)
    if (!embed_tokens_.w.valid()) {
        const bool force_gpu_embed = (std::getenv("MI_CPU_EMBED") && std::atoi(std::getenv("MI_CPU_EMBED")) == 0);
        if (!mtp_loaded_ && !dflash_loaded_ && !force_gpu_embed && cpu_embed_.open(model_dir, model_pfx + "embed_tokens")) {
            use_cpu_embed_ = true;
        } else {
            embed_store_.load_dir_only(ctx_.device(), model_dir, {model_pfx + "embed_tokens."}, /*prefetch_now=*/false);
            embed_tokens_ = load_q4(embed_store_, model_pfx + "embed_tokens");
        }
    }

    // Native Metal Qwen3-VL Vision Tower (lazy init: 0 B vision RAM & 0 ms vision.metal compile until first image!)
    vision_ = QwenVisionTower::create_lazy(ctx_, model_dir);

    // Persistent scratch arena — pack 13 mutually-disjoint private buffers (~225.75 MiB) into ONE
    // tracked MTLHeapTypePlacement (~83.25 MiB -> saves ~142.5 MiB private GPU RAM with 0 ms overhead!).
    size_t val_dim = static_cast<size_t>(lin_num_v_heads_) * lin_head_v_dim_;
    size_t q_full_dim = static_cast<size_t>(num_heads_) * head_dim_;
    size_t kv_full_dim = static_cast<size_t>(num_kv_heads_) * head_dim_;

    buf_token_ids_    = ctx_.alloc_shared(2 * C_attn * sizeof(int));
    buf_hidden_       = use_cpu_embed_
        ? ctx_.alloc_shared(C_attn * hidden_size_ * sizeof(float))
        : ctx_.alloc_private(C_attn * hidden_size_ * sizeof(float));
    if (ane_ && ane_->ready()) {
        buf_hidden_alt_ = use_cpu_embed_
            ? ctx_.alloc_shared(C_attn * hidden_size_ * sizeof(float))
            : ctx_.alloc_private(C_attn * hidden_size_ * sizeof(float));
    }
    // Size buf_normed_ to also hold race-free FP16 SwiGLU output (C_mlp * intermediate_size_ * 2B):
    // swiglu_f32_to_f16 MUST NEVER alias float* gate (4B) and half* out (2B) on the same buffer.
    const size_t sz_normed = std::max(
        C_attn * static_cast<size_t>(hidden_size_) * sizeof(float),
        C_mlp  * static_cast<size_t>(intermediate_size_) * sizeof(uint16_t)
    );
    buf_normed_       = ctx_.alloc_private(sz_normed);
    buf_a_            = ctx_.alloc_private(C_attn * lin_num_v_heads_ * sizeof(float));
    buf_b_            = ctx_.alloc_private(C_attn * lin_num_v_heads_ * sizeof(float));
    buf_g_            = ctx_.alloc_private(C_attn * lin_num_v_heads_ * sizeof(float));
    buf_beta_         = ctx_.alloc_private(C_attn * lin_num_v_heads_ * sizeof(float));
    buf_q_h_          = ctx_.alloc_private(static_cast<size_t>(num_heads_) * 16u * head_dim_ * sizeof(uint16_t));

    const size_t sz_qkv        = C_attn * std::max<size_t>(lin_conv_dim_, 2 * q_full_dim) * sizeof(float);
    const size_t sz_qkv_conv   = C_attn * lin_conv_dim_ * sizeof(float);
    const size_t sz_z          = C_attn * std::max(val_dim, q_full_dim) * sizeof(float);
    const size_t sz_q          = C_attn * std::max<size_t>(lin_num_k_heads_ * lin_head_k_dim_, q_full_dim) * sizeof(float);
    const size_t sz_k          = C_attn * std::max<size_t>(lin_num_k_heads_ * lin_head_k_dim_, kv_full_dim) * sizeof(float);
    const size_t sz_v          = C_attn * kv_full_dim * sizeof(float);
    const size_t sz_attn_out   = C_attn * std::max(val_dim, q_full_dim) * sizeof(float);
    const size_t sz_gdn_normed = C_attn * val_dim * sizeof(float);
    const size_t sz_mlp_gate   = C_mlp * intermediate_size_ * sizeof(float);
    const size_t sz_mlp_up     = C_mlp * intermediate_size_ * sizeof(float);
    const size_t split_o_half_bytes = static_cast<size_t>(MAX_VERIFY) * num_heads_ * 128u * head_dim_ * sizeof(uint16_t);
    const size_t hier_o_f32_bytes   = C_attn * static_cast<size_t>(num_heads_) * head_dim_ * sizeof(float);
    const size_t sz_split_o    = std::max(split_o_half_bytes, hier_o_f32_bytes);
    const size_t sz_mtp_emb    = mtp_loaded_ ? (C_attn * hidden_size_ * sizeof(float)) : 0u;
    const size_t sz_mtp_concat = mtp_loaded_ ? (C_attn * 2 * hidden_size_ * sizeof(float)) : 0u;

    const MTLResourceOptions heap_buf_opts = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked;
    id<MTLDevice> dev = (id<MTLDevice>)ctx_.device();
    auto align_up = [](size_t v, size_t a) -> size_t {
        return (a > 1u) ? ((v + a - 1u) & ~(a - 1u)) : v;
    };
    auto sa = [&](size_t len) -> MTLSizeAndAlign {
        return (len > 0u) ? [dev heapBufferSizeAndAlignWithLength:len options:heap_buf_opts]
                          : MTLSizeAndAlign{0u, 1u};
    };

    MTLSizeAndAlign sa_qkv        = sa(sz_qkv);
    MTLSizeAndAlign sa_qkv_conv   = sa(sz_qkv_conv);
    MTLSizeAndAlign sa_z          = sa(sz_z);
    MTLSizeAndAlign sa_q          = sa(sz_q);
    MTLSizeAndAlign sa_k          = sa(sz_k);
    MTLSizeAndAlign sa_v          = sa(sz_v);
    MTLSizeAndAlign sa_attn_out   = sa(sz_attn_out);
    MTLSizeAndAlign sa_gdn_normed = sa(sz_gdn_normed);
    MTLSizeAndAlign sa_split_o    = sa(sz_split_o);
    MTLSizeAndAlign sa_mlp_gate   = sa(sz_mlp_gate);
    MTLSizeAndAlign sa_mlp_up     = sa(sz_mlp_up);
    MTLSizeAndAlign sa_mtp_emb    = sa(sz_mtp_emb);
    MTLSizeAndAlign sa_mtp_concat = sa(sz_mtp_concat);

    // Phase A/B (Attn & GDN): Slot 0 holds buf_qkv_ (dead after conv1d/q_prep barrier!), aliased AFTER barrier
    // by buf_attn_out_ + (buf_gdn_normed_ [GDN] | buf_split_o_ [Full-Attn]); Slots 1..5 are strictly disjoint!
    const size_t off_qkv        = 0u;
    const size_t off_attn_out   = 0u;
    const size_t off_gdn_normed = align_up(off_attn_out + sa_attn_out.size, sa_gdn_normed.align);
    const size_t off_split_o    = align_up(off_attn_out + sa_attn_out.size, sa_split_o.align);
    const size_t slot0_end      = std::max(off_qkv + sa_qkv.size,
                                           std::max(off_gdn_normed + sa_gdn_normed.size,
                                                    off_split_o + sa_split_o.size));
    const size_t off_qkv_conv   = align_up(slot0_end, sa_qkv_conv.align);
    const size_t off_z          = align_up(off_qkv_conv + sa_qkv_conv.size, sa_z.align);
    const size_t off_q          = align_up(off_z + sa_z.size, sa_q.align);
    const size_t off_k          = align_up(off_q + sa_q.size, sa_k.align);
    const size_t off_v          = align_up(off_k + sa_k.size, sa_v.align);
    const size_t attn_gdn_end   = off_v + sa_v.size;

    // Phase C (MLP): buf_mlp_gate_ and buf_mlp_up_ (also partY for tiled32_sk!) strictly disjoint from each other
    const size_t off_mlp_gate   = 0u;
    const size_t off_mlp_up     = align_up(off_mlp_gate + sa_mlp_gate.size, sa_mlp_up.align);
    const size_t mlp_end        = off_mlp_up + sa_mlp_up.size;

    // Phase D (MTP pre-FC): buf_mtp_emb_ and buf_mtp_concat_ strictly disjoint from each other
    const size_t off_mtp_emb    = 0u;
    const size_t off_mtp_concat = mtp_loaded_ ? align_up(off_mtp_emb + sa_mtp_emb.size, sa_mtp_concat.align) : 0u;
    const size_t mtp_end        = mtp_loaded_ ? (off_mtp_concat + sa_mtp_concat.size) : 0u;

    const size_t heap_bytes = align_up(std::max(attn_gdn_end, std::max(mlp_end, mtp_end)), 16384u);
    MTLHeapDescriptor* hdesc = [[MTLHeapDescriptor alloc] init];
    hdesc.type = MTLHeapTypePlacement;
    hdesc.storageMode = MTLStorageModePrivate;
    hdesc.hazardTrackingMode = MTLHazardTrackingModeTracked;
    hdesc.size = heap_bytes;
    id<MTLHeap> sheap = [dev newHeapWithDescriptor:hdesc];
    scratch_heap_ = sheap;

    auto alloc_in_heap = [&](size_t len, size_t off) -> id {
        if (sheap) {
            id<MTLBuffer> b = [sheap newBufferWithLength:len options:heap_buf_opts offset:off];
            if (b) return b;
        }
        return ctx_.alloc_private(len);
    };

    buf_qkv_        = alloc_in_heap(sz_qkv,        off_qkv);
    buf_qkv_conv_   = alloc_in_heap(sz_qkv_conv,   off_qkv_conv);
    buf_z_          = alloc_in_heap(sz_z,          off_z);
    buf_q_          = alloc_in_heap(sz_q,          off_q);
    buf_k_          = alloc_in_heap(sz_k,          off_k);
    buf_v_          = alloc_in_heap(sz_v,          off_v);
    buf_attn_out_   = alloc_in_heap(sz_attn_out,   off_attn_out);
    buf_gdn_normed_ = alloc_in_heap(sz_gdn_normed, off_gdn_normed);
    buf_split_o_    = alloc_in_heap(sz_split_o,    off_split_o);
    buf_mlp_gate_   = alloc_in_heap(sz_mlp_gate,   off_mlp_gate);
    buf_mlp_up_     = alloc_in_heap(sz_mlp_up,     off_mlp_up);
    if (mtp_loaded_) {
        buf_mtp_emb_    = alloc_in_heap(sz_mtp_emb,    off_mtp_emb);
        buf_mtp_concat_ = alloc_in_heap(sz_mtp_concat, off_mtp_concat);
    }
    buf_split_m_      = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * num_heads_ * 128 * sizeof(float));
    buf_split_l_      = ctx_.alloc_private(static_cast<size_t>(MAX_VERIFY) * num_heads_ * 128 * sizeof(float));
    buf_tree_params_  = ctx_.alloc_shared(sizeof(TreeParams));
    if (sparse_kv_budget_ > 0) {
        buf_quest_scores_ = ctx_.alloc_private(static_cast<size_t>(num_kv_heads_) * 7ull * QUEST_SCORE_STRIDE * sizeof(float));
        buf_quest_pages_  = ctx_.alloc_private(static_cast<size_t>(num_kv_heads_) * QUEST_MAX_ACTIVE_PAGES * sizeof(uint32_t));
        buf_quest_counts_ = ctx_.alloc_private(static_cast<size_t>(num_kv_heads_) * sizeof(uint32_t));
    }
    buf_sk_part_      = ctx_.alloc_private(8ull * static_cast<size_t>(MAX_VERIFY) * hidden_size_ * sizeof(float));
    buf_logits_       = ctx_.alloc_shared(static_cast<size_t>(vocab_size_) * sizeof(float));
    size_t max_am_parts = std::max<size_t>(128u, ceil_div(static_cast<uint32_t>(vocab_size_), 32u));
    size_t min_draft_parts = ceil_div(MAX_DRAFT_ROW_MAP, 8u);
    buf_argmax_parts_ = ctx_.alloc_private(std::max(static_cast<size_t>(MAX_VERIFY) * max_am_parts, min_draft_parts) * 8u);
    buf_argmax_out_   = ctx_.alloc_shared(static_cast<size_t>(MAX_VERIFY) * sizeof(uint32_t));
    buf_draft_row_map_ = ctx_.alloc_shared(static_cast<size_t>(MAX_DRAFT_ROW_MAP) * sizeof(uint32_t));
    // Build BitScreen 1-bit sign(W) x 3-bitplane(X) pre-filter for MTP draft lm_head (general hidden_size via FC_BS_D!)
    bool want_bs = mtp_loaded_ && (hidden_size_ > 0 && (hidden_size_ % 64) == 0 && hidden_size_ <= 8192) &&
                   (!std::getenv("MI_BITSCREEN") || std::atoi(std::getenv("MI_BITSCREEN")) != 0);
    if (want_bs) {
        uint32_t bsv = std::getenv("MI_BITSCREEN_V")
            ? static_cast<uint32_t>(std::atoi(std::getenv("MI_BITSCREEN_V")))
            : 98304u;
        bs_vocab_ = std::clamp(bsv, 32768u, static_cast<uint32_t>(vocab_size_));
        bs_vocab_ = (bs_vocab_ / 2048u) * 2048u; // exact multiple of 2,048 strided buckets
        const size_t bs_kw = static_cast<size_t>(hidden_size_) / 32ull;
        buf_bs_w_        = ctx_.alloc_private(static_cast<size_t>(bs_vocab_) * bs_kw * sizeof(uint32_t));
        buf_bs_meta_     = ctx_.alloc_private(static_cast<size_t>(bs_vocab_) * 2ull * sizeof(float));
        buf_bs_xbp_      = ctx_.alloc_private(8ull * 3ull * bs_kw * sizeof(uint32_t));
        buf_bs_xpc_      = ctx_.alloc_private(8ull * sizeof(float));
        buf_bs_top2_     = ctx_.alloc_private(8ull * 4096ull * 8ull); // ArgmaxOut [8, 4096]
        buf_bs_cands_    = ctx_.alloc_private(8ull * 256ull * sizeof(uint32_t));
        buf_bs_cand_cnt_ = ctx_.alloc_private(8ull * sizeof(uint32_t));
        buf_bs_extra_    = ctx_.alloc_shared(128ull * sizeof(uint32_t));
        kv_bytes += static_cast<size_t>(bs_vocab_) * (bs_kw * 4ull + 8ull);

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_lmh_build_bitscreen_];
            [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_bs_w_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_bs_meta_ offset:0 atIndex:4];
            [enc setBytes:&bs_vocab_ length:sizeof(bs_vocab_) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(bs_vocab_, 4u), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                throw std::runtime_error(std::string("Metal BitScreen build error: ") +
                                         [[cb.error localizedDescription] UTF8String]);
            }
        }
        bs_enabled_ = true;
    }
    if (cfg.contains("eos_token_id")) {
        const auto& e = cfg["eos_token_id"];
        if (e.is_array()) {
            for (const auto& x : e.as_array()) info_.eos_token_ids.push_back(static_cast<int>(x.as_int()));
        } else if (e.is_int()) {
            info_.eos_token_ids.push_back(static_cast<int>(e.as_int()));
        }
    } else if (tc.contains("eos_token_id")) {
        const auto& e = tc["eos_token_id"];
        if (e.is_array()) {
            for (const auto& x : e.as_array()) info_.eos_token_ids.push_back(static_cast<int>(x.as_int()));
        } else if (e.is_int()) {
            info_.eos_token_ids.push_back(static_cast<int>(e.as_int()));
        }
    }
    if (info_.eos_token_ids.empty()) info_.eos_token_ids = {248046, 248044};

    // Dynamically discover special/added token IDs from config.json + tokenizer_config.json (model-agnostic!)
    auto add_special_id = [&](int64_t tid) {
        if (tid >= 0 && tid < vocab_size_) {
            special_tok_ids_.push_back(static_cast<uint32_t>(tid));
        }
    };
    for (int eid : info_.eos_token_ids) add_special_id(eid);
    add_special_id(image_token_id_);
    for (const char* k : {"bos_token_id", "pad_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"}) {
        if (cfg.contains(k) && cfg[k].is_int()) add_special_id(cfg[k].as_int());
        if (tc.contains(k) && tc[k].is_int()) add_special_id(tc[k].as_int());
    }
    std::string tok_cfg_path = model_dir + "/tokenizer_config.json";
    if (std::filesystem::is_regular_file(tok_cfg_path)) {
        try {
            Json tcfg = parse_json(read_file_to_string(tok_cfg_path));
            if (tcfg.contains("added_tokens_decoder") && tcfg["added_tokens_decoder"].is_object()) {
                for (const auto& kv : tcfg["added_tokens_decoder"].as_object()) {
                    char* endp = nullptr;
                    long long val = std::strtoll(kv.first.c_str(), &endp, 10);
                    if (endp && *endp == '\0') add_special_id(static_cast<int64_t>(val));
                }
            }
        } catch (...) {
            // Non-fatal: fall back to config-derived IDs
        }
    }
    std::sort(special_tok_ids_.begin(), special_tok_ids_.end());
    special_tok_ids_.erase(std::unique(special_tok_ids_.begin(), special_tok_ids_.end()), special_tok_ids_.end());
    const uint32_t tail_floor = (vocab_size_ > 4096) ? static_cast<uint32_t>(vocab_size_ - 4096) : FAST_DRAFT_VOCAB;
    bool has_tail_special = false;
    for (uint32_t sid : special_tok_ids_) {
        if (sid >= tail_floor) { has_tail_special = true; break; }
    }
    if (!has_tail_special && vocab_size_ >= 248077) {
        for (uint32_t sid = 248044u; sid < 248077u; ++sid) special_tok_ids_.push_back(sid);
    }
    special_tok_lo_ = static_cast<uint32_t>(vocab_size_);
    for (uint32_t sid : special_tok_ids_) {
        if (sid >= tail_floor) {
            special_tok_lo_ = sid;
            break;
        }
    }

    init_draft_row_map();
    pld4_.resize(PLD_MASK + 1u);
    pld3_.resize(PLD_MASK + 1u);

    info_.arch = "qwen38";
    info_.model_dir = model_dir;
    info_.quant_desc = "Q4_AFFINE_G64 (MMA) + GDN-BF16";
    if (kv_bits_ < 16) {
        info_.quant_desc += " + KV" + std::to_string(kv_bits_);
    }
    if (sparse_kv_budget_ > 0) {
        info_.quant_desc += " + Quest(" + std::to_string(sparse_kv_budget_) + ")";
    }
    if (ane_ && ane_->ready()) {
        info_.quant_desc += " + ANE(" + ane_->tag() + ")";
        if (ane_share_ && ane_share_->shared_count() > 0) {
            info_.quant_desc += " + ANE-Share(" + std::to_string(ane_share_->shared_count()) + "/" + std::to_string(num_layers_) + ")";
        }
    }
    if (vision_ && vision_->ready()) {
        info_.quant_desc += " + Vision";
    }
    if (mtp_loaded_) {
        info_.quant_desc += " + MTP depth " + std::to_string(mtp_depth_) + (pld_enabled_ ? " + PLD" : "");
    }
    if (dflash_loaded_) {
        const DFlashConfig& dcfg = dflash_->config();
        info_.quant_desc += " + DFlash2(b" + std::to_string(dcfg.block_size) + ", " + std::to_string(dcfg.num_layers) + "L)";
    }
    info_.vocab_size = vocab_size_;
    info_.hidden_size = hidden_size_;
    info_.num_layers = num_layers_;
    info_.num_visits = num_layers_;
    info_.max_seq_len = max_seq_len_;
    info_.mapped_weight_bytes = store_.total_mapped_bytes() + mtp_store_.total_mapped_bytes() + embed_store_.total_mapped_bytes()
                              + (dflash_loaded_ ? dflash_->gpu_weight_bytes() : 0ull);
    info_.kv_cache_bytes = kv_bytes;

    // Join async 4-deep ANE warmup (overlapped with safetensors load & BitScreen build!) before returning ready model
    if (ane_ && ane_->ready()) {
        ane_->wait_warmup();
    }

    reset();

    auto t1 = std::chrono::steady_clock::now();
    info_.load_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Qwen38Model::~Qwen38Model() {
    if (save_thread_.joinable()) save_thread_.join();
}

void Qwen38Model::ensure_kv_capacity(int needed) {
    if (needed <= kv_cap_) return;
    if (needed > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds max_seq_len (" + std::to_string(max_seq_len_) + ")");
    }
    int new_cap = needed;
    if (needed <= 16384) {
        new_cap = std::max(needed, std::max(kv_cap_ * 2, 2048));
        new_cap = ((new_cap + 2047) / 2048) * 2048;
    } else {
        // Above 16k: stepwise +16k growth avoids 2x overshoot (e.g. 66k -> 80k instead of 131k!)
        new_cap = ((needed + 16383) / 16384) * 16384;
    }
    new_cap = std::min(new_cap, max_seq_len_);

    const size_t elem_div = (kv_bits_ == 16) ? 1ull : ((kv_bits_ == 8) ? 2ull : 4ull);
    const size_t row_kv_bytes = (static_cast<size_t>(num_kv_heads_) * head_dim_ * sizeof(uint16_t)) / elem_div;
    const size_t row_sb_bytes = (kv_bits_ < 16)
        ? (static_cast<size_t>(num_kv_heads_) * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    const size_t new_kv_sz = static_cast<size_t>(new_cap) * row_kv_bytes;
    const size_t new_sb_sz = static_cast<size_t>(new_cap) * row_sb_bytes;
    const size_t copy_kv = static_cast<size_t>(pos_) * row_kv_bytes;
    const size_t copy_sb = static_cast<size_t>(pos_) * row_sb_bytes;
    const size_t new_kb_sz = (sparse_kv_budget_ > 0)
        ? (static_cast<size_t>(ceil_div(static_cast<uint32_t>(new_cap), QUEST_PAGE_SIZE)) * num_kv_heads_ * head_dim_ * 4ull)
        : 0ull;
    const size_t page_kb_bytes = static_cast<size_t>(num_kv_heads_) * head_dim_ * 4ull;

    // Layer-by-layer alloc -> blit -> swap/release inside @autoreleasepool:
    // eliminates the 2x all-layers KV RAM peak spike (+3.2 GiB -> <= 1 single layer +0.19 GiB, or 0 B when pos_ == 0)!
    for (int i = 0; i < num_layers_; ++i) {
        if (layers_[i].is_linear) continue;
        @autoreleasepool {
            if (copy_kv == 0) {
                states_[i].k_cache = nil;
                states_[i].v_cache = nil;
                states_[i].k_sb = nil;
                states_[i].v_sb = nil;
                states_[i].k_bounds = nil;
            }
            id nk = ctx_.alloc_private(new_kv_sz);
            id nv = ctx_.alloc_private(new_kv_sz);
            id nksb = (row_sb_bytes > 0) ? ctx_.alloc_private(new_sb_sz) : nil;
            id nvsb = (row_sb_bytes > 0) ? ctx_.alloc_private(new_sb_sz) : nil;
            id nkb  = (new_kb_sz > 0) ? ctx_.alloc_private(new_kb_sz) : nil;

            if (copy_kv > 0) {
                id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].k_cache sourceOffset:0 toBuffer:(id<MTLBuffer>)nk destinationOffset:0 size:copy_kv];
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].v_cache sourceOffset:0 toBuffer:(id<MTLBuffer>)nv destinationOffset:0 size:copy_kv];
                if (copy_sb > 0) {
                    [blit copyFromBuffer:(id<MTLBuffer>)states_[i].k_sb sourceOffset:0 toBuffer:(id<MTLBuffer>)nksb destinationOffset:0 size:copy_sb];
                    [blit copyFromBuffer:(id<MTLBuffer>)states_[i].v_sb sourceOffset:0 toBuffer:(id<MTLBuffer>)nvsb destinationOffset:0 size:copy_sb];
                }
                if (new_kb_sz > 0 && states_[i].quest_built_pages > 0) {
                    size_t copy_kb = static_cast<size_t>(states_[i].quest_built_pages) * page_kb_bytes;
                    [blit copyFromBuffer:(id<MTLBuffer>)states_[i].k_bounds sourceOffset:0 toBuffer:(id<MTLBuffer>)nkb destinationOffset:0 size:copy_kb];
                }
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            states_[i].k_cache = nk;
            states_[i].v_cache = nv;
            if (row_sb_bytes > 0) {
                states_[i].k_sb = nksb;
                states_[i].v_sb = nvsb;
            }
            if (new_kb_sz > 0) {
                states_[i].k_bounds = nkb;
            }
        }
    }

    if (mtp_loaded_) {
        const size_t mtp_elem_div = (mtp_kv_bits_ == 16) ? 1ull : ((mtp_kv_bits_ == 8) ? 2ull : 4ull);
        const size_t mtp_row_kv_bytes = (static_cast<size_t>(num_kv_heads_) * head_dim_ * sizeof(uint16_t)) / mtp_elem_div;
        const size_t mtp_row_sb_bytes = (mtp_kv_bits_ < 16)
            ? (static_cast<size_t>(num_kv_heads_) * (head_dim_ / 64) * 2 * sizeof(uint16_t))
            : 0;
        const size_t mtp_new_kv_sz = static_cast<size_t>(new_cap) * mtp_row_kv_bytes;
        const size_t mtp_new_sb_sz = static_cast<size_t>(new_cap) * mtp_row_sb_bytes;
        @autoreleasepool {
            if (mtp_pos_ <= 0) {
                mtp_k_cache_ = nil;
                mtp_v_cache_ = nil;
                mtp_k_sb_ = nil;
                mtp_v_sb_ = nil;
                mtp_k_bounds_ = nil;
            }
            id nk = ctx_.alloc_private(mtp_new_kv_sz);
            id nv = ctx_.alloc_private(mtp_new_kv_sz);
            id nksb = (mtp_row_sb_bytes > 0) ? ctx_.alloc_private(mtp_new_sb_sz) : nil;
            id nvsb = (mtp_row_sb_bytes > 0) ? ctx_.alloc_private(mtp_new_sb_sz) : nil;

            if (mtp_pos_ > 0) {
                size_t copy_mtp = static_cast<size_t>(mtp_pos_) * mtp_row_kv_bytes;
                id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromBuffer:(id<MTLBuffer>)mtp_k_cache_ sourceOffset:0 toBuffer:(id<MTLBuffer>)nk destinationOffset:0 size:copy_mtp];
                [blit copyFromBuffer:(id<MTLBuffer>)mtp_v_cache_ sourceOffset:0 toBuffer:(id<MTLBuffer>)nv destinationOffset:0 size:copy_mtp];
                if (mtp_row_sb_bytes > 0) {
                    size_t copy_mtp_sb = static_cast<size_t>(mtp_pos_) * mtp_row_sb_bytes;
                    [blit copyFromBuffer:(id<MTLBuffer>)mtp_k_sb_ sourceOffset:0 toBuffer:(id<MTLBuffer>)nksb destinationOffset:0 size:copy_mtp_sb];
                    [blit copyFromBuffer:(id<MTLBuffer>)mtp_v_sb_ sourceOffset:0 toBuffer:(id<MTLBuffer>)nvsb destinationOffset:0 size:copy_mtp_sb];
                }
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            mtp_k_cache_ = nk;
            mtp_v_cache_ = nv;
            if (mtp_row_sb_bytes > 0) {
                mtp_k_sb_ = nksb;
                mtp_v_sb_ = nvsb;
            }
            mtp_k_bounds_ = nil;
        }
    }
    kv_cap_ = new_cap;
}

void Qwen38Model::shrink_kv_capacity(int target_cap) {
    int new_cap = std::min(max_seq_len_, std::max(target_cap, 2048));
    if (new_cap >= kv_cap_) return;

    const size_t elem_div = (kv_bits_ == 16) ? 1ull : ((kv_bits_ == 8) ? 2ull : 4ull);
    const size_t row_kv_bytes = (static_cast<size_t>(num_kv_heads_) * head_dim_ * sizeof(uint16_t)) / elem_div;
    const size_t row_sb_bytes = (kv_bits_ < 16)
        ? (static_cast<size_t>(num_kv_heads_) * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    const size_t new_kv_sz = static_cast<size_t>(new_cap) * row_kv_bytes;
    const size_t new_sb_sz = static_cast<size_t>(new_cap) * row_sb_bytes;
    const size_t new_kb_sz = (sparse_kv_budget_ > 0)
        ? (static_cast<size_t>(ceil_div(static_cast<uint32_t>(new_cap), QUEST_PAGE_SIZE)) * num_kv_heads_ * head_dim_ * 4ull)
        : 0ull;

    for (int i = 0; i < num_layers_; ++i) {
        if (layers_[i].is_linear) continue;
        @autoreleasepool {
            states_[i].k_cache = nil;
            states_[i].v_cache = nil;
            states_[i].k_sb = nil;
            states_[i].v_sb = nil;
            states_[i].k_bounds = nil;
            states_[i].k_cache = ctx_.alloc_private(new_kv_sz);
            states_[i].v_cache = ctx_.alloc_private(new_kv_sz);
            if (row_sb_bytes > 0) {
                states_[i].k_sb = ctx_.alloc_private(new_sb_sz);
                states_[i].v_sb = ctx_.alloc_private(new_sb_sz);
            }
            if (new_kb_sz > 0) {
                states_[i].k_bounds = ctx_.alloc_private(new_kb_sz);
            }
            states_[i].quest_built_pages = 0;
        }
    }
    if (mtp_loaded_) {
        const size_t mtp_elem_div = (mtp_kv_bits_ == 16) ? 1ull : ((mtp_kv_bits_ == 8) ? 2ull : 4ull);
        const size_t mtp_row_kv_bytes = (static_cast<size_t>(num_kv_heads_) * head_dim_ * sizeof(uint16_t)) / mtp_elem_div;
        const size_t mtp_row_sb_bytes = (mtp_kv_bits_ < 16)
            ? (static_cast<size_t>(num_kv_heads_) * (head_dim_ / 64) * 2 * sizeof(uint16_t))
            : 0;
        const size_t mtp_new_kv_sz = static_cast<size_t>(new_cap) * mtp_row_kv_bytes;
        const size_t mtp_new_sb_sz = static_cast<size_t>(new_cap) * mtp_row_sb_bytes;
        @autoreleasepool {
            mtp_k_cache_ = nil;
            mtp_v_cache_ = nil;
            mtp_k_sb_ = nil;
            mtp_v_sb_ = nil;
            mtp_k_bounds_ = nil;
            mtp_k_cache_ = ctx_.alloc_private(mtp_new_kv_sz);
            mtp_v_cache_ = ctx_.alloc_private(mtp_new_kv_sz);
            if (mtp_row_sb_bytes > 0) {
                mtp_k_sb_ = ctx_.alloc_private(mtp_new_sb_sz);
                mtp_v_sb_ = ctx_.alloc_private(mtp_new_sb_sz);
            }
            mtp_quest_built_pages_ = 0;
        }
    }
    kv_cap_ = new_cap;
}

void Qwen38Model::reset() {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) {
                id<MTLBuffer> cs = states_[i].conv_state;
                id<MTLBuffer> ss = states_[i].ssm_state;
                [blit fillBuffer:cs range:NSMakeRange(0, cs.length) value:0];
                [blit fillBuffer:ss range:NSMakeRange(0, ss.length) value:0];
            } else {
                states_[i].quest_built_pages = 0;
            }
        }
        mtp_quest_built_pages_ = 0;
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    pos_ = 0;
    rope_delta_ = 0;
    mtp_pos_ = 0;
    df_kv_len_ = 0;
    df_pending_commit_ = 0;
    df_pending_pos_ = 0;
    if (kv_cap_ > 8192) {
        shrink_kv_capacity(2048);
    }
    mtp_seeded_ = false;
    pending_sync_ = false;
    pending_replay_m_ = 0;
    pending_path_noncontig_ = false;
    pending_path_len_ = 0;
    rec_slot_ = 0;
    cur_tree_ = TreeParams{};
    last_t7_m_ = 0;
    last_t7_tree_ = false;
    cur_mrope_buf_ = nil;
    cur_vis_count_ = 0;
    cur_mtp_depth_ = mtp_depth_;
    mtp_alpha_ema_ = 0.72f;
    draft_vocab_size_ = std::min<uint32_t>(FAST_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
    draft_miss_mask_ = 0u;
    reset_draft_row_map();
    token_history_.clear();
    suffix_cache_.begin_turn();
    if (++pld_gen_ == 0u) {
        std::fill(pld4_.begin(), pld4_.end(), PldSlot{});
        std::fill(pld3_.begin(), pld3_.end(), PldSlot{});
        pld_gen_ = 1u;
    }
}

void Qwen38Model::add_suffix_corpus(const int* tokens, int n_tokens) {
    suffix_cache_.add_repo_doc(tokens, n_tokens);
}

void Qwen38Model::init_draft_row_map() {
    uint32_t vb = std::getenv("DRAFT_VBASE") ? static_cast<uint32_t>(std::atoi(std::getenv("DRAFT_VBASE"))) : 32768u;
    vb = std::clamp(vb, 1024u, std::min<uint32_t>(MAX_DRAFT_ROW_MAP - 512u, static_cast<uint32_t>(vocab_size_)));
    draft_vbase_ = vb;
    draft_in_map_.assign(static_cast<size_t>(vocab_size_), 0u);
    uint32_t* rm = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_draft_row_map_ contents]);
    uint32_t n = 0u;
    for (uint32_t id = 0u; id < draft_vbase_; ++id) {
        rm[n++] = id;
        draft_in_map_[id] = 1u;
    }
    // Always include dynamically discovered special/control tokens (e.g. <|im_end|>, <think>, </think>, <tool_call>)
    for (uint32_t id : special_tok_ids_) {
        if (id < static_cast<uint32_t>(vocab_size_) && !draft_in_map_[id] && n < MAX_DRAFT_ROW_MAP) {
            rm[n++] = id;
            draft_in_map_[id] = 1u;
        }
    }
    draft_base_count_ = n;
    draft_row_count_ = n;

    if (bs_enabled_ && buf_bs_extra_) {
        bs_in_extra_.assign(static_cast<size_t>(vocab_size_), 0u);
        uint32_t* ex = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_bs_extra_ contents]);
        uint32_t ne = 0u;
        for (uint32_t id : special_tok_ids_) {
            if (id < static_cast<uint32_t>(vocab_size_) && id >= bs_vocab_ && !bs_in_extra_[id] && ne < 128u) {
                ex[ne++] = id;
                bs_in_extra_[id] = 1u;
            }
        }
        bs_extra_base_ = ne;
        bs_extra_count_ = ne;
    }
}

void Qwen38Model::reset_draft_row_map() {
    if (!buf_draft_row_map_ || draft_in_map_.empty()) return;
    uint32_t* rm = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_draft_row_map_ contents]);
    for (uint32_t i = draft_base_count_; i < draft_row_count_; ++i) {
        uint32_t id = rm[i];
        if (id < draft_in_map_.size()) draft_in_map_[id] = 0u;
    }
    draft_row_count_ = draft_base_count_;
    if (bs_enabled_ && buf_bs_extra_ && !bs_in_extra_.empty()) {
        uint32_t* ex = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_bs_extra_ contents]);
        for (uint32_t i = bs_extra_base_; i < bs_extra_count_; ++i) {
            uint32_t id = ex[i];
            if (id < bs_in_extra_.size()) bs_in_extra_[id] = 0u;
        }
        bs_extra_count_ = bs_extra_base_;
    }
}

void Qwen38Model::add_draft_token(int tok) {
    if (tok < 0 || static_cast<uint32_t>(tok) >= draft_in_map_.size()) return;
    if (!draft_in_map_[tok] && draft_row_count_ < MAX_DRAFT_ROW_MAP) {
        uint32_t* rm = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_draft_row_map_ contents]);
        draft_in_map_[tok] = 1u;
        rm[draft_row_count_++] = static_cast<uint32_t>(tok);
    }
    if (bs_enabled_ && static_cast<uint32_t>(tok) >= bs_vocab_ &&
        buf_bs_extra_ && !bs_in_extra_[tok] && bs_extra_count_ < 128u) {
        uint32_t* ex = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_bs_extra_ contents]);
        bs_in_extra_[tok] = 1u;
        ex[bs_extra_count_++] = static_cast<uint32_t>(tok);
    }
}

id Qwen38Model::encode_draft_lm_head(id enc_id, uint32_t B, uint32_t& out_V, uint32_t& out_n_parts) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    if (bs_enabled_ && B >= 1u && B <= 3u) {
        // Stage 1a: Quantize FP16 buf_normed_[0..B-1] -> 3 bitplanes (<5 us)
        const uint32_t bs_kw = static_cast<uint32_t>(hidden_size_) / 32u;
        const uint32_t quant_tg = (bs_kw + 31u) & ~31u;
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_bs_quant_x_];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_xbp_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_xpc_ offset:0 atIndex:2];
        [enc setBytes:&B length:sizeof(B) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(quant_tg, 1, 1)];
        barrier();

        // Stage 1b: 1-bit sign(W) x 3-bitplane(X) popcount screen + FUSED per-simdgroup register top-2!
        struct BitScreenParams { uint32_t V; uint32_t B; uint32_t n_extra; uint32_t R; };
        BitScreenParams bsp{bs_vocab_, B, bs_extra_count_, bs_vocab_ >> 11};
        id score_pso = (B == 1u) ? pso_bs_score_b1_ : ((B == 2u) ? pso_bs_score_b2_ : pso_bs_score_b3_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)score_pso];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_w_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_meta_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_xbp_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_xpc_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_top2_ offset:0 atIndex:4];
        [enc setBytes:&bsp length:sizeof(bsp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(512u, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        barrier();

        // Stage 1c: Single-pass register-only 64-bin selector over 4,096 L2 top2 items (<8 us!)
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_bs_select_cands_];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_top2_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_extra_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_cands_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_cand_cnt_ offset:0 atIndex:3];
        [enc setBytes:&bsp length:sizeof(bsp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // Stage 2: Exact Q4_g64 rescore on <=256 candidates per batch row (64 TGs -> saturates all 40 M4 Max cores!)
        QmmParams qp{B, 256u, lm_head_.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_bs_rescore_q4_];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_cands_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_bs_cand_cnt_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:7];
        [enc setBytes:&qp length:sizeof(qp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(64u, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        barrier();

        out_V = 256u;
        out_n_parts = 64u;
        return buf_bs_cands_;
    }

    const uint32_t V = draft_row_count_;
    out_V = V;
    if (B >= 2u) {
        uint32_t n_tg_am = ceil_div(V, 32u);
        out_n_parts = n_tg_am;
        QmmParams qp{B, V, lm_head_.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_lmhead_max_f16x_indexed_];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_draft_row_map_ offset:0 atIndex:6];
        [enc setBytes:&qp length:sizeof(qp) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        uint32_t n_tg_am = ceil_div(V, 8u);
        out_n_parts = n_tg_am;
        QmmParams qp{1u, V, lm_head_.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmv_q4_f16x_indexed_max_];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_draft_row_map_ offset:0 atIndex:6];
        [enc setBytes:&qp length:sizeof(qp) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    }
    barrier();
    return buf_draft_row_map_;
}

void Qwen38Model::maybe_widen_draft_vocab(int tok) {
    if (draft_vocab_size_ >= static_cast<uint32_t>(vocab_size_)) return;
    if (tok < 0 || static_cast<uint32_t>(tok) >= special_tok_lo_) return; // Ignore special ChatML / control tokens
    bool missed = (static_cast<uint32_t>(tok) >= draft_vocab_size_);
    draft_miss_mask_ = ((draft_miss_mask_ << 1) | (missed ? 1u : 0u)) & 0xFFFFu;
    if (!missed) return;
    // Widen only when >= 2 out of the last 16 emitted bonus tokens exceeded the current draft tier
    // (prevents a single rare English word like "tokenizer" from bloating 5x draft_head from 1.0 ms -> 7.25 ms!)
    if (__builtin_popcount(draft_miss_mask_) >= 2) {
        if (draft_vocab_size_ < MED_DRAFT_VOCAB) {
            draft_vocab_size_ = std::min<uint32_t>(MED_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
        } else if (draft_vocab_size_ < HI_DRAFT_VOCAB) {
            draft_vocab_size_ = std::min<uint32_t>(HI_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
        } else {
            draft_vocab_size_ = static_cast<uint32_t>(vocab_size_);
        }
        draft_miss_mask_ = 0u;
    }
}

void Qwen38Model::encode_qmm_q4(
    id enc_id, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t add_flag = add_res ? 1u : 0u;

    if (M > 24u) {
        uint32_t r = M % 64u;
        uint32_t main_m = (r > 0u && r <= 24u) ? (M - r) : M;
        QmmParams p{main_m, w.out_dim, w.in_dim, add_flag};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q4_tiled_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 64u), ceil_div(main_m, 64u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (main_m < M) {
            size_t tail_x_off = x_off + static_cast<size_t>(main_m) * w.in_dim * sizeof(float);
            size_t tail_y_off = y_off + static_cast<size_t>(main_m) * w.out_dim * sizeof(float);
            encode_qmm_q4(enc_id, w, x_buf, tail_x_off, y_buf, tail_y_off, r, add_res);
        }
        return;
    }

    if (M >= 9u) {
        QmmParams p{M, w.out_dim, w.in_dim, add_flag};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm16_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), ceil_div(M, 16u), 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }

    if (M >= 2u) {
        QmmParams p{M, w.out_dim, w.in_dim, add_flag};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }

    QmmParams p{M, w.out_dim, w.in_dim, add_flag};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q4_];
    [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 16u), M, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

void Qwen38Model::encode_qmm_q4_f16x(
    id enc_id, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    QmmParams p{M, w.out_dim, w.in_dim, add_res ? 1u : 0u};
    if (M == 1u) {
        // Round 18: Ultra-bandwidth M=1 FP16-X uint4+half4 GEMV (+ Split-K for narrow-N large-K projections!)
        if (w.out_dim <= static_cast<uint32_t>(hidden_size_) && w.in_dim >= 6144u) {
            uint32_t sk = (w.in_dim >= 12288u) ? 4u : 2u;
            QmmSkParams sp{1u, w.out_dim, w.in_dim, add_res ? 1u : 0u, sk};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmv_q4_f16x_sk_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 8u), 1, sk)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(w.out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmv_q4_f16x_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 8u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        return;
    }
    if (M <= 8u) {
        if (w.out_dim == static_cast<uint32_t>(hidden_size_) && w.in_dim >= 6144u) {
            uint32_t sk = (w.in_dim >= 12288u) ? 8u : 2u;
            QmmSkParams sp{M, w.out_dim, w.in_dim, add_res ? 1u : 0u, sk};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_f16x_sk_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), 1, sk)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(M * w.out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_f16x_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return;
    }
    if (w.out_dim <= 64u) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q4_smalln_f16x_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 16u), ceil_div(M, 32u), 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        return;
    }
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q4_tiled_f16x_];
    [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 64u), ceil_div(M, 64u), 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void Qwen38Model::encode_qmm_q4_f16x_and_norm(
    id enc_id, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off,
    uint32_t M, const TensorView& next_norm, id normed_buf
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t sk = (w.in_dim >= 12288u) ? 8u : 2u;
    QmmSkNormParams sp{M, w.out_dim, w.in_dim, 1u, sk, rms_eps_};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_f16x_sk_];
    [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:4];
    [enc setBytes:&sp length:sizeof(QmmSkParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), 1, sk)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_norm_f16_];
    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)next_norm.buffer offset:next_norm.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)normed_buf offset:0 atIndex:3];
    [enc setBytes:&sp length:sizeof(sp) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void Qwen38Model::encode_qmm_q4_f16x_and_dual_norm(
    id enc_id, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off,
    uint32_t M, const TensorView& next_norm, id out_f32_buf, id out_f16_buf
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t sk = (w.in_dim >= 12288u) ? 8u : 2u;
    QmmSkNormParams sp{M, w.out_dim, w.in_dim, 1u, sk, rms_eps_};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_f16x_sk_];
    [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:4];
    [enc setBytes:&sp length:sizeof(QmmSkParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), 1, sk)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_dual_norm_f16_];
    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)next_norm.buffer offset:next_norm.offset atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)out_f32_buf offset:0 atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)out_f16_buf offset:0 atIndex:4];
    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void Qwen38Model::encode_mlp_q4(id enc_id, const LayerWeights& lw, uint32_t M,
                                const TensorView* next_norm, id dual_out_f32, size_t hid_off) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    // Single-Copy ANE 4-bit (model.hwx: hf4 / lut4) Native BF16-AMX MLP Path (4.0 bpw, 0 duplicate GPU weight pages!)
    if (lw.ane_share_idx >= 0 && ane_share_ && ane_share_->has_layer(lw.ane_share_idx)) {
        const AneShareLayer& sl = ane_share_->layer(lw.ane_share_idx);
        const uint32_t tgx_gu_m1 = ceil_div(sl.intermediate_size / 8u, 4u);
        const uint32_t tgx_gu_m8 = sl.intermediate_size / 16u;
        const uint32_t tgx_dn_sk = 16u * ceil_div(sl.hidden_size / 128u, 4u);
        const uint32_t sk_down   = 2u * sl.n_subblocks;

        if (M == 1u) {
            AneShareParams ap = make_ane_share_params(sl, 1u);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_m1_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_m1_sk_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            QmmSkParams sp{1u, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk_down};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(static_cast<uint32_t>(hidden_size_), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
            return;
        }

        if (M <= 8u) {
            AneShareParams ap = make_ane_share_params(sl, M);
            // AMX swiglu_bf16 + KS=4 down_sk4_bf16 beats scalar m4 by -5.16 ms even at M=4 (23.07 vs 28.23 ms)!
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_bf16_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            // KS=4 (128 thr, 4 KB SRAM): 0% SG tail waste on DK=2176!
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_sk4_bf16_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            QmmSkNormParams sp{M, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk_down, rms_eps_};
            if (M >= 2u && next_norm != nullptr) {
                if (dual_out_f32 != nil) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_dual_norm_f16_];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)next_norm->buffer offset:next_norm->offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)dual_out_f32 offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:4];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_norm_f16_];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)next_norm->buffer offset:next_norm->offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
                }
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
                [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:1];
                [enc setBytes:&sp length:sizeof(QmmSkParams) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(M * static_cast<uint32_t>(hidden_size_), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            barrier();
            return;
        }

        if (M <= 24u) {
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.post_attn_norm_share.buffer offset:lw.post_attn_norm_share.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        // Short / tail batches (9 <= M <= 64): M=8 AMX micro-batches (376 GB/s -> 23.1 ms/8 tok, 185 ms for M=64!)
        if (M <= 64u) {
            for (uint32_t m0 = 0u; m0 < M; m0 += 8u) {
                uint32_t sub = std::min<uint32_t>(8u, M - m0);
                AneShareParams ap = make_ane_share_params(sl, sub);
                size_t x_off = static_cast<size_t>(m0) * hidden_size_ * sizeof(uint16_t);
                size_t y_off = hid_off + static_cast<size_t>(m0) * hidden_size_ * sizeof(float);
                if (sub == 1u) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_m1_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:x_off atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_m1_sk_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:x_off atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m8, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_sk4_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                }
                QmmSkParams sp{sub, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk_down};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
                [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:y_off atIndex:1];
                [enc setBytes:&sp length:sizeof(sp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(sub * static_cast<uint32_t>(hidden_size_), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }
            return;
        }
        // Mid-batch / Co-Exec (17 <= M <= 160): Fused BM=32 double-buffered SwiGLU + Split-K down_proj!
        // Uses buf_mlp_up_ (34 MiB, unused since swiglu_tiled32 is fused!) as partY -> NEVER buf_sk_part_ (1.25 MiB)!
        if (M <= 160u && pso_ane_hf4_swiglu_tiled32_ != nil && pso_ane_hf4_down_tiled32_sk_ != nil) {
            AneShareParams ap = make_ane_share_params(sl, M);
            const uint32_t m_tiles  = ceil_div(M, 32u);
            const uint32_t tgx_gu32 = sl.n_subblocks * ceil_div(sl.sub_size / 8u, 4u);
            const uint32_t tgx_dn32 = 16u * ceil_div(sl.hidden_size / 128u, 4u);
            const uint32_t n_q      = 2u * sl.n_subblocks;
            const uint32_t want_sk  = (M <= 64u) ? 4u : 2u;
            const size_t part_cap   = [(id<MTLBuffer>)buf_mlp_up_ length];
            uint32_t sk = 1u;
            for (uint32_t cand : {want_sk, 2u, 1u}) {
                if (cand <= n_q && (n_q % cand == 0u) &&
                    static_cast<size_t>(cand) * M * hidden_size_ * sizeof(float) <= part_cap) {
                    sk = cand;
                    break;
                }
            }

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_tiled32_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_gu32, m_tiles, 1u)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_tiled32_sk_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_dn32, m_tiles, sk)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            QmmSkParams sp{M, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:hid_off atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(M * static_cast<uint32_t>(hidden_size_), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
            return;
        }
        // Large GPU-only chunks (M > 160): BM=64, BN=64 tiled hf4 directly from single-copy model.hwx!
        AneShareParams ap = make_ane_share_params(sl, M);
        const uint32_t m_tiles = ceil_div(M, 64u);
        const uint32_t tgx_tiled_gu = sl.n_subblocks * ceil_div(sl.sub_size / 8u, 8u);
        const uint32_t tgx_tiled_dn = 16u * ceil_div(sl.hidden_size / 128u, 8u);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_tiled_gateup_];
        [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:4];
        [enc setBytes:&ap length:sizeof(ap) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(tgx_tiled_gu, m_tiles, 2u)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        // Write FP16 SwiGLU into dead buf_normed_ (strictly disjoint from FP32 buf_mlp_gate_ -> zero TG race!)
        uint32_t n_elem = M * static_cast<uint32_t>(intermediate_size_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_tiled_down_];
        [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(tgx_tiled_dn, m_tiles, 1u)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();
        return;
    }

    if (M == 1u) {
        // Round 18: End-to-end FP16-X M=1 MLP (qmv_q4_g64_swiglu_f16x -> FP16 Split-K down_proj!)
        QmmParams qp{1u, lw.gate_proj.out_dim, lw.gate_proj.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmv_q4_swiglu_f16x_];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.w.buffer offset:lw.gate_proj.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.s.buffer offset:lw.gate_proj.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.b.buffer offset:lw.gate_proj.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.w.buffer offset:lw.up_proj.w.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.s.buffer offset:lw.up_proj.s.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.b.buffer offset:lw.up_proj.b.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:7];
        [enc setBytes:&qp length:sizeof(qp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(lw.gate_proj.out_dim, 8u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        barrier();
        encode_qmm_q4_f16x(enc, lw.down_proj, buf_mlp_gate_, 0, buf_hidden_, hid_off, 1u, true);
        barrier();
        return;
    } else if (M <= 8u) {
        // End-to-end FP16-X decode MLP: mma_qmm8_swiglu_f16 (405 GB/s) -> FP16 Split-K down_proj!
        QmmParams qp{M, lw.gate_proj.out_dim, lw.gate_proj.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_swiglu_f16_];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.w.buffer offset:lw.gate_proj.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.s.buffer offset:lw.gate_proj.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.b.buffer offset:lw.gate_proj.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.w.buffer offset:lw.up_proj.w.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.s.buffer offset:lw.up_proj.s.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)lw.up_proj.b.buffer offset:lw.up_proj.b.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:7];
        [enc setBytes:&qp length:sizeof(qp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(lw.gate_proj.out_dim, 16u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        if (next_norm != nullptr && lw.down_proj.out_dim == static_cast<uint32_t>(hidden_size_) && lw.down_proj.in_dim >= 6144u) {
            if (dual_out_f32 != nil) {
                encode_qmm_q4_f16x_and_dual_norm(enc, lw.down_proj, buf_mlp_gate_, 0, buf_hidden_, 0, M,
                                                 *next_norm, dual_out_f32, buf_normed_);
            } else {
                encode_qmm_q4_f16x_and_norm(enc, lw.down_proj, buf_mlp_gate_, 0, buf_hidden_, 0, M, *next_norm, buf_normed_);
            }
        } else {
            encode_qmm_q4_f16x(enc, lw.down_proj, buf_mlp_gate_, 0, buf_hidden_, 0, M, true);
            barrier();
        }
        return;
    } else if (M > 24u) {
        // High-throughput FP16-X prefill MLP path (13.48 TFLOPS, half L2 footprint!)
        encode_qmm_q4_f16x(enc, lw.gate_proj, buf_normed_, 0, buf_mlp_gate_, 0, M, false);
        encode_qmm_q4_f16x(enc, lw.up_proj,   buf_normed_, 0, buf_mlp_up_,   0, M, false);
        barrier();

        // Write FP16 SwiGLU into dead buf_normed_ (strictly disjoint from FP32 buf_mlp_gate_ -> zero TG race!)
        uint32_t n_elem = M * static_cast<uint32_t>(intermediate_size_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, lw.down_proj, buf_normed_, 0, buf_hidden_, 0, M, true);
        barrier();
        return;
    }

    encode_qmm_q4(enc, lw.gate_proj, buf_normed_, 0, buf_mlp_gate_, 0, M, false);
    encode_qmm_q4(enc, lw.up_proj,   buf_normed_, 0, buf_mlp_up_,   0, M, false);
    barrier();

    uint32_t n_elem = M * static_cast<uint32_t>(intermediate_size_);
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_];
    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
    [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    encode_qmm_q4(enc, lw.down_proj, buf_mlp_gate_, 0, buf_hidden_, 0, M, true);
    barrier();
}

void Qwen38Model::encode_layer_attn_only(
    id enc_id, int layer_idx, id hid_buf, uint32_t M, int start_pos, bool record_gdn, bool skip_input_norm
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    const LayerWeights& lw = layers_[layer_idx];
    const LayerState& st = states_[layer_idx];
    const bool use_f16 = (M > 24u) || (M <= 8u);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto qmm = [&](const Q4Weight& w, id xb, size_t xo, id yb, size_t yo, bool add_r) {
        if (use_f16) encode_qmm_q4_f16x(enc, w, xb, xo, yb, yo, M, add_r);
        else         encode_qmm_q4(enc, w, xb, xo, yb, yo, M, add_r);
    };

    // Input RMSNorm: hid_buf -> buf_normed_ (FP16 when M > 24; skipped when fused into qwen_ane_add_and_rmsnorm_f16!)
    if (!skip_input_norm) {
        NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
        [enc setBuffer:(id<MTLBuffer>)hid_buf offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    if (!lw.is_linear) {
        qmm(lw.q_proj, buf_normed_, 0, buf_qkv_, 0, false);
        qmm(lw.k_proj, buf_normed_, 0, buf_k_,   0, false);
        qmm(lw.v_proj, buf_normed_, 0, buf_v_,   0, false);
        barrier();

        uint32_t kv_len = static_cast<uint32_t>(start_pos + static_cast<int>(M));
        bool use_vsplit = (M <= 8u && kv_len >= 128u && head_dim_ == 256 && (num_heads_ % num_kv_heads_ == 0));
        bool use_gqa6 = use_vsplit && (num_heads_ == num_kv_heads_ * 6) && (kv_len >= 1024u);
        uint32_t splits = use_gqa6
            ? ((kv_len >= 32768u) ? 128u : ((kv_len >= 8192u) ? 64u : 32u))
            : (use_vsplit ? ((kv_len >= 4096u) ? 16u : ((kv_len >= 1024u) ? 8u : 4u))
                          : ((M == 1u && kv_len >= 256u) ? 16u : 1u));
        FullAttnParams fp{
            M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
            static_cast<uint32_t>(start_pos), kv_len, rope_theta_, rms_eps_,
            1.0f / std::sqrt(static_cast<float>(head_dim_)), splits, rope_delta_
        };

        const bool qkv = (st.k_sb != nil);
        const uint32_t M_pad = (M + 63u) & ~63u;
        const bool use_hier = (!use_vsplit) && (splits <= 1u) && use_f16 && (M >= 64u) && (M <= 1024u) &&
                              (head_dim_ == 256) && (start_pos >= 1024) && (num_heads_ % num_kv_heads_ == 0) &&
                              (static_cast<size_t>(M_pad) * num_heads_ * head_dim_ * 2u <= [(id<MTLBuffer>)buf_qkv_conv_ length]) &&
                              (static_cast<size_t>(M) * num_heads_ * head_dim_ * 4u <= [(id<MTLBuffer>)buf_split_o_ length]);
        if (cur_mrope_buf_ != nil) {
            if (use_hier) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_mrope_hier_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
                [enc setBytes:&fp length:sizeof(fp) atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M_pad, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_mrope_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
                [enc setBytes:&fp length:sizeof(fp) atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            if (qkv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_mrope_q_];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_mrope_];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        } else {
            if (use_hier) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_hier_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
                [enc setBytes:&fp length:sizeof(fp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M_pad, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
                [enc setBytes:&fp length:sizeof(fp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }

            if (qkv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_q_];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
        }
        barrier();

        if (use_vsplit) {
            if (use_gqa6) {
                id pso = nil;
                id q_buf = buf_q_;
                uint32_t tg_thr = 192u;
                if (M == 1u) {
                    pso = qkv ? pso_fa_gqa6_m1_q_ : pso_fa_gqa6_m1_;
                    tg_thr = 128u;
                } else if (M == 2u) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_p2_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_ * 2, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    barrier();
                    pso = qkv ? pso_fa_gqa6_m2_q_ : pso_fa_gqa6_m2_;
                    q_buf = buf_q_h_;
                    tg_thr = 128u;
                } else if (M <= 4u) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_p4_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_ * 3, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    barrier();
                    pso = qkv ? pso_fa_gqa6_m4_q_ : pso_fa_gqa6_m4_;
                    q_buf = buf_q_h_;
                    tg_thr = 128u;
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_h_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    barrier();
                    pso = qkv ? pso_fa_gqa6_v_q_ : pso_fa_gqa6_v_;
                    q_buf = buf_q_h_;
                    tg_thr = 192u;
                }
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
                [enc setBuffer:(id<MTLBuffer>)q_buf offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
                if (qkv) {
                    [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                } else {
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                }
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, splits, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_thr, 1, 1)];
            } else {
                uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
                if (qkv) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_vsplit1_q_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_vsplit1_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                }
                [enc dispatchThreadgroups:MTLSizeMake(num_pairs, splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_vsplit2_f16_ : pso_fa_vsplit2_)];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (splits > 1u) {
            if (qkv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split1_q_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                [enc setBytes:&fp length:sizeof(fp) atIndex:8];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split1_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, splits, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split2_];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else if (M >= 16u && head_dim_ == 256) {
            if (use_hier) {
                // Single-Head Hierarchical FP16-AMX FlashAttention-2 (1 head x 64 Q-rows, 4 barriers/tile, CHUNK_K=1024!)
                // q_h was already packed directly into buf_qkv_conv_ by qwen_full_attn_q_prep*_hier (zero extra pass/barrier!)
                uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                if (qkv) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_hier_q_f16_];
                    [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:7];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_hier_f16_];
                    [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                }
                [enc dispatchThreadgroups:MTLSizeMake(gqa, static_cast<uint32_t>(num_kv_heads_) * (M_pad / 64u), 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else if (qkv) {
                if (use_f16 && start_pos < 1024 && (num_heads_ % num_kv_heads_ == 0)) {
                    uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                    uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_flash_pair_q_f16_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 8u), num_pairs, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_prefill_flash_q_f16_ : pso_fa_prefill_flash_q_)];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 16u), num_heads_, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            } else if (use_f16 && start_pos < 1024 && (num_heads_ % num_kv_heads_ == 0)) {
                uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_flash_pair_f16_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 8u), num_pairs, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_prefill_flash_f16_ : pso_fa_prefill_flash_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 16u), num_heads_, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        } else {
            if (qkv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_attn_q_f16_ : pso_fa_attn_q_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_attn_f16_ : pso_fa_attn_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        barrier();

        qmm(lw.o_proj, buf_attn_out_, 0, hid_buf, 0, true);
        barrier();
        return;
    }

    // Linear attention (GatedDeltaNet) — only called during prefill (record_gdn == false)
    (void)record_gdn;
    id qkv_dst = buf_qkv_;
    id a_dst   = buf_a_;
    id b_dst   = buf_b_;

    qmm(lw.in_proj_qkv, buf_normed_, 0, qkv_dst, 0, false);
    qmm(lw.in_proj_z,   buf_normed_, 0, buf_z_,  0, false);
    if (M >= 2u && M <= 8u && lw.in_proj_a.out_dim == lw.in_proj_b.out_dim) {
        QmmParams qp{M, lw.in_proj_a.out_dim, lw.in_proj_a.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_in_proj_ab_f16x_];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.w.buffer offset:lw.in_proj_a.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.s.buffer offset:lw.in_proj_a.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.b.buffer offset:lw.in_proj_a.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.w.buffer offset:lw.in_proj_b.w.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.s.buffer offset:lw.in_proj_b.s.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.b.buffer offset:lw.in_proj_b.b.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:7];
        [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:8];
        [enc setBytes:&qp length:sizeof(qp) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(2u * ceil_div(lw.in_proj_a.out_dim, 8u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        qmm(lw.in_proj_b,   buf_normed_, 0, b_dst,   0, false);
        qmm(lw.in_proj_a,   buf_normed_, 0, a_dst,   0, false);
    }
    barrier();

    {
        id cs_out = st.conv_state;
        GdnPrepParams pp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_head_k_dim_),
                         static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_conv_dim_), rms_eps_};
        if (M <= 8u && lin_head_k_dim_ == lin_head_v_dim_) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_prework_fused_];
            [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)cs_out offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:8];
            [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:9];
            [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:10];
            [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:11];
            [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:12];
            [enc setBytes:&pp length:sizeof(pp) atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();
        } else {
            GdnConvParams cp{M, static_cast<uint32_t>(lin_conv_dim_)};
            bool par_conv = (M >= 32u);
            if (par_conv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_pf_];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, ceil_div(M, 8u), 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:3];
                [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)cs_out offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_gates_];
            [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:5];
            [enc setBytes:&pp length:sizeof(pp) atIndex:6];
            [enc dispatchThreads:MTLSizeMake(lin_num_v_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();

            if (par_conv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_upd_];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)cs_out offset:0 atIndex:1];
                [enc setBytes:&cp length:sizeof(cp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_qknorm_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:2];
            [enc setBytes:&pp length:sizeof(pp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(lin_num_k_heads_ * 2, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();
        }
    }

    {
        id ss_out = st.ssm_state;
        GdnRecParams rp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_num_v_heads_),
                        static_cast<uint32_t>(lin_head_k_dim_), static_cast<uint32_t>(lin_head_v_dim_),
                        static_cast<uint32_t>(lin_conv_dim_)};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_rec_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
        [enc setBytes:&rp length:sizeof(rp) atIndex:7];
        [enc setBuffer:(id<MTLBuffer>)ss_out offset:0 atIndex:8];
        [enc dispatchThreads:MTLSizeMake(32, lin_head_v_dim_ / 4, lin_num_v_heads_)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        barrier();
    }

    {
        GdnNormParams np{M, static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_head_v_dim_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_gdn_norm_f16_ : pso_gdn_norm_)];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.gdn_norm.buffer offset:lw.gdn_norm.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_gdn_normed_ offset:0 atIndex:3];
        [enc setBytes:&np length:sizeof(np) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(lin_num_v_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();
    }

    qmm(lw.out_proj, buf_gdn_normed_, 0, hid_buf, 0, true);
    barrier();
}

void Qwen38Model::encode_final_norm_and_mtp(
    id enc_id, id hid_buf, id tok_buf, size_t tok_off, uint32_t M, int start_pos,
    bool compute_logits, bool compute_argmax
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    if (mtp_loaded_) {
        size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(float);
        size_t dst_off = (start_pos > 0) ? row_bytes : 0;
        if (start_pos > 0) {
            uint32_t D = static_cast<uint32_t>(hidden_size_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(D) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        {
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:dst_off atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        size_t last_hid_off = dst_off + static_cast<size_t>(M - 1u) * row_bytes;
        {
            uint32_t D = static_cast<uint32_t>(hidden_size_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:last_hid_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(D) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        if (compute_logits) {
            // Round 31: Ultra-bandwidth FP16-X M=1 lm_head on prefill finalize!
            size_t last_off = static_cast<size_t>(M - 1u) * row_bytes;
            NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:last_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            encode_qmm_q4_f16x(enc, lm_head_, buf_normed_, 0, buf_logits_, 0, 1u, false);
            if (compute_argmax) {
                barrier();
                uint32_t V = static_cast<uint32_t>(vocab_size_);
                uint32_t n_tg = 128u;
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
                [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
                [enc setBytes:&V length:sizeof(V) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_out_ offset:0 atIndex:1];
                [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            barrier();
        }
        uint32_t mtp_total = (start_pos == 0) ? (M > 1u ? M - 1u : 0u) : M;
        size_t mtp_tok_base = tok_off + ((start_pos == 0) ? sizeof(int) : 0);
        int mtp_pos_base = (start_pos == 0) ? 0 : (start_pos - 1);
        uint32_t done = 0;
        while (done < mtp_total) {
            uint32_t sub = std::min<uint32_t>(static_cast<uint32_t>(chunk_size_), mtp_total - done);
            encode_mtp_kv_prime(enc, tok_buf, mtp_tok_base + static_cast<size_t>(done) * sizeof(int),
                                buf_mtp_in_hid_, static_cast<size_t>(done) * row_bytes,
                                static_cast<int>(sub), mtp_pos_base + static_cast<int>(done));
            barrier();
            done += sub;
        }
        return;
    }

    if (compute_logits) {
        size_t last_off = static_cast<size_t>(M - 1u) * hidden_size_ * sizeof(float);
        {
            NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:last_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        encode_qmm_q4_f16x(enc, lm_head_, buf_normed_, 0, buf_logits_, 0, 1u, false);
        if (compute_argmax) {
            barrier();
            uint32_t V = static_cast<uint32_t>(vocab_size_);
            uint32_t n_tg = 128u;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
            [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBytes:&V length:sizeof(V) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_out_ offset:0 atIndex:1];
            [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
}

void Qwen38Model::encode_ane_tail_mlp(id enc_id, const LayerWeights& lw, id hid_buf, size_t tail_hid_off, uint32_t gm) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    const TensorView& pan = (lw.ane_share_idx >= 0) ? lw.post_attn_norm_share : lw.post_attn_norm;
    NormParams np{gm, static_cast<uint32_t>(hidden_size_), rms_eps_};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
    [enc setBuffer:(id<MTLBuffer>)hid_buf offset:tail_hid_off atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)pan.buffer offset:pan.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
    [enc setBytes:&np length:sizeof(np) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(gm, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    if (lw.ane_share_idx >= 0 && ane_share_ && ane_share_->has_layer(lw.ane_share_idx)) {
        const AneShareLayer& sl = ane_share_->layer(lw.ane_share_idx);
        if (gm <= 64u) {
            // Fast M<=8 micro-batched hf4/lut4 SwiGLU + Split-K down_proj (376 GB/s -> ~23 ms per 8 tok, ~185 ms for gm=64!)
            const uint32_t tgx_gu_m1 = ceil_div(sl.intermediate_size / 8u, 4u);
            const uint32_t tgx_gu_m8 = sl.intermediate_size / 16u;
            const uint32_t tgx_dn_sk = 16u * ceil_div(sl.hidden_size / 128u, 4u);
            const uint32_t sk_down   = 2u * sl.n_subblocks;
            for (uint32_t m0 = 0u; m0 < gm; m0 += 8u) {
                uint32_t sub = std::min<uint32_t>(8u, gm - m0);
                AneShareParams ap = make_ane_share_params(sl, sub);
                size_t x_off = static_cast<size_t>(m0) * hidden_size_ * sizeof(uint16_t);
                size_t y_off = tail_hid_off + static_cast<size_t>(m0) * hidden_size_ * sizeof(float);
                if (sub == 1u) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_m1_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:x_off atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_m1_sk_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:x_off atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_gu_m8, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_sk4_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(tgx_dn_sk, 1, sk_down)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                }
                QmmSkParams sp{sub, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk_down};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
                [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)hid_buf offset:y_off atIndex:1];
                [enc setBytes:&sp length:sizeof(sp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(sub * static_cast<uint32_t>(hidden_size_), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                if (m0 + 8u < gm) barrier();
            }
        } else if (gm <= 160u && pso_ane_hf4_swiglu_tiled32_ != nil && pso_ane_hf4_down_tiled32_sk_ != nil) {
            // Co-Exec / Short-Prompt Fused BM=32 double-buffered SwiGLU + Split-K down_proj!
            // Binds buf_mlp_up_ (34 MiB, unused by fused swiglu_tiled32!) as partY -> NEVER buf_sk_part_!
            AneShareParams ap = make_ane_share_params(sl, gm);
            const uint32_t m_tiles  = ceil_div(gm, 32u);
            const uint32_t tgx_gu32 = sl.n_subblocks * ceil_div(sl.sub_size / 8u, 4u);
            const uint32_t tgx_dn32 = 16u * ceil_div(sl.hidden_size / 128u, 4u);
            const uint32_t n_q      = 2u * sl.n_subblocks;
            const uint32_t want_sk  = (gm <= 64u) ? 4u : 2u;
            const size_t part_cap   = [(id<MTLBuffer>)buf_mlp_up_ length];
            uint32_t sk = 1u;
            for (uint32_t cand : {want_sk, 2u, 1u}) {
                if (cand <= n_q && (n_q % cand == 0u) &&
                    static_cast<size_t>(cand) * gm * hidden_size_ * sizeof(float) <= part_cap) {
                    sk = cand;
                    break;
                }
            }

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_tiled32_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_gu32, m_tiles, 1u)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_tiled32_sk_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_dn32, m_tiles, sk)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            QmmSkParams sp{gm, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, sk};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:tail_hid_off atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(gm * static_cast<uint32_t>(hidden_size_), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            AneShareParams ap = make_ane_share_params(sl, gm);
            const uint32_t m_tiles = ceil_div(gm, 64u);
            const uint32_t tgx_tiled_gu = sl.n_subblocks * ceil_div(sl.sub_size / 8u, 8u);
            const uint32_t tgx_tiled_dn = 16u * ceil_div(sl.hidden_size / 128u, 8u);

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_tiled_gateup_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:4];
            [enc setBytes:&ap length:sizeof(ap) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_tiled_gu, m_tiles, 2u)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            // Write FP16 SwiGLU into dead buf_normed_ (strictly disjoint from FP32 buf_mlp_gate_ -> zero TG race!)
            uint32_t n_elem = gm * static_cast<uint32_t>(intermediate_size_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_tiled_down_];
            [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:tail_hid_off atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(tgx_tiled_dn, m_tiles, 1u)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    } else {
        encode_qmm_q4_f16x(enc, lw.gate_proj, buf_normed_, 0, buf_mlp_gate_, 0, gm, false);
        encode_qmm_q4_f16x(enc, lw.up_proj,   buf_normed_, 0, buf_mlp_up_,   0, gm, false);
        barrier();
        uint32_t n_elem = gm * static_cast<uint32_t>(intermediate_size_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_up_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&n_elem length:sizeof(n_elem) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n_elem, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        encode_qmm_q4_f16x(enc, lw.down_proj, buf_normed_, 0, hid_buf, tail_hid_off, gm, true);
    }
}

void Qwen38Model::forward_ane_chunk(const int* tokens, int n_chunk, bool is_last, bool compute_argmax,
                                    const int* pos3d, const float* const* vis_ptrs, int df_capture_from_pos) {
    // Two-Half Zero-Copy GPU || ANE Pipelined Prefill:
    //   Half A (slot 0, buf_hidden_)     & Half B (slot 1, buf_hidden_alt_)
    //   While GPU runs Attention/GDN + IOSurface pack on half s, ANE runs 64-layer MLP on half (1 - s)!
    const int blk = ane_->block_rows(); // 256
    const int ane_max_half = AneMlpEngine::MAX_SUBS * blk; // 512
    const int n_pieces = ane_->num_pieces();
    const int single_tail_max = 64; // <= 64 GPU tail rows absorbed onto Half B ONLY (keeps Half A exact 256/512!)
    const int gpu_cores = ctx_.estimated_gpu_cores();
    int mA = 0, mB = 0;
    int ane_m[2] = { 0, 0 };
    int gpu_m[2] = { 0, 0 };
    if (gpu_cores >= 32 && pos_ < 4096 && n_chunk <= blk + 128) {
        // Short-prompt asymmetric qs={1,0} split (176..384 on Max/Ultra):
        // Keep mB=64 for 176..320 (1 Attn+GDN 64-tile + 185 ms M=8 AMX MLP = ~257 ms, 100% hidden inside ~355 ms ANE window -> N=256 ~470 tok/s!)
        if (n_chunk <= blk + 64) {
            mB = std::min(64, n_chunk / 2);
            mA = n_chunk - mB;
        } else {
            // 321..384: fill ANE Half A to exact 256, put 65..128 on GPU Half B via fused tiled32!
            mA = blk;
            mB = n_chunk - blk;
        }
        ane_m[0] = mA; ane_m[1] = 0;
        gpu_m[0] = 0;  gpu_m[1] = mB;
    } else if (gpu_cores >= 18 && gpu_cores < 32 && pos_ < 4096 && n_chunk <= blk + 64) {
        // Pro-tier asymmetric qs={1,0} split (96..320): smaller 64-tok GPU Half B fits inside ANE window!
        mB = std::min(64, n_chunk / 2);
        mA = n_chunk - mB;
        ane_m[0] = mA; ane_m[1] = 0;
        gpu_m[0] = 0;  gpu_m[1] = mB;
    } else if (n_chunk <= 2 * blk) {
        // <= 512: qs={1,1}, zero GPU tail
        mA = n_chunk / 2;
        mB = n_chunk - mA;
        ane_m[0] = mA; ane_m[1] = mB;
    } else if (n_chunk <= 2 * blk + single_tail_max) {
        // 513..576 (e.g. 512+chat = 528): keep Half A exact 256 (0 extra 64-row M-tile!), put 1..64 tail on Half B ONLY!
        mA = blk;
        mB = n_chunk - blk;
        ane_m[0] = blk; ane_m[1] = blk;
        gpu_m[0] = 0;   gpu_m[1] = mB - blk;
    } else if (n_chunk <= 3 * blk) {
        // 577..768: asymmetric 3-sub-block split qs={1,2} (25% less ANE work & faster L0 Half-A pipeline start!)
        mA = blk;
        mB = n_chunk - blk;
        ane_m[0] = mA; ane_m[1] = mB;
    } else if (n_chunk <= 3 * blk + single_tail_max) {
        // 769..832: qs={1,2} ({256,512} on ANE) + 1..64 GPU tail on Half B ONLY (Half A stays exact 256!)
        mA = blk;
        mB = n_chunk - blk;
        ane_m[0] = blk;
        ane_m[1] = 2 * blk;
        gpu_m[0] = 0;
        gpu_m[1] = mB - 2 * blk;
    } else if (n_chunk <= 2 * ane_max_half) {
        // 833..1024: qs={2,2}, zero GPU tail
        mA = n_chunk / 2;
        mB = n_chunk - mA;
        ane_m[0] = mA; ane_m[1] = mB;
    } else if (n_chunk <= 2 * ane_max_half + single_tail_max) {
        // 1025..1088 (e.g. 4k+chat tail = 1040): keep Half A exact 512 (0 extra 64-row M-tile & FA-2 Q-tile!),
        // put 1..64 tail on Half B ONLY (mB <= 576 fits C_attn)!
        mA = ane_max_half;
        mB = n_chunk - ane_max_half;
        ane_m[0] = ane_max_half; ane_m[1] = ane_max_half;
        gpu_m[0] = 0;            gpu_m[1] = mB - ane_max_half;
    } else {
        // 1089..1152: qs={2,2} with 512/half on ANE + split 65..128 tail across both halves (<= 64/half)
        mA = n_chunk / 2;
        mB = n_chunk - mA;
        ane_m[0] = std::min(mA, ane_max_half);
        ane_m[1] = std::min(mB, ane_max_half);
        gpu_m[0] = mA - ane_m[0];
        gpu_m[1] = mB - ane_m[1];
    }
    const int ms[2] = { mA, mB };
    const int qs[2] = { (ane_m[0] > 0) ? ((ane_m[0] + blk - 1) / blk) : 0,
                        (ane_m[1] > 0) ? ((ane_m[1] + blk - 1) / blk) : 0 };
    const int pos_off[2] = { pos_, pos_ + mA };
    const size_t tok_off[2] = { 0, static_cast<size_t>(mA) * sizeof(int) };
    id hid_bufs[2] = { buf_hidden_, buf_hidden_alt_ };

    // DFlash 2 per-half capture ranges overlapping [df_capture_from_pos, INT_MAX) (0 work on early chunks of long prefills!)
    int df_cap_src[2] = { 0, 0 };
    int df_cap_n[2]   = { 0, 0 };
    int df_cap_dst[2] = { 0, 0 };
    int df_cap_pos[2] = { 0, 0 };
    if (dflash_loaded_) {
        for (int s = 0; s < 2; ++s) {
            int c0 = std::max(pos_off[s], df_capture_from_pos);
            int c1 = pos_off[s] + ms[s];
            if (c1 > c0) {
                df_cap_src[s] = c0 - pos_off[s];
                df_cap_n[s]   = c1 - c0;
                df_cap_pos[s] = c0;
            }
        }
        df_cap_dst[0] = 0;
        df_cap_dst[1] = df_cap_n[0];
    }

    std::memcpy([(id<MTLBuffer>)buf_token_ids_ contents], tokens, static_cast<size_t>(n_chunk) * sizeof(int));
    if (use_cpu_embed_) {
        cpu_embed_.gather_f32(tokens, ms[0], reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_ contents]));
        if (ms[1] > 0) {
            cpu_embed_.gather_f32(tokens + mA, ms[1], reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_alt_ contents]));
        }
    }

    // Optional Multimodal 3D M-RoPE + Vision Embedding Scatter setup (zero overhead on pure text!)
    size_t mrope_off[2] = { 0, static_cast<size_t>(mA) * 3 * sizeof(int) };
    uint32_t vis_count[2] = { 0, 0 };
    size_t vis_emb_off[2] = { 0, 0 };
    size_t vis_rows_off[2] = { 0, 0 };
    if (pos3d != nullptr || vis_ptrs != nullptr) {
        size_t max_c = 2ull * (static_cast<size_t>(ane_max_half) + 64ull);
        if (!buf_mrope_pos_)  buf_mrope_pos_  = ctx_.alloc_shared(max_c * 3 * sizeof(int));
        if (!buf_vis_embeds_) buf_vis_embeds_ = ctx_.alloc_shared(max_c * hidden_size_ * sizeof(float));
        if (!buf_vis_rows_)   buf_vis_rows_   = ctx_.alloc_shared(max_c * sizeof(uint32_t));
        if (pos3d != nullptr) {
            std::memcpy([(id<MTLBuffer>)buf_mrope_pos_ contents], pos3d, static_cast<size_t>(n_chunk) * 3 * sizeof(int));
        }
        if (vis_ptrs != nullptr) {
            float* dst_emb = reinterpret_cast<float*>([(id<MTLBuffer>)buf_vis_embeds_ contents]);
            uint32_t* dst_rows = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_vis_rows_ contents]);
            uint32_t k = 0;
            for (int s = 0; s < 2; ++s) {
                vis_emb_off[s]  = static_cast<size_t>(k) * hidden_size_ * sizeof(float);
                vis_rows_off[s] = static_cast<size_t>(k) * sizeof(uint32_t);
                int base = (s == 0) ? 0 : mA;
                for (int r = 0; r < ms[s]; ++r) {
                    const float* vp = vis_ptrs[base + r];
                    if (vp != nullptr) {
                        std::memcpy(dst_emb + static_cast<size_t>(k) * hidden_size_, vp,
                                    static_cast<size_t>(hidden_size_) * sizeof(float));
                        dst_rows[k] = static_cast<uint32_t>(r);
                        ++k;
                        ++vis_count[s];
                    }
                }
            }
        }
    }

    auto encode_ane_add = [&](id<MTLComputeCommandEncoder> enc, int s) {
        int m_tot = ane_m[s];
        for (int sub = 0; sub < qs[s]; ++sub) {
            int row0 = sub * blk;
            uint32_t sub_m = static_cast<uint32_t>(std::min(blk, m_tot - row0));
            size_t hid_off = static_cast<size_t>(row0) * hidden_size_ * sizeof(float);
            AneAddParams ap{sub_m, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(n_pieces)};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_add_];
            [enc setTexture:(id<MTLTexture>)ane_->output_texture(s, sub, 0) atIndex:0];
            [enc setTexture:(id<MTLTexture>)ane_->output_texture(s, sub, n_pieces > 1 ? 1 : 0) atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:hid_off atIndex:0];
            [enc setBytes:&ap length:sizeof(ap) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(hidden_size_, sub_m, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_ane_add_and_norm = [&](id<MTLComputeCommandEncoder> enc, int s, int L) {
        const LayerWeights& lw = layers_[L];
        int m_tot = ane_m[s];
        bool use_f16 = (ms[s] > 24);
        for (int sub = 0; sub < qs[s]; ++sub) {
            int row0 = sub * blk;
            uint32_t sub_m = static_cast<uint32_t>(std::min(blk, m_tot - row0));
            size_t hid_off  = static_cast<size_t>(row0) * hidden_size_ * sizeof(float);
            size_t norm_off = static_cast<size_t>(row0) * hidden_size_ * (use_f16 ? sizeof(uint16_t) : sizeof(float));
            AneAddNormParams ap{sub_m, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(n_pieces), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_ane_add_norm_f16_ : pso_ane_add_norm_)];
            [enc setTexture:(id<MTLTexture>)ane_->output_texture(s, sub, 0) atIndex:0];
            [enc setTexture:(id<MTLTexture>)ane_->output_texture(s, sub, n_pieces > 1 ? 1 : 0) atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:hid_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:norm_off atIndex:2];
            [enc setBytes:&ap length:sizeof(ap) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(sub_m, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        // Co-execution GPU tail rows already had layer (L-1) MLP added in place -> just RMSNorm them!
        if (gpu_m[s] > 0) {
            size_t hid_off  = static_cast<size_t>(ane_m[s]) * hidden_size_ * sizeof(float);
            size_t norm_off = static_cast<size_t>(ane_m[s]) * hidden_size_ * (use_f16 ? sizeof(uint16_t) : sizeof(float));
            NormParams np{static_cast<uint32_t>(gpu_m[s]), static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
            [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:hid_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:norm_off atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<uint32_t>(gpu_m[s]), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    const bool prof = (std::getenv("ANE_PROF") != nullptr);
    double gpu_wait_ms = 0.0, ane_wait_ms = 0.0;
    bool slot_inflight[2] = { false, false };

    auto encode_step_cb = [&](int L, int s) -> id<MTLCommandBuffer> {
        const LayerWeights& lw = layers_[L];
        id inv1_buf = ane_->inv1_buffer(L);
        uint32_t has_inv1 = (inv1_buf != nil) ? 1u : 0u;
        id inv1_bind = (inv1_buf != nil) ? inv1_buf : buf_normed_;

        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

        bool skip_norm = false;
        if (L == 0) {
            if (!use_cpu_embed_) {
                EmbedParams ep{static_cast<uint32_t>(ms[s]), static_cast<uint32_t>(hidden_size_), rms_eps_};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_];
                [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:tok_off[s] atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:0 atIndex:4];
                [enc setBytes:&ep length:sizeof(ep) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(hidden_size_, ms[s], 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }
            if (vis_count[s] > 0) {
                ScatterEmbedParams sp{vis_count[s], static_cast<uint32_t>(hidden_size_)};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_scatter_embeds_];
                [enc setBuffer:(id<MTLBuffer>)buf_vis_embeds_ offset:vis_emb_off[s] atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_vis_rows_ offset:vis_rows_off[s] atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:0 atIndex:2];
                [enc setBytes:&sp length:sizeof(sp) atIndex:3];
                [enc dispatchThreads:MTLSizeMake(hidden_size_, vis_count[s], 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }
        } else {
            // Fused: add layer (L-1)'s ANE MLP output + compute layer L's input RMSNorm in ONE pass!
            encode_ane_add_and_norm(enc, s, L);
            barrier();
            if (dflash_loaded_ && df_cap_n[s] > 0) {
                int df_slot = dflash_->target_capture_slot(L - 1);
                if (df_slot >= 0) {
                    encode_dflash_capture(enc, hid_bufs[s], df_cap_src[s], df_cap_dst[s], df_cap_n[s], df_slot);
                    barrier();
                }
            }
            skip_norm = true;
        }

        // GPU Attention / GatedDeltaNet for layer L on half s (overlapped with ANE on half 1-s!)
        cur_mrope_buf_ = (pos3d != nullptr) ? buf_mrope_pos_ : nil;
        cur_mrope_off_ = mrope_off[s];
        encode_layer_attn_only(enc, L, hid_bufs[s], static_cast<uint32_t>(ms[s]), pos_off[s], false, skip_norm);
        cur_mrope_buf_ = nil;
        cur_mrope_off_ = 0;

        // Fused post_attn RMSNorm + [x'/r | r | 0] FP16 pack directly into ANE input IOSurface!
        int m_tot = ane_m[s];
        for (int sub = 0; sub < qs[s]; ++sub) {
            int row0 = sub * blk;
            uint32_t sub_m = static_cast<uint32_t>(std::min(blk, m_tot - row0));
            size_t hid_off = static_cast<size_t>(row0) * hidden_size_ * sizeof(float);
            AnePackParams pp{
                sub_m, static_cast<uint32_t>(blk), static_cast<uint32_t>(hidden_size_),
                static_cast<uint32_t>(ane_->packed_width()), has_inv1, rms_eps_
            };
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_pack_];
            [enc setBuffer:(id<MTLBuffer>)hid_bufs[s] offset:hid_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.post_attn_norm.buffer offset:lw.post_attn_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)inv1_bind offset:0 atIndex:2];
            [enc setBytes:&pp length:sizeof(pp) atIndex:3];
            [enc setTexture:(id<MTLTexture>)ane_->input_texture(s, sub) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(blk, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Heterogeneous GPU+ANE Co-Execution: run trailing gpu_m[s] MLP rows on GPU reading single-copy model.hwx!
        if (gpu_m[s] > 0) {
            uint32_t gm = static_cast<uint32_t>(gpu_m[s]);
            size_t tail_hid_off = static_cast<size_t>(ane_m[s]) * hidden_size_ * sizeof(float);
            encode_ane_tail_mlp(enc, lw, hid_bufs[s], tail_hid_off, gm);
        }
        [enc endEncoding];
        return cb;
    };

    // Zero-bubble CPU/GPU/ANE pipeline with Cross-Slot Async ANE Ring Pipelining:
    //   - Encode next CB on CPU while GPU executes cur_cb (hides 100% of CPU encoding latency)
    //   - When ANE-bound, submit slot s BEFORE waiting on slot (1-s) so ANE hardware ring never drains!
    //   - When GPU-bound (slot 1-s already done), commit next GPU CB BEFORE submitting slot s!
    @autoreleasepool {
        const int total_steps = num_layers_ * 2;
        id<MTLCommandBuffer> cur_cb = encode_step_cb(0, 0);
        auto tg0 = std::chrono::steady_clock::now();
        [cur_cb commit];

        for (int step = 0; step < total_steps; ++step) {
            int L = step >> 1;
            int s = step & 1;
            int other = 1 - s;

            // Pre-encode NEXT step on CPU while GPU executes cur_cb (hides 100% of CPU encoding latency!)
            id<MTLCommandBuffer> next_cb = nil;
            if (step + 1 < total_steps) {
                next_cb = encode_step_cb((step + 1) >> 1, (step + 1) & 1);
            }

            [cur_cb waitUntilCompleted];
            if (cur_cb.error) {
                if (slot_inflight[0]) { ane_->wait_slot(0); slot_inflight[0] = false; }
                if (slot_inflight[1]) { ane_->wait_slot(1); slot_inflight[1] = false; }
                throw std::runtime_error(std::string("Metal ANE-pipeline GPU error: ") +
                                         [[cur_cb.error localizedDescription] UTF8String]);
            }
            auto tg1 = std::chrono::steady_clock::now();
            if (prof) gpu_wait_ms += std::chrono::duration<double, std::milli>(tg1 - tg0).count();

            if (slot_inflight[other] && ane_->is_slot_done(other)) {
                // GPU-bound: slot `other` already finished on ANE -> acquire & resume GPU immediately, then submit slot `s`!
                ane_->wait_slot(other);
                slot_inflight[other] = false;
                if (next_cb != nil) {
                    tg0 = std::chrono::steady_clock::now();
                    [next_cb commit];
                    cur_cb = next_cb;
                }
                if (qs[s] > 0) {
                    ane_->submit_slot(L, s, qs[s]);
                    slot_inflight[s] = true;
                }
            } else {
                // ANE-bound (or step 0): queue slot `s` into ANE ring FIRST (zero drain gap!), then wait on slot `other`!
                if (qs[s] > 0) {
                    ane_->submit_slot(L, s, qs[s]);
                    slot_inflight[s] = true;
                }
                if (slot_inflight[other]) {
                    auto ta0 = std::chrono::steady_clock::now();
                    ane_->wait_slot(other);
                    slot_inflight[other] = false;
                    if (prof) ane_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta0).count();
                }
                if (next_cb != nil) {
                    tg0 = std::chrono::steady_clock::now();
                    [next_cb commit];
                    cur_cb = next_cb;
                }
            }
        }
    }

    // Overlap Half 0 finalize (L63 add + Half 0 MTP priming) on GPU WHILE L63 Half 1 is still running on ANE!
    // Half 0 buffers (buf_hidden_, output_texture(0,*)) are completely disjoint from Half 1 (ANE in-flight).
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

        if (qs[0] > 0) {
            encode_ane_add(enc, 0);
        }
        if (mtp_loaded_) {
            if (qs[0] > 0) barrier();
            encode_final_norm_and_mtp(enc, buf_hidden_, buf_token_ids_, 0,
                                      static_cast<uint32_t>(mA), pos_, false, false);
        }
        if (dflash_loaded_ && df_cap_n[0] > 0) {
            int slot_last = dflash_->target_capture_slot(num_layers_ - 1);
            if (slot_last >= 0) {
                if (qs[0] > 0) barrier();
                encode_dflash_capture(enc, buf_hidden_, df_cap_src[0], df_cap_dst[0], df_cap_n[0], slot_last);
            }
            barrier();
            encode_dflash_context_commit(enc, df_cap_dst[0], df_cap_n[0], df_cap_pos[0]);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            if (slot_inflight[1]) { ane_->wait_slot(1); slot_inflight[1] = false; }
            throw std::runtime_error(std::string("Metal ANE finalize half0 error: ") +
                                     [[cb.error localizedDescription] UTF8String]);
        }
    }

    if (slot_inflight[1]) {
        auto ta0 = std::chrono::steady_clock::now();
        ane_->wait_slot(1);
        slot_inflight[1] = false;
        if (prof) ane_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta0).count();
    }
    if (prof) {
        std::fprintf(stderr, "[ane_prof] chunk=%d (mA=%d[a%d,g%d],mB=%d[a%d,g%d]): gpu_cb=%.1f ms | ane_extra_wait=%.1f ms\n",
                     n_chunk, mA, ane_m[0], gpu_m[0], mB, ane_m[1], gpu_m[1], gpu_wait_ms, ane_wait_ms);
    }

    // Finalize Half 1 (L63 add + final RMSNorm + lm_head + Half 1 MTP/DFlash commit)
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

        if (qs[1] > 0) {
            encode_ane_add(enc, 1);
        }
        if (mtp_loaded_ || is_last) {
            if (qs[1] > 0) barrier();
            encode_final_norm_and_mtp(enc, buf_hidden_alt_, buf_token_ids_, tok_off[1],
                                      static_cast<uint32_t>(mB), pos_ + mA, is_last, is_last && compute_argmax);
        }
        if (dflash_loaded_ && df_cap_n[1] > 0) {
            barrier();
            int slot_last = dflash_->target_capture_slot(num_layers_ - 1);
            if (slot_last >= 0) {
                encode_dflash_capture(enc, buf_hidden_alt_, df_cap_src[1], df_cap_dst[1], df_cap_n[1], slot_last);
                barrier();
            }
            encode_dflash_context_commit(enc, df_cap_dst[1], df_cap_n[1], df_cap_pos[1]);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal ANE finalize half1 error: ") +
                                     [[cb.error localizedDescription] UTF8String]);
        }
    }

    pos_ += n_chunk;
    if (dflash_loaded_) df_kv_len_ = pos_;
}

void Qwen38Model::encode_full_attn_layer(
    id enc_id, const LayerWeights& lw, id k_cache, id v_cache, uint32_t M, int start_pos,
    id k_sb, id v_sb, bool skip_input_norm, const TensorView* next_norm, id dual_out_f32,
    id k_bounds, int* quest_built_pages, bool is_mtp, bool use_tree_buf
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    const bool use_f16 = (M > 24u) || (M <= 8u);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto bind_tree = [&](uint32_t idx) {
        if (use_tree_buf) [enc setBuffer:(id<MTLBuffer>)buf_tree_params_ offset:0 atIndex:idx];
        else              [enc setBytes:&cur_tree_ length:sizeof(cur_tree_) atIndex:idx];
    };
    auto qmm = [&](const Q4Weight& w, id xb, size_t xo, id yb, size_t yo, bool add_r) {
        if (use_f16) encode_qmm_q4_f16x(enc, w, xb, xo, yb, yo, M, add_r);
        else         encode_qmm_q4(enc, w, xb, xo, yb, yo, M, add_r);
    };

    // Input RMSNorm (skipped on M=2..8 when fused into previous layer's down_proj Split-K reduce!)
    if (!skip_input_norm) {
        NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    // Q+gate, K, V concurrent
    qmm(lw.q_proj, buf_normed_, 0, buf_qkv_, 0, false);
    qmm(lw.k_proj, buf_normed_, 0, buf_k_,   0, false);
    qmm(lw.v_proj, buf_normed_, 0, buf_v_,   0, false);
    barrier();

    uint32_t kv_len = static_cast<uint32_t>(start_pos + static_cast<int>(M));
    bool use_vsplit = ((M <= 8u || (cur_tree_.is_tree != 0u && M <= 16u)) && kv_len >= 128u && head_dim_ == 256 && (num_heads_ % num_kv_heads_ == 0));
    bool use_gqa6 = use_vsplit && (num_heads_ == num_kv_heads_ * 6) && (kv_len >= 1024u);
    // Full dense attention on 1-layer MTP KV4 preserves 100% long-context (8k-128k) acceptance rate (~0.35 ms/step at 128k).
    uint32_t win_sink   = 0u;
    uint32_t win_recent = 0u;
    uint32_t total_pages = ceil_div(kv_len, QUEST_PAGE_SIZE);
    uint32_t budget_pages = (sparse_kv_budget_ > 0)
        ? std::clamp<uint32_t>(static_cast<uint32_t>(sparse_kv_budget_) / QUEST_PAGE_SIZE,
                               QUEST_SINK_PAGES + QUEST_RECENT_PAGES + 16u, QUEST_MAX_ACTIVE_PAGES)
        : 0u;
    bool use_quest = (!is_mtp) && use_gqa6 && (sparse_kv_budget_ > 0) && (k_bounds != nil) && (quest_built_pages != nullptr) &&
                     (total_pages > budget_pages + 64u) && (total_pages <= QUEST_SCORE_STRIDE);
    uint32_t splits = use_quest
        ? ((budget_pages > 512u) ? 128u : 64u)
        : (use_gqa6
            ? ((kv_len >= 32768u) ? 128u : ((kv_len >= 8192u) ? 64u : 32u))
            : (use_vsplit ? ((kv_len >= 4096u) ? 16u : ((kv_len >= 1024u) ? 8u : 4u))
                          : ((M == 1u && kv_len >= 256u) ? 16u : 1u)));
    FullAttnParams fp{
        M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
        static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
        static_cast<uint32_t>(start_pos), kv_len, rope_theta_, rms_eps_,
        1.0f / std::sqrt(static_cast<float>(head_dim_)), splits, rope_delta_,
        win_sink, win_recent
    };

    const bool qkv = (k_sb != nil);
    const bool is_tree = (cur_tree_.is_tree != 0u);
    const uint32_t M_pad = (M + 63u) & ~63u;
    const bool use_hier = (!use_vsplit) && (!is_tree) && (splits <= 1u) && use_f16 && (M >= 64u) && (M <= 1024u) &&
                          (head_dim_ == 256) && (start_pos >= 1024) && (num_heads_ % num_kv_heads_ == 0) &&
                          (static_cast<size_t>(M_pad) * num_heads_ * head_dim_ * 2u <= [(id<MTLBuffer>)buf_qkv_conv_ length]) &&
                          (static_cast<size_t>(M) * num_heads_ * head_dim_ * 4u <= [(id<MTLBuffer>)buf_split_o_ length]);
    if (cur_mrope_buf_ != nil) {
        if (use_hier) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_mrope_hier_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBytes:&fp length:sizeof(fp) atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M_pad, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_mrope_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBytes:&fp length:sizeof(fp) atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_mrope_q_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:6];
            [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_mrope_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)cur_mrope_buf_ offset:cur_mrope_off_ atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    } else if (is_tree) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_tree_];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
        [enc setBytes:&fp length:sizeof(fp) atIndex:4];
        bind_tree(5u);
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(is_mtp ? pso_mtp_kv_prep_tree_ : pso_fa_kv_prep_q_tree_)];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:6];
            [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            bind_tree(8u);
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_tree_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            bind_tree(6u);
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    } else {
        if (use_hier) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_hier_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBytes:&fp length:sizeof(fp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M_pad, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBytes:&fp length:sizeof(fp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(is_mtp ? pso_mtp_kv_prep_ : pso_fa_kv_prep_q_)];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:6];
            [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
    }
    barrier();

    if (use_vsplit) {
        if (use_gqa6) {
            if (use_quest) {
                uint32_t recent_start_page = total_pages - QUEST_RECENT_PAGES;
                QuestParams qp{
                    kv_len, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_), M,
                    static_cast<uint32_t>(*quest_built_pages), recent_start_page, total_pages,
                    QUEST_SINK_PAGES, recent_start_page, budget_pages, QUEST_MAX_ACTIVE_PAGES, QUEST_SCORE_STRIDE
                };
                if (qp.end_page > qp.start_page) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_build_bounds_];
                    [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:0];
                    if (qkv) {
                        [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:1];
                        [enc setBuffer:(id<MTLBuffer>)k_bounds offset:0 atIndex:2];
                        [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                    } else {
                        [enc setBuffer:(id<MTLBuffer>)k_bounds offset:0 atIndex:1];
                        [enc setBytes:&qp length:sizeof(qp) atIndex:2];
                    }
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, qp.end_page - qp.start_page, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();
                    *quest_built_pages = static_cast<int>(qp.end_page);
                }
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_score_pages_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)k_bounds offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_quest_scores_ offset:0 atIndex:2];
                [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, 16, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_select_topk_];
                [enc setBuffer:(id<MTLBuffer>)buf_quest_scores_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:2];
                [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }

            id pso = nil;
            id q_buf = buf_q_;
            uint32_t tg_thr = 192u;
            uint32_t z_tiles = 1u;
            if (M == 1u) {
                pso = use_quest
                    ? (is_tree ? pso_fa_gqa6_m1_sp_tree_ : pso_fa_gqa6_m1_sp_)
                    : (qkv ? (is_mtp ? (is_tree ? pso_mtp_gqa6_m1_tree_ : pso_mtp_gqa6_m1_)
                                     : (is_tree ? pso_fa_gqa6_m1_q_tree_ : pso_fa_gqa6_m1_q_))
                           : (is_tree ? pso_fa_gqa6_m1_tree_ : pso_fa_gqa6_m1_));
                tg_thr = 128u;
            } else if (M == 2u) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_p2_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_ * 2, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                barrier();
                pso = use_quest
                    ? (is_tree ? pso_fa_gqa6_m2_sp_tree_ : pso_fa_gqa6_m2_sp_)
                    : (qkv ? (is_mtp ? (is_tree ? pso_mtp_gqa6_m2_tree_ : pso_mtp_gqa6_m2_)
                                     : (is_tree ? pso_fa_gqa6_m2_q_tree_ : pso_fa_gqa6_m2_q_))
                           : (is_tree ? pso_fa_gqa6_m2_tree_ : pso_fa_gqa6_m2_));
                q_buf = buf_q_h_;
                tg_thr = 128u;
            } else if (M <= 4u) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_p4_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_ * 3, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                barrier();
                pso = use_quest
                    ? (is_tree ? pso_fa_gqa6_m4_sp_tree_ : pso_fa_gqa6_m4_sp_)
                    : (qkv ? (is_mtp ? (is_tree ? pso_mtp_gqa6_m4_tree_ : pso_mtp_gqa6_m4_)
                                     : (is_tree ? pso_fa_gqa6_m4_q_tree_ : pso_fa_gqa6_m4_q_))
                           : (is_tree ? pso_fa_gqa6_m4_tree_ : pso_fa_gqa6_m4_));
                q_buf = buf_q_h_;
                tg_thr = 128u;
            } else {
                z_tiles = is_tree ? ceil_div(M, 8u) : 1u;
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pack_q_h_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_q_h_ offset:0 atIndex:1];
                [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(num_heads_, z_tiles * 8u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                barrier();
                pso = use_quest
                    ? (is_tree ? pso_fa_gqa6_v_sp_tree_ : pso_fa_gqa6_v_sp_)
                    : (qkv ? (is_mtp ? (is_tree ? pso_mtp_gqa6_v_tree_ : pso_mtp_gqa6_v_)
                                     : (is_tree ? pso_fa_gqa6_v_q_tree_ : pso_fa_gqa6_v_q_))
                           : (is_tree ? pso_fa_gqa6_v_tree_ : pso_fa_gqa6_v_));
                q_buf = buf_q_h_;
                tg_thr = 192u;
            }
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
            [enc setBuffer:(id<MTLBuffer>)q_buf offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:2];
            if (qkv) {
                [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                if (is_tree) bind_tree(9u);
                if (use_quest) {
                    uint32_t ap_idx = is_tree ? 10u : 9u;
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:ap_idx];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:ap_idx + 1u];
                }
            } else {
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                if (is_tree) bind_tree(7u);
                if (use_quest) {
                    uint32_t ap_idx = is_tree ? 8u : 7u;
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:ap_idx];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:ap_idx + 1u];
                }
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, splits, z_tiles)
                threadsPerThreadgroup:MTLSizeMake(tg_thr, 1, 1)];
        } else {
            uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
            uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
            uint32_t z_tiles = is_tree ? ceil_div(M, 8u) : 1u;
            if (qkv) {
                id vsp_pso = is_mtp
                    ? (is_tree ? pso_mtp_vsplit1_tree_ : pso_mtp_vsplit1_)
                    : (is_tree ? pso_fa_vsplit1_q_tree_ : pso_fa_vsplit1_q_);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)vsp_pso];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                if (is_tree) bind_tree(9u);
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(is_tree ? pso_fa_vsplit1_tree_ : pso_fa_vsplit1_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                if (is_tree) bind_tree(7u);
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_pairs, splits, z_tiles) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_vsplit2_f16_ : pso_fa_vsplit2_)];
        [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
        [enc setBytes:&fp length:sizeof(fp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (is_tree) {
        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(is_mtp ? pso_mtp_attn_tree_f16_ : pso_fa_attn_q_tree_f16_)];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
            [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            bind_tree(8u);
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_attn_tree_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            bind_tree(6u);
        }
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else if (splits > 1u) {
        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split1_q_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
            [enc setBytes:&fp length:sizeof(fp) atIndex:8];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split1_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
            [enc setBytes:&fp length:sizeof(fp) atIndex:6];
        }
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, splits, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_split2_];
        [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
        [enc setBytes:&fp length:sizeof(fp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else if (M >= 16u && head_dim_ == 256) {
        if (use_hier) {
            uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
            if (qkv) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_hier_q_f16_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:7];
                [enc setBytes:&fp length:sizeof(fp) atIndex:8];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_hier_f16_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
            }
            [enc dispatchThreadgroups:MTLSizeMake(gqa, static_cast<uint32_t>(num_kv_heads_) * (M_pad / 64u), 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else if (qkv) {
            if (use_f16 && start_pos < 1024 && (num_heads_ % num_kv_heads_ == 0)) {
                uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_flash_pair_q_f16_];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 8u), num_pairs, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_prefill_flash_q_f16_ : pso_fa_prefill_flash_q_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 16u), num_heads_, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
        } else if (use_f16 && start_pos < 1024 && (num_heads_ % num_kv_heads_ == 0)) {
            uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
            uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_prefill_flash_pair_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 8u), num_pairs, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_prefill_flash_f16_ : pso_fa_prefill_flash_)];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(M, 16u), num_heads_, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    } else {
        if (qkv) {
            id attn_pso = is_mtp
                ? (use_f16 ? pso_mtp_attn_f16_ : pso_fa_attn_q_)
                : (use_f16 ? pso_fa_attn_q_f16_ : pso_fa_attn_q_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)attn_pso];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)k_sb offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)v_sb offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
            [enc setBytes:&fp length:sizeof(fp) atIndex:7];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_fa_attn_f16_ : pso_fa_attn_)];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
        }
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    barrier();

    const TensorView& pan = (lw.ane_share_idx >= 0) ? lw.post_attn_norm_share : lw.post_attn_norm;
    if (M >= 2u && M <= 8u && lw.o_proj.out_dim == static_cast<uint32_t>(hidden_size_) && lw.o_proj.in_dim >= 6144u) {
        // Round 8: Fuse o_proj Split-K reduce + residual add + post_attn_norm-F16!
        encode_qmm_q4_f16x_and_norm(enc, lw.o_proj, buf_attn_out_, 0, buf_hidden_, 0, M, pan, buf_normed_);
    } else {
        qmm(lw.o_proj, buf_attn_out_, 0, buf_hidden_, 0, true);
        barrier();

        // Post-attention RMSNorm
        NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)pan.buffer offset:pan.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    encode_mlp_q4(enc, lw, M, next_norm, dual_out_f32);
}

void Qwen38Model::encode_chunk(
    id enc_id, int M_int, int start_pos, bool compute_logits, bool compute_argmax,
    bool verify_all_logits, bool record_gdn, float compact_margin, bool use_tree_buf,
    int df_cap_src_row, int df_cap_n_rows
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t M = static_cast<uint32_t>(M_int);
    const bool use_f16 = (M > 24u) || (M <= 8u);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto bind_tree = [&](uint32_t idx) {
        if (use_tree_buf) [enc setBuffer:(id<MTLBuffer>)buf_tree_params_ offset:0 atIndex:idx];
        else              [enc setBytes:&cur_tree_ length:sizeof(cur_tree_) atIndex:idx];
    };
    auto qmm = [&](const Q4Weight& w, id xb, size_t xo, id yb, size_t yo, bool add_r) {
        if (use_f16) encode_qmm_q4_f16x(enc, w, xb, xo, yb, yo, M, add_r);
        else         encode_qmm_q4(enc, w, xb, xo, yb, yo, M, add_r);
    };

    // 1. Embedding lookup
    {
        if (!use_cpu_embed_) {
            EmbedParams ep{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(hidden_size_, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        if (cur_vis_count_ > 0 && cur_vis_emb_buf_ != nil && cur_vis_rows_buf_ != nil) {
            ScatterEmbedParams sp{cur_vis_count_, static_cast<uint32_t>(hidden_size_)};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_scatter_embeds_];
            [enc setBuffer:(id<MTLBuffer>)cur_vis_emb_buf_ offset:cur_vis_emb_off_ atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)cur_vis_rows_buf_ offset:cur_vis_rows_off_ atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:2];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(hidden_size_, cur_vis_count_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
    }

    // 2. 64 Layers (Round 8 + 29: fuse Split-K reduce + residual + next RMSNorm-F16 across ALL 64 layers when M=2..8!)
    const bool fuse_decode_norms = (M >= 2u && M <= 8u);
    const bool fuse_final_dual = (verify_all_logits && fuse_decode_norms);
    const bool use_r7 = (M <= 8u && lin_head_k_dim_ == lin_head_v_dim_);
    const int prev_rec_slot = rec_slot_;
    const int cur_rec_slot  = 1 - rec_slot_;
    PathReplayParams lazy_pr{};
    if (record_gdn && pending_replay_m_ > 0) {
        lazy_pr.len = static_cast<uint32_t>(pending_replay_m_);
        for (uint32_t j = 0; j < lazy_pr.len; ++j) lazy_pr.path[j] = pending_path_[j];
    }
    for (int i = 0; i < num_layers_; ++i) {
        const LayerWeights& lw = layers_[i];
        LayerState& st = states_[i];
        const bool skip_in_norm = fuse_decode_norms && (i > 0);
        const bool is_last_layer = (i + 1 == num_layers_);
        const TensorView* next_in_norm = fuse_decode_norms
            ? (!is_last_layer ? &layers_[i + 1].input_norm : (fuse_final_dual ? &final_norm_ : nullptr))
            : nullptr;
        id dual_f32 = (is_last_layer && fuse_final_dual && mtp_loaded_) ? buf_mtp_verify_hid_ : nil;

        if (!lw.is_linear) {
            encode_full_attn_layer(enc, lw, st.k_cache, st.v_cache, M, start_pos, st.k_sb, st.v_sb,
                                   skip_in_norm, next_in_norm, dual_f32, st.k_bounds, &st.quest_built_pages,
                                   /*is_mtp=*/false, use_tree_buf);
            if (dflash_loaded_ && df_cap_n_rows > 0 && df_cap_src_row >= 0) {
                int df_slot = dflash_->target_capture_slot(i);
                if (df_slot >= 0) {
                    encode_dflash_capture(enc, buf_hidden_, df_cap_src_row, 0, df_cap_n_rows, df_slot);
                    barrier();
                }
            }
            continue;
        }

        // Input RMSNorm (skipped when fused into previous layer's down_proj Split-K reduce!)
        if (!skip_in_norm) {
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        id qkv_dst = record_gdn ? rec_qkv_[cur_rec_slot][i] : buf_qkv_;
        id a_dst   = record_gdn ? rec_a_[cur_rec_slot][i]   : buf_a_;
        id b_dst   = record_gdn ? rec_b_[cur_rec_slot][i]   : buf_b_;

        qmm(lw.in_proj_qkv, buf_normed_, 0, qkv_dst, 0, false);
        qmm(lw.in_proj_z,   buf_normed_, 0, buf_z_,  0, false);
        if (M >= 2u && M <= 8u && lw.in_proj_a.out_dim == lw.in_proj_b.out_dim) {
            QmmParams qp{M, lw.in_proj_a.out_dim, lw.in_proj_a.in_dim, 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_in_proj_ab_f16x_];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.w.buffer offset:lw.in_proj_a.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.s.buffer offset:lw.in_proj_a.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.b.buffer offset:lw.in_proj_a.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.w.buffer offset:lw.in_proj_b.w.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.s.buffer offset:lw.in_proj_b.s.offset atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.b.buffer offset:lw.in_proj_b.b.offset atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:8];
            [enc setBytes:&qp length:sizeof(qp) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(2u * ceil_div(lw.in_proj_a.out_dim, 8u), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            qmm(lw.in_proj_b,   buf_normed_, 0, b_dst,   0, false);
            qmm(lw.in_proj_a,   buf_normed_, 0, a_dst,   0, false);
        }
        barrier();

        {
            const bool is_tree = (cur_tree_.is_tree != 0u);
            GdnPrepParams pp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_head_k_dim_),
                             static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_conv_dim_), rms_eps_};
            if (record_gdn) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_prework_lazy_];
                [enc setBuffer:(id<MTLBuffer>)rec_qkv_[prev_rec_slot][i] offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)rec_a_[prev_rec_slot][i] offset:0 atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)rec_b_[prev_rec_slot][i] offset:0 atIndex:8];
                [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:9];
                [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:10];
                [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:11];
                [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:12];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:13];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:14];
                [enc setBytes:&pp length:sizeof(pp) atIndex:15];
                [enc setBytes:&lazy_pr length:sizeof(lazy_pr) atIndex:16];
                if (use_tree_buf && is_tree) bind_tree(17u);
                else                         [enc setBytes:&cur_tree_ length:sizeof(cur_tree_) atIndex:17];
                [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier();
            } else if (is_tree) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_prework_tree_];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:8];
                [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:9];
                [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:10];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:11];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:12];
                [enc setBytes:&pp length:sizeof(pp) atIndex:13];
                bind_tree(14u);
                [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier();
            } else if (use_r7) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_prework_fused_];
                [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:8];
                [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:9];
                [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:10];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:11];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:12];
                [enc setBytes:&pp length:sizeof(pp) atIndex:13];
                [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier();
            } else {
                GdnConvParams cp{M, static_cast<uint32_t>(lin_conv_dim_)};
                bool par_conv = (M >= 32u);
                if (par_conv) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_pf_];
                    [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:3];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, ceil_div(M, 8u), 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_];
                    [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:3];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_gates_];
                [enc setBuffer:(id<MTLBuffer>)a_dst offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)b_dst offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:5];
                [enc setBytes:&pp length:sizeof(pp) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(lin_num_v_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                barrier();

                if (par_conv) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_conv_upd_];
                    [enc setBuffer:(id<MTLBuffer>)qkv_dst offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:1];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(lin_conv_dim_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_qknorm_];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:2];
                [enc setBytes:&pp length:sizeof(pp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(lin_num_k_heads_ * 2, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                barrier();
            }
        }

        {
            const bool is_tree = (cur_tree_.is_tree != 0u);
            GdnRecParams rp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_num_v_heads_),
                            static_cast<uint32_t>(lin_head_k_dim_), static_cast<uint32_t>(lin_head_v_dim_),
                            static_cast<uint32_t>(lin_conv_dim_)};
            if (record_gdn) {
                id rec_pso = is_tree
                    ? ((cur_tree_.pad0 <= 1u) ? pso_gdn_rec_tree_lazy_s1_
                                              : ((cur_tree_.pad0 == 2u) ? pso_gdn_rec_tree_lazy_s2_ : pso_gdn_rec_tree_lazy_))
                    : pso_gdn_rec_lazy_;
                [enc setComputePipelineState:(id<MTLComputePipelineState>)rec_pso];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&rp length:sizeof(rp) atIndex:7];
                [enc setBytes:&lazy_pr length:sizeof(lazy_pr) atIndex:8];
                if (is_tree) bind_tree(9u);
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(is_tree ? pso_gdn_rec_tree_ : pso_gdn_rec_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&rp length:sizeof(rp) atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:8];
                if (is_tree) bind_tree(9u);
            }
            [enc dispatchThreads:MTLSizeMake(32, lin_head_v_dim_ / 4, lin_num_v_heads_)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            barrier();
        }

        {
            GdnNormParams np{M, static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_head_v_dim_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_gdn_norm_f16_ : pso_gdn_norm_)];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.gdn_norm.buffer offset:lw.gdn_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_gdn_normed_ offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(lin_num_v_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();
        }

        const TensorView& pan = (lw.ane_share_idx >= 0) ? lw.post_attn_norm_share : lw.post_attn_norm;
        if (fuse_decode_norms && lw.out_proj.out_dim == static_cast<uint32_t>(hidden_size_) && lw.out_proj.in_dim >= 6144u) {
            // Round 8: Fuse out_proj Split-K reduce + residual add + post_attn_norm-F16!
            encode_qmm_q4_f16x_and_norm(enc, lw.out_proj, buf_gdn_normed_, 0, buf_hidden_, 0, M, pan, buf_normed_);
        } else {
            qmm(lw.out_proj, buf_gdn_normed_, 0, buf_hidden_, 0, true);
            barrier();

            // Post-attention RMSNorm
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)pan.buffer offset:pan.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        encode_mlp_q4(enc, lw, M, next_in_norm, dual_f32);
        if (dflash_loaded_ && df_cap_n_rows > 0 && df_cap_src_row >= 0) {
            int df_slot = dflash_->target_capture_slot(i);
            if (df_slot >= 0) {
                encode_dflash_capture(enc, buf_hidden_, df_cap_src_row, 0, df_cap_n_rows, df_slot);
                barrier();
            }
        }
    }
    if (record_gdn) {
        rec_slot_ = cur_rec_slot;
        pending_replay_m_ = 0;
    }

    // 3. Final RMSNorm + LM Head (Round 29 + 30: fused Layer-63 dual final norm + fused lm_head row-max epilogue!)
    if (verify_all_logits) {
        const uint32_t V = static_cast<uint32_t>(vocab_size_);
        uint32_t n_tg_am = 128u;
        bool fused_lmh_max = false;
        if (M >= 2u && M <= 8u) {
            // Note: Layer 63 already produced FP32 buf_mtp_verify_hid_ + FP16 buf_normed_ via fuse_final_dual!
            if (compute_argmax || compact_margin > 0.0f) {
                const uint32_t skip_logits = (compute_argmax && compact_margin == 0.0f) ? 1u : 0u;
                QmmParams qp{M, lm_head_.out_dim, lm_head_.in_dim, skip_logits};
                n_tg_am = ceil_div(lm_head_.out_dim, 32u);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_lmhead_max_f16x_];
                [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:5];
                [enc setBytes:&qp length:sizeof(qp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                fused_lmh_max = true;
            } else {
                encode_qmm_q4_f16x(enc, lm_head_, buf_normed_, 0, buf_mtp_logits_, 0, M, false);
            }
        } else {
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_verify_hid_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
            encode_qmm_q4(enc, lm_head_, buf_mtp_verify_hid_, 0, buf_mtp_logits_, 0, M, false);
        }
        if (compute_argmax) {
            barrier();
            if (!fused_lmh_max) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
                [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
                [enc setBytes:&V length:sizeof(V) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_argmax_ offset:0 atIndex:1];
            [enc setBytes:&n_tg_am length:sizeof(n_tg_am) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else if (compact_margin > 0.0f) {
            // Round 9 + 30: GPU threshold compaction using fused lm_head row-max partials!
            barrier();
            if (!fused_lmh_max) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
                [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
                [enc setBytes:&V length:sizeof(V) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }
            VerifyCompactParams vcp{V, n_tg_am, MAX_VERIFY_CANDS, compact_margin};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_logit_max2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_verify_row_max_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_verify_cnt_ offset:0 atIndex:2];
            [enc setBytes:&vcp length:sizeof(vcp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            uint32_t n_tg_comp = 128u;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_logit_compact_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_verify_row_max_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_verify_cands_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_verify_cnt_ offset:0 atIndex:3];
            [enc setBytes:&vcp length:sizeof(vcp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg_comp, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        return;
    }

    if (mtp_loaded_) {
        // Normalize ALL M target rows into buf_mtp_in_hid_ (offset by +1 row when start_pos > 0
        // so row 0 holds previous chunk's buf_mtp_last_hid_ for exact shifted-token MTP KV priming!)
        size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(float);
        size_t dst_off = (start_pos > 0) ? row_bytes : 0;
        if (start_pos > 0) {
            uint32_t D = static_cast<uint32_t>(hidden_size_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(D) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        {
            NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:dst_off atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        // Save last row into buf_mtp_last_hid_ for next chunk / first MTP seed
        size_t last_hid_off = dst_off + static_cast<size_t>(M - 1u) * row_bytes;
        {
            uint32_t D = static_cast<uint32_t>(hidden_size_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:last_hid_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ offset:0 atIndex:1];
            [enc setBytes:&D length:sizeof(D) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        if (compute_logits) {
            // Round 31: Ultra-bandwidth FP16-X M=1 lm_head on prefill finalize!
            size_t last_off = static_cast<size_t>(M - 1u) * row_bytes;
            NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:last_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            encode_qmm_q4_f16x(enc, lm_head_, buf_normed_, 0, buf_logits_, 0, 1u, false);
            if (compute_argmax) {
                barrier();
                uint32_t V = static_cast<uint32_t>(vocab_size_);
                uint32_t n_tg = 128u;
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
                [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
                [enc setBytes:&V length:sizeof(V) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_out_ offset:0 atIndex:1];
                [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            barrier();
        }

        // Prime MTP KV cache ONLY (fc + k/v_proj + kv_prep: skips dead q_proj, O(N^2) attn, o_proj & 267M MLP -> 7.2x faster!)
        if (start_pos == 0 && M > 1u) {
            encode_mtp_kv_prime(enc, buf_token_ids_, sizeof(int), buf_mtp_in_hid_, 0, static_cast<int>(M - 1u), 0);
        } else if (start_pos > 0) {
            encode_mtp_kv_prime(enc, buf_token_ids_, 0, buf_mtp_in_hid_, 0, static_cast<int>(M), start_pos - 1);
        }
        return;
    }

    if (compute_logits) {
        size_t last_off = static_cast<size_t>(M - 1u) * hidden_size_ * sizeof(float);
        {
            NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:last_off atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        encode_qmm_q4_f16x(enc, lm_head_, buf_normed_, 0, buf_logits_, 0, 1u, false);

        if (compute_argmax) {
            barrier();
            uint32_t V = static_cast<uint32_t>(vocab_size_);
            uint32_t n_tg = 128u;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
            [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBytes:&V length:sizeof(V) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_out_ offset:0 atIndex:1];
            [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }

    if (dflash_loaded_ && df_cap_n_rows > 0 && df_cap_src_row >= 0) {
        barrier();
        encode_dflash_context_commit(enc, 0, df_cap_n_rows, start_pos + df_cap_src_row);
    }
}

void Qwen38Model::encode_mtp_kv_prime(
    id enc_id, id tok_buf, size_t tok_off, id hid_buf, size_t hid_off, int M_int, int start_pos
) {
    // 100% Bit-Exact Prefill KV Priming for 1-Layer Causal MTP:
    // Because MTP has only 1 layer and forward() resets mtp_seeded_=false, tokens 0..N-2 ONLY need
    // mtp_k_cache_ & mtp_v_cache_ populated (fc -> input_norm -> k_proj + v_proj -> fa_kv_prep).
    // Eliminates dead q_proj, 17th O(N^2) attention, o_proj, and 267M-param MLP -> 7.2x faster & O(N) scaling!
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t M = static_cast<uint32_t>(M_int);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    const bool use_f16 = (M <= 8u || M > 24u);
    auto qmm = [&](const Q4Weight& w, id xb, size_t xo, id yb, size_t yo, bool add_r) {
        if (use_f16) encode_qmm_q4_f16x(enc, w, xb, xo, yb, yo, M, add_r);
        else         encode_qmm_q4(enc, w, xb, xo, yb, yo, M, add_r);
    };

    if (use_f16) {
        MtpConcatParams cp{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_embed_concat_f16_];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)tok_buf offset:tok_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)hid_buf offset:hid_off atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_embedding.buffer offset:mtp_.pre_fc_norm_embedding.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_hidden.buffer offset:mtp_.pre_fc_norm_hidden.offset atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_concat_ offset:0 atIndex:7];
        [enc setBytes:&cp length:sizeof(cp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, mtp_.fc, buf_mtp_concat_, 0, buf_hidden_, 0, M, false);
        barrier();
    } else {
        EmbedParams ep{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)tok_buf offset:tok_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_emb_ offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(hidden_size_, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        MtpConcatParams cp{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_concat_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_emb_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)hid_buf offset:hid_off atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_embedding.buffer offset:mtp_.pre_fc_norm_embedding.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_hidden.buffer offset:mtp_.pre_fc_norm_hidden.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_concat_ offset:0 atIndex:4];
        [enc setBytes:&cp length:sizeof(cp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4(enc, mtp_.fc, buf_mtp_concat_, 0, buf_hidden_, 0, M, false);
        barrier();
    }

    const LayerWeights& lw = mtp_.layer;
    NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)(use_f16 ? pso_rmsnorm_f16_ : pso_rmsnorm_)];
    [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
    [enc setBytes:&np length:sizeof(np) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    qmm(lw.k_proj, buf_normed_, 0, buf_k_, 0, false);
    qmm(lw.v_proj, buf_normed_, 0, buf_v_, 0, false);
    barrier();

    uint32_t kv_len = static_cast<uint32_t>(start_pos + static_cast<int>(M));
    FullAttnParams fp{
        M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
        static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
        static_cast<uint32_t>(start_pos), kv_len, rope_theta_, rms_eps_,
        1.0f / std::sqrt(static_cast<float>(head_dim_)), 1u, rope_delta_
    };
    const bool qkv = (mtp_k_sb_ != nil);
    if (qkv) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_kv_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)mtp_v_sb_ offset:0 atIndex:6];
        [enc setBytes:&fp length:sizeof(fp) atIndex:7];
    } else {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:4];
        [enc setBytes:&fp length:sizeof(fp) atIndex:5];
    }
    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    barrier();
}

void Qwen38Model::encode_mtp_steps(
    id enc_id, id tok_buf, size_t tok_off, id hid_buf, size_t hid_off,
    int M_int, int start_pos, bool compute_argmax_last,
    id argmax_dst_buf, size_t argmax_dst_off,
    const DraftSampleSpec* sample_spec
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t M = static_cast<uint32_t>(M_int);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    // Round 18 & 19: Fused Q4 Embed + Dual Pre-FC Norm -> FP16 buf_mtp_concat_ + FP16-X mtp.fc!
    const bool use_f16_mtp = (M <= 8u || M > 24u);
    if (use_f16_mtp) {
        MtpConcatParams cp{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_embed_concat_f16_];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)tok_buf offset:tok_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)hid_buf offset:hid_off atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_embedding.buffer offset:mtp_.pre_fc_norm_embedding.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_hidden.buffer offset:mtp_.pre_fc_norm_hidden.offset atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_concat_ offset:0 atIndex:7];
        [enc setBytes:&cp length:sizeof(cp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, mtp_.fc, buf_mtp_concat_, 0, buf_hidden_, 0, M, false);
        barrier();
    } else {
        // 1. Embed tokens -> buf_mtp_emb_
        {
            EmbedParams ep{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)tok_buf offset:tok_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_emb_ offset:0 atIndex:4];
            [enc setBytes:&ep length:sizeof(ep) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(hidden_size_, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        // 2. Concat pre_fc_norm_embedding(emb) + pre_fc_norm_hidden(hid) -> buf_mtp_concat_ [M, 2*D]
        {
            MtpConcatParams cp{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_concat_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_emb_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)hid_buf offset:hid_off atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_embedding.buffer offset:mtp_.pre_fc_norm_embedding.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_hidden.buffer offset:mtp_.pre_fc_norm_hidden.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_concat_ offset:0 atIndex:4];
            [enc setBytes:&cp length:sizeof(cp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        // 3. fc(concat) -> buf_hidden_ [M, D]
        encode_qmm_q4(enc, mtp_.fc, buf_mtp_concat_, 0, buf_hidden_, 0, M, false);
        barrier();
    }

    // 4. 1 Full-Attention Decoder Layer using MTP KV cache (clear M-RoPE binding; MTP uses scalar shifted pos)
    //    Fast-Path for M = 2..8 decode sync: rows 0..M-2 ONLY need k_proj/v_proj/kv_prep (MTP has 1 layer!),
    //    so q_proj, Attention (fa_gqa6_m1!), o_proj, and MLP run ONLY on row M-1 (4x faster attn at 32k-128k!)
    if (M >= 2u && M <= 8u && cur_tree_.is_tree == 0u) {
        const LayerWeights& lw = mtp_.layer;
        NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.input_norm.buffer offset:lw.input_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        const size_t last_norm_off = static_cast<size_t>(M - 1u) * hidden_size_ * sizeof(uint16_t);
        const size_t last_hid_off  = static_cast<size_t>(M - 1u) * hidden_size_ * sizeof(float);
        encode_qmm_q4_f16x(enc, lw.q_proj, buf_normed_, last_norm_off, buf_qkv_, 0, 1u, false);
        encode_qmm_q4_f16x(enc, lw.k_proj, buf_normed_, 0, buf_k_, 0, M, false);
        encode_qmm_q4_f16x(enc, lw.v_proj, buf_normed_, 0, buf_v_, 0, M, false);
        barrier();

        const uint32_t kv_len = static_cast<uint32_t>(start_pos + static_cast<int>(M));
        const uint32_t q_pos  = kv_len - 1u;
        const bool qkv = (mtp_k_sb_ != nil);

        // Concurrent: fa_kv_prep on all M rows [start_pos..q_pos] + fa_q_prep on row M-1 at q_pos!
        FullAttnParams fp_kv{
            M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
            static_cast<uint32_t>(start_pos), kv_len, rope_theta_, rms_eps_,
            1.0f / std::sqrt(static_cast<float>(head_dim_)), 1u, rope_delta_
        };
        if (qkv) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_kv_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)mtp_v_sb_ offset:0 atIndex:6];
            [enc setBytes:&fp_kv length:sizeof(fp_kv) atIndex:7];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_kv_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:4];
            [enc setBytes:&fp_kv length:sizeof(fp_kv) atIndex:5];
        }
        [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        bool use_vsplit = (kv_len >= 128u && head_dim_ == 256 && (num_heads_ % num_kv_heads_ == 0));
        bool use_gqa6   = use_vsplit && (num_heads_ == num_kv_heads_ * 6) && (kv_len >= 1024u);
        uint32_t win_sink   = 0u;
        uint32_t win_recent = 0u;
        uint32_t total_pages = ceil_div(kv_len, QUEST_PAGE_SIZE);
        uint32_t budget_pages = (sparse_kv_budget_ > 0)
            ? std::clamp<uint32_t>(static_cast<uint32_t>(sparse_kv_budget_) / QUEST_PAGE_SIZE,
                                   QUEST_SINK_PAGES + QUEST_RECENT_PAGES + 16u, QUEST_MAX_ACTIVE_PAGES)
            : 0u;
        bool use_quest = use_gqa6 && (sparse_kv_budget_ > 0) && (mtp_k_bounds_ != nil) &&
                         (total_pages > budget_pages + 64u) && (total_pages <= QUEST_SCORE_STRIDE);
        uint32_t splits = use_quest
            ? ((budget_pages > 512u) ? 128u : 64u)
            : (use_gqa6
                ? ((kv_len >= 32768u) ? 128u : ((kv_len >= 8192u) ? 64u : 32u))
                : (use_vsplit ? ((kv_len >= 4096u) ? 16u : ((kv_len >= 1024u) ? 8u : 4u))
                              : ((kv_len >= 256u) ? 16u : 1u)));
        FullAttnParams fp{
            1u, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
            q_pos, kv_len, rope_theta_, rms_eps_,
            1.0f / std::sqrt(static_cast<float>(head_dim_)), splits, rope_delta_,
            win_sink, win_recent
        };
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
        [enc setBytes:&fp length:sizeof(fp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();

        if (use_vsplit) {
            if (use_gqa6) {
                if (use_quest) {
                    uint32_t recent_start_page = total_pages - QUEST_RECENT_PAGES;
                    QuestParams qp{
                        kv_len, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_), 1u,
                        static_cast<uint32_t>(mtp_quest_built_pages_), recent_start_page, total_pages,
                        QUEST_SINK_PAGES, recent_start_page, budget_pages, QUEST_MAX_ACTIVE_PAGES, QUEST_SCORE_STRIDE
                    };
                    if (qp.end_page > qp.start_page) {
                        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_build_bounds_];
                        [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:0];
                        if (qkv) {
                            [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:1];
                            [enc setBuffer:(id<MTLBuffer>)mtp_k_bounds_ offset:0 atIndex:2];
                            [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                        } else {
                            [enc setBuffer:(id<MTLBuffer>)mtp_k_bounds_ offset:0 atIndex:1];
                            [enc setBytes:&qp length:sizeof(qp) atIndex:2];
                        }
                        [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, qp.end_page - qp.start_page, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        barrier();
                        mtp_quest_built_pages_ = static_cast<int>(qp.end_page);
                    }
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_score_pages_];
                    [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)mtp_k_bounds_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_scores_ offset:0 atIndex:2];
                    [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, 16, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_quest_select_topk_];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_scores_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:2];
                    [enc setBytes:&qp length:sizeof(qp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();
                }
                id pso = use_quest ? pso_fa_gqa6_m1_sp_ : (qkv ? pso_mtp_gqa6_m1_ : pso_fa_gqa6_m1_);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:2];
                if (qkv) {
                    [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)mtp_v_sb_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                    if (use_quest) {
                        [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:9];
                        [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:10];
                    }
                } else {
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                    if (use_quest) {
                        [enc setBuffer:(id<MTLBuffer>)buf_quest_pages_ offset:0 atIndex:7];
                        [enc setBuffer:(id<MTLBuffer>)buf_quest_counts_ offset:0 atIndex:8];
                    }
                }
                [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
                uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
                [enc setComputePipelineState:(id<MTLComputePipelineState>)(qkv ? pso_mtp_vsplit1_ : pso_fa_vsplit1_)];
                [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:2];
                if (qkv) {
                    [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)mtp_v_sb_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:8];
                } else {
                    [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                    [enc setBytes:&fp length:sizeof(fp) atIndex:6];
                }
                [enc dispatchThreadgroups:MTLSizeMake(num_pairs, splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            barrier();
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_vsplit2_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(qkv ? pso_mtp_attn_f16_ : pso_fa_attn_f16_)];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)mtp_k_cache_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)mtp_v_cache_ offset:0 atIndex:3];
            if (qkv) {
                [enc setBuffer:(id<MTLBuffer>)mtp_k_sb_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)mtp_v_sb_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            } else {
                [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        barrier();

        encode_qmm_q4_f16x(enc, lw.o_proj, buf_attn_out_, 0, buf_hidden_, last_hid_off, 1u, true);
        barrier();

        NormParams np1{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:last_hid_off atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.post_attn_norm.buffer offset:lw.post_attn_norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np1 length:sizeof(np1) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_mlp_q4(enc, lw, 1u, nullptr, nil, last_hid_off);
    } else {
        id saved_mrope = cur_mrope_buf_;
        cur_mrope_buf_ = nil;
        encode_full_attn_layer(enc, mtp_.layer, mtp_k_cache_, mtp_v_cache_, M, start_pos, mtp_k_sb_, mtp_v_sb_,
                               false, nullptr, nil, nil, nullptr, /*is_mtp=*/true, /*use_tree_buf=*/false);
        cur_mrope_buf_ = saved_mrope;
    }

    // 5. Final MTP norm on last row -> dual FP32 buf_mtp_seed_hid_ + FP16 buf_normed_ for FP16-X draft_head!
    {
        size_t last_off = static_cast<size_t>(M - 1u) * hidden_size_ * sizeof(float);
        NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_dual_];
        [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:last_off atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)mtp_.norm.buffer offset:mtp_.norm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_seed_hid_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
        [enc setBytes:&np length:sizeof(np) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        barrier();
    }

    if (compute_argmax_last) {
        Q4Weight draft_head = lm_head_;
        uint32_t V = draft_vocab_size_;
        draft_head.out_dim = V;
        encode_qmm_q4_f16x(enc, draft_head, buf_normed_, 0, buf_logits_, 0, 1u, false);
        barrier();
        uint32_t n_tg = 128u;
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
        [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
        [enc setBytes:&V length:sizeof(V) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        id dst_b = (argmax_dst_buf != nil) ? argmax_dst_buf : buf_argmax_out_;
        size_t dst_o = (argmax_dst_buf != nil) ? argmax_dst_off : 0;
        if (sample_spec != nullptr && sample_spec->temp > 0.0f && sample_spec->dist_buf != nil) {
            // Round 10 + 26 + Stage 1: Single-CB GPU sampled MTP drafting with exact GPU top-k + top-p truncation!
            DraftSampleParams dsp{
                V, n_tg, 1.0f / sample_spec->temp, 14.0f * sample_spec->temp,
                static_cast<uint32_t>(sample_spec->seed & 0xFFFFFFFFull),
                static_cast<uint32_t>(sample_spec->seed >> 32),
                static_cast<uint32_t>(std::max(0, sample_spec->top_k)),
                sample_spec->top_p
            };
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_sample_draft_];
            [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)dst_b offset:dst_o atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)sample_spec->dist_buf offset:sample_spec->dist_off atIndex:3];
            [enc setBytes:&dsp length:sizeof(dsp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)dst_b offset:dst_o atIndex:1];
            [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        barrier();
    }
}

Qwen38Model::TreeParams Qwen38Model::make_tree_params(const int* parents, int n_nodes, uint32_t mask_base_pos) {
    TreeParams tp{};
    tp.is_tree = 1u;
    tp.n_nodes = static_cast<uint32_t>(n_nodes);
    tp.mask_base_pos = mask_base_pos;
    int child_cnt[16] = {};
    int rem_child[16] = {};
    for (int t = 0; t < n_nodes; ++t) {
        tp.parent[t] = parents[t];
        tp.gdn_load[t] = -1;
        tp.gdn_save[t] = -1;
        if (parents[t] < 0) {
            tp.depth[t] = 0u;
            tp.mask[t] = (1u << t);
        } else {
            int p = parents[t];
            tp.depth[t] = tp.depth[p] + 1u;
            tp.mask[t] = tp.mask[p] | (1u << t);
            child_cnt[p]++;
            rem_child[p]++;
        }
    }
    bool slot_used[4] = {};
    int slot_of[16];
    uint32_t max_slot = 0u;
    for (int t = 0; t < 16; ++t) slot_of[t] = -1;
    for (int t = 0; t < n_nodes; ++t) {
        if (t > 0) {
            int p = parents[t];
            if (p != t - 1) {
                tp.gdn_load[t] = slot_of[p];
            }
            rem_child[p]--;
            if (rem_child[p] == 0 && slot_of[p] >= 0) {
                slot_used[slot_of[p]] = false;
                slot_of[p] = -1;
            }
        }
        if (child_cnt[t] > 1) {
            int s = -1;
            for (int k = 0; k < 4; ++k) {
                if (!slot_used[k]) { s = k; break; }
            }
            if (s < 0) throw std::runtime_error("Tree exceeds 4 simultaneous GDN fork checkpoints");
            slot_used[s] = true;
            slot_of[t] = s;
            tp.gdn_save[t] = s;
            max_slot = std::max(max_slot, static_cast<uint32_t>(s + 1));
        }
    }
    tp.pad0 = max_slot;
    return tp;
}

void Qwen38Model::encode_kv_compact_path(id enc_id, int pre_pos, const uint32_t* path, int len) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    PathReplayParams pr{};
    pr.len = static_cast<uint32_t>(len);
    pr.pre_pos = static_cast<uint32_t>(pre_pos);
    pr.Hkv = static_cast<uint32_t>(num_kv_heads_);
    pr.D = static_cast<uint32_t>(head_dim_);
    pr.kv_bits = static_cast<uint32_t>(kv_bits_);
    for (int j = 0; j < len; ++j) pr.path[j] = path[j];

    for (int i = 0; i < num_layers_; ++i) {
        if (layers_[i].is_linear) continue;
        const LayerState& st = states_[i];
        if (kv_bits_ == 16) {
            uint32_t n = static_cast<uint32_t>(num_kv_heads_ * head_dim_);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_kv_compact_f16_];
            [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:1];
            [enc setBytes:&pr length:sizeof(pr) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            uint32_t n = static_cast<uint32_t>(num_kv_heads_) * ((kv_bits_ == 8) ? static_cast<uint32_t>(head_dim_) : static_cast<uint32_t>(head_dim_ / 2));
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_kv_compact_q_];
            [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:3];
            [enc setBytes:&pr length:sizeof(pr) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void Qwen38Model::encode_gdn_replay_compute(id enc_id, int accepted_plus_one, const uint32_t* path) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    uint32_t M = static_cast<uint32_t>(accepted_plus_one);
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    PathReplayParams pr{};
    if (path != nullptr) {
        pr.len = M;
        for (uint32_t j = 0; j < M; ++j) pr.path[j] = path[j];
    }
    for (int i = 0; i < num_layers_; ++i) {
        if (!layers_[i].is_linear) continue;
        const LayerWeights& lw = layers_[i];
        const LayerState& st = states_[i];

        GdnPrepParams pp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_head_k_dim_),
                         static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_conv_dim_), rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)(path != nullptr ? pso_gdn_prework_replay_path_ : pso_gdn_prework_fused_)];
        [enc setBuffer:(id<MTLBuffer>)rec_qkv_[rec_slot_][i] offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)rec_a_[rec_slot_][i] offset:0 atIndex:7];
        [enc setBuffer:(id<MTLBuffer>)rec_b_[rec_slot_][i] offset:0 atIndex:8];
        [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:9];
        [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:10];
        [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:11];
        [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:12];
        [enc setBytes:&pp length:sizeof(pp) atIndex:13];
        if (path != nullptr) [enc setBytes:&pr length:sizeof(pr) atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        GdnRecParams rp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_num_v_heads_),
                        static_cast<uint32_t>(lin_head_k_dim_), static_cast<uint32_t>(lin_head_v_dim_),
                        static_cast<uint32_t>(lin_conv_dim_)};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_rec_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
        [enc setBytes:&rp length:sizeof(rp) atIndex:7];
        [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:8];
        [enc dispatchThreads:MTLSizeMake(32, lin_head_v_dim_ / 4, lin_num_v_heads_)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        barrier();
    }
}

static inline uint32_t hash_ngram4(int a, int b, int c, int d) {
    uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(a)) * 0x9E3779B97F4A7C15ull
               ^ static_cast<uint64_t>(static_cast<uint32_t>(b)) * 0xBF58476D1CE4E5B9ull
               ^ static_cast<uint64_t>(static_cast<uint32_t>(c)) * 0x94D049BB133111EBull
               ^ static_cast<uint64_t>(static_cast<uint32_t>(d)) * 0xD6E8FEB86659FD93ull;
    h ^= h >> 30;
    h *= 0x94D049BB133111EBull;
    return static_cast<uint32_t>(h ^ (h >> 27));
}

static inline uint32_t hash_ngram3(int a, int b, int c) {
    uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(a)) * 0x9E3779B97F4A7C15ull
               ^ static_cast<uint64_t>(static_cast<uint32_t>(b)) * 0xBF58476D1CE4E5B9ull
               ^ static_cast<uint64_t>(static_cast<uint32_t>(c)) * 0x94D049BB133111EBull;
    h ^= h >> 30;
    h *= 0x94D049BB133111EBull;
    return static_cast<uint32_t>(h ^ (h >> 27));
}

void Qwen38Model::push_history_token(int tok) {
    add_draft_token(tok);
    suffix_cache_.push_session_token(tok);
    const int idx = static_cast<int>(token_history_.size());
    token_history_.push_back(tok);
    const int* h = token_history_.data();
    if (idx >= 2) {
        uint32_t k3 = hash_ngram3(h[idx - 2], h[idx - 1], tok) & PLD_MASK;
        PldSlot& slot = pld3_[k3];
        if (slot.gen != pld_gen_) {
            slot.gen = pld_gen_;
            slot.pos0 = idx;
            slot.pos1 = -1;
        } else {
            slot.pos1 = slot.pos0;
            slot.pos0 = idx;
        }
    }
    if (idx >= 3) {
        uint32_t k4 = hash_ngram4(h[idx - 3], h[idx - 2], h[idx - 1], tok) & PLD_MASK;
        PldSlot& slot = pld4_[k4];
        if (slot.gen != pld_gen_) {
            slot.gen = pld_gen_;
            slot.pos0 = idx;
            slot.pos1 = -1;
        } else {
            slot.pos1 = slot.pos0;
            slot.pos0 = idx;
        }
    }
}

void Qwen38Model::update_mtp_depth_ema(int accepted, int K) {
    // Smooth EMA of geometric per-token acceptance probability; skip K<=1 shallow 3-way root rounds
    // so known-uncertain single-step trees don't distort deep-branch acceptance estimates (d >= 2).
    if (K >= 2) {
        float trials = static_cast<float>(accepted) + ((accepted < K) ? 1.0f : 0.5f);
        float obs_p  = (accepted == K)
            ? std::min(0.95f, (static_cast<float>(accepted) + 0.5f) / (static_cast<float>(K) + 1.0f))
            : (static_cast<float>(accepted) / trials);
        mtp_alpha_ema_ = 0.82f * mtp_alpha_ema_ + 0.18f * obs_p;
    }
    float a = std::clamp(mtp_alpha_ema_, 0.20f, 0.95f);

    int min_k = std::min(3, mtp_depth_);
    int best_k = cur_mtp_depth_;
    float best_rate = -1.0f;
    float pow_a = a;
    float exp_acc = 0.0f;
    for (int k = 1; k <= 7; ++k) {
        exp_acc += pow_a;
        pow_a *= a;
        if (k >= min_k) {
            float rate = (1.0f + exp_acc) / (39.5f + 0.77f * static_cast<float>(k));
            if (rate > best_rate) {
                best_rate = rate;
                best_k = k;
            }
        }
    }
    cur_mtp_depth_ = best_k;
}

int Qwen38Model::try_pld_draft(int anchor_token, int* out_draft, int max_draft, int prefer_first) const {
    if (!pld_enabled_ || max_draft < 3) return 0;
    const int n = static_cast<int>(token_history_.size());
    if (n < 6) return 0;
    const int* h = token_history_.data();
    const int t3 = h[n - 3];
    const int t2 = h[n - 2];
    const int t1 = h[n - 1];
    const int t0 = anchor_token;

    // Round 27 + Stage 4: O(1) 2-way hash-indexed 4-gram lookup (prefers MTP-agreeing continuation when prefer_first >= 0!)
    PldSlot s4 = pld4_[hash_ngram4(t3, t2, t1, t0) & PLD_MASK];
    if (s4.gen != pld_gen_) return 0;

    int best_i = -1;
    int best_avail = 0;
    bool best_pref = false;
    int cands[2] = {s4.pos0, s4.pos1};
    for (int ci = 0; ci < 2; ++ci) {
        int i = cands[ci];
        if (i < 3 || i > n - 2) continue;
        if (h[i] == t0 && h[i - 1] == t1 && h[i - 2] == t2 && h[i - 3] == t3) {
            int avail = std::min(max_draft, (n - 1) - i);
            bool pref = (prefer_first >= 0 && h[i + 1] == prefer_first);
            if (pref && (!best_pref || avail > best_avail)) {
                best_pref = true;
                best_avail = avail;
                best_i = i;
            } else if (!best_pref && avail > best_avail) {
                best_avail = avail;
                best_i = i;
            }
        }
    }
    if (best_i >= 0 && best_avail >= 3) {
        for (int k = 0; k < best_avail; ++k) out_draft[k] = h[best_i + 1 + k];
        return best_avail;
    }
    // Periodic tail repetition (e.g. structured code / tables): extend cyclically to full max_draft!
    if (best_i >= 0 && best_i >= n - 16) {
        int period = n - best_i;
        for (int k = 0; k < max_draft; ++k) {
            int src = best_i + 1 + (k % period);
            out_draft[k] = (src == n) ? anchor_token : h[src];
        }
        return max_draft;
    }
    return 0;
}

int Qwen38Model::try_pld3_draft(int anchor_token, int* out_draft, int max_draft, int prefer_first) const {
    if (!pld_enabled_ || max_draft < 3) return 0;
    const int n = static_cast<int>(token_history_.size());
    if (n < 5) return 0;
    const int* h = token_history_.data();
    const int t2 = h[n - 2];
    const int t1 = h[n - 1];
    const int t0 = anchor_token;

    PldSlot s3 = pld3_[hash_ngram3(t2, t1, t0) & PLD_MASK];
    if (s3.gen != pld_gen_) return 0;

    int best_i = -1;
    int best_avail = 0;
    bool best_pref = false;
    int cands[2] = {s3.pos0, s3.pos1};
    for (int ci = 0; ci < 2; ++ci) {
        int i = cands[ci];
        if (i < 2 || i > n - 2) continue;
        if (h[i] == t0 && h[i - 1] == t1 && h[i - 2] == t2) {
            int avail = std::min(max_draft, (n - 1) - i);
            bool pref = (prefer_first >= 0 && h[i + 1] == prefer_first);
            if (pref && (!best_pref || avail > best_avail)) {
                best_pref = true;
                best_avail = avail;
                best_i = i;
            } else if (!best_pref && avail > best_avail) {
                best_avail = avail;
                best_i = i;
            }
        }
    }
    if (best_i >= 0 && best_avail >= 3) {
        for (int k = 0; k < best_avail; ++k) out_draft[k] = h[best_i + 1 + k];
        return best_avail;
    }
    return 0;
}

const float* Qwen38Model::forward(const int* tokens, int n_tokens, int* out_argmax, StepStats* stats) {
    if (pending_sync_) flush_pending_mtp();
    if (n_tokens <= 0) return reinterpret_cast<const float*>([(id<MTLBuffer>)buf_logits_ contents]);
    if (pos_ + n_tokens > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds max_seq_len (" + std::to_string(max_seq_len_) + ")");
    }
    ensure_kv_capacity(pos_ + n_tokens);

    auto t0 = std::chrono::steady_clock::now();
    int offset = 0;
    const char* env_coexec_str = std::getenv("MI_COEXEC_G");
    const int cur_coexec_g = (ane_ && ane_->ready() && env_coexec_str)
        ? std::clamp(std::atoi(env_coexec_str), 0, 64)
        : 0;
    const int ane_base_chunk = (ane_ && ane_->ready()) ? (2 * AneMlpEngine::MAX_SUBS * ane_->block_rows()) : 0;
    const int ane_max_chunk = (ane_base_chunk > 0) ? (ane_base_chunk + 2 * cur_coexec_g) : 0;
    const int ane_min = ane_min_;
    const int df_cap_from = dflash_loaded_ ? std::max(0, (pos_ + n_tokens) - dflash_->config().sliding_window) : INT_MAX;
    bool used_ane = false;
    while (offset < n_tokens) {
        int rem = n_tokens - offset;
        if (ane_max_chunk > 0 && rem >= ane_min) {
            int m = std::min(ane_max_chunk, rem);
            // Auto tail-absorption co-execution: when rem is 1025..1152 and cur_coexec_g == 0,
            // absorb the <=128-tok tail onto GPU in the SAME 64-layer ANE pass (1.48x faster than 2 ANE passes!)
            if (cur_coexec_g == 0 && rem > ane_max_chunk && rem <= ane_base_chunk + 128) {
                m = rem;
            } else if (rem > ane_max_chunk && rem - m < ane_min && rem <= 2 * ane_max_chunk) {
                // Avoid leaving a tiny tail < ane_min when we can split evenly across two ANE chunks
                m = rem / 2;
            }
            bool is_last = (offset + m == n_tokens);
            forward_ane_chunk(tokens + offset, m, is_last, is_last && (out_argmax != nullptr), nullptr, nullptr, df_cap_from);
            used_ane = true;
            offset += m;
            continue;
        }
        (void)used_ane;

        int m = std::min(chunk_size_, rem);
        bool is_last = (offset + m == n_tokens);
        int df_src = -1, df_n = 0;
        if (dflash_loaded_) {
            int c0 = std::max(pos_, df_cap_from);
            int c1 = pos_ + m;
            if (c1 > c0) { df_src = c0 - pos_; df_n = c1 - c0; }
        }
        std::memcpy([(id<MTLBuffer>)buf_token_ids_ contents], tokens + offset, static_cast<size_t>(m) * sizeof(int));
        if (use_cpu_embed_) {
            cpu_embed_.gather_f32(tokens + offset, m, reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_ contents]));
        }

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_chunk(enc, m, pos_, is_last, is_last && (out_argmax != nullptr), false, false, 0.0f, false, df_src, df_n);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                throw std::runtime_error(std::string("Metal command buffer error: ") +
                                         [[cb.error localizedDescription] UTF8String]);
            }
        }
        pos_ += m;
        if (dflash_loaded_) df_kv_len_ = pos_;
        offset += m;
    }

    for (int i = 0; i < n_tokens; ++i) push_history_token(tokens[i]);
    // Frequency-aware draft vocab sizing: keep 32k (1.0 ms for 5 drafts) on English/code even if prompt has
    // occasional rare identifiers (e.g. "tokenizer"=82637); widen immediately when >10% of prompt is multilingual/CJK!
    if (draft_vocab_size_ < static_cast<uint32_t>(vocab_size_) && n_tokens >= 16) {
        int n_reg = 0, n_gt32 = 0, n_gt64 = 0, n_gt152 = 0;
        for (int i = 0; i < n_tokens; ++i) {
            int tk = tokens[i];
            if (tk < 0 || static_cast<uint32_t>(tk) >= special_tok_lo_) continue;
            ++n_reg;
            if (tk >= static_cast<int>(FAST_DRAFT_VOCAB)) ++n_gt32;
            if (tk >= static_cast<int>(MED_DRAFT_VOCAB))  ++n_gt64;
            if (tk >= static_cast<int>(HI_DRAFT_VOCAB))   ++n_gt152;
        }
        if (n_reg >= 16) {
            if (n_gt152 * 10 > n_reg) {
                draft_vocab_size_ = static_cast<uint32_t>(vocab_size_);
            } else if (n_gt64 * 10 > n_reg && draft_vocab_size_ < HI_DRAFT_VOCAB) {
                draft_vocab_size_ = std::min<uint32_t>(HI_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
            } else if (n_gt32 * 10 > n_reg && draft_vocab_size_ < MED_DRAFT_VOCAB) {
                draft_vocab_size_ = std::min<uint32_t>(MED_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
            }
        }
    }

    // Invalidate MTP seed whenever standard forward() advances target directly (will seed on first MTP round)
    mtp_seeded_ = false;
    mtp_pos_ = pos_ - 1;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = n_tokens;
    }
    if (out_argmax) {
        *out_argmax = static_cast<int>(*reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_argmax_out_ contents]));
    }
    return reinterpret_cast<const float*>([(id<MTLBuffer>)buf_logits_ contents]);
}

EncodedImage Qwen38Model::encode_image_url(const std::string& url, int max_edge) {
    if (!has_vision()) {
        throw std::runtime_error("Model does not have vision tower weights loaded");
    }
    return vision_->encode_url(url, max_edge);
}

const float* Qwen38Model::forward_multimodal(
    const int* tokens, int n_tokens, const std::vector<EncodedImage>& images,
    int* out_argmax, StepStats* stats
) {
    if (images.empty()) {
        return forward(tokens, n_tokens, out_argmax, stats);
    }
    if (pending_sync_) flush_pending_mtp();
    if (n_tokens <= 0) return reinterpret_cast<const float*>([(id<MTLBuffer>)buf_logits_ contents]);
    if (pos_ + n_tokens > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds max_seq_len (" + std::to_string(max_seq_len_) + ")");
    }
    ensure_kv_capacity(pos_ + n_tokens);

    auto t0 = std::chrono::steady_clock::now();

    // 1. Build exact Qwen3-VL 3D M-RoPE positions (t, h, w) and vision embedding pointer map
    std::vector<int> all_pos3d(static_cast<size_t>(n_tokens) * 3);
    std::vector<const float*> all_vis_ptrs(static_cast<size_t>(n_tokens), nullptr);
    int cur_rope = pos_ + rope_delta_;
    size_t img_idx = 0;

    for (int i = 0; i < n_tokens; ) {
        if (tokens[i] == image_token_id_ && img_idx < images.size()) {
            const EncodedImage& img = images[img_idx++];
            int llm_gh = img.grid_h / 2;
            int llm_gw = img.grid_w / 2;
            int n_img = img.num_tokens();
            if (n_img != llm_gh * llm_gw || i + n_img > n_tokens) {
                throw std::runtime_error("Multimodal image_pad token count mismatch with encoded image");
            }
            for (int r = 0; r < llm_gh; ++r) {
                for (int c = 0; c < llm_gw; ++c) {
                    int k = r * llm_gw + c;
                    int idx = i + k;
                    all_pos3d[static_cast<size_t>(idx) * 3 + 0] = cur_rope;
                    all_pos3d[static_cast<size_t>(idx) * 3 + 1] = cur_rope + r;
                    all_pos3d[static_cast<size_t>(idx) * 3 + 2] = cur_rope + c;
                    all_vis_ptrs[static_cast<size_t>(idx)] =
                        img.embeddings.data() + static_cast<size_t>(k) * hidden_size_;
                }
            }
            cur_rope += std::max(llm_gh, llm_gw);
            i += n_img;
        } else {
            all_pos3d[static_cast<size_t>(i) * 3 + 0] = cur_rope;
            all_pos3d[static_cast<size_t>(i) * 3 + 1] = cur_rope;
            all_pos3d[static_cast<size_t>(i) * 3 + 2] = cur_rope;
            cur_rope += 1;
            ++i;
        }
    }

    // 2. Process in ANE (GPU || ANE overlap!) or GPU chunks with M-RoPE + vision embedding scatter
    int offset = 0;
    const char* env_coexec_str = std::getenv("MI_COEXEC_G");
    const int cur_coexec_g = (ane_ && ane_->ready() && env_coexec_str)
        ? std::clamp(std::atoi(env_coexec_str), 0, 64)
        : 0;
    const int ane_base_chunk = (ane_ && ane_->ready()) ? (2 * AneMlpEngine::MAX_SUBS * ane_->block_rows()) : 0;
    const int ane_max_chunk = (ane_base_chunk > 0) ? (ane_base_chunk + 2 * cur_coexec_g) : 0;
    const size_t max_vis_c = std::max<size_t>(
        static_cast<size_t>(chunk_size_),
        ane_base_chunk > 0 ? static_cast<size_t>(ane_base_chunk + 128) : static_cast<size_t>(chunk_size_)
    );

    const int ane_min = ane_min_;
    const int df_cap_from = dflash_loaded_ ? std::max(0, (pos_ + n_tokens) - dflash_->config().sliding_window) : INT_MAX;
    while (offset < n_tokens) {
        int rem = n_tokens - offset;
        if (ane_max_chunk > 0 && rem >= ane_min) {
            int m = std::min(ane_max_chunk, rem);
            if (cur_coexec_g == 0 && rem > ane_max_chunk && rem <= ane_base_chunk + 128) {
                m = rem;
            } else if (rem > ane_max_chunk && rem - m < ane_min && rem <= 2 * ane_max_chunk) {
                m = rem / 2;
            }
            bool is_last = (offset + m == n_tokens);
            forward_ane_chunk(tokens + offset, m, is_last, is_last && (out_argmax != nullptr),
                              all_pos3d.data() + static_cast<size_t>(offset) * 3,
                              all_vis_ptrs.data() + offset, df_cap_from);
            offset += m;
            continue;
        }

        int m = std::min(chunk_size_, rem);
        bool is_last = (offset + m == n_tokens);
        int df_src = -1, df_n = 0;
        if (dflash_loaded_) {
            int c0 = std::max(pos_, df_cap_from);
            int c1 = pos_ + m;
            if (c1 > c0) { df_src = c0 - pos_; df_n = c1 - c0; }
        }
        std::memcpy([(id<MTLBuffer>)buf_token_ids_ contents], tokens + offset, static_cast<size_t>(m) * sizeof(int));
        if (use_cpu_embed_) {
            cpu_embed_.gather_f32(tokens + offset, m, reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_ contents]));
        }

        if (!buf_mrope_pos_)  buf_mrope_pos_  = ctx_.alloc_shared(max_vis_c * 3 * sizeof(int));
        if (!buf_vis_embeds_) buf_vis_embeds_ = ctx_.alloc_shared(max_vis_c * hidden_size_ * sizeof(float));
        if (!buf_vis_rows_)   buf_vis_rows_   = ctx_.alloc_shared(max_vis_c * sizeof(uint32_t));

        std::memcpy([(id<MTLBuffer>)buf_mrope_pos_ contents],
                    all_pos3d.data() + static_cast<size_t>(offset) * 3,
                    static_cast<size_t>(m) * 3 * sizeof(int));

        float* dst_emb = reinterpret_cast<float*>([(id<MTLBuffer>)buf_vis_embeds_ contents]);
        uint32_t* dst_rows = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_vis_rows_ contents]);
        uint32_t vcnt = 0;
        for (int r = 0; r < m; ++r) {
            const float* vp = all_vis_ptrs[offset + r];
            if (vp != nullptr) {
                std::memcpy(dst_emb + static_cast<size_t>(vcnt) * hidden_size_, vp,
                            static_cast<size_t>(hidden_size_) * sizeof(float));
                dst_rows[vcnt] = static_cast<uint32_t>(r);
                ++vcnt;
            }
        }

        cur_mrope_buf_    = buf_mrope_pos_;
        cur_mrope_off_    = 0;
        cur_vis_emb_buf_  = (vcnt > 0) ? buf_vis_embeds_ : nil;
        cur_vis_emb_off_  = 0;
        cur_vis_rows_buf_ = (vcnt > 0) ? buf_vis_rows_ : nil;
        cur_vis_rows_off_ = 0;
        cur_vis_count_    = vcnt;

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_chunk(enc, m, pos_, is_last, is_last && (out_argmax != nullptr), false, false, 0.0f, false, df_src, df_n);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                cur_mrope_buf_ = nil;
                cur_vis_count_ = 0;
                throw std::runtime_error(std::string("Metal multimodal command buffer error: ") +
                                         [[cb.error localizedDescription] UTF8String]);
            }
        }
        cur_mrope_buf_ = nil;
        cur_vis_count_ = 0;
        pos_ += m;
        if (dflash_loaded_) df_kv_len_ = pos_;
        offset += m;
    }

    // Subsequent scalar decode steps use RoPE position (pos + rope_delta_)!
    rope_delta_ = cur_rope - pos_;

    for (int i = 0; i < n_tokens; ++i) push_history_token(tokens[i]);
    mtp_seeded_ = false;
    mtp_pos_ = pos_ - 1;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = n_tokens;
    }
    if (out_argmax) {
        *out_argmax = static_cast<int>(*reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_argmax_out_ contents]));
    }
    return reinterpret_cast<const float*>([(id<MTLBuffer>)buf_logits_ contents]);
}

int Qwen38Model::forward_greedy(int token, StepStats* stats) {
    int next_tok = 0;
    forward(&token, 1, &next_tok, stats);
    return next_tok;
}

void Qwen38Model::flush_pending_mtp() {
    if (!pending_sync_) return;
    if (dflash_loaded_) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            if (pending_replay_m_ > 0) {
                encode_gdn_replay_compute(enc, pending_replay_m_, nullptr);
                pending_replay_m_ = 0;
            }
            if (df_pending_commit_ > 0) {
                encode_dflash_context_commit(enc, 0, df_pending_commit_, df_pending_pos_);
                df_kv_len_ = df_pending_pos_ + df_pending_commit_;
                df_pending_commit_ = 0;
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                throw std::runtime_error(std::string("Metal DFlash flush sync error: ") + [[cb.error localizedDescription] UTF8String]);
            }
        }
        pending_sync_ = false;
        pending_path_noncontig_ = false;
        return;
    }
    int* sync_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_mtp_sync_ids_ contents]);
    std::memcpy(sync_ptr, pending_feed_tokens_,
                static_cast<size_t>(pending_feed_off_ + pending_feed_count_) * sizeof(int));

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

        if (pending_path_noncontig_ && pending_path_len_ > 0) {
            encode_kv_compact_path(enc, pending_kv_pre_pos_, pending_path_, static_cast<int>(pending_path_len_));
        }
        if (pending_replay_m_ > 0) {
            encode_gdn_replay_compute(enc, pending_replay_m_, pending_path_noncontig_ ? pending_path_ : nullptr);
            pending_replay_m_ = 0;
        }

        id feed_hid_buf = buf_mtp_verify_hid_;
        if (pending_path_noncontig_ && pending_path_len_ > 0) {
            PathReplayParams pr{};
            pr.len = pending_path_len_;
            pr.D = static_cast<uint32_t>(hidden_size_);
            for (uint32_t j = 0; j < pending_path_len_; ++j) pr.path[j] = pending_path_[j];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_hid_path_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_verify_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:0 atIndex:1];
            [enc setBytes:&pr length:sizeof(pr) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(hidden_size_, pending_path_len_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
            feed_hid_buf = buf_mtp_in_hid_;
        }

        int cur_pos = pending_mtp_pos_;
        if (pending_seed_first_) {
            encode_mtp_steps(enc, buf_mtp_sync_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_pos, false);
            cur_pos += 1;
            mtp_seeded_ = true;
        }
        encode_mtp_steps(enc, buf_mtp_sync_ids_, static_cast<size_t>(pending_feed_off_) * sizeof(int),
                         feed_hid_buf, 0, pending_feed_count_, cur_pos, true);
        mtp_pos_ = cur_pos + pending_feed_count_;

        // Keep buf_mtp_last_hid_ exact (last committed target final-normed hidden) for save_state / forward()
        size_t last_off = static_cast<size_t>(pending_feed_count_ - 1) * hidden_size_ * sizeof(float);
        uint32_t D = static_cast<uint32_t>(hidden_size_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
        [enc setBuffer:(id<MTLBuffer>)feed_hid_buf offset:last_off atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ offset:0 atIndex:1];
        [enc setBytes:&D length:sizeof(D) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal flush sync error: ") + [[cb.error localizedDescription] UTF8String]);
        }
    }
    mtp_seed_token_ = static_cast<int>(*reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_argmax_out_ contents]));
    mtp_seeded_ = true;
    pending_sync_ = false;
    pending_path_noncontig_ = false;
}

void Qwen38Model::encode_mtp_tree_expand(
    id enc_id, const int* parents, int n_nodes, int pre_draft_mtp_pos,
    bool is_greedy, float draft_temp, int top_k, float top_p,
    uint64_t seed_base, const int* prefilled_mask,
    int start_depth, int max_expand_depth,
    int init_kv_cursor, const int* init_kv_slot, const uint32_t* init_kv_mask,
    bool need_greedy_dist, const int* init_hid_alias
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    int depth[16] = {};
    int children[16][4];
    int n_child[16] = {};
    int max_d = 0;
    for (int t = 0; t < n_nodes; ++t) {
        if (parents[t] >= 0) {
            int p = parents[t];
            depth[t] = depth[p] + 1;
            if (depth[t] > max_d) max_d = depth[t];
            if (n_child[p] < 4) children[p][n_child[p]++] = t;
        }
    }

    int kv_slot[16];
    uint32_t kv_mask[16] = {};
    for (int t = 0; t < 16; ++t) {
        kv_slot[t] = init_kv_slot ? init_kv_slot[t] : -1;
        kv_mask[t] = init_kv_mask ? init_kv_mask[t] : 0u;
    }
    int kv_cursor = init_kv_cursor;

    Q4Weight draft_head = lm_head_;
    const uint32_t V = draft_row_count_;
    draft_head.out_dim = V;
    const float inv_t = is_greedy ? 1.0f : (1.0f / std::max(draft_temp, 1e-4f));
    const float margin = is_greedy ? 12.0f : (14.0f * std::max(draft_temp, 0.01f));

    int d_lim = std::min(max_d, max_expand_depth);
    bool expanded[16] = {};
    bool tok_known[16] = {};
    expanded[0] = true;
    tok_known[0] = true;
    for (int u = 1; u < n_nodes; ++u) {
        if (kv_slot[u] >= 0) expanded[u] = true;
    }
    for (int u = 1; u < n_nodes; ++u) {
        int p = parents[u];
        if (depth[u] == 1 || (prefilled_mask && prefilled_mask[u] != 0) || (p >= 0 && expanded[p])) {
            tok_known[u] = true;
        }
    }

    // Wavefront Tree Expansion: batch ALL ready nodes across independent branches regardless of depth!
    // Cuts CB #2 MTP steps by 33-50% on forked trees (r3w: 2->1, r43: 3->2, df: 3->2, r52: 4->3)!
    for (int wave = 0; wave < 16; ++wave) {
        int batch[8];
        int B = 0;
        for (int u = 1; u < n_nodes; ++u) {
            if (expanded[u] || depth[u] < start_depth || depth[u] >= d_lim || n_child[u] == 0) continue;
            int p = parents[u];
            if (!tok_known[u] || (p > 0 && !expanded[p])) continue;
            bool need = false;
            for (int c = 0; c < n_child[u]; ++c) {
                if (!prefilled_mask || prefilled_mask[children[u][c]] == 0) { need = true; break; }
            }
            if (need && B < 8) batch[B++] = u;
        }
        if (B == 0) break;

        int step_kv_pos = pre_draft_mtp_pos + kv_cursor;
        MtpGatherParams gp{};
        gp.B = static_cast<uint32_t>(B);
        gp.D = static_cast<uint32_t>(hidden_size_);
        gp.eps = rms_eps_;

        TreeParams mtp_tp{};
        mtp_tp.is_tree = 1u;
        mtp_tp.n_nodes = static_cast<uint32_t>(B);
        mtp_tp.mask_base_pos = static_cast<uint32_t>(pre_draft_mtp_pos);

        DraftBatchParams dbp{};
        dbp.B = static_cast<uint32_t>(B);
        dbp.V = V;
        dbp.inv_temp = inv_t;
        dbp.margin = margin;
        uint64_t step_seed = seed_base + static_cast<uint64_t>(wave + start_depth) * 0x9E3779B97F4A7C15ull;
        dbp.seed_lo = static_cast<uint32_t>(step_seed & 0xFFFFFFFFull);
        dbp.seed_hi = static_cast<uint32_t>(step_seed >> 32);
        dbp.top_k = is_greedy ? 0u : static_cast<uint32_t>(std::max(0, top_k));
        dbp.top_p = is_greedy ? 1.0f : top_p;
        dbp.greedy = is_greedy ? 1u : 0u;

        for (int r = 0; r < B; ++r) {
            int u = batch[r];
            int p = parents[u];
            gp.in_tok_idx[r] = static_cast<uint32_t>(u);
            gp.in_hid_idx[r] = (init_hid_alias && init_hid_alias[p] >= 0)
                ? static_cast<uint32_t>(init_hid_alias[p])
                : static_cast<uint32_t>(p);
            gp.out_hid_idx[r] = static_cast<uint32_t>(u);

            int slot = kv_cursor++;
            kv_slot[u] = slot;
            uint32_t m = (1u << slot);
            if (p > 0 && kv_slot[p] >= 0) m |= kv_mask[p];
            kv_mask[u] = m;

            mtp_tp.mask[r] = m;
            mtp_tp.depth[r] = static_cast<uint32_t>(depth[u] - 1);
            mtp_tp.parent[r] = -1;
            mtp_tp.gdn_load[r] = -1;
            mtp_tp.gdn_save[r] = -1;

            // Collect non-prefilled children in DFS order
            int dsts[3] = {-1, -1, -1};
            int nd = 0;
            for (int c = 0; c < n_child[u] && nd < 3; ++c) {
                int ch = children[u][c];
                if (!prefilled_mask || prefilled_mask[ch] == 0) dsts[nd++] = ch;
            }
            dbp.out_tok_idx[r] = static_cast<uint32_t>(dsts[0]);
            dbp.out_alt0_idx[r] = (nd >= 2) ? dsts[1] : -1;
            dbp.out_alt1_idx[r] = (nd >= 3) ? dsts[2] : -1;
            dbp.out_dist_idx[r] = static_cast<uint32_t>(u);
        }

        // 1. Batched gather Q4 embed + dual pre-FC norm -> FP16 buf_mtp_concat_ [B, 2*D]
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_embed_concat_gather_f16_];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_seed_hid_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_embedding.buffer offset:mtp_.pre_fc_norm_embedding.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)mtp_.pre_fc_norm_hidden.buffer offset:mtp_.pre_fc_norm_hidden.offset atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_concat_ offset:0 atIndex:7];
        [enc setBytes:&gp length:sizeof(gp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // 2. mtp.fc -> buf_hidden_ [B, D]
        encode_qmm_q4_f16x(enc, mtp_.fc, buf_mtp_concat_, 0, buf_hidden_, 0, static_cast<uint32_t>(B), false);
        barrier();

        // 3. 1 Full-Attention MTP Layer (uses zero-overhead linear fast path when B==1 on contiguous prefix!)
        id saved_mrope = cur_mrope_buf_;
        TreeParams saved_tree = cur_tree_;
        cur_mrope_buf_ = nil;
        bool is_linear_step = (B == 1 && kv_slot[batch[0]] == depth[batch[0]] - 1 &&
                               kv_mask[batch[0]] == ((1u << depth[batch[0]]) - 1u));
        cur_tree_ = is_linear_step ? TreeParams{} : mtp_tp;
        encode_full_attn_layer(enc, mtp_.layer, mtp_k_cache_, mtp_v_cache_, static_cast<uint32_t>(B), step_kv_pos,
                               mtp_k_sb_, mtp_v_sb_, false, nullptr, nil, nil, nullptr, /*is_mtp=*/true, /*use_tree_buf=*/false);
        cur_tree_ = saved_tree;
        cur_mrope_buf_ = saved_mrope;

        for (int r = 0; r < B; ++r) {
            int u = batch[r];
            expanded[u] = true;
            for (int c = 0; c < n_child[u]; ++c) tok_known[children[u][c]] = true;
        }

        // 4. Batched dual norm -> scatter FP32 to buf_mtp_seed_hid_[out_hid_idx[r]] + contiguous FP16 buf_normed_[r]
        if (B == 1) {
            NormParams np{1u, static_cast<uint32_t>(hidden_size_), rms_eps_};
            size_t dst_off = static_cast<size_t>(gp.out_hid_idx[0]) * hidden_size_ * sizeof(float);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_dual_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)mtp_.norm.buffer offset:mtp_.norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_seed_hid_ offset:dst_off atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_dual_scatter_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)mtp_.norm.buffer offset:mtp_.norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_seed_hid_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        barrier();

        // 5. Draft lm_head: BitScreen 1-bit V=98,304 + exact Q4 top-256 rescore (or Q4 row_map fallback!)
        uint32_t dV = 0u, d_nparts = 0u;
        id d_rmap = encode_draft_lm_head(enc, static_cast<uint32_t>(B), dV, d_nparts);
        dbp.V = dV;
        dbp.n_parts = d_nparts;

        // 6. Batched GPU draft sampling / top-k+top-p / alt extraction -> writes directly to buf_token_ids_!
        if (is_greedy && !need_greedy_dist && B == 1 && dbp.out_alt0_idx[0] < 0) {
            uint32_t n_tg_am = dbp.n_parts;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:static_cast<size_t>(dbp.out_tok_idx[0]) * sizeof(uint32_t) atIndex:1];
            [enc setBytes:&n_tg_am length:sizeof(n_tg_am) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();
        } else {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_sample_draft_batch_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_draft_dists_ offset:0 atIndex:3];
            [enc setBytes:&dbp length:sizeof(dbp) atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)d_rmap offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
    }
}

void Qwen38Model::forward_mtp_tree_impl(int anchor_token, Sampler* sampler, std::vector<int>& out_tokens, StepStats* stats) {
    const int tree_mode = std::getenv("MTP_TREE") ? std::atoi(std::getenv("MTP_TREE")) : 7;
    const bool is_greedy = (sampler == nullptr || sampler->is_greedy());
    const int pre_pos = pos_;
    ensure_kv_capacity(pre_pos + 16);

    auto t0 = std::chrono::steady_clock::now();
    int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);
    tid_ptr[0] = anchor_token;

    const float scfg_temp = is_greedy ? 0.0f : sampler->config().temperature;
    const int   scfg_topk = is_greedy ? 0    : sampler->config().top_k;
    const float scfg_topp = is_greedy ? 1.0f : sampler->config().top_p;
    const float dt_mul = std::getenv("DRAFT_TMUL") ? std::strtof(std::getenv("DRAFT_TMUL"), nullptr) : 0.85f;
    const float draft_temp = is_greedy ? 1.0f : (scfg_temp * dt_mul);
    const float draft_top_p = std::getenv("DRAFT_TOPP")
        ? (std::strtof(std::getenv("DRAFT_TOPP"), nullptr) > 0.0f ? std::strtof(std::getenv("DRAFT_TOPP"), nullptr) : 1.0f)
        : scfg_topp;
    const float v_margin = is_greedy ? 0.0f : sampler->compact_margin();
    const uint64_t round_seed = is_greedy ? 0x123456789ABCDEF0ull : sampler->next_seed();

    // Helper to encode root sync + root draft distribution (writes buf_mtp_seed_hid_[0], buf_draft_dists_[0], and optional root children!)
    auto encode_root_sync = [&](id<MTLComputeCommandEncoder> enc, int out_c0, int out_a0, int out_a1, bool run_lm_head, bool need_dist = true) {
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
        if (pending_sync_) {
            if (pending_path_noncontig_ && pending_path_len_ > 0) {
                encode_kv_compact_path(enc, pending_kv_pre_pos_, pending_path_, static_cast<int>(pending_path_len_));
            }
            // Note: GDN replay (pending_replay_m_) is lazily fused into CB #2b encode_chunk() — ZERO CB #1 GDN replay!
            id feed_hid_buf = buf_mtp_verify_hid_;
            if (pending_path_noncontig_ && pending_path_len_ > 0) {
                PathReplayParams pr{};
                pr.len = pending_path_len_;
                pr.D = static_cast<uint32_t>(hidden_size_);
                for (uint32_t j = 0; j < pending_path_len_; ++j) pr.path[j] = pending_path_[j];
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_hid_path_];
                [enc setBuffer:(id<MTLBuffer>)buf_mtp_verify_hid_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_mtp_in_hid_ offset:0 atIndex:1];
                [enc setBytes:&pr length:sizeof(pr) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(hidden_size_, pending_path_len_, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
                feed_hid_buf = buf_mtp_in_hid_;
            }
            int cur_pos = pending_mtp_pos_;
            if (pending_seed_first_) {
                encode_mtp_steps(enc, buf_mtp_sync_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_pos, false);
                cur_pos += 1;
            }
            // Run MTP layer + dual norm (without lm_head), then run our batch drafter for B=1 so we get alt_id[0..2] & q_top1!
            encode_mtp_steps(enc, buf_mtp_sync_ids_, static_cast<size_t>(pending_feed_off_) * sizeof(int),
                             feed_hid_buf, 0, pending_feed_count_, cur_pos, false);
            mtp_pos_ = cur_pos + pending_feed_count_;
            mtp_seeded_ = true;
            pending_sync_ = false;
            pending_path_noncontig_ = false;
        } else if (!mtp_seeded_) {
            encode_mtp_steps(enc, buf_token_ids_, 0, buf_mtp_last_hid_, 0, 1, mtp_pos_, false);
            mtp_pos_ += 1;
            mtp_seeded_ = true;
        }

        if (run_lm_head) {
            // buf_normed_ already holds FP16 normed root MTP hidden from pso_rmsnorm_dual_!
            uint32_t V = 0u, n_tg = 0u;
            id rmap = encode_draft_lm_head(enc, 1u, V, n_tg);

            if (is_greedy && !need_dist && out_a0 < 0 && out_a1 < 0) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
                [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:static_cast<size_t>(out_c0) * sizeof(uint32_t) atIndex:1];
                [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier();
                return;
            }

            DraftBatchParams dbp{};
            dbp.B = 1u;
            dbp.V = V;
            dbp.n_parts = n_tg;
            dbp.inv_temp = is_greedy ? 1.0f : (1.0f / std::max(draft_temp, 1e-4f));
            dbp.margin = is_greedy ? 12.0f : (14.0f * std::max(draft_temp, 0.01f));
            dbp.seed_lo = static_cast<uint32_t>(round_seed & 0xFFFFFFFFull);
            dbp.seed_hi = static_cast<uint32_t>(round_seed >> 32);
            dbp.top_k = is_greedy ? 0u : static_cast<uint32_t>(std::max(0, scfg_topk));
            dbp.top_p = is_greedy ? 1.0f : draft_top_p;
            dbp.greedy = is_greedy ? 1u : 0u;
            dbp.out_tok_idx[0] = static_cast<uint32_t>(out_c0);
            dbp.out_alt0_idx[0] = out_a0;
            dbp.out_alt1_idx[0] = out_a1;
            dbp.out_dist_idx[0] = 0u;

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_sample_draft_batch_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_draft_dists_ offset:0 atIndex:3];
            [enc setBytes:&dbp length:sizeof(dbp) atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)rmap offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
    };

    if (pending_sync_) {
        int* sync_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_mtp_sync_ids_ contents]);
        std::memcpy(sync_ptr, pending_feed_tokens_,
                    static_cast<size_t>(pending_feed_off_ + pending_feed_count_) * sizeof(int));
    }

    int parents[16] = {};
    int prefilled[16] = {};
    int M = 8;
    bool used_pld = false;
    bool linear_fast = false;
    int pre_draft_mtp_pos = mtp_pos_;

    if (tree_mode == 1 || tree_mode == 2) {
        // Static Single-CB M=8 Trees (ZERO extra CB round-trip!)
        int pld_draft[MAX_VERIFY];
        int pld_k = try_pld_draft(anchor_token, pld_draft, MAX_VERIFY - 1);
        if (pld_k >= 3) {
            used_pld = true;
            linear_fast = true;
            M = pld_k + 1;
            for (int d = 0; d < pld_k; ++d) { tid_ptr[d + 1] = pld_draft[d]; prefilled[d + 1] = 1; }
            for (int i = 0; i < M; ++i) parents[i] = i - 1;
        } else if (tree_mode == 1) {
            // T1: Root-Fork 4+3: 0->1->2->3->4, 0->5->6->7
            const int t1[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
            std::memcpy(parents, t1, sizeof(t1));
        } else {
            // T2: Double-Fork: 0->1->2->3->4, 1->5, 0->6->7
            const int t2[8] = {-1, 0, 1, 2, 3, 1, 0, 6};
            std::memcpy(parents, t2, sizeof(t2));
        }

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

            if (used_pld) {
                encode_root_sync(enc, 1, -1, -1, false);
                pre_draft_mtp_pos = mtp_pos_;
                encode_chunk(enc, M, pre_pos, true, is_greedy, true, true, v_margin);
            } else {
                int root_a0 = (tree_mode == 1) ? 5 : 6;
                encode_root_sync(enc, 1, root_a0, -1, true);
                pre_draft_mtp_pos = mtp_pos_;
                encode_mtp_tree_expand(enc, parents, M, pre_draft_mtp_pos, is_greedy, draft_temp, scfg_topk, draft_top_p, round_seed);
                cur_tree_ = make_tree_params(parents, M, static_cast<uint32_t>(pre_pos));
                encode_chunk(enc, M, pre_pos, true, is_greedy, true, true, v_margin);
                cur_tree_ = TreeParams{};
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) throw std::runtime_error(std::string("Metal tree draft+verify error: ") + [[cb.error localizedDescription] UTF8String]);
        }
    } else if (tree_mode == 7) {
        // T7: Pre-CB1 Suffix Fast-Path + Single-CB1 (d0+d1) + Multi-Branch SuffixCache + 7-Topology Bayesian Solver!
        // Note: For sampled decoding (temp>0), deterministic prefilled tokens lack DraftDist q(x) (accept prob = p(x)
        // vs min(1, p/q)), so Pre-CB1 S0 requires verbatim match_len>=6 (or >=5 greedy) from active session only!
        SuffixResult sr0 = pld_enabled_ ? suffix_cache_.lookup(token_history_, anchor_token) : SuffixResult{};
        bool pre_cb1_suffix = (sr0.n_branches >= 1 && sr0.b[0].cont_len >= 5 && sr0.b[0].from_session &&
                               (is_greedy ? (sr0.b[0].match_len >= 5 || (sr0.b[0].match_len >= 4 && sr0.b[0].count >= 2))
                                          : (sr0.b[0].match_len >= 6)));
        if (pre_cb1_suffix) {
            // Tier S0: Ultra-confident exact suffix match -> 1 SINGLE CB (0 ms draft lm_head, 0 ms d1, 0 ms exp!)
            used_pld = true;
            linear_fast = true;
            M = sr0.b[0].cont_len + 1;
            for (int d = 0; d < sr0.b[0].cont_len; ++d) { tid_ptr[d + 1] = sr0.b[0].cont[d]; prefilled[d + 1] = 1; }
            for (int i = 0; i < M; ++i) parents[i] = i - 1;

            @autoreleasepool {
                id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                encode_root_sync(enc, 1, -1, -1, false);
                pre_draft_mtp_pos = mtp_pos_;
                encode_chunk(enc, M, pre_pos, true, is_greedy, true, true, v_margin);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) throw std::runtime_error(std::string("Metal T7 suffix-fast error: ") + [[cb.error localizedDescription] UTF8String]);
            }
        } else {
        // Adaptive B=3 Tri-Branch Probe in CB #1 + Concurrent CB #2 Pre-Encoding (hides ~0.65 ms CPU encode behind CB #1!):
        const float tri_ema_thresh = std::getenv("MI_TRI_EMA") ? std::strtof(std::getenv("MI_TRI_EMA"), nullptr) : 0.78f;
        const bool tri_probe = (bs_enabled_ ? (mtp_alpha_ema_ < tri_ema_thresh)
                                            : (mtp_alpha_ema_ < 0.78f && pre_pos >= 8192));
        const int probe1_p[3] = {-1, 0, 1};
        const int probe3_p[8] = {-1, 0, 1, 1, 0, 4, 0, 6};
        @autoreleasepool {
            id<MTLCommandBuffer> cb1 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_root_sync(enc1, 1, tri_probe ? 4 : -1, tri_probe ? 6 : -1, true);
            pre_draft_mtp_pos = mtp_pos_;
            encode_mtp_tree_expand(enc1, tri_probe ? probe3_p : probe1_p, tri_probe ? 8 : 3,
                                   pre_draft_mtp_pos, is_greedy, draft_temp,
                                   scfg_topk, draft_top_p, round_seed, nullptr, 1, 2, 0, nullptr, nullptr, true);
            [enc1 endEncoding];
            [cb1 commit];

            // D2: Speculatively pre-encode CB #2 on CPU while GPU executes CB #1!
            // Uncommitted cb2_pre costs 0 GPU work on mispredict and uses dynamic buf_tree_params_ for exact tree reuse.
            const bool allow_pre_cb2 = (sparse_kv_budget_ == 0);
            const int pre_M = (last_t7_m_ > 0) ? last_t7_m_ : ((pre_pos >= 32768) ? 4 : 8);
            const bool pre_tree = (last_t7_m_ > 0) ? last_t7_tree_ : true;
            const int saved_rec_slot = rec_slot_;
            const int saved_pending_replay_m = pending_replay_m_;
            id<MTLCommandBuffer> cb2_pre = nil;
            if (allow_pre_cb2) {
                cb2_pre = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc_pre = [cb2_pre computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                if (pre_tree) {
                    TreeParams dummy_tp{};
                    dummy_tp.is_tree = 1u;
                    dummy_tp.n_nodes = static_cast<uint32_t>(pre_M);
                    dummy_tp.mask_base_pos = static_cast<uint32_t>(pre_pos);
                    dummy_tp.pad0 = 2u; // _s2 (8 KB SRAM, 4 TGs/core) handles both 1-slot and 2-slot T7 trees!
                    cur_tree_ = dummy_tp;
                }
                encode_chunk(enc_pre, pre_M, pre_pos, true, is_greedy, true, true, v_margin, /*use_tree_buf=*/pre_tree);
                cur_tree_ = TreeParams{};
                [enc_pre endEncoding];
            }

            [cb1 waitUntilCompleted];
            if (cb1.error) throw std::runtime_error(std::string("Metal T7 cb1 error: ") + [[cb1.error localizedDescription] UTF8String]);

            DraftDist* all_dd_mut = reinterpret_cast<DraftDist*>([(id<MTLBuffer>)buf_draft_dists_ contents]);
            const DraftDist& d0 = all_dd_mut[0];
            const DraftDist& d1 = all_dd_mut[1];
            const float q0 = d0.q_top1;
            const float q1 = d1.q_top1;
            const int c0 = static_cast<int>(d0.sampled_id);
            const int c1 = static_cast<int>(d1.sampled_id);
            const int a0 = static_cast<int>(d0.alt_id[0]);
            const int a1 = static_cast<int>(d0.alt_id[1]);
            const int a2 = static_cast<int>(d0.alt_id[2]);
            const uint32_t ac0 = d0.alt_count;
            const int a1_0 = static_cast<int>(d1.alt_id[0]);
            const uint32_t ac1 = d1.alt_count;
            const float a = std::clamp(mtp_alpha_ema_, 0.50f, 0.92f);
            const float q1_a0 = (tri_probe && ac0 >= 1u) ? all_dd_mut[4].q_top1 : a;
            const int a0_c1   = (tri_probe && ac0 >= 1u) ? static_cast<int>(all_dd_mut[4].sampled_id) : -1;
            const float q1_a1 = (tri_probe && ac0 >= 2u) ? all_dd_mut[6].q_top1 : a;
            const int a1_c1   = (tri_probe && ac0 >= 2u) ? static_cast<int>(all_dd_mut[6].sampled_id) : -1;

            const float fork_bias = std::getenv("TREE_FBIAS") ? std::strtof(std::getenv("TREE_FBIAS"), nullptr) : 1.15f;
            auto lookup_prob = [](const DraftDist& dd, int tok) -> float {
                for (uint32_t i = 0; i < dd.count && i < 256u; ++i) {
                    if (static_cast<int>(dd.cands[i].id) == tok) return dd.cands[i].val;
                }
                return 0.0f;
            };
            const float p0_a0 = (ac0 >= 1u) ? lookup_prob(d0, a0) : 0.0f;
            const float p0_a1 = (ac0 >= 2u) ? lookup_prob(d0, a1) : 0.0f;
            const float p0_a2 = (ac0 >= 3u) ? lookup_prob(d0, a2) : 0.0f;
            const float p1_a0 = (ac1 >= 1u) ? lookup_prob(d1, a1_0) : 0.0f;

            SuffixResult sr = pld_enabled_ ? suffix_cache_.lookup(token_history_, anchor_token, c0) : SuffixResult{};

            int skip_expand[16] = {};
            int exp_start_d = 1;
            int exp_kv_cur = tri_probe ? 3 : 1;
            int exp_kv_slot[16];
            int exp_hid_alias[16];
            uint32_t exp_kv_mask[16] = {};
            for (int i = 0; i < 16; ++i) { exp_kv_slot[i] = -1; exp_hid_alias[i] = -1; }
            exp_kv_slot[1] = 0;
            exp_kv_mask[1] = 1u;
            skip_expand[1] = 1;
            skip_expand[2] = 1; // tid[2] = c1 already drafted in cb1!

            bool do_hybrid_pld = false;
            // Post-CB1 Suffix Confirmation:
            // - Greedy: accept match_len>=4 with c0 agreement OR match_len==3 with (c0,c1) double agreement.
            // - Sampled: require match_len>=4 with c0 agreement (or repo match with c0+c1 double agreement) AND preserve
            //   cb1's calibrated DraftDist for d=0 (prefilled[1]=0) and d=1 (prefilled[2]=0 when cont[1]==c1)!
            bool s1_hit = (sr.n_branches >= 1 && sr.b[0].cont_len >= 3 && sr.b[0].cont[0] == c0 &&
                           (is_greedy
                               ? (sr.b[0].match_len >= 4 || (sr.b[0].match_len == 3 && sr.b[0].cont_len >= 4 && sr.b[0].cont[1] == c1))
                               : ((sr.b[0].from_session && sr.b[0].match_len >= 4 && (q0 >= 0.85f || (sr.b[0].cont_len >= 2 && sr.b[0].cont[1] == c1))) ||
                                  (!sr.b[0].from_session && sr.b[0].match_len >= 5 && sr.b[0].cont_len >= 4 && sr.b[0].cont[1] == c1))));
            if (s1_hit && sr.n_branches >= 2 && sr.b[0].match_len >= 4 && sr.b[0].cont_len >= 4 &&
                sr.b[1].match_len >= 4 && sr.b[1].cont_len >= 3 && q0 < 0.88f) {
                // Tier S2: 2-Branch Suffix Tree (4+3, ZERO cb2 MTP expansion; preserves d=0/d=1 DraftDist!)
                used_pld = true;
                const int t43[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
                std::memcpy(parents, t43, sizeof(t43));
                for (int j = 0; j < 4; ++j) {
                    tid_ptr[1 + j] = sr.b[0].cont[j];
                    prefilled[1 + j] = (j == 0 || (j == 1 && sr.b[0].cont[1] == c1)) ? 0 : 1;
                }
                for (int j = 0; j < 3; ++j) { tid_ptr[5 + j] = sr.b[1].cont[j]; prefilled[5 + j] = 1; }
            } else if (s1_hit) {
                // Tier S1: MTP-confirmed linear Suffix match (ZERO cb2 MTP expansion; preserves d=0/d=1 DraftDist!)
                used_pld = true;
                linear_fast = true;
                M = sr.b[0].cont_len + 1;
                for (int d = 0; d < sr.b[0].cont_len; ++d) {
                    tid_ptr[d + 1] = sr.b[0].cont[d];
                    prefilled[d + 1] = (d == 0 || (d == 1 && sr.b[0].cont[1] == c1)) ? 0 : 1;
                }
                for (int i = 0; i < M; ++i) parents[i] = i - 1;
            } else if (sr.n_branches >= 1 && sr.b[0].cont_len >= 3 && sr.b[0].cont[0] != c0 &&
                       ((sr.b[0].match_len >= 4 && q0 < 0.985f) ||
                        (sr.b[0].from_session && sr.b[0].match_len == 3 && q0 < 0.95f))) {
                // Tier S3: Hybrid MTP (4 tok, reusing d=1!) || Suffix (3 tok) Parallel Tree!
                const int t_hyb[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
                std::memcpy(parents, t_hyb, sizeof(t_hyb));
                tid_ptr[5] = sr.b[0].cont[0]; prefilled[5] = 1; skip_expand[5] = 1;
                tid_ptr[6] = sr.b[0].cont[1]; prefilled[6] = 1; skip_expand[6] = 1;
                tid_ptr[7] = sr.b[0].cont[2]; prefilled[7] = 1; skip_expand[7] = 1;
                exp_start_d = 2;
                do_hybrid_pld = true;
                used_pld = true;
            } else {
                // D3: Hardware-Adaptive Bayesian E[accepted tokens] / ms(topo) across 9 M=8 + 5 M=4 topologies!
                // Scales priors by GPU bandwidth, tracks live CB #2 GPU time via EMA, and reflects full-context MTP KV4!
                const float bw_scale = 546.0f / std::clamp(static_cast<float>(ctx_.estimated_bandwidth_gbs()), 60.0f, 1200.0f);
                const int eff_kv = (sparse_kv_budget_ > 0 && pre_pos > sparse_kv_budget_ + 2048)
                    ? sparse_kv_budget_
                    : pre_pos;
                const float ctx_frac = static_cast<float>(eff_kv) / 126976.0f;
                const float kv_scale = (kv_bits_ == 16) ? 1.65f : ((kv_bits_ == 4) ? 0.68f : 1.0f);
                const float quest_ovh = (sparse_kv_budget_ > 0 && pre_pos > sparse_kv_budget_ + 2048) ? (0.5f * static_cast<float>(pre_pos) / 126976.0f) : 0.0f;
                const float attn_m8  = (eff_kv >= 1024) ? (36.8f * ctx_frac * kv_scale * bw_scale + quest_ovh) : (0.5f * static_cast<float>(eff_kv) / 1024.0f);
                const float attn_m4  = (eff_kv >= 1024) ? (19.2f * ctx_frac * kv_scale * bw_scale + quest_ovh) : (0.4f * static_cast<float>(eff_kv) / 1024.0f);
                const int eff_mtp_kv = eff_kv;
                const float mtp_frac = static_cast<float>(eff_mtp_kv) / 126976.0f;
                const float mtp_kv_scale = (mtp_kv_bits_ == 4) ? 0.68f : kv_scale;
                const float mtp_cb1  = (1.9f * bw_scale) + ((eff_mtp_kv >= 1024) ? ((tri_probe ? 2.4f : 2.0f) * mtp_frac * mtp_kv_scale * bw_scale) : (tri_probe ? 0.13f : 0.0f));
                const float cpu_ovl  = allow_pre_cb2 ? 0.05f : 0.65f; // D2 hides CB #2 CPU encode behind CB #1!
                // Shared non-attention base EMA (prevents 2-arm bandit inversion where a single cold M=4 sample poisons base_m4 > base_m8!)
                const float prior_m8 = 38.0f * bw_scale;
                const float base_m8  = (ema_cb2_ms_[1] > 0.0f) ? ema_cb2_ms_[1] : prior_m8;
                const float base_m4  = std::max(5.0f, base_m8 - 1.5f * bw_scale);
                const float verify_m8 = base_m8 + attn_m8 + mtp_cb1;
                const float verify_m4 = base_m4 + attn_m4 + mtp_cb1;
                const float wave_b1  = (bs_enabled_ ? 0.76f : 0.80f) * bw_scale + ((eff_mtp_kv >= 1024) ? (0.76f * mtp_frac * mtp_kv_scale * bw_scale) : 0.0f);
                const float wave_b2  = (bs_enabled_ ? 0.89f : 1.02f) * bw_scale + ((eff_mtp_kv >= 1024) ? (1.06f * mtp_frac * mtp_kv_scale * bw_scale) : 0.0f);
                const float wave_b3  = (bs_enabled_ ? 0.89f : 1.10f) * bw_scale + ((eff_mtp_kv >= 1024) ? (1.25f * mtp_frac * mtp_kv_scale * bw_scale) : 0.0f);
                auto m8_cost = [&](float w_ms) { return verify_m8 + std::max(cpu_ovl, w_ms); };
                auto m4_cost = [&](float w_ms) { return verify_m4 + std::max(cpu_ovl, w_ms); };

                const float p01 = q0 * q1;
                const float a2p = a * a, a3 = a2p * a, a4 = a3 * a, a5 = a4 * a;

                float e_lin7 = 1.0f + q0 + p01 * (1.0f + a + a2p + a3 + a4 + a5);
                float r_lin7 = e_lin7 / m8_cost(5.0f * wave_b1);
                float e_lin5 = 1.0f + q0 + p01 * (1.0f + a + a2p + a3);
                float r_lin5 = e_lin5 / m8_cost(3.0f * wave_b1);
                float e_f1   = 1.0f + q0 + p01 * (1.0f + a + a2p) + fork_bias * q0 * p1_a0 * (1.0f + a + a2p);
                float r_f1   = (ac1 >= 1u && p1_a0 > 0.03f) ? (e_f1 / m8_cost(2.0f * wave_b2)) : -1.0f;
                float e_r52  = 1.0f + q0 + p01 * (1.0f + a + a2p + a3) + fork_bias * p0_a0 * (1.0f + q1_a0);
                float c_r52  = tri_probe ? m8_cost(3.0f * wave_b1) : m8_cost(wave_b2 + 2.0f * wave_b1);
                float r_r52  = (ac0 >= 1u && p0_a0 > 0.03f) ? (e_r52 / c_r52) : -1.0f;
                float e_r43  = 1.0f + q0 + p01 * (1.0f + a + a2p) + fork_bias * p0_a0 * (1.0f + q1_a0 * (1.0f + a));
                float c_r43  = tri_probe ? m8_cost(wave_b2 + wave_b1) : m8_cost(2.0f * wave_b2);
                float r_r43  = (ac0 >= 1u && p0_a0 > 0.03f) ? (e_r43 / c_r43) : -1.0f;
                float e_df   = 1.0f + q0 + p01 * (1.0f + a + a2p) + fork_bias * (q0 * p1_a0 + p0_a0 * (1.0f + q1_a0));
                float c_df   = tri_probe ? m8_cost(2.0f * wave_b1) : m8_cost(wave_b2 + wave_b1);
                float r_df   = (ac0 >= 1u && ac1 >= 1u && p0_a0 > 0.03f && p1_a0 > 0.03f) ? (e_df / c_df) : -1.0f;
                float e_r3w  = 1.0f + q0 + p01 * (1.0f + a) + fork_bias * (p0_a0 * (1.0f + q1_a0) + p0_a1 * (1.0f + q1_a1));
                float c_r3w  = tri_probe ? m8_cost(wave_b1) : m8_cost(wave_b3);
                float r_r3w  = (ac0 >= 2u && p0_a1 > 0.03f) ? (e_r3w / c_r3w) : -1.0f;
                // Zero-CB2a-Wave M=8 Trees (unlocked by tri_probe — 0.00 ms CB #2a!)
                float e_t3z  = 1.0f + q0 + p01 + fork_bias * (q0 * p1_a0 + p0_a0 * (1.0f + q1_a0) + p0_a1 * (1.0f + q1_a1));
                float r_t3z  = (tri_probe && ac0 >= 2u && ac1 >= 1u && p0_a1 > 0.03f && p1_a0 > 0.03f) ? (e_t3z / m8_cost(0.0f)) : -1.0f;
                float e_r4w  = 1.0f + q0 + p01 + fork_bias * (p0_a0 * (1.0f + q1_a0) + p0_a1 * (1.0f + q1_a1) + p0_a2);
                float r_r4w  = (tri_probe && ac0 >= 3u && p0_a2 > 0.03f) ? (e_r4w / m8_cost(0.0f)) : -1.0f;

                // M=4 Fast-Verify Topologies (half target attention FLOPs at >=4k; 0 CB #2 draft waves for rf21/rf12/df1/r3w!)
                float e_m4_l3   = 1.0f + q0 + p01 * (1.0f + a);
                float r_m4_l3   = (pre_pos >= 4096) ? (e_m4_l3 / m4_cost(wave_b1)) : -1.0f;
                float e_m4_rf21 = 1.0f + q0 + p01 + fork_bias * p0_a0;
                float r_m4_rf21 = (pre_pos >= 4096 && ac0 >= 1u && p0_a0 > 0.03f) ? (e_m4_rf21 / m4_cost(0.0f)) : -1.0f;
                float e_m4_df1  = 1.0f + q0 + p01 + fork_bias * q0 * p1_a0;
                float r_m4_df1  = (pre_pos >= 4096 && ac1 >= 1u && p1_a0 > 0.03f) ? (e_m4_df1 / m4_cost(0.0f)) : -1.0f;
                float e_m4_r3w  = 1.0f + q0 + fork_bias * (p0_a0 + p0_a1);
                float r_m4_r3w  = (pre_pos >= 4096 && ac0 >= 2u && p0_a1 > 0.03f) ? (e_m4_r3w / m4_cost(0.0f)) : -1.0f;
                float e_m4_rf12 = 1.0f + q0 + fork_bias * p0_a0 * (1.0f + q1_a0);
                float r_m4_rf12 = (pre_pos >= 4096 && tri_probe && ac0 >= 1u && p0_a0 > 0.03f) ? (e_m4_rf12 / m4_cost(0.0f)) : -1.0f;

                float scores[14] = {r_lin7, r_lin5, r_f1, r_r52, r_r43, r_df, r_r3w, r_t3z, r_r4w,
                                    r_m4_l3, r_m4_rf21, r_m4_df1, r_m4_r3w, r_m4_rf12};
                int best = 0;
                for (int k = 1; k < 14; ++k) if (scores[k] > scores[best]) best = k;

                if (best == 0) {
                    linear_fast = true; M = 8;
                    for (int i = 0; i < M; ++i) parents[i] = i - 1;
                    exp_start_d = 2;
                } else if (best == 1) {
                    linear_fast = true; M = 6;
                    for (int i = 0; i < M; ++i) parents[i] = i - 1;
                    exp_start_d = 2;
                } else if (best == 2) {
                    const int tf1[8] = {-1, 0, 1, 2, 3, 1, 5, 6};
                    std::memcpy(parents, tf1, sizeof(tf1));
                    tid_ptr[5] = a1_0; skip_expand[5] = 1;
                    exp_start_d = 2;
                } else if (best == 3) {
                    const int t52[8] = {-1, 0, 1, 2, 3, 4, 0, 6};
                    std::memcpy(parents, t52, sizeof(t52));
                    tid_ptr[6] = a0; skip_expand[6] = 1;
                    if (tri_probe) {
                        tid_ptr[7] = a0_c1; skip_expand[7] = 1;
                        all_dd_mut[6] = all_dd_mut[4];
                        exp_start_d = 2;
                    } else {
                        exp_start_d = 1;
                    }
                } else if (best == 4) {
                    const int t43[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
                    std::memcpy(parents, t43, sizeof(t43));
                    tid_ptr[5] = a0; skip_expand[5] = 1;
                    if (tri_probe) {
                        tid_ptr[6] = a0_c1; skip_expand[6] = 1;
                        all_dd_mut[5] = all_dd_mut[4];
                        exp_kv_slot[5] = 1;
                        exp_kv_mask[5] = 2u;
                        exp_hid_alias[5] = 4; // node 5 (a0)'s MTP hidden sits in seed_hid[4] from CB #1!
                        exp_start_d = 2;
                    } else {
                        exp_start_d = 1;
                    }
                } else if (best == 5) {
                    const int tdf[8] = {-1, 0, 1, 2, 3, 1, 0, 6};
                    std::memcpy(parents, tdf, sizeof(tdf));
                    tid_ptr[5] = a1_0; skip_expand[5] = 1;
                    tid_ptr[6] = a0;   skip_expand[6] = 1;
                    if (tri_probe) {
                        tid_ptr[7] = a0_c1; skip_expand[7] = 1;
                        all_dd_mut[6] = all_dd_mut[4];
                        exp_start_d = 2;
                    } else {
                        exp_start_d = 1;
                    }
                } else if (best == 6) {
                    const int t3w[8] = {-1, 0, 1, 2, 0, 4, 0, 6};
                    std::memcpy(parents, t3w, sizeof(t3w));
                    tid_ptr[4] = a0; skip_expand[4] = 1;
                    tid_ptr[6] = a1; skip_expand[6] = 1;
                    if (tri_probe) {
                        tid_ptr[5] = a0_c1; skip_expand[5] = 1;
                        tid_ptr[7] = a1_c1; skip_expand[7] = 1;
                        exp_start_d = 2; // ONLY 1 B=1 wave (2->3), hidden behind CPU encode!
                    } else {
                        exp_start_d = 1;
                    }
                } else if (best == 7) {
                    // t3_zero: 0 -> { c0 -> {c1, a1_0}, a0 -> a0_c1, a1 -> a1_c1 } (ALL 8 nodes & dists already in place!)
                    std::memcpy(parents, probe3_p, sizeof(probe3_p));
                    exp_start_d = 16; // 0 CB #2a waves!
                } else if (best == 8) {
                    // r4w: 0 -> { c0 -> c1, a0 -> a0_c1, a1 -> a1_c1, a2 } (4-way root coverage, 0 CB #2a waves!)
                    const int tr4w[8] = {-1, 0, 1, 0, 3, 0, 5, 0};
                    std::memcpy(parents, tr4w, sizeof(tr4w));
                    DraftDist dd_a0 = all_dd_mut[4];
                    DraftDist dd_a1 = all_dd_mut[6];
                    tid_ptr[3] = a0;    skip_expand[3] = 1;
                    tid_ptr[4] = a0_c1; skip_expand[4] = 1;
                    tid_ptr[5] = a1;    skip_expand[5] = 1;
                    tid_ptr[6] = a1_c1; skip_expand[6] = 1;
                    tid_ptr[7] = a2;    skip_expand[7] = 1;
                    all_dd_mut[3] = dd_a0;
                    all_dd_mut[5] = dd_a1;
                    exp_start_d = 16; // 0 CB #2a waves!
                } else if (best == 9) {
                    linear_fast = true; M = 4;
                    for (int i = 0; i < M; ++i) parents[i] = i - 1;
                    exp_start_d = 2;
                } else if (best == 10) {
                    M = 4;
                    const int tm4rf[4] = {-1, 0, 1, 0};
                    std::memcpy(parents, tm4rf, sizeof(tm4rf));
                    tid_ptr[3] = a0; skip_expand[3] = 1;
                    exp_start_d = 16; // 0 CB #2 MTP waves!
                } else if (best == 11) {
                    M = 4;
                    const int tm4df[4] = {-1, 0, 1, 1};
                    std::memcpy(parents, tm4df, sizeof(tm4df));
                    tid_ptr[3] = a1_0; skip_expand[3] = 1;
                    exp_start_d = 16; // 0 CB #2 MTP waves!
                } else if (best == 12) {
                    M = 4;
                    const int tm4w[4] = {-1, 0, 0, 0};
                    std::memcpy(parents, tm4w, sizeof(tm4w));
                    tid_ptr[2] = a0; skip_expand[2] = 1;
                    tid_ptr[3] = a1; skip_expand[3] = 1;
                    exp_start_d = 16; // 0 CB #2 MTP waves!
                } else {
                    M = 4;
                    const int tm4rf12[4] = {-1, 0, 0, 2};
                    std::memcpy(parents, tm4rf12, sizeof(tm4rf12));
                    tid_ptr[2] = a0;    skip_expand[2] = 1;
                    tid_ptr[3] = a0_c1; skip_expand[3] = 1;
                    all_dd_mut[2] = all_dd_mut[4];
                    exp_start_d = 16; // 0 CB #2 MTP waves!
                }
            }

            // CB #2: Pipelined split + D2 instant commit of pre-encoded cb2_pre on (M, is_tree) hit!
            double prof_cb2a_gpu = 0.0, prof_cb2b_gpu = 0.0;
            const bool need_expand = (!used_pld || do_hybrid_pld) && (exp_start_d < 16);
            id<MTLCommandBuffer> cb2a = nil;
            if (need_expand) {
                cb2a = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc2a = [cb2a computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                encode_mtp_tree_expand(enc2a, parents, M, pre_draft_mtp_pos, is_greedy, draft_temp, scfg_topk, draft_top_p,
                                       round_seed, skip_expand, exp_start_d, 16, exp_kv_cur, exp_kv_slot, exp_kv_mask, false, exp_hid_alias);
                [enc2a endEncoding];
                [cb2a commit];
            }
            TreeParams final_tp{};
            if (!linear_fast) {
                final_tp = make_tree_params(parents, M, static_cast<uint32_t>(pre_pos));
            }
            const bool pre_hit = (cb2_pre != nil) && (M == pre_M) && ((!linear_fast) == pre_tree) &&
                                 (!pre_tree || final_tp.pad0 <= 2u);
            id<MTLCommandBuffer> cb2 = nil;
            if (pre_hit) {
                if (pre_tree) {
                    std::memcpy([(id<MTLBuffer>)buf_tree_params_ contents], &final_tp, sizeof(final_tp));
                }
                cb2 = cb2_pre;
                [cb2 commit];
            } else {
                if (cb2_pre != nil) {
                    rec_slot_ = saved_rec_slot;
                    pending_replay_m_ = saved_pending_replay_m;
                }
                cb2 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                if (!linear_fast) cur_tree_ = final_tp;
                encode_chunk(enc2, M, pre_pos, true, is_greedy, true, true, v_margin, /*use_tree_buf=*/false);
                cur_tree_ = TreeParams{};
                [enc2 endEncoding];
                [cb2 commit];
            }
            last_t7_m_ = M;
            last_t7_tree_ = !linear_fast;
            [cb2 waitUntilCompleted];
            if (cb2a && cb2a.error) throw std::runtime_error(std::string("Metal T7 expand error: ") + [[cb2a.error localizedDescription] UTF8String]);
            if (cb2.error) throw std::runtime_error(std::string("Metal T7 verify error: ") + [[cb2.error localizedDescription] UTF8String]);

            // D3: Update shared CB #2 non-attention base EMA smoothly from prior (immune to Round-1 cold outliers & M=4/M=8 inversion!)
            const double cb2_ms = (cb2.GPUEndTime - cb2.GPUStartTime) * 1000.0;
            if (cb2_ms > 2.0 && cb2_ms < 500.0 && (M == 4 || M == 8)) {
                const float bw_s = 546.0f / std::clamp(static_cast<float>(ctx_.estimated_bandwidth_gbs()), 60.0f, 1200.0f);
                const int ekv = (sparse_kv_budget_ > 0 && pre_pos > sparse_kv_budget_ + 2048) ? sparse_kv_budget_ : pre_pos;
                const float kvs = (kv_bits_ == 16) ? 1.65f : ((kv_bits_ == 4) ? 0.68f : 1.0f);
                const float est_a = (ekv >= 1024)
                    ? (((M == 4) ? 19.2f : 36.8f) * (static_cast<float>(ekv) / 126976.0f) * kvs * bw_s)
                    : (0.4f * static_cast<float>(ekv) / 1024.0f);
                const float prior_m8 = 38.0f * bw_s;
                const float obs_raw = static_cast<float>(cb2_ms) - est_a + ((M == 4) ? (1.5f * bw_s) : 0.0f);
                const float obs = std::clamp(obs_raw, 0.70f * prior_m8, 1.50f * prior_m8);
                const float cur = (ema_cb2_ms_[1] > 0.0f) ? ema_cb2_ms_[1] : prior_m8;
                ema_cb2_ms_[1] = 0.85f * cur + 0.15f * obs;
            }

            static const bool kMtpProf = (std::getenv("MTP_PROF") != nullptr);
            if (kMtpProf) {
                if (cb2a) prof_cb2a_gpu = (cb2a.GPUEndTime - cb2a.GPUStartTime) * 1000.0;
                prof_cb2b_gpu = cb2_ms;
                const uint32_t* pr0 = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_mtp_argmax_ contents]);
                int tgt0 = static_cast<int>(pr0[0]);
                int hit_rank = (tgt0 == c0) ? 0 : ((ac0 >= 1u && tgt0 == a0) ? 1 : ((ac0 >= 2u && tgt0 == a1) ? 2 : ((ac0 >= 3u && tgt0 == static_cast<int>(d0.alt_id[2])) ? 3 : -1)));
                std::fprintf(stderr, "[mtp_prof] pos=%d M=%d lin=%d pld=%d pre=%d q0=%.3f p0a0=%.3f p0a1=%.3f q1=%.3f ema=%.3f hit0=%d cb2a=%.2f cb2b=%.2f\n",
                             pre_pos, M, linear_fast ? 1 : 0, used_pld ? 1 : 0, pre_hit ? 1 : 0, q0, p0_a0, p0_a1, q1, mtp_alpha_ema_, hit_rank, prof_cb2a_gpu, prof_cb2b_gpu);
            }
        }
        } // end !pre_cb1_suffix
    } else {
        // T3 / T4 / T6: Dynamic Confidence-Guided EAGLE-2 Tree (+ Hybrid 4g/3g PLD+MTP Parallel Tree!)
        @autoreleasepool {
            id<MTLCommandBuffer> cb1 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_root_sync(enc1, 1, -1, -1, true);
            [enc1 endEncoding];
            [cb1 commit];
            [cb1 waitUntilCompleted];
            if (cb1.error) throw std::runtime_error(std::string("Metal tree root sync error: ") + [[cb1.error localizedDescription] UTF8String]);
        }
        pre_draft_mtp_pos = mtp_pos_;

        const DraftDist* d0 = reinterpret_cast<const DraftDist*>([(id<MTLBuffer>)buf_draft_dists_ contents]);
        const float q0 = d0->q_top1;
        const int c0 = static_cast<int>(d0->sampled_id);
        const int a0 = static_cast<int>(d0->alt_id[0]);
        const int a1 = static_cast<int>(d0->alt_id[1]);
        const uint32_t ac = d0->alt_count;

        const float t_hi  = std::getenv("TREE_THI") ? std::strtof(std::getenv("TREE_THI"), nullptr) : 0.94f;
        const float t_mid = std::getenv("TREE_TMID") ? std::strtof(std::getenv("TREE_TMID"), nullptr) : 0.88f;
        const float t_lo  = std::getenv("TREE_TLO") ? std::strtof(std::getenv("TREE_TLO"), nullptr) : 0.68f;
        const float fork_bias = std::getenv("TREE_FBIAS") ? std::strtof(std::getenv("TREE_FBIAS"), nullptr) : 1.15f;

        auto lookup_prob = [](const DraftDist& dd, int tok) -> float {
            for (uint32_t i = 0; i < dd.count && i < 256u; ++i) {
                if (static_cast<int>(dd.cands[i].id) == tok) return dd.cands[i].val;
            }
            return 0.0f;
        };

        int pld4_draft[MAX_VERIFY], pld3_draft[MAX_VERIFY];
        int pld4_k = try_pld_draft(anchor_token, pld4_draft, 7, c0);
        int pld3_k = (tree_mode >= 4 && pld4_k < 3) ? try_pld3_draft(anchor_token, pld3_draft, 3, c0) : 0;

        int skip_expand[16] = {};
        int exp_start_d = 1;
        int exp_kv_cur = 0;
        int exp_kv_slot[16];
        uint32_t exp_kv_mask[16] = {};
        for (int i = 0; i < 16; ++i) exp_kv_slot[i] = -1;

        bool do_hybrid_pld = false;
        if (pld4_k >= 3 && (tree_mode == 3 || pld4_draft[0] == c0)) {
            // Exact 4-gram agreement with MTP root (2-way MTP-guided!) -> full 7-token linear PLD!
            used_pld = true;
            linear_fast = true;
            M = pld4_k + 1;
            for (int d = 0; d < pld4_k; ++d) { tid_ptr[d + 1] = pld4_draft[d]; prefilled[d + 1] = 1; }
            for (int i = 0; i < M; ++i) parents[i] = i - 1;
        } else if (tree_mode >= 4 && (
            (pld4_k >= 3 && q0 < 0.985f) ||
            (pld3_k >= 3 && pld3_draft[0] != c0 && q0 < 0.95f)
        )) {
            // Hybrid PLD + MTP Parallel Tree! Branch A = MTP (4 tok), Branch B = PLD (3 tok)!
            const int* pd = (pld4_k >= 3) ? pld4_draft : pld3_draft;
            const int t_hyb[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
            std::memcpy(parents, t_hyb, sizeof(t_hyb));
            tid_ptr[1] = c0;
            tid_ptr[5] = pd[0]; prefilled[5] = 1; skip_expand[5] = 1;
            tid_ptr[6] = pd[1]; prefilled[6] = 1; skip_expand[6] = 1;
            tid_ptr[7] = pd[2]; prefilled[7] = 1; skip_expand[7] = 1;
            do_hybrid_pld = true;
            used_pld = true;
        } else if (ac == 0u || q0 >= t_hi) {
            if (tree_mode >= 6 && q0 >= 0.985f && mtp_alpha_ema_ >= 0.80f) {
                // Ultra-confident high-EMA code streak (e.g. P2_py_lru): skip probe CB #1b -> direct Depth-7 linear!
                linear_fast = true;
                M = 8;
                for (int i = 0; i < M; ++i) parents[i] = i - 1;
                tid_ptr[1] = c0;
            } else if (tree_mode >= 5) {
                // Multi-Depth Bayesian Expected-Throughput EAGLE-2 M=8 Tree!
                // Probe d=1 and d=2 (only 2 MTP calls = ~1.5 ms), then surgically fork at d=1 or d=2 OR keep adaptive linear!
                tid_ptr[1] = c0;
                const int probe_p[4] = {-1, 0, 1, 2};
                @autoreleasepool {
                    id<MTLCommandBuffer> cb1b = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                    id<MTLComputeCommandEncoder> enc1b = [cb1b computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                    encode_mtp_tree_expand(enc1b, probe_p, 4, pre_draft_mtp_pos, is_greedy, draft_temp,
                                           scfg_topk, draft_top_p, round_seed, nullptr, 1, 3, 0, nullptr, nullptr, true);
                    [enc1b endEncoding];
                    [cb1b commit];
                    [cb1b waitUntilCompleted];
                    if (cb1b.error) throw std::runtime_error(std::string("Metal probe error: ") + [[cb1b.error localizedDescription] UTF8String]);
                }
                const DraftDist* all_dd = reinterpret_cast<const DraftDist*>([(id<MTLBuffer>)buf_draft_dists_ contents]);
                const DraftDist& d1 = all_dd[1];
                const DraftDist& d2 = all_dd[2];
                const float q1 = d1.q_top1;
                const float q2 = d2.q_top1;
                const float p1_a0 = (d1.alt_count >= 1u) ? lookup_prob(d1, static_cast<int>(d1.alt_id[0])) : 0.0f;
                const float p2_a0 = (d2.alt_count >= 1u) ? lookup_prob(d2, static_cast<int>(d2.alt_id[0])) : 0.0f;
                const float a = std::clamp(mtp_alpha_ema_, 0.50f, 0.92f);

                const int lin_k = (tree_mode >= 6) ? std::clamp(cur_mtp_depth_, 5, 7) : 7;
                const float base_e = 1.0f + q0 + q0 * q1 + q0 * q1 * q2;
                const float p012 = q0 * q1 * q2;
                float lin_tail = 0.0f, pa = a;
                for (int j = 3; j < lin_k; ++j) { lin_tail += pa; pa *= a; }
                const float e_lin = base_e + p012 * lin_tail;
                const float e_f1  = base_e + p012 * a + fork_bias * q0 * p1_a0 * (1.0f + a + a * a);
                const float e_f2  = base_e + p012 * (a + a * a) + fork_bias * (q0 * q1) * p2_a0 * (1.0f + a);

                const float lin_ms = 43.4f + 0.78f * static_cast<float>(lin_k - 3);
                const float r_lin = e_lin / lin_ms;
                const float r_f1  = (d1.alt_count >= 1u && p1_a0 > 0.04f) ? (e_f1 / 45.0f) : -1.0f;
                const float r_f2  = (d2.alt_count >= 1u && p2_a0 > 0.04f) ? (e_f2 / 45.0f) : -1.0f;

                exp_kv_cur = 2;
                exp_kv_slot[1] = 0; exp_kv_mask[1] = 1u;
                exp_kv_slot[2] = 1; exp_kv_mask[2] = 3u;
                skip_expand[1] = 1; skip_expand[2] = 1; skip_expand[3] = 1;

                if (r_f1 >= r_lin && r_f1 >= r_f2) {
                    const int tf1[8] = {-1, 0, 1, 2, 3, 1, 5, 6};
                    std::memcpy(parents, tf1, sizeof(tf1));
                    tid_ptr[5] = static_cast<int>(d1.alt_id[0]);
                    skip_expand[5] = 1;
                    exp_start_d = 2;
                } else if (r_f2 >= r_lin) {
                    const int tf2[8] = {-1, 0, 1, 2, 3, 4, 2, 6};
                    std::memcpy(parents, tf2, sizeof(tf2));
                    tid_ptr[6] = static_cast<int>(d2.alt_id[0]);
                    skip_expand[6] = 1;
                    exp_start_d = 3;
                } else {
                    linear_fast = true;
                    M = lin_k + 1;
                    for (int i = 0; i < M; ++i) parents[i] = i - 1;
                    exp_start_d = 3;
                }
            } else {
                // T4-v2: Very confident root -> adaptive Depth-5..7 linear chain!
                linear_fast = true;
                M = std::clamp(cur_mtp_depth_, 5, 7) + 1;
                for (int i = 0; i < M; ++i) parents[i] = i - 1;
                tid_ptr[1] = c0;
            }
        } else if (tree_mode >= 6 && q0 >= t_mid) {
            // Mild root uncertainty (0.84..0.94): Asymmetric 5+2 Root-Fork (preserves Depth-5 primary + 2-tok backup!)
            const int t52[8] = {-1, 0, 1, 2, 3, 4, 0, 6};
            std::memcpy(parents, t52, sizeof(t52));
            tid_ptr[1] = c0;
            tid_ptr[6] = a0;
        } else if (ac < 2u || q0 >= t_lo) {
            // Moderate root uncertainty -> Root-Fork 4+3: 0->1(c0)->2->3->4, 0->5(a0)->6->7
            const int t1[8] = {-1, 0, 1, 2, 3, 0, 5, 6};
            std::memcpy(parents, t1, sizeof(t1));
            tid_ptr[1] = c0;
            tid_ptr[5] = a0;
        } else {
            // High root uncertainty -> 3-Way Root-Fork 3+2+2: 0->1(c0)->2->3, 0->4(a0)->5, 0->6(a1)->7
            const int t3w[8] = {-1, 0, 1, 2, 0, 4, 0, 6};
            std::memcpy(parents, t3w, sizeof(t3w));
            tid_ptr[1] = c0;
            tid_ptr[4] = a0;
            tid_ptr[6] = a1;
        }

        // CB #2: Batched tree expansion + M=8 target verify!
        @autoreleasepool {
            id<MTLCommandBuffer> cb2 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            if (!used_pld || do_hybrid_pld) {
                encode_mtp_tree_expand(enc2, parents, M, pre_draft_mtp_pos, is_greedy, draft_temp, scfg_topk, draft_top_p,
                                       round_seed, skip_expand, exp_start_d, 16, exp_kv_cur,
                                       (exp_kv_cur > 0) ? exp_kv_slot : nullptr,
                                       (exp_kv_cur > 0) ? exp_kv_mask : nullptr);
            }
            if (!linear_fast) {
                cur_tree_ = make_tree_params(parents, M, static_cast<uint32_t>(pre_pos));
            }
            encode_chunk(enc2, M, pre_pos, true, is_greedy, true, true, v_margin);
            cur_tree_ = TreeParams{};
            [enc2 endEncoding];
            [cb2 commit];
            [cb2 waitUntilCompleted];
            if (cb2.error) throw std::runtime_error(std::string("Metal dynamic tree verify error: ") + [[cb2.error localizedDescription] UTF8String]);
        }
    }

    // Exact Lossless Tree Verification (Greedy = bit-exact argmax, Sampled = SpecInfer Theorem 1!)
    int children[16][4];
    int n_child[16] = {};
    for (int t = 1; t < M; ++t) {
        int p = parents[t];
        if (p >= 0 && n_child[p] < 4) children[p][n_child[p]++] = t;
    }

    const uint32_t* preds = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_mtp_argmax_ contents]);
    const float* all_logits = reinterpret_cast<const float*>([(id<MTLBuffer>)buf_mtp_logits_ contents]);
    const LogitCand* all_cands = reinterpret_cast<const LogitCand*>([(id<MTLBuffer>)buf_verify_cands_ contents]);
    const uint32_t* cand_cnts = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_verify_cnt_ contents]);
    const DraftDist* draft_dists = reinterpret_cast<const DraftDist*>([(id<MTLBuffer>)buf_draft_dists_ contents]);

    uint32_t win_path[16];
    int win_len = 0;
    win_path[win_len++] = 0u;
    int cur = 0;
    int accepted = 0;
    int bonus_tok = 0;

    while (true) {
        int nc = n_child[cur];
        if (nc == 0) {
            if (is_greedy) {
                bonus_tok = static_cast<int>(preds[cur]);
            } else {
                const float* row_lg = all_logits + static_cast<size_t>(cur) * vocab_size_;
                const LogitCand* row_cands = all_cands + static_cast<size_t>(cur) * MAX_VERIFY_CANDS;
                bonus_tok = sampler->sample_from_cands(row_cands, cand_cnts[cur], MAX_VERIFY_CANDS, row_lg, vocab_size_);
            }
            out_tokens.push_back(bonus_tok);
            break;
        }

        int ch_toks[4];
        for (int b = 0; b < nc; ++b) ch_toks[b] = tid_ptr[children[cur][b]];

        if (is_greedy) {
            int target_tok = static_cast<int>(preds[cur]);
            int hit = -1;
            for (int b = 0; b < nc; ++b) {
                if (ch_toks[b] == target_tok) { hit = b; break; }
            }
            if (hit >= 0) {
                int nxt = children[cur][hit];
                out_tokens.push_back(target_tok);
                win_path[win_len++] = static_cast<uint32_t>(nxt);
                ++accepted;
                cur = nxt;
            } else {
                bonus_tok = target_tok;
                out_tokens.push_back(bonus_tok);
                break;
            }
        } else {
            const float* row_lg = all_logits + static_cast<size_t>(cur) * vocab_size_;
            const LogitCand* row_cands = all_cands + static_cast<size_t>(cur) * MAX_VERIFY_CANDS;
            const DraftDist* dd0 = (prefilled[children[cur][0]] == 0) ? &draft_dists[cur] : nullptr;
            int chosen_b = -1;
            int tok = sampler->verify_tree_node_dist(
                row_cands, cand_cnts[cur], MAX_VERIFY_CANDS, row_lg, vocab_size_,
                ch_toks, nc, dd0, chosen_b
            );
            if (chosen_b >= 0) {
                int nxt = children[cur][chosen_b];
                out_tokens.push_back(ch_toks[chosen_b]);
                sampler->observe(ch_toks[chosen_b]);
                win_path[win_len++] = static_cast<uint32_t>(nxt);
                ++accepted;
                cur = nxt;
            } else {
                bonus_tok = tok;
                out_tokens.push_back(bonus_tok);
                break;
            }
        }
    }

    add_draft_token(bonus_tok);
    maybe_widen_draft_vocab(bonus_tok);
    if (!used_pld) {
        int ndepth[16] = {};
        int max_d = 1;
        for (int t = 1; t < M; ++t) {
            ndepth[t] = (parents[t] >= 0) ? (ndepth[parents[t]] + 1) : 1;
            if (ndepth[t] > max_d) max_d = ndepth[t];
        }
        update_mtp_depth_ema(accepted, max_d);
    }

    pos_ = pre_pos + win_len;
    for (int j = 0; j < win_len; ++j) push_history_token(tid_ptr[win_path[j]]);

    bool is_contig = true;
    for (int j = 0; j < win_len; ++j) {
        if (win_path[j] != static_cast<uint32_t>(j)) { is_contig = false; break; }
    }

    // Lazy Fused GDN Replay: always commit win_len accepted tokens inside next round's verify pass (zero speculative state writes!)
    pending_replay_m_ = win_len;

    pending_sync_ = true;
    pending_path_noncontig_ = !is_contig;
    pending_kv_pre_pos_ = pre_pos;
    pending_path_len_ = static_cast<uint32_t>(win_len);
    for (int j = 0; j < win_len; ++j) pending_path_[j] = win_path[j];
    pending_seed_first_ = false;
    pending_mtp_pos_ = pre_draft_mtp_pos;
    pending_feed_count_ = win_len;
    pending_feed_off_ = 0;
    for (int j = 0; j < accepted; ++j) pending_feed_tokens_[j] = tid_ptr[win_path[j + 1]];
    pending_feed_tokens_[accepted] = bonus_tok;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = accepted + 1;
        stats->mtp_accepted = accepted;
        stats->mtp_drafted = M - 1;
        stats->pld_hits = used_pld ? 1 : 0;
    }
}

void Qwen38Model::encode_dflash_capture(
    id enc_id, id src_hid_buf, int src_row_off, int dst_row_off, int n_rows, int slot_idx
) {
    if (n_rows <= 0 || slot_idx < 0) return;
    id<MTLComputeCommandEncoder> enc = enc_id;
    const DFlashConfig& dcfg = dflash_->config();
    DFlashCaptureParams cp{
        static_cast<uint32_t>(n_rows),
        static_cast<uint32_t>(hidden_size_),
        static_cast<uint32_t>(dcfg.target_layer_ids.size()),
        static_cast<uint32_t>(slot_idx),
        static_cast<uint32_t>(src_row_off),
        static_cast<uint32_t>(dst_row_off)
    };
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_capture_hidden_];
    [enc setBuffer:(id<MTLBuffer>)src_hid_buf offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)buf_df_capture_ offset:0 atIndex:1];
    [enc setBytes:&cp length:sizeof(cp) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(ceil_div(cp.D, 4u), cp.M, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Qwen38Model::encode_dflash_context_commit(id enc_id, int cap_row_off, int n_rows, int pos_start) {
    if (n_rows <= 0) return;
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    const DFlashConfig& dcfg = dflash_->config();
    const uint32_t H = static_cast<uint32_t>(dcfg.hidden_size);
    const uint32_t Hkv = static_cast<uint32_t>(dcfg.num_kv_heads);
    const uint32_t D_h = static_cast<uint32_t>(dcfg.head_dim);
    const uint32_t win = static_cast<uint32_t>(dcfg.sliding_window);
    const size_t n_tgt = dcfg.target_layer_ids.size();
    const Q4Weight fc_w = as_q4(dflash_->fc());
    const TensorView& hnorm = dflash_->hidden_norm();

    int done = 0;
    while (done < n_rows) {
        const int sub = std::min(n_rows - done, static_cast<int>(c_attn_));
        const uint32_t sub_u = static_cast<uint32_t>(sub);
        const size_t cap_off = static_cast<size_t>(cap_row_off + done) * n_tgt * hidden_size_ * sizeof(uint16_t);

        // 1. fc([sub, 25600] FP16) -> buf_qkv_ ([sub, 5120] FP32)
        encode_qmm_q4_f16x(enc, fc_w, buf_df_capture_, cap_off, buf_qkv_, 0, sub_u, false);
        barrier();

        // 2. Standard Qwen3 RMSNorm (NO +1.0) with hidden_norm -> buf_normed_ ([sub, 5120] FP16)
        DFlashNormParams np{sub_u, H, dcfg.rms_eps};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)hnorm.buffer offset:hnorm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(sub_u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // 3. Project context K/V for all 5 DFlash layers -> k_norm + full-dim D=128 RoPE -> FP16 ring buffer!
        DFlashKvWriteParams kp{
            sub_u, Hkv, D_h, static_cast<uint32_t>(pos_start + done), win, dcfg.rope_theta, dcfg.rms_eps
        };
        for (int l = 0; l < dcfg.num_layers; ++l) {
            const DFlashLayerWeights& dl = dflash_->layer(l);
            encode_qmm_q4_f16x(enc, as_q4(dl.k_proj), buf_normed_, 0, buf_k_, 0, sub_u, false);
            encode_qmm_q4_f16x(enc, as_q4(dl.v_proj), buf_normed_, 0, buf_v_, 0, sub_u, false);
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_kv_write_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)dl.k_norm.buffer offset:dl.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)df_k_cache_[l] offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)df_v_cache_[l] offset:0 atIndex:4];
            [enc setBytes:&kp length:sizeof(kp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(Hkv, sub_u, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();
        }
        done += sub;
    }
}

void Qwen38Model::encode_dflash_draft_step(id enc_id, int start_pos) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    const DFlashConfig& dcfg = dflash_->config();
    const uint32_t B   = static_cast<uint32_t>(dcfg.block_size);
    const uint32_t P   = B - 1u;
    const uint32_t H   = static_cast<uint32_t>(dcfg.hidden_size);
    const uint32_t Hq  = static_cast<uint32_t>(dcfg.num_heads);
    const uint32_t Hkv = static_cast<uint32_t>(dcfg.num_kv_heads);
    const uint32_t D_h = static_cast<uint32_t>(dcfg.head_dim);
    const uint32_t gs  = static_cast<uint32_t>(dcfg.conv_group_size);
    const uint32_t win = static_cast<uint32_t>(dcfg.sliding_window);
    const float eps    = dcfg.rms_eps;

    // 0. Deferred previous-step context commit (merges prior accepted target hiddens into this single CB!)
    if (df_pending_commit_ > 0) {
        encode_dflash_context_commit(enc, 0, df_pending_commit_, df_pending_pos_);
        df_kv_len_ = df_pending_pos_ + df_pending_commit_;
        df_pending_commit_ = 0;
    }

    // 1. Embed [anchor_id, MASK x (B-1)] using target embed_tokens_ -> buf_df_h_ (FP32 [B, H])
    {
        EmbedParams ep{B, H, eps};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.w.buffer offset:embed_tokens_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.s.buffer offset:embed_tokens_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_tokens_.b.buffer offset:embed_tokens_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_ids_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    const uint32_t eff_ctx = static_cast<uint32_t>(std::min(df_kv_len_, dcfg.sliding_window));
    const uint32_t num_splits = (eff_ctx > 256u) ? 8u : 1u;
    const DFlashNormParams np_b{B, H, eps};
    const DFlashConvParams cp_prep{B, H, gs, 0u};
    const DFlashConvParams cp_fin{B, H, gs, 1u};
    const DFlashQkvPrepParams qkvp{B, Hq, Hkv, D_h, static_cast<uint32_t>(start_pos), dcfg.rope_theta, eps};
    const DFlashAttnParams ap{
        B, Hq, Hkv, D_h, static_cast<uint32_t>(start_pos), static_cast<uint32_t>(df_kv_len_),
        win, num_splits, 1.0f / std::sqrt(static_cast<float>(dcfg.head_dim))
    };

    // 2. 5 DFlash 2 Bidirectional Sliding-Window Decoder Layers with GroupedDynamicCausalConv
    for (int l = 0; l < dcfg.num_layers; ++l) {
        const DFlashLayerWeights& dl = dflash_->layer(l);

        // 2a. Input RMSNorm -> attn_conv kernel_projection -> conv prepare (stage=0)
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dl.input_layernorm.buffer offset:dl.input_layernorm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:2];
        [enc setBytes:&np_b length:sizeof(np_b) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, as_q4(dl.attn_kernel_proj), buf_df_x_, 0, buf_df_dyn_, 0, B, false);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_conv_prep_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_dyn_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dl.attn_base_kernel.buffer offset:dl.attn_base_kernel.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x2_ offset:0 atIndex:3];
        [enc setBytes:&cp_prep length:sizeof(cp_prep) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // 2b. Concurrent Q, K, V projections -> q_norm/k_norm + full-dim D=128 RoPE
        encode_qmm_q4_f16x(enc, as_q4(dl.q_proj), buf_df_x2_, 0, buf_q_, 0, B, false);
        encode_qmm_q4_f16x(enc, as_q4(dl.k_proj), buf_df_x2_, 0, buf_k_, 0, B, false);
        encode_qmm_q4_f16x(enc, as_q4(dl.v_proj), buf_df_x2_, 0, buf_v_, 0, B, false);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_qkv_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)dl.q_norm.buffer offset:dl.q_norm.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)dl.k_norm.buffer offset:dl.k_norm.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_df_q_h_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_df_kn_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_df_vn_ offset:0 atIndex:7];
        [enc setBytes:&qkvp length:sizeof(qkvp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(Hq + Hkv, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();

        // 2c. Bidirectional Sliding-Window GQA-4 AMX Attention (ring + noise; does NOT write noise to ring!)
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_attn_p1_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_q_h_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)df_k_cache_[l] offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)df_v_cache_[l] offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_kn_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_df_vn_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)((num_splits == 1u) ? buf_df_attn_out_ : buf_split_o_) offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(Hkv, num_splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();

        if (num_splits > 1u) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_attn_p2_];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_df_attn_out_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(Hq, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();
        }

        // 2d. o_proj -> attn_conv finish + residual add into buf_df_h_
        encode_qmm_q4_f16x(enc, as_q4(dl.o_proj), buf_df_attn_out_, 0, buf_df_o_, 0, B, false);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_conv_finish_f32_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_o_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_dyn_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dl.attn_base_kernel.buffer offset:dl.attn_base_kernel.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:0 atIndex:3];
        [enc setBytes:&cp_fin length:sizeof(cp_fin) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // 2e. Post-Attention RMSNorm -> mlp_conv kernel_projection -> conv prepare (stage=0)
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dl.post_attention_layernorm.buffer offset:dl.post_attention_layernorm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:2];
        [enc setBytes:&np_b length:sizeof(np_b) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, as_q4(dl.mlp_kernel_proj), buf_df_x_, 0, buf_df_dyn_, 0, B, false);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_conv_prep_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_dyn_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dl.mlp_base_kernel.buffer offset:dl.mlp_base_kernel.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x2_ offset:0 atIndex:3];
        [enc setBytes:&cp_prep length:sizeof(cp_prep) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // 2f. Fused SwiGLU (gate + up + SiLU -> FP16 buf_mlp_gate_) -> Split-K down_proj -> mlp_conv finish + residual
        QmmParams qp_mlp{B, dl.gate_proj.out_dim, dl.gate_proj.in_dim, 0u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_swiglu_f16_];
        [enc setBuffer:(id<MTLBuffer>)dl.gate_proj.w.buffer offset:dl.gate_proj.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)dl.gate_proj.s.buffer offset:dl.gate_proj.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dl.gate_proj.b.buffer offset:dl.gate_proj.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)dl.up_proj.w.buffer offset:dl.up_proj.w.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)dl.up_proj.s.buffer offset:dl.up_proj.s.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)dl.up_proj.b.buffer offset:dl.up_proj.b.offset atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x2_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:7];
        [enc setBytes:&qp_mlp length:sizeof(qp_mlp) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(dl.gate_proj.out_dim, 16u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_qmm_q4_f16x(enc, as_q4(dl.down_proj), buf_mlp_gate_, 0, buf_df_o_, 0, B, false);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_conv_finish_f32_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_o_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_dyn_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)dl.mlp_base_kernel.buffer offset:dl.mlp_base_kernel.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:0 atIndex:3];
        [enc setBytes:&cp_fin length:sizeof(cp_fin) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(H, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    // 3. Final Norm on proposal rows 1..B-1 (P = 7 rows!) -> concurrent target lm_head + selector hidden_proj -> exact Top-16
    {
        const TensorView& fnorm = dflash_->norm();
        DFlashNormParams np_p{P, H, eps};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_rmsnorm_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_h_ offset:static_cast<size_t>(H) * sizeof(float) atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)fnorm.buffer offset:fnorm.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:2];
        [enc setBytes:&np_p length:sizeof(np_p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(P, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    encode_qmm_q4_f16x(enc, lm_head_, buf_df_x_, 0, buf_mtp_logits_, 0, P, false);
    {
        const TensorView& hp = dflash_->selector_hidden_proj();
        DFlashHiddenProjParams hpp{P, static_cast<uint32_t>(dcfg.selector_rank), H};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_hidden_proj_f16_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_x_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)hp.buffer offset:hp.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_df_sel_h_ offset:0 atIndex:2];
        [enc setBytes:&hpp length:sizeof(hpp) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(hpp.R, 4u), ceil_div(P, 8u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    barrier();

    {
        const uint32_t V = static_cast<uint32_t>(vocab_size_);
        const uint32_t K = static_cast<uint32_t>(dcfg.selector_top_k);
        const uint32_t num_chunks = ceil_div(V, 2048u);
        DFlashTopKParams tkp{P, V, K, num_chunks};

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_topk16_p1_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_topk_part_ offset:0 atIndex:1];
        [enc setBytes:&tkp length:sizeof(tkp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(num_chunks, P, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_df_topk16_p2_];
        [enc setBuffer:(id<MTLBuffer>)buf_df_topk_part_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_df_topk_out_ offset:0 atIndex:1];
        [enc setBytes:&tkp length:sizeof(tkp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, P, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }
}

void Qwen38Model::forward_dflash_impl(int anchor_token, Sampler* sampler, std::vector<int>& out_tokens, StepStats* stats) {
    const DFlashConfig& dcfg = dflash_->config();
    const int B = dcfg.block_size; // 8
    const int K = B - 1;           // 7
    const bool is_greedy = (sampler == nullptr || sampler->is_greedy());
    const float v_margin = is_greedy ? 0.0f : sampler->compact_margin();
    const int pre_pos = pos_;
    ensure_kv_capacity(pre_pos + B);

    auto t0 = std::chrono::steady_clock::now();

    int draft[MAX_VERIFY] = {};
    DraftDist df_dists[MAX_VERIFY] = {};
    bool use_sampled_q = false;
    bool used_pld = false;
    double cb1_gpu_ms = 0.0, cb2_gpu_ms = 0.0, sel_us = 0.0;

    // Lossless Greedy SuffixCache / PLD Hybrid Fast-Path:
    // On high-confidence, unambiguous long suffix matches (full K-token continuation), bypass the
    // ~5 ms DFlash 5L+lm_head CB #1 and verify directly in a SINGLE CB (while flushing any pending
    // DFlash context KV before encode_chunk overwrites buf_df_capture_ so DFlash stays 100% synced!).
    static const bool kDfPldAllow = !(std::getenv("DF_PLD") && std::atoi(std::getenv("DF_PLD")) == 0);
    if (is_greedy && pld_enabled_ && kDfPldAllow) {
        SuffixResult sr0 = suffix_cache_.lookup(token_history_, anchor_token);
        if (sr0.n_branches == 1 && sr0.b[0].cont_len >= K) {
            const SuffixBranch& top = sr0.b[0];
            const bool pld_strong = top.from_session ? (top.match_len >= 12) : (top.match_len >= 16);
            if (pld_strong) {
                for (int d = 0; d < K; ++d) draft[d] = top.cont[d];
                used_pld = true;
            }
        }
    }

    @autoreleasepool {
        if (used_pld) {
            // Single-CB PLD Verify: [anchor_token, draft[0..K-1]] verified linearly (100% greedy-exact!).
            int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);
            tid_ptr[0] = anchor_token;
            for (int d = 0; d < K; ++d) tid_ptr[d + 1] = draft[d];

            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            if (df_pending_commit_ > 0) {
                encode_dflash_context_commit(enc, 0, df_pending_commit_, df_pending_pos_);
                df_kv_len_ = df_pending_pos_ + df_pending_commit_;
                df_pending_commit_ = 0;
            }
            encode_chunk(enc, B, pre_pos, true, true, true, true, 0.0f, /*use_tree_buf=*/false,
                         /*df_cap_src_row=*/0, /*df_cap_n_rows=*/B);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                throw std::runtime_error(std::string("Metal DFlash-PLD verify CB error: ") +
                                         [[cb.error localizedDescription] UTF8String]);
            }
            cb2_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        } else {
            // Populate [anchor_token, MASK x (B-1)] for DFlash block forward
            int* df_ids = reinterpret_cast<int*>([(id<MTLBuffer>)buf_df_ids_ contents]);
            df_ids[0] = anchor_token;
            for (int i = 1; i < B; ++i) df_ids[i] = dcfg.mask_token_id;

            // CB #1: Single-CB deferred context commit + 5L DFlash block + target lm_head (M=7) + Top-16 + hidden_proj!
            id<MTLCommandBuffer> cb1 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_dflash_draft_step(enc1, pre_pos);
            [enc1 endEncoding];
            [cb1 commit];

            // 100%-hit speculative CB #2 pre-encoding on CPU WHILE GPU executes CB #1 (hides ~0.65 ms CPU encode for free!)
            const int saved_rec_slot = rec_slot_;
            const int saved_pending_replay_m = pending_replay_m_;
            id<MTLCommandBuffer> cb2 = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_chunk(enc2, B, pre_pos, true, is_greedy, true, true, v_margin, /*use_tree_buf=*/false,
                         /*df_cap_src_row=*/0, /*df_cap_n_rows=*/B);
            [enc2 endEncoding];

            [cb1 waitUntilCompleted];
            if (cb1.error) {
                rec_slot_ = saved_rec_slot;
                pending_replay_m_ = saved_pending_replay_m;
                throw std::runtime_error(std::string("Metal DFlash draft CB error: ") +
                                         [[cb1.error localizedDescription] UTF8String]);
            }
            cb1_gpu_ms = (cb1.GPUEndTime - cb1.GPUStartTime) * 1000.0;

            // CPU ARM64 NEON CandidateSelector (<20 us, 0 B Metal wired codebook RAM!)
            const auto ts0 = std::chrono::steady_clock::now();
            const LogitCand* topk_cands = reinterpret_cast<const LogitCand*>([(id<MTLBuffer>)buf_df_topk_out_ contents]);
            const uint16_t* sel_h_f16   = reinterpret_cast<const uint16_t*>([(id<MTLBuffer>)buf_df_sel_h_ contents]);
            if (is_greedy || sampler->config().temperature <= 0.0f) {
                dflash_->select_greedy(anchor_token, topk_cands, sel_h_f16, draft);
                use_sampled_q = false;
            } else {
                dflash_->select_sampled(anchor_token, topk_cands, sel_h_f16,
                                        sampler->config().temperature, sampler->next_seed(), draft, df_dists);
                use_sampled_q = true;
            }
            sel_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ts0).count();

            // Write [anchor_token, draft[0..K-1]] into shared buf_token_ids_ BEFORE committing pre-encoded CB #2!
            int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);
            tid_ptr[0] = anchor_token;
            for (int d = 0; d < K; ++d) tid_ptr[d + 1] = draft[d];

            [cb2 commit];
            [cb2 waitUntilCompleted];
            if (cb2.error) {
                throw std::runtime_error(std::string("Metal DFlash verify CB error: ") +
                                         [[cb2.error localizedDescription] UTF8String]);
            }
            cb2_gpu_ms = (cb2.GPUEndTime - cb2.GPUStartTime) * 1000.0;
        }
    }

    // Exact Verification (Greedy = 100% token-for-token exact AR; Sampled = exact Leviathan rejection sampling!)
    int accepted = 0;
    int bonus_tok = 0;
    if (is_greedy) {
        const uint32_t* preds = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_mtp_argmax_ contents]);
        for (int d = 0; d < K; ++d) {
            if (static_cast<int>(preds[d]) == draft[d]) {
                out_tokens.push_back(draft[d]);
                ++accepted;
            } else {
                break;
            }
        }
        bonus_tok = static_cast<int>(preds[accepted]);
        out_tokens.push_back(bonus_tok);
    } else {
        const float* all_logits    = reinterpret_cast<const float*>([(id<MTLBuffer>)buf_mtp_logits_ contents]);
        const LogitCand* all_cands = reinterpret_cast<const LogitCand*>([(id<MTLBuffer>)buf_verify_cands_ contents]);
        const uint32_t* cand_cnts  = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_verify_cnt_ contents]);
        bool rejected = false;
        for (int d = 0; d < K; ++d) {
            const float* row_lg        = all_logits + static_cast<size_t>(d) * vocab_size_;
            const LogitCand* row_cands = all_cands + static_cast<size_t>(d) * MAX_VERIFY_CANDS;
            const DraftDist* dd        = use_sampled_q ? &df_dists[d] : nullptr;
            bool ok = false;
            int tok = sampler->sample_or_accept_draft_dist(
                row_cands, cand_cnts[d], MAX_VERIFY_CANDS, row_lg, vocab_size_, draft[d], dd, ok
            );
            if (ok) {
                out_tokens.push_back(draft[d]);
                sampler->observe(draft[d]);
                ++accepted;
            } else {
                bonus_tok = tok;
                out_tokens.push_back(bonus_tok);
                rejected = true;
                break;
            }
        }
        if (!rejected) {
            const float* last_lg        = all_logits + static_cast<size_t>(K) * vocab_size_;
            const LogitCand* last_cands = all_cands + static_cast<size_t>(K) * MAX_VERIFY_CANDS;
            bonus_tok = sampler->sample_from_cands(last_cands, cand_cnts[K], MAX_VERIFY_CANDS, last_lg, vocab_size_);
            out_tokens.push_back(bonus_tok);
        }
    }

    pos_ = pre_pos + accepted + 1;
    push_history_token(anchor_token);
    for (int j = 0; j < accepted; ++j) push_history_token(draft[j]);

    // Defer Lazy Fused GDN Replay + DFlash Context KV Commit of accepted rows [0..accepted] to next round!
    pending_sync_ = true;
    pending_replay_m_ = accepted + 1;
    pending_path_noncontig_ = false;
    pending_path_len_ = static_cast<uint32_t>(accepted + 1);
    for (int j = 0; j <= accepted; ++j) pending_path_[j] = static_cast<uint32_t>(j);
    df_pending_commit_ = accepted + 1;
    df_pending_pos_ = pre_pos;

    static const bool kDfProf = (std::getenv("DF_PROF") != nullptr);
    if (kDfProf) {
        std::fprintf(stderr, "[df_prof] pos=%d acc=%d/%d pld=%d cb1=%.2fms sel=%.1fus cb2=%.2fms\n",
                     pre_pos, accepted, K, used_pld ? 1 : 0, cb1_gpu_ms, sel_us, cb2_gpu_ms);
    }

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = accepted + 1;
        stats->mtp_accepted = accepted;
        stats->mtp_drafted = K;
        stats->pld_hits = used_pld ? 1 : 0;
    }
}

void Qwen38Model::forward_mtp_greedy(int anchor_token, std::vector<int>& out_tokens, StepStats* stats) {
    if (dflash_loaded_) {
        if (pos_ + dflash_->config().block_size > max_seq_len_) {
            int nxt = forward_greedy(anchor_token, stats);
            out_tokens.push_back(nxt);
            return;
        }
        forward_dflash_impl(anchor_token, nullptr, out_tokens, stats);
        return;
    }
    if (!mtp_loaded_ || pos_ + MAX_VERIFY > max_seq_len_) {
        int nxt = forward_greedy(anchor_token, stats);
        out_tokens.push_back(nxt);
        return;
    }
    const int tree_mode = std::getenv("MTP_TREE") ? std::atoi(std::getenv("MTP_TREE")) : 6;
    if (tree_mode > 0 && (pending_sync_ || !mtp_seeded_)) {
        forward_mtp_tree_impl(anchor_token, nullptr, out_tokens, stats);
        return;
    }
    ensure_kv_capacity(pos_ + MAX_VERIFY);

    auto t0 = std::chrono::steady_clock::now();
    const int pre_pos = pos_;
    int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);

    // 1. Try zero-cost 4-gram Prompt-Lookup Drafting (PLD) up to 7 tokens!
    int pld_draft[MAX_VERIFY];
    int pld_k = try_pld_draft(anchor_token, pld_draft, 7);
    bool used_pld = (pld_k >= 2);
    int K = used_pld ? pld_k : cur_mtp_depth_;
    int M = K + 1;

    // 2. ONE Single Command Buffer per decode round (ZERO CB #2 round-trip!):
    //    - Deferred sync from previous round (optional GDN replay + MTP KV update; skips 184 MB draft lm_head on PLD!)
    //    - If not PLD: MTP drafts d=1..K-1 writing directly into buf_token_ids_[2..K]
    //    - Full 64-layer target verify on all M = K + 1 tokens + lm_head + GPU argmax!
    tid_ptr[0] = anchor_token;
    if (used_pld) {
        for (int d = 0; d < K; ++d) tid_ptr[d + 1] = pld_draft[d];
    } else if (!pending_sync_ && mtp_seeded_) {
        tid_ptr[1] = mtp_seed_token_;
    }

    if (pending_sync_) {
        int* sync_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_mtp_sync_ids_ contents]);
        std::memcpy(sync_ptr, pending_feed_tokens_,
                    static_cast<size_t>(pending_feed_off_ + pending_feed_count_) * sizeof(int));
    }

    bool was_seeded = mtp_seeded_;
    int pre_draft_mtp_pos = mtp_pos_;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        if (pending_sync_) {
            // Note: GDN replay (pending_replay_m_) is lazily fused into encode_chunk() below!
            int cur_pos = pending_mtp_pos_;
            if (pending_seed_first_) {
                encode_mtp_steps(enc, buf_mtp_sync_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_pos, false);
                cur_pos += 1;
            }
            // Write draft[0] directly into buf_token_ids_[1] on GPU if !used_pld; skip draft lm_head entirely on PLD!
            encode_mtp_steps(enc, buf_mtp_sync_ids_, static_cast<size_t>(pending_feed_off_) * sizeof(int),
                             buf_mtp_verify_hid_, 0, pending_feed_count_, cur_pos,
                             !used_pld, buf_token_ids_, sizeof(uint32_t));
            mtp_pos_ = cur_pos + pending_feed_count_;
            mtp_seeded_ = true;
            was_seeded = true;
            pre_draft_mtp_pos = mtp_pos_;
            pending_sync_ = false;
        }

        if (!used_pld) {
            int cur_mtp_pos = mtp_pos_;
            if (!mtp_seeded_) {
                // Seed (anchor_token, buf_mtp_last_hid_) -> draft[0] directly into buf_token_ids_[1]
                encode_mtp_steps(enc, buf_token_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_mtp_pos,
                                 true, buf_token_ids_, sizeof(uint32_t));
                cur_mtp_pos += 1;
                mtp_seeded_ = true;
            }
            pre_draft_mtp_pos = cur_mtp_pos;
            // Autoregressive drafts d = 1..K-1, each writing draft[d] directly into buf_token_ids_[d + 1]!
            for (int d = 1; d < K; ++d) {
                encode_mtp_steps(enc, buf_token_ids_, static_cast<size_t>(d) * sizeof(int),
                                 buf_mtp_seed_hid_, 0, 1, cur_mtp_pos,
                                 true, buf_token_ids_, static_cast<size_t>(d + 1) * sizeof(uint32_t));
                cur_mtp_pos += 1;
            }
        }

        // Verify all M = K + 1 tokens in ONE flat-cost mma_qmm8 pass (GDN writes to bak_* leaving states_* untouched!)
        encode_chunk(enc, M, pre_pos, true, true, true, true);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal draft+verify error: ") + [[cb.error localizedDescription] UTF8String]);
        }
    }

    // 3. Compare target argmax predictions against drafts (100% exact greedy equivalence!)
    int draft[MAX_VERIFY];
    for (int d = 0; d < K; ++d) draft[d] = tid_ptr[d + 1];
    const uint32_t* preds = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_mtp_argmax_ contents]);

    int accepted = 0;
    for (int d = 0; d < K; ++d) {
        if (static_cast<int>(preds[d]) == draft[d]) {
            out_tokens.push_back(draft[d]);
            ++accepted;
        } else {
            break;
        }
    }
    int bonus_tok = static_cast<int>(preds[accepted]);
    out_tokens.push_back(bonus_tok);

    // Frequency-aware adaptive widening (32k -> 64k -> 152k -> 248k) on sustained out-of-tier bonus tokens
    add_draft_token(bonus_tok);
    maybe_widen_draft_vocab(bonus_tok);

    // Round 28: Update EMA-driven adaptive MTP draft depth (2..7)
    if (!used_pld) {
        update_mtp_depth_ema(accepted, K);
    }

    // Update target pos & O(1) PLD token history with the (accepted + 1) committed tokens [anchor, draft[0..accepted-1]]
    pos_ = pre_pos + accepted + 1;
    push_history_token(anchor_token);
    for (int j = 0; j < accepted; ++j) push_history_token(draft[j]);

    // 4. Defer Lazy GDN commit + MTP KV sync to next round (streams tokens IMMEDIATELY!)
    pending_sync_ = true;
    pending_replay_m_ = accepted + 1;
    pending_path_noncontig_ = false;
    pending_path_len_ = static_cast<uint32_t>(accepted + 1);
    for (int j = 0; j <= accepted; ++j) pending_path_[j] = static_cast<uint32_t>(j);
    pending_seed_first_ = (used_pld && !was_seeded);
    pending_mtp_pos_ = pre_draft_mtp_pos;
    pending_feed_count_ = accepted + 1;
    pending_feed_off_ = pending_seed_first_ ? 1 : 0;
    if (pending_seed_first_) pending_feed_tokens_[0] = anchor_token;
    for (int j = 0; j < accepted; ++j) pending_feed_tokens_[pending_feed_off_ + j] = draft[j];
    pending_feed_tokens_[pending_feed_off_ + accepted] = bonus_tok;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = accepted + 1;
        stats->mtp_accepted = accepted;
        stats->mtp_drafted = K;
        stats->pld_hits = used_pld ? 1 : 0;
    }
}

void Qwen38Model::forward_mtp_sampled(int anchor_token, Sampler& sampler, std::vector<int>& out_tokens, StepStats* stats) {
    if (dflash_loaded_) {
        if (pos_ + dflash_->config().block_size > max_seq_len_) {
            const float* lg = forward(&anchor_token, 1, nullptr, stats);
            out_tokens.push_back(sampler.sample(lg, vocab_size_));
            return;
        }
        forward_dflash_impl(anchor_token, &sampler, out_tokens, stats);
        return;
    }
    if (!mtp_loaded_ || pos_ + MAX_VERIFY > max_seq_len_) {
        const float* lg = forward(&anchor_token, 1, nullptr, stats);
        out_tokens.push_back(sampler.sample(lg, vocab_size_));
        return;
    }
    const int tree_mode = std::getenv("MTP_TREE") ? std::atoi(std::getenv("MTP_TREE")) : 6;
    if (tree_mode > 0 && (pending_sync_ || !mtp_seeded_)) {
        forward_mtp_tree_impl(anchor_token, &sampler, out_tokens, stats);
        return;
    }
    ensure_kv_capacity(pos_ + MAX_VERIFY);

    auto t0 = std::chrono::steady_clock::now();
    const int pre_pos = pos_;
    int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);

    int pld_draft[MAX_VERIFY];
    int pld_k = try_pld_draft(anchor_token, pld_draft, 7);
    bool used_pld = (pld_k >= 2);
    int K = used_pld ? pld_k : cur_mtp_depth_;
    int M = K + 1;

    tid_ptr[0] = anchor_token;
    if (used_pld) {
        for (int d = 0; d < K; ++d) tid_ptr[d + 1] = pld_draft[d];
    } else if (!pending_sync_ && mtp_seeded_) {
        tid_ptr[1] = mtp_seed_token_;
    }

    if (pending_sync_) {
        int* sync_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_mtp_sync_ids_ contents]);
        std::memcpy(sync_ptr, pending_feed_tokens_,
                    static_cast<size_t>(pending_feed_off_ + pending_feed_count_) * sizeof(int));
    }

    bool was_seeded = mtp_seeded_;
    int pre_draft_mtp_pos = mtp_pos_;
    const auto& scfg = sampler.config();
    const bool do_sampled_draft = (!used_pld && scfg.temperature > 0.0f);
    // Empirically optimal draft sharpening (T_d = 0.6*T with top_k/top_p -> +24.3% faster sampled decode, 100% lossless!)
    const float dt_mul = std::getenv("DRAFT_TMUL") ? std::strtof(std::getenv("DRAFT_TMUL"), nullptr) : 0.85f;
    const float draft_temp = scfg.temperature * dt_mul;
    const float draft_top_p = std::getenv("DRAFT_TOPP")
        ? (std::strtof(std::getenv("DRAFT_TOPP"), nullptr) > 0.0f ? std::strtof(std::getenv("DRAFT_TOPP"), nullptr) : 1.0f)
        : scfg.top_p;
    bool draft_has_q[MAX_VERIFY] = {};
    const float v_margin = sampler.compact_margin();

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        if (pending_sync_) {
            // Note: GDN replay (pending_replay_m_) is lazily fused into encode_chunk() below!
            int cur_pos = pending_mtp_pos_;
            if (pending_seed_first_) {
                encode_mtp_steps(enc, buf_mtp_sync_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_pos, false);
                cur_pos += 1;
            }
            DraftSampleSpec ds0{draft_temp, sampler.next_seed(), buf_draft_dists_, 0, scfg.top_k, draft_top_p};
            const DraftSampleSpec* pds0 = do_sampled_draft ? &ds0 : nullptr;
            if (do_sampled_draft) draft_has_q[0] = true;
            encode_mtp_steps(enc, buf_mtp_sync_ids_, static_cast<size_t>(pending_feed_off_) * sizeof(int),
                             buf_mtp_verify_hid_, 0, pending_feed_count_, cur_pos,
                             !used_pld, buf_token_ids_, sizeof(uint32_t), pds0);
            mtp_pos_ = cur_pos + pending_feed_count_;
            mtp_seeded_ = true;
            was_seeded = true;
            pre_draft_mtp_pos = mtp_pos_;
            pending_sync_ = false;
        }

        if (!used_pld) {
            int cur_mtp_pos = mtp_pos_;
            if (!mtp_seeded_) {
                DraftSampleSpec ds0{draft_temp, sampler.next_seed(), buf_draft_dists_, 0, scfg.top_k, draft_top_p};
                const DraftSampleSpec* pds0 = do_sampled_draft ? &ds0 : nullptr;
                if (do_sampled_draft) draft_has_q[0] = true;
                encode_mtp_steps(enc, buf_token_ids_, 0, buf_mtp_last_hid_, 0, 1, cur_mtp_pos,
                                 true, buf_token_ids_, sizeof(uint32_t), pds0);
                cur_mtp_pos += 1;
                mtp_seeded_ = true;
            }
            pre_draft_mtp_pos = cur_mtp_pos;
            for (int d = 1; d < K; ++d) {
                DraftSampleSpec dsd{
                    draft_temp, sampler.next_seed(), buf_draft_dists_,
                    static_cast<size_t>(d) * sizeof(DraftDist), scfg.top_k, draft_top_p
                };
                const DraftSampleSpec* pdsd = do_sampled_draft ? &dsd : nullptr;
                if (do_sampled_draft) draft_has_q[d] = true;
                encode_mtp_steps(enc, buf_token_ids_, static_cast<size_t>(d) * sizeof(int),
                                 buf_mtp_seed_hid_, 0, 1, cur_mtp_pos,
                                 true, buf_token_ids_, static_cast<size_t>(d + 1) * sizeof(uint32_t), pdsd);
                cur_mtp_pos += 1;
            }
        }

        encode_chunk(enc, M, pre_pos, true, false, true, true, v_margin);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal sampled draft+verify error: ") + [[cb.error localizedDescription] UTF8String]);
        }
    }

    // Round 9 + 10: Exact Leviathan speculative rejection sampling using GPU-compacted target candidates & GPU draft dists!
    int draft[MAX_VERIFY];
    for (int d = 0; d < K; ++d) draft[d] = tid_ptr[d + 1];
    const float* all_logits = reinterpret_cast<const float*>([(id<MTLBuffer>)buf_mtp_logits_ contents]);
    const LogitCand* all_cands = reinterpret_cast<const LogitCand*>([(id<MTLBuffer>)buf_verify_cands_ contents]);
    const uint32_t* cand_cnts = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_verify_cnt_ contents]);
    const DraftDist* draft_dists = reinterpret_cast<const DraftDist*>([(id<MTLBuffer>)buf_draft_dists_ contents]);

    int accepted = 0;
    int bonus_tok = 0;
    bool rejected = false;
    for (int d = 0; d < K; ++d) {
        const float* row_lg = all_logits + static_cast<size_t>(d) * vocab_size_;
        const LogitCand* row_cands = all_cands + static_cast<size_t>(d) * MAX_VERIFY_CANDS;
        const DraftDist* dd = draft_has_q[d] ? &draft_dists[d] : nullptr;
        bool ok = false;
        int tok = sampler.sample_or_accept_draft_dist(
            row_cands, cand_cnts[d], MAX_VERIFY_CANDS, row_lg, vocab_size_, draft[d], dd, ok
        );
        if (ok) {
            out_tokens.push_back(draft[d]);
            sampler.observe(draft[d]);
            ++accepted;
        } else {
            bonus_tok = tok;
            out_tokens.push_back(bonus_tok);
            rejected = true;
            break;
        }
    }
    if (!rejected) {
        const float* last_lg = all_logits + static_cast<size_t>(K) * vocab_size_;
        const LogitCand* last_cands = all_cands + static_cast<size_t>(K) * MAX_VERIFY_CANDS;
        bonus_tok = sampler.sample_from_cands(last_cands, cand_cnts[K], MAX_VERIFY_CANDS, last_lg, vocab_size_);
        out_tokens.push_back(bonus_tok);
    }

    add_draft_token(bonus_tok);
    maybe_widen_draft_vocab(bonus_tok);

    if (!used_pld) {
        update_mtp_depth_ema(accepted, K);
    }

    pos_ = pre_pos + accepted + 1;
    push_history_token(anchor_token);
    for (int j = 0; j < accepted; ++j) push_history_token(draft[j]);

    pending_sync_ = true;
    pending_replay_m_ = accepted + 1;
    pending_path_noncontig_ = false;
    pending_path_len_ = static_cast<uint32_t>(accepted + 1);
    for (int j = 0; j <= accepted; ++j) pending_path_[j] = static_cast<uint32_t>(j);
    pending_seed_first_ = (used_pld && !was_seeded);
    pending_mtp_pos_ = pre_draft_mtp_pos;
    pending_feed_count_ = accepted + 1;
    pending_feed_off_ = pending_seed_first_ ? 1 : 0;
    if (pending_seed_first_) pending_feed_tokens_[0] = anchor_token;
    for (int j = 0; j < accepted; ++j) pending_feed_tokens_[pending_feed_off_ + j] = draft[j];
    pending_feed_tokens_[pending_feed_off_ + accepted] = bonus_tok;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = accepted + 1;
        stats->mtp_accepted = accepted;
        stats->mtp_drafted = K;
        stats->pld_hits = used_pld ? 1 : 0;
    }
}

bool Qwen38Model::save_state_to_file(const std::string& path, const std::vector<int>& tokens,
                                     const std::vector<uint64_t>& image_hashes) {
    if (pending_sync_) flush_pending_mtp();
    if (pos_ <= 0 || static_cast<int>(tokens.size()) != pos_) return false;
    if (save_thread_.joinable()) save_thread_.join();

    size_t cs_bytes = 3ull * lin_conv_dim_ * sizeof(float);
    size_t ss_bytes = static_cast<size_t>(lin_num_v_heads_) * lin_head_v_dim_ * lin_head_k_dim_ * sizeof(uint16_t);
    size_t elem_div = (kv_bits_ == 16) ? 1ull : ((kv_bits_ == 8) ? 2ull : 4ull);
    size_t kv_active = (static_cast<size_t>(pos_) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / elem_div;
    size_t sb_active = (kv_bits_ < 16)
        ? (static_cast<size_t>(pos_) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    size_t mtp_elem_div = (mtp_kv_bits_ == 16) ? 1ull : ((mtp_kv_bits_ == 8) ? 2ull : 4ull);
    size_t mtp_kv_active = mtp_loaded_
        ? ((static_cast<size_t>(mtp_pos_) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / mtp_elem_div)
        : 0;
    size_t mtp_sb_active = (mtp_loaded_ && mtp_kv_bits_ < 16)
        ? (static_cast<size_t>(mtp_pos_) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    size_t mtp_hid_bytes = mtp_loaded_ ? (static_cast<size_t>(hidden_size_) * sizeof(float)) : 0;
    size_t df_ring_sz = dflash_loaded_
        ? (static_cast<size_t>(dflash_->config().num_kv_heads) * dflash_->config().sliding_window * dflash_->config().head_dim * sizeof(uint16_t))
        : 0;

    size_t total_state = 0;
    for (int i = 0; i < num_layers_; ++i) {
        total_state += layers_[i].is_linear ? (cs_bytes + ss_bytes) : (2 * kv_active + 2 * sb_active);
    }
    if (mtp_loaded_) total_state += 2 * mtp_kv_active + 2 * mtp_sb_active + mtp_hid_bytes;
    if (dflash_loaded_) total_state += static_cast<size_t>(dflash_->config().num_layers) * 2 * df_ring_sz;

    @autoreleasepool {
        id<MTLBuffer> stage = ctx_.alloc_shared(total_state);
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t off = 0;
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) {
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].conv_state sourceOffset:0 toBuffer:stage destinationOffset:off size:cs_bytes];
                off += cs_bytes;
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].ssm_state sourceOffset:0 toBuffer:stage destinationOffset:off size:ss_bytes];
                off += ss_bytes;
            } else {
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].k_cache sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_active];
                off += kv_active;
                [blit copyFromBuffer:(id<MTLBuffer>)states_[i].v_cache sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_active];
                off += kv_active;
                if (sb_active > 0) {
                    [blit copyFromBuffer:(id<MTLBuffer>)states_[i].k_sb sourceOffset:0 toBuffer:stage destinationOffset:off size:sb_active];
                    off += sb_active;
                    [blit copyFromBuffer:(id<MTLBuffer>)states_[i].v_sb sourceOffset:0 toBuffer:stage destinationOffset:off size:sb_active];
                    off += sb_active;
                }
            }
        }
        if (mtp_loaded_) {
            if (mtp_kv_active > 0) {
                [blit copyFromBuffer:(id<MTLBuffer>)mtp_k_cache_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mtp_kv_active];
                off += mtp_kv_active;
                [blit copyFromBuffer:(id<MTLBuffer>)mtp_v_cache_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mtp_kv_active];
                off += mtp_kv_active;
                if (mtp_sb_active > 0) {
                    [blit copyFromBuffer:(id<MTLBuffer>)mtp_k_sb_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mtp_sb_active];
                    off += mtp_sb_active;
                    [blit copyFromBuffer:(id<MTLBuffer>)mtp_v_sb_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mtp_sb_active];
                    off += mtp_sb_active;
                }
            }
            [blit copyFromBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mtp_hid_bytes];
            off += mtp_hid_bytes;
        }
        if (dflash_loaded_) {
            for (int l = 0; l < dflash_->config().num_layers; ++l) {
                [blit copyFromBuffer:(id<MTLBuffer>)df_k_cache_[l] sourceOffset:0 toBuffer:stage destinationOffset:off size:df_ring_sz];
                off += df_ring_sz;
                [blit copyFromBuffer:(id<MTLBuffer>)df_v_cache_[l] sourceOffset:0 toBuffer:stage destinationOffset:off size:df_ring_sz];
                off += df_ring_sz;
            }
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // Round 11 + 32: Async background disk serialization (MICACHE2 when multimodal, MICACHE1 for pure text!)
        const bool is_v2 = (!image_hashes.empty() || rope_delta_ != 0);
        PrefixCacheHeader hdr{};
        std::memcpy(hdr.magic, is_v2 ? "MICACHE2" : "MICACHE1", 8);
        hdr.model_hash = compute_model_hash(info_);
        hdr.pos = static_cast<uint32_t>(pos_);
        hdr.reserved = mtp_loaded_
            ? (0x80000000u | (mtp_kv_bits_ < 16 ? 0x40000000u : 0u) | (mtp_kv_bits_ == 4 ? 0x20000000u : 0u) | (static_cast<uint32_t>(mtp_pos_) & 0x1FFFFFFFu))
            : (dflash_loaded_ ? (0x40000000u | (static_cast<uint32_t>(df_kv_len_) & 0x1FFFFFFFu)) : 0u);
        hdr.state_bytes = total_state;
        MultimodalCacheMeta meta{static_cast<int32_t>(rope_delta_), static_cast<uint32_t>(image_hashes.size())};

        void* retained_stage = (void*)CFBridgingRetain(stage);
        const void* stage_ptr = [stage contents];
        std::vector<int> toks_copy = tokens;
        std::vector<uint64_t> imgs_copy = image_hashes;
        std::string path_copy = path;

        save_thread_ = std::thread([retained_stage, stage_ptr, hdr, is_v2, meta,
                                    toks_copy = std::move(toks_copy), imgs_copy = std::move(imgs_copy),
                                    path_copy = std::move(path_copy), total_state]() {
            std::string tmp_path = path_copy + ".tmp";
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
                ofs.write(reinterpret_cast<const char*>(toks_copy.data()), toks_copy.size() * sizeof(int));
                if (is_v2) {
                    ofs.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
                    if (!imgs_copy.empty()) {
                        ofs.write(reinterpret_cast<const char*>(imgs_copy.data()), imgs_copy.size() * sizeof(uint64_t));
                    }
                }
                ofs.write(reinterpret_cast<const char*>(stage_ptr), total_state);
                ofs.close();
                if (ofs.good()) {
                    std::error_code ec;
                    std::filesystem::rename(tmp_path, path_copy, ec);
                } else {
                    std::error_code ec;
                    std::filesystem::remove(tmp_path, ec);
                }
            }
            CFRelease(retained_stage);
        });
        return true;
    }
}

void Qwen38Model::keep_warm() {
    // Round 11: Non-mutating lightweight GPU + ANE + mmap keep-warm ping (<3 ms)
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        uint32_t n = 256u;
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    if (ane_ && ane_->ready()) {
        ane_->submit(0, 0, 1);
        ane_->wait();
    }
    store_.prefetch();
}

bool Qwen38Model::load_state_from_file(const std::string& path, std::vector<int>& out_tokens,
                                       std::vector<uint64_t>& out_image_hashes) {
    out_image_hashes.clear();
    if (save_thread_.joinable()) save_thread_.join();
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < sizeof(PrefixCacheHeader)) {
        ::close(fd);
        return false;
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    const size_t page_sz = static_cast<size_t>(::getpagesize());
    const size_t map_len = ((file_size + page_sz - 1) / page_sz) * page_sz;
    void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) return false;

    auto unmap_guard = [&]() { ::munmap(mapped, file_size); };

    const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped);
    PrefixCacheHeader hdr{};
    std::memcpy(&hdr, base, sizeof(hdr));
    bool is_v1 = (std::memcmp(hdr.magic, "MICACHE1", 8) == 0);
    bool is_v2 = (std::memcmp(hdr.magic, "MICACHE2", 8) == 0);
    if ((!is_v1 && !is_v2) ||
        hdr.model_hash != compute_model_hash(info_) ||
        static_cast<int>(hdr.pos) > max_seq_len_) {
        unmap_guard();
        return false;
    }

    const size_t tok_bytes = static_cast<size_t>(hdr.pos) * sizeof(int);
    size_t cursor = sizeof(PrefixCacheHeader) + tok_bytes;
    if (cursor > file_size) {
        unmap_guard();
        return false;
    }

    int loaded_rope_delta = 0;
    if (is_v2) {
        if (cursor + sizeof(MultimodalCacheMeta) > file_size) {
            unmap_guard();
            return false;
        }
        MultimodalCacheMeta meta{};
        std::memcpy(&meta, base + cursor, sizeof(meta));
        cursor += sizeof(meta);
        size_t img_bytes = static_cast<size_t>(meta.num_images) * sizeof(uint64_t);
        if (cursor + img_bytes > file_size) {
            unmap_guard();
            return false;
        }
        out_image_hashes.resize(meta.num_images);
        if (img_bytes > 0) {
            std::memcpy(out_image_hashes.data(), base + cursor, img_bytes);
        }
        cursor += img_bytes;
        loaded_rope_delta = meta.rope_delta;
    }

    const size_t data_off = cursor;
    if (data_off + hdr.state_bytes > file_size) {
        unmap_guard();
        return false;
    }

    out_tokens.resize(hdr.pos);
    std::memcpy(out_tokens.data(), base + sizeof(PrefixCacheHeader), tok_bytes);

    // Round 33: Ensure lazy KV buffers can hold restored hdr.pos!
    ensure_kv_capacity(static_cast<int>(hdr.pos));

    size_t cs_bytes = 3ull * lin_conv_dim_ * sizeof(float);
    size_t ss_bytes = static_cast<size_t>(lin_num_v_heads_) * lin_head_v_dim_ * lin_head_k_dim_ * sizeof(uint16_t);
    size_t elem_div = (kv_bits_ == 16) ? 1ull : ((kv_bits_ == 8) ? 2ull : 4ull);
    size_t kv_active = (static_cast<size_t>(hdr.pos) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / elem_div;
    size_t sb_active = (kv_bits_ < 16)
        ? (static_cast<size_t>(hdr.pos) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    bool has_mtp_payload = (hdr.reserved & 0x80000000u) != 0u;
    bool has_df_payload  = (!has_mtp_payload) && ((hdr.reserved & 0x40000000u) != 0u);
    bool mtp_is_q        = (hdr.reserved & 0x40000000u) != 0u;
    bool mtp_is_q4       = (hdr.reserved & 0x20000000u) != 0u;
    uint32_t saved_mtp_pos = hdr.reserved & 0x1FFFFFFFu;
    int saved_mtp_bits   = mtp_is_q ? (mtp_is_q4 ? 4 : 8) : 16;
    if (has_mtp_payload && mtp_loaded_ && (saved_mtp_bits != mtp_kv_bits_)) {
        unmap_guard();
        return false;
    }
    if (dflash_loaded_ && !has_df_payload) {
        unmap_guard();
        return false;
    }
    size_t mtp_elem_div = (saved_mtp_bits == 16) ? 1ull : ((saved_mtp_bits == 8) ? 2ull : 4ull);
    size_t mtp_kv_active = has_mtp_payload
        ? ((static_cast<size_t>(saved_mtp_pos) * num_kv_heads_ * head_dim_ * sizeof(uint16_t)) / mtp_elem_div)
        : 0;
    size_t mtp_sb_active = (has_mtp_payload && mtp_is_q)
        ? (static_cast<size_t>(saved_mtp_pos) * num_kv_heads_ * (head_dim_ / 64) * 2 * sizeof(uint16_t))
        : 0;
    size_t mtp_hid_bytes = has_mtp_payload ? (static_cast<size_t>(hidden_size_) * sizeof(float)) : 0;
    size_t df_ring_sz = (has_df_payload && dflash_loaded_)
        ? (static_cast<size_t>(dflash_->config().num_kv_heads) * dflash_->config().sliding_window * dflash_->config().head_dim * sizeof(uint16_t))
        : 0;
    size_t expected_state = 0;
    for (int i = 0; i < num_layers_; ++i) {
        expected_state += layers_[i].is_linear ? (cs_bytes + ss_bytes) : (2 * kv_active + 2 * sb_active);
    }
    if (has_mtp_payload) expected_state += 2 * mtp_kv_active + 2 * mtp_sb_active + mtp_hid_bytes;
    if (has_df_payload && dflash_loaded_) expected_state += static_cast<size_t>(dflash_->config().num_layers) * 2 * df_ring_sz;
    if (expected_state != hdr.state_bytes) {
        unmap_guard();
        return false;
    }

    @autoreleasepool {
        // Zero-copy mmap -> MTLBuffer (eliminates 750 MB CPU memcpy on prefix cache restore!)
        id<MTLBuffer> stage = [(id<MTLDevice>)ctx_.device() newBufferWithBytesNoCopy:mapped
                                                                              length:map_len
                                                                             options:MTLResourceStorageModeShared
                                                                         deallocator:nil];
        size_t base_off = data_off;
        if (!stage) {
            stage = ctx_.alloc_shared(hdr.state_bytes);
            std::memcpy([stage contents], base + data_off, hdr.state_bytes);
            base_off = 0;
        }

        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t off = base_off;
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) {
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].conv_state destinationOffset:0 size:cs_bytes];
                off += cs_bytes;
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].ssm_state destinationOffset:0 size:ss_bytes];
                off += ss_bytes;
            } else {
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].k_cache destinationOffset:0 size:kv_active];
                off += kv_active;
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].v_cache destinationOffset:0 size:kv_active];
                off += kv_active;
                if (sb_active > 0) {
                    [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].k_sb destinationOffset:0 size:sb_active];
                    off += sb_active;
                    [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)states_[i].v_sb destinationOffset:0 size:sb_active];
                    off += sb_active;
                }
            }
        }
        if (has_mtp_payload && mtp_loaded_) {
            if (mtp_kv_active > 0) {
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)mtp_k_cache_ destinationOffset:0 size:mtp_kv_active];
                off += mtp_kv_active;
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)mtp_v_cache_ destinationOffset:0 size:mtp_kv_active];
                off += mtp_kv_active;
                if (mtp_sb_active > 0) {
                    [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)mtp_k_sb_ destinationOffset:0 size:mtp_sb_active];
                    off += mtp_sb_active;
                    [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)mtp_v_sb_ destinationOffset:0 size:mtp_sb_active];
                    off += mtp_sb_active;
                }
            }
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)buf_mtp_last_hid_ destinationOffset:0 size:mtp_hid_bytes];
            off += mtp_hid_bytes;
        }
        if (has_df_payload && dflash_loaded_) {
            for (int l = 0; l < dflash_->config().num_layers; ++l) {
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)df_k_cache_[l] destinationOffset:0 size:df_ring_sz];
                off += df_ring_sz;
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)df_v_cache_[l] destinationOffset:0 size:df_ring_sz];
                off += df_ring_sz;
            }
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    unmap_guard();

    pos_ = static_cast<int>(hdr.pos);
    rope_delta_ = loaded_rope_delta;
    for (int i = 0; i < num_layers_; ++i) states_[i].quest_built_pages = 0;
    mtp_quest_built_pages_ = 0;
    cur_mrope_buf_ = nil;
    cur_vis_count_ = 0;
    mtp_pos_ = (has_mtp_payload && mtp_loaded_) ? static_cast<int>(saved_mtp_pos) : 0;
    mtp_seeded_ = false;
    df_kv_len_ = (has_df_payload && dflash_loaded_) ? static_cast<int>(saved_mtp_pos) : 0;
    df_pending_commit_ = 0;
    df_pending_pos_ = 0;
    pending_sync_ = false;
    token_history_.clear();
    suffix_cache_.begin_turn();
    reset_draft_row_map();
    if (++pld_gen_ == 0u) {
        std::fill(pld4_.begin(), pld4_.end(), PldSlot{});
        std::fill(pld3_.begin(), pld3_.end(), PldSlot{});
        pld_gen_ = 1u;
    }
    token_history_.reserve(out_tokens.size() + 256);
    for (int tk : out_tokens) push_history_token(tk);
    draft_miss_mask_ = 0u;
    // Apply same frequency-aware draft vocab sizing to restored tokens
    draft_vocab_size_ = std::min<uint32_t>(FAST_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
    if (out_tokens.size() >= 16) {
        int n_reg = 0, n_gt32 = 0, n_gt64 = 0, n_gt152 = 0;
        for (int tk : out_tokens) {
            if (tk < 0 || static_cast<uint32_t>(tk) >= special_tok_lo_) continue;
            ++n_reg;
            if (tk >= static_cast<int>(FAST_DRAFT_VOCAB)) ++n_gt32;
            if (tk >= static_cast<int>(MED_DRAFT_VOCAB))  ++n_gt64;
            if (tk >= static_cast<int>(HI_DRAFT_VOCAB))   ++n_gt152;
        }
        if (n_reg >= 16) {
            if (n_gt152 * 10 > n_reg) draft_vocab_size_ = static_cast<uint32_t>(vocab_size_);
            else if (n_gt64 * 10 > n_reg) draft_vocab_size_ = std::min<uint32_t>(HI_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
            else if (n_gt32 * 10 > n_reg) draft_vocab_size_ = std::min<uint32_t>(MED_DRAFT_VOCAB, static_cast<uint32_t>(vocab_size_));
        }
    }
    return true;
}

void Qwen38Model::profile_ops() {
    ensure_kv_capacity(1024);
    reset();
    // Run 1 short warmup forward to fault pages & stabilize clocks
    std::vector<int> warm(64, 100);
    forward(warm.data(), 64);
    pos_ = 512;

    int* tid_ptr = reinterpret_cast<int*>([(id<MTLBuffer>)buf_token_ids_ contents]);
    for (int i = 0; i < 16; ++i) tid_ptr[i] = 100 + i;

    auto bench_gpu = [&](const char* label, double gb_weights, auto&& encode_fn) -> double {
        double sum_ms = 0.0;
        const int WARM = 2;
        const int REPS = 5;
        for (int r = 0; r < WARM + REPS; ++r) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                encode_fn(enc);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) {
                    throw std::runtime_error(std::string("profile_ops GPU error: ") +
                                             [[cb.error localizedDescription] UTF8String]);
                }
                if (r >= WARM) {
                    double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    sum_ms += ms;
                }
            }
        }
        double avg_ms = sum_ms / REPS;
        if (gb_weights > 0.0) {
            std::printf("  %-46s %6.2f ms  (%5.1f GB/s on %.2f GB)\n",
                        label, avg_ms, gb_weights / (avg_ms / 1000.0), gb_weights);
        } else {
            std::printf("  %-46s %6.2f ms\n", label, avg_ms);
        }
        std::fflush(stdout);
        return avg_ms;
    };

    const bool has_share = (ane_share_ && ane_share_->shared_count() == num_layers_);
    const double bpw_mlp = has_share ? 4.0 : 4.5;
    const double bpw_g64 = 4.5;

    std::printf("\n================ PROBE 1: HARDWARE GPU/ANE TIMING BREAKDOWN ================\n");
    std::printf("Config: %s | ane_share=%d/%d | kv_bits=%d\n\n",
                info_.quant_desc.c_str(), has_share ? num_layers_ : 0, num_layers_, kv_bits_);

    // 1. Whole-pass baselines (M=8 verify vs M=1 greedy at pos=512)
    double t_full_m8 = bench_gpu("FULL M=8 verify (64L + lm_head + argmax)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        if (use_cpu_embed_) cpu_embed_.gather_f32(tid_ptr, 8, reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_ contents]));
        encode_chunk(enc, 8, 512, false, true, true, true, 0.0f);
    });
    double t_full_m1 = bench_gpu("FULL M=1 greedy (64L + lm_head + argmax)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        if (use_cpu_embed_) cpu_embed_.gather_f32(tid_ptr, 1, reinterpret_cast<float*>([(id<MTLBuffer>)buf_hidden_ contents]));
        encode_chunk(enc, 1, 512, true, true, false, false, 0.0f);
    });
    std::printf("\n--- Isolated Op Families across all 64 real layers (M=8, pos=512) ---\n");

    const uint32_t M = 8u;
    double sum_ops = 0.0;

    // 2. 64x MLP gate+up
    double gb_gateup = 64.0 * 2.0 * hidden_size_ * intermediate_size_ * (bpw_mlp / 8.0) / 1e9;
    sum_ops += bench_gpu(has_share
            ? (ane_share_nonuniform_lut_ ? "64x MLP gate+up (ane_lut4_swiglu_bf16, M=8)" : "64x MLP gate+up (ane_hf4_swiglu_bf16, M=8)")
            : "64x MLP gate+up (g64 mma_swiglu_f16, M=8)",
        gb_gateup, [&](id<MTLComputeCommandEncoder> enc) {
            for (int i = 0; i < num_layers_; ++i) {
                const LayerWeights& lw = layers_[i];
                if (lw.ane_share_idx >= 0 && ane_share_ && ane_share_->has_layer(lw.ane_share_idx)) {
                    const AneShareLayer& sl = ane_share_->layer(lw.ane_share_idx);
                    AneShareParams ap = make_ane_share_params(sl, M);
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(sl.intermediate_size / 16u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    QmmParams qp{M, lw.gate_proj.out_dim, lw.gate_proj.in_dim, 0u};
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_swiglu_f16_];
                    [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.w.buffer offset:lw.gate_proj.w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.s.buffer offset:lw.gate_proj.s.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)lw.gate_proj.b.buffer offset:lw.gate_proj.b.offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)lw.up_proj.w.buffer offset:lw.up_proj.w.offset atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)lw.up_proj.s.buffer offset:lw.up_proj.s.offset atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)lw.up_proj.b.buffer offset:lw.up_proj.b.offset atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:7];
                    [enc setBytes:&qp length:sizeof(qp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(lw.gate_proj.out_dim, 16u), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        });

    // 3. 64x MLP down_proj
    double gb_down = 64.0 * hidden_size_ * intermediate_size_ * (bpw_mlp / 8.0) / 1e9;
    sum_ops += bench_gpu(has_share
            ? (ane_share_nonuniform_lut_ ? "64x MLP down_proj (ane_lut4_down_sk4, M=8)" : "64x MLP down_proj (ane_hf4_down_sk4, M=8)")
            : "64x MLP down_proj (g64 mma_qmm8_sk, M=8)",
        gb_down, [&](id<MTLComputeCommandEncoder> enc) {
            for (int i = 0; i < num_layers_; ++i) {
                const LayerWeights& lw = layers_[i];
                if (lw.ane_share_idx >= 0 && ane_share_ && ane_share_->has_layer(lw.ane_share_idx)) {
                    const AneShareLayer& sl = ane_share_->layer(lw.ane_share_idx);
                    AneShareParams ap = make_ane_share_params(sl, M);
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_sk4_bf16_];
                    [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                    [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(16u * ceil_div(sl.hidden_size / 128u, 4u), 1, 2u * sl.n_subblocks)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else {
                    QmmSkParams sp{M, lw.down_proj.out_dim, lw.down_proj.in_dim, 1u, 8u};
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_qmm8_f16x_sk_];
                    [enc setBuffer:(id<MTLBuffer>)lw.down_proj.w.buffer offset:lw.down_proj.w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.down_proj.s.buffer offset:lw.down_proj.s.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)lw.down_proj.b.buffer offset:lw.down_proj.b.offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:4];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(lw.down_proj.out_dim, 32u), 1, 8)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        });

    // 4. 64x MLP sk_reduce_norm_f16
    sum_ops += bench_gpu("64x MLP sk_reduce_norm_f16 (SK=8, M=8)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        QmmSkNormParams sp{M, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), 1u, 8u, rms_eps_};
        for (int i = 0; i < num_layers_; ++i) {
            const TensorView& nn = (i + 1 < num_layers_) ? layers_[i + 1].input_norm : final_norm_;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_sk_reduce_norm_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_hidden_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)nn.buffer offset:nn.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    });

    // 5. 48x GDN in_proj (qkv + z + ab)
    double gb_gdn_in = 0.0;
    for (int i = 0; i < num_layers_; ++i) {
        if (!layers_[i].is_linear) continue;
        const auto& lw = layers_[i];
        gb_gdn_in += (double(lw.in_proj_qkv.out_dim) * lw.in_proj_qkv.in_dim
                    + double(lw.in_proj_z.out_dim) * lw.in_proj_z.in_dim
                    + double(lw.in_proj_a.out_dim) * lw.in_proj_a.in_dim
                    + double(lw.in_proj_b.out_dim) * lw.in_proj_b.in_dim) * (bpw_g64 / 8.0) / 1e9;
    }
    sum_ops += bench_gpu("48x GDN in_proj (qkv + z + ab, M=8)", gb_gdn_in, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            encode_qmm_q4_f16x(enc, lw.in_proj_qkv, buf_normed_, 0, rec_qkv_[0][i], 0, M, false);
            encode_qmm_q4_f16x(enc, lw.in_proj_z,   buf_normed_, 0, buf_z_,         0, M, false);
            QmmParams qp{M, lw.in_proj_a.out_dim, lw.in_proj_a.in_dim, 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_in_proj_ab_f16x_];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.w.buffer offset:lw.in_proj_a.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.s.buffer offset:lw.in_proj_a.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_a.b.buffer offset:lw.in_proj_a.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.w.buffer offset:lw.in_proj_b.w.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.s.buffer offset:lw.in_proj_b.s.offset atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)lw.in_proj_b.b.buffer offset:lw.in_proj_b.b.offset atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)rec_a_[0][i] offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)rec_b_[0][i] offset:0 atIndex:8];
            [enc setBytes:&qp length:sizeof(qp) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(2u * ceil_div(lw.in_proj_a.out_dim, 8u), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    });

    // 6. 48x GDN prework + rec + norm (Lazy Fused Replay R=4 + Verify M=8!)
    sum_ops += bench_gpu("48x GDN lazy(R=4)+rec(M=8)+norm", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        GdnPrepParams pp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_head_k_dim_),
                         static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_conv_dim_), rms_eps_};
        GdnRecParams rp{M, static_cast<uint32_t>(lin_num_k_heads_), static_cast<uint32_t>(lin_num_v_heads_),
                        static_cast<uint32_t>(lin_head_k_dim_), static_cast<uint32_t>(lin_head_v_dim_),
                        static_cast<uint32_t>(lin_conv_dim_)};
        GdnNormParams np{M, static_cast<uint32_t>(lin_num_v_heads_), static_cast<uint32_t>(lin_head_v_dim_), rms_eps_};
        PathReplayParams pr{};
        pr.len = 4u;
        for (uint32_t j = 0; j < 4u; ++j) pr.path[j] = j;
        TreeParams tp{};
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            const auto& st = states_[i];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_prework_lazy_];
            [enc setBuffer:(id<MTLBuffer>)rec_qkv_[0][i] offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)rec_qkv_[1][i] offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.conv1d_w.buffer offset:lw.conv1d_w.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)st.conv_state offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:6];
            [enc setBuffer:(id<MTLBuffer>)rec_a_[0][i] offset:0 atIndex:7];
            [enc setBuffer:(id<MTLBuffer>)rec_b_[0][i] offset:0 atIndex:8];
            [enc setBuffer:(id<MTLBuffer>)rec_a_[1][i] offset:0 atIndex:9];
            [enc setBuffer:(id<MTLBuffer>)rec_b_[1][i] offset:0 atIndex:10];
            [enc setBuffer:(id<MTLBuffer>)lw.A_log.buffer offset:lw.A_log.offset atIndex:11];
            [enc setBuffer:(id<MTLBuffer>)lw.dt_bias.buffer offset:lw.dt_bias.offset atIndex:12];
            [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:13];
            [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:14];
            [enc setBytes:&pp length:sizeof(pp) atIndex:15];
            [enc setBytes:&pr length:sizeof(pr) atIndex:16];
            [enc setBytes:&tp length:sizeof(tp) atIndex:17];
            [enc dispatchThreadgroups:MTLSizeMake(2 * lin_num_k_heads_ + lin_num_v_heads_, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_rec_lazy_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_conv_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_g_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_beta_ offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)st.ssm_state offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
            [enc setBytes:&rp length:sizeof(rp) atIndex:7];
            [enc setBytes:&pr length:sizeof(pr) atIndex:8];
            [enc dispatchThreads:MTLSizeMake(32, lin_head_v_dim_ / 4, lin_num_v_heads_)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_gdn_norm_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.gdn_norm.buffer offset:lw.gdn_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_gdn_normed_ offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(lin_num_v_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    });

    // 7. 48x GDN out_proj + sk_reduce_norm
    double gb_gdn_out = 0.0;
    for (int i = 0; i < num_layers_; ++i) {
        if (!layers_[i].is_linear) continue;
        gb_gdn_out += double(layers_[i].out_proj.out_dim) * layers_[i].out_proj.in_dim * (bpw_g64 / 8.0) / 1e9;
    }
    sum_ops += bench_gpu("48x GDN out_proj_sk + reduce_norm (M=8)", gb_gdn_out, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            const TensorView& pan = (lw.ane_share_idx >= 0) ? lw.post_attn_norm_share : lw.post_attn_norm;
            encode_qmm_q4_f16x_and_norm(enc, lw.out_proj, buf_gdn_normed_, 0, buf_hidden_, 0, M, pan, buf_normed_);
        }
    });

    // 8. 16x GQA qkv_proj
    double gb_gqa_qkv = 0.0;
    for (int i = 0; i < num_layers_; ++i) {
        if (layers_[i].is_linear) continue;
        const auto& lw = layers_[i];
        gb_gqa_qkv += (double(lw.q_proj.out_dim) * lw.q_proj.in_dim
                     + double(lw.k_proj.out_dim) * lw.k_proj.in_dim
                     + double(lw.v_proj.out_dim) * lw.v_proj.in_dim) * (bpw_g64 / 8.0) / 1e9;
    }
    sum_ops += bench_gpu("16x GQA qkv_proj (M=8)", gb_gqa_qkv, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            encode_qmm_q4_f16x(enc, lw.q_proj, buf_normed_, 0, buf_qkv_, 0, M, false);
            encode_qmm_q4_f16x(enc, lw.k_proj, buf_normed_, 0, buf_k_,   0, M, false);
            encode_qmm_q4_f16x(enc, lw.v_proj, buf_normed_, 0, buf_v_,   0, M, false);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    });

    // 9. 16x GQA prep + attn + vsplit2 (pos=512)
    sum_ops += bench_gpu("16x GQA prep + attn + vsplit2 (M=8, L=512)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t kv_len = 520u;
        uint32_t splits = 4u;
        FullAttnParams fp{
            M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(rotary_dim_),
            512u, kv_len, rope_theta_, rms_eps_, 1.0f / std::sqrt(static_cast<float>(head_dim_)), splits, 0
        };
        uint32_t gqa = static_cast<uint32_t>(num_heads_ / num_kv_heads_);
        uint32_t num_pairs = static_cast<uint32_t>(num_kv_heads_) * ((gqa + 1u) / 2u);
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            const auto& st = states_[i];
            const bool qkv = (st.k_sb != nil);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_q_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.q_norm.buffer offset:lw.q_norm.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBytes:&fp length:sizeof(fp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:(id<MTLComputePipelineState>)(qkv ? pso_fa_kv_prep_q_ : pso_fa_kv_prep_)];
            [enc setBuffer:(id<MTLBuffer>)buf_k_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.k_norm.buffer offset:lw.k_norm.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:4];
            if (qkv) {
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:6];
                [enc setBytes:&fp length:sizeof(fp) atIndex:7];
            } else {
                [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)(qkv ? pso_fa_vsplit1_q_ : pso_fa_vsplit1_)];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)st.k_cache offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)st.v_cache offset:0 atIndex:2];
            if (qkv) {
                [enc setBuffer:(id<MTLBuffer>)st.k_sb offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)st.v_sb offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:7];
                [enc setBytes:&fp length:sizeof(fp) atIndex:8];
            } else {
                [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:5];
                [enc setBytes:&fp length:sizeof(fp) atIndex:6];
            }
            [enc dispatchThreadgroups:MTLSizeMake(num_pairs, splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_fa_vsplit2_f16_];
            [enc setBuffer:(id<MTLBuffer>)buf_split_o_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_split_m_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_split_l_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_z_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    });

    // 10. 16x GQA o_proj + sk_reduce_norm
    double gb_gqa_o = 0.0;
    for (int i = 0; i < num_layers_; ++i) {
        if (layers_[i].is_linear) continue;
        gb_gqa_o += double(layers_[i].o_proj.out_dim) * layers_[i].o_proj.in_dim * (bpw_g64 / 8.0) / 1e9;
    }
    sum_ops += bench_gpu("16x GQA o_proj_sk + reduce_norm (M=8)", gb_gqa_o, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) continue;
            const auto& lw = layers_[i];
            const TensorView& pan = (lw.ane_share_idx >= 0) ? lw.post_attn_norm_share : lw.post_attn_norm;
            encode_qmm_q4_f16x_and_norm(enc, lw.o_proj, buf_attn_out_, 0, buf_hidden_, 0, M, pan, buf_normed_);
        }
    });

    // 11. lm_head + argmax2 (M=8, greedy write-skip)
    double gb_lmh = double(lm_head_.out_dim) * lm_head_.in_dim * (bpw_g64 / 8.0) / 1e9;
    sum_ops += bench_gpu("1x lm_head_max_f16x + argmax2 (M=8)", gb_lmh, [&](id<MTLComputeCommandEncoder> enc) {
        QmmParams qp{M, lm_head_.out_dim, lm_head_.in_dim, 1u};
        uint32_t n_tg_am = ceil_div(lm_head_.out_dim, 32u);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mma_lmhead_max_f16x_];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.w.buffer offset:lm_head_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.s.buffer offset:lm_head_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)lm_head_.b.buffer offset:lm_head_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_logits_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:5];
        [enc setBytes:&qp length:sizeof(qp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg_am, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_argmax_ offset:0 atIndex:1];
        [enc setBytes:&n_tg_am length:sizeof(n_tg_am) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });

    std::printf("\n  SUM of isolated op families:                   %6.2f ms  (vs FULL M=8 %.2f ms, M=1 %.2f ms)\n",
                sum_ops, t_full_m8, t_full_m1);

    if (has_share) {
        bench_gpu(ane_share_nonuniform_lut_ ? "64x MLP M=4 (ane_lut4_swiglu_bf16 + down_sk4)" : "64x MLP M=4 (ane_hf4_swiglu_bf16 + down_sk4)",
            gb_gateup + gb_down, [&](id<MTLComputeCommandEncoder> enc) {
                for (int i = 0; i < num_layers_; ++i) {
                    const LayerWeights& lw = layers_[i];
                    if (lw.ane_share_idx >= 0 && ane_share_->has_layer(lw.ane_share_idx)) {
                        const AneShareLayer& sl = ane_share_->layer(lw.ane_share_idx);
                        AneShareParams ap = make_ane_share_params(sl, 4u);
                        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_swiglu_bf16_];
                        [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                        [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                        [enc setBuffer:(id<MTLBuffer>)buf_normed_ offset:0 atIndex:2];
                        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:3];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(sl.intermediate_size / 16u, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ane_hf4_down_sk4_bf16_];
                        [enc setBuffer:(id<MTLBuffer>)sl.hwx_buf offset:0 atIndex:0];
                        [enc setBuffer:(id<MTLBuffer>)sl.buf_scales_offs offset:0 atIndex:1];
                        [enc setBuffer:(id<MTLBuffer>)buf_mlp_gate_ offset:0 atIndex:2];
                        [enc setBuffer:(id<MTLBuffer>)buf_sk_part_ offset:0 atIndex:3];
                        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
                        [enc dispatchThreadgroups:MTLSizeMake(16u * ceil_div(sl.hidden_size / 128u, 4u), 1, 2u * sl.n_subblocks)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                }
            });
    }

    // 12. MTP drafting & Prefill GPU vs ANE component timings
    if (mtp_loaded_) {
        std::printf("\n--- MTP Drafting Component Timings (%s, V=%u) ---\n",
                    bs_enabled_ ? "BitScreen 1-bit + Q4 rescore" : "Q4 row_map",
                    bs_enabled_ ? bs_vocab_ : draft_row_count_);
        for (int B : {1, 2, 3}) {
            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "1x MTP draft_lm_head ONLY (B=%d)", B);
            bench_gpu(lbl, 0.0, [&](id<MTLComputeCommandEncoder> enc) {
                uint32_t dV = 0u, dnp = 0u;
                (void)encode_draft_lm_head(enc, static_cast<uint32_t>(B), dV, dnp);
            });
            std::snprintf(lbl, sizeof(lbl), "1x MTP full step (fc+layer+draft_lmh, B=%d)", B);
            bench_gpu(lbl, 0.0, [&](id<MTLComputeCommandEncoder> enc) {
                encode_mtp_steps(enc, buf_token_ids_, 0, buf_mtp_seed_hid_, 0, B, 512, false);
                uint32_t dV = 0u, dnp = 0u;
                (void)encode_draft_lm_head(enc, static_cast<uint32_t>(B), dV, dnp);
            });
        }
        bench_gpu("1x MTP prefill KV-prime (M=512, pos=512)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            encode_mtp_kv_prime(enc, buf_token_ids_, 0, buf_mtp_in_hid_, 0, 512, 512);
        });
        bench_gpu("1x MTP decode root-sync M=6 (pos=512)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            encode_mtp_steps(enc, buf_token_ids_, 0, buf_mtp_verify_hid_, 0, 6, 512, false);
        });
    }
    if (dflash_loaded_) {
        const DFlashConfig& dcfg = dflash_->config();
        std::printf("\n--- DFlash 2 Drafting Component Timings (%dL, block=%d, win=%d) ---\n",
                    dcfg.num_layers, dcfg.block_size, dcfg.sliding_window);
        int* df_ids = reinterpret_cast<int*>([(id<MTLBuffer>)buf_df_ids_ contents]);
        df_ids[0] = 100;
        for (int i = 1; i < dcfg.block_size; ++i) df_ids[i] = dcfg.mask_token_id;
        bench_gpu("1x DFlash2 context commit (M=6, fc+5L KV)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            encode_dflash_context_commit(enc, 0, 6, 512);
        });
        df_pending_commit_ = 0;
        df_kv_len_ = 512;
        bench_gpu("1x DFlash2 draft block (5L+lmh M=7+topk16, L=512)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            encode_dflash_draft_step(enc, 512);
        });
        df_kv_len_ = dcfg.sliding_window;
        bench_gpu("1x DFlash2 draft block (5L+lmh M=7+topk16, L=2048)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            encode_dflash_draft_step(enc, dcfg.sliding_window);
        });
        const LogitCand* topk_ptr = reinterpret_cast<const LogitCand*>([(id<MTLBuffer>)buf_df_topk_out_ contents]);
        const uint16_t* sel_ptr   = reinterpret_cast<const uint16_t*>([(id<MTLBuffer>)buf_df_sel_h_ contents]);
        int tmp_draft[MAX_VERIFY] = {};
        auto ts0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 20; ++r) dflash_->select_greedy(100, topk_ptr, sel_ptr, tmp_draft);
        double sel_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ts0).count() / 20.0;
        std::printf("  %-46s %6.1f us\n", "1x DFlash2 CPU NEON CandidateSelector (greedy)", sel_us);
        std::fflush(stdout);
    }

    std::printf("\n--- Prefill GPU vs ANE Component Timings (64 layers, M=256 / M=512) ---\n");
    bench_gpu("64x GPU Attn+GDN ONLY (M=512, pos=0)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            encode_layer_attn_only(enc, i, buf_hidden_, 512u, 0, false, false);
        }
    });
    bench_gpu("48x GPU GDN-only (M=512, pos=0)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (layers_[i].is_linear) encode_layer_attn_only(enc, i, buf_hidden_, 512u, 0, false, false);
        }
    });
    bench_gpu("16x GPU Full-Attn-only (M=512, pos=0)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) encode_layer_attn_only(enc, i, buf_hidden_, 512u, 0, false, false);
        }
    });
    ensure_kv_capacity(4096);
    bench_gpu("16x GPU Full-Attn-only (M=512, pos=3072)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
        for (int i = 0; i < num_layers_; ++i) {
            if (!layers_[i].is_linear) encode_layer_attn_only(enc, i, buf_hidden_, 512u, 3072, false, false);
        }
    });
    if (!has_share) {
        bench_gpu("64x GPU g64 MLP tiled_f16x (M=256)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            for (int i = 0; i < num_layers_; ++i) {
                encode_mlp_q4(enc, layers_[i], 256u, nullptr, nil);
            }
        });
        bench_gpu("64x GPU g64 MLP tiled_f16x (M=512)", 0.0, [&](id<MTLComputeCommandEncoder> enc) {
            for (int i = 0; i < num_layers_; ++i) {
                encode_mlp_q4(enc, layers_[i], 512u, nullptr, nil);
            }
        });
    }
    if (ane_ && ane_->ready()) {
        for (int nsub : {1, 2}) {
            // Warmup 1 pass
            for (int i = 0; i < num_layers_; ++i) { ane_->submit(i, 0, nsub); ane_->wait(); }
            auto ta0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 2; ++r) {
                for (int i = 0; i < num_layers_; ++i) { ane_->submit(i, 0, nsub); ane_->wait(); }
            }
            double ane_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta0).count() / 2.0;
            std::printf("  64x ANE MLP (%dx256 = %d tok)                  %6.2f ms  (%.1f tok/s MLP-only)\n",
                        nsub, nsub * 256, ane_ms, (nsub * 256) / (ane_ms / 1000.0));
        }
    }
    std::printf("============================================================================\n");
    std::fflush(stdout);
}

} // namespace mi
