#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "inkling2.hpp"
#include "json.hpp"
#include "prefix_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace mi {

namespace {

struct QmmParams { uint32_t M, N, K, add_residual; };
struct EmbedParams { uint32_t M, D; float eps; };
struct NormParams { uint32_t M, D; float eps; };
struct AttnResParams {
    uint32_t M, D, n_closed, use_partial, n_q, closed_stride;
    float eps, scale;
};
struct CtrlParams { uint32_t M, D, R; float eps; };
struct SconvParams { uint32_t M, C, H, has_add; float inv_div; uint32_t save_tape; };
struct InkAttnParams {
    uint32_t M, Hq, Hkv, D, d_rel, rel_extent, window, pos, kv_len, kv_cap;
    float qk_clip, rel_clip, log_alpha, log_floor, eps, scale;
    uint32_t num_splits;
};
struct RouterParams { uint32_t M, D, E, K; float route_scale; };
struct ActQuantParams { uint32_t rows, width; };
struct MoeNvfp4Params { uint32_t M, D, I, K_top; float gate_cap, up_cap; uint32_t use_act_quant; };
struct MoeGroupParams { uint32_t total_slots, E; };
struct SoftcapParams { uint32_t N; float cap; };
struct MtpMixParams { uint32_t M_out, D, row_offset; float eps, b0, b1, b2, b3; };
struct MtpAttnParams {
    uint32_t M, Hq, Hkv, D, pos, ctx_len, kv_cap, window;
    float eps, rope_theta, scale;
};
struct MtpAddNormParams { uint32_t M, D, has_res, write_sum; float eps; };
struct MarkovPrepParams { uint32_t rank, D, step, prev_idx; };

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

inline float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

} // namespace

Inkling2Model::LinearWeight Inkling2Model::load_linear(
    const std::string& full_key, const std::unordered_map<std::string, int>& quant_bits
) {
    LinearWeight lw;
    lw.w = store_.get(full_key);
    auto it = quant_bits.find(full_key);
    if (it != quant_bits.end() && store_.has(full_key + ".mtplx_scales")) {
        lw.s = store_.get(full_key + ".mtplx_scales");
        lw.b = store_.get(full_key + ".mtplx_biases");
        lw.out_dim = static_cast<uint32_t>(lw.w.shape[0]);
        if (it->second == 8) {
            lw.kind = LinKind::Q8_G64;
            lw.in_dim = static_cast<uint32_t>(lw.w.shape[1] * 4);
        } else if (it->second == 4) {
            lw.kind = LinKind::Q4_G64;
            lw.in_dim = static_cast<uint32_t>(lw.w.shape[1] * 8);
        } else {
            throw std::runtime_error("Unsupported mtplx bit width for " + full_key);
        }
    } else {
        lw.kind = LinKind::BF16;
        lw.out_dim = static_cast<uint32_t>(lw.w.shape[0]);
        lw.in_dim  = static_cast<uint32_t>(lw.w.shape[1]);
    }
    return lw;
}

Inkling2Model::Inkling2Model(
    MetalContext& ctx, const std::string& model_dir, int max_seq_len,
    bool exact_nvfp4_act, const std::string& mtp_dir, int mtp_depth
)
    : ctx_(ctx), max_seq_len_(max_seq_len), exact_nvfp4_act_(exact_nvfp4_act),
      mtp_depth_(std::clamp(mtp_depth, 1, 3)) {
    auto t0 = std::chrono::steady_clock::now();

    std::string shader_dir = MOMETAL_SHADER_DIR;
    ctx_.load_shader_file("common", shader_dir + "/common.metal");
    ctx_.load_shader_file("inkling2", shader_dir + "/inkling2.metal");

    pso_qmm_q4_              = ctx_.pipeline("common", "qmm_q4_g64_f32");
    pso_qmm_q4_b4_           = ctx_.pipeline("common", "qmm_q4_g64_m8");
    pso_qmm_q4_tiled_        = ctx_.pipeline("common", "qmm_q4_g64_tiled");
    pso_qmm_q8_              = ctx_.pipeline("common", "qmm_q8_g64_f32");
    pso_qmm_q8_b4_           = ctx_.pipeline("common", "qmm_q8_g64_m8");
    pso_qmm_q8_tiled_        = ctx_.pipeline("common", "qmm_q8_g64_tiled");
    pso_qmm_bf16_            = ctx_.pipeline("common", "qmm_bf16_f32");
    pso_qmm_bf16_b4_         = ctx_.pipeline("common", "qmm_bf16_m8");
    pso_qmm_bf16_tiled_      = ctx_.pipeline("common", "qmm_bf16_tiled");
    pso_embed_q4_            = ctx_.pipeline("common", "embed_q4_g64_f32");
    pso_rmsnorm_             = ctx_.pipeline("common", "rmsnorm_f32_bf16w");
    pso_swiglu_              = ctx_.pipeline("common", "swiglu_f32");
    pso_copy_                = ctx_.pipeline("common", "vec_copy_f32");
    pso_argmax1_             = ctx_.pipeline("common", "argmax_stage1");
    pso_argmax2_             = ctx_.pipeline("common", "argmax_stage2");

    pso_ink_embed_           = ctx_.pipeline("inkling2", "inkling_embed_bf16_norm");
    pso_ink_attn_res_        = ctx_.pipeline("inkling2", "inkling_attn_res");
    pso_ink_attn_res_norm1_  = ctx_.pipeline("inkling2", "inkling_attn_res_norm1");
    pso_ink_ctrl_fused_      = ctx_.pipeline("inkling2", "inkling_controller_norm_fused");
    pso_ink_ctrl_prep_       = ctx_.pipeline("inkling2", "inkling_controller_prep");
    pso_ink_ctrl_apply_      = ctx_.pipeline("inkling2", "inkling_controller_apply");
    pso_ink_sconv_st_        = ctx_.pipeline("inkling2", "inkling_sconv_static");
    pso_ink_sconv_dy_        = ctx_.pipeline("inkling2", "inkling_sconv_dynamic");
    pso_ink_q_rel_prep_      = ctx_.pipeline("inkling2", "inkling_q_rel_prep");
    pso_ink_kv_prep_         = ctx_.pipeline("inkling2", "inkling_kv_prep_write");
    pso_ink_rel_prof_        = ctx_.pipeline("inkling2", "inkling_rel_profiles");
    pso_ink_attn_            = ctx_.pipeline("inkling2", "inkling_rel_sink_attn_gated");
    pso_ink_router_sel_      = ctx_.pipeline("inkling2", "inkling_router_topk_select");
    pso_ink_act_q_           = ctx_.pipeline("inkling2", "inkling_nvfp4_quant_act");
    pso_ink_mxfp8_act_q_     = ctx_.pipeline("inkling2", "inkling_mxfp8_quant_act");
    pso_ink_moe_w13_         = ctx_.pipeline("inkling2", "inkling_moe_nvfp4_w13_swiglu");
    pso_ink_moe_w2_          = ctx_.pipeline("inkling2", "inkling_moe_nvfp4_w2_combine");
    pso_ink_moe_mxfp8_w13_   = ctx_.pipeline("inkling2", "inkling_moe_mxfp8_w13_swiglu");
    pso_ink_moe_mxfp8_w2_    = ctx_.pipeline("inkling2", "inkling_moe_mxfp8_w2_combine");
    pso_ink_moe_sort_        = ctx_.pipeline("inkling2", "inkling_moe_group_sort");
    pso_ink_moe_w13_grp_     = ctx_.pipeline("inkling2", "inkling_moe_nvfp4_w13_grouped");
    pso_ink_moe_w2_grp_      = ctx_.pipeline("inkling2", "inkling_moe_nvfp4_w2_grouped");
    pso_ink_moe_mxfp8_w13_grp_ = ctx_.pipeline("inkling2", "inkling_moe_mxfp8_w13_grouped");
    pso_ink_moe_mxfp8_w2_grp_  = ctx_.pipeline("inkling2", "inkling_moe_mxfp8_w2_grouped");
    pso_ink_moe_reduce_      = ctx_.pipeline("inkling2", "inkling_moe_reduce_slots");
    pso_ink_softcap_         = ctx_.pipeline("inkling2", "inkling_logit_softcap");

    pso_mtp_mix_norm_        = ctx_.pipeline("inkling2", "inkling_mtp_mix_norm");
    pso_mtp_k_rope_          = ctx_.pipeline("inkling2", "inkling_mtp_k_rope_write");
    pso_mtp_qkv_prep_        = ctx_.pipeline("inkling2", "inkling_mtp_qkv_rope_prep");
    pso_mtp_attn_            = ctx_.pipeline("inkling2", "inkling_mtp_attn");
    pso_mtp_add_norm_        = ctx_.pipeline("inkling2", "inkling_mtp_add_norm");
    pso_mtp_fill_masks_      = ctx_.pipeline("inkling2", "inkling_mtp_fill_masks");
    pso_mtp_markov_prep_     = ctx_.pipeline("inkling2", "inkling_mtp_markov_prep");
    pso_mtp_markov_gate_     = ctx_.pipeline("inkling2", "inkling_mtp_markov_gate");
    pso_mtp_add_h_bias_      = ctx_.pipeline("inkling2", "inkling_mtp_add_h_bias");

    Json cfg = parse_json(read_file_to_string(model_dir + "/config.json"));
    const Json& tc = cfg["text_config"];

    hidden_size_         = static_cast<int>(tc["hidden_size"].as_int(2304));
    num_layers_          = static_cast<int>(tc["num_hidden_layers"].as_int(24));
    vocab_size_          = static_cast<int>(tc["vocab_size"].as_int(201024));
    unpadded_vocab_size_ = static_cast<int>(tc["unpadded_vocab_size"].as_int(200058));
    num_heads_           = static_cast<int>(tc["num_attention_heads"].as_int(24));
    num_kv_heads_        = static_cast<int>(tc["num_key_value_heads"].as_int(8));
    head_dim_            = static_cast<int>(tc["head_dim"].as_int(128));
    d_rel_               = static_cast<int>(tc["d_rel"].as_int(16));
    rel_extent_          = static_cast<int>(tc["rel_extent"].as_int(1024));
    sliding_window_      = static_cast<int>(tc["sliding_window_size"].as_int(1024));
    attn_res_block_      = static_cast<int>(tc["n_layers_per_attn_res_block"].as_int(6));
    num_experts_         = static_cast<int>(tc["n_routed_experts"].as_int(128));
    top_k_               = static_cast<int>(tc["num_experts_per_tok"].as_int(8));
    moe_inter_           = static_cast<int>(tc["intermediate_size"].as_int(1152));
    route_scale_         = static_cast<float>(tc["route_scale"].as_double(8.0));
    gate_softcap_        = static_cast<float>(tc["mlp_gate_softcap"].as_double(4.0));
    up_softcap_          = static_cast<float>(tc["mlp_up_softcap"].as_double(25.0));
    logit_softcap_       = static_cast<float>(tc["final_logit_softcapping"].as_double(30.0));
    qk_clip_             = static_cast<float>(tc["qk_l2_clip_threshold"].as_double(22.627416997969522));
    rel_clip_            = static_cast<float>(tc["rel_l2_clip_threshold"].as_double(8.0));
    log_alpha_           = static_cast<float>(tc["log_scaling_alpha"].as_double(0.1));
    log_floor_           = static_cast<float>(tc["log_scaling_n_floor"].as_double(8192.0));
    rms_eps_             = static_cast<float>(tc["rms_norm_eps"].as_double(1e-6));

    const Json& lc = tc["loop_config"];
    core_start_ = static_cast<int>(lc["core_start"].as_int(6));
    core_end_   = static_cast<int>(lc["core_end"].as_int(18));
    num_passes_ = static_cast<int>(lc["num_passes"].as_int(3));
    gate_rank_  = static_cast<int>(lc["gate_rank"].as_int(64));

    std::set<int> local_ids;
    for (const auto& lid : tc["local_layer_ids"].as_array()) {
        local_ids.insert(static_cast<int>(lid.as_int()));
    }

    // Parse optional mtplx_inkling_precision overrides
    std::unordered_map<std::string, int> quant_bits;
    if (cfg.contains("mtplx_inkling_precision") && cfg["mtplx_inkling_precision"].is_object()) {
        for (const auto& [k, v] : cfg["mtplx_inkling_precision"].as_object()) {
            quant_bits[k] = static_cast<int>(v["bits"].as_int(8));
        }
    }

    // Zero-copy mmap weights (skip vision tower)
    store_.load_dir(ctx_.device(), model_dir, {"model.visual."});

    const std::string pfx = "model.llm.";
    embed_        = load_linear(pfx + "embed.weight", quant_bits);
    embed_norm_   = store_.get(pfx + "embed_norm.weight");
    final_norm_   = store_.get(pfx + "norm.weight");
    output_query_ = store_.get(pfx + "attn_res.output_query");
    unembed_      = load_linear(pfx + "unembed.weight", quant_bits);

    // Assemble cross-pass message queries [core_size, 4, D] from main [core_size*2, D] and kv [core_size*2, D]
    int core_size = core_end_ - core_start_;
    const TensorView& mq_main = store_.get(pfx + "middle_cycle_cross_pass_queries");
    const TensorView& mq_kv   = store_.get(pfx + "middle_cycle_cross_pass_kv_queries");
    const float* main_p = reinterpret_cast<const float*>(mq_main.host_ptr);
    const float* kv_p   = reinterpret_cast<const float*>(mq_kv.host_ptr);
    msg_queries_buf_.resize(core_size);
    for (int i = 0; i < core_size; ++i) {
        std::vector<float> q4(static_cast<size_t>(4) * hidden_size_);
        // [main[i,0], kv[i,0], kv[i,1], main[i,1]]
        std::memcpy(q4.data() + 0 * hidden_size_, main_p + (2 * i + 0) * hidden_size_, hidden_size_ * sizeof(float));
        std::memcpy(q4.data() + 1 * hidden_size_, kv_p   + (2 * i + 0) * hidden_size_, hidden_size_ * sizeof(float));
        std::memcpy(q4.data() + 2 * hidden_size_, kv_p   + (2 * i + 1) * hidden_size_, hidden_size_ * sizeof(float));
        std::memcpy(q4.data() + 3 * hidden_size_, main_p + (2 * i + 1) * hidden_size_, hidden_size_ * sizeof(float));
        msg_queries_buf_[i] = ctx_.alloc_shared_copy(q4.data(), q4.size() * sizeof(float));
    }

    // Load 24 physical layers
    layers_.resize(num_layers_);
    for (int i = 0; i < num_layers_; ++i) {
        std::string lp = pfx + "layers." + std::to_string(i) + ".";
        LayerWeights& lw = layers_[i];
        lw.is_sliding  = (local_ids.count(i) > 0);
        lw.is_boundary = (i % attn_res_block_) == 0;
        if (i > 0) lw.attn_query = store_.get(pfx + "attn_res.attn_queries." + std::to_string(i - 1));
        lw.mlp_query = store_.get(pfx + "attn_res.mlp_queries." + std::to_string(i));

        if (lw.is_sliding) {
            lw.attn_norm = store_.get(lp + "attn_norm.weight");
        } else {
            lw.q_input_norm = store_.get(lp + "q_input_norm.weight");
            lw.k_input_norm = store_.get(lp + "k_input_norm.weight");
            lw.v_input_norm = store_.get(lp + "v_input_norm.weight");
        }
        lw.mlp_norm = store_.get(lp + "mlp_norm.weight");

        lw.wq_du     = load_linear(lp + "attn.wq_du.weight", quant_bits);
        lw.wk_dv     = load_linear(lp + "attn.wk_dv.weight", quant_bits);
        lw.wv_dv     = load_linear(lp + "attn.wv_dv.weight", quant_bits);
        lw.wr_du     = load_linear(lp + "attn.wr_du.weight", quant_bits);
        lw.sink_proj = load_linear(lp + "attn.attn_sink_proj.weight", quant_bits);
        lw.gate_attn = load_linear(lp + "attn.gate_attn.weight", quant_bits);
        lw.wo_ud     = load_linear(lp + "attn.wo_ud.weight", quant_bits);

        lw.q_scale   = store_.get(lp + "attn.q_l2_clip.scale");
        lw.r_scale   = store_.get(lp + "attn.rel_l2_clip.scale");
        lw.rel_basis = store_.get(lp + "attn.rel_logits_proj.proj");

        if (lw.is_sliding) {
            lw.k_sconv_w = store_.get(lp + "attn.k_sconv.weight");
            lw.v_sconv_w = store_.get(lp + "attn.v_sconv.weight");
        } else {
            lw.k_sconv_w        = store_.get(lp + "attn.k_sconv.static_w");
            lw.k_sconv_dyn_proj = store_.get(lp + "attn.k_sconv.weight_proj");
            lw.v_sconv_w        = store_.get(lp + "attn.v_sconv.static_w");
            lw.v_sconv_dyn_proj = store_.get(lp + "attn.v_sconv.weight_proj");
        }
        lw.attn_sconv_w = store_.get(lp + "attn_sconv.weight");
        lw.mlp_sconv_w  = store_.get(lp + "mlp_sconv.weight");

        lw.router_w    = store_.get(lp + "mlp.gate.weight");
        lw.router_bias = store_.get(lp + "mlp.gate.bias");
        lw.w13_w       = store_.get(lp + "mlp.experts.w13_weight");
        lw.w13_scale   = store_.get(lp + "mlp.experts.w13_weight_scale");
        lw.w2_w        = store_.get(lp + "mlp.experts.w2_weight");
        lw.w2_scale    = store_.get(lp + "mlp.experts.w2_weight_scale");
        lw.is_mxfp8    = (lw.w13_w.dtype == DType::F8_E4M3);
        if (!lw.is_mxfp8) {
            lw.w13_amax  = store_.get(lp + "mlp.experts.w13_weight_amax");
            lw.w13_denom = store_.get(lp + "mlp.experts.w13_weight_outer_scale_denominator");
            lw.w2_amax   = store_.get(lp + "mlp.experts.w2_weight_amax");
            lw.w2_denom  = store_.get(lp + "mlp.experts.w2_weight_outer_scale_denominator");
        }

        if (store_.has(lp + "controllers.attn.down_weight")) {
            lw.has_controllers = true;
            const char* cnames[4] = {"attn", "k", "v", "mlp"};
            for (int c = 0; c < 4; ++c) {
                std::string cp = lp + "controllers." + cnames[c] + ".";
                lw.ctrl[c].down   = store_.get(cp + "down_weight");
                lw.ctrl[c].forget = store_.get(cp + "forget_weight");
                lw.ctrl[c].gain   = store_.get(cp + "gain_weight");
            }
        }
    }

    // Build 48-visit execution schedule: [0..5, (6..17) x 3, 18..23]
    for (int i = 0; i < core_start_; ++i) visit_order_.push_back(i);
    for (int p = 0; p < num_passes_; ++p) {
        for (int i = core_start_; i < core_end_; ++i) visit_order_.push_back(i);
    }
    for (int i = core_end_; i < num_layers_; ++i) visit_order_.push_back(i);

    // Allocate per-visit KV + short-conv caches (sliding kv_cap = window + 16 for overwrite-free MTP verify!)
    visit_states_.resize(visit_order_.size());
    size_t kv_bytes = 0;
    size_t kv_dim = static_cast<size_t>(num_kv_heads_) * head_dim_;
    for (size_t v = 0; v < visit_order_.size(); ++v) {
        int lidx = visit_order_[v];
        VisitState& vs = visit_states_[v];
        vs.is_sliding = layers_[lidx].is_sliding;
        vs.kv_cap = vs.is_sliding ? (sliding_window_ + 16) : max_seq_len_;
        size_t kv_sz = static_cast<size_t>(vs.kv_cap) * kv_dim * sizeof(uint16_t);
        vs.k_cache = ctx_.alloc_private(kv_sz);
        vs.v_cache = ctx_.alloc_private(kv_sz);
        vs.conv_k    = ctx_.alloc_private(3ull * kv_dim * sizeof(float));
        vs.conv_v    = ctx_.alloc_private(3ull * kv_dim * sizeof(float));
        vs.conv_attn = ctx_.alloc_private(3ull * hidden_size_ * sizeof(float));
        vs.conv_mlp  = ctx_.alloc_private(3ull * hidden_size_ * sizeof(float));
        kv_bytes += 2 * kv_sz + 3ull * (2 * kv_dim + 2 * hidden_size_) * sizeof(float);
    }

    // Optional DSpark MTP drafter loading (auto-detects <model_dir>/mtp.safetensors)
    std::string resolved_mtp = mtp_dir;
    if (resolved_mtp == "off" || resolved_mtp == "none") {
        resolved_mtp.clear();
    } else if (resolved_mtp == "auto") {
        std::string cand = model_dir + "/mtp.safetensors";
        if (std::filesystem::exists(cand)) resolved_mtp = cand;
        else resolved_mtp.clear();
    }

    if (!resolved_mtp.empty()) {
        if (std::filesystem::is_directory(resolved_mtp)) {
            resolved_mtp = resolved_mtp + "/mtp.safetensors";
        }
        mtp_store_.load_file(ctx_.device(), resolved_mtp);

        std::set<int> mtp_local = {0, 2};
        if (tc.contains("mtp_config") && tc["mtp_config"].is_object()) {
            const Json& mc = tc["mtp_config"];
            mtp_num_layers_  = static_cast<int>(mc["num_hidden_layers"].as_int(3));
            mtp_inter_       = static_cast<int>(mc["intermediate_size"].as_int(9216));
            mtp_block_size_  = static_cast<int>(mc["block_size"].as_int(8));
            mtp_markov_rank_ = static_cast<int>(mc["markov_rank"].as_int(256));
            mtp_rope_theta_  = static_cast<float>(mc["rope_theta"].as_double(1000000.0));
            if (mc.contains("local_layer_ids") && mc["local_layer_ids"].is_array()) {
                mtp_local.clear();
                for (const auto& lid : mc["local_layer_ids"].as_array()) mtp_local.insert(static_cast<int>(lid.as_int()));
            }
            if (mc.contains("target_layer_ids") && mc["target_layer_ids"].is_array()) {
                const auto& tids = mc["target_layer_ids"].as_array();
                for (size_t i = 0; i < std::min<size_t>(4, tids.size()); ++i) {
                    target_capture_ids_[i] = static_cast<int>(tids[i].as_int());
                }
            }
        }

        // Compute softmax(fuse_weight [4, 4], axis=-1) once on CPU
        TensorView fw = mtp_store_.get("fuse_weight");
        const uint16_t* fw_ptr = reinterpret_cast<const uint16_t*>(fw.host_ptr);
        for (int r = 0; r < 4; ++r) {
            float vals[4];
            float mx = -1e30f;
            for (int c = 0; c < 4; ++c) {
                vals[c] = bf16_bits_to_f32(fw_ptr[r * 4 + c]);
                mx = std::max(mx, vals[c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) { vals[c] = std::exp(vals[c] - mx); sum += vals[c]; }
            for (int c = 0; c < 4; ++c) mtp_blend_[r][c] = vals[c] / sum;
        }

        mtp_mask_embed_         = mtp_store_.get("mask_embed");
        mtp_output_norm_        = mtp_store_.get("norm.weight");
        mtp_markov_dflare_norm_ = mtp_store_.get("markov_head.dflare_norm.weight");
        mtp_markov_w1_          = mtp_store_.get("markov_head.markov_w1.weight");
        mtp_markov_rnn_w_       = mtp_store_.get("markov_head.rnn_proj.weight");
        mtp_markov_rnn_b_       = mtp_store_.get("markov_head.rnn_proj.bias");
        mtp_markov_hid_w_       = mtp_store_.get("markov_head.hidden_out.weight");
        mtp_markov_hid_b_       = mtp_store_.get("markov_head.hidden_out.bias");

        mtp_blocks_.resize(mtp_num_layers_);
        mtp_states_.resize(mtp_num_layers_);
        for (int b = 0; b < mtp_num_layers_; ++b) {
            std::string lp = "layers." + std::to_string(b) + ".";
            DraftBlockWeights& bw = mtp_blocks_[b];
            bw.is_sliding          = (mtp_local.count(b) > 0);
            bw.dflare_norm         = mtp_store_.get(lp + "dflare_norm.weight");
            bw.input_layernorm     = mtp_store_.get(lp + "input_layernorm.weight");
            bw.post_attn_layernorm = mtp_store_.get(lp + "post_attention_layernorm.weight");
            bw.q_proj              = mtp_store_.get(lp + "self_attn.q_proj.weight");
            bw.k_proj              = mtp_store_.get(lp + "self_attn.k_proj.weight");
            bw.v_proj              = mtp_store_.get(lp + "self_attn.v_proj.weight");
            bw.o_proj              = mtp_store_.get(lp + "self_attn.o_proj.weight");
            bw.k_proj_ctx          = mtp_store_.get(lp + "self_attn.k_proj_ctx.weight");
            bw.v_proj_ctx          = mtp_store_.get(lp + "self_attn.v_proj_ctx.weight");
            bw.q_norm              = mtp_store_.get(lp + "self_attn.q_norm.weight");
            bw.k_norm              = mtp_store_.get(lp + "self_attn.k_norm.weight");
            bw.k_norm_ctx          = mtp_store_.get(lp + "self_attn.k_norm_ctx.weight");
            bw.gate_proj           = mtp_store_.get(lp + "mlp.gate_proj.weight");
            bw.up_proj             = mtp_store_.get(lp + "mlp.up_proj.weight");
            bw.down_proj           = mtp_store_.get(lp + "mlp.down_proj.weight");

            DraftBlockState& bs = mtp_states_[b];
            bs.is_sliding = bw.is_sliding;
            bs.kv_cap = bs.is_sliding ? sliding_window_ : max_seq_len_;
            size_t dkv_sz = static_cast<size_t>(bs.kv_cap) * kv_dim * sizeof(uint16_t);
            bs.k_cache = ctx_.alloc_private(dkv_sz);
            bs.v_cache = ctx_.alloc_private(dkv_sz);
            kv_bytes += 2 * dkv_sz;
        }

        // Per-visit 4-step conv tapes for instant zero-replay rollback (~15.3 MB total)
        for (auto& vs : visit_states_) {
            vs.tape_k    = ctx_.alloc_private(4ull * 3ull * kv_dim * sizeof(float));
            vs.tape_v    = ctx_.alloc_private(4ull * 3ull * kv_dim * sizeof(float));
            vs.tape_attn = ctx_.alloc_private(4ull * 3ull * hidden_size_ * sizeof(float));
            vs.tape_mlp  = ctx_.alloc_private(4ull * 3ull * hidden_size_ * sizeof(float));
            kv_bytes += 4ull * 3ull * (2 * kv_dim + 2 * hidden_size_) * sizeof(float);
        }

        size_t C_sz = static_cast<size_t>(chunk_size_);
        size_t D_sz = static_cast<size_t>(hidden_size_);
        for (int c = 0; c < 4; ++c) buf_mtp_cap_[c] = ctx_.alloc_private(C_sz * D_sz * sizeof(float));
        buf_mtp_mix_           = ctx_.alloc_private(C_sz * D_sz * sizeof(float));
        buf_mtp_markov_ctx_    = ctx_.alloc_private(D_sz * sizeof(float));
        buf_mtp_h_             = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * D_sz * sizeof(float));
        buf_mtp_res_           = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * D_sz * sizeof(float));
        buf_mtp_normed_        = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * D_sz * sizeof(float));
        buf_mtp_k_self_        = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * kv_dim * sizeof(uint16_t));
        buf_mtp_v_self_        = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * kv_dim * sizeof(uint16_t));
        buf_mtp_mlp_g_         = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * mtp_inter_ * sizeof(float));
        buf_mtp_mlp_u_         = ctx_.alloc_private(static_cast<size_t>(mtp_block_size_) * mtp_inter_ * sizeof(float));
        buf_mtp_rnn_in_        = ctx_.alloc_private((2ull * mtp_markov_rank_ + 2ull * D_sz) * sizeof(float));
        buf_mtp_rnn_out_       = ctx_.alloc_private(3ull * mtp_markov_rank_ * sizeof(float));
        buf_mtp_rnn_state_     = ctx_.alloc_private(static_cast<size_t>(mtp_markov_rank_) * sizeof(float));
        buf_mtp_signal_        = ctx_.alloc_private(static_cast<size_t>(mtp_markov_rank_) * sizeof(float));
        buf_mtp_eff_h_         = ctx_.alloc_private(D_sz * sizeof(float));
        buf_mtp_tok_ring_      = ctx_.alloc_shared(8 * sizeof(uint32_t));
        buf_mtp_verify_logits_ = ctx_.alloc_private(4ull * unpadded_vocab_size_ * sizeof(float));
        mtp_loaded_ = true;
    }

    // Allocate persistent GPU scratch arena
    size_t C = static_cast<size_t>(chunk_size_);
    size_t D = static_cast<size_t>(hidden_size_);
    size_t q_dim = static_cast<size_t>(num_heads_) * head_dim_;

    buf_token_ids_     = ctx_.alloc_shared(C * sizeof(int));
    buf_x_             = ctx_.alloc_private(C * D * sizeof(float));
    buf_entry_x_       = ctx_.alloc_private(C * D * sizeof(float));
    buf_closed_        = ctx_.alloc_private(6ull * C * D * sizeof(float));
    buf_entry_closed_  = ctx_.alloc_private(4ull * C * D * sizeof(float));
    buf_msg_bank_      = ctx_.alloc_private(4ull * C * D * sizeof(float));
    buf_msg_out_       = ctx_.alloc_private(4ull * C * D * sizeof(float));
    buf_read_out_      = ctx_.alloc_private(3ull * C * D * sizeof(float));
    buf_ctrl_mixed_    = ctx_.alloc_private(C * 2 * D * sizeof(float));
    buf_ctrl_state_    = ctx_.alloc_private(C * gate_rank_ * sizeof(float));
    buf_q_in_          = ctx_.alloc_private(C * D * sizeof(float));
    buf_k_in_          = ctx_.alloc_private(C * D * sizeof(float));
    buf_v_in_          = ctx_.alloc_private(C * D * sizeof(float));
    buf_q_lin_         = ctx_.alloc_private(C * q_dim * sizeof(float));
    buf_k_lin_         = ctx_.alloc_private(C * kv_dim * sizeof(float));
    buf_v_lin_         = ctx_.alloc_private(C * kv_dim * sizeof(float));
    buf_k_conv_        = ctx_.alloc_private(C * kv_dim * sizeof(float));
    buf_v_conv_        = ctx_.alloc_private(C * kv_dim * sizeof(float));
    buf_dyn_w_k_       = ctx_.alloc_private(C * num_kv_heads_ * 4 * sizeof(float));
    buf_dyn_w_v_       = ctx_.alloc_private(C * num_kv_heads_ * 4 * sizeof(float));
    buf_router_logits_ = ctx_.alloc_private(C * num_experts_ * sizeof(float));
    buf_r_lin_         = ctx_.alloc_private(C * num_heads_ * d_rel_ * sizeof(float));
    buf_sink_          = ctx_.alloc_private(C * num_heads_ * sizeof(float));
    buf_gate_          = ctx_.alloc_private(C * q_dim * sizeof(float));
    buf_q_ready_       = ctx_.alloc_private(C * q_dim * sizeof(float));
    buf_r_ready_       = ctx_.alloc_private(C * num_heads_ * d_rel_ * sizeof(float));
    buf_profiles_      = ctx_.alloc_private(C * num_heads_ * rel_extent_ * sizeof(float));
    buf_attn_out_      = ctx_.alloc_private(C * q_dim * sizeof(float));
    buf_proj_out_      = ctx_.alloc_private(C * D * sizeof(float));
    buf_residual_      = ctx_.alloc_private(C * D * sizeof(float));
    buf_ffn_in_        = ctx_.alloc_private(C * D * sizeof(float));
    buf_ffn_q_         = ctx_.alloc_private(C * D * sizeof(float));
    buf_ffn_outer_     = ctx_.alloc_private(C * sizeof(float));
    buf_top_idx_       = ctx_.alloc_private(C * top_k_ * sizeof(uint32_t));
    buf_top_w_         = ctx_.alloc_private(C * top_k_ * sizeof(float));
    buf_moe_mid_       = ctx_.alloc_private(C * top_k_ * moe_inter_ * sizeof(float));
    buf_moe_mid_q_     = ctx_.alloc_private(C * top_k_ * moe_inter_ * sizeof(float));
    buf_moe_mid_outer_ = ctx_.alloc_private(C * top_k_ * sizeof(float));
    buf_moe_out_       = ctx_.alloc_private(C * D * sizeof(float));
    buf_moe_exp_cnt_   = ctx_.alloc_private(static_cast<size_t>(num_experts_) * sizeof(uint32_t));
    buf_moe_exp_off_   = ctx_.alloc_private(static_cast<size_t>(num_experts_) * sizeof(uint32_t));
    buf_moe_sorted_    = ctx_.alloc_private(C * top_k_ * sizeof(uint32_t));
    buf_moe_slot_out_  = ctx_.alloc_private(C * top_k_ * D * sizeof(float));
    buf_logits_        = ctx_.alloc_shared(static_cast<size_t>(vocab_size_) * sizeof(float));
    buf_argmax_parts_  = ctx_.alloc_private(4 * 128 * 8);
    buf_argmax_out_    = ctx_.alloc_shared(4 * sizeof(uint32_t));

    // Populate ModelInfo
    info_.arch = "inkling2";
    info_.model_dir = model_dir;
    info_.quant_desc = quant_bits.empty()
        ? "BF16 attn + NVFP4 MoE (Official)"
        : "Q8_AFFINE_G64 attn/head + NVFP4 MoE (MTPLX pack)";
    if (mtp_loaded_) info_.quant_desc += " + MTP(d=" + std::to_string(mtp_depth_) + ")";
    info_.vocab_size = unpadded_vocab_size_;
    info_.hidden_size = hidden_size_;
    info_.num_layers = num_layers_;
    info_.num_visits = static_cast<int>(visit_order_.size());
    info_.max_seq_len = max_seq_len_;
    info_.mapped_weight_bytes = store_.total_mapped_bytes() + mtp_store_.total_mapped_bytes();
    info_.kv_cache_bytes = kv_bytes;

    if (cfg.contains("eos_token_id")) {
        const auto& e = cfg["eos_token_id"];
        if (e.is_array()) {
            for (const auto& x : e.as_array()) info_.eos_token_ids.push_back(static_cast<int>(x.as_int()));
        } else if (e.is_int()) {
            info_.eos_token_ids.push_back(static_cast<int>(e.as_int()));
        }
    }
    if (info_.eos_token_ids.empty()) info_.eos_token_ids = {200006, 200010, 199999};

    reset();

    auto t1 = std::chrono::steady_clock::now();
    info_.load_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Inkling2Model::~Inkling2Model() = default;

void Inkling2Model::reset() {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (const auto& vs : visit_states_) {
            id<MTLBuffer> ck = vs.conv_k, cv = vs.conv_v, ca = vs.conv_attn, cm = vs.conv_mlp;
            [blit fillBuffer:ck range:NSMakeRange(0, ck.length) value:0];
            [blit fillBuffer:cv range:NSMakeRange(0, cv.length) value:0];
            [blit fillBuffer:ca range:NSMakeRange(0, ca.length) value:0];
            [blit fillBuffer:cm range:NSMakeRange(0, cm.length) value:0];
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    pos_ = 0;
    mtp_pos_ = 0;
}

void Inkling2Model::encode_linear(
    id enc_id, const LinearWeight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    QmmParams p{M, w.out_dim, w.in_dim, add_res ? 1u : 0u};

    if (M > 8u && (w.in_dim % 32u == 0u)) {
        if (w.kind == LinKind::BF16) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_tiled_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), ceil_div(M, 32u), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (w.kind == LinKind::Q8_G64) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q8_tiled_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 32u), ceil_div(M, 32u), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            // Q4_G64 upgraded to BM=64, BN=64, BK=32 (128 threads, 4x4 AMX + hw transpose)
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_q4_tiled_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(w.out_dim, 64u), ceil_div(M, 64u), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        return;
    }

    if (M > 1u && (w.in_dim % 32u == 0u)) {
        MTLSize tg = MTLSizeMake(128, 1, 1);
        MTLSize grid = MTLSizeMake(ceil_div(w.out_dim, 32u), ceil_div(M, 8u), 1);
        if (w.kind == LinKind::BF16) {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_b4_];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
        } else {
            id pso = (w.kind == LinKind::Q8_G64) ? pso_qmm_q8_b4_ : pso_qmm_q4_b4_;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
            [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
        }
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        return;
    }

    MTLSize tg = MTLSizeMake(32, 4, 1);
    MTLSize grid = MTLSizeMake(ceil_div(w.out_dim, 16u), M, 1);

    if (w.kind == LinKind::BF16) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
    } else {
        id pso = (w.kind == LinKind::Q8_G64) ? pso_qmm_q8_ : pso_qmm_q4_;
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
        [enc setBuffer:(id<MTLBuffer>)w.w.buffer offset:w.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)w.s.buffer offset:w.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)w.b.buffer offset:w.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
    }
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

void Inkling2Model::encode_bf16_mat(
    id enc_id, const TensorView& w, uint32_t out_dim, uint32_t in_dim,
    id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    QmmParams p{M, out_dim, in_dim, add_res ? 1u : 0u};
    if (M > 8u && (in_dim % 32u == 0u)) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_tiled_];
        [enc setBuffer:(id<MTLBuffer>)w.buffer offset:w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(out_dim, 32u), ceil_div(M, 32u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    if (M > 1u && (in_dim % 32u == 0u)) {
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_b4_];
        [enc setBuffer:(id<MTLBuffer>)w.buffer offset:w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(ceil_div(out_dim, 32u), ceil_div(M, 8u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qmm_bf16_];
    [enc setBuffer:(id<MTLBuffer>)w.buffer offset:w.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(out_dim, 16u), M, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
}

void Inkling2Model::encode_copy(id enc_id, id src, size_t src_off, id dst, size_t dst_off, uint32_t count) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_copy_];
    [enc setBuffer:(id<MTLBuffer>)src offset:src_off atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)dst offset:dst_off atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Inkling2Model::encode_rmsnorm(id enc_id, id x_buf, size_t x_off, const TensorView& w, id y_buf, size_t y_off, uint32_t M) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    NormParams np{M, static_cast<uint32_t>(hidden_size_), rms_eps_};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_rmsnorm_];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)w.buffer offset:w.offset atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)y_buf offset:y_off atIndex:2];
    [enc setBytes:&np length:sizeof(np) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Inkling2Model::encode_controller(
    id enc_id, const ControllerWeights& cw,
    id x_buf, size_t x_off, id msg_buf, size_t msg_off, id out_buf, size_t out_off, uint32_t M
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    CtrlParams cp{M, static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(gate_rank_), rms_eps_};

    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_ctrl_prep_];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)msg_buf offset:msg_off atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_ctrl_mixed_ offset:0 atIndex:2];
    [enc setBytes:&cp length:sizeof(cp) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    encode_bf16_mat(enc, cw.down, static_cast<uint32_t>(gate_rank_), static_cast<uint32_t>(2 * hidden_size_),
                    buf_ctrl_mixed_, 0, buf_ctrl_state_, 0, M);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_ctrl_apply_];
    [enc setBuffer:(id<MTLBuffer>)x_buf offset:x_off atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)msg_buf offset:msg_off atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_ctrl_state_ offset:0 atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)cw.forget.buffer offset:cw.forget.offset atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)cw.gain.buffer offset:cw.gain.offset atIndex:4];
    [enc setBuffer:(id<MTLBuffer>)out_buf offset:out_off atIndex:5];
    [enc setBytes:&cp length:sizeof(cp) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(hidden_size_, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void Inkling2Model::encode_chunk(
    id enc_id, int M_int, int start_pos, bool compute_logits, bool compute_argmax,
    bool verify_all_logits, bool save_conv_tape, bool capture_mtp
) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    uint32_t M = static_cast<uint32_t>(M_int);
    uint32_t D = static_cast<uint32_t>(hidden_size_);
    uint32_t MD = M * D;
    size_t MD_bytes = static_cast<size_t>(MD) * sizeof(float);
    uint32_t closed_stride = static_cast<uint32_t>(chunk_size_) * D;
    size_t closed_stride_bytes = static_cast<size_t>(closed_stride) * sizeof(float);
    float inv_sqrt_D = 1.0f / std::sqrt(static_cast<float>(D));
    float inv_sqrt_5 = 1.0f / std::sqrt(5.0f);
    float inv_sqrt_9 = 1.0f / 3.0f;

    // 1. Embedding + embed_norm
    if (embed_.kind == LinKind::BF16) {
        EmbedParams ep{M, D, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_embed_];
        [enc setBuffer:(id<MTLBuffer>)embed_.w.buffer offset:embed_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_norm_.buffer offset:embed_norm_.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:3];
        [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        EmbedParams ep{M, D, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_q4_];
        [enc setBuffer:(id<MTLBuffer>)embed_.w.buffer offset:embed_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_.s.buffer offset:embed_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_.b.buffer offset:embed_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_token_ids_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_q_in_ offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        encode_rmsnorm(enc, buf_q_in_, 0, embed_norm_, buf_x_, 0, M);
    }
    barrier();

    int n_closed = 0;
    int entry_n_closed = 0;
    int msg_bank_count = 0;

    // 2. Execute all 48 visits
    for (size_t visit = 0; visit < visit_order_.size(); ++visit) {
        int lidx = visit_order_[visit];
        const LayerWeights& lw = layers_[lidx];
        const VisitState& vs = visit_states_[visit];

        // Loop pass boundary management (copies are independent -> single barrier after)
        if (lidx == core_start_) {
            if (static_cast<int>(visit) == core_start_) {
                encode_copy(enc, buf_x_, 0, buf_entry_x_, 0, MD);
                entry_n_closed = n_closed;
                for (int b = 0; b < n_closed; ++b) {
                    encode_copy(enc, buf_closed_, b * closed_stride_bytes,
                                buf_entry_closed_, b * closed_stride_bytes, MD);
                }
                msg_bank_count = 0;
                barrier();
            } else {
                msg_bank_count = 0;
                for (int b = entry_n_closed + 1; b < n_closed; ++b) {
                    encode_copy(enc, buf_closed_, b * closed_stride_bytes,
                                buf_msg_bank_, msg_bank_count * closed_stride_bytes, MD);
                    ++msg_bank_count;
                }
                encode_copy(enc, buf_x_, 0, buf_msg_bank_, msg_bank_count * closed_stride_bytes, MD);
                ++msg_bank_count;
                barrier();

                encode_copy(enc, buf_entry_x_, 0, buf_x_, 0, MD);
                n_closed = entry_n_closed;
                for (int b = 0; b < entry_n_closed; ++b) {
                    encode_copy(enc, buf_entry_closed_, b * closed_stride_bytes,
                                buf_closed_, b * closed_stride_bytes, MD);
                }
                barrier();
            }
        }

        // Capture pre-visit hidden for DSpark MTP drafter (target_layer_ids = [5, 23, 41, 48])
        if (capture_mtp) {
            for (int c = 0; c < 3; ++c) {
                if (static_cast<int>(visit) == target_capture_ids_[c]) {
                    encode_copy(enc, buf_x_, 0, buf_mtp_cap_[c], 0, MD);
                }
            }
        }

        // Cross-pass message retrieval (can run concurrently with attn_res!)
        bool has_messages = (lidx >= core_start_ && lidx < core_end_ && msg_bank_count > 0);
        if (has_messages) {
            AttnResParams mp{M, D, static_cast<uint32_t>(msg_bank_count), 0u, 4u, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_];
            [enc setBuffer:(id<MTLBuffer>)buf_msg_bank_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)msg_queries_buf_[lidx - core_start_] offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_msg_out_ offset:0 atIndex:3];
            [enc setBytes:&mp length:sizeof(mp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Attention input via Block Attention Residual (fused norm when sliding & no messages!)
        bool shared_qkv_in = false;
        if (lidx == 0) {
            encode_rmsnorm(enc, buf_x_, 0, lw.attn_norm, buf_q_in_, 0, M);
            shared_qkv_in = true;
            barrier();
        } else if (lw.is_sliding && !has_messages) {
            AttnResParams ap{M, D, static_cast<uint32_t>(n_closed), 1u, 1u, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_norm1_];
            [enc setBuffer:(id<MTLBuffer>)buf_closed_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.attn_query.buffer offset:lw.attn_query.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.attn_norm.buffer offset:lw.attn_norm.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_q_in_ offset:0 atIndex:4];
            [enc setBytes:&ap length:sizeof(ap) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            shared_qkv_in = true;
            barrier();
        } else {
            uint32_t n_q = lw.is_sliding ? 1u : 3u;
            AttnResParams ap{M, D, static_cast<uint32_t>(n_closed), 1u, n_q, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_];
            [enc setBuffer:(id<MTLBuffer>)buf_closed_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.attn_query.buffer offset:lw.attn_query.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_read_out_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            size_t src_off[3] = {
                0,
                lw.is_sliding ? 0 : (1 * MD_bytes),
                lw.is_sliding ? 0 : (2 * MD_bytes)
            };
            id dst_bufs[3] = {buf_q_in_, buf_k_in_, buf_v_in_};
            const TensorView* norms[3] = {
                lw.is_sliding ? &lw.attn_norm : &lw.q_input_norm,
                lw.is_sliding ? &lw.attn_norm : &lw.k_input_norm,
                lw.is_sliding ? &lw.attn_norm : &lw.v_input_norm
            };
            if (has_messages) {
                // ALL 3 fused controller+RMSNorm kernels run CONCURRENTLY with ZERO intermediate buffers!
                CtrlParams cp{M, D, static_cast<uint32_t>(gate_rank_), rms_eps_};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_ctrl_fused_];
                for (int s = 0; s < 3; ++s) {
                    [enc setBuffer:(id<MTLBuffer>)buf_read_out_ offset:src_off[s] atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_msg_out_ offset:s * MD_bytes atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)lw.ctrl[s].down.buffer offset:lw.ctrl[s].down.offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)lw.ctrl[s].forget.buffer offset:lw.ctrl[s].forget.offset atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)lw.ctrl[s].gain.buffer offset:lw.ctrl[s].gain.offset atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)norms[s]->buffer offset:norms[s]->offset atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)dst_bufs[s] offset:0 atIndex:6];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                barrier();
            } else {
                // Global layer, pass 0: all 3 RMSNorms run CONCURRENTLY!
                for (int s = 0; s < 3; ++s) {
                    encode_rmsnorm(enc, buf_read_out_, src_off[s], *norms[s], dst_bufs[s], 0, M);
                }
                barrier();
            }
        }

        id k_in_buf = shared_qkv_in ? buf_q_in_ : buf_k_in_;
        id v_in_buf = shared_qkv_in ? buf_q_in_ : buf_v_in_;

        // ALL 6 attention projections (+ dynamic conv weight projs on global layers) run CONCURRENTLY!
        encode_linear(enc, lw.wq_du,     buf_q_in_, 0, buf_q_lin_, 0, M);
        encode_linear(enc, lw.wr_du,     buf_q_in_, 0, buf_r_lin_, 0, M);
        encode_linear(enc, lw.sink_proj, buf_q_in_, 0, buf_sink_,  0, M);
        encode_linear(enc, lw.gate_attn, buf_q_in_, 0, buf_gate_,  0, M);
        encode_linear(enc, lw.wk_dv,     k_in_buf,  0, buf_k_lin_, 0, M);
        encode_linear(enc, lw.wv_dv,     v_in_buf,  0, buf_v_lin_, 0, M);
        if (!lw.is_sliding) {
            encode_bf16_mat(enc, lw.k_sconv_dyn_proj, static_cast<uint32_t>(num_kv_heads_ * 4), D,
                            k_in_buf, 0, buf_dyn_w_k_, 0, M);
            encode_bf16_mat(enc, lw.v_sconv_dyn_proj, static_cast<uint32_t>(num_kv_heads_ * 4), D,
                            v_in_buf, 0, buf_dyn_w_v_, 0, M);
        }
        barrier();

        // K/V Short Convs + Q/Rel head prep all read independent inputs -> run CONCURRENTLY!
        uint32_t kv_dim = static_cast<uint32_t>(num_kv_heads_ * head_dim_);
        InkAttnParams ap{
            M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(d_rel_), static_cast<uint32_t>(rel_extent_),
            lw.is_sliding ? static_cast<uint32_t>(sliding_window_) : 0u,
            static_cast<uint32_t>(start_pos), static_cast<uint32_t>(start_pos + M_int),
            static_cast<uint32_t>(vs.kv_cap),
            qk_clip_, rel_clip_, log_alpha_, log_floor_, rms_eps_,
            1.0f / std::sqrt(static_cast<float>(head_dim_)), 1u
        };

        // Q + Rel prep runs in parallel with K + V short convs!
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_q_rel_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_lin_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)lw.q_scale.buffer offset:lw.q_scale.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_r_lin_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)lw.r_scale.buffer offset:lw.r_scale.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ready_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_r_ready_ offset:0 atIndex:5];
        [enc setBytes:&ap length:sizeof(ap) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        if (lw.is_sliding) {
            SconvParams sp{M, kv_dim, static_cast<uint32_t>(num_kv_heads_), 0u, inv_sqrt_5, save_conv_tape ? 1u : 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_sconv_st_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_lin_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.k_sconv_w.buffer offset:lw.k_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_k offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_k_conv_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_k : vs.conv_k) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:(id<MTLBuffer>)buf_v_lin_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.v_sconv_w.buffer offset:lw.v_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_v offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_v_conv_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_v : vs.conv_v) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            SconvParams sp{M, kv_dim, static_cast<uint32_t>(num_kv_heads_), 0u, inv_sqrt_9, save_conv_tape ? 1u : 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_sconv_dy_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_lin_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.k_sconv_w.buffer offset:lw.k_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_dyn_w_k_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_k offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_k_conv_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_k : vs.conv_k) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:(id<MTLBuffer>)buf_v_lin_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.v_sconv_w.buffer offset:lw.v_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_dyn_w_v_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_v offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_v_conv_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_v : vs.conv_v) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        barrier();

        // KV prep + Relative bias profiles (only up to min(rel_extent, kv_len)!) run CONCURRENTLY!
        {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_kv_prep_];
            [enc setBuffer:(id<MTLBuffer>)buf_k_conv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_v_conv_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)vs.k_cache offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)vs.v_cache offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            uint32_t active_dist = std::min<uint32_t>(rel_extent_, ap.kv_len);
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_rel_prof_];
            [enc setBuffer:(id<MTLBuffer>)buf_r_ready_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.rel_basis.buffer offset:lw.rel_basis.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_profiles_ offset:0 atIndex:2];
            [enc setBytes:&ap length:sizeof(ap) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(active_dist, num_heads_, M)
                threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(64u, active_dist), 4, 1)];
            barrier();

            // Flash GQA + Relative Bias + Sink + Output Gate
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_];
            [enc setBuffer:(id<MTLBuffer>)buf_q_ready_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_profiles_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_sink_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_gate_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)vs.k_cache offset:0 atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)vs.v_cache offset:0 atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:6];
            [enc setBytes:&ap length:sizeof(ap) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            barrier();
        }

        // Output projection wo_ud
        encode_linear(enc, lw.wo_ud, buf_attn_out_, 0, buf_proj_out_, 0, M);
        barrier();

        // Post-attention short conv + optional block boundary copy (independent -> concurrent!)
        {
            SconvParams sp{M, D, 1u, lw.is_boundary ? 0u : 1u, inv_sqrt_5, save_conv_tape ? 1u : 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_sconv_st_];
            [enc setBuffer:(id<MTLBuffer>)buf_proj_out_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.attn_sconv_w.buffer offset:lw.attn_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_attn offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_residual_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_attn : vs.conv_attn) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            if (lw.is_boundary) {
                encode_copy(enc, buf_x_, 0, buf_closed_, n_closed * closed_stride_bytes, MD);
                ++n_closed;
            }
            barrier();
        }

        // MLP input via Block Attention Residual (fused norm when no messages!)
        if (!has_messages) {
            AttnResParams mp{M, D, static_cast<uint32_t>(n_closed), 1u, 1u, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_norm1_];
            [enc setBuffer:(id<MTLBuffer>)buf_closed_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_residual_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.mlp_query.buffer offset:lw.mlp_query.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.mlp_norm.buffer offset:lw.mlp_norm.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_ffn_in_ offset:0 atIndex:4];
            [enc setBytes:&mp length:sizeof(mp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        } else {
            AttnResParams mp{M, D, static_cast<uint32_t>(n_closed), 1u, 1u, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_];
            [enc setBuffer:(id<MTLBuffer>)buf_closed_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_residual_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.mlp_query.buffer offset:lw.mlp_query.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_read_out_ offset:0 atIndex:3];
            [enc setBytes:&mp length:sizeof(mp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            CtrlParams cp{M, D, static_cast<uint32_t>(gate_rank_), rms_eps_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_ctrl_fused_];
            [enc setBuffer:(id<MTLBuffer>)buf_read_out_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_msg_out_ offset:3 * MD_bytes atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)lw.ctrl[3].down.buffer offset:lw.ctrl[3].down.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)lw.ctrl[3].forget.buffer offset:lw.ctrl[3].forget.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)lw.ctrl[3].gain.buffer offset:lw.ctrl[3].gain.offset atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)lw.mlp_norm.buffer offset:lw.mlp_norm.offset atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)buf_ffn_in_ offset:0 atIndex:6];
            [enc setBytes:&cp length:sizeof(cp) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        // Sparse MoE: Router matmul + NVFP4/MXFP8 activation quant run CONCURRENTLY!
        {
            encode_bf16_mat(enc, lw.router_w, static_cast<uint32_t>(num_experts_), D,
                            buf_ffn_in_, 0, buf_router_logits_, 0, M);

            id w13_x_buf = buf_ffn_in_;
            if (exact_nvfp4_act_) {
                ActQuantParams aq{M, D};
                if (lw.is_mxfp8) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_mxfp8_act_q_];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_in_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_q_ offset:0 atIndex:1];
                    [enc setBytes:&aq length:sizeof(aq) atIndex:2];
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_act_q_];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_in_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_q_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_outer_ offset:0 atIndex:2];
                    [enc setBytes:&aq length:sizeof(aq) atIndex:3];
                }
                [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                w13_x_buf = buf_ffn_q_;
            }
            barrier();

            RouterParams rp{M, D, static_cast<uint32_t>(num_experts_), static_cast<uint32_t>(top_k_), route_scale_};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_router_sel_];
            [enc setBuffer:(id<MTLBuffer>)buf_router_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.router_bias.buffer offset:lw.router_bias.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_top_w_ offset:0 atIndex:3];
            [enc setBytes:&rp length:sizeof(rp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            MoeNvfp4Params mp{M, D, static_cast<uint32_t>(moe_inter_), static_cast<uint32_t>(top_k_),
                              gate_softcap_, up_softcap_, exact_nvfp4_act_ ? 1u : 0u};
            bool use_grouped_moe = (M >= 4u);
            if (use_grouped_moe) {
                MoeGroupParams gp{M * static_cast<uint32_t>(top_k_), static_cast<uint32_t>(num_experts_)};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_sort_];
                [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_cnt_ offset:0 atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_off_ offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_sorted_ offset:0 atIndex:3];
                [enc setBytes:&gp length:sizeof(gp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                barrier();

                if (lw.is_mxfp8) {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_mxfp8_w13_grp_];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_w.buffer offset:lw.w13_w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_scale.buffer offset:lw.w13_scale.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)w13_x_buf offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_cnt_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_off_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_sorted_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:6];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(2u * moe_inter_, 32u), num_experts_, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();

                    id w2_mid_buf = buf_moe_mid_;
                    if (exact_nvfp4_act_) {
                        ActQuantParams aq{M * static_cast<uint32_t>(top_k_), static_cast<uint32_t>(moe_inter_)};
                        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_mxfp8_act_q_];
                        [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:0];
                        [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_q_ offset:0 atIndex:1];
                        [enc setBytes:&aq length:sizeof(aq) atIndex:2];
                        [enc dispatchThreadgroups:MTLSizeMake(M * top_k_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        barrier();
                        w2_mid_buf = buf_moe_mid_q_;
                    }

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_mxfp8_w2_grp_];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_w.buffer offset:lw.w2_w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_scale.buffer offset:lw.w2_scale.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)w2_mid_buf offset:0 atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)buf_top_w_ offset:0 atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_cnt_ offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_off_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_sorted_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_slot_out_ offset:0 atIndex:7];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(D, 32u), num_experts_, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                } else {
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_w13_grp_];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_w.buffer offset:lw.w13_w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_scale.buffer offset:lw.w13_scale.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_amax.buffer offset:lw.w13_amax.offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)lw.w13_denom.buffer offset:lw.w13_denom.offset atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)w13_x_buf offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_ffn_outer_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_cnt_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_off_ offset:0 atIndex:7];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_sorted_ offset:0 atIndex:8];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:9];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:10];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(2u * moe_inter_, 32u), num_experts_, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();

                    id w2_mid_buf = buf_moe_mid_;
                    if (exact_nvfp4_act_) {
                        ActQuantParams aq{M * static_cast<uint32_t>(top_k_), static_cast<uint32_t>(moe_inter_)};
                        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_act_q_];
                        [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:0];
                        [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_q_ offset:0 atIndex:1];
                        [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_outer_ offset:0 atIndex:2];
                        [enc setBytes:&aq length:sizeof(aq) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(M * top_k_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        barrier();
                        w2_mid_buf = buf_moe_mid_q_;
                    }

                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_w2_grp_];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_w.buffer offset:lw.w2_w.offset atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_scale.buffer offset:lw.w2_scale.offset atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_amax.buffer offset:lw.w2_amax.offset atIndex:2];
                    [enc setBuffer:(id<MTLBuffer>)lw.w2_denom.buffer offset:lw.w2_denom.offset atIndex:3];
                    [enc setBuffer:(id<MTLBuffer>)w2_mid_buf offset:0 atIndex:4];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_outer_ offset:0 atIndex:5];
                    [enc setBuffer:(id<MTLBuffer>)buf_top_w_ offset:0 atIndex:6];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_cnt_ offset:0 atIndex:7];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_exp_off_ offset:0 atIndex:8];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_sorted_ offset:0 atIndex:9];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_slot_out_ offset:0 atIndex:10];
                    [enc setBytes:&mp length:sizeof(mp) atIndex:11];
                    [enc dispatchThreadgroups:MTLSizeMake(ceil_div(D, 32u), num_experts_, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    barrier();
                }

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_reduce_];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_slot_out_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_out_ offset:0 atIndex:1];
                [enc setBytes:&mp length:sizeof(mp) atIndex:2];
                [enc dispatchThreads:MTLSizeMake(D, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            } else if (lw.is_mxfp8) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_mxfp8_w13_];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_w.buffer offset:lw.w13_w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_scale.buffer offset:lw.w13_scale.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)w13_x_buf offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:4];
                [enc setBytes:&mp length:sizeof(mp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(moe_inter_, 16u), M * top_k_, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                barrier();

                id w2_mid_buf = buf_moe_mid_;
                if (exact_nvfp4_act_) {
                    ActQuantParams aq{M * static_cast<uint32_t>(top_k_), static_cast<uint32_t>(moe_inter_)};
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_mxfp8_act_q_];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_q_ offset:0 atIndex:1];
                    [enc setBytes:&aq length:sizeof(aq) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(M * top_k_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();
                    w2_mid_buf = buf_moe_mid_q_;
                }

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_mxfp8_w2_];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_w.buffer offset:lw.w2_w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_scale.buffer offset:lw.w2_scale.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)w2_mid_buf offset:0 atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)buf_top_w_ offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_out_ offset:0 atIndex:5];
                [enc setBytes:&mp length:sizeof(mp) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(D, 16u), M, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                barrier();
            } else {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_w13_];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_w.buffer offset:lw.w13_w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_scale.buffer offset:lw.w13_scale.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_amax.buffer offset:lw.w13_amax.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)lw.w13_denom.buffer offset:lw.w13_denom.offset atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)w13_x_buf offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_ffn_outer_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:7];
                [enc setBytes:&mp length:sizeof(mp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(moe_inter_, 16u), M * top_k_, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                barrier();

                id w2_mid_buf = buf_moe_mid_;
                if (exact_nvfp4_act_) {
                    ActQuantParams aq{M * static_cast<uint32_t>(top_k_), static_cast<uint32_t>(moe_inter_)};
                    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_act_q_];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_ offset:0 atIndex:0];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_q_ offset:0 atIndex:1];
                    [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_outer_ offset:0 atIndex:2];
                    [enc setBytes:&aq length:sizeof(aq) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(M * top_k_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    barrier();
                    w2_mid_buf = buf_moe_mid_q_;
                }

                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_moe_w2_];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_w.buffer offset:lw.w2_w.offset atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_scale.buffer offset:lw.w2_scale.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_amax.buffer offset:lw.w2_amax.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)lw.w2_denom.buffer offset:lw.w2_denom.offset atIndex:3];
                [enc setBuffer:(id<MTLBuffer>)w2_mid_buf offset:0 atIndex:4];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_mid_outer_ offset:0 atIndex:5];
                [enc setBuffer:(id<MTLBuffer>)buf_top_idx_ offset:0 atIndex:6];
                [enc setBuffer:(id<MTLBuffer>)buf_top_w_ offset:0 atIndex:7];
                [enc setBuffer:(id<MTLBuffer>)buf_moe_out_ offset:0 atIndex:8];
                [enc setBytes:&mp length:sizeof(mp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(ceil_div(D, 16u), M, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
                barrier();
            }
        }

        // Post-MLP short conv + residual -> new x
        {
            SconvParams sp{M, D, 1u, 1u, inv_sqrt_5, save_conv_tape ? 1u : 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_sconv_st_];
            [enc setBuffer:(id<MTLBuffer>)buf_moe_out_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)lw.mlp_sconv_w.buffer offset:lw.mlp_sconv_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)vs.conv_mlp offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)buf_residual_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)(save_conv_tape ? vs.tape_mlp : vs.conv_mlp) offset:0 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
    }

    // 3. Final Block Attention Residual + Final Norm + optional capture[3] + Unembed
    if (compute_logits || capture_mtp) {
        {
            AttnResParams fp{M, D, static_cast<uint32_t>(n_closed), 1u, 1u, closed_stride, rms_eps_, inv_sqrt_D};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_attn_res_norm1_];
            [enc setBuffer:(id<MTLBuffer>)buf_closed_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_x_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)output_query_.buffer offset:output_query_.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)final_norm_.buffer offset:final_norm_.offset atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)buf_read_out_ offset:0 atIndex:4];
            [enc setBytes:&fp length:sizeof(fp) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }
        if (capture_mtp) {
            encode_copy(enc, buf_read_out_, 0, buf_mtp_cap_[3], 0, MD);
            barrier();
        }

        if (verify_all_logits) {
            LinearWeight unembed_unpadded = unembed_;
            unembed_unpadded.out_dim = static_cast<uint32_t>(unpadded_vocab_size_);
            encode_linear(enc, unembed_unpadded, buf_read_out_, 0, buf_mtp_verify_logits_, 0, M);
            barrier();

            uint32_t V = static_cast<uint32_t>(unpadded_vocab_size_);
            uint32_t n_tg = 128u;
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
            [enc setBuffer:(id<MTLBuffer>)buf_mtp_verify_logits_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
            [enc setBytes:&V length:sizeof(V) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(n_tg, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)buf_argmax_out_ offset:0 atIndex:1];
            [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (compute_logits) {
            size_t last_off = static_cast<size_t>(M - 1u) * D * sizeof(float);
            LinearWeight unembed_unpadded = unembed_;
            unembed_unpadded.out_dim = static_cast<uint32_t>(unpadded_vocab_size_);
            encode_linear(enc, unembed_unpadded, buf_read_out_, last_off, buf_logits_, 0, 1u);
            barrier();

            if (logit_softcap_ > 0.0f) {
                SoftcapParams sc{static_cast<uint32_t>(unpadded_vocab_size_), logit_softcap_};
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_softcap_];
                [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
                [enc setBytes:&sc length:sizeof(sc) atIndex:1];
                [enc dispatchThreads:MTLSizeMake(unpadded_vocab_size_, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }

            if (compute_argmax) {
                uint32_t V = static_cast<uint32_t>(unpadded_vocab_size_);
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
}

void Inkling2Model::encode_mtp_observe(id enc_id, int M_int, int start_pos) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    uint32_t M = static_cast<uint32_t>(M_int);
    uint32_t D = static_cast<uint32_t>(hidden_size_);
    uint32_t kv_dim = static_cast<uint32_t>(num_kv_heads_ * head_dim_);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    for (int b = 0; b < mtp_num_layers_; ++b) {
        const DraftBlockWeights& bw = mtp_blocks_[b];
        const DraftBlockState& bs = mtp_states_[b];

        MtpMixParams mp{M, D, 0u, rms_eps_, mtp_blend_[b][0], mtp_blend_[b][1], mtp_blend_[b][2], mtp_blend_[b][3]};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_mix_norm_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[0] offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[1] offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[2] offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[3] offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)bw.dflare_norm.buffer offset:bw.dflare_norm.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_mix_ offset:0 atIndex:5];
        [enc setBytes:&mp length:sizeof(mp) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_bf16_mat(enc, bw.k_proj_ctx, kv_dim, D, buf_mtp_mix_, 0, buf_k_lin_, 0, M);
        encode_bf16_mat(enc, bw.v_proj_ctx, kv_dim, D, buf_mtp_mix_, 0, buf_v_lin_, 0, M);
        barrier();

        MtpAttnParams ap{
            M, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(start_pos),
            static_cast<uint32_t>(start_pos + M_int), static_cast<uint32_t>(bs.kv_cap),
            bs.is_sliding ? static_cast<uint32_t>(sliding_window_) : 0u,
            rms_eps_, mtp_rope_theta_, attn_scale
        };
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_k_rope_];
        [enc setBuffer:(id<MTLBuffer>)buf_k_lin_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_v_lin_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)bw.k_norm_ctx.buffer offset:bw.k_norm_ctx.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)bs.k_cache offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)bs.v_cache offset:0 atIndex:4];
        [enc setBytes:&ap length:sizeof(ap) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(num_kv_heads_, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();
    }

    // Markov context uses only the last committed row (M - 1)
    MtpMixParams mmp{1u, D, M - 1u, rms_eps_, mtp_blend_[3][0], mtp_blend_[3][1], mtp_blend_[3][2], mtp_blend_[3][3]};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_mix_norm_];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[0] offset:0 atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[1] offset:0 atIndex:1];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[2] offset:0 atIndex:2];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_cap_[3] offset:0 atIndex:3];
    [enc setBuffer:(id<MTLBuffer>)mtp_markov_dflare_norm_.buffer offset:mtp_markov_dflare_norm_.offset atIndex:4];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_markov_ctx_ offset:0 atIndex:5];
    [enc setBytes:&mmp length:sizeof(mmp) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void Inkling2Model::encode_mtp_propose(id enc_id, int /*anchor_token*/, int count) {
    id<MTLComputeCommandEncoder> enc = enc_id;
    auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    uint32_t B = static_cast<uint32_t>(mtp_block_size_); // 8
    uint32_t D = static_cast<uint32_t>(hidden_size_);
    uint32_t q_dim = static_cast<uint32_t>(num_heads_ * head_dim_);
    uint32_t kv_dim = static_cast<uint32_t>(num_kv_heads_ * head_dim_);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // 1. Row 0 = embed_norm(embed(anchor)) from buf_mtp_tok_ring_[0], rows 1..7 = mask_embed
    if (embed_.kind == LinKind::BF16) {
        EmbedParams ep{1u, D, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_ink_embed_];
        [enc setBuffer:(id<MTLBuffer>)embed_.w.buffer offset:embed_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_norm_.buffer offset:embed_norm_.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_tok_ring_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_h_ offset:0 atIndex:3];
        [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        EmbedParams ep{1u, D, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_embed_q4_];
        [enc setBuffer:(id<MTLBuffer>)embed_.w.buffer offset:embed_.w.offset atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)embed_.s.buffer offset:embed_.s.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)embed_.b.buffer offset:embed_.b.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_tok_ring_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_q_in_ offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
        encode_rmsnorm(enc, buf_q_in_, 0, embed_norm_, buf_mtp_h_, 0, 1u);
    }

    [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_fill_masks_];
    [enc setBuffer:(id<MTLBuffer>)mtp_mask_embed_.buffer offset:mtp_mask_embed_.offset atIndex:0];
    [enc setBuffer:(id<MTLBuffer>)buf_mtp_h_ offset:0 atIndex:1];
    [enc setBytes:&D length:sizeof(D) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(D, B - 1u, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();

    // 2. 3 Bidirectional Block-8 DraftBlocks
    for (int b = 0; b < mtp_num_layers_; ++b) {
        const DraftBlockWeights& bw = mtp_blocks_[b];
        const DraftBlockState& bs = mtp_states_[b];

        // summed = (b==0) ? h : (h + res) -> buf_mtp_res_, normed -> buf_mtp_normed_
        MtpAddNormParams anp{B, D, (b > 0) ? 1u : 0u, 1u, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_add_norm_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_h_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_res_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)bw.input_layernorm.buffer offset:bw.input_layernorm.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_res_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_normed_ offset:0 atIndex:4];
        [enc setBytes:&anp length:sizeof(anp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // Concurrent Q, K, V projections (M=8 -> fast tiled GEMM!)
        encode_bf16_mat(enc, bw.q_proj, q_dim, D, buf_mtp_normed_, 0, buf_q_lin_, 0, B);
        encode_bf16_mat(enc, bw.k_proj, kv_dim, D, buf_mtp_normed_, 0, buf_k_lin_, 0, B);
        encode_bf16_mat(enc, bw.v_proj, kv_dim, D, buf_mtp_normed_, 0, buf_v_lin_, 0, B);
        barrier();

        MtpAttnParams ap{
            B, static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(num_kv_heads_),
            static_cast<uint32_t>(head_dim_), static_cast<uint32_t>(mtp_pos_),
            static_cast<uint32_t>(mtp_pos_), static_cast<uint32_t>(bs.kv_cap),
            bs.is_sliding ? static_cast<uint32_t>(sliding_window_) : 0u,
            rms_eps_, mtp_rope_theta_, attn_scale
        };
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_qkv_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_lin_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_k_lin_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_v_lin_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)bw.q_norm.buffer offset:bw.q_norm.offset atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)bw.k_norm.buffer offset:bw.k_norm.offset atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ready_ offset:0 atIndex:5];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_k_self_ offset:0 atIndex:6];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_v_self_ offset:0 atIndex:7];
        [enc setBytes:&ap length:sizeof(ap) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_ + num_kv_heads_, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_attn_];
        [enc setBuffer:(id<MTLBuffer>)buf_q_ready_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)bs.k_cache offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)bs.v_cache offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_k_self_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_v_self_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_attn_out_ offset:0 atIndex:5];
        [enc setBytes:&ap length:sizeof(ap) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads_, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        barrier();

        // summed += o_proj(attn)
        encode_bf16_mat(enc, bw.o_proj, D, q_dim, buf_attn_out_, 0, buf_mtp_res_, 0, B, true);
        barrier();

        // SwiGLU MLP -> buf_mtp_h_ (residual stays in buf_mtp_res_)
        encode_rmsnorm(enc, buf_mtp_res_, 0, bw.post_attn_layernorm, buf_mtp_normed_, 0, B);
        barrier();

        encode_bf16_mat(enc, bw.gate_proj, static_cast<uint32_t>(mtp_inter_), D, buf_mtp_normed_, 0, buf_mtp_mlp_g_, 0, B);
        encode_bf16_mat(enc, bw.up_proj,   static_cast<uint32_t>(mtp_inter_), D, buf_mtp_normed_, 0, buf_mtp_mlp_u_, 0, B);
        barrier();

        uint32_t n_mlp = B * static_cast<uint32_t>(mtp_inter_);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_swiglu_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_mlp_g_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_mlp_u_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_mlp_g_ offset:0 atIndex:2];
        [enc setBytes:&n_mlp length:sizeof(n_mlp) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n_mlp, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_bf16_mat(enc, bw.down_proj, D, static_cast<uint32_t>(mtp_inter_), buf_mtp_mlp_g_, 0, buf_mtp_h_, 0, B, false);
        barrier();
    }

    // 3. Final output norm: rms_norm(h + res, norm.weight) -> buf_mtp_normed_[0..7, D]
    {
        MtpAddNormParams fnp{B, D, 1u, 0u, rms_eps_};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_add_norm_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_h_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_res_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)mtp_output_norm_.buffer offset:mtp_output_norm_.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_res_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_normed_ offset:0 atIndex:4];
        [enc setBytes:&fnp length:sizeof(fnp) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();
    }

    // 4. Markov RNN head: produce `count` draft tokens into buf_mtp_tok_ring_[1..count] with ZERO CPU sync!
    uint32_t rank = static_cast<uint32_t>(mtp_markov_rank_);
    uint32_t rnn_in_dim = 2u * rank + 2u * D;
    uint32_t V = static_cast<uint32_t>(unpadded_vocab_size_);
    uint32_t n_tg = 128u;
    LinearWeight unembed_unpadded = unembed_;
    unembed_unpadded.out_dim = V;

    for (int s = 0; s < count; ++s) {
        uint32_t step_u = static_cast<uint32_t>(s);
        MarkovPrepParams mpp{rank, D, step_u, step_u};
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_markov_prep_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_rnn_state_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)mtp_markov_w1_.buffer offset:mtp_markov_w1_.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_tok_ring_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_normed_ offset:0 atIndex:3];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_markov_ctx_ offset:0 atIndex:4];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_rnn_in_ offset:0 atIndex:5];
        [enc setBytes:&mpp length:sizeof(mpp) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(rnn_in_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_bf16_mat(enc, mtp_markov_rnn_w_, 3u * rank, rnn_in_dim, buf_mtp_rnn_in_, 0, buf_mtp_rnn_out_, 0, 1u);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_markov_gate_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_rnn_out_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)mtp_markov_rnn_b_.buffer offset:mtp_markov_rnn_b_.offset atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_rnn_state_ offset:0 atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_signal_ offset:0 atIndex:3];
        [enc setBytes:&rank length:sizeof(rank) atIndex:4];
        [enc setBytes:&step_u length:sizeof(step_u) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(rank, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        // Algebraic fold-free identity: unembed @ (h[s+1] + hidden_out.weight @ signal + hidden_out.bias)
        encode_bf16_mat(enc, mtp_markov_hid_w_, D, rank, buf_mtp_signal_, 0, buf_proj_out_, 0, 1u);
        barrier();

        size_t h_step_off = static_cast<size_t>(s + 1) * D * sizeof(float);
        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_mtp_add_h_bias_];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_normed_ offset:h_step_off atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_proj_out_ offset:0 atIndex:1];
        [enc setBuffer:(id<MTLBuffer>)mtp_markov_hid_b_.buffer offset:mtp_markov_hid_b_.offset atIndex:2];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_eff_h_ offset:0 atIndex:3];
        [enc setBytes:&D length:sizeof(D) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        encode_linear(enc, unembed_unpadded, buf_mtp_eff_h_, 0, buf_logits_, 0, 1u);
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax1_];
        [enc setBuffer:(id<MTLBuffer>)buf_logits_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:1];
        [enc setBytes:&V length:sizeof(V) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        barrier();

        [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_argmax2_];
        [enc setBuffer:(id<MTLBuffer>)buf_argmax_parts_ offset:0 atIndex:0];
        [enc setBuffer:(id<MTLBuffer>)buf_mtp_tok_ring_ offset:static_cast<size_t>(s + 1) * sizeof(uint32_t) atIndex:1];
        [enc setBytes:&n_tg length:sizeof(n_tg) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        barrier();
    }
}

void Inkling2Model::rollback_conv_tapes(int kept_minus_one) {
    size_t kv_bytes = 3ull * static_cast<size_t>(num_kv_heads_ * head_dim_) * sizeof(float);
    size_t d_bytes  = 3ull * static_cast<size_t>(hidden_size_) * sizeof(float);
    size_t kv_off = static_cast<size_t>(kept_minus_one) * kv_bytes;
    size_t d_off  = static_cast<size_t>(kept_minus_one) * d_bytes;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (const auto& vs : visit_states_) {
            [blit copyFromBuffer:(id<MTLBuffer>)vs.tape_k    sourceOffset:kv_off toBuffer:(id<MTLBuffer>)vs.conv_k    destinationOffset:0 size:kv_bytes];
            [blit copyFromBuffer:(id<MTLBuffer>)vs.tape_v    sourceOffset:kv_off toBuffer:(id<MTLBuffer>)vs.conv_v    destinationOffset:0 size:kv_bytes];
            [blit copyFromBuffer:(id<MTLBuffer>)vs.tape_attn sourceOffset:d_off  toBuffer:(id<MTLBuffer>)vs.conv_attn destinationOffset:0 size:d_bytes];
            [blit copyFromBuffer:(id<MTLBuffer>)vs.tape_mlp  sourceOffset:d_off  toBuffer:(id<MTLBuffer>)vs.conv_mlp  destinationOffset:0 size:d_bytes];
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

const float* Inkling2Model::forward(const int* tokens, int n_tokens, int* out_argmax, StepStats* stats) {
    if (n_tokens <= 0) return reinterpret_cast<const float*>([(id<MTLBuffer>)buf_logits_ contents]);
    if (pos_ + n_tokens > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds max_seq_len (" + std::to_string(max_seq_len_) + ")");
    }

    auto t0 = std::chrono::steady_clock::now();
    int offset = 0;
    while (offset < n_tokens) {
        int m = std::min(chunk_size_, n_tokens - offset);
        bool is_last = (offset + m == n_tokens);
        std::memcpy([(id<MTLBuffer>)buf_token_ids_ contents], tokens + offset, static_cast<size_t>(m) * sizeof(int));

        @autoreleasepool {
            id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            encode_chunk(enc, m, pos_, is_last, is_last && (out_argmax != nullptr), false, false, mtp_loaded_);
            if (mtp_loaded_) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_mtp_observe(enc, m, pos_);
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                throw std::runtime_error(std::string("Metal command buffer error: ") +
                                         [[cb.error localizedDescription] UTF8String]);
            }
        }
        pos_ += m;
        if (mtp_loaded_) mtp_pos_ += m;
        offset += m;
    }

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

int Inkling2Model::forward_greedy(int token, StepStats* stats) {
    int next_tok = 0;
    forward(&token, 1, &next_tok, stats);
    return next_tok;
}

void Inkling2Model::forward_mtp_greedy(int anchor_token, std::vector<int>& out_tokens, StepStats* stats) {
    if (!mtp_loaded_ || pos_ + mtp_depth_ + 1 > max_seq_len_) {
        int nxt = forward_greedy(anchor_token, stats);
        out_tokens.push_back(nxt);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    int K = mtp_depth_; // 1..3 draft tokens; verify M = K + 1 <= 4
    int M = K + 1;
    int pre_pos = pos_;

    // Seed anchor_token into tok_ring[0]
    uint32_t* ring = reinterpret_cast<uint32_t*>([(id<MTLBuffer>)buf_mtp_tok_ring_ contents]);
    ring[0] = static_cast<uint32_t>(anchor_token);

    // Execute ENTIRE DSpark propose (3 blocks + K Markov steps) AND target M-token verify in ONE command buffer!
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        encode_mtp_propose(enc, anchor_token, K);
        [enc endEncoding];

        // Exact integer blit copy (never float-copy token IDs under fast-math!)
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:(id<MTLBuffer>)buf_mtp_tok_ring_ sourceOffset:0
                    toBuffer:(id<MTLBuffer>)buf_token_ids_ destinationOffset:0
                        size:static_cast<size_t>(M) * sizeof(uint32_t)];
        [blit endEncoding];

        id<MTLComputeCommandEncoder> enc2 = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        encode_chunk(enc2, M, pre_pos, true, true, true, true, true);
        [enc2 endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal Inkling MTP error: ") +
                                     [[cb.error localizedDescription] UTF8String]);
        }
    }

    // Exact greedy verification: compare target argmax predictions against drafts
    const uint32_t* preds = reinterpret_cast<const uint32_t*>([(id<MTLBuffer>)buf_argmax_out_ contents]);
    int accepted = 0;
    for (int d = 0; d < K; ++d) {
        int draft_tok = static_cast<int>(ring[d + 1]);
        if (static_cast<int>(preds[d]) == draft_tok) {
            out_tokens.push_back(draft_tok);
            ++accepted;
        } else {
            break;
        }
    }
    int bonus_tok = static_cast<int>(preds[accepted]);
    out_tokens.push_back(bonus_tok);

    int kept = accepted + 1;
    // Sliding KV has kv_cap = sliding_window + 16 so rejected tail NEVER overwrites valid window!
    // Only conv states need a tiny blit rollback if kept < M, followed by drafter observe(kept).
    if (kept < M) {
        rollback_conv_tapes(kept - 1);
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        encode_mtp_observe(enc, kept, pre_pos);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    pos_ = pre_pos + kept;
    mtp_pos_ = pre_pos + kept;

    auto t1 = std::chrono::steady_clock::now();
    if (stats) {
        stats->gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->tokens_processed = kept;
        stats->mtp_accepted = accepted;
        stats->mtp_drafted = K;
    }
}

bool Inkling2Model::save_state_to_file(const std::string& path, const std::vector<int>& tokens) {
    if (pos_ <= 0 || static_cast<int>(tokens.size()) != pos_) return false;

    size_t kv_dim = static_cast<size_t>(num_kv_heads_) * head_dim_;
    size_t conv_kv_bytes = 3ull * kv_dim * sizeof(float);
    size_t conv_d_bytes  = 3ull * hidden_size_ * sizeof(float);

    size_t total_state = 0;
    for (const auto& vs : visit_states_) {
        size_t kv_rows = (vs.is_sliding && pos_ >= vs.kv_cap) ? static_cast<size_t>(vs.kv_cap) : static_cast<size_t>(pos_);
        total_state += 2 * kv_rows * kv_dim * sizeof(uint16_t) + 2 * conv_kv_bytes + 2 * conv_d_bytes;
    }
    if (mtp_loaded_) {
        for (const auto& bs : mtp_states_) {
            size_t kv_rows = (bs.is_sliding && mtp_pos_ >= bs.kv_cap) ? static_cast<size_t>(bs.kv_cap) : static_cast<size_t>(mtp_pos_);
            total_state += 2 * kv_rows * kv_dim * sizeof(uint16_t);
        }
        total_state += static_cast<size_t>(hidden_size_) * sizeof(float); // buf_mtp_markov_ctx_
    }

    @autoreleasepool {
        id<MTLBuffer> stage = ctx_.alloc_shared(total_state);
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t off = 0;
        for (const auto& vs : visit_states_) {
            size_t kv_rows = (vs.is_sliding && pos_ >= vs.kv_cap) ? static_cast<size_t>(vs.kv_cap) : static_cast<size_t>(pos_);
            size_t kv_sz = kv_rows * kv_dim * sizeof(uint16_t);
            [blit copyFromBuffer:(id<MTLBuffer>)vs.k_cache   sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_sz]; off += kv_sz;
            [blit copyFromBuffer:(id<MTLBuffer>)vs.v_cache   sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_sz]; off += kv_sz;
            [blit copyFromBuffer:(id<MTLBuffer>)vs.conv_k    sourceOffset:0 toBuffer:stage destinationOffset:off size:conv_kv_bytes]; off += conv_kv_bytes;
            [blit copyFromBuffer:(id<MTLBuffer>)vs.conv_v    sourceOffset:0 toBuffer:stage destinationOffset:off size:conv_kv_bytes]; off += conv_kv_bytes;
            [blit copyFromBuffer:(id<MTLBuffer>)vs.conv_attn sourceOffset:0 toBuffer:stage destinationOffset:off size:conv_d_bytes];  off += conv_d_bytes;
            [blit copyFromBuffer:(id<MTLBuffer>)vs.conv_mlp  sourceOffset:0 toBuffer:stage destinationOffset:off size:conv_d_bytes];  off += conv_d_bytes;
        }
        if (mtp_loaded_) {
            for (const auto& bs : mtp_states_) {
                size_t kv_rows = (bs.is_sliding && mtp_pos_ >= bs.kv_cap) ? static_cast<size_t>(bs.kv_cap) : static_cast<size_t>(mtp_pos_);
                size_t kv_sz = kv_rows * kv_dim * sizeof(uint16_t);
                [blit copyFromBuffer:(id<MTLBuffer>)bs.k_cache sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_sz]; off += kv_sz;
                [blit copyFromBuffer:(id<MTLBuffer>)bs.v_cache sourceOffset:0 toBuffer:stage destinationOffset:off size:kv_sz]; off += kv_sz;
            }
            size_t mctx_sz = static_cast<size_t>(hidden_size_) * sizeof(float);
            [blit copyFromBuffer:(id<MTLBuffer>)buf_mtp_markov_ctx_ sourceOffset:0 toBuffer:stage destinationOffset:off size:mctx_sz]; off += mctx_sz;
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;
        PrefixCacheHeader hdr{};
        std::memcpy(hdr.magic, "MICACHE1", 8);
        hdr.model_hash = compute_model_hash(info_);
        hdr.pos = static_cast<uint32_t>(pos_);
        hdr.reserved = mtp_loaded_ ? (0x80000000u | static_cast<uint32_t>(mtp_pos_)) : 0u;
        hdr.state_bytes = total_state;
        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        ofs.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int));
        ofs.write(reinterpret_cast<const char*>([stage contents]), total_state);
        return ofs.good();
    }
}

bool Inkling2Model::load_state_from_file(const std::string& path, std::vector<int>& out_tokens) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    PrefixCacheHeader hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!ifs || std::memcmp(hdr.magic, "MICACHE1", 8) != 0) return false;
    if (hdr.model_hash != compute_model_hash(info_)) return false;
    if (static_cast<int>(hdr.pos) > max_seq_len_) return false;

    out_tokens.resize(hdr.pos);
    ifs.read(reinterpret_cast<char*>(out_tokens.data()), hdr.pos * sizeof(int));
    if (!ifs) return false;

    size_t kv_dim = static_cast<size_t>(num_kv_heads_) * head_dim_;
    size_t conv_kv_bytes = 3ull * kv_dim * sizeof(float);
    size_t conv_d_bytes  = 3ull * hidden_size_ * sizeof(float);
    bool has_mtp_payload = (hdr.reserved & 0x80000000u) != 0u;
    int saved_mtp_pos = static_cast<int>(hdr.reserved & 0x7FFFFFFFu);

    @autoreleasepool {
        id<MTLBuffer> stage = ctx_.alloc_shared(hdr.state_bytes);
        ifs.read(reinterpret_cast<char*>([stage contents]), hdr.state_bytes);
        if (!ifs) return false;

        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        size_t off = 0;
        for (const auto& vs : visit_states_) {
            size_t kv_rows = (vs.is_sliding && static_cast<int>(hdr.pos) >= vs.kv_cap)
                ? static_cast<size_t>(vs.kv_cap) : static_cast<size_t>(hdr.pos);
            size_t kv_sz = kv_rows * kv_dim * sizeof(uint16_t);
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.k_cache   destinationOffset:0 size:kv_sz]; off += kv_sz;
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.v_cache   destinationOffset:0 size:kv_sz]; off += kv_sz;
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.conv_k    destinationOffset:0 size:conv_kv_bytes]; off += conv_kv_bytes;
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.conv_v    destinationOffset:0 size:conv_kv_bytes]; off += conv_kv_bytes;
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.conv_attn destinationOffset:0 size:conv_d_bytes];  off += conv_d_bytes;
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)vs.conv_mlp  destinationOffset:0 size:conv_d_bytes];  off += conv_d_bytes;
        }
        if (has_mtp_payload && mtp_loaded_) {
            for (const auto& bs : mtp_states_) {
                size_t kv_rows = (bs.is_sliding && saved_mtp_pos >= bs.kv_cap)
                    ? static_cast<size_t>(bs.kv_cap) : static_cast<size_t>(saved_mtp_pos);
                size_t kv_sz = kv_rows * kv_dim * sizeof(uint16_t);
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)bs.k_cache destinationOffset:0 size:kv_sz]; off += kv_sz;
                [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)bs.v_cache destinationOffset:0 size:kv_sz]; off += kv_sz;
            }
            size_t mctx_sz = static_cast<size_t>(hidden_size_) * sizeof(float);
            [blit copyFromBuffer:stage sourceOffset:off toBuffer:(id<MTLBuffer>)buf_mtp_markov_ctx_ destinationOffset:0 size:mctx_sz]; off += mctx_sz;
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    pos_ = static_cast<int>(hdr.pos);
    mtp_pos_ = (has_mtp_payload && mtp_loaded_) ? saved_mtp_pos : 0;
    return true;
}

} // namespace mi
