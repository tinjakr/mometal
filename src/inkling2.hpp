#pragma once

#include "metal_ctx.hpp"
#include "model.hpp"
#include "safetensors.hpp"

#include <string>
#include <vector>

namespace mi {

class Inkling2Model : public Model {
public:
    Inkling2Model(MetalContext& ctx, const std::string& model_dir, int max_seq_len = 8192,
                  bool exact_nvfp4_act = true, const std::string& mtp_dir = "", int mtp_depth = 3);
    ~Inkling2Model() override;

    const ModelInfo& info() const override { return info_; }
    void reset() override;
    int position() const override { return pos_; }

    const float* forward(const int* tokens, int n_tokens, int* out_argmax = nullptr, StepStats* stats = nullptr) override;
    int forward_greedy(int token, StepStats* stats = nullptr) override;

    bool has_mtp() const override { return mtp_loaded_; }
    int mtp_depth() const override { return mtp_loaded_ ? mtp_depth_ : 0; }
    void forward_mtp_greedy(int anchor_token, std::vector<int>& out_tokens, StepStats* stats = nullptr) override;

    bool save_state_to_file(const std::string& path, const std::vector<int>& tokens) override;
    bool load_state_from_file(const std::string& path, std::vector<int>& out_tokens) override;

private:
    enum class LinKind { BF16, Q8_G64, Q4_G64 };

    struct LinearWeight {
        LinKind kind = LinKind::BF16;
        TensorView w;
        TensorView s;
        TensorView b;
        uint32_t out_dim = 0;
        uint32_t in_dim = 0;
    };

    struct ControllerWeights {
        TensorView down;   // bf16 [64, 4608]
        TensorView forget; // bf16 [2304, 64]
        TensorView gain;   // bf16 [2304, 64]
    };

    struct LayerWeights {
        bool is_sliding = true;
        bool is_boundary = false;
        bool has_controllers = false;

        TensorView attn_query;  // f32 [D] or [3, D] (nil for layer 0)
        TensorView mlp_query;   // f32 [D]

        // Input norms
        TensorView attn_norm;    // sliding only
        TensorView q_input_norm; // global only
        TensorView k_input_norm;
        TensorView v_input_norm;
        TensorView mlp_norm;

        // Attention projections
        LinearWeight wq_du;
        LinearWeight wk_dv;
        LinearWeight wv_dv;
        LinearWeight wr_du;
        LinearWeight sink_proj;
        LinearWeight gate_attn;
        LinearWeight wo_ud;

        TensorView q_scale;   // bf16 [Hq, D]
        TensorView r_scale;   // bf16 [Hq, d_rel]
        TensorView rel_basis; // bf16 [d_rel, rel_extent]

        // Short convolutions
        TensorView k_sconv_w;          // static: [1024, 1, 4], dynamic: static_w [4, 1024]
        TensorView k_sconv_dyn_proj;   // dynamic only: [32, 2304]
        TensorView v_sconv_w;
        TensorView v_sconv_dyn_proj;
        TensorView attn_sconv_w;       // [2304, 1, 4]
        TensorView mlp_sconv_w;        // [2304, 1, 4]

        // MoE router + NVFP4 / MXFP8 experts
        bool is_mxfp8 = false;
        TensorView router_w;    // bf16 [128, 2304]
        TensorView router_bias; // f32 [128]
        TensorView w13_w;       // nvfp4: u8 [128, 2304, 1152], mxfp8: f8_e4m3 [128, 2304, 2304]
        TensorView w13_scale;   // nvfp4: f8_e4m3 [128, 2304, 144], mxfp8: u8 [128, 2304, 72]
        TensorView w13_amax;    // nvfp4 only: f32 [128]
        TensorView w13_denom;   // nvfp4 only: f32 [1]
        TensorView w2_w;
        TensorView w2_scale;
        TensorView w2_amax;
        TensorView w2_denom;

        // Loop controllers (layers core_start..core_end-1)
        ControllerWeights ctrl[4]; // 0=attn(q), 1=k, 2=v, 3=mlp
    };

    struct VisitState {
        bool is_sliding = true;
        int kv_cap = 1040;
        id k_cache = nil;   // half [kv_cap, Hkv, head_dim]
        id v_cache = nil;   // half [kv_cap, Hkv, head_dim]
        id conv_k = nil;    // f32 [3, Hkv*head_dim]
        id conv_v = nil;    // f32 [3, Hkv*head_dim]
        id conv_attn = nil; // f32 [3, D]
        id conv_mlp = nil;  // f32 [3, D]
        // Post-step conv state tapes [4, 3, C] for instant zero-replay MTP rollback
        id tape_k = nil;
        id tape_v = nil;
        id tape_attn = nil;
        id tape_mlp = nil;
    };

    struct DraftBlockWeights {
        bool is_sliding = true;
        TensorView dflare_norm;
        TensorView input_layernorm;
        TensorView post_attn_layernorm;
        TensorView q_proj, k_proj, v_proj, o_proj;
        TensorView k_proj_ctx, v_proj_ctx;
        TensorView q_norm, k_norm, k_norm_ctx;
        TensorView gate_proj, up_proj, down_proj;
    };

    struct DraftBlockState {
        bool is_sliding = true;
        int kv_cap = 1024;
        id k_cache = nil; // half [kv_cap, Hkv, D]
        id v_cache = nil; // half [kv_cap, Hkv, D]
    };

    MetalContext& ctx_;
    SafetensorsStore store_;
    SafetensorsStore mtp_store_;
    ModelInfo info_;

    // Hyperparameters
    int hidden_size_ = 2304;
    int num_layers_ = 24;
    int core_start_ = 6;
    int core_end_ = 18;
    int num_passes_ = 3;
    int gate_rank_ = 64;
    int vocab_size_ = 201024;
    int unpadded_vocab_size_ = 200058;
    int num_heads_ = 24;
    int num_kv_heads_ = 8;
    int head_dim_ = 128;
    int d_rel_ = 16;
    int rel_extent_ = 1024;
    int sliding_window_ = 1024;
    int attn_res_block_ = 6;
    int num_experts_ = 128;
    int top_k_ = 8;
    int moe_inter_ = 1152;
    float route_scale_ = 8.0f;
    float gate_softcap_ = 4.0f;
    float up_softcap_ = 25.0f;
    float logit_softcap_ = 30.0f;
    float qk_clip_ = 22.627416997969522f;
    float rel_clip_ = 8.0f;
    float log_alpha_ = 0.1f;
    float log_floor_ = 8192.0f;
    float rms_eps_ = 1e-6f;
    int max_seq_len_ = 8192;
    int chunk_size_ = 128;
    int pos_ = 0;
    bool exact_nvfp4_act_ = true;

    std::vector<int> visit_order_;
    LinearWeight embed_;
    TensorView embed_norm_;
    TensorView final_norm_;
    TensorView output_query_;
    LinearWeight unembed_;
    std::vector<id> msg_queries_buf_; // 12 shared f32[4, D] buffers assembled at load time
    std::vector<LayerWeights> layers_;
    std::vector<VisitState> visit_states_;

    // Scratch buffers
    id buf_token_ids_ = nil;     // shared int[chunk_size]
    id buf_x_ = nil;             // private f32[chunk_size, D]
    id buf_entry_x_ = nil;       // private f32[chunk_size, D]
    id buf_closed_ = nil;        // private f32[6, chunk_size, D]
    id buf_entry_closed_ = nil;  // private f32[4, chunk_size, D]
    id buf_msg_bank_ = nil;      // private f32[4, chunk_size, D]
    id buf_msg_out_ = nil;       // private f32[4, chunk_size, D]
    id buf_read_out_ = nil;      // private f32[3, chunk_size, D]
    id buf_ctrl_mixed_ = nil;    // private f32[chunk_size, 2*D]
    id buf_ctrl_state_ = nil;    // private f32[chunk_size, 64]
    id buf_q_in_ = nil;          // private f32[chunk_size, D]
    id buf_k_in_ = nil;          // private f32[chunk_size, D]
    id buf_v_in_ = nil;          // private f32[chunk_size, D]
    id buf_q_lin_ = nil;         // private f32[chunk_size, Hq*head_dim]
    id buf_k_lin_ = nil;         // private f32[chunk_size, Hkv*head_dim]
    id buf_v_lin_ = nil;         // private f32[chunk_size, Hkv*head_dim]
    id buf_k_conv_ = nil;        // private f32[chunk_size, Hkv*head_dim]
    id buf_v_conv_ = nil;        // private f32[chunk_size, Hkv*head_dim]
    id buf_dyn_w_k_ = nil;       // private f32[chunk_size, Hkv*4]
    id buf_dyn_w_v_ = nil;       // private f32[chunk_size, Hkv*4]
    id buf_router_logits_ = nil; // private f32[chunk_size, 128]
    id buf_r_lin_ = nil;         // private f32[chunk_size, Hq*d_rel]
    id buf_sink_ = nil;          // private f32[chunk_size, Hq]
    id buf_gate_ = nil;          // private f32[chunk_size, Hq*head_dim]
    id buf_q_ready_ = nil;       // private f32[chunk_size, Hq*head_dim]
    id buf_r_ready_ = nil;       // private f32[chunk_size, Hq*d_rel]
    id buf_profiles_ = nil;      // private f32[chunk_size, Hq*rel_extent]
    id buf_attn_out_ = nil;      // private f32[chunk_size, Hq*head_dim]
    id buf_proj_out_ = nil;      // private f32[chunk_size, D]
    id buf_residual_ = nil;      // private f32[chunk_size, D]
    id buf_ffn_in_ = nil;        // private f32[chunk_size, D]
    id buf_ffn_q_ = nil;         // private f32[chunk_size, D]
    id buf_ffn_outer_ = nil;     // private f32[chunk_size]
    id buf_top_idx_ = nil;       // private uint32[chunk_size, 8]
    id buf_top_w_ = nil;         // private f32[chunk_size, 8]
    id buf_moe_mid_ = nil;       // private f32[chunk_size, 8, moe_inter]
    id buf_moe_mid_q_ = nil;     // private f32[chunk_size, 8, moe_inter]
    id buf_moe_mid_outer_ = nil; // private f32[chunk_size, 8]
    id buf_moe_out_ = nil;       // private f32[chunk_size, D]
    id buf_moe_exp_cnt_ = nil;   // private uint32[128]
    id buf_moe_exp_off_ = nil;   // private uint32[128]
    id buf_moe_sorted_ = nil;    // private uint32[chunk_size * 8]
    id buf_moe_slot_out_ = nil;  // private f32[chunk_size * 8, D]
    id buf_logits_ = nil;        // shared f32[vocab_size]
    id buf_argmax_parts_ = nil;  // private ArgmaxOut[4 * 128]
    id buf_argmax_out_ = nil;    // shared uint32[4]

    // Inkling DSpark MTP drafter state
    bool mtp_loaded_ = false;
    int mtp_depth_ = 3;          // 1..3 draft tokens (verify M = depth + 1 <= 4)
    int mtp_num_layers_ = 3;
    int mtp_inter_ = 9216;
    int mtp_block_size_ = 8;
    int mtp_markov_rank_ = 256;
    float mtp_rope_theta_ = 1000000.0f;
    int mtp_pos_ = 0;            // committed tokens observed by drafter
    int target_capture_ids_[4] = {5, 23, 41, 48};
    float mtp_blend_[4][4] = {}; // softmax(fuse_weight, axis=-1)
    TensorView mtp_mask_embed_;
    TensorView mtp_output_norm_;
    TensorView mtp_markov_dflare_norm_;
    TensorView mtp_markov_w1_;
    TensorView mtp_markov_rnn_w_;
    TensorView mtp_markov_rnn_b_;
    TensorView mtp_markov_hid_w_;
    TensorView mtp_markov_hid_b_;
    std::vector<DraftBlockWeights> mtp_blocks_;
    std::vector<DraftBlockState> mtp_states_;

    id buf_mtp_cap_[4] = {nil, nil, nil, nil}; // private f32[chunk_size, D]
    id buf_mtp_mix_ = nil;       // private f32[chunk_size, D]
    id buf_mtp_markov_ctx_ = nil;// private f32[D]
    id buf_mtp_h_ = nil;         // private f32[8, D]
    id buf_mtp_res_ = nil;       // private f32[8, D]
    id buf_mtp_normed_ = nil;    // private f32[8, D]
    id buf_mtp_k_self_ = nil;    // private half[8, Hkv, head_dim]
    id buf_mtp_v_self_ = nil;    // private half[8, Hkv, head_dim]
    id buf_mtp_mlp_g_ = nil;     // private f32[8, 9216]
    id buf_mtp_mlp_u_ = nil;     // private f32[8, 9216]
    id buf_mtp_rnn_in_ = nil;    // private f32[5120]
    id buf_mtp_rnn_out_ = nil;   // private f32[768]
    id buf_mtp_rnn_state_ = nil; // private f32[256]
    id buf_mtp_signal_ = nil;    // private f32[256]
    id buf_mtp_eff_h_ = nil;     // private f32[D]
    id buf_mtp_tok_ring_ = nil;  // shared uint32[8] ([anchor, draft0, draft1, ...])
    id buf_mtp_verify_logits_ = nil; // private f32[4, unpadded_vocab_size]

    // Cached pipelines
    id pso_qmm_q4_ = nil;
    id pso_qmm_q4_b4_ = nil;
    id pso_qmm_q4_tiled_ = nil;
    id pso_qmm_q8_ = nil;
    id pso_qmm_q8_b4_ = nil;
    id pso_qmm_q8_tiled_ = nil;
    id pso_qmm_bf16_ = nil;
    id pso_qmm_bf16_b4_ = nil;
    id pso_qmm_bf16_tiled_ = nil;
    id pso_embed_q4_ = nil;
    id pso_rmsnorm_ = nil;
    id pso_swiglu_ = nil;
    id pso_copy_ = nil;
    id pso_argmax1_ = nil;
    id pso_argmax2_ = nil;

    id pso_ink_embed_ = nil;
    id pso_ink_attn_res_ = nil;
    id pso_ink_attn_res_norm1_ = nil;
    id pso_ink_ctrl_fused_ = nil;
    id pso_ink_ctrl_prep_ = nil;
    id pso_ink_ctrl_apply_ = nil;
    id pso_ink_sconv_st_ = nil;
    id pso_ink_sconv_dy_ = nil;
    id pso_ink_q_rel_prep_ = nil;
    id pso_ink_kv_prep_ = nil;
    id pso_ink_rel_prof_ = nil;
    id pso_ink_attn_ = nil;
    id pso_ink_router_sel_ = nil;
    id pso_ink_act_q_ = nil;
    id pso_ink_mxfp8_act_q_ = nil;
    id pso_ink_moe_w13_ = nil;
    id pso_ink_moe_w2_ = nil;
    id pso_ink_moe_mxfp8_w13_ = nil;
    id pso_ink_moe_mxfp8_w2_ = nil;
    id pso_ink_moe_sort_ = nil;
    id pso_ink_moe_w13_grp_ = nil;
    id pso_ink_moe_w2_grp_ = nil;
    id pso_ink_moe_mxfp8_w13_grp_ = nil;
    id pso_ink_moe_mxfp8_w2_grp_ = nil;
    id pso_ink_moe_reduce_ = nil;
    id pso_ink_softcap_ = nil;

    id pso_mtp_mix_norm_ = nil;
    id pso_mtp_k_rope_ = nil;
    id pso_mtp_qkv_prep_ = nil;
    id pso_mtp_attn_ = nil;
    id pso_mtp_add_norm_ = nil;
    id pso_mtp_fill_masks_ = nil;
    id pso_mtp_markov_prep_ = nil;
    id pso_mtp_markov_gate_ = nil;
    id pso_mtp_add_h_bias_ = nil;

    LinearWeight load_linear(const std::string& full_key, const std::unordered_map<std::string, int>& quant_bits);
    void encode_linear(id enc, const LinearWeight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res = false);
    void encode_bf16_mat(id enc, const TensorView& w, uint32_t out_dim, uint32_t in_dim, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res = false);
    void encode_copy(id enc, id src, size_t src_off, id dst, size_t dst_off, uint32_t count);
    void encode_rmsnorm(id enc, id x_buf, size_t x_off, const TensorView& w, id y_buf, size_t y_off, uint32_t M);
    void encode_controller(id enc, const ControllerWeights& cw, id x_buf, size_t x_off, id msg_buf, size_t msg_off, id out_buf, size_t out_off, uint32_t M);
    void encode_chunk(id enc, int M, int start_pos, bool compute_logits, bool compute_argmax,
                      bool verify_all_logits = false, bool save_conv_tape = false, bool capture_mtp = false);
    void encode_mtp_observe(id enc, int M, int start_pos);
    void encode_mtp_propose(id enc, int anchor_token, int count);
    void rollback_conv_tapes(int kept_minus_one);
};

} // namespace mi
