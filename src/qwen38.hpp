#pragma once

#include "ane_mlp.hpp"
#include "ane_share.hpp"
#include "dflash.hpp"
#include "metal_ctx.hpp"
#include "model.hpp"
#include "safetensors.hpp"
#include "suffix_cache.hpp"
#include "vision.hpp"

#include <climits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace mi {

class Qwen38Model : public Model {
public:
    Qwen38Model(MetalContext& ctx, const std::string& model_dir, int max_seq_len = 8192,
                const std::string& mtp_dir = "", int mtp_depth = 5, bool enable_pld = true,
                const std::string& ane_models_dir = "auto", const std::string& ane_tag = "auto",
                int kv_bits = 8, int ane_share_layers = 0, int sparse_kv_budget = 0,
                const std::string& dflash_dir = "");
    ~Qwen38Model() override;

    const ModelInfo& info() const override { return info_; }
    void reset() override;
    int position() const override { return pos_; }

    const float* forward(const int* tokens, int n_tokens, int* out_argmax = nullptr, StepStats* stats = nullptr) override;
    int forward_greedy(int token, StepStats* stats = nullptr) override;

    bool has_mtp() const override { return mtp_loaded_ || dflash_loaded_; }
    bool has_dflash() const override { return dflash_loaded_; }
    int mtp_depth() const override {
        return dflash_loaded_ ? (dflash_->config().block_size - 1) : (mtp_loaded_ ? mtp_depth_ : 0);
    }
    void forward_mtp_greedy(int anchor_token, std::vector<int>& out_tokens, StepStats* stats = nullptr) override;
    bool supports_mtp_sampled() const override { return mtp_loaded_ || dflash_loaded_; }
    void forward_mtp_sampled(int anchor_token, Sampler& sampler, std::vector<int>& out_tokens, StepStats* stats = nullptr) override;

    bool has_vision() const override { return vision_ && vision_->ready(); }
    int image_token_id() const override { return image_token_id_; }
    EncodedImage encode_image_url(const std::string& url, int max_edge = 1280) override;
    const float* forward_multimodal(
        const int* tokens, int n_tokens, const std::vector<EncodedImage>& images,
        int* out_argmax = nullptr, StepStats* stats = nullptr
    ) override;

    bool save_state_to_file(const std::string& path, const std::vector<int>& tokens) override {
        return save_state_to_file(path, tokens, {});
    }
    bool load_state_from_file(const std::string& path, std::vector<int>& out_tokens) override {
        std::vector<uint64_t> ignored;
        return load_state_from_file(path, out_tokens, ignored);
    }
    bool save_state_to_file(const std::string& path, const std::vector<int>& tokens,
                            const std::vector<uint64_t>& image_hashes) override;
    bool load_state_from_file(const std::string& path, std::vector<int>& out_tokens,
                              std::vector<uint64_t>& out_image_hashes) override;
    void wait_pending_save() override;
    void keep_warm() override;
    void add_suffix_corpus(const int* tokens, int n_tokens) override;
    void profile_ops() override;

private:
    struct Q4Weight {
        TensorView w;
        TensorView s;
        TensorView b;
        uint32_t out_dim = 0;
        uint32_t in_dim = 0;
    };

    struct LayerWeights {
        bool is_linear = true;
        TensorView input_norm;
        TensorView post_attn_norm;

        // Linear attention (GatedDeltaNet)
        Q4Weight in_proj_qkv;
        Q4Weight in_proj_z;
        Q4Weight in_proj_b;
        Q4Weight in_proj_a;
        Q4Weight out_proj;
        TensorView conv1d_w;
        TensorView A_log;
        TensorView dt_bias;
        TensorView gdn_norm;

        // Full attention (GQA + gate + partial RoPE)
        Q4Weight q_proj;
        Q4Weight k_proj;
        Q4Weight v_proj;
        Q4Weight o_proj;
        TensorView q_norm;
        TensorView k_norm;

        // Dense SwiGLU MLP
        Q4Weight gate_proj;
        Q4Weight up_proj;
        Q4Weight down_proj;
        int ane_share_idx = -1;
        TensorView post_attn_norm_share;
    };

    struct LayerState {
        // Linear attention state
        id conv_state = nil; // f32 [3, conv_dim]
        id ssm_state = nil;  // f32 [Hv, Dv, Dk]
        // Full attention KV cache (FP16 or 8/4-bit group-64 affine quantized)
        id k_cache = nil;    // f16 / u8 / u4 [max_seq, Hkv, D]
        id v_cache = nil;    // f16 / u8 / u4 [max_seq, Hkv, D]
        id k_sb = nil;       // half2 [max_seq, Hkv, D/64] (scale, bias) when kv_bits < 16
        id v_sb = nil;       // half2 [max_seq, Hkv, D/64] (scale, bias) when kv_bits < 16
        // Optional Quest query-aware block-sparse K min/max bounds (allocated ONLY when sparse_kv_budget_ > 0)
        id k_bounds = nil;   // half2 [max_pages, Hkv, D] (k_min, k_max) per 32-token page
        int quest_built_pages = 0;
    };

    struct MtpWeights {
        Q4Weight fc;
        TensorView pre_fc_norm_embedding;
        TensorView pre_fc_norm_hidden;
        LayerWeights layer; // 1 full-attention decoder layer
        TensorView norm;
    };

    MetalContext& ctx_;
    SafetensorsStore store_;
    SafetensorsStore mtp_store_;
    SafetensorsStore embed_store_; // Dedicated un-prefetched store for embed_tokens when MTP GPU drafting is active
    CpuMmapQ4Weight cpu_embed_;    // Zero-Metal-wiring CPU Q4 mmap embedder when !mtp_loaded_ (-682 MiB wired RAM!)
    bool use_cpu_embed_ = false;
    ModelInfo info_;

    // Hyperparameters
    int hidden_size_ = 5120;
    int intermediate_size_ = 17408;
    int num_layers_ = 64;
    int vocab_size_ = 248320;
    int full_attn_interval_ = 4;
    int num_heads_ = 24;
    int num_kv_heads_ = 4;
    int head_dim_ = 256;
    int rotary_dim_ = 64;
    float rope_theta_ = 10000000.0f;
    float rms_eps_ = 1e-6f;
    int lin_num_k_heads_ = 16;
    int lin_num_v_heads_ = 48;
    int lin_head_k_dim_ = 128;
    int lin_head_v_dim_ = 128;
    int lin_conv_dim_ = 10240;
    int max_seq_len_ = 8192;
    int kv_cap_ = 0; // Round 33: Lazy KV capacity grown on demand up to max_seq_len_!
    int kv_bits_ = 8; // 16 (FP16), 8 (Q8_G64), or 4 (Q4_G64)
    int mtp_kv_bits_ = 4; // Lossless KV4 for 1-layer MTP drafter (independent of target kv_bits_!)
    int sparse_kv_budget_ = 0; // Optional Quest block-sparse decode budget (0 = exact dense)
    static constexpr uint32_t QUEST_PAGE_SIZE = 32u;
    static constexpr uint32_t QUEST_SINK_PAGES = 4u;        // 128 sink tokens
    static constexpr uint32_t QUEST_RECENT_PAGES = 64u;     // 2048 recent tokens (always dense!)
    static constexpr uint32_t QUEST_MAX_ACTIVE_PAGES = 1024u;
    static constexpr uint32_t QUEST_SCORE_STRIDE = 8192u;   // up to 262k context pages
    int chunk_size_ = 512;
    int ane_min_ = 176;
    int pos_ = 0;
    int rope_delta_ = 0;
    int image_token_id_ = 248056;
    uint32_t special_tok_lo_ = 248000u;
    std::vector<uint32_t> special_tok_ids_;

    // Model weights & state
    Q4Weight embed_tokens_;
    TensorView final_norm_;
    Q4Weight lm_head_;
    std::vector<LayerWeights> layers_;
    std::vector<LayerState> states_;

    // MTP drafter state + Prompt-Lookup Drafting (PLD)
    static constexpr int MAX_VERIFY = 8;
    static constexpr uint32_t FAST_DRAFT_VOCAB = 32768u;
    static constexpr uint32_t MED_DRAFT_VOCAB  = 65536u;
    static constexpr uint32_t HI_DRAFT_VOCAB   = 151936u;
    static constexpr uint32_t MAX_DRAFT_ROW_MAP = 65536u;
    bool mtp_loaded_ = false;
    bool pld_enabled_ = true;
    int mtp_depth_ = 5;     // base draft depth (1..7)
    int cur_mtp_depth_ = 5; // adaptive draft depth in [2, 7]
    float mtp_alpha_ema_ = 0.72f; // Round 28: smooth EMA of acceptance rate
    uint32_t draft_vocab_size_ = FAST_DRAFT_VOCAB;
    uint32_t draft_miss_mask_ = 0u; // 16-bit sliding window of out-of-range bonus tokens
    // Dynamic compact row_map for MTP tree draft lm_head (0 MB weight duplication, zero CJK bloat!)
    uint32_t draft_vbase_ = 24576u;
    uint32_t draft_base_count_ = 0u;
    uint32_t draft_row_count_ = 0u;
    id buf_draft_row_map_ = nil; // shared uint32 [MAX_DRAFT_ROW_MAP]
    std::vector<uint8_t> draft_in_map_;
    // BitScreen 1-bit sign(W) x 3-bitplane(X) draft lm_head pre-filter + exact Q4 top-256 rescore!
    bool     bs_enabled_ = false;
    uint32_t bs_vocab_ = 0u;
    uint32_t bs_extra_base_ = 0u;
    uint32_t bs_extra_count_ = 0u;
    std::vector<uint8_t> bs_in_extra_;
    id buf_bs_w_        = nil; // private uint [bs_vocab, 160] (62.9 MB for 98,304; stride-2048 interleaved)
    id buf_bs_meta_     = nil; // private float2 [bs_vocab] (0.78 MB)
    id buf_bs_xbp_      = nil; // private uint [8, 3, 160]
    id buf_bs_xpc_      = nil; // private float [8]
    id buf_bs_top2_     = nil; // private ArgmaxOut [8, 4096] (256 KB L2-resident fused top-2!)
    id buf_bs_cands_    = nil; // private uint [8, 256]
    id buf_bs_cand_cnt_ = nil; // private uint [8]
    id buf_bs_extra_    = nil; // shared uint [128] (special + session IDs >= bs_vocab)
    int mtp_pos_ = 0;
    bool mtp_seeded_ = false;
    int mtp_seed_token_ = 0;
    std::vector<int> token_history_;
    // Round 27 + Stage 3: O(1) 2-way rolling hash-indexed 4-gram & 3-gram Prompt-Lookup Drafting
    struct PldSlot { uint32_t gen = 0; int32_t pos0 = -1; int32_t pos1 = -1; };
    static constexpr uint32_t PLD_MASK = 262143u;
    uint32_t pld_gen_ = 1u;
    std::vector<PldSlot> pld4_;
    std::vector<PldSlot> pld3_;
    SuffixCache suffix_cache_;
    MtpWeights mtp_;
    id mtp_k_cache_ = nil;        // u4 / f16 [max_seq, Hkv, head_dim]
    id mtp_v_cache_ = nil;        // u4 / f16 [max_seq, Hkv, head_dim]
    id mtp_k_sb_ = nil;           // half2 [max_seq, Hkv, head_dim/64] when mtp_kv_bits_ < 16
    id mtp_v_sb_ = nil;           // half2 [max_seq, Hkv, head_dim/64] when mtp_kv_bits_ < 16
    id mtp_k_bounds_ = nil;       // half2 [max_pages, Hkv, head_dim] when sparse_kv_budget_ > 0
    int mtp_quest_built_pages_ = 0;
    id buf_mtp_seed_hid_ = nil;   // private f32 [hidden]
    id buf_mtp_last_hid_ = nil;   // private f32 [hidden] (last target normed hidden from forward)
    id buf_mtp_verify_hid_ = nil; // private f32 [MAX_VERIFY, hidden]
    id buf_mtp_in_hid_ = nil;     // private f32 [chunk_size + 1, hidden]
    id buf_mtp_logits_ = nil;     // shared f32 [MAX_VERIFY, vocab_size] (verify batch logits)
    id buf_mtp_argmax_ = nil;     // shared uint32 [MAX_VERIFY]
    id buf_mtp_sync_ids_ = nil;   // shared int [MAX_VERIFY + 2] for single-CB deferred sync
    // Round 9 & 10: GPU verify candidate compaction + GPU sampled MTP draft distributions
    static constexpr uint32_t MAX_VERIFY_CANDS = 1024u;
    id buf_verify_row_max_ = nil; // private f32 [MAX_VERIFY]
    id buf_verify_cands_ = nil;   // shared LogitCand [MAX_VERIFY, MAX_VERIFY_CANDS]
    id buf_verify_cnt_ = nil;     // shared uint32 [MAX_VERIFY]
    id buf_draft_dists_ = nil;    // shared DraftDist [MAX_VERIFY]
    // Deferred MTP sync state (merges CB #2 into next round's CB #1 -> 1 CB/round!)
    bool pending_sync_ = false;
    int  pending_replay_m_ = 0;
    bool pending_seed_first_ = false;
    int  pending_mtp_pos_ = 0;
    int  pending_feed_count_ = 0;
    int  pending_feed_off_ = 0;
    int  pending_feed_tokens_[MAX_VERIFY + 10] = {};
    // Stage 2: Arbitrary speculative tree support (up to 16 nodes, 4 active GDN fork checkpoints)
    struct TreeParams {
        uint32_t is_tree = 0;
        uint32_t n_nodes = 0;
        uint32_t mask_base_pos = 0;
        uint32_t pad0 = 0;
        uint32_t mask[16] = {};
        uint32_t depth[16] = {};
        int32_t  parent[16] = {};
        int32_t  gdn_load[16] = {};
        int32_t  gdn_save[16] = {};
    };
    bool     pending_path_noncontig_ = false;
    int      pending_kv_pre_pos_ = 0;
    uint32_t pending_path_len_ = 0;
    uint32_t pending_path_[16] = {};
    TreeParams cur_tree_{};
    id buf_tree_params_ = nil;   // shared 256B TreeParams buffer for pre-encoded CB #2b!
    float ema_cb2_ms_[2] = {0.0f, 0.0f}; // live GPU ms EMA (base non-attn) for [0]=M=4, [1]=M=8
    int last_t7_m_ = 0;
    bool last_t7_tree_ = false;
    bool ane_share_nonuniform_lut_ = false;
    static TreeParams make_tree_params(const int* parents, int n_nodes, uint32_t mask_base_pos = 0);
    // Ping-pong MAX_VERIFY-token record buffers for Lazy Fused GDN Replay (ZERO bak_ssm_state_ -> saves 66 MB RAM!)
    std::vector<id> rec_qkv_[2]; // f32 [MAX_VERIFY, conv_dim] per linear layer
    std::vector<id> rec_a_[2];   // f32 [MAX_VERIFY, Hv] per linear layer
    std::vector<id> rec_b_[2];   // f32 [MAX_VERIFY, Hv] per linear layer
    int rec_slot_ = 0;           // slot holding LAST recorded verify round's projections

    // Native DFlash 2 block-diffusion speculative drafter (active when --dflash <dir> is provided)
    std::unique_ptr<DFlashEngine> dflash_;
    bool dflash_loaded_ = false;
    int df_kv_len_ = 0;
    int df_pending_commit_ = 0;
    int df_pending_pos_ = 0;
    size_t c_attn_ = 512;
    size_t df_cap_rows_ = 0;
    std::vector<id> df_k_cache_; // private half [Hkv, sliding_window, D] per DFlash layer
    std::vector<id> df_v_cache_; // private half [Hkv, sliding_window, D] per DFlash layer
    id buf_df_capture_   = nil;  // private half [df_cap_rows_, num_targets * hidden_size]
    id buf_df_ids_       = nil;  // shared int [block_size]
    id buf_df_h_         = nil;  // private float [block_size, hidden_size]
    id buf_df_x_         = nil;  // private half [block_size, hidden_size]
    id buf_df_x2_        = nil;  // private half [block_size, hidden_size]
    id buf_df_dyn_       = nil;  // private float [block_size, 2 * kernel_size * (H / group_size)]
    id buf_df_o_         = nil;  // private float [block_size, hidden_size]
    id buf_df_q_h_       = nil;  // private half [Hq, 8, D]
    id buf_df_kn_        = nil;  // private half [Hkv, block_size, D]
    id buf_df_vn_        = nil;  // private half [Hkv, block_size, D]
    id buf_df_attn_out_  = nil;  // private half [block_size, Hq * D]
    id buf_df_topk_part_ = nil;  // private LogitCand [P, num_chunks, K]
    id buf_df_topk_out_  = nil;  // shared LogitCand [P * K]
    id buf_df_sel_h_     = nil;  // shared half [P * selector_rank]
    std::vector<float> df_mask_embed_; // Pre-gathered MASK token embedding when use_cpu_embed_ is active (-682 MiB wired RAM!)

    // Native CoreML/IOSurface ANE MLP accelerator for prefill (+ optional single-copy hf4 weight sharing!)
    std::unique_ptr<AneMlpEngine> ane_;
    std::unique_ptr<AneShareManager> ane_share_;
    id buf_hidden_alt_ = nil; // Half-B hidden buffer for 2-half GPU || ANE pipeline

    // Round 11: Async background disk checkpoint writer thread
    std::thread save_thread_;

    // Native Metal Vision Tower + M-RoPE / multimodal scatter scratch (allocated lazily!)
    std::unique_ptr<QwenVisionTower> vision_;
    id buf_mrope_pos_ = nil;    // shared int [2 * C_attn * 3]
    id buf_vis_embeds_ = nil;   // shared float [2 * C_attn * hidden_size]
    id buf_vis_rows_ = nil;     // shared uint32 [2 * C_attn]
    // Active per-chunk multimodal bindings (non-nil only during an image chunk)
    id cur_mrope_buf_ = nil;
    size_t cur_mrope_off_ = 0;
    id cur_vis_emb_buf_ = nil;
    size_t cur_vis_emb_off_ = 0;
    id cur_vis_rows_buf_ = nil;
    size_t cur_vis_rows_off_ = 0;
    uint32_t cur_vis_count_ = 0;

    // Pre-allocated GPU scratch arena (placement heap aliases mutually-disjoint Attn/GDN, MLP, and MTP scratch!)
    id scratch_heap_ = nil;
    id buf_token_ids_ = nil;
    id buf_hidden_ = nil;
    id buf_normed_ = nil;
    id buf_qkv_ = nil;
    id buf_qkv_conv_ = nil;
    id buf_z_ = nil;
    id buf_a_ = nil;
    id buf_b_ = nil;
    id buf_g_ = nil;
    id buf_beta_ = nil;
    id buf_q_ = nil;
    id buf_q_h_ = nil;
    id buf_k_ = nil;
    id buf_v_ = nil;
    id buf_attn_out_ = nil;
    id buf_gdn_normed_ = nil;
    id buf_mlp_gate_ = nil;
    id buf_mlp_up_ = nil;
    id buf_split_o_ = nil;
    id buf_mtp_concat_ = nil;     // heap-placed private f32 [chunk_size, 2*hidden]
    id buf_mtp_emb_ = nil;        // heap-placed private f32 [chunk_size, hidden]
    id buf_split_m_ = nil;
    id buf_split_l_ = nil;
    id buf_quest_scores_ = nil; // private float [Hkv*7, QUEST_SCORE_STRIDE] when sparse_kv_budget_ > 0
    id buf_quest_pages_ = nil;  // private uint32 [Hkv, QUEST_MAX_ACTIVE_PAGES]
    id buf_quest_counts_ = nil; // private uint32 [Hkv]
    id buf_sk_part_ = nil; // Split-K partial buffer for decode down_proj / out_proj (640 KB, stays in L2!)
    id buf_logits_ = nil;
    id buf_argmax_parts_ = nil;
    id buf_argmax_out_ = nil;

    // Cached pipelines
    id pso_embed_ = nil;
    id pso_qmm_q4_ = nil;
    id pso_mma_qmm8_ = nil;
    id pso_mma_qmm8_f16x_ = nil;
    id pso_mma_qmm8_f16x_sk_ = nil;
    id pso_mma_sk_reduce_ = nil;
    id pso_mma_sk_reduce_norm_f16_ = nil;
    id pso_mma_sk_reduce_dual_norm_f16_ = nil;
    id pso_mma_lmhead_max_f16x_ = nil;
    id pso_mma_lmhead_max_f16x_indexed_ = nil;
    id pso_qmv_q4_f16x_max_ = nil;
    id pso_qmv_q4_f16x_indexed_max_ = nil;
    id pso_mma_qmm8_swiglu_ = nil;
    id pso_mma_qmm8_swiglu_f16_ = nil;
    id pso_mma_qmm16_ = nil;
    id pso_qmm_q4_tiled_ = nil;
    id pso_qmm_q4_tiled_f16x_ = nil;
    id pso_qmm_q4_swiglu_f16x_ = nil;
    id pso_qmm_q4_smalln_f16x_ = nil;
    id pso_qmm_q4_smalln_ab_f16x_ = nil;
    id pso_mma_in_proj_ab_f16x_ = nil;
    id pso_qmv_q4_swiglu_ = nil;
    id pso_qmv_q4_f16x_ = nil;
    id pso_qmv_q4_f16x_sk_ = nil;
    id pso_qmv_q4_swiglu_f16x_ = nil;
    id pso_mtp_embed_concat_f16_ = nil;
    id pso_rmsnorm_ = nil;
    id pso_rmsnorm_f16_ = nil;
    id pso_rmsnorm_dual_ = nil;
    id pso_swiglu_ = nil;
    id pso_swiglu_f16_ = nil;
    id pso_copy_ = nil;
    id pso_gdn_conv_ = nil;
    id pso_gdn_conv_pf_ = nil;
    id pso_gdn_conv_qknorm_gates_pf_ = nil;
    id pso_gdn_conv_upd_ = nil;
    id pso_gdn_qknorm_ = nil;
    id pso_gdn_gates_ = nil;
    id pso_gdn_prework_fused_ = nil;
    id pso_gdn_prework_tree_ = nil;
    id pso_gdn_prework_replay_path_ = nil;
    id pso_gdn_prework_lazy_ = nil;
    id pso_gdn_rec_ = nil;
    id pso_gdn_rec_tree_ = nil;
    id pso_gdn_rec_lazy_ = nil;
    id pso_gdn_rec_tree_lazy_ = nil;
    id pso_gdn_rec_tree_lazy_s1_ = nil;
    id pso_gdn_rec_tree_lazy_s2_ = nil;
    id pso_gdn_norm_ = nil;
    id pso_gdn_norm_f16_ = nil;
    id pso_fa_q_prep_ = nil;
    id pso_fa_q_prep_hier_ = nil;
    id pso_fa_q_prep_tree_ = nil;
    id pso_fa_kv_prep_ = nil;
    id pso_fa_kv_prep_tree_ = nil;
    id pso_fa_q_prep_mrope_ = nil;
    id pso_fa_q_prep_mrope_hier_ = nil;
    id pso_fa_kv_prep_mrope_ = nil;
    id pso_scatter_embeds_ = nil;
    id pso_fa_attn_ = nil;
    id pso_fa_attn_f16_ = nil;
    id pso_fa_attn_tree_f16_ = nil;
    id pso_fa_prefill_flash_ = nil;
    id pso_fa_prefill_flash_f16_ = nil;
    id pso_fa_prefill_flash_pair_f16_ = nil;
    id pso_fa_prefill_hier_f16_ = nil;
    id pso_pack_q_pf_h_ = nil;
    id pso_fa_split1_ = nil;
    id pso_fa_split2_ = nil;
    id pso_fa_vsplit1_ = nil;
    id pso_fa_vsplit1_tree_ = nil;
    id pso_fa_vsplit2_ = nil;
    id pso_fa_vsplit2_f16_ = nil;
    id pso_fa_gqa6_m1_ = nil;
    id pso_fa_gqa6_m1_tree_ = nil;
    id pso_fa_gqa6_m2_ = nil;
    id pso_fa_gqa6_m2_tree_ = nil;
    id pso_fa_gqa6_m4_ = nil;
    id pso_fa_gqa6_m4_tree_ = nil;
    id pso_fa_gqa6_v_ = nil;
    id pso_fa_gqa6_v_tree_ = nil;
    id pso_pack_q_h_ = nil;
    id pso_pack_q_p2_ = nil;
    id pso_pack_q_p4_ = nil;
    // Quantized KV (8-bit / 4-bit group-64 affine) pipelines
    id pso_fa_kv_prep_q_ = nil;
    id pso_fa_kv_prep_q_tree_ = nil;
    id pso_fa_kv_prep_mrope_q_ = nil;
    id pso_fa_attn_q_ = nil;
    id pso_fa_attn_q_f16_ = nil;
    id pso_fa_attn_q_tree_f16_ = nil;
    id pso_fa_prefill_flash_q_ = nil;
    id pso_fa_prefill_flash_q_f16_ = nil;
    id pso_fa_prefill_flash_pair_q_f16_ = nil;
    id pso_fa_prefill_hier_q_f16_ = nil;
    id pso_fa_split1_q_ = nil;
    id pso_fa_vsplit1_q_ = nil;
    id pso_fa_vsplit1_q_tree_ = nil;
    id pso_fa_gqa6_m1_q_ = nil;
    id pso_fa_gqa6_m1_q_tree_ = nil;
    id pso_fa_gqa6_m2_q_ = nil;
    id pso_fa_gqa6_m2_q_tree_ = nil;
    id pso_fa_gqa6_m4_q_ = nil;
    id pso_fa_gqa6_m4_q_tree_ = nil;
    id pso_fa_gqa6_v_q_ = nil;
    id pso_fa_gqa6_v_q_tree_ = nil;
    // Dedicated lossless KV4 pipelines for 1-layer MTP drafter (independent of target kv_bits_!)
    id pso_mtp_kv_prep_ = nil;
    id pso_mtp_kv_prep_tree_ = nil;
    id pso_mtp_attn_f16_ = nil;
    id pso_mtp_attn_tree_f16_ = nil;
    id pso_mtp_vsplit1_ = nil;
    id pso_mtp_vsplit1_tree_ = nil;
    id pso_mtp_gqa6_m1_ = nil;
    id pso_mtp_gqa6_m1_tree_ = nil;
    id pso_mtp_gqa6_m2_ = nil;
    id pso_mtp_gqa6_m2_tree_ = nil;
    id pso_mtp_gqa6_m4_ = nil;
    id pso_mtp_gqa6_m4_tree_ = nil;
    id pso_mtp_gqa6_v_ = nil;
    id pso_mtp_gqa6_v_tree_ = nil;
    // Quest query-aware block-sparse decode pipelines (loaded ONLY when sparse_kv_budget_ > 0)
    id pso_quest_build_bounds_ = nil;
    id pso_quest_score_pages_ = nil;
    id pso_quest_select_topk_ = nil;
    id pso_fa_gqa6_m1_sp_ = nil;
    id pso_fa_gqa6_m1_sp_tree_ = nil;
    id pso_fa_gqa6_m2_sp_ = nil;
    id pso_fa_gqa6_m2_sp_tree_ = nil;
    id pso_fa_gqa6_m4_sp_ = nil;
    id pso_fa_gqa6_m4_sp_tree_ = nil;
    id pso_fa_gqa6_v_sp_ = nil;
    id pso_fa_gqa6_v_sp_tree_ = nil;
    id pso_kv_compact_f16_ = nil;
    id pso_kv_compact_q_ = nil;
    id pso_copy_hid_path_ = nil;
    id pso_argmax1_ = nil;
    id pso_argmax2_ = nil;
    id pso_logit_max2_ = nil;
    id pso_logit_compact_ = nil;
    id pso_mtp_sample_draft_ = nil;
    id pso_mtp_sample_draft_batch_ = nil;
    id pso_mtp_embed_concat_gather_f16_ = nil;
    id pso_rmsnorm_dual_scatter_f16_ = nil;
    id pso_mtp_concat_ = nil;
    id pso_ane_pack_ = nil;
    id pso_ane_add_ = nil;
    id pso_ane_add_norm_ = nil;
    id pso_ane_add_norm_f16_ = nil;
    id pso_ane_hf4_swiglu_bf16_ = nil;
    id pso_ane_hf4_swiglu_m1_ = nil;
    id pso_ane_hf4_swiglu_m4_ = nil;
    id pso_ane_hf4_down_sk_bf16_ = nil;
    id pso_ane_hf4_down_sk4_bf16_ = nil;
    id pso_ane_hf4_down_m1_sk_ = nil;
    id pso_ane_hf4_down_sk_m4_ = nil;
    id pso_ane_hf4_tiled_gateup_ = nil;
    id pso_ane_hf4_tiled_down_ = nil;
    id pso_ane_hf4_swiglu_tiled32_ = nil;
    id pso_ane_hf4_down_tiled32_sk_ = nil;
    id pso_lmh_build_bitscreen_ = nil;
    id pso_bs_quant_x_ = nil;
    id pso_bs_score_b1_ = nil;
    id pso_bs_score_b2_ = nil;
    id pso_bs_score_b3_ = nil;
    id pso_bs_select_cands_ = nil;
    id pso_bs_rescore_q4_ = nil;
    // DFlash 2 block-diffusion drafter pipelines (loaded ONLY when --dflash <dir> is active)
    id pso_df_rmsnorm_f16_ = nil;
    id pso_df_conv_prep_f16_ = nil;
    id pso_df_conv_finish_f32_ = nil;
    id pso_df_conv_fin_norm_f16_ = nil;
    id pso_df_kv_write_ = nil;
    id pso_df_qkv_prep_ = nil;
    id pso_df_attn_p1_ = nil;
    id pso_df_attn_p2_ = nil;
    id pso_df_topk16_p1_ = nil;
    id pso_df_topk16_p2_ = nil;
    id pso_df_capture_hidden_ = nil;
    id pso_df_hidden_proj_f16_ = nil;

    id encode_draft_lm_head(id enc, uint32_t B, uint32_t& out_V, uint32_t& out_n_parts);

    static Q4Weight as_q4(const DFlashQ4Weight& w) { return Q4Weight{w.w, w.s, w.b, w.out_dim, w.in_dim}; }
    Q4Weight load_q4(const SafetensorsStore& st, const std::string& prefix);
    void ensure_kv_capacity(int needed);
    void shrink_kv_capacity(int target_cap);
    void encode_ane_tail_mlp(id enc, const LayerWeights& lw, id hid_buf, size_t tail_hid_off, uint32_t gm);
    void init_draft_row_map();
    void reset_draft_row_map();
    void add_draft_token(int tok);
    void maybe_widen_draft_vocab(int tok);
    void push_history_token(int tok);
    void update_mtp_depth_ema(int accepted, int K);
    void encode_qmm_q4(id enc, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res);
    void encode_qmm_q4_f16x(id enc, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off, uint32_t M, bool add_res);
    void encode_qmm_q4_f16x_and_norm(id enc, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off,
                                     uint32_t M, const TensorView& next_norm, id normed_buf);
    void encode_qmm_q4_f16x_and_dual_norm(id enc, const Q4Weight& w, id x_buf, size_t x_off, id y_buf, size_t y_off,
                                          uint32_t M, const TensorView& next_norm, id out_f32_buf, id out_f16_buf);
    void encode_mlp_q4(id enc, const LayerWeights& lw, uint32_t M, const TensorView* next_norm = nullptr,
                       id dual_out_f32 = nil, size_t hid_off = 0);
    void encode_layer_attn_only(id enc, int layer_idx, id hid_buf, uint32_t M, int start_pos, bool record_gdn = false, bool skip_input_norm = false);
    void encode_full_attn_layer(id enc, const LayerWeights& lw, id k_cache, id v_cache,
                                uint32_t M, int start_pos, id k_sb = nil, id v_sb = nil,
                                bool skip_input_norm = false, const TensorView* next_norm = nullptr,
                                id dual_out_f32 = nil, id k_bounds = nil, int* quest_built_pages = nullptr,
                                bool is_mtp = false, bool use_tree_buf = false);
    struct DraftSampleSpec {
        float temp = 0.0f;
        uint64_t seed = 0;
        id dist_buf = nil;
        size_t dist_off = 0;
        int top_k = 0;
        float top_p = 1.0f;
    };

    void encode_chunk(id enc, int M, int start_pos, bool compute_logits, bool compute_argmax,
                      bool verify_all_logits = false, bool record_gdn = false, float compact_margin = 0.0f,
                      bool use_tree_buf = false, int df_cap_src_row = -1, int df_cap_n_rows = 0);
    void encode_final_norm_and_mtp(id enc, id hid_buf, id tok_buf, size_t tok_off, uint32_t M, int start_pos,
                                   bool compute_logits, bool compute_argmax);
    void forward_ane_chunk(const int* tokens, int n_chunk, bool is_last, bool compute_argmax,
                           const int* pos3d = nullptr, const float* const* vis_ptrs = nullptr,
                           int df_capture_from_pos = INT_MAX);
    void encode_dflash_capture(id enc, id src_hid_buf, int src_row_off, int dst_row_off, int n_rows, int slot_idx);
    void encode_dflash_context_commit(id enc, int cap_row_off, int n_rows, int pos_start);
    void encode_dflash_draft_step(id enc, int start_pos);
    void forward_dflash_impl(int anchor_token, Sampler* sampler, std::vector<int>& out_tokens, StepStats* stats);
    void encode_mtp_kv_prime(id enc, id tok_buf, size_t tok_off, id hid_buf, size_t hid_off, int M, int start_pos);
    void encode_mtp_steps(id enc, id tok_buf, size_t tok_off, id hid_buf, size_t hid_off, int M, int start_pos,
                          bool compute_argmax_last, id argmax_dst_buf = nil, size_t argmax_dst_off = 0,
                          const DraftSampleSpec* sample_spec = nullptr);
    void encode_mtp_tree_expand(id enc, const int* parents, int n_nodes, int pre_draft_mtp_pos,
                                bool is_greedy, float draft_temp, int top_k, float top_p,
                                uint64_t seed_base, const int* prefilled_mask = nullptr,
                                int start_depth = 1, int max_expand_depth = 16,
                                int init_kv_cursor = 0, const int* init_kv_slot = nullptr,
                                const uint32_t* init_kv_mask = nullptr, bool need_greedy_dist = false,
                                const int* init_hid_alias = nullptr);
    void forward_mtp_tree_impl(int anchor_token, Sampler* sampler, std::vector<int>& out_tokens, StepStats* stats);
    void encode_gdn_replay_compute(id enc, int accepted_plus_one, const uint32_t* path = nullptr);
    void encode_kv_compact_path(id enc, int pre_pos, const uint32_t* path, int len);
    void flush_pending_mtp();
    int try_pld_draft(int anchor_token, int* out_draft, int max_draft, int prefer_first = -1) const;
    int try_pld3_draft(int anchor_token, int* out_draft, int max_draft, int prefer_first = -1) const;
};

} // namespace mi
