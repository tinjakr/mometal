#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mi {

class Sampler;

struct EncodedImage {
    uint64_t content_hash = 0;
    int grid_t = 1;
    int grid_h = 0; // patch grid height (multiple of spatial_merge_size=2)
    int grid_w = 0; // patch grid width  (multiple of spatial_merge_size=2)
    int num_tokens() const { return grid_t * (grid_h / 2) * (grid_w / 2); }
    std::vector<float> embeddings; // [num_tokens, out_hidden_size] FP32
};

struct ModelInfo {
    std::string arch;         // "qwen38" or "inkling2"
    std::string model_dir;
    std::string quant_desc;
    int vocab_size = 0;
    int hidden_size = 0;
    int num_layers = 0;
    int num_visits = 0;
    int max_seq_len = 0;
    std::vector<int> eos_token_ids;
    size_t mapped_weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    double load_time_ms = 0.0;
};

struct StepStats {
    double gpu_ms = 0.0;
    int tokens_processed = 0;
    int mtp_accepted = 0;
    int mtp_drafted = 0;
    int pld_hits = 0;
};

class Model {
public:
    virtual ~Model() = default;

    virtual const ModelInfo& info() const = 0;

    // Reset KV / recurrent / conv caches to position 0
    virtual void reset() = 0;

    // Current sequence position (number of tokens in cache)
    virtual int position() const = 0;

    // Run forward pass on tokens[0..n-1], advancing cache by n.
    // Returns pointer to shared GPU buffer containing logits [vocab_size] for the LAST token.
    // If out_argmax != nullptr, also computes argmax on GPU and writes top token ID (zero vocab copy!).
    virtual const float* forward(const int* tokens, int n_tokens, int* out_argmax = nullptr, StepStats* stats = nullptr) = 0;

    // Fast greedy decode step: feeds a single token, runs entire forward + GPU argmax in ONE command buffer,
    // returns next token ID without copying logits to CPU!
    virtual int forward_greedy(int token, StepStats* stats = nullptr) = 0;

    // MTP / DFlash speculative decoding support (optional, active when --mtp or --dflash loaded)
    virtual bool has_mtp() const { return false; }
    virtual bool has_dflash() const { return false; }
    virtual int mtp_depth() const { return 0; }

    // Run one speculative draft+verify round starting from `anchor_token` (the most recently emitted
    // token that has NOT yet been fed into the target cache).
    // Appends 1..(depth+1) guaranteed-exact greedy tokens to `out_tokens` (ending with the bonus/correction token).
    // Advances both target and drafter caches cleanly (rolling back rejected draft tokens).
    virtual void forward_mtp_greedy(int anchor_token, std::vector<int>& out_tokens, StepStats* stats = nullptr) {
        int next = forward_greedy(anchor_token, stats);
        out_tokens.push_back(next);
    }

    // Speculative draft+verify round with exact rejection sampling for temperature > 0 / penalties!
    virtual bool supports_mtp_sampled() const { return false; }
    virtual void forward_mtp_sampled(int anchor_token, Sampler& sampler, std::vector<int>& out_tokens, StepStats* stats = nullptr) {
        (void)sampler;
        int next = forward_greedy(anchor_token, stats);
        out_tokens.push_back(next);
    }

    // Vision / multimodal support (optional, active when model has vision_tower)
    virtual bool has_vision() const { return false; }
    virtual int image_token_id() const { return -1; }
    virtual EncodedImage encode_image_url(const std::string& url, int max_edge = 1280) {
        (void)url; (void)max_edge;
        throw std::runtime_error("This model does not support vision inputs");
    }
    virtual const float* forward_multimodal(
        const int* tokens, int n_tokens, const std::vector<EncodedImage>& images,
        int* out_argmax = nullptr, StepStats* stats = nullptr
    ) {
        (void)images;
        return forward(tokens, n_tokens, out_argmax, stats);
    }

    // Disk-backed prefix/state cache serialization (MICACHE1 text + MICACHE2 multimodal with image hashes & rope_delta)
    virtual bool save_state_to_file(const std::string& path, const std::vector<int>& tokens) { return false; }
    virtual bool load_state_from_file(const std::string& path, std::vector<int>& out_tokens) { return false; }
    virtual bool save_state_to_file(const std::string& path, const std::vector<int>& tokens,
                                    const std::vector<uint64_t>& image_hashes) {
        return image_hashes.empty() ? save_state_to_file(path, tokens) : false;
    }
    virtual bool load_state_from_file(const std::string& path, std::vector<int>& out_tokens,
                                      std::vector<uint64_t>& out_image_hashes) {
        out_image_hashes.clear();
        return load_state_from_file(path, out_tokens);
    }

    // Lightweight non-mutating GPU + ANE keep-warm ping (prevents macOS idle clock/power sleep between turns)
    virtual void keep_warm() {}

    // Optional workspace/repo suffix corpus indexing for zero-cost multi-branch speculative drafting
    virtual void add_suffix_corpus(const int* tokens, int n_tokens) { (void)tokens; (void)n_tokens; }

    // Hardware per-op GPU + ANE timing breakdown (for optimization diagnostics)
    virtual void profile_ops() {}
};

} // namespace mi
