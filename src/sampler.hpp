#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace mi {

struct SamplerConfig {
    float temperature = 0.0f;      // 0.0 = greedy
    int   top_k = 40;
    float top_p = 0.95f;
    float presence_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    uint64_t seed = 42;
};

struct LogitCand {
    float val;
    uint32_t id;
};

struct DraftDist {
    uint32_t count;
    uint32_t sampled_id;
    float    sampled_q;
    float    q_top1;
    uint32_t alt_id[3];
    uint32_t alt_count;
    LogitCand cands[256];
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg = {});

    void reset(uint64_t seed = 42);
    void observe(int token);
    int sample(const float* logits, int vocab_size);

    // Round 9: Sample from GPU-compacted candidates (fallback to full logits if cand_cnt > max_cands)
    int sample_from_cands(const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
                          const float* full_logits, int vocab_size);

    // Exact speculative rejection sampling for a deterministic draft token `draft_tok` (q(draft_tok)=1)
    int sample_or_accept_draft(const float* logits, int vocab_size, int draft_tok, bool& accepted);

    // Round 9 + 10: Exact Leviathan speculative rejection sampling for a SAMPLED draft distribution q(x)!
    //   Accepts `draft_tok` with probability min(1, p(draft_tok) / q(draft_tok));
    //   otherwise samples from exact residual distribution norm(max(0, p(y) - q(y))).
    //   Uses GPU-compacted target candidates when cand_cnt <= max_cands for ~150x faster CPU verify!
    int sample_or_accept_draft_dist(
        const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
        const float* full_logits, int vocab_size,
        int draft_tok, const DraftDist* draft_dist, bool& accepted
    );

    // Stage 3: Exact SpecInfer Theorem 1 multi-child tree verification!
    // Tests child_toks[0] against q0 = child0_dist (or delta if nullptr), then tests distinct
    // backup children child_toks[1..num_children-1] sequentially against exact residual p'(x).
    // Returns chosen_child in [0..num_children-1] if accepted, or -1 if all rejected (bonus token returned).
    int verify_tree_node_dist(
        const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
        const float* full_logits, int vocab_size,
        const int* child_toks, int num_children,
        const DraftDist* child0_dist, int& chosen_child
    );

    const SamplerConfig& config() const { return cfg_; }
    uint64_t next_seed() { return rng_(); }
    float compact_margin() const;

    bool is_greedy() const {
        return cfg_.temperature <= 0.0f && cfg_.presence_penalty == 0.0f && cfg_.repeat_penalty == 1.0f;
    }

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<int> seen_counts_;
    std::vector<int> seen_tokens_;
    std::vector<std::pair<float, int>> cands_;
    std::vector<float> probs_;

    // Build filtered target distribution in cands_[0..keep-1], probs_[0..keep-1] (normalized, sum == 1)
    size_t build_distribution(const float* logits, int vocab_size);
    size_t build_distribution_from_cands(const LogitCand* gpu_cands, uint32_t cand_cnt);
    size_t finalize_candidates(float max_v);
};

} // namespace mi
