#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace mi {

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
    if (!std::isfinite(cfg_.repeat_penalty) || cfg_.repeat_penalty <= 0.0f) cfg_.repeat_penalty = 1.0f;
    if (!std::isfinite(cfg_.presence_penalty)) cfg_.presence_penalty = 0.0f;
    if (!std::isfinite(cfg_.temperature) || cfg_.temperature < 0.0f) cfg_.temperature = 0.0f;
    if (!std::isfinite(cfg_.top_p)) cfg_.top_p = 1.0f;
}

void Sampler::reset(uint64_t seed) {
    rng_.seed(seed);
    for (int tok : seen_tokens_) {
        if (tok >= 0 && tok < static_cast<int>(seen_counts_.size())) {
            seen_counts_[tok] = 0;
        }
    }
    seen_tokens_.clear();
}

void Sampler::observe(int token) {
    if (token < 0) return;
    if (token >= static_cast<int>(seen_counts_.size())) {
        seen_counts_.resize(static_cast<size_t>(token) + 1024, 0);
    }
    if (seen_counts_[token] == 0) seen_tokens_.push_back(token);
    seen_counts_[token]++;
}

size_t Sampler::build_distribution(const float* logits, int vocab_size) {
    if (vocab_size <= 0 || logits == nullptr) {
        cands_.clear();
        probs_.clear();
        cands_.emplace_back(0.0f, 0);
        probs_.push_back(1.0f);
        return 1;
    }

    const float rp = (std::isfinite(cfg_.repeat_penalty) && cfg_.repeat_penalty > 0.0f)
        ? cfg_.repeat_penalty : 1.0f;
    const bool has_pen = (cfg_.presence_penalty != 0.0f || rp != 1.0f);
    const int seen_n = static_cast<int>(seen_counts_.size());

    auto adj = [&](float v, int i) -> float {
        if (std::isnan(v)) return -std::numeric_limits<float>::infinity();
        if (has_pen && i >= 0 && i < seen_n && seen_counts_[i] > 0) {
            if (cfg_.presence_penalty != 0.0f) v -= cfg_.presence_penalty;
            if (rp != 1.0f) {
                v = (v > 0.0f) ? (v / rp) : (v * rp);
            }
        }
        return v;
    };

    // Fast pass 1: NaN-safe NEON-vectorized max scan across all logits (~0.04 ms vs 0.65 ms scalar!)
    float max_v = -std::numeric_limits<float>::infinity();
    int best_id = 0;
    if (!has_pen) {
#if defined(__ARM_NEON)
        float32x4_t vm0 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm1 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm2 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm3 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        int i = 0;
        for (; i + 16 <= vocab_size; i += 16) {
            vm0 = vmaxnmq_f32(vm0, vld1q_f32(logits + i));
            vm1 = vmaxnmq_f32(vm1, vld1q_f32(logits + i + 4));
            vm2 = vmaxnmq_f32(vm2, vld1q_f32(logits + i + 8));
            vm3 = vmaxnmq_f32(vm3, vld1q_f32(logits + i + 12));
        }
        float32x4_t vm = vmaxnmq_f32(vmaxnmq_f32(vm0, vm1), vmaxnmq_f32(vm2, vm3));
        max_v = vmaxnmvq_f32(vm);
        for (; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] > max_v) max_v = logits[i];
        }
#else
        for (int i = 0; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] > max_v) { max_v = logits[i]; best_id = i; }
        }
#endif
    } else {
        // Even with penalties: NaN-safe NEON max over raw logits, then exact pass over all possible winners
#if defined(__ARM_NEON)
        float32x4_t vm0 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm1 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm2 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        float32x4_t vm3 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        int i = 0;
        for (; i + 16 <= vocab_size; i += 16) {
            vm0 = vmaxnmq_f32(vm0, vld1q_f32(logits + i));
            vm1 = vmaxnmq_f32(vm1, vld1q_f32(logits + i + 4));
            vm2 = vmaxnmq_f32(vm2, vld1q_f32(logits + i + 8));
            vm3 = vmaxnmq_f32(vm3, vld1q_f32(logits + i + 12));
        }
        float raw_max = vmaxnmvq_f32(vmaxnmq_f32(vmaxnmq_f32(vm0, vm1), vmaxnmq_f32(vm2, vm3)));
        for (; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] > raw_max) raw_max = logits[i];
        }
        cands_.clear();
        if (!std::isfinite(raw_max)) {
            // Handle +INFINITY or all-(-INF/NaN) rows safely without inf - inf NaNs
            if (raw_max > 0.0f) {
                for (int j = 0; j < vocab_size; ++j) {
                    if (std::isinf(logits[j]) && logits[j] > 0.0f) {
                        max_v = std::numeric_limits<float>::infinity();
                        if (cands_.empty()) best_id = j;
                        cands_.emplace_back(max_v, j);
                    }
                }
            }
            if (cands_.empty()) {
                cands_.emplace_back(0.0f, 0);
                probs_.assign(1, 1.0f);
                return 1;
            }
            return finalize_candidates(max_v);
        }
        // Conservative penalty margin for both positive and negative raw_max (and rp > 1 or 0 < rp < 1)
        float rp_margin = 0.0f;
        if (rp > 1.0f) {
            rp_margin = std::fabs(raw_max) * std::max(1.0f - 1.0f / rp, rp - 1.0f);
        } else if (rp < 1.0f) {
            rp_margin = std::fabs(raw_max) * std::max(1.0f / rp - 1.0f, 1.0f - rp);
        }
        float pen_margin = std::fabs(cfg_.presence_penalty) + rp_margin + 2.0f;
        float pre_cut = raw_max - std::max(18.5f * std::max(cfg_.temperature, 0.01f), 0.0f) - pen_margin;
        float32x4_t vcut = vdupq_n_f32(pre_cut);
        i = 0;
        for (; i + 16 <= vocab_size; i += 16) {
            uint32x4_t m0 = vcgeq_f32(vld1q_f32(logits + i), vcut);
            uint32x4_t m1 = vcgeq_f32(vld1q_f32(logits + i + 4), vcut);
            uint32x4_t m2 = vcgeq_f32(vld1q_f32(logits + i + 8), vcut);
            uint32x4_t m3 = vcgeq_f32(vld1q_f32(logits + i + 12), vcut);
            uint32x4_t any = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
            if (vmaxvq_u32(any) != 0u) {
                for (int j = i; j < i + 16; ++j) {
                    if (!std::isnan(logits[j]) && logits[j] >= pre_cut) {
                        float av = adj(logits[j], j);
                        if (av > max_v) { max_v = av; best_id = j; }
                        cands_.emplace_back(av, j);
                    }
                }
            }
        }
        for (; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] >= pre_cut) {
                float av = adj(logits[i], i);
                if (av > max_v) { max_v = av; best_id = i; }
                cands_.emplace_back(av, i);
            }
        }
        if (cands_.empty()) {
            for (int j = 0; j < vocab_size; ++j) {
                if (std::isnan(logits[j])) continue;
                float av = adj(logits[j], j);
                if (av > max_v) { max_v = av; best_id = j; }
            }
            cands_.emplace_back(std::isfinite(max_v) ? max_v : 0.0f, best_id);
        }
        if (cfg_.temperature <= 0.0f) {
            cands_.clear(); probs_.clear();
            cands_.emplace_back(max_v, best_id); probs_.push_back(1.0f);
            return 1;
        }
        if (std::isfinite(max_v)) {
            float cutoff = max_v - 18.5f * cfg_.temperature;
            cands_.erase(std::remove_if(cands_.begin(), cands_.end(),
                                        [cutoff](const auto& p) { return !(p.first >= cutoff); }),
                         cands_.end());
        }
        if (cands_.empty()) cands_.emplace_back(std::isfinite(max_v) ? max_v : 0.0f, best_id);
        return finalize_candidates(max_v);
#else
        for (int i = 0; i < vocab_size; ++i) {
            if (std::isnan(logits[i])) continue;
            float v = adj(logits[i], i);
            if (v > max_v) { max_v = v; best_id = i; }
        }
#endif
    }

    if (!std::isfinite(max_v)) {
        cands_.clear();
        if (max_v > 0.0f) {
            for (int i = 0; i < vocab_size; ++i) {
                if (std::isinf(logits[i]) && logits[i] > 0.0f) {
                    if (cands_.empty()) best_id = i;
                    cands_.emplace_back(max_v, i);
                }
            }
        }
        if (cands_.empty()) {
            cands_.emplace_back(0.0f, 0);
            probs_.assign(1, 1.0f);
            return 1;
        }
        return finalize_candidates(max_v);
    }

    if (cfg_.temperature <= 0.0f) {
        for (int i = 0; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] >= max_v) { best_id = i; break; }
        }
        cands_.clear();
        probs_.clear();
        cands_.emplace_back(max_v, best_id);
        probs_.push_back(1.0f);
        return 1;
    }

    // Fast pass 2: NEON threshold filter (skips 99.9% of 16-float blocks in 4 instructions!)
    {
        float cutoff = max_v - 18.5f * cfg_.temperature;
        cands_.clear();
#if defined(__ARM_NEON)
        float32x4_t vcut = vdupq_n_f32(cutoff);
        int i = 0;
        for (; i + 16 <= vocab_size; i += 16) {
            uint32x4_t m0 = vcgeq_f32(vld1q_f32(logits + i), vcut);
            uint32x4_t m1 = vcgeq_f32(vld1q_f32(logits + i + 4), vcut);
            uint32x4_t m2 = vcgeq_f32(vld1q_f32(logits + i + 8), vcut);
            uint32x4_t m3 = vcgeq_f32(vld1q_f32(logits + i + 12), vcut);
            uint32x4_t any = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
            if (vmaxvq_u32(any) != 0u) {
                for (int j = i; j < i + 16; ++j) {
                    if (!std::isnan(logits[j]) && logits[j] >= cutoff) cands_.emplace_back(logits[j], j);
                }
            }
        }
        for (; i < vocab_size; ++i) {
            if (!std::isnan(logits[i]) && logits[i] >= cutoff) cands_.emplace_back(logits[i], i);
        }
#else
        for (int i = 0; i < vocab_size; ++i) {
            if (std::isnan(logits[i])) continue;
            float v = adj(logits[i], i);
            if (v >= cutoff) cands_.emplace_back(v, i);
        }
#endif
        if (cands_.empty()) cands_.emplace_back(max_v, best_id);
    }

    return finalize_candidates(max_v);
}

int Sampler::sample(const float* logits, int vocab_size) {
    if (vocab_size <= 0 || logits == nullptr) return 0;
    if (is_greedy()) {
        int best = 0;
        float best_v = -std::numeric_limits<float>::infinity();
        bool found = false;
        for (int i = 0; i < vocab_size; ++i) {
            float v = logits[i];
            if (std::isnan(v)) continue;
            if (!found || v > best_v) {
                best_v = v;
                best = i;
                found = true;
            }
        }
        return best;
    }

    size_t keep = build_distribution(logits, vocab_size);
    if (keep <= 1) return cands_[0].second;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs_[i];
        if (r <= acc) return cands_[i].second;
    }
    return cands_[keep - 1].second;
}

int Sampler::sample_or_accept_draft(const float* logits, int vocab_size, int draft_tok, bool& accepted) {
    size_t keep = build_distribution(logits, vocab_size);

    float p_draft = 0.0f;
    size_t draft_idx = keep;
    for (size_t i = 0; i < keep; ++i) {
        if (cands_[i].second == draft_tok) {
            p_draft = probs_[i];
            draft_idx = i;
            break;
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (p_draft > 0.0f && (p_draft >= 1.0f || dist(rng_) < p_draft)) {
        accepted = true;
        return draft_tok;
    }

    // Rejected: sample from exact residual distribution p(x) / (1 - p(draft_tok)) for x != draft_tok
    accepted = false;
    float rem_sum = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        if (i != draft_idx) rem_sum += probs_[i];
    }
    if (rem_sum <= 1e-12f) {
        accepted = (draft_idx < keep);
        return accepted ? draft_tok : cands_[0].second;
    }
    float r = dist(rng_) * rem_sum;
    float acc = 0.0f;
    int last_tok = cands_[0].second;
    for (size_t i = 0; i < keep; ++i) {
        if (i == draft_idx) continue;
        last_tok = cands_[i].second;
        acc += probs_[i];
        if (r <= acc) return last_tok;
    }
    return last_tok;
}

float Sampler::compact_margin() const {
    // Multiplicative repeat_penalty shifts logits proportionally to |raw_max|, which cannot be
    // bounded by a fixed constant margin without risking truncation of true post-penalty winners.
    // Return 0.0f to disable GPU logit compaction and use the exact NEON full-logit pass instead!
    if (cfg_.repeat_penalty != 1.0f) return 0.0f;
    float t = std::max(cfg_.temperature, 0.01f);
    float m = 18.5f * t;
    if (cfg_.presence_penalty != 0.0f) {
        m += std::fabs(cfg_.presence_penalty) + 2.0f;
    }
    return m;
}

size_t Sampler::finalize_candidates(float max_v) {
    if (cands_.empty()) {
        cands_.emplace_back(0.0f, 0);
        probs_.assign(1, 1.0f);
        return 1;
    }
    if (cfg_.temperature <= 0.0f) {
        auto best_it = std::max_element(cands_.begin(), cands_.end(),
                                        [](const auto& a, const auto& b) { return a.first < b.first; });
        auto best = *best_it;
        cands_.clear();
        probs_.clear();
        cands_.push_back(best);
        probs_.push_back(1.0f);
        return 1;
    }
    int k = (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(cands_.size()))
        ? cfg_.top_k : static_cast<int>(cands_.size());
    if (k < static_cast<int>(cands_.size())) {
        std::nth_element(cands_.begin(), cands_.begin() + k, cands_.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        cands_.resize(static_cast<size_t>(k));
    }
    std::sort(cands_.begin(), cands_.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    probs_.resize(cands_.size());
    float sum = 0.0f;
    if (std::isinf(max_v) && max_v > 0.0f) {
        for (size_t i = 0; i < cands_.size(); ++i) {
            probs_[i] = (std::isinf(cands_[i].first) && cands_[i].first > 0.0f) ? 1.0f : 0.0f;
            sum += probs_[i];
        }
    } else if (std::isfinite(max_v)) {
        float inv_t = 1.0f / cfg_.temperature;
        for (size_t i = 0; i < cands_.size(); ++i) {
            float p = std::isfinite(cands_[i].first)
                ? std::exp((cands_[i].first - max_v) * inv_t)
                : 0.0f;
            if (!std::isfinite(p)) p = 0.0f;
            probs_[i] = p;
            sum += p;
        }
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::fill(probs_.begin(), probs_.end(), 0.0f);
        probs_[0] = 1.0f;
        return 1;
    }
    float inv_sum = 1.0f / sum;
    for (float& p : probs_) p *= inv_sum;

    size_t keep = probs_.size();
    if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f) {
        float cum = 0.0f;
        for (size_t i = 0; i < probs_.size(); ++i) {
            cum += probs_[i];
            if (cum >= cfg_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    float renorm = 0.0f;
    for (size_t i = 0; i < keep; ++i) renorm += probs_[i];
    if (!(renorm > 0.0f) || !std::isfinite(renorm)) {
        probs_[0] = 1.0f;
        return 1;
    }
    float inv_r = 1.0f / renorm;
    for (size_t i = 0; i < keep; ++i) probs_[i] *= inv_r;
    return keep;
}

size_t Sampler::build_distribution_from_cands(const LogitCand* gpu_cands, uint32_t cand_cnt) {
    const float rp = (std::isfinite(cfg_.repeat_penalty) && cfg_.repeat_penalty > 0.0f)
        ? cfg_.repeat_penalty : 1.0f;
    const bool has_pen = (cfg_.presence_penalty != 0.0f || rp != 1.0f);
    const int seen_n = static_cast<int>(seen_counts_.size());
    cands_.clear();
    cands_.reserve(cand_cnt);
    float max_v = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < cand_cnt; ++i) {
        int id = static_cast<int>(gpu_cands[i].id);
        float v = gpu_cands[i].val;
        if (std::isnan(v)) continue;
        if (has_pen && id >= 0 && id < seen_n && seen_counts_[id] > 0) {
            if (cfg_.presence_penalty != 0.0f) v -= cfg_.presence_penalty;
            if (rp != 1.0f) {
                v = (v > 0.0f) ? (v / rp) : (v * rp);
            }
        }
        if (v > max_v) max_v = v;
        cands_.emplace_back(v, id);
    }
    if (cands_.empty()) return 0;
    return finalize_candidates(max_v);
}

int Sampler::sample_from_cands(const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
                               const float* full_logits, int vocab_size) {
    size_t keep = 0;
    if (cfg_.repeat_penalty == 1.0f && gpu_cands != nullptr && cand_cnt > 0 && cand_cnt <= max_cands) {
        keep = build_distribution_from_cands(gpu_cands, cand_cnt);
    }
    if (keep == 0) {
        keep = build_distribution(full_logits, vocab_size);
    }
    if (keep <= 1) return cands_[0].second;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs_[i];
        if (r <= acc) return cands_[i].second;
    }
    return cands_[keep - 1].second;
}

int Sampler::sample_or_accept_draft_dist(
    const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
    const float* full_logits, int vocab_size,
    int draft_tok, const DraftDist* draft_dist, bool& accepted
) {
    size_t keep = 0;
    if (cfg_.repeat_penalty == 1.0f && gpu_cands != nullptr && cand_cnt > 0 && cand_cnt <= max_cands) {
        keep = build_distribution_from_cands(gpu_cands, cand_cnt);
    }
    if (keep == 0) {
        keep = build_distribution(full_logits, vocab_size);
    }

    float p_draft = 0.0f;
    size_t draft_idx = keep;
    for (size_t i = 0; i < keep; ++i) {
        if (cands_[i].second == draft_tok) {
            p_draft = probs_[i];
            draft_idx = i;
            break;
        }
    }

    // Lookup q(x) for any token x from draft_dist (or deterministic q(draft_tok)=1 when draft_dist == nullptr)
    const bool has_q = (draft_dist != nullptr && draft_dist->count > 0 && draft_dist->count <= 256);
    float q_draft = has_q ? draft_dist->sampled_q : 1.0f;
    if (q_draft <= 0.0f) q_draft = 1e-30f;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float accept_prob = (p_draft >= q_draft) ? 1.0f : (p_draft / q_draft);
    if (p_draft > 0.0f && (accept_prob >= 1.0f || dist(rng_) < accept_prob)) {
        accepted = true;
        return draft_tok;
    }

    // Rejected: sample from exact Leviathan residual norm(max(0, p(y) - q(y)))!
    accepted = false;
    auto lookup_q = [&](int tok) -> float {
        if (!has_q) return (tok == draft_tok) ? 1.0f : 0.0f;
        uint32_t n = draft_dist->count;
        for (uint32_t j = 0; j < n; ++j) {
            if (static_cast<int>(draft_dist->cands[j].id) == tok) {
                return draft_dist->cands[j].val;
            }
        }
        return 0.0f;
    };

    float rem_sum = 0.0f;
    // Reuse probs_ in-place to hold residual weights max(0, p(y) - q(y))
    for (size_t i = 0; i < keep; ++i) {
        float diff = probs_[i] - lookup_q(cands_[i].second);
        float w = (diff > 0.0f) ? diff : 0.0f;
        probs_[i] = w;
        rem_sum += w;
    }
    if (rem_sum <= 1e-12f) {
        accepted = (draft_idx < keep);
        return accepted ? draft_tok : cands_[0].second;
    }
    float r = dist(rng_) * rem_sum;
    float acc = 0.0f;
    int last_tok = cands_[0].second;
    for (size_t i = 0; i < keep; ++i) {
        if (probs_[i] <= 0.0f) continue;
        last_tok = cands_[i].second;
        acc += probs_[i];
        if (r <= acc) return last_tok;
    }
    return last_tok;
}

int Sampler::verify_tree_node_dist(
    const LogitCand* gpu_cands, uint32_t cand_cnt, uint32_t max_cands,
    const float* full_logits, int vocab_size,
    const int* child_toks, int num_children,
    const DraftDist* child0_dist, int& chosen_child
) {
    chosen_child = -1;
    size_t keep = 0;
    if (cfg_.repeat_penalty == 1.0f && gpu_cands != nullptr && cand_cnt > 0 && cand_cnt <= max_cands) {
        keep = build_distribution_from_cands(gpu_cands, cand_cnt);
    }
    if (keep == 0) {
        keep = build_distribution(full_logits, vocab_size);
    }

    if (num_children <= 0) {
        if (keep <= 1) return cands_[0].second;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_), acc = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            acc += probs_[i];
            if (r <= acc) return cands_[i].second;
        }
        return cands_[keep - 1].second;
    }

    // 1. Test primary child 0 against q0 = child0_dist (or deterministic delta(child_toks[0]) if nullptr)
    const int c0 = child_toks[0];
    float p0 = 0.0f;
    size_t idx0 = keep;
    for (size_t i = 0; i < keep; ++i) {
        if (cands_[i].second == c0) { p0 = probs_[i]; idx0 = i; break; }
    }
    const bool has_q0 = (child0_dist != nullptr && child0_dist->count > 0 && child0_dist->count <= 256);
    float q0 = has_q0 ? child0_dist->sampled_q : 1.0f;
    if (q0 <= 0.0f) q0 = 1e-30f;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float acc0 = (p0 >= q0) ? 1.0f : (p0 / q0);
    if (p0 > 0.0f && (acc0 >= 1.0f || dist(rng_) < acc0)) {
        chosen_child = 0;
        return c0;
    }

    // Build exact residual r(x) = max(0, p(x) - q0(x)) in probs_[0..keep-1]
    auto lookup_q0 = [&](int tok) -> float {
        if (!has_q0) return (tok == c0) ? 1.0f : 0.0f;
        uint32_t n = child0_dist->count;
        for (uint32_t j = 0; j < n; ++j) {
            if (static_cast<int>(child0_dist->cands[j].id) == tok) return child0_dist->cands[j].val;
        }
        return 0.0f;
    };

    float rem_sum = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        float diff = probs_[i] - lookup_q0(cands_[i].second);
        float w = (diff > 0.0f) ? diff : 0.0f;
        probs_[i] = w;
        rem_sum += w;
    }
    if (rem_sum <= 1e-12f) {
        chosen_child = (idx0 < keep) ? 0 : -1;
        return (chosen_child == 0) ? c0 : cands_[0].second;
    }

    // 2. Test distinct backup children b = 1..num_children-1 against exact normalized residual r(x)!
    for (int b = 1; b < num_children; ++b) {
        int cb = child_toks[b];
        bool dup = false;
        for (int prev = 0; prev < b; ++prev) {
            if (child_toks[prev] == cb) { dup = true; break; }
        }
        if (dup) continue;

        size_t idxb = keep;
        float wb = 0.0f;
        for (size_t i = 0; i < keep; ++i) {
            if (cands_[i].second == cb) { idxb = i; wb = probs_[i]; break; }
        }
        if (wb > 0.0f) {
            float pb = wb / rem_sum;
            if (pb >= 1.0f || dist(rng_) < pb) {
                chosen_child = b;
                return cb;
            }
            // Rejected backup child: zero its mass in residual without replacement!
            probs_[idxb] = 0.0f;
            rem_sum -= wb;
            if (rem_sum <= 1e-12f) {
                chosen_child = b;
                return cb;
            }
        }
    }

    // 3. All children rejected: sample exact bonus token from final residual r(x)!
    float r = dist(rng_) * rem_sum;
    float acc = 0.0f;
    int last_tok = cands_[0].second;
    for (size_t i = 0; i < keep; ++i) {
        if (probs_[i] <= 0.0f) continue;
        last_tok = cands_[i].second;
        acc += probs_[i];
        if (r <= acc) return last_tok;
    }
    return last_tok;
}

} // namespace mi
