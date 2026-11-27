#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "dflash.hpp"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fs = std::filesystem;

namespace mi {

namespace {

inline size_t align256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

inline float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        return static_cast<uint16_t>((u >> 16) | ((u & 0x007FFFFFu) ? 0x0040u : 0u));
    }
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline uint16_t f32_to_f16_bits(float f) {
    __fp16 h = static_cast<__fp16>(f);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, 2);
    return bits;
}

inline float f16_bits_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<float>(h);
}

inline uint16_t load_u16_unaligned(const uint8_t* p) {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline float load_f32_unaligned(const uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

#if defined(__aarch64__) || defined(__ARM_NEON)
inline uint16x4_t load_u16x4_unaligned(const uint8_t* p) {
    uint16_t tmp[4];
    std::memcpy(tmp, p, sizeof(tmp));
    return vld1_u16(tmp);
}

inline float32x4_t load_f32x4_unaligned(const uint8_t* p) {
    float tmp[4];
    std::memcpy(tmp, p, sizeof(tmp));
    return vld1q_f32(tmp);
}
#endif

struct RawTensorMeta {
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    size_t byte_offset = 0; // offset from mmap_ptr_
    size_t nbytes = 0;
};

// Fast ARM64 NEON quantization of one BF16 row [K] -> MLX Q4_AFFINE_G64 (w [K/8] u32, s [K/64] bf16, b [K/64] bf16).
// Quantizes against exact rounded BF16 scale & bias so dequantization matches common.metal bit-for-bit!
void quantize_bf16_row_q4_g64(const uint8_t* src_bf16_bytes, uint32_t K,
                              uint32_t* dst_w, uint16_t* dst_s, uint16_t* dst_b) {
    const uint32_t num_groups = K >> 6; // K / 64
#if defined(__aarch64__) || defined(__ARM_NEON)
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vf15  = vdupq_n_f32(15.0f);
    const int32x4_t sh_lo   = {0, 4, 8, 12};
    const int32x4_t sh_hi   = {16, 20, 24, 28};

    for (uint32_t g = 0; g < num_groups; ++g) {
        const uint8_t* gs = src_bf16_bytes + (static_cast<size_t>(g) << 7);
        float32x4_t v[16];
        v[0] = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(gs + 0), 16));
        float32x4_t vmin = v[0];
        float32x4_t vmax = v[0];
        for (int i = 1; i < 16; ++i) {
            v[i] = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(gs + static_cast<size_t>(i) * 8u), 16));
            vmin = vminq_f32(vmin, v[i]);
            vmax = vmaxq_f32(vmax, v[i]);
        }
        const float w_min = vminvq_f32(vmin);
        const float w_max = vmaxvq_f32(vmax);
        const float raw_scale = (w_max - w_min) * (1.0f / 15.0f);

        const uint16_t s_bf16 = f32_to_bf16_rne(raw_scale);
        const uint16_t b_bf16 = f32_to_bf16_rne(w_min);
        dst_s[g] = s_bf16;
        dst_b[g] = b_bf16;

        const float sc = bf16_bits_to_f32(s_bf16);
        const float bi = bf16_bits_to_f32(b_bf16);
        uint32_t* gw = dst_w + (static_cast<size_t>(g) << 3);

        if (sc <= 0.0f) {
            for (int wi = 0; wi < 8; ++wi) gw[wi] = 0u;
            continue;
        }
        const float32x4_t vbi  = vdupq_n_f32(bi);
        const float32x4_t vinv = vdupq_n_f32(1.0f / sc);

        for (int wi = 0; wi < 8; ++wi) {
            float32x4_t f0 = vmaxq_f32(vzero, vminq_f32(vf15, vmulq_f32(vsubq_f32(v[wi * 2 + 0], vbi), vinv)));
            float32x4_t f1 = vmaxq_f32(vzero, vminq_f32(vf15, vmulq_f32(vsubq_f32(v[wi * 2 + 1], vbi), vinv)));
            uint32x4_t u0 = vcvtnq_u32_f32(f0);
            uint32x4_t u1 = vcvtnq_u32_f32(f1);
            uint32x4_t packed = vorrq_u32(vshlq_u32(u0, sh_lo), vshlq_u32(u1, sh_hi));
            gw[wi] = vaddvq_u32(packed);
        }
    }
#else
    for (uint32_t g = 0; g < num_groups; ++g) {
        const uint8_t* gs = src_bf16_bytes + (static_cast<size_t>(g) << 7);
        float vals[64];
        float w_min = bf16_bits_to_f32(load_u16_unaligned(gs));
        float w_max = w_min;
        for (int i = 0; i < 64; ++i) {
            float x = bf16_bits_to_f32(load_u16_unaligned(gs + static_cast<size_t>(i) * 2u));
            vals[i] = x;
            if (x < w_min) w_min = x;
            if (x > w_max) w_max = x;
        }
        const uint16_t s_bf16 = f32_to_bf16_rne((w_max - w_min) * (1.0f / 15.0f));
        const uint16_t b_bf16 = f32_to_bf16_rne(w_min);
        dst_s[g] = s_bf16;
        dst_b[g] = b_bf16;
        const float sc = bf16_bits_to_f32(s_bf16);
        const float bi = bf16_bits_to_f32(b_bf16);
        uint32_t* gw = dst_w + (static_cast<size_t>(g) << 3);
        if (sc <= 0.0f) {
            for (int wi = 0; wi < 8; ++wi) gw[wi] = 0u;
            continue;
        }
        const float inv_sc = 1.0f / sc;
        for (int wi = 0; wi < 8; ++wi) {
            uint32_t word = 0u;
            for (int ni = 0; ni < 8; ++ni) {
                float qf = std::clamp((vals[wi * 8 + ni] - bi) * inv_sc, 0.0f, 15.0f);
                uint32_t q = static_cast<uint32_t>(std::nearbyint(qf));
                word |= (q & 0xFu) << (ni * 4);
            }
            gw[wi] = word;
        }
    }
#endif
}

// Generic fallback for F16 / F32 source checkpoints (converts row to BF16/F32 and quantizes identically)
void quantize_generic_row_q4_g64(const uint8_t* src_row, DType dtype, uint32_t K,
                                 uint32_t* dst_w, uint16_t* dst_s, uint16_t* dst_b) {
    if (dtype == DType::BF16) {
        quantize_bf16_row_q4_g64(src_row, K, dst_w, dst_s, dst_b);
        return;
    }
    const uint32_t num_groups = K >> 6;
    float vals[64];
    for (uint32_t g = 0; g < num_groups; ++g) {
        const size_t base = static_cast<size_t>(g) << 6;
        float w_min = 1e30f, w_max = -1e30f;
        for (int i = 0; i < 64; ++i) {
            float x = 0.0f;
            if (dtype == DType::F16) {
                x = f16_bits_to_f32(load_u16_unaligned(src_row + (base + i) * 2u));
            } else {
                x = load_f32_unaligned(src_row + (base + i) * 4u);
            }
            vals[i] = x;
            if (x < w_min) w_min = x;
            if (x > w_max) w_max = x;
        }
        const uint16_t s_bf16 = f32_to_bf16_rne((w_max - w_min) * (1.0f / 15.0f));
        const uint16_t b_bf16 = f32_to_bf16_rne(w_min);
        dst_s[g] = s_bf16;
        dst_b[g] = b_bf16;
        const float sc = bf16_bits_to_f32(s_bf16);
        const float bi = bf16_bits_to_f32(b_bf16);
        uint32_t* gw = dst_w + (static_cast<size_t>(g) << 3);
        if (sc <= 0.0f) {
            for (int wi = 0; wi < 8; ++wi) gw[wi] = 0u;
            continue;
        }
        const float inv_sc = 1.0f / sc;
        for (int wi = 0; wi < 8; ++wi) {
            uint32_t word = 0u;
            for (int ni = 0; ni < 8; ++ni) {
                float qf = std::clamp((vals[wi * 8 + ni] - bi) * inv_sc, 0.0f, 15.0f);
                uint32_t q = static_cast<uint32_t>(std::nearbyint(qf));
                word |= (q & 0xFu) << (ni * 4);
            }
            gw[wi] = word;
        }
    }
}

void convert_tensor_to_f16(const uint8_t* src, DType dtype, size_t numel, uint16_t* dst_f16) {
    if (dtype == DType::F16) {
        std::memcpy(dst_f16, src, numel * sizeof(uint16_t));
        return;
    }
    if (dtype == DType::BF16) {
        size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        __fp16* d = reinterpret_cast<__fp16*>(dst_f16);
        for (; i + 4 <= numel; i += 4) {
            float32x4_t fv = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(src + i * 2u), 16));
            vst1_f16(d + i, vcvt_f16_f32(fv));
        }
#endif
        for (; i < numel; ++i) {
            dst_f16[i] = f32_to_f16_bits(bf16_bits_to_f32(load_u16_unaligned(src + i * 2u)));
        }
        return;
    }
    if (dtype == DType::F32) {
        size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        __fp16* d = reinterpret_cast<__fp16*>(dst_f16);
        for (; i + 4 <= numel; i += 4) {
            vst1_f16(d + i, vcvt_f16_f32(load_f32x4_unaligned(src + i * 4u)));
        }
#endif
        for (; i < numel; ++i) {
            dst_f16[i] = f32_to_f16_bits(load_f32_unaligned(src + i * 4u));
        }
        return;
    }
    throw std::runtime_error("DFlash: unsupported source dtype for FP16 conversion");
}

// Fast ARM64 NEON CandidateSelector helper (shared by FP16/FP32 sel_h and greedy/sampled paths)
void run_candidate_selector(
    const DFlashConfig& cfg,
    const uint8_t* pred_cb_bytes,
    const uint8_t* succ_cb_bytes,
    int anchor_token,
    const LogitCand* topk_cands,
    const uint16_t* sel_h_f16,
    const float* sel_h_f32,
    bool sampled,
    float temperature,
    uint64_t seed,
    int* out_draft_ids,
    DraftDist* out_draft_dists,
    int top_k = 0,
    float top_p = 1.0f
) {
    const int n_pos = cfg.block_size - 1;
    const int K = cfg.selector_top_k;
    const int R = cfg.selector_rank;
    const uint32_t V = static_cast<uint32_t>(cfg.vocab_size);
    if (n_pos <= 0 || K <= 0 || R <= 0 || !pred_cb_bytes || !succ_cb_bytes) return;

    alignas(16) float v_stack[512];
    std::vector<float> v_heap;
    float* v_buf = v_stack;
    if (R > 512) {
        v_heap.resize(static_cast<size_t>(R));
        v_buf = v_heap.data();
    }

    float scores[256];
    float probs[256];
    int order[256];
    const int K_clamped = std::min(K, 256);
    const size_t row_bytes = static_cast<size_t>(R) * sizeof(uint16_t);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);

    int pred = anchor_token;
    for (int p = 0; p < n_pos; ++p) {
        const uint32_t pred_u = (pred >= 0 && static_cast<uint32_t>(pred) < V)
            ? static_cast<uint32_t>(pred) : 0u;
        const uint8_t* pred_row = pred_cb_bytes + static_cast<size_t>(pred_u) * row_bytes;

        // 1. v[r] = pred_cb[pred, r] * sel_h[p, r]
        int r = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        if (sel_h_f16 != nullptr) {
            const __fp16* sh = reinterpret_cast<const __fp16*>(sel_h_f16 + static_cast<size_t>(p) * static_cast<size_t>(R));
            for (; r + 8 <= R; r += 8) {
                float32x4_t p0 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(pred_row + static_cast<size_t>(r + 0) * 2u), 16));
                float32x4_t p1 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(pred_row + static_cast<size_t>(r + 4) * 2u), 16));
                float32x4_t h0 = vcvt_f32_f16(vld1_f16(sh + r + 0));
                float32x4_t h1 = vcvt_f32_f16(vld1_f16(sh + r + 4));
                vst1q_f32(v_buf + r + 0, vmulq_f32(p0, h0));
                vst1q_f32(v_buf + r + 4, vmulq_f32(p1, h1));
            }
        } else {
            const float* sh = sel_h_f32 + static_cast<size_t>(p) * static_cast<size_t>(R);
            for (; r + 8 <= R; r += 8) {
                float32x4_t p0 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(pred_row + static_cast<size_t>(r + 0) * 2u), 16));
                float32x4_t p1 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(pred_row + static_cast<size_t>(r + 4) * 2u), 16));
                vst1q_f32(v_buf + r + 0, vmulq_f32(p0, vld1q_f32(sh + r + 0)));
                vst1q_f32(v_buf + r + 4, vmulq_f32(p1, vld1q_f32(sh + r + 4)));
            }
        }
#endif
        for (; r < R; ++r) {
            float sh_val = (sel_h_f16 != nullptr)
                ? f16_bits_to_f32(sel_h_f16[static_cast<size_t>(p) * static_cast<size_t>(R) + r])
                : sel_h_f32[static_cast<size_t>(p) * static_cast<size_t>(R) + r];
            v_buf[r] = bf16_bits_to_f32(load_u16_unaligned(pred_row + static_cast<size_t>(r) * 2u)) * sh_val;
        }

        // 2. score[k] = unary[p][k] + dot(v, succ_cb[cand[p][k]])
        //    Strictly filter invalid / sentinel (id >= V, e.g. 0xFFFFFFFFu) or non-finite unary logits!
        const LogitCand* row_cands = topk_cands + static_cast<size_t>(p) * static_cast<size_t>(K);
        for (int k = 0; k < K_clamped; ++k) {
            const uint32_t raw_id = row_cands[k].id;
            if (raw_id >= V || !std::isfinite(row_cands[k].val)) {
                scores[k] = -std::numeric_limits<float>::infinity();
                continue;
            }
            const uint8_t* succ_row = succ_cb_bytes + static_cast<size_t>(raw_id) * row_bytes;
            if (k + 1 < K_clamped && row_cands[k + 1].id < V) {
                const uint32_t next_cid = row_cands[k + 1].id;
                const char* pf = reinterpret_cast<const char*>(succ_cb_bytes + static_cast<size_t>(next_cid) * row_bytes);
                for (size_t off = 0; off < row_bytes; off += 128u) {
                    __builtin_prefetch(pf + off, 0, 1);
                }
            }
            float dot = 0.0f;
            int j = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            for (; j + 16 <= R; j += 16) {
                float32x4_t s0 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(succ_row + static_cast<size_t>(j + 0) * 2u), 16));
                float32x4_t s1 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(succ_row + static_cast<size_t>(j + 4) * 2u), 16));
                float32x4_t s2 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(succ_row + static_cast<size_t>(j + 8) * 2u), 16));
                float32x4_t s3 = vreinterpretq_f32_u32(vshll_n_u16(load_u16x4_unaligned(succ_row + static_cast<size_t>(j + 12) * 2u), 16));
                acc0 = vfmaq_f32(acc0, vld1q_f32(v_buf + j + 0),  s0);
                acc1 = vfmaq_f32(acc1, vld1q_f32(v_buf + j + 4),  s1);
                acc2 = vfmaq_f32(acc2, vld1q_f32(v_buf + j + 8),  s2);
                acc3 = vfmaq_f32(acc3, vld1q_f32(v_buf + j + 12), s3);
            }
            dot = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#endif
            for (; j < R; ++j) {
                dot += v_buf[j] * bf16_bits_to_f32(load_u16_unaligned(succ_row + static_cast<size_t>(j) * 2u));
            }
            scores[k] = row_cands[k].val + dot;
        }

        // 3. NaN/Inf/sentinel-safe selection of greedy or top_k/top_p/0.85xT-sharpened sampled token & exact DraftDist q(x)
        int n_valid = 0;
        int best = -1;
        float max_s = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < K_clamped; ++k) {
            if (row_cands[k].id >= V) continue;
            float sc = scores[k];
            if (!std::isfinite(sc)) continue;
            order[n_valid++] = k;
            if (best < 0 || sc > max_s || (sc == max_s && row_cands[k].id < row_cands[best].id)) {
                max_s = sc;
                best = k;
            }
        }

        if (n_valid == 0) {
            // All candidates were invalid/NaN/sentinel (0xFFFFFFFFu): clamp safely to token 0 (never -1!)
            pred = 0;
            out_draft_ids[p] = 0;
            if (out_draft_dists != nullptr) {
                DraftDist& dd = out_draft_dists[p];
                dd.count = 1u;
                dd.sampled_id = 0u;
                dd.sampled_q = 1.0f;
                dd.q_top1 = 1.0f;
                dd.alt_count = 0u;
                dd.cands[0] = LogitCand{1.0f, 0u};
            }
            continue;
        }

        if (!sampled || temperature <= 0.0f) {
            const uint32_t chosen_id = row_cands[best].id;
            pred = static_cast<int>(chosen_id);
            out_draft_ids[p] = pred;
            if (out_draft_dists != nullptr) {
                DraftDist& dd = out_draft_dists[p];
                dd.count = 1u;
                dd.sampled_id = chosen_id;
                dd.sampled_q = 1.0f;
                dd.q_top1 = 1.0f;
                dd.alt_count = 0u;
                dd.cands[0] = LogitCand{1.0f, chosen_id};
            }
        } else {
            // Sort valid candidates descending by combined score (deterministic tie-break by smaller token ID)
            std::sort(order, order + n_valid, [&](int a, int b) {
                if (scores[a] != scores[b]) return scores[a] > scores[b];
                return row_cands[a].id < row_cands[b].id;
            });

            // Apply top_k truncation to proposal distribution q(x) (100% lossless under Leviathan rejection sampling!)
            int keep = (top_k > 0 && top_k < n_valid) ? top_k : n_valid;

            // 0.85x temperature sharpening reduces tail probability waste on 4-bit draft logits
            const float eff_temp = std::max(temperature * 0.85f, 1e-4f);
            const float inv_t = 1.0f / eff_temp;
            const float top_s = scores[order[0]];
            float sum_q = 0.0f;
            for (int i = 0; i < keep; ++i) {
                float w = std::exp((scores[order[i]] - top_s) * inv_t);
                if (!std::isfinite(w)) w = 0.0f;
                probs[i] = w;
                sum_q += w;
            }
            if (!(sum_q > 0.0f) || !std::isfinite(sum_q)) {
                probs[0] = 1.0f;
                keep = 1;
            } else {
                const float inv_sum = 1.0f / sum_q;
                for (int i = 0; i < keep; ++i) probs[i] *= inv_sum;

                // Apply top_p nucleus truncation + exact renormalization
                if (top_p > 0.0f && top_p < 1.0f && keep > 1) {
                    float cum = 0.0f;
                    int keep_p = keep;
                    for (int i = 0; i < keep; ++i) {
                        cum += probs[i];
                        if (cum >= top_p) {
                            keep_p = i + 1;
                            break;
                        }
                    }
                    keep = std::max(1, keep_p);
                    float renorm = 0.0f;
                    for (int i = 0; i < keep; ++i) renorm += probs[i];
                    if (!(renorm > 0.0f) || !std::isfinite(renorm)) {
                        probs[0] = 1.0f;
                        keep = 1;
                    } else {
                        const float inv_r = 1.0f / renorm;
                        for (int i = 0; i < keep; ++i) probs[i] *= inv_r;
                    }
                }
            }

            const float u = udist(rng);
            float acc = 0.0f;
            int chosen_i = keep - 1;
            for (int i = 0; i < keep; ++i) {
                acc += probs[i];
                if (u <= acc) {
                    chosen_i = i;
                    break;
                }
            }
            const uint32_t chosen_id = row_cands[order[chosen_i]].id;
            pred = static_cast<int>(chosen_id);
            out_draft_ids[p] = pred;

            if (out_draft_dists != nullptr) {
                DraftDist& dd = out_draft_dists[p];
                dd.count = static_cast<uint32_t>(keep);
                dd.sampled_id = chosen_id;
                dd.sampled_q = probs[chosen_i];
                dd.q_top1 = probs[0];
                dd.alt_count = 0u;
                for (int i = 0; i < keep; ++i) {
                    dd.cands[i].val = probs[i];
                    dd.cands[i].id = row_cands[order[i]].id;
                }
            }
        }
    }
}

uint64_t fnv1a_update(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

constexpr size_t kDFlashQ4CacheHeaderBytes = 256u;
constexpr uint32_t kDFlashQ4CacheVersion = 1u;

struct DFlashQ4CacheHeader {
    char     magic[8];          // "MIDFQ4\0\1"
    uint32_t version;           // kDFlashQ4CacheVersion
    uint32_t header_bytes;      // kDFlashQ4CacheHeaderBytes (256)
    uint64_t fingerprint;       // DFlashEngine::fingerprint_
    uint64_t source_file_size;  // exact model.safetensors st.st_size
    int64_t  source_mtime_sec;  // st.st_mtimespec.tv_sec
    int64_t  source_mtime_nsec; // st.st_mtimespec.tv_nsec
    uint64_t arena_bytes;       // exact q4_cursor
    uint32_t num_q4_jobs;       // number of Q4 weight tensors
    uint32_t flags;             // 0
    uint64_t layout_hash;       // FNV-1a over all Q4Job (out_dim, in_dim, offsets, sizes)
    uint8_t  reserved[kDFlashQ4CacheHeaderBytes - 72u];
};
static_assert(sizeof(DFlashQ4CacheHeader) == kDFlashQ4CacheHeaderBytes,
              "DFlashQ4CacheHeader must be 256 bytes");

std::string dflash_q4_cache_dir() {
    const char* env_dir = std::getenv("MOMETAL_DFLASH_CACHE_DIR");
    if (env_dir && env_dir[0] != '\0') return std::string(env_dir);
    const char* home = std::getenv("HOME");
    return home ? (std::string(home) + "/.cache/mometal/dflash") : "/tmp/mometal-dflash";
}

std::string dflash_q4_cache_path(uint64_t fp) {
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%016llx.q4bin", static_cast<unsigned long long>(fp));
    return dflash_q4_cache_dir() + "/" + hex;
}

bool validate_q4_cache_header(const DFlashQ4CacheHeader& h,
                              uint64_t exp_fp, uint64_t exp_src_sz,
                              int64_t exp_mt_sec, int64_t exp_mt_nsec,
                              uint64_t exp_arena_bytes, uint32_t exp_n_jobs,
                              uint64_t exp_layout_hash) {
    static constexpr char kMagic[8] = {'M', 'I', 'D', 'F', 'Q', '4', '\0', '\1'};
    return std::memcmp(h.magic, kMagic, 8) == 0 &&
           h.version == kDFlashQ4CacheVersion &&
           h.header_bytes == static_cast<uint32_t>(kDFlashQ4CacheHeaderBytes) &&
           h.fingerprint == exp_fp &&
           h.source_file_size == exp_src_sz &&
           h.source_mtime_sec == exp_mt_sec &&
           h.source_mtime_nsec == exp_mt_nsec &&
           h.arena_bytes == exp_arena_bytes &&
           h.num_q4_jobs == exp_n_jobs &&
           h.flags == 0u &&
           h.layout_hash == exp_layout_hash;
}

} // namespace

DFlashEngine::DFlashEngine(MetalContext& ctx, const std::string& dflash_dir) {
    auto t0 = std::chrono::steady_clock::now();

    try {
        // 1. Parse config.json
        std::string cfg_path = dflash_dir + "/config.json";
        if (!fs::is_regular_file(cfg_path)) {
            throw std::runtime_error("DFlash config.json not found in: " + dflash_dir);
        }
        Json cfg = parse_json(read_file_to_string(cfg_path));
        cfg_.num_layers        = static_cast<int>(cfg["num_hidden_layers"].as_int(5));
        cfg_.hidden_size       = static_cast<int>(cfg["hidden_size"].as_int(5120));
        cfg_.intermediate_size = static_cast<int>(cfg["intermediate_size"].as_int(17408));
        cfg_.num_heads         = static_cast<int>(cfg["num_attention_heads"].as_int(32));
        cfg_.num_kv_heads      = static_cast<int>(cfg["num_key_value_heads"].as_int(8));
        cfg_.head_dim          = static_cast<int>(cfg["head_dim"].as_int(128));
        cfg_.vocab_size        = static_cast<int>(cfg["vocab_size"].as_int(248320));
        cfg_.sliding_window    = static_cast<int>(cfg["sliding_window"].as_int(2048));
        cfg_.rms_eps           = static_cast<float>(cfg["rms_norm_eps"].as_double(1e-6));
        if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object() &&
            cfg["rope_parameters"].contains("rope_theta")) {
            cfg_.rope_theta = static_cast<float>(cfg["rope_parameters"]["rope_theta"].as_double(10000000.0));
        } else {
            cfg_.rope_theta = static_cast<float>(cfg["rope_theta"].as_double(10000000.0));
        }

        const Json& dc = cfg.contains("dflash_config") ? cfg["dflash_config"] : cfg;
        cfg_.block_size       = static_cast<int>(dc["block_size"].as_int(8));
        cfg_.mask_token_id    = static_cast<int>(dc["mask_token_id"].as_int(248070));
        cfg_.conv_group_size  = static_cast<int>(dc["conv_group_size"].as_int(16));
        cfg_.conv_kernel_size = static_cast<int>(dc["conv_kernel_size"].as_int(2));
        cfg_.selector_rank    = static_cast<int>(dc["selector_rank"].as_int(256));
        cfg_.selector_top_k   = static_cast<int>(dc["selector_top_k"].as_int(16));
        cfg_.target_layer_ids.clear();
        if (dc.contains("target_layer_ids") && dc["target_layer_ids"].is_array()) {
            for (const auto& v : dc["target_layer_ids"].as_array()) {
                cfg_.target_layer_ids.push_back(static_cast<int>(v.as_int()));
            }
        }
        if (cfg_.target_layer_ids.empty()) {
            cfg_.target_layer_ids = {5, 19, 33, 47, 61};
        }

        if (cfg_.num_layers <= 0 || cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0 ||
            cfg_.num_heads <= 0 || cfg_.num_kv_heads <= 0 || cfg_.head_dim <= 0 ||
            cfg_.vocab_size <= 0 || cfg_.sliding_window <= 0 || cfg_.block_size < 2 ||
            cfg_.conv_group_size <= 0 || cfg_.conv_kernel_size <= 0 ||
            cfg_.selector_rank <= 0 || cfg_.selector_top_k <= 0 ||
            (cfg_.hidden_size % cfg_.conv_group_size) != 0) {
            throw std::runtime_error("Invalid DFlash config dimensions in: " + cfg_path);
        }

        int max_tid = 0;
        for (int tid : cfg_.target_layer_ids) {
            if (tid < 0 || tid > 4096) {
                throw std::runtime_error("Invalid DFlash target_layer_id in: " + cfg_path);
            }
            if (tid > max_tid) max_tid = tid;
        }
        target_slot_lut_.assign(static_cast<size_t>(max_tid + 1), -1);
        for (size_t s = 0; s < cfg_.target_layer_ids.size(); ++s) {
            int tid = cfg_.target_layer_ids[s];
            target_slot_lut_[static_cast<size_t>(tid)] = static_cast<int>(s);
        }

        // 2. Read-only page-aligned mmap of model.safetensors (enables zero-copy Metal sub-ranges for native Q4!)
        std::string st_path = dflash_dir + "/model.safetensors";
        if (!fs::is_regular_file(st_path)) {
            throw std::runtime_error("DFlash model.safetensors not found in: " + dflash_dir);
        }
        int fd = ::open(st_path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open DFlash weights: " + st_path);
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < 16) {
            ::close(fd);
            throw std::runtime_error("Invalid DFlash safetensors file: " + st_path);
        }
        const size_t ps = static_cast<size_t>(getpagesize());
        auto align_down_ps = [ps](size_t x) -> size_t { return x & ~(ps - 1u); };
        auto align_up_ps   = [ps](size_t x) -> size_t { return (x + ps - 1u) & ~(ps - 1u); };

        mmap_file_size_ = static_cast<size_t>(st.st_size);
        mmap_size_ = align_up_ps(mmap_file_size_);
        const int64_t src_mtime_sec  = static_cast<int64_t>(st.st_mtimespec.tv_sec);
        const int64_t src_mtime_nsec = static_cast<int64_t>(st.st_mtimespec.tv_nsec);

        mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        }
        ::close(fd);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            mmap_size_ = 0;
            mmap_file_size_ = 0;
            throw std::runtime_error("mmap failed for DFlash weights: " + st_path);
        }

        const uint8_t* base = reinterpret_cast<const uint8_t*>(mmap_ptr_);
        uint64_t header_len = 0;
        std::memcpy(&header_len, base, sizeof(uint64_t));
        if (mmap_file_size_ < 8u || header_len > mmap_file_size_ - 8u || header_len > (128ull << 20)) {
            throw std::runtime_error("Corrupt DFlash safetensors header: " + st_path);
        }
        const size_t data_base = 8u + static_cast<size_t>(header_len);
        const size_t data_avail = mmap_file_size_ - data_base;
        Json hdr = parse_json(std::string_view(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(header_len)));
        if (!hdr.is_object()) {
            throw std::runtime_error("Corrupt DFlash safetensors header JSON (not an object): " + st_path);
        }

        std::unordered_map<std::string, RawTensorMeta> meta;
        meta.reserve(hdr.as_object().size());
        for (const auto& [k, info] : hdr.as_object()) {
            if (k == "__metadata__") continue;
            if (!info.is_object() || !info.contains("dtype") || !info.contains("shape") || !info.contains("data_offsets") ||
                !info["dtype"].is_string() || !info["shape"].is_array() || !info["data_offsets"].is_array()) {
                throw std::runtime_error("Corrupt DFlash safetensors tensor entry: " + k);
            }
            RawTensorMeta rm;
            rm.dtype = parse_dtype(info["dtype"].as_string());
            for (const auto& d : info["shape"].as_array()) {
                int64_t dv = d.as_int();
                if (dv < 0) throw std::runtime_error("Negative shape dimension in DFlash tensor: " + k);
                rm.shape.push_back(dv);
            }
            const auto& offs = info["data_offsets"].as_array();
            if (offs.size() != 2) {
                throw std::runtime_error("Invalid data_offsets length in DFlash tensor: " + k);
            }
            int64_t s_i64 = offs[0].as_int();
            int64_t e_i64 = offs[1].as_int();
            if (s_i64 < 0 || e_i64 < s_i64) {
                throw std::runtime_error("Invalid data_offsets range in DFlash tensor: " + k);
            }
            size_t s = static_cast<size_t>(s_i64);
            size_t e = static_cast<size_t>(e_i64);
            if (s > data_avail || e > data_avail) {
                throw std::runtime_error("DFlash safetensors data_offsets out of bounds for tensor: " + k);
            }
            rm.byte_offset = data_base + s;
            rm.nbytes = e - s;
            meta.emplace(k, std::move(rm));
        }

        auto has_meta = [&](const std::string& key) -> bool {
            return meta.count(key) > 0 || meta.count("model." + key) > 0;
        };
        auto has_cb_meta = [&](const std::string& key) -> bool {
            return has_meta(key) || has_meta(key + ".weight");
        };
        if (!has_cb_meta("candidate_selector.predecessor_codebook") ||
            !has_cb_meta("candidate_selector.successor_codebook") ||
            !has_meta("candidate_selector.hidden_projection.weight") ||
            !has_meta("layers.0.attention_conv.base_kernel") ||
            !has_meta("layers.0.attention_conv.kernel_projection.weight")) {
            throw std::runtime_error(
                "Unsupported DFlash checkpoint in " + dflash_dir +
                ": missing DFlash 2 candidate_selector / attention_conv tensors "
                "(z-lab/dflash v1 checkpoints are not supported; expected a DFlash 2 checkpoint)");
        }

        auto find_meta = [&](const std::string& key) -> const RawTensorMeta& {
            auto it = meta.find(key);
            if (it != meta.end()) return it->second;
            auto it2 = meta.find("model." + key);
            if (it2 != meta.end()) return it2->second;
            throw std::runtime_error("Missing DFlash tensor: " + key);
        };
        auto find_cb_meta = [&](const std::string& key) -> const RawTensorMeta& {
            if (has_meta(key)) return find_meta(key);
            return find_meta(key + ".weight");
        };

        // 3. Bind zero-Metal-wiring CPU mmap BF16 pointers for CandidateSelector codebooks!
        const RawTensorMeta& m_pred = find_cb_meta("candidate_selector.predecessor_codebook");
        const RawTensorMeta& m_succ = find_cb_meta("candidate_selector.successor_codebook");
        if (m_pred.dtype != DType::BF16 || m_succ.dtype != DType::BF16) {
            throw std::runtime_error("DFlash codebooks expected BF16 dtype");
        }
        const size_t cb_v = static_cast<size_t>(cfg_.vocab_size);
        const size_t cb_r = static_cast<size_t>(cfg_.selector_rank);
        if (cb_r > 0 && cb_v > (std::numeric_limits<size_t>::max() / 2u) / cb_r) {
            throw std::runtime_error("DFlash codebook dimensions overflow size_t");
        }
        const size_t expected_cb_bytes = cb_v * cb_r * sizeof(uint16_t);
        auto check_cb = [&](const RawTensorMeta& m, const char* name) {
            if (m.shape.size() != 2 ||
                m.shape[0] != static_cast<int64_t>(cfg_.vocab_size) ||
                m.shape[1] != static_cast<int64_t>(cfg_.selector_rank) ||
                m.nbytes != expected_cb_bytes) {
                throw std::runtime_error(std::string("DFlash ") + name + " shape/nbytes mismatch with [vocab_size, selector_rank]");
            }
        };
        check_cb(m_pred, "predecessor_codebook");
        check_cb(m_succ, "successor_codebook");
        pred_cb_bytes_ = base + m_pred.byte_offset;
        succ_cb_bytes_ = base + m_succ.byte_offset;

        // Deterministic 64-bit checkpoint identity fingerprint (for MICACHE2 & .q4bin disk cache disambiguation)
        {
            uint64_t fp = 14695981039346656037ull;
            std::error_code ec;
            std::string canon = fs::weakly_canonical(dflash_dir, ec).string();
            if (canon.empty()) canon = dflash_dir;
            fp = fnv1a_update(fp, canon.data(), canon.size());
            fp = fnv1a_update(fp, &mmap_file_size_, sizeof(mmap_file_size_));
            fp = fnv1a_update(fp, &header_len, sizeof(header_len));
            fp = fnv1a_update(fp, base + 8, std::min<size_t>(static_cast<size_t>(header_len), 4096u));
            fp = fnv1a_update(fp, &cfg_.num_layers, sizeof(cfg_.num_layers));
            fp = fnv1a_update(fp, &cfg_.hidden_size, sizeof(cfg_.hidden_size));
            fp = fnv1a_update(fp, &cfg_.sliding_window, sizeof(cfg_.sliding_window));
            fp = fnv1a_update(fp, &cfg_.block_size, sizeof(cfg_.block_size));
            fp = fnv1a_update(fp, &cfg_.mask_token_id, sizeof(cfg_.mask_token_id));
            if (!cfg_.target_layer_ids.empty()) {
                fp = fnv1a_update(fp, cfg_.target_layer_ids.data(), cfg_.target_layer_ids.size() * sizeof(int));
            }
            fingerprint_ = fp;
        }

        // Kick off kernel sequential read-ahead on both BF16 codebooks (~242.5 MiB) IMMEDIATELY!
        // (Zero Metal wired RAM: pages stay clean file-backed OS page cache!)
        const size_t pred_pg_lo = align_down_ps(m_pred.byte_offset);
        const size_t pred_pg_hi = std::min(mmap_file_size_, align_up_ps(m_pred.byte_offset + m_pred.nbytes));
        const size_t succ_pg_lo = align_down_ps(m_succ.byte_offset);
        const size_t succ_pg_hi = std::min(mmap_file_size_, align_up_ps(m_succ.byte_offset + m_succ.nbytes));
        if (pred_pg_hi > pred_pg_lo) {
            ::madvise(const_cast<uint8_t*>(base + pred_pg_lo), pred_pg_hi - pred_pg_lo, MADV_WILLNEED);
        }
        if (succ_pg_hi > succ_pg_lo) {
            ::madvise(const_cast<uint8_t*>(base + succ_pg_lo), succ_pg_hi - succ_pg_lo, MADV_WILLNEED);
        }

        // 4. Plan small FP16 arena (~3.1 MiB) and Q4 weights (either native MLX Q4 MAP_SHARED or disk-cached .q4bin!)
        struct F16Job {
            const RawTensorMeta* src;
            TensorView* dst;
            size_t numel;
            size_t dst_off;
        };
        std::vector<F16Job> f16_jobs;
        std::vector<std::pair<size_t, size_t>> evict_ranges;
        size_t f16_cursor = 0;

        auto add_f16 = [&](const std::string& key, TensorView& dst) {
            const RawTensorMeta& rm = find_meta(key);
            if (rm.dtype != DType::BF16 && rm.dtype != DType::F16 && rm.dtype != DType::F32) {
                throw std::runtime_error("DFlash F16 tensor unsupported source dtype: " + key);
            }
            const size_t elem_sz = dtype_size(rm.dtype);
            size_t numel = 1;
            for (int64_t d : rm.shape) {
                if (d <= 0) throw std::runtime_error("DFlash F16 tensor has non-positive dim: " + key);
                size_t du = static_cast<size_t>(d);
                if (du > (std::numeric_limits<size_t>::max() / std::max<size_t>(elem_sz, 2u)) / numel) {
                    throw std::runtime_error("DFlash F16 tensor shape overflows size_t: " + key);
                }
                numel *= du;
            }
            if (rm.nbytes != numel * elem_sz) {
                throw std::runtime_error("DFlash F16 tensor nbytes mismatch with shape: " + key);
            }
            size_t bytes = numel * sizeof(uint16_t);
            F16Job j{};
            j.src = &rm;
            j.dst = &dst;
            j.numel = numel;
            j.dst_off = f16_cursor;
            f16_cursor += align256(bytes);
            f16_jobs.push_back(j);
            evict_ranges.emplace_back(rm.byte_offset, rm.byte_offset + rm.nbytes);
        };

        layers_.resize(static_cast<size_t>(cfg_.num_layers));
        add_f16("hidden_norm.weight", hidden_norm_);
        add_f16("norm.weight", norm_);
        add_f16("candidate_selector.hidden_projection.weight", sel_hidden_proj_);
        for (int l = 0; l < cfg_.num_layers; ++l) {
            std::string lp = "layers." + std::to_string(l) + ".";
            DFlashLayerWeights& lw = layers_[static_cast<size_t>(l)];
            add_f16(lp + "input_layernorm.weight", lw.input_layernorm);
            add_f16(lp + "post_attention_layernorm.weight", lw.post_attention_layernorm);
            add_f16(lp + "self_attn.q_norm.weight", lw.q_norm);
            add_f16(lp + "self_attn.k_norm.weight", lw.k_norm);
            add_f16(lp + "attention_conv.base_kernel", lw.attn_base_kernel);
            add_f16(lp + "mlp_conv.base_kernel", lw.mlp_base_kernel);
        }

        buf_f16_arena_ = ctx.alloc_shared(std::max<size_t>(f16_cursor, 256u));
        uint8_t* f16_base = reinterpret_cast<uint8_t*>([(id<MTLBuffer>)buf_f16_arena_ contents]);
        for (const auto& j : f16_jobs) {
            uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(f16_base + j.dst_off);
            convert_tensor_to_f16(base + j.src->byte_offset, j.src->dtype, j.numel, dst_ptr);
            TensorView& tv = *j.dst;
            tv.buffer   = buf_f16_arena_;
            tv.offset   = j.dst_off;
            tv.nbytes   = j.numel * sizeof(uint16_t);
            tv.host_ptr = reinterpret_cast<const uint8_t*>(dst_ptr);
            tv.dtype    = DType::F16;
            tv.shape    = j.src->shape;
        }

        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned n_thr = std::min<unsigned>(12u, std::max<unsigned>(1u, hw ? hw : 8u));
        const RawTensorMeta& m_fc_probe = find_meta("fc.weight");
        const bool is_native_q4 = (m_fc_probe.dtype == DType::U32 && has_meta("fc.scales") && has_meta("fc.biases"));

        size_t q4_total_bytes = 0;
        const char* q4_mode_label = "in-memory";

        if (is_native_q4) {
            // Validate quantization config if present (must be MLX 4-bit group-64 affine)
            const Json* qcfg = cfg.contains("quantization") ? &cfg["quantization"]
                             : (cfg.contains("quantization_config") ? &cfg["quantization_config"] : nullptr);
            if (qcfg && qcfg->is_object()) {
                if (qcfg->contains("bits") && qcfg->operator[]("bits").as_int(4) != 4) {
                    throw std::runtime_error("DFlash native quantized checkpoint requires bits=4");
                }
                if (qcfg->contains("group_size") && qcfg->operator[]("group_size").as_int(64) != 64) {
                    throw std::runtime_error("DFlash native quantized checkpoint requires group_size=64");
                }
                if (qcfg->contains("mode") && qcfg->operator[]("mode").is_string() &&
                    qcfg->operator[]("mode").as_string() != "affine") {
                    throw std::runtime_error("DFlash native quantized checkpoint requires affine mode");
                }
            }

            struct NativeQ4Job {
                const RawTensorMeta* w;
                const RawTensorMeta* s;
                const RawTensorMeta* b;
                DFlashQ4Weight* dst;
                uint32_t out_dim;
                uint32_t in_dim;
            };
            std::vector<NativeQ4Job> native_jobs;
            native_jobs.reserve(1u + static_cast<size_t>(cfg_.num_layers) * 9u);
            std::vector<std::pair<size_t, size_t>> q4_pg_ranges;
            q4_pg_ranges.reserve(native_jobs.capacity() * 3u);

            auto add_native_q4 = [&](const std::string& pfx, DFlashQ4Weight& dst) {
                const RawTensorMeta& mw = find_meta(pfx + ".weight");
                const RawTensorMeta& ms = find_meta(pfx + ".scales");
                const RawTensorMeta& mb = find_meta(pfx + ".biases");
                if (mw.dtype != DType::U32 || ms.dtype != DType::BF16 || mb.dtype != DType::BF16) {
                    throw std::runtime_error("DFlash native Q4 tensor dtype mismatch (expected U32/BF16/BF16): " + pfx);
                }
                if (mw.shape.size() != 2 || ms.shape.size() != 2 || mb.shape.size() != 2 ||
                    mw.shape[0] <= 0 || mw.shape[1] <= 0 ||
                    ms.shape != mb.shape || ms.shape[0] != mw.shape[0] || ms.shape[1] <= 0) {
                    throw std::runtime_error("DFlash native Q4 tensor shape mismatch: " + pfx);
                }
                const uint64_t N64 = static_cast<uint64_t>(mw.shape[0]);
                const uint64_t Kw64 = static_cast<uint64_t>(mw.shape[1]);
                const uint64_t K64 = Kw64 * 8ull;
                if (N64 > UINT32_MAX || K64 > UINT32_MAX || (K64 % 64ull) != 0ull ||
                    static_cast<uint64_t>(ms.shape[1]) != (K64 / 64ull)) {
                    throw std::runtime_error("DFlash native Q4 tensor dims incompatible with group_size=64: " + pfx);
                }
                const size_t exp_w_bytes  = static_cast<size_t>(N64) * static_cast<size_t>(Kw64) * sizeof(uint32_t);
                const size_t exp_sb_bytes = static_cast<size_t>(N64) * static_cast<size_t>(K64 / 64ull) * sizeof(uint16_t);
                if (mw.nbytes != exp_w_bytes || ms.nbytes != exp_sb_bytes || mb.nbytes != exp_sb_bytes) {
                    throw std::runtime_error("DFlash native Q4 tensor nbytes mismatch: " + pfx);
                }
                native_jobs.push_back(NativeQ4Job{
                    &mw, &ms, &mb, &dst, static_cast<uint32_t>(N64), static_cast<uint32_t>(K64)
                });
                q4_total_bytes += exp_w_bytes + 2u * exp_sb_bytes;
                for (const RawTensorMeta* rm : {&mw, &ms, &mb}) {
                    size_t p0 = align_down_ps(rm->byte_offset);
                    size_t p1 = std::min(mmap_size_, align_up_ps(rm->byte_offset + rm->nbytes));
                    if (p1 > p0) q4_pg_ranges.emplace_back(p0, p1);
                }
            };

            add_native_q4("fc", fc_);
            for (int l = 0; l < cfg_.num_layers; ++l) {
                std::string lp = "layers." + std::to_string(l) + ".";
                DFlashLayerWeights& lw = layers_[static_cast<size_t>(l)];
                add_native_q4(lp + "attention_conv.kernel_projection", lw.attn_kernel_proj);
                add_native_q4(lp + "mlp_conv.kernel_projection", lw.mlp_kernel_proj);
                add_native_q4(lp + "self_attn.q_proj", lw.q_proj);
                add_native_q4(lp + "self_attn.k_proj", lw.k_proj);
                add_native_q4(lp + "self_attn.v_proj", lw.v_proj);
                add_native_q4(lp + "self_attn.o_proj", lw.o_proj);
                add_native_q4(lp + "mlp.gate_proj", lw.gate_proj);
                add_native_q4(lp + "mlp.up_proj", lw.up_proj);
                add_native_q4(lp + "mlp.down_proj", lw.down_proj);
            }

            // Merge page-aligned Q4 sub-ranges (keeps 242.5 MiB codebooks strictly OUTSIDE Metal buffers!)
            std::sort(q4_pg_ranges.begin(), q4_pg_ranges.end());
            std::vector<std::pair<size_t, size_t>> merged;
            for (const auto& r : q4_pg_ranges) {
                if (merged.empty() || r.first > merged.back().second) {
                    merged.push_back(r);
                } else if (r.second > merged.back().second) {
                    merged.back().second = r.second;
                }
            }

            id<MTLDevice> dev = (id<MTLDevice>)ctx.device();
            q4_sub_bufs_.reserve(merged.size());
            for (const auto& [r_lo, r_hi] : merged) {
                const size_t r_len = r_hi - r_lo;
                void* r_ptr = const_cast<uint8_t*>(base + r_lo);
                ::madvise(r_ptr, r_len, MADV_WILLNEED);
                id<MTLBuffer> sub_buf = [dev newBufferWithBytesNoCopy:r_ptr
                                                               length:r_len
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
                if (!sub_buf) {
                    throw std::runtime_error("DFlash native Q4 newBufferWithBytesNoCopy failed");
                }
                q4_sub_bufs_.push_back(sub_buf);
            }

            auto bind_sub_tv = [&](const RawTensorMeta& rm, TensorView& tv) {
                size_t lo = 0, hi = merged.size();
                while (lo + 1u < hi) {
                    size_t mid = (lo + hi) >> 1;
                    if (merged[mid].first <= rm.byte_offset) lo = mid;
                    else hi = mid;
                }
                tv.buffer   = q4_sub_bufs_[lo];
                tv.offset   = rm.byte_offset - merged[lo].first;
                tv.nbytes   = rm.nbytes;
                tv.host_ptr = base + rm.byte_offset;
                tv.dtype    = rm.dtype;
                tv.shape    = rm.shape;
            };

            for (const auto& j : native_jobs) {
                DFlashQ4Weight& d = *j.dst;
                d.out_dim = j.out_dim;
                d.in_dim  = j.in_dim;
                bind_sub_tv(*j.w, d.w);
                bind_sub_tv(*j.s, d.s);
                bind_sub_tv(*j.b, d.b);
            }
            q4_mode_label = "native-Q4 MAP_SHARED";
        } else {
            // BF16 / F16 / F32 checkpoint -> 256-byte aligned Q4_AFFINE_G64 arena (guaranteed sh==0 fast path!)
            // Cached atomically to ~/.cache/mometal/dflash/<fp>.q4bin and MAP_SHARED zero-copy bound!
            struct Q4Job {
                const RawTensorMeta* src;
                DFlashQ4Weight* dst;
                uint32_t out_dim;
                uint32_t in_dim;
                size_t w_off;
                size_t s_off;
                size_t b_off;
                size_t w_bytes;
                size_t sb_bytes;
            };
            std::vector<Q4Job> q4_jobs;
            q4_jobs.reserve(1u + static_cast<size_t>(cfg_.num_layers) * 9u);
            size_t q4_cursor = 0;

            auto add_q4 = [&](const std::string& key, DFlashQ4Weight& dst) {
                const RawTensorMeta& rm = find_meta(key);
                if (rm.shape.size() != 2 || rm.shape[0] <= 0 || rm.shape[1] <= 0) {
                    throw std::runtime_error("DFlash Q4 tensor must be non-empty 2D: " + key);
                }
                if (rm.dtype != DType::BF16 && rm.dtype != DType::F16 && rm.dtype != DType::F32) {
                    throw std::runtime_error("DFlash Q4 tensor unsupported source dtype: " + key);
                }
                if (rm.shape[0] > UINT32_MAX || rm.shape[1] > UINT32_MAX) {
                    throw std::runtime_error("DFlash Q4 tensor dimensions overflow uint32: " + key);
                }
                uint32_t N = static_cast<uint32_t>(rm.shape[0]);
                uint32_t K = static_cast<uint32_t>(rm.shape[1]);
                if ((K % 64u) != 0u) {
                    throw std::runtime_error("DFlash Q4 tensor K not divisible by 64: " + key);
                }
                const size_t elem_sz = dtype_size(rm.dtype);
                if (static_cast<size_t>(N) > (std::numeric_limits<size_t>::max() / elem_sz) / static_cast<size_t>(K) ||
                    rm.nbytes != static_cast<size_t>(N) * static_cast<size_t>(K) * elem_sz) {
                    throw std::runtime_error("DFlash Q4 tensor nbytes mismatch with shape: " + key);
                }
                size_t w_bytes  = static_cast<size_t>(N) * (K / 8u) * sizeof(uint32_t);
                size_t sb_bytes = static_cast<size_t>(N) * (K / 64u) * sizeof(uint16_t);
                Q4Job j{};
                j.src = &rm;
                j.dst = &dst;
                j.out_dim = N;
                j.in_dim = K;
                j.w_off = q4_cursor;  q4_cursor += align256(w_bytes);
                j.s_off = q4_cursor;  q4_cursor += align256(sb_bytes);
                j.b_off = q4_cursor;  q4_cursor += align256(sb_bytes);
                j.w_bytes = w_bytes;
                j.sb_bytes = sb_bytes;
                q4_jobs.push_back(j);
                evict_ranges.emplace_back(rm.byte_offset, rm.byte_offset + rm.nbytes);
            };

            add_q4("fc.weight", fc_);
            for (int l = 0; l < cfg_.num_layers; ++l) {
                std::string lp = "layers." + std::to_string(l) + ".";
                DFlashLayerWeights& lw = layers_[static_cast<size_t>(l)];
                add_q4(lp + "attention_conv.kernel_projection.weight", lw.attn_kernel_proj);
                add_q4(lp + "mlp_conv.kernel_projection.weight", lw.mlp_kernel_proj);
                add_q4(lp + "self_attn.q_proj.weight", lw.q_proj);
                add_q4(lp + "self_attn.k_proj.weight", lw.k_proj);
                add_q4(lp + "self_attn.v_proj.weight", lw.v_proj);
                add_q4(lp + "self_attn.o_proj.weight", lw.o_proj);
                add_q4(lp + "mlp.gate_proj.weight", lw.gate_proj);
                add_q4(lp + "mlp.up_proj.weight", lw.up_proj);
                add_q4(lp + "mlp.down_proj.weight", lw.down_proj);
            }
            q4_total_bytes = q4_cursor;

            // Compute deterministic layout hash over all Q4Job descriptors
            uint64_t layout_hash = 14695981039346656037ull;
            for (const auto& j : q4_jobs) {
                layout_hash = fnv1a_update(layout_hash, &j.out_dim, sizeof(j.out_dim));
                layout_hash = fnv1a_update(layout_hash, &j.in_dim, sizeof(j.in_dim));
                layout_hash = fnv1a_update(layout_hash, &j.w_off, sizeof(j.w_off));
                layout_hash = fnv1a_update(layout_hash, &j.s_off, sizeof(j.s_off));
                layout_hash = fnv1a_update(layout_hash, &j.b_off, sizeof(j.b_off));
                layout_hash = fnv1a_update(layout_hash, &j.w_bytes, sizeof(j.w_bytes));
                layout_hash = fnv1a_update(layout_hash, &j.sb_bytes, sizeof(j.sb_bytes));
            }

            const size_t cache_file_bytes = kDFlashQ4CacheHeaderBytes + q4_cursor;
            const std::string cache_dir  = dflash_q4_cache_dir();
            const std::string cache_path = dflash_q4_cache_path(fingerprint_);
            id<MTLDevice> dev = (id<MTLDevice>)ctx.device();

            auto try_map_q4_cache = [&](const std::string& cpath) -> bool {
                int cfd = ::open(cpath.c_str(), O_RDONLY);
                if (cfd < 0) return false;
                struct stat cst{};
                if (::fstat(cfd, &cst) != 0 || static_cast<size_t>(cst.st_size) != cache_file_bytes) {
                    ::close(cfd);
                    return false;
                }
                DFlashQ4CacheHeader ch{};
                ssize_t nr = ::pread(cfd, &ch, sizeof(ch), 0);
                if (nr != static_cast<ssize_t>(sizeof(ch)) ||
                    !validate_q4_cache_header(ch, fingerprint_, static_cast<uint64_t>(mmap_file_size_),
                                              src_mtime_sec, src_mtime_nsec,
                                              static_cast<uint64_t>(q4_cursor),
                                              static_cast<uint32_t>(q4_jobs.size()),
                                              layout_hash)) {
                    ::close(cfd);
                    return false;
                }
                const size_t map_len = align_up_ps(cache_file_bytes);
                void* ptr = ::mmap(nullptr, map_len, PROT_READ, MAP_SHARED, cfd, 0);
                ::close(cfd);
                if (ptr == MAP_FAILED) return false;
                ::madvise(ptr, cache_file_bytes, MADV_WILLNEED);
                id<MTLBuffer> mbuf = [dev newBufferWithBytesNoCopy:ptr
                                                            length:map_len
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
                if (!mbuf) {
                    ::munmap(ptr, map_len);
                    return false;
                }
                q4_mmap_ptr_  = ptr;
                q4_mmap_size_ = map_len;
                buf_q4_arena_ = mbuf;
                return true;
            };

            auto run_parallel_quantize = [&](uint8_t* dst_arena_base) {
                struct RowSlice {
                    size_t job_idx;
                    uint32_t row0;
                    uint32_t row1;
                };
                constexpr uint32_t ROW_TILE = 128u;
                std::vector<RowSlice> slices;
                slices.reserve(4096);
                for (size_t ji = 0; ji < q4_jobs.size(); ++ji) {
                    const uint32_t N = q4_jobs[ji].out_dim;
                    for (uint32_t r0 = 0; r0 < N; r0 += ROW_TILE) {
                        slices.push_back(RowSlice{ji, r0, std::min(N, r0 + ROW_TILE)});
                    }
                }
                std::atomic<size_t> next_slice{0};
                std::vector<std::thread> workers;
                workers.reserve(n_thr);
                for (unsigned ti = 0; ti < n_thr; ++ti) {
                    workers.emplace_back([&]() {
                        while (true) {
                            size_t si = next_slice.fetch_add(1, std::memory_order_relaxed);
                            if (si >= slices.size()) break;
                            const RowSlice& sl = slices[si];
                            const Q4Job& j = q4_jobs[sl.job_idx];
                            const uint32_t K = j.in_dim;
                            const size_t src_row_bytes = static_cast<size_t>(K) * dtype_size(j.src->dtype);
                            const uint8_t* src_base = base + j.src->byte_offset;
                            uint32_t* w_base = reinterpret_cast<uint32_t*>(dst_arena_base + j.w_off);
                            uint16_t* s_base = reinterpret_cast<uint16_t*>(dst_arena_base + j.s_off);
                            uint16_t* b_base = reinterpret_cast<uint16_t*>(dst_arena_base + j.b_off);
                            const uint32_t w_stride = K >> 3;
                            const uint32_t g_stride = K >> 6;
                            for (uint32_t r = sl.row0; r < sl.row1; ++r) {
                                quantize_generic_row_q4_g64(
                                    src_base + static_cast<size_t>(r) * src_row_bytes,
                                    j.src->dtype,
                                    K,
                                    w_base + static_cast<size_t>(r) * w_stride,
                                    s_base + static_cast<size_t>(r) * g_stride,
                                    b_base + static_cast<size_t>(r) * g_stride
                                );
                            }
                        }
                    });
                }
                for (auto& th : workers) th.join();
            };

            size_t q4_data_offset = 0;
            bool mapped_cache = try_map_q4_cache(cache_path);
            if (mapped_cache) {
                q4_data_offset = kDFlashQ4CacheHeaderBytes;
                q4_mode_label  = "disk-cache MAP_SHARED hit";
            } else {
                // Cache miss: try quantizing directly into an atomic .q4bin.tmp.<pid> file (0 extra anonymous RAM!)
                bool built_cache = false;
                std::error_code ec_dir;
                fs::create_directories(cache_dir, ec_dir);
                if (!ec_dir) {
                    std::string tmp_path = cache_path + ".tmp." + std::to_string(::getpid());
                    int tfd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
                    if (tfd >= 0) {
                        bool wr_ok = (::ftruncate(tfd, static_cast<off_t>(cache_file_bytes)) == 0);
                        void* tptr = MAP_FAILED;
                        if (wr_ok) {
                            tptr = ::mmap(nullptr, cache_file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, tfd, 0);
                            if (tptr == MAP_FAILED) wr_ok = false;
                        }
                        if (wr_ok) {
                            // Zero-initialize header + alignment padding via OS sparse zero pages; quantize directly into [256..]
                            uint8_t* tbytes = reinterpret_cast<uint8_t*>(tptr);
                            run_parallel_quantize(tbytes + kDFlashQ4CacheHeaderBytes);

                            DFlashQ4CacheHeader ch{};
                            static constexpr char kMagic[8] = {'M', 'I', 'D', 'F', 'Q', '4', '\0', '\1'};
                            std::memcpy(ch.magic, kMagic, 8);
                            ch.version           = kDFlashQ4CacheVersion;
                            ch.header_bytes      = static_cast<uint32_t>(kDFlashQ4CacheHeaderBytes);
                            ch.fingerprint       = fingerprint_;
                            ch.source_file_size  = static_cast<uint64_t>(mmap_file_size_);
                            ch.source_mtime_sec  = src_mtime_sec;
                            ch.source_mtime_nsec = src_mtime_nsec;
                            ch.arena_bytes       = static_cast<uint64_t>(q4_cursor);
                            ch.num_q4_jobs       = static_cast<uint32_t>(q4_jobs.size());
                            ch.flags             = 0u;
                            ch.layout_hash       = layout_hash;
                            std::memcpy(tbytes, &ch, sizeof(ch));
                        }
                        if (tptr != MAP_FAILED) {
                            ::munmap(tptr, cache_file_bytes);
                        }
                        ::close(tfd);
                        if (wr_ok) {
                            std::error_code ec_rn;
                            fs::rename(tmp_path, cache_path, ec_rn);
                            if (!ec_rn && try_map_q4_cache(cache_path)) {
                                built_cache    = true;
                                q4_data_offset = kDFlashQ4CacheHeaderBytes;
                                q4_mode_label  = "disk-cache MAP_SHARED built";
                            }
                        }
                        if (!built_cache) {
                            std::error_code ec_rm;
                            fs::remove(tmp_path, ec_rm);
                        }
                    }
                }

                if (!built_cache) {
                    // Clean fallback if cache directory isn't writable or disk is full
                    buf_q4_arena_  = ctx.alloc_shared(std::max<size_t>(q4_cursor, 256u));
                    q4_data_offset = 0;
                    uint8_t* q4_dst = reinterpret_cast<uint8_t*>([(id<MTLBuffer>)buf_q4_arena_ contents]);
                    run_parallel_quantize(q4_dst);
                    q4_mode_label = "in-memory alloc_shared";
                }

                // Drop quantized BF16 source pages ONLY (strictly excluding codebook pages!)
                for (const auto& [lo_off, hi_off] : evict_ranges) {
                    size_t p_lo = align_up_ps(lo_off);
                    size_t p_hi = align_down_ps(hi_off);
                    if (p_hi <= p_lo) continue;
                    if (p_lo < pred_pg_hi && p_hi > pred_pg_lo) {
                        if (p_lo < pred_pg_lo) ::madvise(const_cast<uint8_t*>(base + p_lo), pred_pg_lo - p_lo, MADV_DONTNEED);
                        if (p_hi > pred_pg_hi) ::madvise(const_cast<uint8_t*>(base + pred_pg_hi), p_hi - pred_pg_hi, MADV_DONTNEED);
                        continue;
                    }
                    if (p_lo < succ_pg_hi && p_hi > succ_pg_lo) {
                        if (p_lo < succ_pg_lo) ::madvise(const_cast<uint8_t*>(base + p_lo), succ_pg_lo - p_lo, MADV_DONTNEED);
                        if (p_hi > succ_pg_hi) ::madvise(const_cast<uint8_t*>(base + succ_pg_hi), p_hi - succ_pg_hi, MADV_DONTNEED);
                        continue;
                    }
                    ::madvise(const_cast<uint8_t*>(base + p_lo), p_hi - p_lo, MADV_DONTNEED);
                }
            }

            const uint8_t* q4_buf_base = reinterpret_cast<const uint8_t*>([(id<MTLBuffer>)buf_q4_arena_ contents]);
            for (const auto& j : q4_jobs) {
                DFlashQ4Weight& d = *j.dst;
                d.out_dim = j.out_dim;
                d.in_dim  = j.in_dim;

                d.w.buffer   = buf_q4_arena_;
                d.w.offset   = q4_data_offset + j.w_off;
                d.w.nbytes   = j.w_bytes;
                d.w.host_ptr = q4_buf_base + d.w.offset;
                d.w.dtype    = DType::U32;
                d.w.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 8u)};

                d.s.buffer   = buf_q4_arena_;
                d.s.offset   = q4_data_offset + j.s_off;
                d.s.nbytes   = j.sb_bytes;
                d.s.host_ptr = q4_buf_base + d.s.offset;
                d.s.dtype    = DType::BF16;
                d.s.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 64u)};

                d.b.buffer   = buf_q4_arena_;
                d.b.offset   = q4_data_offset + j.b_off;
                d.b.nbytes   = j.sb_bytes;
                d.b.host_ptr = q4_buf_base + d.b.offset;
                d.b.dtype    = DType::BF16;
                d.b.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 64u)};
            }
        }

        gpu_weight_bytes_ = q4_total_bytes + f16_cursor;

        // 5. Parallel 1-byte-per-page PTE warmup over pred_cb + succ_cb (0 B Metal wired RAM, race-free!)
        {
            const std::pair<size_t, size_t> cb_ranges[2] = {
                {pred_pg_lo, pred_pg_hi},
                {succ_pg_lo, succ_pg_hi}
            };
            const volatile uint8_t* vbase = base;
            std::vector<std::thread> warmers;
            warmers.reserve(n_thr);
            for (unsigned ti = 0; ti < n_thr; ++ti) {
                warmers.emplace_back([&, ti]() {
                    for (const auto& [r_lo, r_hi] : cb_ranges) {
                        if (r_hi <= r_lo) continue;
                        const size_t n_pages = (r_hi - r_lo + ps - 1u) / ps;
                        const size_t p0 = (n_pages * ti) / n_thr;
                        const size_t p1 = (n_pages * (ti + 1u)) / n_thr;
                        for (size_t pg = p0; pg < p1; ++pg) {
                            const size_t off = r_lo + pg * ps;
                            if (off < r_hi) (void)vbase[off];
                        }
                    }
                });
            }
            for (auto& th : warmers) th.join();
        }

        ready_ = true;
        load_time_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
            "[dflash] Loaded %d-layer DFlash2 drafter (block=%d, win=%d, H=%d, Hq=%d/%d, D=%d, rank=%d, topk=%d) | "
            "Q4=%.1f MiB (%s) + F16=%.1f MiB (codebooks CPU-mmap 0B Metal) in %.1f ms (%u threads)\n",
            cfg_.num_layers, cfg_.block_size, cfg_.sliding_window, cfg_.hidden_size,
            cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, cfg_.selector_rank, cfg_.selector_top_k,
            q4_total_bytes / (1024.0 * 1024.0), q4_mode_label, f16_cursor / (1024.0 * 1024.0), load_time_ms_, n_thr);
    } catch (...) {
        // Release ALL Metal buffers BEFORE unmapping backing mmaps!
        buf_q4_arena_ = nil;
        q4_sub_bufs_.clear();
        buf_f16_arena_ = nil;
        if (q4_mmap_ptr_ && q4_mmap_ptr_ != MAP_FAILED) {
            ::munmap(q4_mmap_ptr_, q4_mmap_size_);
        }
        q4_mmap_ptr_ = nullptr;
        q4_mmap_size_ = 0;
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_ = nullptr;
        mmap_size_ = 0;
        mmap_file_size_ = 0;
        pred_cb_bytes_ = nullptr;
        succ_cb_bytes_ = nullptr;
        ready_ = false;
        throw;
    }
}

DFlashEngine::~DFlashEngine() {
    // Release ALL Metal buffers BEFORE unmapping backing mmaps!
    buf_q4_arena_ = nil;
    q4_sub_bufs_.clear();
    buf_f16_arena_ = nil;
    if (q4_mmap_ptr_ && q4_mmap_ptr_ != MAP_FAILED) {
        ::munmap(q4_mmap_ptr_, q4_mmap_size_);
    }
    q4_mmap_ptr_ = nullptr;
    q4_mmap_size_ = 0;
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        ::munmap(mmap_ptr_, mmap_size_);
    }
    mmap_ptr_ = nullptr;
    mmap_size_ = 0;
    mmap_file_size_ = 0;
    pred_cb_bytes_ = nullptr;
    succ_cb_bytes_ = nullptr;
}

int DFlashEngine::target_capture_slot(int target_layer_idx) const {
    if (target_layer_idx < 0 || static_cast<size_t>(target_layer_idx) >= target_slot_lut_.size()) {
        return -1;
    }
    return target_slot_lut_[static_cast<size_t>(target_layer_idx)];
}

void DFlashEngine::select_greedy(int anchor_token,
                                 const LogitCand* topk_cands,
                                 const uint16_t* sel_h_f16,
                                 int* out_draft_ids) const {
    run_candidate_selector(cfg_, pred_cb_bytes_, succ_cb_bytes_, anchor_token,
                           topk_cands, sel_h_f16, nullptr, false, 0.0f, 0ull,
                           out_draft_ids, nullptr, 0, 1.0f);
}

void DFlashEngine::select_sampled(int anchor_token,
                                  const LogitCand* topk_cands,
                                  const uint16_t* sel_h_f16,
                                  float temperature,
                                  uint64_t seed,
                                  int* out_draft_ids,
                                  DraftDist* out_draft_dists,
                                  int top_k,
                                  float top_p) const {
    run_candidate_selector(cfg_, pred_cb_bytes_, succ_cb_bytes_, anchor_token,
                           topk_cands, sel_h_f16, nullptr, true, temperature, seed,
                           out_draft_ids, out_draft_dists, top_k, top_p);
}

void DFlashEngine::select_greedy_f32(int anchor_token,
                                     const LogitCand* topk_cands,
                                     const float* sel_h_f32,
                                     int* out_draft_ids) const {
    run_candidate_selector(cfg_, pred_cb_bytes_, succ_cb_bytes_, anchor_token,
                           topk_cands, nullptr, sel_h_f32, false, 0.0f, 0ull,
                           out_draft_ids, nullptr, 0, 1.0f);
}

void DFlashEngine::select_sampled_f32(int anchor_token,
                                      const LogitCand* topk_cands,
                                      const float* sel_h_f32,
                                      float temperature,
                                      uint64_t seed,
                                      int* out_draft_ids,
                                      DraftDist* out_draft_dists,
                                      int top_k,
                                      float top_p) const {
    run_candidate_selector(cfg_, pred_cb_bytes_, succ_cb_bytes_, anchor_token,
                           topk_cands, nullptr, sel_h_f32, true, temperature, seed,
                           out_draft_ids, out_draft_dists, top_k, top_p);
}

} // namespace mi
