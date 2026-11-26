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

struct RawTensorMeta {
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    size_t byte_offset = 0; // offset from mmap_ptr_
    size_t nbytes = 0;
};

// Fast ARM64 NEON quantization of one BF16 row [K] -> MLX Q4_AFFINE_G64 (w [K/8] u32, s [K/64] bf16, b [K/64] bf16).
// Quantizes against exact rounded BF16 scale & bias so dequantization matches common.metal bit-for-bit!
void quantize_bf16_row_q4_g64(const uint16_t* src_bf16, uint32_t K,
                              uint32_t* dst_w, uint16_t* dst_s, uint16_t* dst_b) {
    const uint32_t num_groups = K >> 6; // K / 64
#if defined(__aarch64__) || defined(__ARM_NEON)
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vf15  = vdupq_n_f32(15.0f);
    const int32x4_t sh_lo   = {0, 4, 8, 12};
    const int32x4_t sh_hi   = {16, 20, 24, 28};

    for (uint32_t g = 0; g < num_groups; ++g) {
        const uint16_t* gs = src_bf16 + (static_cast<size_t>(g) << 6);
        float32x4_t v[16];
        v[0] = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(gs + 0), 16));
        float32x4_t vmin = v[0];
        float32x4_t vmax = v[0];
        for (int i = 1; i < 16; ++i) {
            v[i] = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(gs + i * 4), 16));
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
        const uint16_t* gs = src_bf16 + (static_cast<size_t>(g) << 6);
        float vals[64];
        float w_min = bf16_bits_to_f32(gs[0]);
        float w_max = w_min;
        for (int i = 0; i < 64; ++i) {
            float x = bf16_bits_to_f32(gs[i]);
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
        quantize_bf16_row_q4_g64(reinterpret_cast<const uint16_t*>(src_row), K, dst_w, dst_s, dst_b);
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
                uint16_t bits;
                std::memcpy(&bits, src_row + (base + i) * 2u, 2);
                x = f16_bits_to_f32(bits);
            } else {
                std::memcpy(&x, src_row + (base + i) * 4u, 4);
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
        const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
        size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        __fp16* d = reinterpret_cast<__fp16*>(dst_f16);
        for (; i + 4 <= numel; i += 4) {
            float32x4_t fv = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(s + i), 16));
            vst1_f16(d + i, vcvt_f16_f32(fv));
        }
#endif
        for (; i < numel; ++i) {
            dst_f16[i] = f32_to_f16_bits(bf16_bits_to_f32(s[i]));
        }
        return;
    }
    if (dtype == DType::F32) {
        const float* s = reinterpret_cast<const float*>(src);
        size_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        __fp16* d = reinterpret_cast<__fp16*>(dst_f16);
        for (; i + 4 <= numel; i += 4) {
            vst1_f16(d + i, vcvt_f16_f32(vld1q_f32(s + i)));
        }
#endif
        for (; i < numel; ++i) {
            dst_f16[i] = f32_to_f16_bits(s[i]);
        }
        return;
    }
    throw std::runtime_error("DFlash: unsupported source dtype for FP16 conversion");
}

// Fast ARM64 NEON CandidateSelector helper (shared by FP16/FP32 sel_h and greedy/sampled paths)
void run_candidate_selector(
    const DFlashConfig& cfg,
    const uint16_t* pred_cb_bf16,
    const uint16_t* succ_cb_bf16,
    int anchor_token,
    const LogitCand* topk_cands,
    const uint16_t* sel_h_f16,
    const float* sel_h_f32,
    bool sampled,
    float temperature,
    uint64_t seed,
    int* out_draft_ids,
    DraftDist* out_draft_dists
) {
    const int n_pos = cfg.block_size - 1;
    const int K = cfg.selector_top_k;
    const int R = cfg.selector_rank;
    const uint32_t V = static_cast<uint32_t>(cfg.vocab_size);
    if (n_pos <= 0 || K <= 0 || R <= 0) return;

    alignas(16) float v_stack[512];
    std::vector<float> v_heap;
    float* v_buf = v_stack;
    if (R > 512) {
        v_heap.resize(static_cast<size_t>(R));
        v_buf = v_heap.data();
    }

    float scores[256];
    float probs[256];
    const int K_clamped = std::min(K, 256);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);

    int pred = anchor_token;
    for (int p = 0; p < n_pos; ++p) {
        const uint32_t pred_u = (pred >= 0 && static_cast<uint32_t>(pred) < V)
            ? static_cast<uint32_t>(pred) : 0u;
        const uint16_t* pred_row = pred_cb_bf16 + static_cast<size_t>(pred_u) * static_cast<size_t>(R);

        // 1. v[r] = pred_cb[pred, r] * sel_h[p, r]
        int r = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        if (sel_h_f16 != nullptr) {
            const __fp16* sh = reinterpret_cast<const __fp16*>(sel_h_f16 + static_cast<size_t>(p) * static_cast<size_t>(R));
            for (; r + 8 <= R; r += 8) {
                float32x4_t p0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pred_row + r + 0), 16));
                float32x4_t p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pred_row + r + 4), 16));
                float32x4_t h0 = vcvt_f32_f16(vld1_f16(sh + r + 0));
                float32x4_t h1 = vcvt_f32_f16(vld1_f16(sh + r + 4));
                vst1q_f32(v_buf + r + 0, vmulq_f32(p0, h0));
                vst1q_f32(v_buf + r + 4, vmulq_f32(p1, h1));
            }
        } else {
            const float* sh = sel_h_f32 + static_cast<size_t>(p) * static_cast<size_t>(R);
            for (; r + 8 <= R; r += 8) {
                float32x4_t p0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pred_row + r + 0), 16));
                float32x4_t p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pred_row + r + 4), 16));
                vst1q_f32(v_buf + r + 0, vmulq_f32(p0, vld1q_f32(sh + r + 0)));
                vst1q_f32(v_buf + r + 4, vmulq_f32(p1, vld1q_f32(sh + r + 4)));
            }
        }
#endif
        for (; r < R; ++r) {
            float sh_val = (sel_h_f16 != nullptr)
                ? f16_bits_to_f32(sel_h_f16[static_cast<size_t>(p) * static_cast<size_t>(R) + r])
                : sel_h_f32[static_cast<size_t>(p) * static_cast<size_t>(R) + r];
            v_buf[r] = bf16_bits_to_f32(pred_row[r]) * sh_val;
        }

        // 2. score[k] = unary[p][k] + dot(v, succ_cb[cand[p][k]])
        const LogitCand* row_cands = topk_cands + static_cast<size_t>(p) * static_cast<size_t>(K);
        for (int k = 0; k < K_clamped; ++k) {
            const uint32_t cid = (row_cands[k].id < V) ? row_cands[k].id : 0u;
            const uint16_t* succ_row = succ_cb_bf16 + static_cast<size_t>(cid) * static_cast<size_t>(R);
            if (k + 1 < K_clamped) {
                const uint32_t next_cid = (row_cands[k + 1].id < V) ? row_cands[k + 1].id : 0u;
                const char* pf = reinterpret_cast<const char*>(succ_cb_bf16 + static_cast<size_t>(next_cid) * static_cast<size_t>(R));
                const size_t row_bytes = static_cast<size_t>(R) * sizeof(uint16_t);
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
                float32x4_t s0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(succ_row + j + 0), 16));
                float32x4_t s1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(succ_row + j + 4), 16));
                float32x4_t s2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(succ_row + j + 8), 16));
                float32x4_t s3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(succ_row + j + 12), 16));
                acc0 = vfmaq_f32(acc0, vld1q_f32(v_buf + j + 0),  s0);
                acc1 = vfmaq_f32(acc1, vld1q_f32(v_buf + j + 4),  s1);
                acc2 = vfmaq_f32(acc2, vld1q_f32(v_buf + j + 8),  s2);
                acc3 = vfmaq_f32(acc3, vld1q_f32(v_buf + j + 12), s3);
            }
            dot = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#endif
            for (; j < R; ++j) {
                dot += v_buf[j] * bf16_bits_to_f32(succ_row[j]);
            }
            scores[k] = row_cands[k].val + dot;
        }

        // 3. Select greedy or sampled token & optionally populate exact DraftDist q(x)
        int best = 0;
        float max_s = scores[0];
        for (int k = 1; k < K_clamped; ++k) {
            if (scores[k] > max_s) {
                max_s = scores[k];
                best = k;
            }
        }

        if (!sampled || temperature <= 0.0f) {
            pred = static_cast<int>(row_cands[best].id);
            out_draft_ids[p] = pred;
            if (out_draft_dists != nullptr) {
                DraftDist& dd = out_draft_dists[p];
                dd.count = static_cast<uint32_t>(K_clamped);
                dd.sampled_id = static_cast<uint32_t>(pred);
                dd.sampled_q = 1.0f;
                dd.q_top1 = 1.0f;
                dd.alt_count = 0u;
                for (int k = 0; k < K_clamped; ++k) {
                    dd.cands[k].val = (k == best) ? 1.0f : 0.0f;
                    dd.cands[k].id = row_cands[k].id;
                }
            }
        } else {
            const float inv_t = 1.0f / std::max(temperature, 1e-4f);
            float sum_q = 0.0f;
            for (int k = 0; k < K_clamped; ++k) {
                probs[k] = std::exp((scores[k] - max_s) * inv_t);
                sum_q += probs[k];
            }
            const float inv_sum = (sum_q > 0.0f) ? (1.0f / sum_q) : 1.0f;
            float max_q = 0.0f;
            for (int k = 0; k < K_clamped; ++k) {
                probs[k] *= inv_sum;
                if (probs[k] > max_q) max_q = probs[k];
            }

            const float u = udist(rng);
            float acc = 0.0f;
            int chosen = K_clamped - 1;
            for (int k = 0; k < K_clamped; ++k) {
                acc += probs[k];
                if (u <= acc) {
                    chosen = k;
                    break;
                }
            }
            pred = static_cast<int>(row_cands[chosen].id);
            out_draft_ids[p] = pred;

            if (out_draft_dists != nullptr) {
                DraftDist& dd = out_draft_dists[p];
                dd.count = static_cast<uint32_t>(K_clamped);
                dd.sampled_id = static_cast<uint32_t>(pred);
                dd.sampled_q = probs[chosen];
                dd.q_top1 = max_q;
                dd.alt_count = 0u;
                for (int k = 0; k < K_clamped; ++k) {
                    dd.cands[k].val = probs[k];
                    dd.cands[k].id = row_cands[k].id;
                }
            }
        }
    }
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

        int max_tid = 0;
        for (int tid : cfg_.target_layer_ids) {
            if (tid > max_tid) max_tid = tid;
        }
        target_slot_lut_.assign(static_cast<size_t>(max_tid + 1), -1);
        for (size_t s = 0; s < cfg_.target_layer_ids.size(); ++s) {
            int tid = cfg_.target_layer_ids[s];
            if (tid >= 0) target_slot_lut_[static_cast<size_t>(tid)] = static_cast<int>(s);
        }

        // 2. Read-only mmap of model.safetensors
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
        mmap_size_ = static_cast<size_t>(st.st_size);
        mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd, 0);
        }
        ::close(fd);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            throw std::runtime_error("mmap failed for DFlash weights: " + st_path);
        }

        const uint8_t* base = reinterpret_cast<const uint8_t*>(mmap_ptr_);
        uint64_t header_len = 0;
        std::memcpy(&header_len, base, sizeof(uint64_t));
        if (8u + header_len > mmap_size_) {
            throw std::runtime_error("Corrupt DFlash safetensors header: " + st_path);
        }
        const size_t data_base = 8u + static_cast<size_t>(header_len);
        Json hdr = parse_json(std::string_view(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(header_len)));

        std::unordered_map<std::string, RawTensorMeta> meta;
        meta.reserve(hdr.as_object().size());
        for (const auto& [k, info] : hdr.as_object()) {
            if (k == "__metadata__") continue;
            RawTensorMeta rm;
            rm.dtype = parse_dtype(info["dtype"].as_string());
            for (const auto& d : info["shape"].as_array()) rm.shape.push_back(d.as_int());
            const auto& offs = info["data_offsets"].as_array();
            size_t s = static_cast<size_t>(offs[0].as_int());
            size_t e = static_cast<size_t>(offs[1].as_int());
            rm.byte_offset = data_base + s;
            rm.nbytes = e - s;
            meta.emplace(k, std::move(rm));
        }

        auto find_meta = [&](const std::string& key) -> const RawTensorMeta& {
            auto it = meta.find(key);
            if (it != meta.end()) return it->second;
            auto it2 = meta.find("model." + key);
            if (it2 != meta.end()) return it2->second;
            throw std::runtime_error("Missing DFlash tensor: " + key);
        };

        // 3. Bind zero-Metal-wiring CPU mmap BF16 pointers for CandidateSelector codebooks!
        const RawTensorMeta& m_pred = find_meta("candidate_selector.predecessor_codebook");
        const RawTensorMeta& m_succ = find_meta("candidate_selector.successor_codebook");
        if (m_pred.dtype != DType::BF16 || m_succ.dtype != DType::BF16) {
            throw std::runtime_error("DFlash codebooks expected BF16 dtype");
        }
        pred_cb_bf16_ = reinterpret_cast<const uint16_t*>(base + m_pred.byte_offset);
        succ_cb_bf16_ = reinterpret_cast<const uint16_t*>(base + m_succ.byte_offset);

        // Kick off kernel sequential read-ahead on both BF16 codebooks (~242.5 MiB) IMMEDIATELY
        // so SSD -> OS page-cache I/O overlaps 100% with the 12-thread Q4 quantization pass below!
        // (Zero Metal wired RAM: pages stay clean file-backed OS page cache!)
        const size_t ps = static_cast<size_t>(getpagesize());
        auto align_down_ps = [ps](size_t x) -> size_t { return x & ~(ps - 1u); };
        auto align_up_ps   = [ps](size_t x) -> size_t { return (x + ps - 1u) & ~(ps - 1u); };
        const size_t pred_pg_lo = align_down_ps(m_pred.byte_offset);
        const size_t pred_pg_hi = std::min(mmap_size_, align_up_ps(m_pred.byte_offset + m_pred.nbytes));
        const size_t succ_pg_lo = align_down_ps(m_succ.byte_offset);
        const size_t succ_pg_hi = std::min(mmap_size_, align_up_ps(m_succ.byte_offset + m_succ.nbytes));
        if (pred_pg_hi > pred_pg_lo) {
            ::madvise(const_cast<uint8_t*>(base + pred_pg_lo), pred_pg_hi - pred_pg_lo, MADV_WILLNEED);
        }
        if (succ_pg_hi > succ_pg_lo) {
            ::madvise(const_cast<uint8_t*>(base + succ_pg_lo), succ_pg_hi - succ_pg_lo, MADV_WILLNEED);
        }

        // 4. Plan contiguous Q4_AFFINE_G64 and FP16 Metal arenas (256-byte aligned views)
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
        struct F16Job {
            const RawTensorMeta* src;
            TensorView* dst;
            size_t numel;
            size_t dst_off;
        };

        std::vector<Q4Job> q4_jobs;
        std::vector<F16Job> f16_jobs;
        std::vector<std::pair<size_t, size_t>> evict_ranges; // BF16 mmap ranges to MADV_DONTNEED after load
        size_t q4_cursor = 0;
        size_t f16_cursor = 0;

        auto add_q4 = [&](const std::string& key, DFlashQ4Weight& dst) {
            const RawTensorMeta& rm = find_meta(key);
            if (rm.shape.size() != 2) {
                throw std::runtime_error("DFlash Q4 tensor must be 2D: " + key);
            }
            uint32_t N = static_cast<uint32_t>(rm.shape[0]);
            uint32_t K = static_cast<uint32_t>(rm.shape[1]);
            if ((K % 64u) != 0u) {
                throw std::runtime_error("DFlash Q4 tensor K not divisible by 64: " + key);
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

        auto add_f16 = [&](const std::string& key, TensorView& dst) {
            const RawTensorMeta& rm = find_meta(key);
            size_t numel = 1;
            for (int64_t d : rm.shape) numel *= static_cast<size_t>(d);
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
        add_q4("fc.weight", fc_);
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

        buf_q4_arena_  = ctx.alloc_shared(std::max<size_t>(q4_cursor, 256u));
        buf_f16_arena_ = ctx.alloc_shared(std::max<size_t>(f16_cursor, 256u));
        gpu_weight_bytes_ = q4_cursor + f16_cursor;

        uint8_t* q4_base  = reinterpret_cast<uint8_t*>([(id<MTLBuffer>)buf_q4_arena_ contents]);
        uint8_t* f16_base = reinterpret_cast<uint8_t*>([(id<MTLBuffer>)buf_f16_arena_ contents]);

        // Populate all TensorView descriptors
        for (const auto& j : q4_jobs) {
            DFlashQ4Weight& d = *j.dst;
            d.out_dim = j.out_dim;
            d.in_dim  = j.in_dim;

            d.w.buffer   = buf_q4_arena_;
            d.w.offset   = j.w_off;
            d.w.nbytes   = j.w_bytes;
            d.w.host_ptr = q4_base + j.w_off;
            d.w.dtype    = DType::U32;
            d.w.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 8u)};

            d.s.buffer   = buf_q4_arena_;
            d.s.offset   = j.s_off;
            d.s.nbytes   = j.sb_bytes;
            d.s.host_ptr = q4_base + j.s_off;
            d.s.dtype    = DType::BF16;
            d.s.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 64u)};

            d.b.buffer   = buf_q4_arena_;
            d.b.offset   = j.b_off;
            d.b.nbytes   = j.sb_bytes;
            d.b.host_ptr = q4_base + j.b_off;
            d.b.dtype    = DType::BF16;
            d.b.shape    = {static_cast<int64_t>(j.out_dim), static_cast<int64_t>(j.in_dim / 64u)};
        }

        // Convert all small FP16 tensors synchronously (~3.1 MB -> <1 ms)
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

        // 5. Fine-grained parallel 12-thread ARM64 NEON BF16 -> Q4_AFFINE_G64 quantization (~0.25-0.4s!)
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

        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned n_thr = std::min<unsigned>(12u, std::max<unsigned>(1u, hw ? hw : 8u));
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
                    uint32_t* w_base = reinterpret_cast<uint32_t*>(q4_base + j.w_off);
                    uint16_t* s_base = reinterpret_cast<uint16_t*>(q4_base + j.s_off);
                    uint16_t* b_base = reinterpret_cast<uint16_t*>(q4_base + j.b_off);
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

        // 6. Drop quantized BF16 source pages ONLY (strictly excluding codebook pages!),
        //    then parallel pre-fault every 16 KB page of pred_cb + succ_cb into CPU PTEs (~2-5 ms)
        //    so CandidateSelector has ZERO major/minor page-fault spikes (stays 7-20 us on 100% of rounds!).
        for (const auto& [lo_off, hi_off] : evict_ranges) {
            size_t p_lo = align_up_ps(lo_off);
            size_t p_hi = align_down_ps(hi_off);
            if (p_hi <= p_lo) continue;
            // Defensive: never evict any page intersecting pred_cb or succ_cb!
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

        // Parallel 1-byte-per-page PTE warmup over pred_cb + succ_cb (0 B Metal wired RAM, race-free!)
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
            "Q4=%.1f MiB + F16=%.1f MiB (codebooks CPU-mmap 0B Metal) in %.1f ms (%u threads)\n",
            cfg_.num_layers, cfg_.block_size, cfg_.sliding_window, cfg_.hidden_size,
            cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, cfg_.selector_rank, cfg_.selector_top_k,
            q4_cursor / (1024.0 * 1024.0), f16_cursor / (1024.0 * 1024.0), load_time_ms_, n_thr);
    } catch (...) {
        buf_q4_arena_ = nil;
        buf_f16_arena_ = nil;
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
            ::munmap(mmap_ptr_, mmap_size_);
        }
        mmap_ptr_ = nullptr;
        mmap_size_ = 0;
        pred_cb_bf16_ = nullptr;
        succ_cb_bf16_ = nullptr;
        ready_ = false;
        throw;
    }
}

DFlashEngine::~DFlashEngine() {
    buf_q4_arena_ = nil;
    buf_f16_arena_ = nil;
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        ::munmap(mmap_ptr_, mmap_size_);
    }
    mmap_ptr_ = nullptr;
    mmap_size_ = 0;
    pred_cb_bf16_ = nullptr;
    succ_cb_bf16_ = nullptr;
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
    run_candidate_selector(cfg_, pred_cb_bf16_, succ_cb_bf16_, anchor_token,
                           topk_cands, sel_h_f16, nullptr, false, 0.0f, 0ull,
                           out_draft_ids, nullptr);
}

void DFlashEngine::select_sampled(int anchor_token,
                                  const LogitCand* topk_cands,
                                  const uint16_t* sel_h_f16,
                                  float temperature,
                                  uint64_t seed,
                                  int* out_draft_ids,
                                  DraftDist* out_draft_dists) const {
    run_candidate_selector(cfg_, pred_cb_bf16_, succ_cb_bf16_, anchor_token,
                           topk_cands, sel_h_f16, nullptr, true, temperature, seed,
                           out_draft_ids, out_draft_dists);
}

void DFlashEngine::select_greedy_f32(int anchor_token,
                                     const LogitCand* topk_cands,
                                     const float* sel_h_f32,
                                     int* out_draft_ids) const {
    run_candidate_selector(cfg_, pred_cb_bf16_, succ_cb_bf16_, anchor_token,
                           topk_cands, nullptr, sel_h_f32, false, 0.0f, 0ull,
                           out_draft_ids, nullptr);
}

void DFlashEngine::select_sampled_f32(int anchor_token,
                                      const LogitCand* topk_cands,
                                      const float* sel_h_f32,
                                      float temperature,
                                      uint64_t seed,
                                      int* out_draft_ids,
                                      DraftDist* out_draft_dists) const {
    run_candidate_selector(cfg_, pred_cb_bf16_, succ_cb_bf16_, anchor_token,
                           topk_cands, nullptr, sel_h_f32, true, temperature, seed,
                           out_draft_ids, out_draft_dists);
}

} // namespace mi
