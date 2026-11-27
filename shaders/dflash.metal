#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ============================================================================
// DFlash 2 Metal 3.0 Kernels (incoai/Qwen3.8-27B-DFlash2 & z-lab/dflash)
//   - Standard Qwen3 RMSNorm (NO +1.0! FP16 norm weights)
//   - GroupedDynamicCausalConv (group_size=16, kernel_size=2, prepare + finish)
//   - Context KV ring commit (k_norm + full-dim D=128 RoPE -> [Hkv, window, D])
//   - Draft QKV prep (q_norm/k_norm + full-dim D=128 RoPE, 8-row padded Q)
//   - Bidirectional sliding-window GQA-4 AMX attention (Split-K p1 + p2)
//   - Exact 2-pass Top-16 over FP32 lm_head logits (ABI-exact mi::LogitCand)
//   - Target hidden capture (FP32 -> FP16 [cap_M, num_targets * D])
//   - CandidateSelector hidden projection ([M, D] x [R, D]^T -> FP16 [M, R])
// ============================================================================

struct DFlashNormParams {
    uint  M;
    uint  D;
    float eps;
};

struct DFlashConvParams {
    uint M;          // block_size (8)
    uint D;          // hidden_size (5120)
    uint group_size; // conv_group_size (16)
    uint stage;      // 0 = prepare, 1 = finish
};

struct DFlashConvNormParams {
    uint  M;              // block_size (8)
    uint  D;              // hidden_size (5120)
    uint  group_size;     // conv_group_size (16)
    uint  stage;          // 1 = finish
    uint  norm_row_start; // 0 for layer internal norms (all M rows), 1 for final norm (P = M-1 rows)
    float eps;            // rms_eps (1e-6f)
};

struct DFlashKvWriteParams {
    uint  M;          // rows to commit (prefill <=512, decode <=8)
    uint  Hkv;        // 8
    uint  D;          // 128
    uint  pos_start;  // sequence position of row 0
    uint  window;     // sliding_window (2048)
    float rope_theta; // 10000000.0f
    float eps;        // 1e-6f
};

struct DFlashQkvPrepParams {
    uint  M;          // block_size (<= 8)
    uint  Hq;         // 32
    uint  Hkv;        // 8
    uint  D;          // 128
    uint  pos_start;  // sequence position of anchor (row 0)
    float rope_theta; // 10000000.0f
    float eps;        // 1e-6f
};

struct DFlashAttnParams {
    uint  M;          // block_size (<= 8)
    uint  Hq;         // 32
    uint  Hkv;        // 8
    uint  D;          // 128
    uint  pos_start;  // sequence position of anchor (row 0)
    uint  ctx_len;    // total committed context tokens (normally == pos_start)
    uint  window;     // sliding_window (2048)
    uint  num_splits; // 1 (short ctx <= 256) or 8 (longer ctx <= 2048)
    float scale;      // rsqrt(128.0f)
};

struct DFlashTopKParams {
    uint M;          // proposal rows (7)
    uint V;          // vocab_size (248320)
    uint K;          // selector_top_k (16, max 32)
    uint num_chunks; // ceil_div(V, 2048) = 122
};

struct DFlashCaptureParams {
    uint M;           // rows to copy
    uint D;           // hidden_size (5120)
    uint num_targets; // 5
    uint slot_idx;    // 0..4
    uint src_row_off; // starting row offset in src (buf_hidden_)
    uint dst_row_off; // starting row offset in dst (buf_df_capture_)
};

struct DFlashHiddenProjParams {
    uint M; // proposal rows (7)
    uint R; // selector_rank (256)
    uint D; // hidden_size (5120)
};

// ABI-identical to mi::LogitCand in src/sampler.hpp
struct DFlashLogitCand {
    float val;
    uint  id;
};

// ============================================================================
// 1. Standard Qwen3 RMSNorm (NO +1.0): float [M, D] * half [D] -> half [M, D]
//    Grid: (M, 1, 1), TG: (256, 1, 1)
// ============================================================================
kernel void dflash_rmsnorm_f16(
    const device float*        X [[buffer(0)]],
    const device half*         W [[buffer(1)]],
    device half*               Y [[buffer(2)]],
    constant DFlashNormParams& p [[buffer(3)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;

    const device float* xr = X + (ulong)row * p.D;
    device half*        yr = Y + (ulong)row * p.D;

    float sq = 0.0f;
    if ((p.D & 3u) == 0u) {
        const uint d4 = p.D >> 2;
        const device float4* xr4 = (const device float4*)xr;
        for (uint i = tid; i < d4; i += tg_size) {
            float4 v = xr4[i];
            sq = fma(v.x, v.x, fma(v.y, v.y, fma(v.z, v.z, fma(v.w, v.w, sq))));
        }
    } else {
        for (uint i = tid; i < p.D; i += tg_size) {
            float v = xr[i];
            sq = fma(v, v, sq);
        }
    }

    sq = simd_sum(sq);
    threadgroup float smem[32];
    const uint sg   = tid >> 5;
    const uint lane = tid & 31u;
    const uint nsg  = tg_size >> 5;
    if (lane == 0u) smem[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float tot = (lane < nsg) ? smem[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = smem[0];

    if ((p.D & 3u) == 0u) {
        const uint d4 = p.D >> 2;
        const device float4* xr4 = (const device float4*)xr;
        const device half4*  w4  = (const device half4*)W;
        device half4*        yr4 = (device half4*)yr;
        for (uint i = tid; i < d4; i += tg_size) {
            float4 v = xr4[i] * inv_rms * float4(w4[i]);
            yr4[i] = half4(v);
        }
    } else {
        for (uint i = tid; i < p.D; i += tg_size) {
            yr[i] = half(xr[i] * inv_rms * float(W[i]));
        }
    }
}

// ============================================================================
// 2. GroupedDynamicCausalConv (group_size=16, kernel_size=2)
//    base_k: half  [2_stages, 2_taps, D]
//    dyn:    float [M, 2_stages, 2_taps, D / group_size]
//    Stage 0 (prepare):    y[r,c]  = x[r,c]*k0 + (r>0 ? x[r-1,c]*k1 : 0)  (half -> half)
//    Stage 1 (finish+add): h[r,c] += x[r,c]*k0 + (r>0 ? x[r-1,c]*k1 : 0)  (float -> float+=)
//    DispatchThreads: (D, M, 1), TG: (256, 1, 1)
// ============================================================================
kernel void dflash_conv_prepare_f16(
    const device half*         x      [[buffer(0)]],
    const device float*        dyn    [[buffer(1)]],
    const device half*         base_k [[buffer(2)]],
    device half*               y      [[buffer(3)]],
    constant DFlashConvParams& p      [[buffer(4)]],
    uint2 gid                         [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= p.D || r >= p.M) return;

    const uint G = p.D / p.group_size;
    const uint g = c / p.group_size;
    const ulong dyn_base = ((ulong)r * 4ul + (ulong)p.stage * 2ul) * G;
    const ulong bk_base  = (ulong)p.stage * 2ul * p.D;

    const float k0 = float(base_k[bk_base + c])       + dyn[dyn_base + g];
    const float k1 = float(base_k[bk_base + p.D + c]) + dyn[dyn_base + G + g];

    const float cur  = float(x[(ulong)r * p.D + c]);
    const float prev = (r > 0u) ? float(x[(ulong)(r - 1u) * p.D + c]) : 0.0f;
    y[(ulong)r * p.D + c] = half(fma(cur, k0, prev * k1));
}

kernel void dflash_conv_finish_add_f32(
    const device float*        x      [[buffer(0)]],
    const device float*        dyn    [[buffer(1)]],
    const device half*         base_k [[buffer(2)]],
    device float*              h      [[buffer(3)]],
    constant DFlashConvParams& p      [[buffer(4)]],
    uint2 gid                         [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint r = gid.y;
    if (c >= p.D || r >= p.M) return;

    const uint G = p.D / p.group_size;
    const uint g = c / p.group_size;
    const ulong dyn_base = ((ulong)r * 4ul + (ulong)p.stage * 2ul) * G;
    const ulong bk_base  = (ulong)p.stage * 2ul * p.D;

    const float k0 = float(base_k[bk_base + c])       + dyn[dyn_base + g];
    const float k1 = float(base_k[bk_base + p.D + c]) + dyn[dyn_base + G + g];

    const float cur  = x[(ulong)r * p.D + c];
    const float prev = (r > 0u) ? x[(ulong)(r - 1u) * p.D + c] : 0.0f;
    h[(ulong)r * p.D + c] += fma(cur, k0, prev * k1);
}

// ============================================================================
// 2b. Fused GroupedDynamicCausalConv Finish+ResidualAdd + Standard Qwen3 RMSNorm
//     Eliminates 10 kernel launches, 10 barriers, and 10 full FP32 h re-reads/round!
//     Updates h[r, :] in-place for all r in [0..M-1], and for r >= norm_row_start
//     writes RMSNorm(h[r, :]) * norm_w -> y[r - norm_row_start, :] (FP16).
//     Grid: (M, 1, 1), TG: (256, 1, 1)
// ============================================================================
kernel void dflash_conv_finish_add_rmsnorm_f16(
    const device float*            x      [[buffer(0)]], // [M, D]
    const device float*            dyn    [[buffer(1)]], // [M, 2_stages, 2_taps, D / group_size]
    const device half*             base_k [[buffer(2)]], // [2_stages, 2_taps, D]
    device float*                  h      [[buffer(3)]], // [M, D] in-place residual +=
    const device half*             norm_w [[buffer(4)]], // [D] FP16 (NO +1.0!)
    device half*                   y      [[buffer(5)]], // [M - norm_row_start, D] FP16
    constant DFlashConvNormParams& p      [[buffer(6)]],
    uint row                              [[threadgroup_position_in_grid]],
    uint tid                              [[thread_position_in_threadgroup]],
    uint tg_size                          [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;

    const uint G = p.D / p.group_size;
    const ulong dyn_base = ((ulong)row * 4ul + (ulong)p.stage * 2ul) * G;
    const ulong bk_base  = (ulong)p.stage * 2ul * p.D;
    const device float* xr = x + (ulong)row * p.D;
    const device float* xp = (row > 0u) ? (x + (ulong)(row - 1u) * p.D) : nullptr;
    device float*       hr = h + (ulong)row * p.D;

    const bool do_norm = (row >= p.norm_row_start);

    // Fast vectorized register-cached path (covers D=5120 -> 5 float4s/thread, up to D=8192)
    if ((p.D & 3u) == 0u && (p.group_size & 3u) == 0u && p.D <= 8192u && tg_size == 256u) {
        const uint d4 = p.D >> 2;
        const uint gs4 = p.group_size >> 2;
        const device float4* xr4  = (const device float4*)xr;
        const device float4* xp4  = (const device float4*)xp;
        const device half4*  bk04 = (const device half4*)(base_k + bk_base);
        const device half4*  bk14 = (const device half4*)(base_k + bk_base + p.D);
        device float4*       hr4  = (device float4*)hr;

        float4 hv[8];
        float sq = 0.0f;
        uint n_it = 0u;
        for (uint i = tid; i < d4; i += 256u) {
            const uint g = i / gs4;
            const float d0 = dyn[dyn_base + g];
            const float d1 = dyn[dyn_base + G + g];
            const float4 k0 = float4(bk04[i]) + d0;
            const float4 k1 = float4(bk14[i]) + d1;
            const float4 cur  = xr4[i];
            const float4 prev = (row > 0u) ? xp4[i] : float4(0.0f);
            const float4 hnew = hr4[i] + fma(cur, k0, prev * k1);
            hr4[i] = hnew;
            if (do_norm) {
                hv[n_it] = hnew;
                sq = fma(hnew.x, hnew.x, fma(hnew.y, hnew.y, fma(hnew.z, hnew.z, fma(hnew.w, hnew.w, sq))));
            }
            ++n_it;
        }
        if (!do_norm) return;

        sq = simd_sum(sq);
        threadgroup float smem[32];
        const uint sg   = tid >> 5;
        const uint lane = tid & 31u;
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = (lane < 8u) ? smem[lane] : 0.0f;
            tot = simd_sum(tot);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = smem[0];

        const device half4* w4  = (const device half4*)norm_w;
        device half4*       yr4 = (device half4*)(y + (ulong)(row - p.norm_row_start) * p.D);
        uint it = 0u;
        for (uint i = tid; i < d4; i += 256u) {
            yr4[i] = half4(hv[it] * inv_rms * float4(w4[i]));
            ++it;
        }
        return;
    }

    // General fallback for arbitrary D / group_size / tg_size
    float sq = 0.0f;
    for (uint c = tid; c < p.D; c += tg_size) {
        const uint g = c / p.group_size;
        const float k0 = float(base_k[bk_base + c])       + dyn[dyn_base + g];
        const float k1 = float(base_k[bk_base + p.D + c]) + dyn[dyn_base + G + g];
        const float cur  = xr[c];
        const float prev = (row > 0u) ? xp[c] : 0.0f;
        const float hnew = hr[c] + fma(cur, k0, prev * k1);
        hr[c] = hnew;
        if (do_norm) sq = fma(hnew, hnew, sq);
    }
    if (!do_norm) return;

    sq = simd_sum(sq);
    threadgroup float smem[32];
    const uint sg   = tid >> 5;
    const uint lane = tid & 31u;
    const uint nsg  = tg_size >> 5;
    if (lane == 0u) smem[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = (lane < nsg) ? smem[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = smem[0];

    device half* yr = y + (ulong)(row - p.norm_row_start) * p.D;
    for (uint c = tid; c < p.D; c += tg_size) {
        yr[c] = half(hr[c] * inv_rms * float(norm_w[c]));
    }
}

// ============================================================================
// Helper: Standard Qwen3 Head RMSNorm (NO +1.0) + Full-Dim Rotate-Half RoPE
//         1 simdgroup (32 lanes) per (head, row), specialized for D=128
// ============================================================================
inline void dflash_norm_rope_128(
    const device float* src,
    const device half*  norm_w,
    device half*        dst,
    uint  pos,
    float rope_theta,
    float eps,
    uint  lane
) {
    float v0 = src[0u  + lane];
    float v1 = src[32u + lane];
    float v2 = src[64u + lane];
    float v3 = src[96u + lane];

    float sq = fma(v0, v0, fma(v1, v1, fma(v2, v2, v3 * v3)));
    float inv_rms = rsqrt(simd_sum(sq) * (1.0f / 128.0f) + eps);

    v0 *= inv_rms * float(norm_w[0u  + lane]);
    v1 *= inv_rms * float(norm_w[32u + lane]);
    v2 *= inv_rms * float(norm_w[64u + lane]);
    v3 *= inv_rms * float(norm_w[96u + lane]);

    // Full-dim rotate-half RoPE (D=128, half_d=64):
    //   Pair 0 (i = lane     in 0..31):  (v0, v2) = (x[i], x[i + 64])
    //   Pair 1 (i = 32+lane  in 32..63): (v1, v3) = (x[i], x[i + 64])
    const float fpos = float(pos);
    const float freq0 = 1.0f / pow(rope_theta, float(2u * lane) * (1.0f / 128.0f));
    const float ang0  = fpos * freq0;
    const float c0 = cos(ang0), s0 = sin(ang0);
    const float r0 = fma(v0, c0, -v2 * s0);
    const float r2 = fma(v2, c0,  v0 * s0);

    const float freq1 = 1.0f / pow(rope_theta, float(2u * (32u + lane)) * (1.0f / 128.0f));
    const float ang1  = fpos * freq1;
    const float c1 = cos(ang1), s1 = sin(ang1);
    const float r1 = fma(v1, c1, -v3 * s1);
    const float r3 = fma(v3, c1,  v1 * s1);

    dst[0u  + lane] = half(r0);
    dst[32u + lane] = half(r1);
    dst[64u + lane] = half(r2);
    dst[96u + lane] = half(r3);
}

// ============================================================================
// 3. Context KV Ring Commit:
//    k_proj, v_proj: float [M, Hkv, D]
//    k_norm:         half  [D]
//    k_ring, v_ring: half  [Hkv, window, D]  (contiguous per KV head!)
//    Grid: (Hkv, M, 1), TG: (32, 1, 1)
// ============================================================================
kernel void dflash_context_kv_write(
    const device float*           k_proj [[buffer(0)]],
    const device float*           v_proj [[buffer(1)]],
    const device half*            k_norm [[buffer(2)]],
    device half*                  k_ring [[buffer(3)]],
    device half*                  v_ring [[buffer(4)]],
    constant DFlashKvWriteParams& p      [[buffer(5)]],
    uint2 tgid                           [[threadgroup_position_in_grid]],
    uint  lane                           [[thread_index_in_simdgroup]]
) {
    const uint hkv = tgid.x;
    const uint r   = tgid.y;
    if (hkv >= p.Hkv || r >= p.M) return;

    const uint pos  = p.pos_start + r;
    const uint slot = ((p.window & (p.window - 1u)) == 0u) ? (pos & (p.window - 1u)) : (pos % p.window);

    const ulong src_off = ((ulong)r * p.Hkv + hkv) * p.D;
    const ulong dst_off = ((ulong)hkv * p.window + slot) * p.D;

    const device float* ksrc = k_proj + src_off;
    const device float* vsrc = v_proj + src_off;
    device half*        kdst = k_ring + dst_off;
    device half*        vdst = v_ring + dst_off;

    if (p.D == 128u) {
        dflash_norm_rope_128(ksrc, k_norm, kdst, pos, p.rope_theta, p.eps, lane);
        vdst[0u  + lane] = half(vsrc[0u  + lane]);
        vdst[32u + lane] = half(vsrc[32u + lane]);
        vdst[64u + lane] = half(vsrc[64u + lane]);
        vdst[96u + lane] = half(vsrc[96u + lane]);
    } else {
        float sq = 0.0f;
        for (uint i = lane; i < p.D; i += 32u) {
            float x = ksrc[i];
            sq = fma(x, x, sq);
        }
        float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);
        const uint half_d = p.D >> 1;
        for (uint i = lane; i < half_d; i += 32u) {
            float x0 = ksrc[i]          * inv_rms * float(k_norm[i]);
            float x1 = ksrc[i + half_d] * inv_rms * float(k_norm[i + half_d]);
            float freq = 1.0f / pow(p.rope_theta, float(2u * i) / float(p.D));
            float ang  = float(pos) * freq;
            float c = cos(ang), s = sin(ang);
            kdst[i]          = half(fma(x0, c, -x1 * s));
            kdst[i + half_d] = half(fma(x1, c,  x0 * s));
            vdst[i]          = half(vsrc[i]);
            vdst[i + half_d] = half(vsrc[i + half_d]);
        }
    }
}

// ============================================================================
// 4. Draft Block QKV Prep (M <= 8):
//    q_proj:  float [M, Hq, D]    -> q_out:   half [Hq, 8, D]  (rows M..7 zero-padded!)
//    k_proj:  float [M, Hkv, D]   -> k_noise: half [Hkv, M, D]
//    v_proj:  float [M, Hkv, D]   -> v_noise: half [Hkv, M, D]
//    Grid: (Hq + Hkv, 8, 1), TG: (32, 1, 1)
// ============================================================================
kernel void dflash_qkv_prep(
    const device float*           q_proj  [[buffer(0)]],
    const device float*           k_proj  [[buffer(1)]],
    const device float*           v_proj  [[buffer(2)]],
    const device half*            q_norm  [[buffer(3)]],
    const device half*            k_norm  [[buffer(4)]],
    device half*                  q_out   [[buffer(5)]],
    device half*                  k_noise [[buffer(6)]],
    device half*                  v_noise [[buffer(7)]],
    constant DFlashQkvPrepParams& p       [[buffer(8)]],
    uint2 tgid                            [[threadgroup_position_in_grid]],
    uint  lane                            [[thread_index_in_simdgroup]]
) {
    const uint r = tgid.y;
    if (tgid.x < p.Hq) {
        const uint h = tgid.x;
        if (r >= 8u) return;
        device half* qdst = q_out + ((ulong)h * 8ul + r) * p.D;
        if (r >= p.M) {
            for (uint i = lane; i < p.D; i += 32u) qdst[i] = 0.0h;
            return;
        }
        const device float* qsrc = q_proj + ((ulong)r * p.Hq + h) * p.D;
        const uint pos = p.pos_start + r;
        if (p.D == 128u) {
            dflash_norm_rope_128(qsrc, q_norm, qdst, pos, p.rope_theta, p.eps, lane);
        } else {
            float sq = 0.0f;
            for (uint i = lane; i < p.D; i += 32u) { float x = qsrc[i]; sq = fma(x, x, sq); }
            float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);
            const uint half_d = p.D >> 1;
            for (uint i = lane; i < half_d; i += 32u) {
                float x0 = qsrc[i]          * inv_rms * float(q_norm[i]);
                float x1 = qsrc[i + half_d] * inv_rms * float(q_norm[i + half_d]);
                float freq = 1.0f / pow(p.rope_theta, float(2u * i) / float(p.D));
                float ang  = float(pos) * freq;
                float c = cos(ang), s = sin(ang);
                qdst[i]          = half(fma(x0, c, -x1 * s));
                qdst[i + half_d] = half(fma(x1, c,  x0 * s));
            }
        }
    } else {
        const uint hkv = tgid.x - p.Hq;
        if (hkv >= p.Hkv || r >= p.M) return;
        const ulong src_off = ((ulong)r * p.Hkv + hkv) * p.D;
        const ulong dst_off = ((ulong)hkv * p.M + r) * p.D;
        const device float* ksrc = k_proj + src_off;
        const device float* vsrc = v_proj + src_off;
        device half*        kdst = k_noise + dst_off;
        device half*        vdst = v_noise + dst_off;
        const uint pos = p.pos_start + r;
        if (p.D == 128u) {
            dflash_norm_rope_128(ksrc, k_norm, kdst, pos, p.rope_theta, p.eps, lane);
            vdst[0u  + lane] = half(vsrc[0u  + lane]);
            vdst[32u + lane] = half(vsrc[32u + lane]);
            vdst[64u + lane] = half(vsrc[64u + lane]);
            vdst[96u + lane] = half(vsrc[96u + lane]);
        } else {
            float sq = 0.0f;
            for (uint i = lane; i < p.D; i += 32u) { float x = ksrc[i]; sq = fma(x, x, sq); }
            float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);
            const uint half_d = p.D >> 1;
            for (uint i = lane; i < half_d; i += 32u) {
                float x0 = ksrc[i]          * inv_rms * float(k_norm[i]);
                float x1 = ksrc[i + half_d] * inv_rms * float(k_norm[i + half_d]);
                float freq = 1.0f / pow(p.rope_theta, float(2u * i) / float(p.D));
                float ang  = float(pos) * freq;
                float c = cos(ang), s = sin(ang);
                kdst[i]          = half(fma(x0, c, -x1 * s));
                kdst[i + half_d] = half(fma(x1, c,  x0 * s));
                vdst[i]          = half(vsrc[i]);
                vdst[i + half_d] = half(vsrc[i + half_d]);
            }
        }
    }
}

// ============================================================================
// 5. Bidirectional Sliding-Window GQA-4 AMX Attention (D=128, M<=8, is_causal=false)
//    Query at pos_start+r attends to:
//      - Ring entries with k_pos in [max(0, pos_start + r - (window - 1)) .. ctx_len - 1]
//      - ALL M noise entries [0 .. M-1]
//    Pass 1 (dflash_attn_bidir_window_p1):
//      Grid: (Hkv, num_splits, 1), TG: (128, 1, 1)  (4 SGs = 1 SG per GQA-4 head)
//      SRAM: sKV 8.70 KiB + sScore 4.61 KiB + sP 2.30 KiB + sAlpha 0.13 KiB = 15.74 KiB
//      When num_splits == 1: writes final out [M, Hq, D] directly into buffer(5) (skip p2!)
//      When num_splits >  1: writes part_o [M, Hq, num_splits, D] into buffer(5),
//                            part_m [M, Hq, num_splits] into buffer(6),
//                            part_l [M, Hq, num_splits] into buffer(7).
// ============================================================================
kernel void dflash_attn_bidir_window_p1(
    const device half*         q_h           [[buffer(0)]], // [Hq, 8, 128]
    const device half*         k_ring        [[buffer(1)]], // [Hkv, window, 128]
    const device half*         v_ring        [[buffer(2)]], // [Hkv, window, 128]
    const device half*         k_noise       [[buffer(3)]], // [Hkv, M, 128]
    const device half*         v_noise       [[buffer(4)]], // [Hkv, M, 128]
    device half*               out_or_part_o [[buffer(5)]], // [M, Hq, D] (S=1) or [M, Hq, S, D] (S>1)
    device float*              part_m        [[buffer(6)]], // [M, Hq, S] (unused if S=1)
    device float*              part_l        [[buffer(7)]], // [M, Hq, S] (unused if S=1)
    constant DFlashAttnParams& p             [[buffer(8)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint  tid                                [[thread_index_in_threadgroup]],
    uint  lane                               [[thread_index_in_simdgroup]],
    uint  sg                                 [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BK = 32u;
    const uint hkv = tgid.x;
    const uint sp  = tgid.y;
    if (hkv >= p.Hkv || sp >= p.num_splits) return;

    const uint gqa = p.Hq / p.Hkv; // 4
    const uint k_begin  = (p.pos_start + 1u > p.window) ? (p.pos_start + 1u - p.window) : 0u;
    const uint n_ring   = (p.ctx_len > k_begin) ? (p.ctx_len - k_begin) : 0u;
    const uint total_kv = n_ring + p.M;

    const uint total_tiles     = (total_kv + BK - 1u) / BK;
    const uint tiles_per_split = (total_tiles + p.num_splits - 1u) / p.num_splits;
    const uint tile_start      = sp * tiles_per_split;
    const uint tile_end        = min(total_tiles, (sp + 1u) * tiles_per_split);

    // Empty split fast-exit
    if (tile_start >= tile_end) {
        if (p.num_splits > 1u) {
            for (uint u = tid; u < gqa * 8u; u += 128u) {
                const uint h_loc = u >> 3;
                const uint r     = u & 7u;
                if (r < p.M) {
                    const ulong slot = ((ulong)r * p.Hq + (hkv * gqa + h_loc)) * p.num_splits + sp;
                    part_m[slot] = -INFINITY;
                    part_l[slot] = 0.0f;
                }
            }
        }
        return;
    }

    const bool pow2_win = (p.window & (p.window - 1u)) == 0u;
    const uint win_mask = p.window - 1u;

    // 16-byte aligned 32 x 136 half tile (8.70 KiB) + FP32 score + FP16 P (total 15.74 KiB)
    threadgroup uint4 sKV_u4[32][17];
    threadgroup half* sKV = (threadgroup half*)sKV_u4;
    threadgroup float sScore[32][36];
    threadgroup half  sP[32][36];
    threadgroup float sAlpha[32];

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const float scale_log2 = p.scale * 1.4426950408889634f;

    // Max q_min across all query rows r in [0..M-1] (used for branchless interior tile check)
    const uint q_min_max = (p.pos_start + p.M > p.window) ? (p.pos_start + p.M - p.window) : 0u;

    // Supports gqa=4 in 1 pass (and any gqa multiple of 4 cleanly)
    for (uint h_base = 0u; h_base < gqa; h_base += 4u) {
        const uint h_loc = h_base + sg;
        const bool active_sg = (h_loc < gqa);
        const uint hq = hkv * gqa + h_loc;
        const device half* q_ptr = q_h + (ulong)(active_sg ? hq : 0u) * (8ul * 128ul);

        simdgroup_float8x8 O[16];
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 16u; ++j) {
            O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }

        float m_row[8], l_row[8];
        #pragma clang loop unroll(full)
        for (uint rr = 0u; rr < 8u; ++rr) {
            m_row[rr] = -INFINITY;
            l_row[rr] = 0.0f;
        }

        for (uint tile = tile_start; tile < tile_end; ++tile) {
            const uint tile_base = tile * BK;

            // 1. Coalesced 128-bit load of 32 x 128 FP16 K tile (ring + noise unified)
            #pragma clang loop unroll(full)
            for (uint it4 = 0u; it4 < 4u; ++it4) {
                const uint tk   = it4 * 8u + (tid >> 4); // 0..31
                const uint vec  = tid & 15u;             // 0..15 (x8 halfs = 128)
                const uint idx  = tile_base + tk;
                uint4 raw = uint4(0u);
                if (idx < n_ring) {
                    const uint k_pos = k_begin + idx;
                    const uint slot  = pow2_win ? (k_pos & win_mask) : (k_pos % p.window);
                    raw = *(const device uint4*)(k_ring + ((ulong)hkv * p.window + slot) * 128ul + vec * 8u);
                } else if (idx < total_kv) {
                    const uint j = idx - n_ring;
                    raw = *(const device uint4*)(k_noise + ((ulong)hkv * p.M + j) * 128ul + vec * 8u);
                }
                sKV_u4[tk][vec] = raw;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 2. AMX Q[8, 128] @ K[32, 128]^T -> FP32 s0..s3
            if (active_sg) {
                simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 s3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

                #pragma clang loop unroll(full)
                for (uint kk = 0u; kk < 128u; kk += 8u) {
                    simdgroup_half8x8 qm, km0, km1, km2, km3;
                    simdgroup_load(qm,  q_ptr + kk, 128);
                    simdgroup_load(km0, sKV + 0u  * 136u + kk, 136, ulong2(0, 0), true);
                    simdgroup_load(km1, sKV + 8u  * 136u + kk, 136, ulong2(0, 0), true);
                    simdgroup_load(km2, sKV + 16u * 136u + kk, 136, ulong2(0, 0), true);
                    simdgroup_load(km3, sKV + 24u * 136u + kk, 136, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s0, qm, km0, s0);
                    simdgroup_multiply_accumulate(s1, qm, km1, s1);
                    simdgroup_multiply_accumulate(s2, qm, km2, s2);
                    simdgroup_multiply_accumulate(s3, qm, km3, s3);
                }

                const uint s_row = sg * 8u + fm;
                sScore[s_row][0u  + fn]      = s0.thread_elements()[0];
                sScore[s_row][0u  + fn + 1u] = s0.thread_elements()[1];
                sScore[s_row][8u  + fn]      = s1.thread_elements()[0];
                sScore[s_row][8u  + fn + 1u] = s1.thread_elements()[1];
                sScore[s_row][16u + fn]      = s2.thread_elements()[0];
                sScore[s_row][16u + fn + 1u] = s2.thread_elements()[1];
                sScore[s_row][24u + fn]      = s3.thread_elements()[0];
                sScore[s_row][24u + fn + 1u] = s3.thread_elements()[1];
                simdgroup_barrier(mem_flags::mem_threadgroup);

                // 3. Online softmax over 32 keys with exact per-row sliding-window mask
                const bool interior = (tile_base + BK <= total_kv) &&
                                      (tile_base >= n_ring || (k_begin + tile_base) >= q_min_max);
                if (interior) {
                    #pragma clang loop unroll(full)
                    for (uint rr = 0u; rr < 8u; ++rr) {
                        const uint r = sg * 8u + rr;
                        const float score   = sScore[r][lane] * scale_log2;
                        const float row_max = simd_max(score);
                        const float m_old   = m_row[rr];
                        const float m_new   = max(m_old, row_max);
                        const float alpha   = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                        const float p_val   = exp2(score - m_new);
                        l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                        m_row[rr] = m_new;
                        sP[r][lane] = half(p_val);
                        if (lane == 0u) sAlpha[r] = alpha;
                    }
                } else {
                    const uint idx = tile_base + lane;
                    const bool in_bounds = (idx < total_kv);
                    const bool is_ring   = (idx < n_ring);
                    const uint k_pos     = k_begin + idx;
                    #pragma clang loop unroll(full)
                    for (uint rr = 0u; rr < 8u; ++rr) {
                        const uint r = sg * 8u + rr;
                        const uint q_min = (p.pos_start + rr + 1u > p.window)
                                               ? (p.pos_start + rr + 1u - p.window)
                                               : 0u;
                        const bool valid = (rr < p.M) && in_bounds && (!is_ring || k_pos >= q_min);
                        const float score   = valid ? (sScore[r][lane] * scale_log2) : -INFINITY;
                        const float row_max = simd_max(score);
                        const float m_old   = m_row[rr];
                        const float m_new   = max(m_old, row_max);
                        const float alpha   = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                        const float p_val   = (valid && m_new != -INFINITY) ? exp2(score - m_new) : 0.0f;
                        l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                        m_row[rr] = m_new;
                        sP[r][lane] = half(p_val);
                        if (lane == 0u) sAlpha[r] = alpha;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 4. Coalesced 128-bit load of 32 x 128 FP16 V tile
            #pragma clang loop unroll(full)
            for (uint it4 = 0u; it4 < 4u; ++it4) {
                const uint tk   = it4 * 8u + (tid >> 4);
                const uint vec  = tid & 15u;
                const uint idx  = tile_base + tk;
                uint4 raw = uint4(0u);
                if (idx < n_ring) {
                    const uint k_pos = k_begin + idx;
                    const uint slot  = pow2_win ? (k_pos & win_mask) : (k_pos % p.window);
                    raw = *(const device uint4*)(v_ring + ((ulong)hkv * p.window + slot) * 128ul + vec * 8u);
                } else if (idx < total_kv) {
                    const uint j = idx - n_ring;
                    raw = *(const device uint4*)(v_noise + ((ulong)hkv * p.M + j) * 128ul + vec * 8u);
                }
                sKV_u4[tk][vec] = raw;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 5. Rescale FP32 accumulators + AMX P[8, 32] @ V[32, 128]
            if (active_sg) {
                const float a_row = sAlpha[sg * 8u + fm];
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 16u; ++j) {
                    O[j].thread_elements()[0] *= a_row;
                    O[j].thread_elements()[1] *= a_row;
                }
                #pragma clang loop unroll(full)
                for (uint kk = 0u; kk < 32u; kk += 8u) {
                    simdgroup_half8x8 pm;
                    simdgroup_load(pm, &sP[sg * 8u][kk], 36);
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 16u; ++j) {
                        simdgroup_half8x8 vm;
                        simdgroup_load(vm, sKV + kk * 136u + 8u * j, 136);
                        simdgroup_multiply_accumulate(O[j], pm, vm, O[j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 6. Store normalized output (direct to out if S==1, or to part_o/part_m/part_l if S>1)
        if (active_sg) {
            if (fm < p.M) {
                const float l_val = l_row[fm];
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                device half* dst;
                if (p.num_splits == 1u) {
                    dst = out_or_part_o + ((ulong)fm * p.Hq + hq) * 128ul;
                } else {
                    const ulong slot = ((ulong)fm * p.Hq + hq) * p.num_splits + sp;
                    dst = out_or_part_o + slot * 128ul;
                    if (fn == 0u) {
                        part_m[slot] = m_row[fm];
                        part_l[slot] = l_val;
                    }
                }
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 16u; ++j) {
                    dst[8u * j + fn]      = half(O[j].thread_elements()[0] * inv_l);
                    dst[8u * j + fn + 1u] = half(O[j].thread_elements()[1] * inv_l);
                }
            }
        }
        if (h_base + 4u < gqa) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Pass 2: Reduce Split-K partials -> final half [M, Hq, D] (only needed when num_splits > 1)
//   Grid: (Hq, M, 1), TG: (32, 1, 1)
kernel void dflash_attn_bidir_window_p2(
    const device half*         part_o [[buffer(0)]], // [M, Hq, S, 128]
    const device float*        part_m [[buffer(1)]], // [M, Hq, S]
    const device float*        part_l [[buffer(2)]], // [M, Hq, S]
    device half*               out    [[buffer(3)]], // [M, Hq, 128]
    constant DFlashAttnParams& p      [[buffer(4)]],
    uint2 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]]
) {
    const uint hq = tgid.x;
    const uint r  = tgid.y;
    if (hq >= p.Hq || r >= p.M) return;

    const ulong base_slot = ((ulong)r * p.Hq + hq) * p.num_splits;

    // 1. Find global max m across splits (in base-2 log space matching p1)
    float m_max = -INFINITY;
    for (uint s = lane; s < p.num_splits; s += 32u) {
        m_max = max(m_max, part_m[base_slot + s]);
    }
    m_max = simd_max(m_max);

    // 2. Accumulate weighted normalized partials across splits
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float sum_w = 0.0f;
    for (uint s = 0u; s < p.num_splits; ++s) {
        const float ls = part_l[base_slot + s];
        if (ls <= 0.0f || m_max == -INFINITY) continue;
        const float ms = part_m[base_slot + s];
        const float w  = ls * exp2(ms - m_max);
        sum_w += w;
        const device half* po = part_o + (base_slot + s) * 128ul;
        acc0 = fma(w, float(po[0u  + lane]), acc0);
        acc1 = fma(w, float(po[32u + lane]), acc1);
        acc2 = fma(w, float(po[64u + lane]), acc2);
        acc3 = fma(w, float(po[96u + lane]), acc3);
    }

    const float inv_w = (sum_w > 0.0f) ? (1.0f / sum_w) : 0.0f;
    device half* dst = out + ((ulong)r * p.Hq + hq) * 128ul;
    dst[0u  + lane] = half(acc0 * inv_w);
    dst[32u + lane] = half(acc1 * inv_w);
    dst[64u + lane] = half(acc2 * inv_w);
    dst[96u + lane] = half(acc3 * inv_w);
}

// ============================================================================
// 6. Exact 2-Pass Top-K (K <= 32, typically 16) over FP32 lm_head Logits [M, V]
//    Deterministic tie-breaking by smallest token ID.
//    1-barrier per-SG reduction + packed_float4 coalesced loads (32 barriers -> 1).
//    Pass 1 (dflash_lmhead_topk16_p1):
//      num_chunks = ceil_div(V, 2048) (= 122 for V=248077/248320)
//      Grid: (num_chunks, M, 1), TG: (256, 1, 1)
//    Pass 2 (dflash_lmhead_topk16_p2):
//      Grid: (1, M, 1), TG: (256, 1, 1) -> outputs flat [M * K] DFlashLogitCand
// ============================================================================
kernel void dflash_lmhead_topk16_p1(
    const device float*       logits     [[buffer(0)]],
    device DFlashLogitCand*   part_cands [[buffer(1)]],
    constant DFlashTopKParams& p         [[buffer(2)]],
    uint2 tgid                           [[threadgroup_position_in_grid]],
    uint  tid                            [[thread_index_in_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]],
    uint  sg                             [[simdgroup_index_in_threadgroup]]
) {
    (void)tid;
    const uint chunk = tgid.x;
    const uint row   = tgid.y;
    if (chunk >= p.num_chunks || row >= p.M || p.K == 0u || p.K > 32u) return;

    const device float* lrow = logits + (ulong)row * p.V;
    const uint sg_base = chunk * 2048u + sg * 256u;

    float vals[8];
    uint  ids[8];
    float my_v    = -INFINITY;
    uint  my_slot = 0u;
    uint  my_id   = 0xFFFFFFFFu;

    // 2 coalesced 4B-aligned packed_float4 loads per lane (handles non-mult-4 V like 248077 safely!)
    const uint idx0 = sg_base + lane * 4u;
    const uint idx1 = sg_base + 128u + lane * 4u;
    float4 f0 = float4(-INFINITY);
    float4 f1 = float4(-INFINITY);
    if (idx0 + 4u <= p.V) {
        f0 = float4(*(const device packed_float4*)(lrow + idx0));
    } else {
        for (uint j = 0u; j < 4u; ++j) if (idx0 + j < p.V) f0[j] = lrow[idx0 + j];
    }
    if (idx1 + 4u <= p.V) {
        f1 = float4(*(const device packed_float4*)(lrow + idx1));
    } else {
        for (uint j = 0u; j < 4u; ++j) if (idx1 + j < p.V) f1[j] = lrow[idx1 + j];
    }

    vals[0] = isnan(f0.x) ? -INFINITY : f0.x; ids[0] = (idx0 + 0u < p.V) ? (idx0 + 0u) : 0xFFFFFFFFu;
    vals[1] = isnan(f0.y) ? -INFINITY : f0.y; ids[1] = (idx0 + 1u < p.V) ? (idx0 + 1u) : 0xFFFFFFFFu;
    vals[2] = isnan(f0.z) ? -INFINITY : f0.z; ids[2] = (idx0 + 2u < p.V) ? (idx0 + 2u) : 0xFFFFFFFFu;
    vals[3] = isnan(f0.w) ? -INFINITY : f0.w; ids[3] = (idx0 + 3u < p.V) ? (idx0 + 3u) : 0xFFFFFFFFu;
    vals[4] = isnan(f1.x) ? -INFINITY : f1.x; ids[4] = (idx1 + 0u < p.V) ? (idx1 + 0u) : 0xFFFFFFFFu;
    vals[5] = isnan(f1.y) ? -INFINITY : f1.y; ids[5] = (idx1 + 1u < p.V) ? (idx1 + 1u) : 0xFFFFFFFFu;
    vals[6] = isnan(f1.z) ? -INFINITY : f1.z; ids[6] = (idx1 + 2u < p.V) ? (idx1 + 2u) : 0xFFFFFFFFu;
    vals[7] = isnan(f1.w) ? -INFINITY : f1.w; ids[7] = (idx1 + 3u < p.V) ? (idx1 + 3u) : 0xFFFFFFFFu;

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 8u; ++i) {
        const float v = vals[i];
        const uint id = ids[i];
        if (v > my_v || (v == my_v && id < my_id && v > -INFINITY)) {
            my_v    = v;
            my_slot = i;
            my_id   = id;
        }
    }

    // Each of the 8 SGs selects its own Top-K purely in warp ops (0 threadgroup_barriers!)
    threadgroup float sm_v[8][32];
    threadgroup uint  sm_id[8][32];

    for (uint k = 0u; k < p.K; ++k) {
        const float sg_v  = simd_max(my_v);
        const uint  sg_id = simd_min((my_v == sg_v) ? my_id : 0xFFFFFFFFu);
        if (lane == 0u) {
            sm_v[sg][k]  = sg_v;
            sm_id[sg][k] = sg_id;
        }
        if (my_id == sg_id && my_id != 0xFFFFFFFFu) {
            vals[my_slot] = -INFINITY;
            ids[my_slot]  = 0xFFFFFFFFu;
            my_v    = -INFINITY;
            my_slot = 0u;
            my_id   = 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 8u; ++i) {
                const float v  = vals[i];
                const uint  id = ids[i];
                if (v > my_v || (v == my_v && id < my_id && v > -INFINITY)) {
                    my_v    = v;
                    my_slot = i;
                    my_id   = id;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 merges 8 * K <= 256 candidates into final chunk Top-K (0 barriers!)
    if (sg == 0u) {
        const uint total_c = 8u * p.K;
        float mv_a[8];
        uint  mi_a[8];
        float b_v = -INFINITY;
        uint  b_s = 0u;
        uint  b_i = 0xFFFFFFFFu;

        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            const uint pos = i * 32u + lane;
            float v  = -INFINITY;
            uint  id = 0xFFFFFFFFu;
            if (pos < total_c) {
                const uint s_idx = pos / p.K;
                const uint k_idx = pos - s_idx * p.K;
                v  = sm_v[s_idx][k_idx];
                id = sm_id[s_idx][k_idx];
            }
            mv_a[i] = v;
            mi_a[i] = id;
            if (v > b_v || (v == b_v && id < b_i && v > -INFINITY)) {
                b_v = v;
                b_s = i;
                b_i = id;
            }
        }

        const ulong out_base = ((ulong)row * p.num_chunks + chunk) * p.K;
        for (uint k = 0u; k < p.K; ++k) {
            const float wv  = simd_max(b_v);
            const uint  wid = simd_min((b_v == wv) ? b_i : 0xFFFFFFFFu);
            if (lane == 0u) {
                part_cands[out_base + k] = DFlashLogitCand{ wv, wid };
            }
            if (b_i == wid && b_i != 0xFFFFFFFFu) {
                mv_a[b_s] = -INFINITY;
                mi_a[b_s] = 0xFFFFFFFFu;
                b_v = -INFINITY;
                b_s = 0u;
                b_i = 0xFFFFFFFFu;
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 8u; ++i) {
                    const float v  = mv_a[i];
                    const uint  id = mi_a[i];
                    if (v > b_v || (v == b_v && id < b_i && v > -INFINITY)) {
                        b_v = v;
                        b_s = i;
                        b_i = id;
                    }
                }
            }
        }
    }
}

kernel void dflash_lmhead_topk16_p2(
    const device DFlashLogitCand* part_cands [[buffer(0)]],
    device DFlashLogitCand*       out_cands  [[buffer(1)]],
    constant DFlashTopKParams&    p          [[buffer(2)]],
    uint2 tgid                               [[threadgroup_position_in_grid]],
    uint  tid                                [[thread_index_in_threadgroup]],
    uint  lane                               [[thread_index_in_simdgroup]],
    uint  sg                                 [[simdgroup_index_in_threadgroup]]
) {
    const uint row = tgid.y;
    if (row >= p.M || p.K == 0u || p.K > 32u) return;

    const uint total = p.num_chunks * p.K; // 122 * 16 = 1952 <= 4096
    const device DFlashLogitCand* src = part_cands + (ulong)row * total;

    float vals[16];
    uint  ids[16];
    float my_v    = -INFINITY;
    uint  my_slot = 0u;
    uint  my_id   = 0xFFFFFFFFu;

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 16u; ++i) {
        const uint pos = i * 256u + tid;
        float v  = -INFINITY;
        uint  id = 0xFFFFFFFFu;
        if (pos < total) {
            DFlashLogitCand c = src[pos];
            if (!isnan(c.val) && c.id != 0xFFFFFFFFu) {
                v  = c.val;
                id = c.id;
            }
        }
        vals[i] = v;
        ids[i]  = id;
        if (v > my_v || (v == my_v && id < my_id && v > -INFINITY)) {
            my_v    = v;
            my_slot = i;
            my_id   = id;
        }
    }

    threadgroup float sm_v[8][32];
    threadgroup uint  sm_id[8][32];

    for (uint k = 0u; k < p.K; ++k) {
        const float sg_v  = simd_max(my_v);
        const uint  sg_id = simd_min((my_v == sg_v) ? my_id : 0xFFFFFFFFu);
        if (lane == 0u) {
            sm_v[sg][k]  = sg_v;
            sm_id[sg][k] = sg_id;
        }
        if (my_id == sg_id && my_id != 0xFFFFFFFFu) {
            vals[my_slot] = -INFINITY;
            ids[my_slot]  = 0xFFFFFFFFu;
            my_v    = -INFINITY;
            my_slot = 0u;
            my_id   = 0xFFFFFFFFu;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 16u; ++i) {
                const float v  = vals[i];
                const uint  id = ids[i];
                if (v > my_v || (v == my_v && id < my_id && v > -INFINITY)) {
                    my_v    = v;
                    my_slot = i;
                    my_id   = id;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        const uint total_c = 8u * p.K;
        float mv_a[8];
        uint  mi_a[8];
        float b_v = -INFINITY;
        uint  b_s = 0u;
        uint  b_i = 0xFFFFFFFFu;

        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            const uint pos = i * 32u + lane;
            float v  = -INFINITY;
            uint  id = 0xFFFFFFFFu;
            if (pos < total_c) {
                const uint s_idx = pos / p.K;
                const uint k_idx = pos - s_idx * p.K;
                v  = sm_v[s_idx][k_idx];
                id = sm_id[s_idx][k_idx];
            }
            mv_a[i] = v;
            mi_a[i] = id;
            if (v > b_v || (v == b_v && id < b_i && v > -INFINITY)) {
                b_v = v;
                b_s = i;
                b_i = id;
            }
        }

        device DFlashLogitCand* dst = out_cands + (ulong)row * p.K;
        for (uint k = 0u; k < p.K; ++k) {
            const float wv  = simd_max(b_v);
            const uint  wid = simd_min((b_v == wv) ? b_i : 0xFFFFFFFFu);
            if (lane == 0u) {
                dst[k] = DFlashLogitCand{ wv, wid };
            }
            if (b_i == wid && b_i != 0xFFFFFFFFu) {
                mv_a[b_s] = -INFINITY;
                mi_a[b_s] = 0xFFFFFFFFu;
                b_v = -INFINITY;
                b_s = 0u;
                b_i = 0xFFFFFFFFu;
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 8u; ++i) {
                    const float v  = mv_a[i];
                    const uint  id = mi_a[i];
                    if (v > b_v || (v == b_v && id < b_i && v > -INFINITY)) {
                        b_v = v;
                        b_s = i;
                        b_i = id;
                    }
                }
            }
        }
    }
}

// ============================================================================
// 7. Target Hidden State Capture:
//    Copies M rows of buf_hidden_ (float [src_M, D]) starting at src_row_off
//    into slot_idx of buf_df_capture_ (half [cap_M, num_targets * D]) at dst_row_off.
//    DispatchThreads: (D / 4, M, 1) when D%4==0 (or (D, M, 1) via scalar fallback), TG: (256, 1, 1)
//    Note: uses gid.x in [0 .. D/4 - 1] (1280 for D=5120) for 128-bit float4 -> half4 copy!
// ============================================================================
kernel void dflash_capture_hidden(
    const device float*           src [[buffer(0)]],
    device half*                  dst [[buffer(1)]],
    constant DFlashCaptureParams& p   [[buffer(2)]],
    uint2 gid                         [[thread_position_in_grid]]
) {
    const uint c4 = gid.x;
    const uint r  = gid.y;
    const uint c  = c4 << 2;
    if (c >= p.D || r >= p.M) return;

    const ulong src_off = (ulong)(p.src_row_off + r) * p.D + c;
    const ulong dst_off = (ulong)(p.dst_row_off + r) * ((ulong)p.num_targets * p.D)
                        + (ulong)p.slot_idx * p.D + c;

    if (((p.D & 3u) == 0u) && (c + 4u <= p.D)) {
        const float4 v = *(const device float4*)(src + src_off);
        *(device half4*)(dst + dst_off) = half4(v);
    } else {
        for (uint i = 0u; c + i < p.D; ++i) {
            dst[dst_off + i] = half(src[src_off + i]);
        }
    }
}

// ============================================================================
// 8. CandidateSelector Hidden Projection:
//    Y[M, R] = X[M, D] @ W[R, D]^T  (half x half -> FP32 accum -> half)
//    Reads W[R, D] ONCE across up to 8 rows of X in registers (~8 us total!).
//    Grid: (ceil_div(R, 4), ceil_div(M, 8), 1), TG: (128, 1, 1)  (4 SGs = 4 output rows/TG)
// ============================================================================
kernel void dflash_hidden_proj_f16(
    const device half*               X [[buffer(0)]], // [M, D]
    const device half*               W [[buffer(1)]], // [R, D]
    device half*                     Y [[buffer(2)]], // [M, R]
    constant DFlashHiddenProjParams& p [[buffer(3)]],
    uint2 tgid                         [[threadgroup_position_in_grid]],
    uint  lane                         [[thread_index_in_simdgroup]],
    uint  sg                           [[simdgroup_index_in_threadgroup]]
) {
    const uint r_out = tgid.x * 4u + sg;
    const uint m0    = tgid.y * 8u;
    if (r_out >= p.R || m0 >= p.M) return;

    const uint tm = min(8u, p.M - m0);
    const device half* wr = W + (ulong)r_out * p.D;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    if ((p.D & 7u) == 0u) {
        const uint d8 = p.D >> 3;
        for (uint i = lane; i < d8; i += 32u) {
            const uint k = i << 3;
            const half4 w0 = *(const device half4*)(wr + k);
            const half4 w1 = *(const device half4*)(wr + k + 4u);
            const float4 fw0 = float4(w0);
            const float4 fw1 = float4(w1);
            #pragma clang loop unroll(full)
            for (uint m = 0u; m < 8u; ++m) {
                if (m < tm) {
                    const device half* xr = X + (ulong)(m0 + m) * p.D + k;
                    const float4 fx0 = float4(*(const device half4*)(xr));
                    const float4 fx1 = float4(*(const device half4*)(xr + 4u));
                    acc[m] = fma(fw0.x, fx0.x, fma(fw0.y, fx0.y, fma(fw0.z, fx0.z, fma(fw0.w, fx0.w, acc[m]))));
                    acc[m] = fma(fw1.x, fx1.x, fma(fw1.y, fx1.y, fma(fw1.z, fx1.z, fma(fw1.w, fx1.w, acc[m]))));
                }
            }
        }
    } else {
        for (uint k = lane; k < p.D; k += 32u) {
            const float fw = float(wr[k]);
            for (uint m = 0u; m < tm; ++m) {
                acc[m] = fma(fw, float(X[(ulong)(m0 + m) * p.D + k]), acc[m]);
            }
        }
    }

    for (uint m = 0u; m < tm; ++m) {
        const float sum = simd_sum(acc[m]);
        if (lane == 0u) {
            Y[(ulong)(m0 + m) * p.R + r_out] = half(sum);
        }
    }
}
