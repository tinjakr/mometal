#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float softplus_f32(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// ============================================================================
// 1. GatedDeltaNet: Causal Depthwise Conv1d (K=4) + SiLU + Conv State Update
//    conv_state: [3, C] (oldest at row 0, newest at row 2)
//    conv_w:     bf16 [C, 4] (taps 0..3 chronological: 0=oldest, 3=current)
//    qkv_in:     f32 [M, C]
//    qkv_out:    f32 [M, C]
// ============================================================================

struct GdnConvParams {
    uint M;
    uint C;
};

kernel void qwen_gdn_conv1d_silu(
    const device float*  qkv_in         [[buffer(0)]],
    const device ushort* conv_w         [[buffer(1)]],
    const device float*  conv_state_in  [[buffer(2)]],
    device float*        qkv_out        [[buffer(3)]],
    constant GdnConvParams& p           [[buffer(4)]],
    device float*        conv_state_out [[buffer(5)]],
    uint c                              [[thread_position_in_grid]]
) {
    if (c >= p.C) return;

    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);

    float s0 = conv_state_in[0u * p.C + c];
    float s1 = conv_state_in[1u * p.C + c];
    float s2 = conv_state_in[2u * p.C + c];

    for (uint t = 0; t < p.M; t++) {
        float x = qkv_in[(ulong)t * p.C + c];
        float acc = fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3)));
        qkv_out[(ulong)t * p.C + c] = silu_f32(acc);
        s0 = s1;
        s1 = s2;
        s2 = x;
    }

    conv_state_out[0u * p.C + c] = s0;
    conv_state_out[1u * p.C + c] = s1;
    conv_state_out[2u * p.C + c] = s2;
}

// Parallel prefill Conv1d (K=4) + SiLU: 64x more parallelism over (C, ceil_div(M, 8))!
// Reads conv_state_in read-only (100% race-free); conv_state_out updated concurrently behind existing barrier!
kernel void qwen_gdn_conv1d_silu_prefill(
    const device float*  qkv_in        [[buffer(0)]],
    const device ushort* conv_w        [[buffer(1)]],
    const device float*  conv_state_in [[buffer(2)]],
    device float*        qkv_out       [[buffer(3)]],
    constant GdnConvParams& p          [[buffer(4)]],
    uint2 gid                          [[thread_position_in_grid]]
) {
    uint c  = gid.x;
    uint t0 = gid.y * 8u;
    if (c >= p.C || t0 >= p.M) return;

    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);

    float s0, s1, s2;
    if (t0 == 0u) {
        s0 = conv_state_in[0u * p.C + c];
        s1 = conv_state_in[1u * p.C + c];
        s2 = conv_state_in[2u * p.C + c];
    } else {
        s0 = qkv_in[(ulong)(t0 - 3u) * p.C + c];
        s1 = qkv_in[(ulong)(t0 - 2u) * p.C + c];
        s2 = qkv_in[(ulong)(t0 - 1u) * p.C + c];
    }

    uint t_end = min(t0 + 8u, p.M);
    for (uint t = t0; t < t_end; t++) {
        float x = qkv_in[(ulong)t * p.C + c];
        float acc = fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3)));
        qkv_out[(ulong)t * p.C + c] = silu_f32(acc);
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
}

kernel void qwen_gdn_conv_state_update(
    const device float* qkv_in         [[buffer(0)]],
    device float*       conv_state_out [[buffer(1)]],
    constant GdnConvParams& p          [[buffer(2)]],
    uint c                             [[thread_position_in_grid]]
) {
    if (c >= p.C) return;
    conv_state_out[0u * p.C + c] = qkv_in[(ulong)(p.M - 3u) * p.C + c];
    conv_state_out[1u * p.C + c] = qkv_in[(ulong)(p.M - 2u) * p.C + c];
    conv_state_out[2u * p.C + c] = qkv_in[(ulong)(p.M - 1u) * p.C + c];
}

// ============================================================================
// 2. GatedDeltaNet: Q/K Unweighted RMSNorm + Scaling + Beta/G Computation
//    qkv_conv: [M, 2*key_dim + value_dim]  (key_dim = Hk*Dk = 2048, value_dim = Hv*Dv = 6144)
//    q_out:    [M, Hk, Dk]  q = (1/Dk) * rmsnorm(q)
//    k_out:    [M, Hk, Dk]  k = (1/sqrt(Dk)) * rmsnorm(k)
// ============================================================================

struct GdnPrepParams {
    uint M;
    uint Hk;
    uint Dk;
    uint Hv;
    uint C;      // conv_dim = 2*Hk*Dk + Hv*Dv
    float eps;
};

kernel void qwen_gdn_qknorm(
    const device float* qkv_conv [[buffer(0)]],
    device float*       q_out    [[buffer(1)]],
    device float*       k_out    [[buffer(2)]],
    constant GdnPrepParams& p    [[buffer(3)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    // Grid: (Hk * 2, M) threadgroups, each 1 simdgroup (32 threads)
    uint h_idx = tgid.x;
    uint t = tgid.y;
    if (t >= p.M || h_idx >= p.Hk * 2u) return;

    bool is_k = (h_idx >= p.Hk);
    uint hk = is_k ? (h_idx - p.Hk) : h_idx;
    uint key_dim = p.Hk * p.Dk;
    const device float* src = qkv_conv + (ulong)t * p.C + (is_k ? key_dim : 0u) + hk * p.Dk;
    device float* dst = (is_k ? k_out : q_out) + ((ulong)t * p.Hk + hk) * p.Dk;

    // Dk = 128 -> 4 elements per lane
    float v0 = src[lane];
    float v1 = src[lane + 32u];
    float v2 = src[lane + 64u];
    float v3 = src[lane + 96u];
    float sq = fma(v0, v0, fma(v1, v1, fma(v2, v2, v3 * v3)));
    sq = simd_sum(sq);

    float inv_rms = rsqrt(sq / float(p.Dk) + p.eps);
    float inv_scale = rsqrt(float(p.Dk));
    float mul = is_k ? (inv_rms * inv_scale) : (inv_rms * inv_scale * inv_scale);

    dst[lane]       = v0 * mul;
    dst[lane + 32u] = v1 * mul;
    dst[lane + 64u] = v2 * mul;
    dst[lane + 96u] = v3 * mul;
}

kernel void qwen_gdn_gates(
    const device float*  a_in    [[buffer(0)]],
    const device float*  b_in    [[buffer(1)]],
    const device ushort* A_log   [[buffer(2)]],
    const device ushort* dt_bias [[buffer(3)]],
    device float*        g_out   [[buffer(4)]],
    device float*        beta_out[[buffer(5)]],
    constant GdnPrepParams& p    [[buffer(6)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint hv = gid.x;
    uint t  = gid.y;
    if (hv >= p.Hv || t >= p.M) return;

    ulong idx = (ulong)t * p.Hv + hv;
    float av = a_in[idx];
    float bv = b_in[idx];
    float alog = bf16_to_f32(A_log[hv]);
    float dtb  = bf16_to_f32(dt_bias[hv]);

    g_out[idx]    = exp(-exp(alog) * softplus_f32(av + dtb));
    beta_out[idx] = sigmoid_f32(bv);
}

// ============================================================================
// 2b. Round 7b: Fused GDN Decode Prework (M = 1..8, 128 threads/TG = 1 ch/thread!)
//     Fuses conv1d + SiLU + conv_state_update + Q/K RMSNorm/scaling + gates (g, beta)
//     into ONE single dispatch (80 TGs x 128 threads = 10,240 threads, 7 regs/thread)
//     with ZERO Q/K buf_qkv_conv_ traffic!
// ============================================================================
kernel void qwen_gdn_prework_fused(
    const device float*  qkv_in         [[buffer(0)]],
    const device ushort* conv_w         [[buffer(1)]],
    const device float*  conv_state_in  [[buffer(2)]],
    device float*        conv_state_out [[buffer(3)]],
    device float*        qkv_conv_out   [[buffer(4)]], // only V channels written
    device float*        q_out          [[buffer(5)]],
    device float*        k_out          [[buffer(6)]],
    const device float*  a_in           [[buffer(7)]],
    const device float*  b_in           [[buffer(8)]],
    const device ushort* A_log          [[buffer(9)]],
    const device ushort* dt_bias        [[buffer(10)]],
    device float*        g_out          [[buffer(11)]],
    device float*        beta_out       [[buffer(12)]],
    constant GdnPrepParams& p           [[buffer(13)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint sg                             [[simdgroup_index_in_threadgroup]]
) {
    const uint key_dim = p.Hk * p.Dk;
    if (tgid < 2u * p.Hk) {
        bool is_k = (tgid >= p.Hk);
        uint hk = is_k ? (tgid - p.Hk) : tgid;
        uint c  = (is_k ? key_dim : 0u) + hk * p.Dk + tid;

        float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
        float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
        float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
        float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
        float s0 = conv_state_in[0u * p.C + c];
        float s1 = conv_state_in[1u * p.C + c];
        float s2 = conv_state_in[2u * p.C + c];

        const float inv_scale = rsqrt(float(p.Dk));
        const float qk_scale  = is_k ? inv_scale : (inv_scale * inv_scale);
        device float* out_base = is_k ? k_out : q_out;
        threadgroup float smem[4];

        for (uint t = 0; t < p.M; t++) {
            float x = qkv_in[(ulong)t * p.C + c];
            float sv = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
            s0 = s1; s1 = s2; s2 = x;

            float sq = simd_sum(sv * sv);
            if (lane == 0u) smem[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = smem[0] + smem[1] + smem[2] + smem[3];
            float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
            out_base[((ulong)t * p.Hk + hk) * p.Dk + tid] = sv * mul;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        conv_state_out[0u * p.C + c] = s0;
        conv_state_out[1u * p.C + c] = s1;
        conv_state_out[2u * p.C + c] = s2;
        return;
    }

    uint hv = tgid - 2u * p.Hk;
    if (hv >= p.Hv) return;

    uint c = 2u * key_dim + hv * p.Dk + tid; // Dv == Dk == 128
    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
    float s0 = conv_state_in[0u * p.C + c];
    float s1 = conv_state_in[1u * p.C + c];
    float s2 = conv_state_in[2u * p.C + c];

    float neg_ea = 0.0f, dtb = 0.0f;
    if (tid == 0u) {
        neg_ea = -exp(bf16_to_f32(A_log[hv]));
        dtb    = bf16_to_f32(dt_bias[hv]);
    }

    for (uint t = 0; t < p.M; t++) {
        float x = qkv_in[(ulong)t * p.C + c];
        qkv_conv_out[(ulong)t * p.C + c] = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
        s0 = s1; s1 = s2; s2 = x;
        if (tid == 0u) {
            ulong gi = (ulong)t * p.Hv + hv;
            g_out[gi]    = exp(neg_ea * softplus_f32(a_in[gi] + dtb));
            beta_out[gi] = sigmoid_f32(b_in[gi]);
        }
    }

    conv_state_out[0u * p.C + c] = s0;
    conv_state_out[1u * p.C + c] = s1;
    conv_state_out[2u * p.C + c] = s2;
}

// ============================================================================
// 3. GatedDeltaNet Recurrence Kernel (4x-dv Vectorized!)
//    1 simdgroup (32 threads) handles 4 consecutive dv values -> 4x fewer q/k/g/beta
//    reads, coalesced float4 v_val & y_out, 16 state floats/thread in registers!
//    state: [Hv, Dv, Dk] (f32)
//    y_out: [M, Hv, Dv] (f32)
// ============================================================================

struct GdnRecParams {
    uint M;
    uint Hk;
    uint Hv;
    uint Dk;
    uint Dv;
    uint C;      // conv_dim for v offset inside qkv_conv
};

inline float4 bf16x4_to_f32x4(ushort4 b) {
    return as_type<float4>(uint4(b) << 16u);
}

inline ushort4 f32x4_to_bf16x4(float4 v) {
    uint4 u = as_type<uint4>(v);
    u += 0x7FFFu + ((u >> 16u) & 1u);
    return ushort4(u >> 16u);
}

// Round 17: BF16 ssm_state (exact FP32 exponent range, 2x smaller state & 2x bandwidth, FP32 register math!)
kernel void qwen_gdn_recurrence(
    const device float*  q_in      [[buffer(0)]],
    const device float*  k_in      [[buffer(1)]],
    const device float*  qkv_conv  [[buffer(2)]],
    const device float*  g_in      [[buffer(3)]],
    const device float*  beta_in   [[buffer(4)]],
    const device ushort* state_in  [[buffer(5)]],
    device float*        y_out     [[buffer(6)]],
    constant GdnRecParams& p       [[buffer(7)]],
    device ushort*       state_out [[buffer(8)]],
    uint3 gid                      [[thread_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]]
) {
    uint dv0 = gid.y * 4u;
    uint hv  = gid.z;
    if (dv0 >= p.Dv || hv >= p.Hv) return;

    uint hk = hv / (p.Hv / p.Hk);
    uint v_offset = 2u * p.Hk * p.Dk + hv * p.Dv + dv0;
    uint base_k = lane * 4u;

    ulong st_base = ((ulong)hv * p.Dv + dv0) * p.Dk + base_k;
    float4 s0 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 0u * p.Dk)));
    float4 s1 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 1u * p.Dk)));
    float4 s2 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 2u * p.Dk)));
    float4 s3 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 3u * p.Dk)));

    for (uint t = 0; t < p.M; t++) {
        float g    = g_in[(ulong)t * p.Hv + hv];
        float beta = beta_in[(ulong)t * p.Hv + hv];

        float4 kv = *((const device float4*)(k_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
        float4 qv = *((const device float4*)(q_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
        float4 vv = *((const device float4*)(qkv_conv + (ulong)t * p.C + v_offset));

        s0 *= g; s1 *= g; s2 *= g; s3 *= g;

        float d0 = simd_sum(dot(s0, kv));
        float d1 = simd_sum(dot(s1, kv));
        float d2 = simd_sum(dot(s2, kv));
        float d3 = simd_sum(dot(s3, kv));

        float del0 = (vv.x - d0) * beta;
        float del1 = (vv.y - d1) * beta;
        float del2 = (vv.z - d2) * beta;
        float del3 = (vv.w - d3) * beta;

        s0 = fma(kv, float4(del0), s0);
        s1 = fma(kv, float4(del1), s1);
        s2 = fma(kv, float4(del2), s2);
        s3 = fma(kv, float4(del3), s3);

        float o0 = simd_sum(dot(s0, qv));
        float o1 = simd_sum(dot(s1, qv));
        float o2 = simd_sum(dot(s2, qv));
        float o3 = simd_sum(dot(s3, qv));

        if (lane == 0u) {
            *((device float4*)(y_out + ((ulong)t * p.Hv + hv) * p.Dv + dv0)) = float4(o0, o1, o2, o3);
        }
    }

    *((device ushort4*)(state_out + st_base + 0u * p.Dk)) = f32x4_to_bf16x4(s0);
    *((device ushort4*)(state_out + st_base + 1u * p.Dk)) = f32x4_to_bf16x4(s1);
    *((device ushort4*)(state_out + st_base + 2u * p.Dk)) = f32x4_to_bf16x4(s2);
    *((device ushort4*)(state_out + st_base + 3u * p.Dk)) = f32x4_to_bf16x4(s3);
}

// ============================================================================
// 4. GatedDeltaNet: Gated RMSNorm (rmsnorm(y, weight) * silu(z))
//    1 simdgroup (32 threads) per (hv, t), Dv=128 -> 4 elements/lane
// ============================================================================

struct GdnNormParams {
    uint M;
    uint Hv;
    uint Dv;
    float eps;
};

kernel void qwen_gdn_rmsnorm_gated(
    const device float*  y_in   [[buffer(0)]],
    const device float*  z_in   [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant GdnNormParams& p   [[buffer(4)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint hv = tgid.x;
    uint t  = tgid.y;
    if (hv >= p.Hv || t >= p.M) return;

    ulong off = ((ulong)t * p.Hv + hv) * p.Dv;
    const device float* yp = y_in + off;
    const device float* zp = z_in + off;
    device float* op = out + off;

    float y0 = yp[lane], y1 = yp[lane + 32u], y2 = yp[lane + 64u], y3 = yp[lane + 96u];
    float sq = fma(y0, y0, fma(y1, y1, fma(y2, y2, y3 * y3)));
    float inv_rms = rsqrt(simd_sum(sq) / float(p.Dv) + p.eps);

    op[lane]       = y0 * inv_rms * bf16_to_f32(weight[lane])       * silu_f32(zp[lane]);
    op[lane + 32u] = y1 * inv_rms * bf16_to_f32(weight[lane + 32u]) * silu_f32(zp[lane + 32u]);
    op[lane + 64u] = y2 * inv_rms * bf16_to_f32(weight[lane + 64u]) * silu_f32(zp[lane + 64u]);
    op[lane + 96u] = y3 * inv_rms * bf16_to_f32(weight[lane + 96u]) * silu_f32(zp[lane + 96u]);
}

// FP16-output variant feeding 13.5 TFLOPS qmm_q4_g64_tiled_f16x directly (halves output bandwidth!)
kernel void qwen_gdn_rmsnorm_gated_f16(
    const device float*  y_in   [[buffer(0)]],
    const device float*  z_in   [[buffer(1)]],
    const device ushort* weight [[buffer(2)]],
    device half*         out    [[buffer(3)]],
    constant GdnNormParams& p   [[buffer(4)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint hv = tgid.x;
    uint t  = tgid.y;
    if (hv >= p.Hv || t >= p.M) return;

    ulong off = ((ulong)t * p.Hv + hv) * p.Dv;
    const device float* yp = y_in + off;
    const device float* zp = z_in + off;
    device half* op = out + off;

    float y0 = yp[lane], y1 = yp[lane + 32u], y2 = yp[lane + 64u], y3 = yp[lane + 96u];
    float sq = fma(y0, y0, fma(y1, y1, fma(y2, y2, y3 * y3)));
    float inv_rms = rsqrt(simd_sum(sq) / float(p.Dv) + p.eps);

    op[lane]       = half(y0 * inv_rms * bf16_to_f32(weight[lane])       * silu_f32(zp[lane]));
    op[lane + 32u] = half(y1 * inv_rms * bf16_to_f32(weight[lane + 32u]) * silu_f32(zp[lane + 32u]));
    op[lane + 64u] = half(y2 * inv_rms * bf16_to_f32(weight[lane + 64u]) * silu_f32(zp[lane + 64u]));
    op[lane + 96u] = half(y3 * inv_rms * bf16_to_f32(weight[lane + 96u]) * silu_f32(zp[lane + 96u]));
}

// ============================================================================
// 5. Full Attention Prep:
//    - Split q_proj [M, Hq, 2*D] into Q [M, Hq, D] + gate [M, Hq, D]
//    - Apply q_norm + partial NeoX RoPE (rotary_dim=64) to Q
//    - Apply k_norm + partial NeoX RoPE (rotary_dim=64) to K, write K & V (half) to KV cache
// ============================================================================

struct FullAttnParams {
    uint  M;
    uint  Hq;
    uint  Hkv;
    uint  D;            // 256
    uint  rotary_dim;   // 64
    uint  pos;          // starting KV cache position
    uint  kv_len;       // total KV length after appending M tokens (pos + M)
    float rope_theta;
    float eps;
    float scale;        // 1/sqrt(D)
    uint  num_splits;
    int   rope_offset;  // M-RoPE delta after images (0 for pure text!)
};

kernel void qwen_full_attn_q_prep(
    const device float*  q_proj  [[buffer(0)]],
    const device ushort* q_norm  [[buffer(1)]],
    device float*        q_out   [[buffer(2)]],
    device float*        gate_out[[buffer(3)]],
    constant FullAttnParams& p   [[buffer(4)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    const device float* src = q_proj + ((ulong)t * p.Hq + h) * (2u * p.D);
    device float* qdst = q_out + ((ulong)t * p.Hq + h) * p.D;
    device float* gdst = gate_out + ((ulong)t * p.Hq + h) * p.D;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = src[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        gdst[idx] = src[p.D + idx];
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(q_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1; // 32
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle = float(int(p.pos + t) + p.rope_offset) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        qdst[i * 32u + lane] = v[i];
    }
}

kernel void qwen_full_attn_kv_prep(
    const device float*  k_proj   [[buffer(0)]],
    const device float*  v_proj   [[buffer(1)]],
    const device ushort* k_norm   [[buffer(2)]],
    device half*         k_cache  [[buffer(3)]],  // [max_seq, Hkv, D]
    device half*         v_cache  [[buffer(4)]],  // [max_seq, Hkv, D]
    constant FullAttnParams& p    [[buffer(5)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    const device float* ksrc = k_proj + ((ulong)t * p.Hkv + h) * p.D;
    const device float* vsrc = v_proj + ((ulong)t * p.Hkv + h) * p.D;
    ulong cache_off = ((ulong)(p.pos + t) * p.Hkv + h) * p.D;
    device half* kdst = k_cache + cache_off;
    device half* vdst = v_cache + cache_off;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = ksrc[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        vdst[idx] = half(vsrc[idx]);
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(k_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle = float(int(p.pos + t) + p.rope_offset) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        kdst[i * 32u + lane] = half(v[i]);
    }
}

// 3D Interleaved M-RoPE variants (mrope_section=[11, 11, 10] -> axis = lane % 3!)
// pos3d: int [M, 3] = (pos_t, pos_h, pos_w) per token in this chunk!
kernel void qwen_full_attn_q_prep_mrope(
    const device float*  q_proj  [[buffer(0)]],
    const device ushort* q_norm  [[buffer(1)]],
    device float*        q_out   [[buffer(2)]],
    device float*        gate_out[[buffer(3)]],
    constant FullAttnParams& p   [[buffer(4)]],
    const device int*    pos3d   [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    const device float* src = q_proj + ((ulong)t * p.Hq + h) * (2u * p.D);
    device float* qdst = q_out + ((ulong)t * p.Hq + h) * p.D;
    device float* gdst = gate_out + ((ulong)t * p.Hq + h) * p.D;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = src[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        gdst[idx] = src[p.D + idx];
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(q_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1; // 32
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        uint axis = lane % 3u;
        float angle = float(pos3d[t * 3u + axis]) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        qdst[i * 32u + lane] = v[i];
    }
}

kernel void qwen_full_attn_kv_prep_mrope(
    const device float*  k_proj   [[buffer(0)]],
    const device float*  v_proj   [[buffer(1)]],
    const device ushort* k_norm   [[buffer(2)]],
    device half*         k_cache  [[buffer(3)]],
    device half*         v_cache  [[buffer(4)]],
    constant FullAttnParams& p    [[buffer(5)]],
    const device int*    pos3d    [[buffer(6)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    const device float* ksrc = k_proj + ((ulong)t * p.Hkv + h) * p.D;
    const device float* vsrc = v_proj + ((ulong)t * p.Hkv + h) * p.D;
    ulong cache_off = ((ulong)(p.pos + t) * p.Hkv + h) * p.D;
    device half* kdst = k_cache + cache_off;
    device half* vdst = v_cache + cache_off;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = ksrc[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        vdst[idx] = half(vsrc[idx]);
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(k_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        uint axis = lane % 3u;
        float angle = float(pos3d[t * 3u + axis]) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        kdst[i * 32u + lane] = half(v[i]);
    }
}

struct ScatterEmbedParams {
    uint count;
    uint D;
};

// Scatter pre-encoded vision embeddings onto <|image_pad|> rows of hid [M, D]
kernel void qwen_scatter_embeds(
    const device float*  vis_emb  [[buffer(0)]], // [count, D]
    const device uint*   row_ids  [[buffer(1)]], // [count]
    device float*        hid      [[buffer(2)]], // [M, D]
    constant ScatterEmbedParams& p [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint k = gid.y;
    if (d >= p.D || k >= p.count) return;
    hid[(ulong)row_ids[k] * p.D + d] = vis_emb[(ulong)k * p.D + d];
}

// ============================================================================
// 6. Flash GQA Attention + Fused Output Sigmoid Gate
//    Standard single-pass (prefill & short decode) + Split-K (long decode)
// ============================================================================

template<typename OutT>
inline void qwen_gqa_attn_gated_impl(
    const device float* q_in,
    const device float* gate_in,
    const device half*  k_cache,
    const device half*  v_cache,
    device OutT*        out,
    constant FullAttnParams& p,
    uint2 tgid,
    uint  lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device float* gp = gate_in + q_off;
    device OutT* op = out + q_off;

    float qv[8];
    float acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        qv[i] = qp[i * 32u + lane];
        acc[i] = 0.0f;
    }

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    uint limit = p.pos + t + 1u;
    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    for (uint kpos = 0; kpos < limit; kpos++) {
        const device half* kp = k_cache + (ulong)kpos * kv_stride + head_off;
        float dot = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            dot = fma(qv[i], float(kp[i * 32u + lane]), dot);
        }
        float score = simd_sum(dot) * p.scale;

        float m_new = max(m_max, score);
        float alpha = exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device half* vp = v_cache + (ulong)kpos * kv_stride + head_off;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            acc[i] = fma(acc[i], alpha, beta * float(vp[i * 32u + lane]));
        }
    }

    float inv_l = 1.0f / l_sum;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = OutT((acc[i] * inv_l) * sigmoid_f32(gp[idx]));
    }
}

kernel void qwen_gqa_attn_gated(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device float*       out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    qwen_gqa_attn_gated_impl<float>(q_in, gate_in, k_cache, v_cache, out, p, tgid, lane);
}

kernel void qwen_gqa_attn_gated_f16(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device half*        out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    qwen_gqa_attn_gated_impl<half>(q_in, gate_in, k_cache, v_cache, out, p, tgid, lane);
}

// ============================================================================
// 6b. Hardware simdgroup_half8x8 AMX FlashAttention-2 Prefill Kernel (D=256)
//     BQ=16, BK=32, 128 threads (4 simdgroups, 2x2 O-layout -> 32 regs/thread, 19.9 KB TG mem)
//     Vectorized threadgroup half4 loads, single-cycle exp2(), mask-free interior fast-path!
//     Templated for both float (standard) and half (direct 13.5 TFLOPS o_proj input!) outputs.
// ============================================================================

template<typename OutT>
inline void qwen_gqa_prefill_flash_impl(
    const device float* q_in,
    const device float* gate_in,
    const device half*  k_cache,
    const device half*  v_cache,
    device OutT*        out,
    constant FullAttnParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sQ)[16][260],
    threadgroup half  (&sKV)[32][132],
    threadgroup float (&sS)[16][32],
    threadgroup half  (&sP)[16][36],
    threadgroup float (&sAlpha)[16],
    threadgroup float (&sInvL)[16]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint t0 = tgid.x * BQ;
    uint h  = tgid.y;
    if (h >= p.Hq || t0 >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    // Step 0: Load Q[t0..t0+15, 0..255] once into sQ (128 threads x 32 floats = 8 float4/thread)
    {
        uint qr = tid >> 3;          // 0..15
        uint qc = (tid & 7u) * 32u;  // 0, 32, ..., 224
        uint gt = t0 + qr;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (gt < p.M) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)gt * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) {
                sq_dst[i] = half4(qp[i]);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2x2 simdgroup layout for O: sg_m in {0, 8}, sg_d in {0, 64}
    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;
    const bool full_q_tile = (t0 + BQ <= p.M);

    uint max_kv = p.pos + min(t0 + BQ, p.M);
    uint kr = tid >> 2;          // 0..31
    uint kc = (tid & 3u) * 32u;  // 0, 32, 64, 96
    threadgroup half4* skv_dst = (threadgroup half4*)(&sKV[kr][kc]);

    for (uint kb = 0; kb < max_kv; kb += BK) {
        uint tile_k = min(BK, max_kv - kb);

        // 1. S = Q @ K^T [16, 32] via AMX (sg computes 8 K-cols: sg*8 .. sg*8+7)
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        // d_half = 0 (cols 0..127)
        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // d_half = 1 (cols 128..255)
        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        // No extra barrier needed before writing sS (separate from sKV!)
        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Single-cycle hardware exp2() online softmax (mask-free interior fast-path!)
        bool interior = full_q_tile && (kb + BK <= p.pos + t0);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == 0u) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint q_pos = p.pos + t0 + r;
                uint k_pos = kb + lane;
                bool valid = (t0 + r < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        // Load V[:, 0..127] immediately (sKV was already free after K half 1!)
        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Rescale O0, O1 in registers + O0 += P @ V[:, 0..127]
        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row;
            O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row;
            O1[j].thread_elements()[1] *= a_row;
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4b. O1 += P @ V[:, 128..255] via AMX
        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Final normalization + fused sigmoid output gate directly from registers!
    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            sInvL[sg * 4u + rr] = (l_row[rr] > 0.0f) ? (1.0f / l_row[rr]) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gt = t0 + sg_m + fm;
    if (gt < p.M) {
        float inv_l = sInvL[sg_m + fm];
        ulong q_off = ((ulong)gt * p.Hq + h) * p.D;
        const device float* gp = gate_in + q_off;
        device OutT* op = out + q_off;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            op[c0]      = OutT((O0[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c0]));
            op[c0 + 1u] = OutT((O0[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c0 + 1u]));
            uint c1 = 128u + c0;
            op[c1]      = OutT((O1[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c1]));
            op[c1 + 1u] = OutT((O1[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c1 + 1u]));
        }
    }
}

kernel void qwen_gqa_prefill_flash(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device float*       out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  lane                  [[thread_index_in_simdgroup]],
    uint  sg                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half  sQ[16][260];
    threadgroup half  sKV[32][132];
    threadgroup float sS[16][32];
    threadgroup half  sP[16][36];
    threadgroup float sAlpha[16];
    threadgroup float sInvL[16];
    qwen_gqa_prefill_flash_impl<float>(q_in, gate_in, k_cache, v_cache, out, p, tgid, tid, lane, sg,
                                       sQ, sKV, sS, sP, sAlpha, sInvL);
}

kernel void qwen_gqa_prefill_flash_f16(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device half*        out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  lane                  [[thread_index_in_simdgroup]],
    uint  sg                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half  sQ[16][260];
    threadgroup half  sKV[32][132];
    threadgroup float sS[16][32];
    threadgroup half  sP[16][36];
    threadgroup float sAlpha[16];
    threadgroup float sInvL[16];
    qwen_gqa_prefill_flash_impl<half>(q_in, gate_in, k_cache, v_cache, out, p, tgid, tid, lane, sg,
                                      sQ, sKV, sS, sP, sAlpha, sInvL);
}

// GQA-Pair (2 GQA heads x 8 Q-tokens = 16 Q-rows!) AMX FlashAttention-2 Prefill (FP16 output)
// Exact same 19.9 KB TG mem & 128 threads as single-head BQ16 (full 2x occupancy!), 1.97x faster on causal diagonal!
kernel void qwen_gqa_prefill_flash_pair_f16(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device half*        out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  lane                  [[thread_index_in_simdgroup]],
    uint  sg                    [[simdgroup_index_in_threadgroup]]
) {
    uint t0 = tgid.x * 8u;
    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint hkv = tgid.y / pairs_per_kv;
    uint gp_idx = tgid.y % pairs_per_kv;
    uint h0  = hkv * gqa + gp_idx * 2u;
    uint h1  = h0 + 1u;
    bool h1_ok = (gp_idx * 2u + 1u < gqa);
    if (hkv >= p.Hkv || t0 >= p.M) return;

    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    threadgroup half  sQ[16][260];
    threadgroup half  sKV[32][132];
    threadgroup float sS[16][32];
    threadgroup half  sP[16][36];
    threadgroup float sAlpha[16];
    threadgroup float sInvL[16];

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint lh = qr >> 3;
        uint lt = qr & 7u;
        uint gt = t0 + lt;
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (gt < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)gt * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;
    const bool full_q_tile = (t0 + 8u <= p.M);
    uint max_kv = p.pos + min(t0 + 8u, p.M);

    uint kr = tid >> 2;
    uint kc = (tid & 3u) * 32u;
    threadgroup half4* skv_dst = (threadgroup half4*)(&sKV[kr][kc]);

    for (uint kb = 0; kb < max_kv; kb += 32u) {
        uint tile_k = min(32u, max_kv - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = full_q_tile && (kb + 32u <= p.pos + t0);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == 0u) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint lt = r & 7u;
                uint q_pos = p.pos + t0 + lt;
                uint k_pos = kb + lane;
                bool valid = (t0 + lt < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) sInvL[sg * 4u + rr] = (l_row[rr] > 0.0f) ? (1.0f / l_row[rr]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = sg_m + fm;
    uint lh = row >> 3;
    uint lt = row & 7u;
    uint gt = t0 + lt;
    uint h  = (lh == 0u) ? h0 : h1;
    if (gt < p.M && (lh == 0u || h1_ok)) {
        float inv_l = sInvL[row];
        ulong q_off = ((ulong)gt * p.Hq + h) * p.D;
        const device float* gp = gate_in + q_off;
        device half* op = out + q_off;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            op[c0]      = half((O0[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c0]));
            op[c0 + 1u] = half((O0[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c0 + 1u]));
            uint c1 = 128u + c0;
            op[c1]      = half((O1[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c1]));
            op[c1 + 1u] = half((O1[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c1 + 1u]));
        }
    }
}

// Split-K Pass 1 for M=1 long-context decode (saturates all 40 M4 Max cores)
kernel void qwen_gqa_decode_splitk_p1(
    const device float* q_in     [[buffer(0)]],
    const device half*  k_cache  [[buffer(1)]],
    const device half*  v_cache  [[buffer(2)]],
    device float*       part_o   [[buffer(3)]], // [Hq, num_splits, D]
    device float*       part_m   [[buffer(4)]], // [Hq, num_splits]
    device float*       part_l   [[buffer(5)]], // [Hq, num_splits]
    constant FullAttnParams& p   [[buffer(6)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint s = tgid.y;
    if (h >= p.Hq || s >= p.num_splits) return;

    uint hkv = h / (p.Hq / p.Hkv);
    const device float* qp = q_in + (ulong)h * p.D;

    float qv[8];
    float acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        qv[i] = qp[i * 32u + lane];
        acc[i] = 0.0f;
    }

    uint total = p.kv_len;
    uint chunk = (total + p.num_splits - 1u) / p.num_splits;
    uint start = s * chunk;
    uint end   = min(total, start + chunk);

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    for (uint kpos = start; kpos < end; kpos++) {
        const device half* kp = k_cache + (ulong)kpos * kv_stride + head_off;
        float dot = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            dot = fma(qv[i], float(kp[i * 32u + lane]), dot);
        }
        float score = simd_sum(dot) * p.scale;

        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device half* vp = v_cache + (ulong)kpos * kv_stride + head_off;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            acc[i] = fma(acc[i], alpha, beta * float(vp[i * 32u + lane]));
        }
    }

    ulong slot = (ulong)h * p.num_splits + s;
    device float* po = part_o + slot * p.D;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        po[i * 32u + lane] = acc[i];
    }
    if (lane == 0u) {
        part_m[slot] = m_max;
        part_l[slot] = l_sum;
    }
}

kernel void qwen_gqa_decode_splitk_p2(
    const device float* part_o   [[buffer(0)]],
    const device float* part_m   [[buffer(1)]],
    const device float* part_l   [[buffer(2)]],
    const device float* gate_in  [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant FullAttnParams& p   [[buffer(5)]],
    uint  h                      [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    if (h >= p.Hq) return;

    float m_global = -INFINITY;
    for (uint s = 0; s < p.num_splits; s++) {
        float m = part_m[(ulong)h * p.num_splits + s];
        if (m > m_global) m_global = m;
    }

    float acc[8] = {0,0,0,0,0,0,0,0};
    float l_global = 0.0f;
    for (uint s = 0; s < p.num_splits; s++) {
        ulong slot = (ulong)h * p.num_splits + s;
        float l = part_l[slot];
        if (l == 0.0f) continue;
        float w = exp(part_m[slot] - m_global);
        l_global += l * w;
        const device float* po = part_o + slot * p.D;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            acc[i] = fma(po[i * 32u + lane], w, acc[i]);
        }
    }

    float inv_l = 1.0f / l_global;
    const device float* gp = gate_in + (ulong)h * p.D;
    device float* op = out + (ulong)h * p.D;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = (acc[i] * inv_l) * sigmoid_f32(gp[idx]);
    }
}

// ============================================================================
// 6c. GQA-Pair + M-Batched (M = 1..8) + Split-K AMX FlashAttention-2 Verify
//     1 threadgroup handles 2 GQA heads x up to 8 verify tokens = 16 query rows
//     sharing ONE KV tile load (16x KV bandwidth reuse vs scalar qwen_gqa_attn_gated!)
//     19.9 KB TG mem, 128 threads, single-cycle exp2(), mask-free interior fast-path!
//     Eliminates +54 ms (8k) / +112 ms (16k) MTP verify attention stall!
// ============================================================================

kernel void qwen_gqa_verify_splitk_p1(
    const device float* q_in     [[buffer(0)]],
    const device half*  k_cache  [[buffer(1)]],
    const device half*  v_cache  [[buffer(2)]],
    device float*       part_o   [[buffer(3)]], // [M, Hq, num_splits, D]
    device float*       part_m   [[buffer(4)]], // [M, Hq, num_splits] (log2 space)
    device float*       part_l   [[buffer(5)]], // [M, Hq, num_splits]
    constant FullAttnParams& p   [[buffer(6)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_index_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint pair_idx = tgid.x;
    uint s        = tgid.y;
    if (pair_idx >= p.Hkv * pairs_per_kv || s >= p.num_splits) return;

    uint hkv   = pair_idx / pairs_per_kv;
    uint gp    = pair_idx % pairs_per_kv;
    uint h0    = hkv * gqa + gp * 2u;
    uint h1    = h0 + 1u;
    bool h1_ok = (gp * 2u + 1u < gqa);

    uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
    uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
    uint kb_start        = s * tiles_per_split * BK;
    uint kb_end          = min(p.kv_len, (s + 1u) * tiles_per_split * BK);

    if (kb_start >= kb_end) {
        if (tid < BQ) {
            uint lh = tid >> 3;
            uint t  = tid & 7u;
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = -INFINITY;
                part_l[slot] = 0.0f;
            }
        }
        return;
    }

    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    threadgroup half  sQ[BQ][260];   // 8.32 KB: rows 0..7 = h0[t=0..7], rows 8..15 = h1[t=0..7]
    threadgroup half  sKV[BK][132];  // 8.45 KB: 32 x 128 strip of K or V
    threadgroup float sS[BQ][BK];    // 2.05 KB
    threadgroup half  sP[BQ][36];    // 1.15 KB
    threadgroup float sAlpha[BQ];

    // Step 0: Load Q for (h0, t=0..M-1) and (h1, t=0..M-1) once into sQ
    {
        uint qr = tid >> 3;          // 0..15
        uint qc = (tid & 7u) * 32u;  // 0, 32, ..., 224
        uint lh = qr >> 3;
        uint t  = qr & 7u;
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (t < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)t * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;

    uint kr = tid >> 2;          // 0..31
    uint kc = (tid & 3u) * 32u;  // 0, 32, 64, 96
    threadgroup half4* skv_dst = (threadgroup half4*)(&sKV[kr][kc]);

    for (uint kb = kb_start; kb < kb_end; kb += BK) {
        uint tile_k = min(BK, kb_end - kb);

        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        // K d_half = 0
        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K d_half = 1
        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax in log2 space (interior fast-path for all tiles strictly before p.pos!)
        bool interior = (kb + BK <= p.pos) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == kb_start) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint t = r & 7u;
                uint q_pos = p.pos + t;
                uint k_pos = kb + lane;
                bool valid = (t < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        // Load V d_half = 0
        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row;
            O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row;
            O1[j].thread_elements()[1] *= a_row;
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V d_half = 1
        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial m, l (log2 space)
    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            uint r  = sg * 4u + rr;
            uint lh = r >> 3;
            uint t  = r & 7u;
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = m_row[rr];
                part_l[slot] = l_row[rr];
            }
        }
    }

    // Write partial unnormalized O directly from registers!
    uint row = sg_m + fm;
    uint lh  = row >> 3;
    uint t   = row & 7u;
    uint h   = (lh == 0u) ? h0 : h1;
    if (t < p.M && (lh == 0u || h1_ok)) {
        ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
        device float* po = part_o + slot * p.D;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            po[c0]           = O0[j].thread_elements()[0];
            po[c0 + 1u]      = O0[j].thread_elements()[1];
            po[128u + c0]    = O1[j].thread_elements()[0];
            po[128u + c0 + 1u] = O1[j].thread_elements()[1];
        }
    }
}

template<typename OutT>
inline void qwen_gqa_verify_splitk_p2_impl(
    const device float* part_o,
    const device float* part_m,
    const device float* part_l,
    const device float* gate_in,
    device OutT*        out,
    constant FullAttnParams& p,
    uint2 tgid,
    uint  lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    ulong base_slot = ((ulong)t * p.Hq + h) * p.num_splits;
    float m_global = -INFINITY;
    for (uint s = 0; s < p.num_splits; s++) {
        float m = part_m[base_slot + s];
        if (m > m_global) m_global = m;
    }

    float acc[8] = {0,0,0,0,0,0,0,0};
    float l_global = 0.0f;
    for (uint s = 0; s < p.num_splits; s++) {
        ulong slot = base_slot + s;
        float l = part_l[slot];
        if (l == 0.0f) continue;
        float w = exp2(part_m[slot] - m_global);
        l_global += l * w;
        const device float* po = part_o + slot * p.D;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) {
            acc[i] = fma(po[i * 32u + lane], w, acc[i]);
        }
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* gp = gate_in + q_off;
    device OutT* op = out + q_off;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = OutT((acc[i] * inv_l) * sigmoid_f32(gp[idx]));
    }
}

kernel void qwen_gqa_verify_splitk_p2(
    const device float* part_o   [[buffer(0)]],
    const device float* part_m   [[buffer(1)]],
    const device float* part_l   [[buffer(2)]],
    const device float* gate_in  [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant FullAttnParams& p   [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    qwen_gqa_verify_splitk_p2_impl<float>(part_o, part_m, part_l, gate_in, out, p, tgid, lane);
}

kernel void qwen_gqa_verify_splitk_p2_f16(
    const device float* part_o   [[buffer(0)]],
    const device float* part_m   [[buffer(1)]],
    const device float* part_l   [[buffer(2)]],
    const device float* gate_in  [[buffer(3)]],
    device half*        out      [[buffer(4)]],
    constant FullAttnParams& p   [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    qwen_gqa_verify_splitk_p2_impl<half>(part_o, part_m, part_l, gate_in, out, p, tgid, lane);
}

// ============================================================================
// 7. MTP Drafter Input Fusion:
//    out[row, 0..D-1]   = rmsnorm(embed[row], w_emb)
//    out[row, D..2*D-1] = rmsnorm(hidden[row], w_hid)
// ============================================================================

struct MtpConcatParams {
    uint  M;
    uint  D;
    float eps;
};

kernel void qwen_mtp_concat_norm(
    const device float*  emb   [[buffer(0)]],
    const device float*  hid   [[buffer(1)]],
    const device ushort* w_emb [[buffer(2)]],
    const device ushort* w_hid [[buffer(3)]],
    device float*        out   [[buffer(4)]], // [M, 2*D]
    constant MtpConcatParams& p [[buffer(5)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* ep = emb + (ulong)row * p.D;
    const device float* hp = hid + (ulong)row * p.D;
    device float* op = out + (ulong)row * (2u * p.D);

    float sq_e = 0.0f, sq_h = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float ve = ep[i], vh = hp[i];
        sq_e = fma(ve, ve, sq_e);
        sq_h = fma(vh, vh, sq_h);
    }
    sq_e = simd_sum(sq_e);
    sq_h = simd_sum(sq_h);

    threadgroup float se[32], sh[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) { se[sg] = sq_e; sh[sg] = sq_h; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float te = simd_sum((lane < nsg) ? se[lane] : 0.0f);
        float th = simd_sum((lane < nsg) ? sh[lane] : 0.0f);
        if (lane == 0u) {
            se[0] = rsqrt(te / float(p.D) + p.eps);
            sh[0] = rsqrt(th / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ire = se[0], irh = sh[0];

    for (uint i = tid; i < p.D; i += tg_size) {
        op[i]       = ep[i] * ire * bf16_to_f32(w_emb[i]);
        op[p.D + i] = hp[i] * irh * bf16_to_f32(w_hid[i]);
    }
}

// Round 19: Fused Q4 Embed + Dual Pre-FC RMSNorm -> FP16 [M, 2*D] for FP16-X mtp.fc!
kernel void qwen_mtp_embed_concat_f16(
    const device uint*   emb_W [[buffer(0)]],
    const device ushort* emb_S [[buffer(1)]],
    const device ushort* emb_B [[buffer(2)]],
    const device int*    ids   [[buffer(3)]],
    const device float*  hid   [[buffer(4)]],
    const device ushort* w_emb [[buffer(5)]],
    const device ushort* w_hid [[buffer(6)]],
    device half*         out   [[buffer(7)]], // FP16 [M, 2*D]
    constant MtpConcatParams& p [[buffer(8)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const uint tok = uint(ids[row]);
    const uint KW = p.D >> 3;
    const uint KG = p.D >> 6;
    const device uint*   wr = emb_W + (ulong)tok * KW;
    const device ushort* sr = emb_S + (ulong)tok * KG;
    const device ushort* br = emb_B + (ulong)tok * KG;
    const device float*  hp = hid + (ulong)row * p.D;
    device half*         op = out + (ulong)row * (2u * p.D);

    threadgroup float se[32], sh[32];
    const uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;

    if (p.D == 5120u && tg_size == 256u) {
        float ve_reg[20], vh_reg[20];
        float sq_e = 0.0f, sq_h = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint i = k * 256u + tid;
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            float vh = hp[i];
            ve_reg[k] = ve;
            vh_reg[k] = vh;
            sq_e = fma(ve, ve, sq_e);
            sq_h = fma(vh, vh, sq_h);
        }
        sq_e = simd_sum(sq_e);
        sq_h = simd_sum(sq_h);
        if (lane == 0u) { se[sg] = sq_e; sh[sg] = sq_h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float te = simd_sum((lane < nsg) ? se[lane] : 0.0f);
            float th = simd_sum((lane < nsg) ? sh[lane] : 0.0f);
            if (lane == 0u) {
                se[0] = rsqrt(te / float(p.D) + p.eps);
                sh[0] = rsqrt(th / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ire = se[0], irh = sh[0];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint i = k * 256u + tid;
            op[i]       = half(ve_reg[k] * ire * bf16_to_f32(w_emb[i]));
            op[p.D + i] = half(vh_reg[k] * irh * bf16_to_f32(w_hid[i]));
        }
    } else {
        float sq_e = 0.0f, sq_h = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            float vh = hp[i];
            sq_e = fma(ve, ve, sq_e);
            sq_h = fma(vh, vh, sq_h);
        }
        sq_e = simd_sum(sq_e);
        sq_h = simd_sum(sq_h);
        if (lane == 0u) { se[sg] = sq_e; sh[sg] = sq_h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float te = simd_sum((lane < nsg) ? se[lane] : 0.0f);
            float th = simd_sum((lane < nsg) ? sh[lane] : 0.0f);
            if (lane == 0u) {
                se[0] = rsqrt(te / float(p.D) + p.eps);
                sh[0] = rsqrt(th / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ire = se[0], irh = sh[0];
        for (uint i = tid; i < p.D; i += tg_size) {
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            op[i]       = half(ve * ire * bf16_to_f32(w_emb[i]));
            op[p.D + i] = half(hp[i] * irh * bf16_to_f32(w_hid[i]));
        }
    }
}

// ============================================================================
// 8. Native Zero-Copy IOSurface ANE MLP Packing & Output Addition Kernels
//    - qwen_rmsnorm_ane_pack: Fuses post_attn RMSNorm + optional inv1 SmoothQuant
//      + per-row scale r = max|x'| + 1e-4 + [x'/r | r | 0] FP16 straight into
//      the ANE CVPixelBuffer/IOSurface MTLTexture (ZERO CPU memcpy!)
//    - qwen_ane_add_output: Adds ANE FP16 down_proj directly from output IOSurface
//      onto buf_hidden_!
// ============================================================================

struct AnePackParams {
    uint  M;          // active rows in this sub-block (<= block_rows, e.g. 256)
    uint  block_rows; // ANE IOSurface height (e.g. 256)
    uint  D;          // hidden_size (5120)
    uint  PW;         // packed width (5184)
    uint  has_inv1;   // 0 if inv1 == 1.0 everywhere, 1 if inv1 buffer provided
    float eps;
};

kernel void qwen_rmsnorm_ane_pack(
    const device float*  hid    [[buffer(0)]],
    const device ushort* norm_w [[buffer(1)]],
    const device float*  inv1   [[buffer(2)]],
    texture2d<half, access::write> ane_in [[texture(0)]],
    constant AnePackParams& p   [[buffer(3)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    if (row >= p.block_rows) return;

    if (row >= p.M) {
        for (uint i = tid; i < p.PW; i += tg_size) {
            ane_in.write(half4(0.0h), uint2(i, row));
        }
        return;
    }

    const device float* hp = hid + (ulong)row * p.D;

    // D = 5120, tg_size = 256 -> exactly 20 elements per thread in registers!
    float vals[20];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        float v = hp[idx];
        vals[k] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);

    threadgroup float smem_sq[32];
    threadgroup float smem_max[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) smem_sq[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? smem_sq[lane] : 0.0f);
        if (lane == 0u) smem_sq[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = smem_sq[0];

    float amax = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        float w = bf16_to_f32(norm_w[idx]);
        float s = p.has_inv1 ? inv1[idx] : 1.0f;
        float v = vals[k] * inv_rms * w * s;
        vals[k] = v;
        amax = max(amax, fabs(v));
    }
    amax = simd_max(amax);
    if (lane == 0u) smem_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_max((lane < nsg) ? smem_max[lane] : 0.0f);
        if (lane == 0u) smem_max[0] = tot + 1e-4f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r_scale = smem_max[0];
    float inv_r = 1.0f / r_scale;

    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        ane_in.write(half4(half(vals[k] * inv_r), 0.0h, 0.0h, 0.0h), uint2(idx, row));
    }
    if (tid == 0u) {
        ane_in.write(half4(half(r_scale), 0.0h, 0.0h, 0.0h), uint2(p.D, row));
    }
    if (tid < (p.PW - p.D - 1u)) {
        ane_in.write(half4(0.0h), uint2(p.D + 1u + tid, row));
    }
}

struct AneAddParams {
    uint M;
    uint D;
    uint num_pieces;
};

kernel void qwen_ane_add_output(
    texture2d<half, access::read> ane_out0 [[texture(0)]],
    texture2d<half, access::read> ane_out1 [[texture(1)]],
    device float*                 hid      [[buffer(0)]],
    constant AneAddParams&        p        [[buffer(1)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= p.D || row >= p.M) return;

    float v = float(ane_out0.read(uint2(col, row)).x);
    if (p.num_pieces > 1u) {
        v += float(ane_out1.read(uint2(col, row)).x);
    }
    hid[(ulong)row * p.D + col] += v;
}

struct AneAddNormParams {
    uint  M;
    uint  D;
    uint  num_pieces;
    float eps;
};

// Fused ANE output residual add + next layer's input RMSNorm in ONE pass!
// Eliminates 126 kernel launches, 126 barriers, and 2.5 GB of hid_buf traffic per 1024-tok chunk!
kernel void qwen_ane_add_and_rmsnorm(
    texture2d<half, access::read> ane_out0 [[texture(0)]],
    texture2d<half, access::read> ane_out1 [[texture(1)]],
    device float*                 hid      [[buffer(0)]],
    const device ushort*          norm_w   [[buffer(1)]],
    device float*                 normed   [[buffer(2)]],
    constant AneAddNormParams&    p        [[buffer(3)]],
    uint row                               [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint tg_size                           [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    device float* hp = hid + (ulong)row * p.D;
    device float* np = normed + (ulong)row * p.D;

    // D = 5120, tg_size = 256 -> 20 exact elements per thread in registers!
    float vals[20];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        float av = float(ane_out0.read(uint2(idx, row)).x);
        if (p.num_pieces > 1u) av += float(ane_out1.read(uint2(idx, row)).x);
        float h = hp[idx] + av;
        hp[idx] = h;
        vals[k] = h;
        sq = fma(h, h, sq);
    }
    sq = simd_sum(sq);

    threadgroup float smem[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) smem[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
        if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = smem[0];

    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        np[idx] = vals[k] * inv_rms * bf16_to_f32(norm_w[idx]);
    }
}

// FP16-output variant: adds ANE output + writes FP16 normed directly for 13.5 TFLOPS qmm_q4_g64_tiled_f16x!
kernel void qwen_ane_add_and_rmsnorm_f16(
    texture2d<half, access::read> ane_out0 [[texture(0)]],
    texture2d<half, access::read> ane_out1 [[texture(1)]],
    device float*                 hid      [[buffer(0)]],
    const device ushort*          norm_w   [[buffer(1)]],
    device half*                  normed   [[buffer(2)]],
    constant AneAddNormParams&    p        [[buffer(3)]],
    uint row                               [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint tg_size                           [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    device float* hp = hid + (ulong)row * p.D;
    device half*  np = normed + (ulong)row * p.D;

    float vals[20];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        float av = float(ane_out0.read(uint2(idx, row)).x);
        if (p.num_pieces > 1u) av += float(ane_out1.read(uint2(idx, row)).x);
        float h = hp[idx] + av;
        hp[idx] = h;
        vals[k] = h;
        sq = fma(h, h, sq);
    }
    sq = simd_sum(sq);

    threadgroup float smem[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) smem[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
        if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = smem[0];

    #pragma clang loop unroll(full)
    for (uint k = 0; k < 20u; k++) {
        uint idx = k * 256u + tid;
        np[idx] = half(vals[k] * inv_rms * bf16_to_f32(norm_w[idx]));
    }
}

// ============================================================================
// 10. Quantized 8-bit & 4-bit Group-64 Affine KV Cache Kernels (Round 6)
//     Cuts KV RAM 1.88x (Q8) to 3.56x (Q4), unlocking 64k-128k context on 48 GB
//     and accelerating long-context decode attention bandwidth!
//     Layout (D=256, 4 groups of 64):
//       Q8: k_cache/v_cache [max_seq, Hkv, 256] uchar,  k_sb/v_sb [max_seq, Hkv, 4] half2(scale, min)
//       Q4: k_cache/v_cache [max_seq, Hkv, 128] uchar,  k_sb/v_sb [max_seq, Hkv, 4] half2(scale, min)
//           In group g (0..3), byte (g*32 + j) packs elem (64*g + j) in low nibble
//           and elem (64*g + 32 + j) in high nibble -> zero-shuffle coalesced across all kernels!
// ============================================================================

template<int KV_BITS, bool USE_MROPE>
inline void qwen_full_attn_kv_prep_quant_impl(
    const device float*  k_proj,
    const device float*  v_proj,
    const device ushort* k_norm,
    device uchar*        k_cache,
    device uchar*        v_cache,
    device float2*       k_sb,
    device float2*       v_sb,
    constant FullAttnParams& p,
    const device int*    pos3d,
    uint2 tgid,
    uint  lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    const device float* ksrc = k_proj + ((ulong)t * p.Hkv + h) * p.D;
    const device float* vsrc = v_proj + ((ulong)t * p.Hkv + h) * p.D;
    constexpr uint BYTES_PER_HEAD = (KV_BITS == 8) ? 256u : 128u;
    ulong head_idx = (ulong)(p.pos + t) * p.Hkv + h;
    device uchar*  kdst = k_cache + head_idx * BYTES_PER_HEAD;
    device uchar*  vdst = v_cache + head_idx * BYTES_PER_HEAD;
    device float2* ksbd = k_sb + head_idx * 4ul;
    device float2* vsbd = v_sb + head_idx * 4ul;

    float kv[8];
    float vv[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = ksrc[idx];
        kv[i] = x;
        sq = fma(x, x, sq);
        vv[i] = vsrc[idx];
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        kv[i] = kv[i] * inv_rms * bf16_to_f32(k_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle;
        if (USE_MROPE) {
            uint axis = lane % 3u;
            angle = float(pos3d[t * 3u + axis]) * freq;
        } else {
            angle = float(int(p.pos + t) + p.rope_offset) * freq;
        }
        float c = cos(angle);
        float s = sin(angle);
        float x0 = kv[0];
        float x1 = kv[1];
        kv[0] = fma(x0, c, -x1 * s);
        kv[1] = fma(x1, c,  x0 * s);
    }

    constexpr float QMAX = (KV_BITS == 8) ? 255.0f : 15.0f;
    #pragma unroll
    for (uint g = 0; g < 4u; g++) {
        // Quantize K group g (64 elements: kv[2*g] & kv[2*g+1] across 32 lanes) using exact FP32 scale+min!
        float ka = kv[2u * g];
        float kb = kv[2u * g + 1u];
        float kmn = simd_min(min(ka, kb));
        float kmx = simd_max(max(ka, kb));
        float krng = kmx - kmn;
        float fs_k = (krng > 1e-12f) ? (krng / QMAX) : 0.0f;
        float inv_k = (krng > 1e-12f) ? (QMAX / krng) : 0.0f;
        if (lane == 0u) ksbd[g] = float2(fs_k, kmn);
        uint qk0 = uint(clamp(rint((ka - kmn) * inv_k), 0.0f, QMAX));
        uint qk1 = uint(clamp(rint((kb - kmn) * inv_k), 0.0f, QMAX));
        if (KV_BITS == 8) {
            kdst[(2u * g) * 32u + lane]      = uchar(qk0);
            kdst[(2u * g + 1u) * 32u + lane] = uchar(qk1);
        } else {
            kdst[g * 32u + lane] = uchar(qk0 | (qk1 << 4u));
        }

        // Quantize V group g
        float va = vv[2u * g];
        float vb = vv[2u * g + 1u];
        float vmn = simd_min(min(va, vb));
        float vmx = simd_max(max(va, vb));
        float vrng = vmx - vmn;
        float fs_v = (vrng > 1e-12f) ? (vrng / QMAX) : 0.0f;
        float inv_v = (vrng > 1e-12f) ? (QMAX / vrng) : 0.0f;
        if (lane == 0u) vsbd[g] = float2(fs_v, vmn);
        uint qv0 = uint(clamp(rint((va - vmn) * inv_v), 0.0f, QMAX));
        uint qv1 = uint(clamp(rint((vb - vmn) * inv_v), 0.0f, QMAX));
        if (KV_BITS == 8) {
            vdst[(2u * g) * 32u + lane]      = uchar(qv0);
            vdst[(2u * g + 1u) * 32u + lane] = uchar(qv1);
        } else {
            vdst[g * 32u + lane] = uchar(qv0 | (qv1 << 4u));
        }
    }
}

kernel void qwen_full_attn_kv_prep_q8(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_impl<8, false>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, nullptr, tgid, lane);
}

kernel void qwen_full_attn_kv_prep_q8_mrope(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    const device int* pos3d [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_impl<8, true>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, pos3d, tgid, lane);
}

kernel void qwen_full_attn_kv_prep_q4(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_impl<4, false>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, nullptr, tgid, lane);
}

kernel void qwen_full_attn_kv_prep_q4_mrope(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    const device int* pos3d [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_impl<4, true>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, pos3d, tgid, lane);
}

// Helper: load & dequantize 32 x 128 strip of Q8 or Q4 KV cache into threadgroup sKV[32][132]
template<int KV_BITS>
inline void load_kv_tile_quant(
    const device uchar* kv_cache,
    const device float2* kv_sb,
    threadgroup half (&sKV)[32][132],
    uint kb, uint tile_k, uint d_half, uint hkv, uint Hkv, uint tid
) {
    uint kr = tid >> 2;
    if (kr < tile_k) {
        ulong head_idx = (ulong)(kb + kr) * Hkv + hkv;
        if (KV_BITS == 8) {
            uint kc  = (tid & 3u) * 32u;
            uint grp = d_half * 2u + ((tid & 3u) >> 1);
            float2 sb = kv_sb[head_idx * 4ul + grp];
            float4 fs = float4(sb.x);
            float4 fb = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 256ul + d_half * 128u + kc);
            threadgroup half4* dst  = (threadgroup half4*)(&sKV[kr][kc]);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) {
                dst[i] = half4(fma(float4(qp[i]), fs, fb));
            }
        } else {
            uint g_loc = (tid & 3u) >> 1;
            uint sub16 = (tid & 1u) * 16u;
            uint grp   = d_half * 2u + g_loc;
            float2 sb  = kv_sb[head_idx * 4ul + grp];
            float4 fs  = float4(sb.x);
            float4 fb  = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 128ul + grp * 32u + sub16);
            threadgroup half4* dst_lo = (threadgroup half4*)(&sKV[kr][g_loc * 64u + sub16]);
            threadgroup half4* dst_hi = (threadgroup half4*)(&sKV[kr][g_loc * 64u + 32u + sub16]);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; i++) {
                uchar4 u = qp[i];
                dst_lo[i] = half4(fma(float4(u & uchar4(0x0Fu)), fs, fb));
                dst_hi[i] = half4(fma(float4(u >> uchar4(4u)),   fs, fb));
            }
        }
    } else {
        uint kc = (tid & 3u) * 32u;
        threadgroup half4* dst = (threadgroup half4*)(&sKV[kr][kc]);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i] = half4(0.0h);
    }
}

template<int KV_BITS, typename OutT>
inline void qwen_gqa_attn_gated_quant_impl(
    const device float* q_in,
    const device float* gate_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device OutT*        out,
    constant FullAttnParams& p,
    uint2 tgid, uint lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device float* gp = gate_in + q_off;
    device OutT* op = out + q_off;

    float qv[8], acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) { qv[i] = qp[i * 32u + lane]; acc[i] = 0.0f; }

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    uint limit = p.pos + t + 1u;
    constexpr uint BYTES_PER_HEAD = (KV_BITS == 8) ? 256u : 128u;

    for (uint kpos = 0; kpos < limit; kpos++) {
        ulong head_idx = (ulong)kpos * p.Hkv + hkv;
        const device uchar* kp = k_cache + head_idx * BYTES_PER_HEAD;
        const device float2* ksb = k_sb + head_idx * 4ul;
        float dot = 0.0f;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(ksb[g].x);
            float mn = float(ksb[g].y);
            float k0, k1;
            if (KV_BITS == 8) {
                k0 = fma(float(kp[(2u * g) * 32u + lane]), sc, mn);
                k1 = fma(float(kp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pk = kp[g * 32u + lane];
                k0 = fma(float(pk & 0x0Fu), sc, mn);
                k1 = fma(float(pk >> 4u), sc, mn);
            }
            dot = fma(qv[2u * g], k0, fma(qv[2u * g + 1u], k1, dot));
        }
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device uchar* vp = v_cache + head_idx * BYTES_PER_HEAD;
        const device float2* vsb = v_sb + head_idx * 4ul;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(vsb[g].x);
            float mn = float(vsb[g].y);
            float v0, v1;
            if (KV_BITS == 8) {
                v0 = fma(float(vp[(2u * g) * 32u + lane]), sc, mn);
                v1 = fma(float(vp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pv = vp[g * 32u + lane];
                v0 = fma(float(pv & 0x0Fu), sc, mn);
                v1 = fma(float(pv >> 4u), sc, mn);
            }
            acc[2u * g]      = fma(acc[2u * g],      alpha, beta * v0);
            acc[2u * g + 1u] = fma(acc[2u * g + 1u], alpha, beta * v1);
        }
    }

    float inv_l = 1.0f / l_sum;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = OutT((acc[i] * inv_l) * sigmoid_f32(gp[idx]));
    }
}

kernel void qwen_gqa_attn_gated_q8(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device float* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_impl<8, float>(q, g, kc, vc, ksb, vsb, out, p, tgid, lane); }

kernel void qwen_gqa_attn_gated_q8_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_impl<8, half>(q, g, kc, vc, ksb, vsb, out, p, tgid, lane); }

kernel void qwen_gqa_attn_gated_q4(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device float* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_impl<4, float>(q, g, kc, vc, ksb, vsb, out, p, tgid, lane); }

kernel void qwen_gqa_attn_gated_q4_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_impl<4, half>(q, g, kc, vc, ksb, vsb, out, p, tgid, lane); }

template<int KV_BITS, typename OutT>
inline void qwen_gqa_prefill_flash_quant_impl(
    const device float* q_in,
    const device float* gate_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device OutT*        out,
    constant FullAttnParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sQ)[16][260],
    threadgroup half  (&sKV)[32][132],
    threadgroup float (&sS)[16][32],
    threadgroup half  (&sP)[16][36],
    threadgroup float (&sAlpha)[16],
    threadgroup float (&sInvL)[16]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint t0 = tgid.x * BQ;
    uint h  = tgid.y;
    if (h >= p.Hq || t0 >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint gt = t0 + qr;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (gt < p.M) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)gt * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;
    const bool full_q_tile = (t0 + BQ <= p.M);
    uint max_kv = p.pos + min(t0 + BQ, p.M);

    for (uint kb = 0; kb < max_kv; kb += BK) {
        uint tile_k = min(BK, max_kv - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = full_q_tile && (kb + BK <= p.pos + t0);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == 0u) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint q_pos = p.pos + t0 + r;
                uint k_pos = kb + lane;
                bool valid = (t0 + r < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) sInvL[sg * 4u + rr] = (l_row[rr] > 0.0f) ? (1.0f / l_row[rr]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gt = t0 + sg_m + fm;
    if (gt < p.M) {
        float inv_l = sInvL[sg_m + fm];
        ulong q_off = ((ulong)gt * p.Hq + h) * p.D;
        const device float* gp = gate_in + q_off;
        device OutT* op = out + q_off;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            op[c0]      = OutT((O0[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c0]));
            op[c0 + 1u] = OutT((O0[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c0 + 1u]));
            uint c1 = 128u + c0;
            op[c1]      = OutT((O1[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c1]));
            op[c1 + 1u] = OutT((O1[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c1 + 1u]));
        }
    }
}

kernel void qwen_gqa_prefill_flash_q8(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device float* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_quant_impl<8, float>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}

kernel void qwen_gqa_prefill_flash_q8_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_quant_impl<8, half>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}

kernel void qwen_gqa_prefill_flash_q4(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device float* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_quant_impl<4, float>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}

kernel void qwen_gqa_prefill_flash_q4_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_quant_impl<4, half>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}

template<int KV_BITS>
inline void qwen_gqa_decode_splitk_p1_quant_impl(
    const device float* q_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*       part_o,
    device float*       part_m,
    device float*       part_l,
    constant FullAttnParams& p,
    uint2 tgid, uint lane
) {
    uint h = tgid.x;
    uint s = tgid.y;
    if (h >= p.Hq || s >= p.num_splits) return;

    uint hkv = h / (p.Hq / p.Hkv);
    const device float* qp = q_in + (ulong)h * p.D;
    float qv[8], acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) { qv[i] = qp[i * 32u + lane]; acc[i] = 0.0f; }

    uint total = p.kv_len;
    uint chunk = (total + p.num_splits - 1u) / p.num_splits;
    uint start = s * chunk;
    uint end   = min(total, start + chunk);

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    constexpr uint BYTES_PER_HEAD = (KV_BITS == 8) ? 256u : 128u;

    for (uint kpos = start; kpos < end; kpos++) {
        ulong head_idx = (ulong)kpos * p.Hkv + hkv;
        const device uchar* kp = k_cache + head_idx * BYTES_PER_HEAD;
        const device float2* ksb = k_sb + head_idx * 4ul;
        float dot = 0.0f;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(ksb[g].x), mn = float(ksb[g].y);
            float k0, k1;
            if (KV_BITS == 8) {
                k0 = fma(float(kp[(2u * g) * 32u + lane]), sc, mn);
                k1 = fma(float(kp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pk = kp[g * 32u + lane];
                k0 = fma(float(pk & 0x0Fu), sc, mn);
                k1 = fma(float(pk >> 4u), sc, mn);
            }
            dot = fma(qv[2u * g], k0, fma(qv[2u * g + 1u], k1, dot));
        }
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device uchar* vp = v_cache + head_idx * BYTES_PER_HEAD;
        const device float2* vsb = v_sb + head_idx * 4ul;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(vsb[g].x), mn = float(vsb[g].y);
            float v0, v1;
            if (KV_BITS == 8) {
                v0 = fma(float(vp[(2u * g) * 32u + lane]), sc, mn);
                v1 = fma(float(vp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pv = vp[g * 32u + lane];
                v0 = fma(float(pv & 0x0Fu), sc, mn);
                v1 = fma(float(pv >> 4u), sc, mn);
            }
            acc[2u * g]      = fma(acc[2u * g],      alpha, beta * v0);
            acc[2u * g + 1u] = fma(acc[2u * g + 1u], alpha, beta * v1);
        }
    }

    ulong slot = (ulong)h * p.num_splits + s;
    device float* po = part_o + slot * p.D;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) po[i * 32u + lane] = acc[i];
    if (lane == 0u) { part_m[slot] = m_max; part_l[slot] = l_sum; }
}

kernel void qwen_gqa_decode_splitk_p1_q8(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_decode_splitk_p1_quant_impl<8>(q, kc, vc, ksb, vsb, po, pm, pl, p, tgid, lane); }

kernel void qwen_gqa_decode_splitk_p1_q4(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_decode_splitk_p1_quant_impl<4>(q, kc, vc, ksb, vsb, po, pm, pl, p, tgid, lane); }

template<int KV_BITS>
inline void qwen_gqa_verify_splitk_p1_quant_impl(
    const device float* q_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*       part_o,
    device float*       part_m,
    device float*       part_l,
    constant FullAttnParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sQ)[16][260],
    threadgroup half  (&sKV)[32][132],
    threadgroup float (&sS)[16][32],
    threadgroup half  (&sP)[16][36],
    threadgroup float (&sAlpha)[16]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint pair_idx = tgid.x;
    uint s        = tgid.y;
    if (pair_idx >= p.Hkv * pairs_per_kv || s >= p.num_splits) return;

    uint hkv   = pair_idx / pairs_per_kv;
    uint gp    = pair_idx % pairs_per_kv;
    uint h0    = hkv * gqa + gp * 2u;
    uint h1    = h0 + 1u;
    bool h1_ok = (gp * 2u + 1u < gqa);

    uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
    uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
    uint kb_start        = s * tiles_per_split * BK;
    uint kb_end          = min(p.kv_len, (s + 1u) * tiles_per_split * BK);

    if (kb_start >= kb_end) {
        if (tid < BQ) {
            uint lh = tid >> 3;
            uint t  = tid & 7u;
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = -INFINITY;
                part_l[slot] = 0.0f;
            }
        }
        return;
    }

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint lh = qr >> 3;
        uint t  = qr & 7u;
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (t < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)t * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint kb = kb_start; kb < kb_end; kb += BK) {
        uint tile_k = min(BK, kb_end - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = (kb + BK <= p.pos) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == kb_start) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint t = r & 7u;
                uint q_pos = p.pos + t;
                uint k_pos = kb + lane;
                bool valid = (t < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            uint r  = sg * 4u + rr;
            uint lh = r >> 3;
            uint t  = r & 7u;
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = m_row[rr];
                part_l[slot] = l_row[rr];
            }
        }
    }

    uint row = sg_m + fm;
    uint lh = row >> 3;
    uint t  = row & 7u;
    uint h  = (lh == 0u) ? h0 : h1;
    if (t < p.M && (lh == 0u || h1_ok)) {
        ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
        device float* po = part_o + slot * p.D;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            po[c0]      = O0[j].thread_elements()[0];
            po[c0 + 1u] = O0[j].thread_elements()[1];
            uint c1 = 128u + c0;
            po[c1]      = O1[j].thread_elements()[0];
            po[c1 + 1u] = O1[j].thread_elements()[1];
        }
    }
}

kernel void qwen_gqa_verify_splitk_p1_q8(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16];
    qwen_gqa_verify_splitk_p1_quant_impl<8>(q, kc, vc, ksb, vsb, po, pm, pl, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha);
}

kernel void qwen_gqa_verify_splitk_p1_q4(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16];
    qwen_gqa_verify_splitk_p1_quant_impl<4>(q, kc, vc, ksb, vsb, po, pm, pl, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha);
}

// Round 23: Quantized KV8/KV4 GQA-Pair (2 GQA heads x 8 Q-tokens = 16 rows) AMX FlashAttention-2 Prefill!
template<int KV_BITS>
inline void qwen_gqa_prefill_flash_pair_quant_impl(
    const device float* q_in,
    const device float* gate_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device half*        out,
    constant FullAttnParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sQ)[16][260],
    threadgroup half  (&sKV)[32][132],
    threadgroup float (&sS)[16][32],
    threadgroup half  (&sP)[16][36],
    threadgroup float (&sAlpha)[16],
    threadgroup float (&sInvL)[16]
) {
    uint t0 = tgid.x * 8u;
    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint hkv = tgid.y / pairs_per_kv;
    uint gp_idx = tgid.y % pairs_per_kv;
    uint h0  = hkv * gqa + gp_idx * 2u;
    uint h1  = h0 + 1u;
    bool h1_ok = (gp_idx * 2u + 1u < gqa);
    if (hkv >= p.Hkv || t0 >= p.M) return;

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint lh = qr >> 3;
        uint lt = qr & 7u;
        uint gt = t0 + lt;
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (gt < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)gt * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;
    const bool full_q_tile = (t0 + 8u <= p.M);
    uint max_kv = p.pos + min(t0 + 8u, p.M);

    for (uint kb = 0; kb < max_kv; kb += 32u) {
        uint tile_k = min(32u, max_kv - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = full_q_tile && (kb + 32u <= p.pos + t0);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == 0u) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint lt = r & 7u;
                uint q_pos = p.pos + t0 + lt;
                uint k_pos = kb + lane;
                bool valid = (t0 + lt < p.M) && (lane < tile_k) && (k_pos <= q_pos);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) sInvL[sg * 4u + rr] = (l_row[rr] > 0.0f) ? (1.0f / l_row[rr]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = sg_m + fm;
    uint lh = row >> 3;
    uint lt = row & 7u;
    uint gt = t0 + lt;
    uint h  = (lh == 0u) ? h0 : h1;
    if (gt < p.M && (lh == 0u || h1_ok)) {
        float inv_l = sInvL[row];
        ulong q_off = ((ulong)gt * p.Hq + h) * p.D;
        const device float* gp = gate_in + q_off;
        device half* op = out + q_off;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            op[c0]      = half((O0[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c0]));
            op[c0 + 1u] = half((O0[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c0 + 1u]));
            uint c1 = 128u + c0;
            op[c1]      = half((O1[j].thread_elements()[0] * inv_l) * sigmoid_f32(gp[c1]));
            op[c1 + 1u] = half((O1[j].thread_elements()[1] * inv_l) * sigmoid_f32(gp[c1 + 1u]));
        }
    }
}

kernel void qwen_gqa_prefill_flash_pair_q8_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_pair_quant_impl<8>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}

kernel void qwen_gqa_prefill_flash_pair_q4_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16], sInvL[16];
    qwen_gqa_prefill_flash_pair_quant_impl<4>(q, g, kc, vc, ksb, vsb, out, p, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha, sInvL);
}


// ============================================================================
// 11. Stage 2: Arbitrary Speculative Tree Verification Kernels (M <= 16)
//     Supports ANY multi-fork tree across all 48 GatedDeltaNet layers (4-slot
//     on-chip BF16 checkpoint stack in threadgroup SRAM, zero DRAM traffic!)
//     and all 16 Full-Attention layers (bitmask causal mask + depth-indexed RoPE
//     + winning-path KV cache compaction & exact GDN path replay).
// ============================================================================

struct TreeParams {
    uint is_tree;
    uint n_nodes;
    uint mask_base_pos;
    uint pad0;
    uint mask[16];
    uint depth[16];
    int  parent[16];
    int  gdn_load[16];
    int  gdn_save[16];
};

struct PathReplayParams {
    uint len;
    uint pre_pos;
    uint Hkv;
    uint D;
    uint kv_bits;
    uint pad0;
    uint pad1;
    uint pad2;
    uint path[16];
};

kernel void qwen_gdn_prework_tree(
    const device float*  qkv_in         [[buffer(0)]],
    const device ushort* conv_w         [[buffer(1)]],
    const device float*  conv_state_in  [[buffer(2)]],
    device float*        conv_state_out [[buffer(3)]],
    device float*        qkv_conv_out   [[buffer(4)]],
    device float*        q_out          [[buffer(5)]],
    device float*        k_out          [[buffer(6)]],
    const device float*  a_in           [[buffer(7)]],
    const device float*  b_in           [[buffer(8)]],
    const device ushort* A_log          [[buffer(9)]],
    const device ushort* dt_bias        [[buffer(10)]],
    device float*        g_out          [[buffer(11)]],
    device float*        beta_out       [[buffer(12)]],
    constant GdnPrepParams& p           [[buffer(13)]],
    constant TreeParams& tp             [[buffer(14)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint sg                             [[simdgroup_index_in_threadgroup]]
) {
    (void)conv_state_out;
    const uint key_dim = p.Hk * p.Dk;
    if (tgid < 2u * p.Hk) {
        bool is_k = (tgid >= p.Hk);
        uint hk = is_k ? (tgid - p.Hk) : tgid;
        uint c  = (is_k ? key_dim : 0u) + hk * p.Dk + tid;

        float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
        float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
        float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
        float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
        float cs0 = conv_state_in[0u * p.C + c];
        float cs1 = conv_state_in[1u * p.C + c];
        float cs2 = conv_state_in[2u * p.C + c];

        const float inv_scale = rsqrt(float(p.Dk));
        const float qk_scale  = is_k ? inv_scale : (inv_scale * inv_scale);
        device float* out_base = is_k ? k_out : q_out;
        threadgroup float smem[4];

        for (uint t = 0; t < p.M; t++) {
            int p1 = tp.parent[t];
            int p2 = (p1 >= 0) ? tp.parent[p1] : (p1 - 1);
            int p3 = (p2 >= 0) ? tp.parent[p2] : (p2 - 1);
            float x3 = (p3 >= 0) ? qkv_in[(ulong)p3 * p.C + c] : ((p3 == -3) ? cs0 : ((p3 == -2) ? cs1 : cs2));
            float x2 = (p2 >= 0) ? qkv_in[(ulong)p2 * p.C + c] : ((p2 == -2) ? cs1 : cs2);
            float x1 = (p1 >= 0) ? qkv_in[(ulong)p1 * p.C + c] : cs2;
            float x0 = qkv_in[(ulong)t * p.C + c];
            float sv = silu_f32(fma(x3, w0, fma(x2, w1, fma(x1, w2, x0 * w3))));

            float sq = simd_sum(sv * sv);
            if (lane == 0u) smem[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = smem[0] + smem[1] + smem[2] + smem[3];
            float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
            out_base[((ulong)t * p.Hk + hk) * p.Dk + tid] = sv * mul;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    uint hv = tgid - 2u * p.Hk;
    if (hv >= p.Hv) return;

    uint c = 2u * key_dim + hv * p.Dk + tid;
    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
    float cs0 = conv_state_in[0u * p.C + c];
    float cs1 = conv_state_in[1u * p.C + c];
    float cs2 = conv_state_in[2u * p.C + c];

    float neg_ea = 0.0f, dtb = 0.0f;
    if (tid == 0u) {
        neg_ea = -exp(bf16_to_f32(A_log[hv]));
        dtb    = bf16_to_f32(dt_bias[hv]);
    }

    for (uint t = 0; t < p.M; t++) {
        int p1 = tp.parent[t];
        int p2 = (p1 >= 0) ? tp.parent[p1] : (p1 - 1);
        int p3 = (p2 >= 0) ? tp.parent[p2] : (p2 - 1);
        float x3 = (p3 >= 0) ? qkv_in[(ulong)p3 * p.C + c] : ((p3 == -3) ? cs0 : ((p3 == -2) ? cs1 : cs2));
        float x2 = (p2 >= 0) ? qkv_in[(ulong)p2 * p.C + c] : ((p2 == -2) ? cs1 : cs2);
        float x1 = (p1 >= 0) ? qkv_in[(ulong)p1 * p.C + c] : cs2;
        float x0 = qkv_in[(ulong)t * p.C + c];
        qkv_conv_out[(ulong)t * p.C + c] = silu_f32(fma(x3, w0, fma(x2, w1, fma(x1, w2, x0 * w3))));
        if (tid == 0u) {
            ulong gi = (ulong)t * p.Hv + hv;
            g_out[gi]    = exp(neg_ea * softplus_f32(a_in[gi] + dtb));
            beta_out[gi] = sigmoid_f32(b_in[gi]);
        }
    }
}

kernel void qwen_gdn_prework_replay_path(
    const device float*  qkv_in         [[buffer(0)]],
    const device ushort* conv_w         [[buffer(1)]],
    const device float*  conv_state_in  [[buffer(2)]],
    device float*        conv_state_out [[buffer(3)]],
    device float*        qkv_conv_out   [[buffer(4)]],
    device float*        q_out          [[buffer(5)]],
    device float*        k_out          [[buffer(6)]],
    const device float*  a_in           [[buffer(7)]],
    const device float*  b_in           [[buffer(8)]],
    const device ushort* A_log          [[buffer(9)]],
    const device ushort* dt_bias        [[buffer(10)]],
    device float*        g_out          [[buffer(11)]],
    device float*        beta_out       [[buffer(12)]],
    constant GdnPrepParams& p           [[buffer(13)]],
    constant PathReplayParams& pr       [[buffer(14)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tid                            [[thread_position_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint sg                             [[simdgroup_index_in_threadgroup]]
) {
    const uint key_dim = p.Hk * p.Dk;
    if (tgid < 2u * p.Hk) {
        bool is_k = (tgid >= p.Hk);
        uint hk = is_k ? (tgid - p.Hk) : tgid;
        uint c  = (is_k ? key_dim : 0u) + hk * p.Dk + tid;

        float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
        float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
        float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
        float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
        float s0 = conv_state_in[0u * p.C + c];
        float s1 = conv_state_in[1u * p.C + c];
        float s2 = conv_state_in[2u * p.C + c];

        const float inv_scale = rsqrt(float(p.Dk));
        const float qk_scale  = is_k ? inv_scale : (inv_scale * inv_scale);
        device float* out_base = is_k ? k_out : q_out;
        threadgroup float smem[4];

        for (uint t = 0; t < p.M; t++) {
            uint src_t = pr.path[t];
            float x = qkv_in[(ulong)src_t * p.C + c];
            float sv = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
            s0 = s1; s1 = s2; s2 = x;

            float sq = simd_sum(sv * sv);
            if (lane == 0u) smem[sg] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = smem[0] + smem[1] + smem[2] + smem[3];
            float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
            out_base[((ulong)t * p.Hk + hk) * p.Dk + tid] = sv * mul;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        conv_state_out[0u * p.C + c] = s0;
        conv_state_out[1u * p.C + c] = s1;
        conv_state_out[2u * p.C + c] = s2;
        return;
    }

    uint hv = tgid - 2u * p.Hk;
    if (hv >= p.Hv) return;

    uint c = 2u * key_dim + hv * p.Dk + tid;
    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
    float s0 = conv_state_in[0u * p.C + c];
    float s1 = conv_state_in[1u * p.C + c];
    float s2 = conv_state_in[2u * p.C + c];

    float neg_ea = 0.0f, dtb = 0.0f;
    if (tid == 0u) {
        neg_ea = -exp(bf16_to_f32(A_log[hv]));
        dtb    = bf16_to_f32(dt_bias[hv]);
    }

    for (uint t = 0; t < p.M; t++) {
        uint src_t = pr.path[t];
        float x = qkv_in[(ulong)src_t * p.C + c];
        qkv_conv_out[(ulong)t * p.C + c] = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
        s0 = s1; s1 = s2; s2 = x;
        if (tid == 0u) {
            ulong sgi = (ulong)src_t * p.Hv + hv;
            ulong dgi = (ulong)t * p.Hv + hv;
            g_out[dgi]    = exp(neg_ea * softplus_f32(a_in[sgi] + dtb));
            beta_out[dgi] = sigmoid_f32(b_in[sgi]);
        }
    }

    conv_state_out[0u * p.C + c] = s0;
    conv_state_out[1u * p.C + c] = s1;
    conv_state_out[2u * p.C + c] = s2;
}

// 4-slot on-chip BF16 checkpoint stack in threadgroup SRAM (16 KB -> 2 TGs/core, ZERO DRAM cost, ZERO barriers!)
kernel void qwen_gdn_recurrence_tree(
    const device float*  q_in      [[buffer(0)]],
    const device float*  k_in      [[buffer(1)]],
    const device float*  qkv_conv  [[buffer(2)]],
    const device float*  g_in      [[buffer(3)]],
    const device float*  beta_in   [[buffer(4)]],
    const device ushort* state_in  [[buffer(5)]],
    device float*        y_out     [[buffer(6)]],
    constant GdnRecParams& p       [[buffer(7)]],
    device ushort*       state_out [[buffer(8)]],
    constant TreeParams& tp        [[buffer(9)]],
    uint3 gid                      [[thread_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]]
) {
    (void)state_out;
    uint dv0 = gid.y * 4u;
    uint hv  = gid.z;
    if (dv0 >= p.Dv || hv >= p.Hv) return;

    uint hk = hv / (p.Hv / p.Hk);
    uint v_offset = 2u * p.Hk * p.Dk + hv * p.Dv + dv0;
    uint base_k = lane * 4u;

    threadgroup ushort4 s_stack[4][4][128];

    ulong st_base = ((ulong)hv * p.Dv + dv0) * p.Dk + base_k;
    float4 s0 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 0u * p.Dk)));
    float4 s1 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 1u * p.Dk)));
    float4 s2 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 2u * p.Dk)));
    float4 s3 = bf16x4_to_f32x4(*((const device ushort4*)(state_in + st_base + 3u * p.Dk)));

    for (uint t = 0; t < p.M; t++) {
        int ld = tp.gdn_load[t];
        if (ld >= 0) {
            s0 = bf16x4_to_f32x4(s_stack[ld][0][tid]);
            s1 = bf16x4_to_f32x4(s_stack[ld][1][tid]);
            s2 = bf16x4_to_f32x4(s_stack[ld][2][tid]);
            s3 = bf16x4_to_f32x4(s_stack[ld][3][tid]);
        }

        float g    = g_in[(ulong)t * p.Hv + hv];
        float beta = beta_in[(ulong)t * p.Hv + hv];

        float4 kv = *((const device float4*)(k_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
        float4 qv = *((const device float4*)(q_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
        float4 vv = *((const device float4*)(qkv_conv + (ulong)t * p.C + v_offset));

        s0 *= g; s1 *= g; s2 *= g; s3 *= g;

        float d0 = simd_sum(dot(s0, kv));
        float d1 = simd_sum(dot(s1, kv));
        float d2 = simd_sum(dot(s2, kv));
        float d3 = simd_sum(dot(s3, kv));

        float del0 = (vv.x - d0) * beta;
        float del1 = (vv.y - d1) * beta;
        float del2 = (vv.z - d2) * beta;
        float del3 = (vv.w - d3) * beta;

        s0 = fma(kv, float4(del0), s0);
        s1 = fma(kv, float4(del1), s1);
        s2 = fma(kv, float4(del2), s2);
        s3 = fma(kv, float4(del3), s3);

        float o0 = simd_sum(dot(s0, qv));
        float o1 = simd_sum(dot(s1, qv));
        float o2 = simd_sum(dot(s2, qv));
        float o3 = simd_sum(dot(s3, qv));

        if (lane == 0u) {
            *((device float4*)(y_out + ((ulong)t * p.Hv + hv) * p.Dv + dv0)) = float4(o0, o1, o2, o3);
        }

        int sv = tp.gdn_save[t];
        if (sv >= 0) {
            s_stack[sv][0][tid] = f32x4_to_bf16x4(s0);
            s_stack[sv][1][tid] = f32x4_to_bf16x4(s1);
            s_stack[sv][2][tid] = f32x4_to_bf16x4(s2);
            s_stack[sv][3][tid] = f32x4_to_bf16x4(s3);
        }
    }
}

// ============================================================================
// Lazy Fused GDN Replay + Speculative Verify (Zero CB #1 replay pass, Zero bak_ssm_state_!)
//   Phase A: replay pr.len accepted tokens from prev_qkv/prev_a/prev_b (writing slots 0..pr.len-1)
//            and commit exact conv_state_io / state_io ONCE in-place!
//   Phase B: immediately verify p.M tokens from cur_qkv/cur_a/cur_b (linear OR tree, slots 8..8+M-1)
//            keeping conv/ssm state hot in registers with ZERO speculative state writes!
// ============================================================================
kernel void qwen_gdn_prework_lazy(
    const device float*        prev_qkv      [[buffer(0)]],
    const device float*        cur_qkv       [[buffer(1)]],
    const device ushort*       conv_w        [[buffer(2)]],
    device float*              conv_state_io [[buffer(3)]],
    device float*              qkv_conv_out  [[buffer(4)]],
    device float*              q_out         [[buffer(5)]],
    device float*              k_out         [[buffer(6)]],
    const device float*        prev_a        [[buffer(7)]],
    const device float*        prev_b        [[buffer(8)]],
    const device float*        cur_a         [[buffer(9)]],
    const device float*        cur_b         [[buffer(10)]],
    const device ushort*       A_log         [[buffer(11)]],
    const device ushort*       dt_bias       [[buffer(12)]],
    device float*              g_out         [[buffer(13)]],
    device float*              beta_out      [[buffer(14)]],
    constant GdnPrepParams&    p             [[buffer(15)]],
    constant PathReplayParams& pr            [[buffer(16)]],
    constant TreeParams&       tp            [[buffer(17)]],
    uint tgid                                [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]],
    uint lane                                [[thread_index_in_simdgroup]],
    uint sg                                  [[simdgroup_index_in_threadgroup]]
) {
    const uint key_dim = p.Hk * p.Dk;
    if (tgid < 2u * p.Hk) {
        bool is_k = (tgid >= p.Hk);
        uint hk = is_k ? (tgid - p.Hk) : tgid;
        uint c  = (is_k ? key_dim : 0u) + hk * p.Dk + tid;

        float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
        float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
        float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
        float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
        float s0 = conv_state_io[0u * p.C + c];
        float s1 = conv_state_io[1u * p.C + c];
        float s2 = conv_state_io[2u * p.C + c];

        const float inv_scale = rsqrt(float(p.Dk));
        const float qk_scale  = is_k ? inv_scale : (inv_scale * inv_scale);
        device float* out_base = is_k ? k_out : q_out;
        threadgroup float smem[4];

        // Phase A: Lazy replay of previous round's accepted tokens + commit conv_state_io
        if (pr.len > 0u) {
            if (is_k) {
                for (uint t = 0u; t < pr.len; t++) {
                    uint src_t = pr.path[t];
                    float x = prev_qkv[(ulong)src_t * p.C + c];
                    float sv = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
                    s0 = s1; s1 = s2; s2 = x;

                    float sq = simd_sum(sv * sv);
                    if (lane == 0u) smem[sg] = sq;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float tot = smem[0] + smem[1] + smem[2] + smem[3];
                    float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
                    out_base[((ulong)t * p.Hk + hk) * p.Dk + tid] = sv * mul;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                // Q heads don't need replay q_out (recurrence replay skips y_out), only exact conv_state shift!
                for (uint t = 0u; t < pr.len; t++) {
                    uint src_t = pr.path[t];
                    float x = prev_qkv[(ulong)src_t * p.C + c];
                    s0 = s1; s1 = s2; s2 = x;
                }
            }
            conv_state_io[0u * p.C + c] = s0;
            conv_state_io[1u * p.C + c] = s1;
            conv_state_io[2u * p.C + c] = s2;
        }

        // Phase B: Current round verify tokens -> slots [8 .. 8 + M - 1]
        if (tp.is_tree == 0u) {
            for (uint t = 0u; t < p.M; t++) {
                float x = cur_qkv[(ulong)t * p.C + c];
                float sv = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
                s0 = s1; s1 = s2; s2 = x;

                float sq = simd_sum(sv * sv);
                if (lane == 0u) smem[sg] = sq;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = smem[0] + smem[1] + smem[2] + smem[3];
                float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
                out_base[((ulong)(8u + t) * p.Hk + hk) * p.Dk + tid] = sv * mul;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            float cs0 = s0, cs1 = s1, cs2 = s2;
            for (uint t = 0u; t < p.M; t++) {
                int p1 = tp.parent[t];
                int p2 = (p1 >= 0) ? tp.parent[p1] : (p1 - 1);
                int p3 = (p2 >= 0) ? tp.parent[p2] : (p2 - 1);
                float x3 = (p3 >= 0) ? cur_qkv[(ulong)p3 * p.C + c] : ((p3 == -3) ? cs0 : ((p3 == -2) ? cs1 : cs2));
                float x2 = (p2 >= 0) ? cur_qkv[(ulong)p2 * p.C + c] : ((p2 == -2) ? cs1 : cs2);
                float x1 = (p1 >= 0) ? cur_qkv[(ulong)p1 * p.C + c] : cs2;
                float x0 = cur_qkv[(ulong)t * p.C + c];
                float sv = silu_f32(fma(x3, w0, fma(x2, w1, fma(x1, w2, x0 * w3))));

                float sq = simd_sum(sv * sv);
                if (lane == 0u) smem[sg] = sq;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = smem[0] + smem[1] + smem[2] + smem[3];
                float mul = rsqrt(tot / float(p.Dk) + p.eps) * qk_scale;
                out_base[((ulong)(8u + t) * p.Hk + hk) * p.Dk + tid] = sv * mul;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        return;
    }

    uint hv = tgid - 2u * p.Hk;
    if (hv >= p.Hv) return;

    uint c = 2u * key_dim + hv * p.Dk + tid;
    float w0 = bf16_to_f32(conv_w[c * 4u + 0u]);
    float w1 = bf16_to_f32(conv_w[c * 4u + 1u]);
    float w2 = bf16_to_f32(conv_w[c * 4u + 2u]);
    float w3 = bf16_to_f32(conv_w[c * 4u + 3u]);
    float s0 = conv_state_io[0u * p.C + c];
    float s1 = conv_state_io[1u * p.C + c];
    float s2 = conv_state_io[2u * p.C + c];

    float neg_ea = 0.0f, dtb = 0.0f;
    if (tid == 0u) {
        neg_ea = -exp(bf16_to_f32(A_log[hv]));
        dtb    = bf16_to_f32(dt_bias[hv]);
    }

    if (pr.len > 0u) {
        for (uint t = 0u; t < pr.len; t++) {
            uint src_t = pr.path[t];
            float x = prev_qkv[(ulong)src_t * p.C + c];
            qkv_conv_out[(ulong)t * p.C + c] = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
            s0 = s1; s1 = s2; s2 = x;
            if (tid == 0u) {
                ulong sgi = (ulong)src_t * p.Hv + hv;
                ulong dgi = (ulong)t * p.Hv + hv;
                g_out[dgi]    = exp(neg_ea * softplus_f32(prev_a[sgi] + dtb));
                beta_out[dgi] = sigmoid_f32(prev_b[sgi]);
            }
        }
        conv_state_io[0u * p.C + c] = s0;
        conv_state_io[1u * p.C + c] = s1;
        conv_state_io[2u * p.C + c] = s2;
    }

    if (tp.is_tree == 0u) {
        for (uint t = 0u; t < p.M; t++) {
            float x = cur_qkv[(ulong)t * p.C + c];
            uint dst_t = 8u + t;
            qkv_conv_out[(ulong)dst_t * p.C + c] = silu_f32(fma(s0, w0, fma(s1, w1, fma(s2, w2, x * w3))));
            s0 = s1; s1 = s2; s2 = x;
            if (tid == 0u) {
                ulong sgi = (ulong)t * p.Hv + hv;
                ulong dgi = (ulong)dst_t * p.Hv + hv;
                g_out[dgi]    = exp(neg_ea * softplus_f32(cur_a[sgi] + dtb));
                beta_out[dgi] = sigmoid_f32(cur_b[sgi]);
            }
        }
    } else {
        float cs0 = s0, cs1 = s1, cs2 = s2;
        for (uint t = 0u; t < p.M; t++) {
            int p1 = tp.parent[t];
            int p2 = (p1 >= 0) ? tp.parent[p1] : (p1 - 1);
            int p3 = (p2 >= 0) ? tp.parent[p2] : (p2 - 1);
            float x3 = (p3 >= 0) ? cur_qkv[(ulong)p3 * p.C + c] : ((p3 == -3) ? cs0 : ((p3 == -2) ? cs1 : cs2));
            float x2 = (p2 >= 0) ? cur_qkv[(ulong)p2 * p.C + c] : ((p2 == -2) ? cs1 : cs2);
            float x1 = (p1 >= 0) ? cur_qkv[(ulong)p1 * p.C + c] : cs2;
            float x0 = cur_qkv[(ulong)t * p.C + c];
            uint dst_t = 8u + t;
            qkv_conv_out[(ulong)dst_t * p.C + c] = silu_f32(fma(x3, w0, fma(x2, w1, fma(x1, w2, x0 * w3))));
            if (tid == 0u) {
                ulong sgi = (ulong)t * p.Hv + hv;
                ulong dgi = (ulong)dst_t * p.Hv + hv;
                g_out[dgi]    = exp(neg_ea * softplus_f32(cur_a[sgi] + dtb));
                beta_out[dgi] = sigmoid_f32(cur_b[sgi]);
            }
        }
    }
}

kernel void qwen_gdn_recurrence_lazy(
    const device float*        q_in     [[buffer(0)]],
    const device float*        k_in     [[buffer(1)]],
    const device float*        qkv_conv [[buffer(2)]],
    const device float*        g_in     [[buffer(3)]],
    const device float*        beta_in  [[buffer(4)]],
    device ushort*             state_io [[buffer(5)]],
    device float*              y_out    [[buffer(6)]],
    constant GdnRecParams&     p        [[buffer(7)]],
    constant PathReplayParams& pr       [[buffer(8)]],
    uint3 gid                           [[thread_position_in_grid]],
    uint  lane                          [[thread_index_in_simdgroup]]
) {
    uint dv0 = gid.y * 4u;
    uint hv  = gid.z;
    if (dv0 >= p.Dv || hv >= p.Hv) return;

    uint hk = hv / (p.Hv / p.Hk);
    uint v_offset = 2u * p.Hk * p.Dk + hv * p.Dv + dv0;
    uint base_k = lane * 4u;

    ulong st_base = ((ulong)hv * p.Dv + dv0) * p.Dk + base_k;
    float4 s0 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 0u * p.Dk)));
    float4 s1 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 1u * p.Dk)));
    float4 s2 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 2u * p.Dk)));
    float4 s3 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 3u * p.Dk)));

    // Phase A: Replay pr.len accepted tokens -> commit ONCE to state_io (with exact BF16 boundary parity!)
    if (pr.len > 0u) {
        for (uint t = 0u; t < pr.len; t++) {
            float g    = g_in[(ulong)t * p.Hv + hv];
            float beta = beta_in[(ulong)t * p.Hv + hv];
            float4 kv  = *((const device float4*)(k_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
            float4 vv  = *((const device float4*)(qkv_conv + (ulong)t * p.C + v_offset));

            s0 *= g; s1 *= g; s2 *= g; s3 *= g;
            float d0 = simd_sum(dot(s0, kv));
            float d1 = simd_sum(dot(s1, kv));
            float d2 = simd_sum(dot(s2, kv));
            float d3 = simd_sum(dot(s3, kv));

            s0 = fma(kv, float4((vv.x - d0) * beta), s0);
            s1 = fma(kv, float4((vv.y - d1) * beta), s1);
            s2 = fma(kv, float4((vv.z - d2) * beta), s2);
            s3 = fma(kv, float4((vv.w - d3) * beta), s3);
        }
        ushort4 c0 = f32x4_to_bf16x4(s0);
        ushort4 c1 = f32x4_to_bf16x4(s1);
        ushort4 c2 = f32x4_to_bf16x4(s2);
        ushort4 c3 = f32x4_to_bf16x4(s3);
        *((device ushort4*)(state_io + st_base + 0u * p.Dk)) = c0;
        *((device ushort4*)(state_io + st_base + 1u * p.Dk)) = c1;
        *((device ushort4*)(state_io + st_base + 2u * p.Dk)) = c2;
        *((device ushort4*)(state_io + st_base + 3u * p.Dk)) = c3;
        s0 = bf16x4_to_f32x4(c0);
        s1 = bf16x4_to_f32x4(c1);
        s2 = bf16x4_to_f32x4(c2);
        s3 = bf16x4_to_f32x4(c3);
    }

    // Phase B: Linear verify on slots [8 .. 8 + M - 1] (zero speculative state write!)
    for (uint t = 0u; t < p.M; t++) {
        uint st_t  = 8u + t;
        float g    = g_in[(ulong)st_t * p.Hv + hv];
        float beta = beta_in[(ulong)st_t * p.Hv + hv];
        float4 kv  = *((const device float4*)(k_in + ((ulong)st_t * p.Hk + hk) * p.Dk + base_k));
        float4 qv  = *((const device float4*)(q_in + ((ulong)st_t * p.Hk + hk) * p.Dk + base_k));
        float4 vv  = *((const device float4*)(qkv_conv + (ulong)st_t * p.C + v_offset));

        s0 *= g; s1 *= g; s2 *= g; s3 *= g;
        float d0 = simd_sum(dot(s0, kv));
        float d1 = simd_sum(dot(s1, kv));
        float d2 = simd_sum(dot(s2, kv));
        float d3 = simd_sum(dot(s3, kv));

        s0 = fma(kv, float4((vv.x - d0) * beta), s0);
        s1 = fma(kv, float4((vv.y - d1) * beta), s1);
        s2 = fma(kv, float4((vv.z - d2) * beta), s2);
        s3 = fma(kv, float4((vv.w - d3) * beta), s3);

        float o0 = simd_sum(dot(s0, qv));
        float o1 = simd_sum(dot(s1, qv));
        float o2 = simd_sum(dot(s2, qv));
        float o3 = simd_sum(dot(s3, qv));

        if (lane == 0u) {
            *((device float4*)(y_out + ((ulong)t * p.Hv + hv) * p.Dv + dv0)) = float4(o0, o1, o2, o3);
        }
    }
}

kernel void qwen_gdn_recurrence_tree_lazy(
    const device float*        q_in     [[buffer(0)]],
    const device float*        k_in     [[buffer(1)]],
    const device float*        qkv_conv [[buffer(2)]],
    const device float*        g_in     [[buffer(3)]],
    const device float*        beta_in  [[buffer(4)]],
    device ushort*             state_io [[buffer(5)]],
    device float*              y_out    [[buffer(6)]],
    constant GdnRecParams&     p        [[buffer(7)]],
    constant PathReplayParams& pr       [[buffer(8)]],
    constant TreeParams&       tp       [[buffer(9)]],
    uint3 gid                           [[thread_position_in_grid]],
    uint  tid                           [[thread_index_in_threadgroup]],
    uint  lane                          [[thread_index_in_simdgroup]]
) {
    uint dv0 = gid.y * 4u;
    uint hv  = gid.z;
    if (dv0 >= p.Dv || hv >= p.Hv) return;

    uint hk = hv / (p.Hv / p.Hk);
    uint v_offset = 2u * p.Hk * p.Dk + hv * p.Dv + dv0;
    uint base_k = lane * 4u;

    threadgroup ushort4 s_stack[4][4][128];

    ulong st_base = ((ulong)hv * p.Dv + dv0) * p.Dk + base_k;
    float4 s0 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 0u * p.Dk)));
    float4 s1 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 1u * p.Dk)));
    float4 s2 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 2u * p.Dk)));
    float4 s3 = bf16x4_to_f32x4(*((const device ushort4*)(state_io + st_base + 3u * p.Dk)));

    // Phase A: Replay pr.len accepted tokens -> commit ONCE to state_io (with exact BF16 boundary parity!)
    if (pr.len > 0u) {
        for (uint t = 0u; t < pr.len; t++) {
            float g    = g_in[(ulong)t * p.Hv + hv];
            float beta = beta_in[(ulong)t * p.Hv + hv];
            float4 kv  = *((const device float4*)(k_in + ((ulong)t * p.Hk + hk) * p.Dk + base_k));
            float4 vv  = *((const device float4*)(qkv_conv + (ulong)t * p.C + v_offset));

            s0 *= g; s1 *= g; s2 *= g; s3 *= g;
            float d0 = simd_sum(dot(s0, kv));
            float d1 = simd_sum(dot(s1, kv));
            float d2 = simd_sum(dot(s2, kv));
            float d3 = simd_sum(dot(s3, kv));

            s0 = fma(kv, float4((vv.x - d0) * beta), s0);
            s1 = fma(kv, float4((vv.y - d1) * beta), s1);
            s2 = fma(kv, float4((vv.z - d2) * beta), s2);
            s3 = fma(kv, float4((vv.w - d3) * beta), s3);
        }
        ushort4 c0 = f32x4_to_bf16x4(s0);
        ushort4 c1 = f32x4_to_bf16x4(s1);
        ushort4 c2 = f32x4_to_bf16x4(s2);
        ushort4 c3 = f32x4_to_bf16x4(s3);
        *((device ushort4*)(state_io + st_base + 0u * p.Dk)) = c0;
        *((device ushort4*)(state_io + st_base + 1u * p.Dk)) = c1;
        *((device ushort4*)(state_io + st_base + 2u * p.Dk)) = c2;
        *((device ushort4*)(state_io + st_base + 3u * p.Dk)) = c3;
        s0 = bf16x4_to_f32x4(c0);
        s1 = bf16x4_to_f32x4(c1);
        s2 = bf16x4_to_f32x4(c2);
        s3 = bf16x4_to_f32x4(c3);
    }

    // Phase B: Tree verify on slots [8 .. 8 + M - 1]
    for (uint t = 0u; t < p.M; t++) {
        int ld = tp.gdn_load[t];
        if (ld >= 0) {
            s0 = bf16x4_to_f32x4(s_stack[ld][0][tid]);
            s1 = bf16x4_to_f32x4(s_stack[ld][1][tid]);
            s2 = bf16x4_to_f32x4(s_stack[ld][2][tid]);
            s3 = bf16x4_to_f32x4(s_stack[ld][3][tid]);
        }

        uint st_t  = 8u + t;
        float g    = g_in[(ulong)st_t * p.Hv + hv];
        float beta = beta_in[(ulong)st_t * p.Hv + hv];
        float4 kv  = *((const device float4*)(k_in + ((ulong)st_t * p.Hk + hk) * p.Dk + base_k));
        float4 qv  = *((const device float4*)(q_in + ((ulong)st_t * p.Hk + hk) * p.Dk + base_k));
        float4 vv  = *((const device float4*)(qkv_conv + (ulong)st_t * p.C + v_offset));

        s0 *= g; s1 *= g; s2 *= g; s3 *= g;
        float d0 = simd_sum(dot(s0, kv));
        float d1 = simd_sum(dot(s1, kv));
        float d2 = simd_sum(dot(s2, kv));
        float d3 = simd_sum(dot(s3, kv));

        s0 = fma(kv, float4((vv.x - d0) * beta), s0);
        s1 = fma(kv, float4((vv.y - d1) * beta), s1);
        s2 = fma(kv, float4((vv.z - d2) * beta), s2);
        s3 = fma(kv, float4((vv.w - d3) * beta), s3);

        float o0 = simd_sum(dot(s0, qv));
        float o1 = simd_sum(dot(s1, qv));
        float o2 = simd_sum(dot(s2, qv));
        float o3 = simd_sum(dot(s3, qv));

        if (lane == 0u) {
            *((device float4*)(y_out + ((ulong)t * p.Hv + hv) * p.Dv + dv0)) = float4(o0, o1, o2, o3);
        }

        int sv = tp.gdn_save[t];
        if (sv >= 0) {
            s_stack[sv][0][tid] = f32x4_to_bf16x4(s0);
            s_stack[sv][1][tid] = f32x4_to_bf16x4(s1);
            s_stack[sv][2][tid] = f32x4_to_bf16x4(s2);
            s_stack[sv][3][tid] = f32x4_to_bf16x4(s3);
        }
    }
}

kernel void qwen_full_attn_q_prep_tree(
    const device float*  q_proj  [[buffer(0)]],
    const device ushort* q_norm  [[buffer(1)]],
    device float*        q_out   [[buffer(2)]],
    device float*        gate_out[[buffer(3)]],
    constant FullAttnParams& p   [[buffer(4)]],
    constant TreeParams& tp      [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    const device float* src = q_proj + ((ulong)t * p.Hq + h) * (2u * p.D);
    device float* qdst = q_out + ((ulong)t * p.Hq + h) * p.D;
    device float* gdst = gate_out + ((ulong)t * p.Hq + h) * p.D;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = src[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        gdst[idx] = src[p.D + idx];
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(q_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle = float(int(tp.mask_base_pos + tp.depth[t]) + p.rope_offset) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        qdst[i * 32u + lane] = v[i];
    }
}

kernel void qwen_full_attn_kv_prep_tree(
    const device float*  k_proj   [[buffer(0)]],
    const device float*  v_proj   [[buffer(1)]],
    const device ushort* k_norm   [[buffer(2)]],
    device half*         k_cache  [[buffer(3)]],
    device half*         v_cache  [[buffer(4)]],
    constant FullAttnParams& p    [[buffer(5)]],
    constant TreeParams& tp       [[buffer(6)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    const device float* ksrc = k_proj + ((ulong)t * p.Hkv + h) * p.D;
    const device float* vsrc = v_proj + ((ulong)t * p.Hkv + h) * p.D;
    ulong cache_off = ((ulong)(p.pos + t) * p.Hkv + h) * p.D;
    device half* kdst = k_cache + cache_off;
    device half* vdst = v_cache + cache_off;

    float v[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = ksrc[idx];
        v[i] = x;
        sq = fma(x, x, sq);
        vdst[idx] = half(vsrc[idx]);
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        v[i] = v[i] * inv_rms * bf16_to_f32(k_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle = float(int(tp.mask_base_pos + tp.depth[t]) + p.rope_offset) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = v[0];
        float x1 = v[1];
        v[0] = fma(x0, c, -x1 * s);
        v[1] = fma(x1, c,  x0 * s);
    }

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        kdst[i * 32u + lane] = half(v[i]);
    }
}

template<int KV_BITS>
inline void qwen_full_attn_kv_prep_quant_tree_impl(
    const device float*  k_proj,
    const device float*  v_proj,
    const device ushort* k_norm,
    device uchar*        k_cache,
    device uchar*        v_cache,
    device float2*       k_sb,
    device float2*       v_sb,
    constant FullAttnParams& p,
    constant TreeParams& tp,
    uint2 tgid,
    uint  lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    const device float* ksrc = k_proj + ((ulong)t * p.Hkv + h) * p.D;
    const device float* vsrc = v_proj + ((ulong)t * p.Hkv + h) * p.D;
    constexpr uint BYTES_PER_HEAD = (KV_BITS == 8) ? 256u : 128u;
    ulong head_idx = (ulong)(p.pos + t) * p.Hkv + h;
    device uchar*  kdst = k_cache + head_idx * BYTES_PER_HEAD;
    device uchar*  vdst = v_cache + head_idx * BYTES_PER_HEAD;
    device float2* ksbd = k_sb + head_idx * 4ul;
    device float2* vsbd = v_sb + head_idx * 4ul;

    float kv[8], vv[8];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        float x = ksrc[idx];
        kv[i] = x;
        sq = fma(x, x, sq);
        vv[i] = vsrc[idx];
    }
    float inv_rms = rsqrt(simd_sum(sq) / float(p.D) + p.eps);

    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        kv[i] = kv[i] * inv_rms * bf16_to_f32(k_norm[idx]);
    }

    uint half_rot = p.rotary_dim >> 1;
    if (lane < half_rot) {
        float freq = 1.0f / pow(p.rope_theta, float(2u * lane) / float(p.rotary_dim));
        float angle = float(int(tp.mask_base_pos + tp.depth[t]) + p.rope_offset) * freq;
        float c = cos(angle);
        float s = sin(angle);
        float x0 = kv[0];
        float x1 = kv[1];
        kv[0] = fma(x0, c, -x1 * s);
        kv[1] = fma(x1, c,  x0 * s);
    }

    constexpr float QMAX = (KV_BITS == 8) ? 255.0f : 15.0f;
    #pragma unroll
    for (uint g = 0; g < 4u; g++) {
        float ka = kv[2u * g], kb = kv[2u * g + 1u];
        float kmn = simd_min(min(ka, kb)), kmx = simd_max(max(ka, kb));
        float krng = kmx - kmn;
        float fs_k = (krng > 1e-12f) ? (krng / QMAX) : 0.0f;
        float inv_k = (krng > 1e-12f) ? (QMAX / krng) : 0.0f;
        if (lane == 0u) ksbd[g] = float2(fs_k, kmn);
        uint qk0 = uint(clamp(rint((ka - kmn) * inv_k), 0.0f, QMAX));
        uint qk1 = uint(clamp(rint((kb - kmn) * inv_k), 0.0f, QMAX));
        if (KV_BITS == 8) {
            kdst[(2u * g) * 32u + lane]      = uchar(qk0);
            kdst[(2u * g + 1u) * 32u + lane] = uchar(qk1);
        } else {
            kdst[g * 32u + lane] = uchar(qk0 | (qk1 << 4u));
        }

        float va = vv[2u * g], vb = vv[2u * g + 1u];
        float vmn = simd_min(min(va, vb)), vmx = simd_max(max(va, vb));
        float vrng = vmx - vmn;
        float fs_v = (vrng > 1e-12f) ? (vrng / QMAX) : 0.0f;
        float inv_v = (vrng > 1e-12f) ? (QMAX / vrng) : 0.0f;
        if (lane == 0u) vsbd[g] = float2(fs_v, vmn);
        uint qv0 = uint(clamp(rint((va - vmn) * inv_v), 0.0f, QMAX));
        uint qv1 = uint(clamp(rint((vb - vmn) * inv_v), 0.0f, QMAX));
        if (KV_BITS == 8) {
            vdst[(2u * g) * 32u + lane]      = uchar(qv0);
            vdst[(2u * g + 1u) * 32u + lane] = uchar(qv1);
        } else {
            vdst[g * 32u + lane] = uchar(qv0 | (qv1 << 4u));
        }
    }
}

kernel void qwen_full_attn_kv_prep_q8_tree(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    constant TreeParams& tp [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_tree_impl<8>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, tp, tgid, lane);
}

kernel void qwen_full_attn_kv_prep_q4_tree(
    const device float* k_proj [[buffer(0)]], const device float* v_proj [[buffer(1)]],
    const device ushort* k_norm [[buffer(2)]], device uchar* k_cache [[buffer(3)]],
    device uchar* v_cache [[buffer(4)]], device float2* k_sb [[buffer(5)]],
    device float2* v_sb [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    constant TreeParams& tp [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    qwen_full_attn_kv_prep_quant_tree_impl<4>(k_proj, v_proj, k_norm, k_cache, v_cache, k_sb, v_sb, p, tp, tgid, lane);
}

// Short-context (< 64) tree-masked GQA attention (FP16 / Q8 / Q4 KV, FP16 output)
kernel void qwen_gqa_attn_gated_tree_f16(
    const device float* q_in    [[buffer(0)]],
    const device float* gate_in [[buffer(1)]],
    const device half*  k_cache [[buffer(2)]],
    const device half*  v_cache [[buffer(3)]],
    device half*        out     [[buffer(4)]],
    constant FullAttnParams& p  [[buffer(5)]],
    constant TreeParams& tp     [[buffer(6)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device float* gp = gate_in + q_off;
    device half* op = out + q_off;

    float qv[8], acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) { qv[i] = qp[i * 32u + lane]; acc[i] = 0.0f; }

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    uint tmask = tp.mask[t];
    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    for (uint kpos = 0; kpos < p.kv_len; kpos++) {
        if (kpos >= tp.mask_base_pos && (((tmask >> (kpos - tp.mask_base_pos)) & 1u) == 0u)) continue;
        const device half* kp = k_cache + (ulong)kpos * kv_stride + head_off;
        float dot = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) dot = fma(qv[i], float(kp[i * 32u + lane]), dot);
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;
        const device half* vp = v_cache + (ulong)kpos * kv_stride + head_off;
        #pragma unroll
        for (uint i = 0; i < 8u; i++) acc[i] = fma(acc[i], alpha, beta * float(vp[i * 32u + lane]));
    }

    float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = half((acc[i] * inv_l) * sigmoid_f32(gp[idx]));
    }
}

template<int KV_BITS>
inline void qwen_gqa_attn_gated_quant_tree_f16_impl(
    const device float* q_in,
    const device float* gate_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device half*        out,
    constant FullAttnParams& p,
    constant TreeParams& tp,
    uint2 tgid, uint lane
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device float* gp = gate_in + q_off;
    device half* op = out + q_off;

    float qv[8], acc[8];
    #pragma unroll
    for (uint i = 0; i < 8u; i++) { qv[i] = qp[i * 32u + lane]; acc[i] = 0.0f; }

    float m_max = -INFINITY;
    float l_sum = 0.0f;
    uint tmask = tp.mask[t];
    constexpr uint BYTES_PER_HEAD = (KV_BITS == 8) ? 256u : 128u;

    for (uint kpos = 0; kpos < p.kv_len; kpos++) {
        if (kpos >= tp.mask_base_pos && (((tmask >> (kpos - tp.mask_base_pos)) & 1u) == 0u)) continue;
        ulong head_idx = (ulong)kpos * p.Hkv + hkv;
        const device uchar* kp = k_cache + head_idx * BYTES_PER_HEAD;
        const device float2* ksb = k_sb + head_idx * 4ul;
        float dot = 0.0f;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(ksb[g].x), mn = float(ksb[g].y);
            float k0, k1;
            if (KV_BITS == 8) {
                k0 = fma(float(kp[(2u * g) * 32u + lane]), sc, mn);
                k1 = fma(float(kp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pk = kp[g * 32u + lane];
                k0 = fma(float(pk & 0x0Fu), sc, mn);
                k1 = fma(float(pk >> 4u), sc, mn);
            }
            dot = fma(qv[2u * g], k0, fma(qv[2u * g + 1u], k1, dot));
        }
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device uchar* vp = v_cache + head_idx * BYTES_PER_HEAD;
        const device float2* vsb = v_sb + head_idx * 4ul;
        #pragma unroll
        for (uint g = 0; g < 4u; g++) {
            float sc = float(vsb[g].x), mn = float(vsb[g].y);
            float v0, v1;
            if (KV_BITS == 8) {
                v0 = fma(float(vp[(2u * g) * 32u + lane]), sc, mn);
                v1 = fma(float(vp[(2u * g + 1u) * 32u + lane]), sc, mn);
            } else {
                uchar pv = vp[g * 32u + lane];
                v0 = fma(float(pv & 0x0Fu), sc, mn);
                v1 = fma(float(pv >> 4u), sc, mn);
            }
            acc[2u * g]      = fma(acc[2u * g],      alpha, beta * v0);
            acc[2u * g + 1u] = fma(acc[2u * g + 1u], alpha, beta * v1);
        }
    }

    float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) {
        uint idx = i * 32u + lane;
        op[idx] = half((acc[i] * inv_l) * sigmoid_f32(gp[idx]));
    }
}

kernel void qwen_gqa_attn_gated_q8_tree_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    constant TreeParams& tp [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_tree_f16_impl<8>(q, g, kc, vc, ksb, vsb, out, p, tp, tgid, lane); }

kernel void qwen_gqa_attn_gated_q4_tree_f16(
    const device float* q [[buffer(0)]], const device float* g [[buffer(1)]],
    const device uchar* kc [[buffer(2)]], const device uchar* vc [[buffer(3)]],
    const device float2* ksb [[buffer(4)]], const device float2* vsb [[buffer(5)]],
    device half* out [[buffer(6)]], constant FullAttnParams& p [[buffer(7)]],
    constant TreeParams& tp [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) { qwen_gqa_attn_gated_quant_tree_f16_impl<4>(q, g, kc, vc, ksb, vsb, out, p, tp, tgid, lane); }

// Tree-masked AMX Split-K Verify Pass 1 (FP16 KV, handles M <= 16 via tgid.z = 0..1!)
kernel void qwen_gqa_verify_splitk_p1_tree(
    const device float* q_in     [[buffer(0)]],
    const device half*  k_cache  [[buffer(1)]],
    const device half*  v_cache  [[buffer(2)]],
    device float*       part_o   [[buffer(3)]],
    device float*       part_m   [[buffer(4)]],
    device float*       part_l   [[buffer(5)]],
    constant FullAttnParams& p   [[buffer(6)]],
    constant TreeParams& tp      [[buffer(7)]],
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_index_in_threadgroup]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint pair_idx = tgid.x;
    uint s        = tgid.y;
    uint t_base   = tgid.z * 8u;
    if (pair_idx >= p.Hkv * pairs_per_kv || s >= p.num_splits || t_base >= p.M) return;

    uint hkv   = pair_idx / pairs_per_kv;
    uint gp    = pair_idx % pairs_per_kv;
    uint h0    = hkv * gqa + gp * 2u;
    uint h1    = h0 + 1u;
    bool h1_ok = (gp * 2u + 1u < gqa);

    uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
    uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
    uint kb_start        = s * tiles_per_split * BK;
    uint kb_end          = min(p.kv_len, (s + 1u) * tiles_per_split * BK);

    if (kb_start >= kb_end) {
        if (tid < BQ) {
            uint lh = tid >> 3;
            uint t  = t_base + (tid & 7u);
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = -INFINITY;
                part_l[slot] = 0.0f;
            }
        }
        return;
    }

    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    threadgroup half  sQ[BQ][260];
    threadgroup half  sKV[BK][132];
    threadgroup float sS[BQ][BK];
    threadgroup half  sP[BQ][36];
    threadgroup float sAlpha[BQ];

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint lh = qr >> 3;
        uint t  = t_base + (qr & 7u);
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (t < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)t * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;

    uint kr = tid >> 2;
    uint kc = (tid & 3u) * 32u;
    threadgroup half4* skv_dst = (threadgroup half4*)(&sKV[kr][kc]);

    for (uint kb = kb_start; kb < kb_end; kb += BK) {
        uint tile_k = min(BK, kb_end - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (kr < tile_k) {
            const device half4* kp = (const device half4*)(k_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = kp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = (kb + BK <= tp.mask_base_pos) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == kb_start) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint t = t_base + (r & 7u);
                uint k_pos = kb + lane;
                bool tree_ok = (t < p.M) && ((k_pos < tp.mask_base_pos) || (((tp.mask[t] >> (k_pos - tp.mask_base_pos)) & 1u) != 0u));
                bool valid = tree_ok && (lane < tile_k);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (kr < tile_k) {
            const device half4* vp = (const device half4*)(v_cache + (ulong)(kb + kr) * kv_stride + head_off + 128u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = vp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) skv_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            uint r  = sg * 4u + rr;
            uint lh = r >> 3;
            uint t  = t_base + (r & 7u);
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = m_row[rr];
                part_l[slot] = l_row[rr];
            }
        }
    }

    uint row = sg_m + fm;
    uint lh  = row >> 3;
    uint t   = t_base + (row & 7u);
    uint h   = (lh == 0u) ? h0 : h1;
    if (t < p.M && (lh == 0u || h1_ok)) {
        ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
        device float* po = part_o + slot * p.D;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            po[c0]           = O0[j].thread_elements()[0];
            po[c0 + 1u]      = O0[j].thread_elements()[1];
            po[128u + c0]    = O1[j].thread_elements()[0];
            po[128u + c0 + 1u] = O1[j].thread_elements()[1];
        }
    }
}

template<int KV_BITS>
inline void qwen_gqa_verify_splitk_p1_quant_tree_impl(
    const device float* q_in,
    const device uchar* k_cache,
    const device uchar* v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*       part_o,
    device float*       part_m,
    device float*       part_l,
    constant FullAttnParams& p,
    constant TreeParams& tp,
    uint3 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sQ)[16][260],
    threadgroup half  (&sKV)[32][132],
    threadgroup float (&sS)[16][32],
    threadgroup half  (&sP)[16][36],
    threadgroup float (&sAlpha)[16]
) {
    constexpr uint BQ = 16u;
    constexpr uint BK = 32u;

    uint gqa = p.Hq / p.Hkv;
    uint pairs_per_kv = (gqa + 1u) >> 1;
    uint pair_idx = tgid.x;
    uint s        = tgid.y;
    uint t_base   = tgid.z * 8u;
    if (pair_idx >= p.Hkv * pairs_per_kv || s >= p.num_splits || t_base >= p.M) return;

    uint hkv   = pair_idx / pairs_per_kv;
    uint gp    = pair_idx % pairs_per_kv;
    uint h0    = hkv * gqa + gp * 2u;
    uint h1    = h0 + 1u;
    bool h1_ok = (gp * 2u + 1u < gqa);

    uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
    uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
    uint kb_start        = s * tiles_per_split * BK;
    uint kb_end          = min(p.kv_len, (s + 1u) * tiles_per_split * BK);

    if (kb_start >= kb_end) {
        if (tid < BQ) {
            uint lh = tid >> 3;
            uint t  = t_base + (tid & 7u);
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = -INFINITY;
                part_l[slot] = 0.0f;
            }
        }
        return;
    }

    {
        uint qr = tid >> 3;
        uint qc = (tid & 7u) * 32u;
        uint lh = qr >> 3;
        uint t  = t_base + (qr & 7u);
        uint h  = (lh == 0u) ? h0 : h1;
        threadgroup half4* sq_dst = (threadgroup half4*)(&sQ[qr][qc]);
        if (t < p.M && (lh == 0u || h1_ok)) {
            const device float4* qp = (const device float4*)(q_in + ((ulong)t * p.Hq + h) * p.D + qc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(qp[i]);
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sq_dst[i] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = (sg >> 1) * 8u;
    uint sg_d = (sg & 1u) * 64u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O1[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_row[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint kb = kb_start; kb < kb_end; kb += BK) {
        uint tile_k = min(BK, kb_end - kb);
        simdgroup_float8x8 s_acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s_acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][kk], 260);
            simdgroup_load(q1, &sQ[8][kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(k_cache, k_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 q0, q1, k0;
            simdgroup_load(q0, &sQ[0][128u + kk], 260);
            simdgroup_load(q1, &sQ[8][128u + kk], 260);
            simdgroup_load(k0, &sKV[sg * 8u][kk], 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s_acc0, q0, k0, s_acc0);
            simdgroup_multiply_accumulate(s_acc1, q1, k0, s_acc1);
        }

        sS[fm][sg * 8u + fn]           = s_acc0.thread_elements()[0];
        sS[fm][sg * 8u + fn + 1u]      = s_acc0.thread_elements()[1];
        sS[8u + fm][sg * 8u + fn]      = s_acc1.thread_elements()[0];
        sS[8u + fm][sg * 8u + fn + 1u] = s_acc1.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool interior = (kb + BK <= tp.mask_base_pos) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float score = sS[r][lane] * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == kb_start) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                uint t = t_base + (r & 7u);
                uint k_pos = kb + lane;
                bool tree_ok = (t < p.M) && ((k_pos < tp.mask_base_pos) || (((tp.mask[t] >> (k_pos - tp.mask_base_pos)) & 1u) != 0u));
                bool valid = tree_ok && (lane < tile_k);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile_quant<KV_BITS>(v_cache, v_sb, sKV, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][sg_d + 8u * j], 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            uint r  = sg * 4u + rr;
            uint lh = r >> 3;
            uint t  = t_base + (r & 7u);
            uint h  = (lh == 0u) ? h0 : h1;
            if (t < p.M && (lh == 0u || h1_ok)) {
                ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
                part_m[slot] = m_row[rr];
                part_l[slot] = l_row[rr];
            }
        }
    }

    uint row = sg_m + fm;
    uint lh = row >> 3;
    uint t  = t_base + (row & 7u);
    uint h  = (lh == 0u) ? h0 : h1;
    if (t < p.M && (lh == 0u || h1_ok)) {
        ulong slot = ((ulong)t * p.Hq + h) * p.num_splits + s;
        device float* po = part_o + slot * p.D;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = sg_d + 8u * j + fn;
            po[c0]      = O0[j].thread_elements()[0];
            po[c0 + 1u] = O0[j].thread_elements()[1];
            uint c1 = 128u + c0;
            po[c1]      = O1[j].thread_elements()[0];
            po[c1 + 1u] = O1[j].thread_elements()[1];
        }
    }
}

kernel void qwen_gqa_verify_splitk_p1_q8_tree(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16];
    qwen_gqa_verify_splitk_p1_quant_tree_impl<8>(q, kc, vc, ksb, vsb, po, pm, pl, p, tp, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha);
}

kernel void qwen_gqa_verify_splitk_p1_q4_tree(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sQ[16][260], sKV[32][132], sP[16][36];
    threadgroup float sS[16][32], sAlpha[16];
    qwen_gqa_verify_splitk_p1_quant_tree_impl<4>(q, kc, vc, ksb, vsb, po, pm, pl, p, tp, tgid, tid, lane, sg, sQ, sKV, sS, sP, sAlpha);
}

// Winning-path KV cache in-place compaction (safe because DFS path[j] >= j strictly increasing!)
kernel void qwen_kv_compact_f16(
    device half* k_cache [[buffer(0)]],
    device half* v_cache [[buffer(1)]],
    constant PathReplayParams& pr [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint row_elems = pr.Hkv * pr.D;
    if (tid >= row_elems) return;
    for (uint j = 0; j < pr.len; ++j) {
        uint src = pr.path[j];
        if (src != j) {
            ulong d_off = (ulong)(pr.pre_pos + j) * row_elems + tid;
            ulong s_off = (ulong)(pr.pre_pos + src) * row_elems + tid;
            k_cache[d_off] = k_cache[s_off];
            v_cache[d_off] = v_cache[s_off];
        }
    }
}

kernel void qwen_kv_compact_q(
    device uchar*  k_cache [[buffer(0)]],
    device uchar*  v_cache [[buffer(1)]],
    device float2* k_sb    [[buffer(2)]],
    device float2* v_sb    [[buffer(3)]],
    constant PathReplayParams& pr [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint row_bytes = pr.Hkv * ((pr.kv_bits == 8u) ? pr.D : (pr.D >> 1));
    uint sb_elems  = pr.Hkv * (pr.D >> 6);
    if (tid < row_bytes) {
        for (uint j = 0; j < pr.len; ++j) {
            uint src = pr.path[j];
            if (src != j) {
                ulong d_off = (ulong)(pr.pre_pos + j) * row_bytes + tid;
                ulong s_off = (ulong)(pr.pre_pos + src) * row_bytes + tid;
                k_cache[d_off] = k_cache[s_off];
                v_cache[d_off] = v_cache[s_off];
            }
        }
    }
    if (tid < sb_elems) {
        for (uint j = 0; j < pr.len; ++j) {
            uint src = pr.path[j];
            if (src != j) {
                ulong d_off = (ulong)(pr.pre_pos + j) * sb_elems + tid;
                ulong s_off = (ulong)(pr.pre_pos + src) * sb_elems + tid;
                k_sb[d_off] = k_sb[s_off];
                v_sb[d_off] = v_sb[s_off];
            }
        }
    }
}

// Gather winning path hidden rows src_hid[path[j]] -> dst_hid[j] for exact MTP KV sync
kernel void qwen_copy_hid_path(
    const device float* src_hid [[buffer(0)]],
    device float*       dst_hid [[buffer(1)]],
    constant PathReplayParams& pr [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint j = gid.y;
    if (d >= pr.D || j >= pr.len) return;
    dst_hid[(ulong)j * pr.D + d] = src_hid[(ulong)pr.path[j] * pr.D + d];
}


struct MtpGatherParams {
    uint  B;
    uint  D;
    float eps;
    uint  pad;
    uint  in_tok_idx[8];
    uint  in_hid_idx[8];
    uint  out_hid_idx[8];
};

// Stage 3: Batched Q4 Embed + Dual Pre-FC RMSNorm gathering arbitrary tree parent hidden rows & token IDs -> FP16 [B, 2*D]
kernel void qwen_mtp_embed_concat_gather_f16(
    const device uint*   emb_W [[buffer(0)]],
    const device ushort* emb_S [[buffer(1)]],
    const device ushort* emb_B [[buffer(2)]],
    const device int*    ids   [[buffer(3)]],
    const device float*  hid   [[buffer(4)]],
    const device ushort* w_emb [[buffer(5)]],
    const device ushort* w_hid [[buffer(6)]],
    device half*         out   [[buffer(7)]],
    constant MtpGatherParams& p [[buffer(8)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    if (row >= p.B) return;
    const uint tok = uint(ids[p.in_tok_idx[row]]);
    const uint KW = p.D >> 3;
    const uint KG = p.D >> 6;
    const device uint*   wr = emb_W + (ulong)tok * KW;
    const device ushort* sr = emb_S + (ulong)tok * KG;
    const device ushort* br = emb_B + (ulong)tok * KG;
    const device float*  hp = hid + (ulong)p.in_hid_idx[row] * p.D;
    device half*         op = out + (ulong)row * (2u * p.D);

    threadgroup float se[32], sh[32];
    const uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;

    if (p.D == 5120u && tg_size == 256u) {
        float ve_reg[20], vh_reg[20];
        float sq_e = 0.0f, sq_h = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint i = k * 256u + tid;
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            float vh = hp[i];
            ve_reg[k] = ve;
            vh_reg[k] = vh;
            sq_e = fma(ve, ve, sq_e);
            sq_h = fma(vh, vh, sq_h);
        }
        sq_e = simd_sum(sq_e);
        sq_h = simd_sum(sq_h);
        if (lane == 0u) { se[sg] = sq_e; sh[sg] = sq_h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float te = simd_sum((lane < nsg) ? se[lane] : 0.0f);
            float th = simd_sum((lane < nsg) ? sh[lane] : 0.0f);
            if (lane == 0u) {
                se[0] = rsqrt(te / float(p.D) + p.eps);
                sh[0] = rsqrt(th / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ire = se[0], irh = sh[0];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint i = k * 256u + tid;
            op[i]       = half(ve_reg[k] * ire * bf16_to_f32(w_emb[i]));
            op[p.D + i] = half(vh_reg[k] * irh * bf16_to_f32(w_hid[i]));
        }
    } else {
        float sq_e = 0.0f, sq_h = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            float vh = hp[i];
            sq_e = fma(ve, ve, sq_e);
            sq_h = fma(vh, vh, sq_h);
        }
        sq_e = simd_sum(sq_e);
        sq_h = simd_sum(sq_h);
        if (lane == 0u) { se[sg] = sq_e; sh[sg] = sq_h; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float te = simd_sum((lane < nsg) ? se[lane] : 0.0f);
            float th = simd_sum((lane < nsg) ? sh[lane] : 0.0f);
            if (lane == 0u) {
                se[0] = rsqrt(te / float(p.D) + p.eps);
                sh[0] = rsqrt(th / float(p.D) + p.eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ire = se[0], irh = sh[0];
        for (uint i = tid; i < p.D; i += tg_size) {
            uint word = wr[i >> 3];
            float q = float((word >> ((i & 7u) * 4u)) & 15u);
            uint g = i >> 6;
            float ve = fma(q, bf16_to_f32(sr[g]), bf16_to_f32(br[g]));
            op[i]       = half(ve * ire * bf16_to_f32(w_emb[i]));
            op[p.D + i] = half(hp[i] * irh * bf16_to_f32(w_hid[i]));
        }
    }
}

// Stage 3: Batched dual RMSNorm scattering FP32 hidden rows to dst_f32[out_hid_idx[row]] AND contiguous FP16 dst_f16[row]
kernel void qwen_rmsnorm_dual_scatter_f16(
    const device float*  x       [[buffer(0)]],
    const device ushort* w       [[buffer(1)]],
    device float*        dst_f32 [[buffer(2)]],
    device half*         dst_f16 [[buffer(3)]],
    constant MtpGatherParams& p  [[buffer(4)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    if (row >= p.B) return;
    const device float* xp = x + (ulong)row * p.D;
    device float*       of = dst_f32 + (ulong)p.out_hid_idx[row] * p.D;
    device half*        oh = dst_f16 + (ulong)row * p.D;

    float sq = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = xp[i];
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float smem[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) smem[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
        if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = smem[0];
    for (uint i = tid; i < p.D; i += tg_size) {
        float val = xp[i] * inv_rms * bf16_to_f32(w[i]);
        of[i] = val;
        oh[i] = half(val);
    }
}

// ============================================================================
// GQA-6 Grouped KV-Sharing AMX Split-K Attention Ladder (1x KV DRAM read across all 6 Q heads!)
//   1. qwen_gqa6_m1_splitk_p1_{f16,q8,q4}{,_tree}: M==1 (128 thr = 4 SGs x 64 dims, BCF sKV[32][264] -> 386+ GB/s!)
//   2. qwen_gqa6_m2_splitk_p1_{f16,q8,q4}{,_tree}: M==2 (128 thr = 4 SGs x 64 dims, BCF sKV[32][264] -> 276+ GB/s!)
//   3. qwen_gqa6_m4_splitk_p1_{f16,q8,q4}{,_tree}: M=3..4 (128 thr = 4 SGs x 64 dims, 3x8 O[3][8], BCF sKV[32][264] -> 19.56 ms @ 128k!)
//   4. qwen_gqa6_verify_splitk_p1_{f16,q8,q4}{,_tree}: M=5..8 (192 thr = 6 SGs, 11.7 KB SRAM, 74% of M4 Max AMX ceiling!)
// Output layout matches qwen_gqa_verify_splitk_p2(_f16) (log2 m_max, slot = (t*Hq+h)*num_splits+s).
// ============================================================================

// Helper: load [BK=32, 256] full-D half tile from FP16 / Q8 / Q4 KV cache using 128 threads (32 rows x 4 groups of 64)
// Bank-conflict-free stride STR=264 eliminates SRAM bank conflicts on transposed AMX simdgroup_load!
template<int KV_BITS, uint STR = 264u>
inline void load_kv_tile32_full256(
    const device uchar*  kv_cache,
    const device float2* kv_sb,
    threadgroup half (&sKV)[32][STR],
    uint kb, uint tile_k, uint hkv, uint Hkv, uint tid
) {
    uint kr  = tid >> 2;
    uint grp = tid & 3u;
    threadgroup half4* dst = (threadgroup half4*)(&sKV[kr][grp * 64u]);
    if (kr < tile_k) {
        ulong head_idx = (ulong)(kb + kr) * Hkv + hkv;
        if (KV_BITS == 16) {
            const device half4* hp = (const device half4*)((const device half*)kv_cache + head_idx * 256ul + grp * 64u);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; i++) dst[i] = hp[i];
        } else if (KV_BITS == 8) {
            float2 sb = kv_sb[head_idx * 4ul + grp];
            float4 fs = float4(sb.x), fb = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 256ul + grp * 64u);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; i++) dst[i] = half4(fma(float4(qp[i]), fs, fb));
        } else {
            float2 sb = kv_sb[head_idx * 4ul + grp];
            float4 fs = float4(sb.x), fb = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 128ul + grp * 32u);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) {
                uchar4 u = qp[i];
                dst[i]      = half4(fma(float4(u & uchar4(0x0Fu)), fs, fb));
                dst[8u + i] = half4(fma(float4(u >> uchar4(4u)),   fs, fb));
            }
        }
    } else {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16u; i++) dst[i] = half4(0.0h);
    }
}

// 1. M=1 AMX GQA-6: packs all 6 query heads into ONE 8x256 AMX tile, 128 threads = 4 SGs x 64 dims (O[8] only!)
template<int KV_BITS, bool IS_TREE, bool IS_SPARSE = false>
inline void qwen_gqa6_m1_splitk_p1_impl(
    const device float*  q_in,
    const device uchar*  k_cache,
    const device uchar*  v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*        part_o,
    device float*        part_m,
    device float*        part_l,
    constant FullAttnParams& p,
    constant TreeParams* tp,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half (&sKV)[32][264],
    threadgroup half (&sS)[4][8][32],
    threadgroup half (&sP)[8][36],
    threadgroup half (&sA)[8],
    const device uint* active_pages = nullptr,
    const device uint* active_counts = nullptr
) {
    constexpr uint BK = 32u, STR = 264u;
    uint hkv = tgid.x;
    uint s   = tgid.y;
    if (hkv >= p.Hkv || s >= p.num_splits) return;

    uint iter_start, iter_end;
    if (IS_SPARSE) {
        uint n_active        = active_counts[hkv];
        uint pages_per_split = (n_active + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * pages_per_split;
        iter_end             = min(n_active, (s + 1u) * pages_per_split);
    } else {
        uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
        uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * tiles_per_split;
        iter_end             = min(num_k_tiles, (s + 1u) * tiles_per_split);
    }

    if (iter_start >= iter_end) {
        if (tid < 6u) {
            ulong slot = (ulong)(hkv * 6u + tid) * p.num_splits + s;
            part_m[slot] = -INFINITY;
            part_l[slot] = 0.0f;
        }
        return;
    }

    // Load 6 query heads into sKV[0..7][0..255] once, then keep Qm[8] in SG registers for the entire loop!
    for (uint i = tid; i < 8u * 256u; i += 128u) {
        uint h_loc = i >> 8;
        uint d     = i & 255u;
        sKV[h_loc][d] = (h_loc < 6u) ? half(q_in[(ulong)(hkv * 6u + h_loc) * 256ul + d]) : 0.0h;
    }
    if (tid < 64u) {
        sP[6u + (tid >> 5)][tid & 31u] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 Qm[8], O[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        simdgroup_load(Qm[j], &sKV[0][sg * 64u + j * 8u], STR);
        O[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    float m_row[6], l_row[6];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 6u; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint it = iter_start; it < iter_end; ++it) {
        uint kb = IS_SPARSE ? (active_pages[(ulong)hkv * 1024ul + it] * BK) : (it * BK);
        uint tile_k = min(BK, p.kv_len - kb);
        load_kv_tile32_full256<KV_BITS, STR>(k_cache, k_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 s0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        simdgroup_half8x8 s1 = s0, s2 = s0, s3 = s0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint kk = sg * 64u + j * 8u;
            simdgroup_half8x8 km0, km1, km2, km3;
            simdgroup_load(km0, &sKV[0][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km1, &sKV[8][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km2, &sKV[16][kk], STR, ulong2(0, 0), true);
            simdgroup_load(km3, &sKV[24][kk], STR, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, Qm[j], km0, s0);
            simdgroup_multiply_accumulate(s1, Qm[j], km1, s1);
            simdgroup_multiply_accumulate(s2, Qm[j], km2, s2);
            simdgroup_multiply_accumulate(s3, Qm[j], km3, s3);
        }
        sS[sg][fm][0u  + fn] = s0.thread_elements()[0]; sS[sg][fm][0u  + fn + 1u] = s0.thread_elements()[1];
        sS[sg][fm][8u  + fn] = s1.thread_elements()[0]; sS[sg][fm][8u  + fn + 1u] = s1.thread_elements()[1];
        sS[sg][fm][16u + fn] = s2.thread_elements()[0]; sS[sg][fm][16u + fn + 1u] = s2.thread_elements()[1];
        sS[sg][fm][24u + fn] = s3.thread_elements()[0]; sS[sg][fm][24u + fn + 1u] = s3.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u) {
            uint mask_base = IS_TREE ? tp->mask_base_pos : p.pos;
            bool interior  = (kb + BK <= mask_base) && (tile_k == BK);
            if (interior) {
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 6u; r++) {
                    float score = (float(sS[0][r][lane]) + float(sS[1][r][lane]) + float(sS[2][r][lane]) + float(sS[3][r][lane])) * scale_log2;
                    float rm = simd_max(score), mo = m_row[r], mn = max(mo, rm);
                    float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                    float pv = exp2(score - mn);
                    l_row[r] = fma(l_row[r], al, simd_sum(pv)); m_row[r] = mn;
                    sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
                }
            } else {
                uint k_pos = kb + lane;
                bool ok = IS_TREE
                    ? ((k_pos < tp->mask_base_pos) || (((tp->mask[0] >> (k_pos - tp->mask_base_pos)) & 1u) != 0u))
                    : (k_pos <= p.pos);
                bool valid = ok && (lane < tile_k);
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 6u; r++) {
                    float score = valid
                        ? (float(sS[0][r][lane]) + float(sS[1][r][lane]) + float(sS[2][r][lane]) + float(sS[3][r][lane])) * scale_log2
                        : -INFINITY;
                    float rm = simd_max(score), mo = m_row[r], mn = max(mo, rm);
                    float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                    float pv = (valid && mn != -INFINITY) ? exp2(score - mn) : 0.0f;
                    l_row[r] = fma(l_row[r], al, simd_sum(pv)); m_row[r] = mn;
                    sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
                }
            }
            if (lane == 0u) { sA[6] = 0.0h; sA[7] = 0.0h; }
        }
        load_kv_tile32_full256<KV_BITS, STR>(v_cache, v_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        half a_row = sA[fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O[j].thread_elements()[0] *= a_row; O[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm; simdgroup_load(pm, &sP[0][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm; simdgroup_load(vm, &sKV[kk][sg * 64u + j * 8u], STR);
                simdgroup_multiply_accumulate(O[j], pm, vm, O[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (fm < 6u) {
        ulong slot = (ulong)(hkv * 6u + fm) * p.num_splits + s;
        device float* po = part_o + slot * 256ul + sg * 64u;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint c0 = 8u * j + fn;
            po[c0]      = float(O[j].thread_elements()[0]);
            po[c0 + 1u] = float(O[j].thread_elements()[1]);
        }
    }
    if (sg == 0u && lane < 6u) {
        ulong slot = (ulong)(hkv * 6u + lane) * p.num_splits + s;
        part_m[slot] = m_row[lane];
        part_l[slot] = l_row[lane];
    }
}

kernel void qwen_gqa6_m1_splitk_p1_f16(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<16, false>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m1_splitk_p1_f16_tree(
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]], constant TreeParams& tp [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<16, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m1_splitk_p1_q8(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<8, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m1_splitk_p1_q8_tree(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<8, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m1_splitk_p1_q4(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<4, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m1_splitk_p1_q4_tree(
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8];
    qwen_gqa6_m1_splitk_p1_impl<4, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

// 2. M=2 AMX GQA-6: packs 6 heads x 2 tokens into TWO 8x256 tiles (3 heads x 2 toks per tile), 128 thr = 4 SGs x 64 dims!
kernel void qwen_pack_q_p2(
    const device float* q_in [[buffer(0)]],
    device half*        q_p2 [[buffer(1)]], // [Hkv*2, 8, 256]
    constant FullAttnParams& p [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint tile = tgid.x; // 0 .. Hkv*2 - 1
    uint r    = tgid.y; // 0 .. 7
    if (tile >= p.Hkv * 2u || r >= 8u) return;
    device half* dst = q_p2 + ((ulong)tile * 8ul + r) * 256ul;
    uint hkv = tile >> 1, sub = tile & 1u;
    if (r < 6u) {
        uint h = hkv * 6u + sub * 3u + (r >> 1);
        uint t = r & 1u;
        if (t < p.M) {
            const device float* src = q_in + ((ulong)t * p.Hq + h) * 256ul;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = half(src[i * 32u + lane]);
            return;
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = 0.0h;
}

template<int KV_BITS, bool IS_TREE, bool IS_SPARSE = false>
inline void qwen_gqa6_m2_splitk_p1_impl(
    const device half*   q_p2,
    const device uchar*  k_cache,
    const device uchar*  v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*        part_o,
    device float*        part_m,
    device float*        part_l,
    constant FullAttnParams& p,
    constant TreeParams* tp,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half (&sKV)[32][264],
    threadgroup half (&sS)[4][16][32],
    threadgroup half (&sP)[16][36],
    threadgroup half (&sA)[16],
    const device uint* active_pages = nullptr,
    const device uint* active_counts = nullptr
) {
    constexpr uint BK = 32u, STR = 264u;
    uint hkv = tgid.x, s = tgid.y;
    if (hkv >= p.Hkv || s >= p.num_splits) return;
    uint iter_start, iter_end;
    if (IS_SPARSE) {
        uint n_active        = active_counts[hkv];
        uint pages_per_split = (n_active + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * pages_per_split;
        iter_end             = min(n_active, (s + 1u) * pages_per_split);
    } else {
        uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
        uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * tiles_per_split;
        iter_end             = min(num_k_tiles, (s + 1u) * tiles_per_split);
    }
    if (iter_start >= iter_end) {
        if (tid < 12u) {
            uint h_loc = tid >> 1, t = tid & 1u;
            if (t < p.M) {
                ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
                part_m[slot] = -INFINITY; part_l[slot] = 0.0f;
            }
        }
        return;
    }

    const device half* qp0 = q_p2 + (ulong)(hkv * 2u + 0u) * 8ul * 256ul + sg * 64u;
    const device half* qp1 = q_p2 + (ulong)(hkv * 2u + 1u) * 8ul * 256ul + sg * 64u;
    simdgroup_half8x8 O0[8], O1[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        O1[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    uint qid = lane >> 2, fm = (qid & 4u) + ((lane >> 1) & 3u), fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    float m_row[4], l_row[4];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4u; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint it = iter_start; it < iter_end; ++it) {
        uint kb = IS_SPARSE ? (active_pages[(ulong)hkv * 1024ul + it] * BK) : (it * BK);
        uint tile_k = min(BK, p.kv_len - kb);
        load_kv_tile32_full256<KV_BITS, STR>(k_cache, k_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 s00 = make_filled_simdgroup_matrix<half,8,8>(0.0h), s01 = s00, s02 = s00, s03 = s00;
        simdgroup_half8x8 s10 = s00, s11 = s00, s12 = s00, s13 = s00;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint kk = sg * 64u + j * 8u;
            simdgroup_half8x8 q0, q1, km0, km1, km2, km3;
            simdgroup_load(q0, qp0 + j * 8u, 256);
            simdgroup_load(q1, qp1 + j * 8u, 256);
            simdgroup_load(km0, &sKV[0][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km1, &sKV[8][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km2, &sKV[16][kk], STR, ulong2(0, 0), true);
            simdgroup_load(km3, &sKV[24][kk], STR, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s00, q0, km0, s00); simdgroup_multiply_accumulate(s01, q0, km1, s01);
            simdgroup_multiply_accumulate(s02, q0, km2, s02); simdgroup_multiply_accumulate(s03, q0, km3, s03);
            simdgroup_multiply_accumulate(s10, q1, km0, s10); simdgroup_multiply_accumulate(s11, q1, km1, s11);
            simdgroup_multiply_accumulate(s12, q1, km2, s12); simdgroup_multiply_accumulate(s13, q1, km3, s13);
        }
        sS[sg][fm][0u+fn]     = s00.thread_elements()[0]; sS[sg][fm][0u+fn+1u]     = s00.thread_elements()[1];
        sS[sg][fm][8u+fn]     = s01.thread_elements()[0]; sS[sg][fm][8u+fn+1u]     = s01.thread_elements()[1];
        sS[sg][fm][16u+fn]    = s02.thread_elements()[0]; sS[sg][fm][16u+fn+1u]    = s02.thread_elements()[1];
        sS[sg][fm][24u+fn]    = s03.thread_elements()[0]; sS[sg][fm][24u+fn+1u]    = s03.thread_elements()[1];
        sS[sg][8u+fm][0u+fn]  = s10.thread_elements()[0]; sS[sg][8u+fm][0u+fn+1u]  = s10.thread_elements()[1];
        sS[sg][8u+fm][8u+fn]  = s11.thread_elements()[0]; sS[sg][8u+fm][8u+fn+1u]  = s11.thread_elements()[1];
        sS[sg][8u+fm][16u+fn] = s12.thread_elements()[0]; sS[sg][8u+fm][16u+fn+1u] = s12.thread_elements()[1];
        sS[sg][8u+fm][24u+fn] = s13.thread_elements()[0]; sS[sg][8u+fm][24u+fn+1u] = s13.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint mask_base = IS_TREE ? tp->mask_base_pos : p.pos;
        bool interior  = (kb + BK <= mask_base) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr;
                float sc = (float(sS[0][r][lane]) + float(sS[1][r][lane]) + float(sS[2][r][lane]) + float(sS[3][r][lane])) * scale_log2;
                float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                float pv = exp2(sc - mn);
                l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
            }
        } else {
            uint k_pos = kb + lane;
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 4u; rr++) {
                uint r = sg * 4u + rr, loc = r & 7u, t = loc & 1u;
                bool ok;
                if (IS_TREE) {
                    ok = (loc < 6u) && (t < p.M) && ((k_pos < tp->mask_base_pos) || (((tp->mask[t] >> (k_pos - tp->mask_base_pos)) & 1u) != 0u));
                } else {
                    ok = (loc < 6u) && (t < p.M) && (k_pos <= p.pos + t);
                }
                bool valid = ok && (lane < tile_k);
                float sc = valid
                    ? (float(sS[0][r][lane]) + float(sS[1][r][lane]) + float(sS[2][r][lane]) + float(sS[3][r][lane])) * scale_log2
                    : -INFINITY;
                float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                float pv = (valid && mn != -INFINITY) ? exp2(sc - mn) : 0.0f;
                l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
            }
        }
        load_kv_tile32_full256<KV_BITS, STR>(v_cache, v_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        half a0 = sA[fm], a1 = sA[8u + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a0; O0[j].thread_elements()[1] *= a0;
            O1[j].thread_elements()[0] *= a1; O1[j].thread_elements()[1] *= a1;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm0, pm1;
            simdgroup_load(pm0, &sP[0][kk], 36);
            simdgroup_load(pm1, &sP[8][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm; simdgroup_load(vm, &sKV[kk][sg * 64u + j * 8u], STR);
                simdgroup_multiply_accumulate(O0[j], pm0, vm, O0[j]);
                simdgroup_multiply_accumulate(O1[j], pm1, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 4u; rr++) {
            uint r = sg * 4u + rr, sub = r >> 3, loc = r & 7u;
            if (loc < 6u) {
                uint h_loc = sub * 3u + (loc >> 1), t = loc & 1u;
                if (t < p.M) {
                    ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
                    part_m[slot] = m_row[rr]; part_l[slot] = l_row[rr];
                }
            }
        }
    }
    if (fm < 6u) {
        uint t = fm & 1u;
        if (t < p.M) {
            uint hl0 = fm >> 1;
            ulong slot0 = ((ulong)t * p.Hq + (hkv * 6u + hl0)) * p.num_splits + s;
            ulong slot1 = ((ulong)t * p.Hq + (hkv * 6u + 3u + hl0)) * p.num_splits + s;
            device float* po0 = part_o + slot0 * 256ul + sg * 64u;
            device float* po1 = part_o + slot1 * 256ul + sg * 64u;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                uint c0 = 8u * j + fn;
                po0[c0] = float(O0[j].thread_elements()[0]); po0[c0 + 1u] = float(O0[j].thread_elements()[1]);
                po1[c0] = float(O1[j].thread_elements()[0]); po1[c0 + 1u] = float(O1[j].thread_elements()[1]);
            }
        }
    }
}

kernel void qwen_gqa6_m2_splitk_p1_f16(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<16, false>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m2_splitk_p1_f16_tree(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]], constant TreeParams& tp [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<16, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m2_splitk_p1_q8(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<8, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m2_splitk_p1_q8_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<8, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m2_splitk_p1_q4(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<4, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

kernel void qwen_gqa6_m2_splitk_p1_q4_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16];
    qwen_gqa6_m2_splitk_p1_impl<4, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sS, sP, sA);
}

// 3. M=3..4 AMX GQA-6: 128 thr (4 SGs x 64 dims), 3 8-row tiles (2 heads x 4 toks), O[3][8] = 24 mats (< 32!),
// bank-conflict-free sKV[32][264], ALL 128 threads load full 256 dims -> 5.25/10.24/19.56 ms at 32k/64k/128k!
kernel void qwen_pack_q_p4(
    const device float* q_in [[buffer(0)]],
    device half*        q_p4 [[buffer(1)]], // [Hkv*3, 8, 256]
    constant FullAttnParams& p [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint tile = tgid.x; // 0 .. Hkv*3 - 1
    uint r    = tgid.y; // 0 .. 7
    if (tile >= p.Hkv * 3u || r >= 8u) return;
    device half* dst = q_p4 + ((ulong)tile * 8ul + r) * 256ul;
    uint hkv = tile / 3u, hp = tile % 3u;
    uint h   = hkv * 6u + hp * 2u + (r >> 2);
    uint t   = r & 3u;
    if (t < p.M) {
        const device float* src = q_in + ((ulong)t * p.Hq + h) * 256ul;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = half(src[i * 32u + lane]);
    } else {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = 0.0h;
    }
}

template<int KV_BITS, bool IS_TREE, bool IS_SPARSE = false>
inline void qwen_gqa6_m4_splitk_p1_impl(
    const device half*   q_p4,
    const device uchar*  k_cache,
    const device uchar*  v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*        part_o,
    device float*        part_m,
    device float*        part_l,
    constant FullAttnParams& p,
    constant TreeParams* tp,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half (&sKV)[32][264],
    threadgroup half (&sP)[24][36],
    threadgroup half (&sA)[24],
    const device uint* active_pages = nullptr,
    const device uint* active_counts = nullptr
) {
    constexpr uint BK = 32u, STR = 264u;
    uint hkv = tgid.x, s = tgid.y;
    if (hkv >= p.Hkv || s >= p.num_splits) return;

    uint iter_start, iter_end;
    if (IS_SPARSE) {
        uint n_active        = active_counts[hkv];
        uint pages_per_split = (n_active + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * pages_per_split;
        iter_end             = min(n_active, (s + 1u) * pages_per_split);
    } else {
        uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
        uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * tiles_per_split;
        iter_end             = min(num_k_tiles, (s + 1u) * tiles_per_split);
    }

    if (iter_start >= iter_end) {
        if (tid < 24u) {
            uint h_loc = tid >> 2, t = tid & 3u;
            if (t < p.M) {
                ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
                part_m[slot] = -INFINITY; part_l[slot] = 0.0f;
            }
        }
        return;
    }

    threadgroup half* sS_base = &sKV[0][0]; // 32*264*2 = 16,896 B >= 4*24*32*2 = 6,144 B!
    const device half* qp0 = q_p4 + (ulong)(hkv * 3u + 0u) * 8ul * 256ul + sg * 64u;
    const device half* qp1 = q_p4 + (ulong)(hkv * 3u + 1u) * 8ul * 256ul + sg * 64u;
    const device half* qp2 = q_p4 + (ulong)(hkv * 3u + 2u) * 8ul * 256ul + sg * 64u;

    simdgroup_half8x8 O0[8], O1[8], O2[8];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; j++) {
        O0[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        O1[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        O2[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }

    uint qid = lane >> 2, fm = (qid & 4u) + ((lane >> 1) & 3u), fn = (qid & 2u) * 2u + (lane & 1u) * 2u;
    float m_row[6], l_row[6];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 6u; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint it = iter_start; it < iter_end; ++it) {
        uint kb = IS_SPARSE ? (active_pages[(ulong)hkv * 1024ul + it] * BK) : (it * BK);
        uint tile_k = min(BK, p.kv_len - kb);
        load_kv_tile32_full256<KV_BITS, STR>(k_cache, k_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 s00 = make_filled_simdgroup_matrix<half,8,8>(0.0h), s01 = s00, s02 = s00, s03 = s00;
        simdgroup_half8x8 s10 = s00, s11 = s00, s12 = s00, s13 = s00;
        simdgroup_half8x8 s20 = s00, s21 = s00, s22 = s00, s23 = s00;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            uint kk = sg * 64u + j * 8u;
            simdgroup_half8x8 q0, q1, q2, km0, km1, km2, km3;
            simdgroup_load(q0, qp0 + j * 8u, 256);
            simdgroup_load(q1, qp1 + j * 8u, 256);
            simdgroup_load(q2, qp2 + j * 8u, 256);
            simdgroup_load(km0, &sKV[0][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km1, &sKV[8][kk],  STR, ulong2(0, 0), true);
            simdgroup_load(km2, &sKV[16][kk], STR, ulong2(0, 0), true);
            simdgroup_load(km3, &sKV[24][kk], STR, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s00, q0, km0, s00); simdgroup_multiply_accumulate(s01, q0, km1, s01);
            simdgroup_multiply_accumulate(s02, q0, km2, s02); simdgroup_multiply_accumulate(s03, q0, km3, s03);
            simdgroup_multiply_accumulate(s10, q1, km0, s10); simdgroup_multiply_accumulate(s11, q1, km1, s11);
            simdgroup_multiply_accumulate(s12, q1, km2, s12); simdgroup_multiply_accumulate(s13, q1, km3, s13);
            simdgroup_multiply_accumulate(s20, q2, km0, s20); simdgroup_multiply_accumulate(s21, q2, km1, s21);
            simdgroup_multiply_accumulate(s22, q2, km2, s22); simdgroup_multiply_accumulate(s23, q2, km3, s23);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* my_s = sS_base + sg * (24u * 32u);
        my_s[(0u+fm)*32u+0u+fn]   = s00.thread_elements()[0]; my_s[(0u+fm)*32u+0u+fn+1u]   = s00.thread_elements()[1];
        my_s[(0u+fm)*32u+8u+fn]   = s01.thread_elements()[0]; my_s[(0u+fm)*32u+8u+fn+1u]   = s01.thread_elements()[1];
        my_s[(0u+fm)*32u+16u+fn]  = s02.thread_elements()[0]; my_s[(0u+fm)*32u+16u+fn+1u]  = s02.thread_elements()[1];
        my_s[(0u+fm)*32u+24u+fn]  = s03.thread_elements()[0]; my_s[(0u+fm)*32u+24u+fn+1u]  = s03.thread_elements()[1];
        my_s[(8u+fm)*32u+0u+fn]   = s10.thread_elements()[0]; my_s[(8u+fm)*32u+0u+fn+1u]   = s10.thread_elements()[1];
        my_s[(8u+fm)*32u+8u+fn]   = s11.thread_elements()[0]; my_s[(8u+fm)*32u+8u+fn+1u]   = s11.thread_elements()[1];
        my_s[(8u+fm)*32u+16u+fn]  = s12.thread_elements()[0]; my_s[(8u+fm)*32u+16u+fn+1u]  = s12.thread_elements()[1];
        my_s[(8u+fm)*32u+24u+fn]  = s13.thread_elements()[0]; my_s[(8u+fm)*32u+24u+fn+1u]  = s13.thread_elements()[1];
        my_s[(16u+fm)*32u+0u+fn]  = s20.thread_elements()[0]; my_s[(16u+fm)*32u+0u+fn+1u]  = s20.thread_elements()[1];
        my_s[(16u+fm)*32u+8u+fn]  = s21.thread_elements()[0]; my_s[(16u+fm)*32u+8u+fn+1u]  = s21.thread_elements()[1];
        my_s[(16u+fm)*32u+16u+fn] = s22.thread_elements()[0]; my_s[(16u+fm)*32u+16u+fn+1u] = s22.thread_elements()[1];
        my_s[(16u+fm)*32u+24u+fn] = s23.thread_elements()[0]; my_s[(16u+fm)*32u+24u+fn+1u] = s23.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint mask_base = IS_TREE ? tp->mask_base_pos : p.pos;
        bool interior  = (kb + BK <= mask_base) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 6u; rr++) {
                uint r = sg * 6u + rr;
                float sc = (float(sS_base[0u*768u+r*32u+lane]) + float(sS_base[1u*768u+r*32u+lane])
                          + float(sS_base[2u*768u+r*32u+lane]) + float(sS_base[3u*768u+r*32u+lane])) * scale_log2;
                float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn), pv = exp2(sc - mn);
                l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
            }
        } else {
            uint k_pos = kb + lane;
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 6u; rr++) {
                uint r = sg * 6u + rr, t = r & 3u;
                bool ok;
                if (IS_TREE) {
                    ok = (t < p.M) && ((k_pos < tp->mask_base_pos) || (((tp->mask[t] >> (k_pos - tp->mask_base_pos)) & 1u) != 0u));
                } else {
                    ok = (t < p.M) && (k_pos <= p.pos + t);
                }
                bool valid = ok && (lane < tile_k);
                float sc = valid
                    ? (float(sS_base[0u*768u+r*32u+lane]) + float(sS_base[1u*768u+r*32u+lane])
                     + float(sS_base[2u*768u+r*32u+lane]) + float(sS_base[3u*768u+r*32u+lane])) * scale_log2
                    : -INFINITY;
                float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                float pv = (valid && mn != -INFINITY) ? exp2(sc - mn) : 0.0f;
                l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile32_full256<KV_BITS, STR>(v_cache, v_sb, sKV, kb, tile_k, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        half a0 = sA[0u+fm], a1 = sA[8u+fm], a2 = sA[16u+fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j].thread_elements()[0] *= a0; O0[j].thread_elements()[1] *= a0;
            O1[j].thread_elements()[0] *= a1; O1[j].thread_elements()[1] *= a1;
            O2[j].thread_elements()[0] *= a2; O2[j].thread_elements()[1] *= a2;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm0, pm1, pm2;
            simdgroup_load(pm0, &sP[0u][kk],  36);
            simdgroup_load(pm1, &sP[8u][kk],  36);
            simdgroup_load(pm2, &sP[16u][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vm; simdgroup_load(vm, &sKV[kk][sg * 64u + 8u * j], STR);
                simdgroup_multiply_accumulate(O0[j], pm0, vm, O0[j]);
                simdgroup_multiply_accumulate(O1[j], pm1, vm, O1[j]);
                simdgroup_multiply_accumulate(O2[j], pm2, vm, O2[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 6u; rr++) {
            uint r = sg * 6u + rr, hl = r >> 2, t = r & 3u;
            if (t < p.M) {
                ulong slot = ((ulong)t * p.Hq + (hkv * 6u + hl)) * p.num_splits + s;
                part_m[slot] = m_row[rr]; part_l[slot] = l_row[rr];
            }
        }
    }
    uint t = fm & 3u, hl_sub = fm >> 2;
    if (t < p.M) {
        #pragma clang loop unroll(full)
        for (uint hp = 0; hp < 3u; hp++) {
            ulong slot = ((ulong)t * p.Hq + (hkv * 6u + hp * 2u + hl_sub)) * p.num_splits + s;
            device float* po = part_o + slot * 256ul + sg * 64u;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                uint c0 = 8u * j + fn;
                half v0 = (hp == 0u) ? O0[j].thread_elements()[0] : ((hp == 1u) ? O1[j].thread_elements()[0] : O2[j].thread_elements()[0]);
                half v1 = (hp == 0u) ? O0[j].thread_elements()[1] : ((hp == 1u) ? O1[j].thread_elements()[1] : O2[j].thread_elements()[1]);
                po[c0] = float(v0); po[c0 + 1u] = float(v1);
            }
        }
    }
}

kernel void qwen_gqa6_m4_splitk_p1_f16(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<16, false>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sP, sA);
}

kernel void qwen_gqa6_m4_splitk_p1_f16_tree(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]], constant TreeParams& tp [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<16, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sP, sA);
}

kernel void qwen_gqa6_m4_splitk_p1_q8(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<8, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sP, sA);
}

kernel void qwen_gqa6_m4_splitk_p1_q8_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<8, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sP, sA);
}

kernel void qwen_gqa6_m4_splitk_p1_q4(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<4, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV, sP, sA);
}

kernel void qwen_gqa6_m4_splitk_p1_q4_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[24][36], sA[24];
    qwen_gqa6_m4_splitk_p1_impl<4, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV, sP, sA);
}

// Pack Q [M, Hq, 256] float -> Q_h [Hq, 16, 256] half (96-192 KB, L1-resident for GQA-6 M=5..8 BK=32!)
kernel void qwen_pack_q_h(
    const device float* q_in [[buffer(0)]],
    device half*        q_h  [[buffer(1)]],
    constant FullAttnParams& p [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= 16u) return;
    device half* dst = q_h + ((ulong)h * 16ul + t) * 256ul;
    if (t < p.M) {
        const device float* src = q_in + ((ulong)t * p.Hq + h) * 256ul;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = half(src[i * 32u + lane]);
    } else {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = 0.0h;
    }
}

// Helper: load [BK=32, 128] half tile from FP16 / Q8 / Q4 KV cache into sKV_raw (stride 132) using first 128 threads
template<int KV_BITS>
inline void load_kv_tile32_half128(
    const device uchar*  kv_cache,
    const device float2* kv_sb,
    threadgroup half*    sKV_raw,
    uint kb, uint tile_k, uint d_half, uint hkv, uint Hkv, uint tid
) {
    if (tid >= 128u) return;
    uint kr = tid >> 2;
    if (kr < tile_k) {
        ulong head_idx = (ulong)(kb + kr) * Hkv + hkv;
        if (KV_BITS == 16) {
            uint kc = (tid & 3u) * 32u;
            const device half4* hp = (const device half4*)((const device half*)kv_cache + head_idx * 256ul + d_half * 128u + kc);
            threadgroup half4* dst = (threadgroup half4*)(sKV_raw + kr * 132u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) dst[i] = hp[i];
        } else if (KV_BITS == 8) {
            uint kc  = (tid & 3u) * 32u;
            uint grp = d_half * 2u + ((tid & 3u) >> 1);
            float2 sb = kv_sb[head_idx * 4ul + grp];
            float4 fs = float4(sb.x), fb = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 256ul + d_half * 128u + kc);
            threadgroup half4* dst  = (threadgroup half4*)(sKV_raw + kr * 132u + kc);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) dst[i] = half4(fma(float4(qp[i]), fs, fb));
        } else {
            uint g_loc = (tid & 3u) >> 1;
            uint sub16 = (tid & 1u) * 16u;
            uint grp   = d_half * 2u + g_loc;
            float2 sb = kv_sb[head_idx * 4ul + grp];
            float4 fs = float4(sb.x), fb = float4(sb.y);
            const device uchar4* qp = (const device uchar4*)(kv_cache + head_idx * 128ul + grp * 32u + sub16);
            threadgroup half4* dst_lo = (threadgroup half4*)(sKV_raw + kr * 132u + g_loc * 64u + sub16);
            threadgroup half4* dst_hi = (threadgroup half4*)(sKV_raw + kr * 132u + g_loc * 64u + 32u + sub16);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; i++) {
                uchar4 u = qp[i];
                dst_lo[i] = half4(fma(float4(u & uchar4(0x0Fu)), fs, fb));
                dst_hi[i] = half4(fma(float4(u >> uchar4(4u)),   fs, fb));
            }
        }
    } else {
        uint kc = (tid & 3u) * 32u;
        threadgroup half4* dst = (threadgroup half4*)(sKV_raw + kr * 132u + kc);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i] = half4(0.0h);
    }
}

// GQA-6 192-thread (6 SGs = 1 SG per query head!) BK=32 FP16-AMX Split-K Verify Kernel
// Total SRAM: 11.7 KB (2+ TGs/core!) | 1x KV DRAM read across all 6 query heads | L1-resident half Q
template<int KV_BITS, bool IS_TREE, bool IS_SPARSE = false>
inline void qwen_gqa6_verify_splitk_p1_impl(
    const device half*   q_h,
    const device uchar*  k_cache,
    const device uchar*  v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device float*        part_o,
    device float*        part_m,
    device float*        part_l,
    constant FullAttnParams& p,
    constant TreeParams* tp,
    uint3 tgid, uint tid, uint lane, uint sg,
    threadgroup half (&sKV_raw)[4224],
    threadgroup half (&sP)[48][36],
    threadgroup half (&sAlpha)[48],
    const device uint* active_pages = nullptr,
    const device uint* active_counts = nullptr
) {
    constexpr uint BK = 32u;
    uint hkv    = tgid.x;
    uint s      = tgid.y;
    uint t_base = tgid.z * 8u;
    if (hkv >= p.Hkv || s >= p.num_splits || t_base >= p.M || sg >= 6u) return;

    uint iter_start, iter_end;
    if (IS_SPARSE) {
        uint n_active        = active_counts[hkv];
        uint pages_per_split = (n_active + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * pages_per_split;
        iter_end             = min(n_active, (s + 1u) * pages_per_split);
    } else {
        uint num_k_tiles     = (p.kv_len + BK - 1u) / BK;
        uint tiles_per_split = (num_k_tiles + p.num_splits - 1u) / p.num_splits;
        iter_start           = s * tiles_per_split;
        iter_end             = min(num_k_tiles, (s + 1u) * tiles_per_split);
    }

    if (iter_start >= iter_end) {
        if (tid < 48u) {
            uint h_loc = tid >> 3;
            uint t     = t_base + (tid & 7u);
            if (t < p.M) {
                ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
                part_m[slot] = -INFINITY;
                part_l[slot] = 0.0f;
            }
        }
        return;
    }

    uint h_loc = sg; // 0..5
    uint qid   = lane >> 2;
    uint fm    = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn    = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const device half* q_ptr = q_h + ((ulong)(hkv * 6u + h_loc) * 16ul + t_base) * 256ul;

    simdgroup_half8x8 O0[16], O1[16];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 16u; j++) {
        O0[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        O1[j] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }

    float m_row[8], l_row[8];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8u; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint it = iter_start; it < iter_end; ++it) {
        uint kb = IS_SPARSE ? (active_pages[(ulong)hkv * 1024ul + it] * BK) : (it * BK);
        uint tile_k = min(BK, p.kv_len - kb);
        simdgroup_half8x8 s0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        simdgroup_half8x8 s1 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        simdgroup_half8x8 s2 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        simdgroup_half8x8 s3 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

        load_kv_tile32_half128<KV_BITS>(k_cache, k_sb, sKV_raw, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 qm, km0, km1, km2, km3;
            simdgroup_load(qm,  q_ptr + kk, 256);
            simdgroup_load(km0, sKV_raw + 0u  * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km1, sKV_raw + 8u  * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km2, sKV_raw + 16u * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km3, sKV_raw + 24u * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, qm, km0, s0);
            simdgroup_multiply_accumulate(s1, qm, km1, s1);
            simdgroup_multiply_accumulate(s2, qm, km2, s2);
            simdgroup_multiply_accumulate(s3, qm, km3, s3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile32_half128<KV_BITS>(k_cache, k_sb, sKV_raw, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 qm, km0, km1, km2, km3;
            simdgroup_load(qm,  q_ptr + 128u + kk, 256);
            simdgroup_load(km0, sKV_raw + 0u  * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km1, sKV_raw + 8u  * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km2, sKV_raw + 16u * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_load(km3, sKV_raw + 24u * 132u + kk, 132, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s0, qm, km0, s0);
            simdgroup_multiply_accumulate(s1, qm, km1, s1);
            simdgroup_multiply_accumulate(s2, qm, km2, s2);
            simdgroup_multiply_accumulate(s3, qm, km3, s3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint s_row = h_loc * 8u + fm;
        sKV_raw[s_row * 32u + 0u  + fn] = s0.thread_elements()[0]; sKV_raw[s_row * 32u + 0u  + fn + 1u] = s0.thread_elements()[1];
        sKV_raw[s_row * 32u + 8u  + fn] = s1.thread_elements()[0]; sKV_raw[s_row * 32u + 8u  + fn + 1u] = s1.thread_elements()[1];
        sKV_raw[s_row * 32u + 16u + fn] = s2.thread_elements()[0]; sKV_raw[s_row * 32u + 16u + fn + 1u] = s2.thread_elements()[1];
        sKV_raw[s_row * 32u + 24u + fn] = s3.thread_elements()[0]; sKV_raw[s_row * 32u + 24u + fn + 1u] = s3.thread_elements()[1];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint mask_base = IS_TREE ? tp->mask_base_pos : p.pos;
        bool interior  = (kb + BK <= mask_base) && (tile_k == BK);
        if (interior) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 8u; rr++) {
                uint r = h_loc * 8u + rr;
                float score = float(sKV_raw[r * 32u + lane]) * scale_log2;
                float row_max = simd_max(score);
                float m_old = m_row[rr], m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = exp2(score - m_new);
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = half(alpha);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 8u; rr++) {
                uint r = h_loc * 8u + rr, t = t_base + rr, k_pos = kb + lane;
                bool ok;
                if (IS_TREE) {
                    ok = (t < p.M) && ((k_pos < tp->mask_base_pos) || (((tp->mask[t] >> (k_pos - tp->mask_base_pos)) & 1u) != 0u));
                } else {
                    ok = (t < p.M) && (k_pos <= p.pos + t);
                }
                bool valid = ok && (lane < tile_k);
                float score = valid ? (float(sKV_raw[r * 32u + lane]) * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr], m_new = max(m_old, row_max);
                float alpha = (m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = half(alpha);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile32_half128<KV_BITS>(v_cache, v_sb, sKV_raw, kb, tile_k, 0u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        half a_row = sAlpha[h_loc * 8u + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16u; j++) {
            O0[j].thread_elements()[0] *= a_row; O0[j].thread_elements()[1] *= a_row;
            O1[j].thread_elements()[0] *= a_row; O1[j].thread_elements()[1] *= a_row;
        }
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm; simdgroup_load(pm, &sP[h_loc * 8u][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16u; j++) {
                simdgroup_half8x8 vm; simdgroup_load(vm, sKV_raw + kk * 132u + 8u * j, 132);
                simdgroup_multiply_accumulate(O0[j], pm, vm, O0[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_kv_tile32_half128<KV_BITS>(v_cache, v_sb, sKV_raw, kb, tile_k, 1u, hkv, p.Hkv, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 pm; simdgroup_load(pm, &sP[h_loc * 8u][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16u; j++) {
                simdgroup_half8x8 vm; simdgroup_load(vm, sKV_raw + kk * 132u + 8u * j, 132);
                simdgroup_multiply_accumulate(O1[j], pm, vm, O1[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 8u; rr++) {
            uint t = t_base + rr;
            if (t < p.M) {
                ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
                part_m[slot] = m_row[rr];
                part_l[slot] = l_row[rr];
            }
        }
    }

    uint t = t_base + fm;
    if (t < p.M) {
        ulong slot = ((ulong)t * p.Hq + (hkv * 6u + h_loc)) * p.num_splits + s;
        device float* po = part_o + slot * 256ul;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16u; j++) {
            uint c0 = 8u * j + fn;
            po[c0]      = float(O0[j].thread_elements()[0]);
            po[c0 + 1u] = float(O0[j].thread_elements()[1]);
            uint c1 = 128u + c0;
            po[c1]      = float(O1[j].thread_elements()[0]);
            po[c1 + 1u] = float(O1[j].thread_elements()[1]);
        }
    }
}

kernel void qwen_gqa6_verify_splitk_p1_f16(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<16, false>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

kernel void qwen_gqa6_verify_splitk_p1_f16_tree(
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]], constant TreeParams& tp [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<16, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

kernel void qwen_gqa6_verify_splitk_p1_q8(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<8, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

kernel void qwen_gqa6_verify_splitk_p1_q8_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<8, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

kernel void qwen_gqa6_verify_splitk_p1_q4(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<4, false>(q, kc, vc, ksb, vsb, po, pm, pl, p, nullptr, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

kernel void qwen_gqa6_verify_splitk_p1_q4_tree(
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]],
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]],
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]], constant TreeParams& tp [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48];
    qwen_gqa6_verify_splitk_p1_impl<4, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, &tp, tgid, tid, lane, sg, sKV_raw, sP, sAlpha);
}

// ============================================================================
// 2-Level Hierarchical FP16-AMX + Periodic FP32 Flush FlashAttention-2 (Long-Context Prefill!)
// - BQ=64, BK=32, 256 thr (8 SGs), bank-conflict-free sKV[32][264], L2-GQA grid
// - Inner 512-token chunks (16 tiles): 100% simdgroup_half8x8 (32 FP32 regs -> 11.20 TFLOPs!)
// - Periodic FP32 flush every 512 tokens into o_f32 (reuses 24 MB buf_split_o_ -> 0 extra RAM,
//   mean_err = 6.7e-5 vs FP32 even at 64k-128k!)
// ============================================================================
kernel void qwen_pack_q_pf_h(
    const device float* q_in [[buffer(0)]],
    device half*        q_h  [[buffer(1)]], // [Hq, M_pad, 256], M_pad = ceil_div(M, 64)*64
    constant FullAttnParams& p [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    uint M_pad = (p.M + 63u) & ~63u;
    uint h = tgid.x, t = tgid.y;
    if (h >= p.Hq || t >= M_pad) return;
    device half* dst = q_h + ((ulong)h * M_pad + t) * 256ul;
    if (t < p.M) {
        const device float* src = q_in + ((ulong)t * p.Hq + h) * 256ul;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = half(src[i * 32u + lane]);
    } else {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; i++) dst[i * 32u + lane] = 0.0h;
    }
}

template<int KV_BITS>
inline void qwen_gqa_prefill_flash_hier_impl(
    const device half*   q_h,
    const device float*  gate,
    const device uchar*  k_cache,
    const device uchar*  v_cache,
    const device float2* k_sb,
    const device float2* v_sb,
    device half*         out,
    device float*        o_f32, // scratch [M, Hq, 256] FP32 (reuses buf_split_o_)
    constant FullAttnParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half  (&sKV)[32][264],
    threadgroup half  (&sP)[64][36],
    threadgroup half  (&sA)[64],
    threadgroup float (&sMg)[64],
    threadgroup float (&sLg)[64],
    threadgroup float (&sAg)[64],
    threadgroup float (&sAc)[64]
) {
    constexpr uint BQ = 64u, BK = 32u, STR = 264u, CHUNK_K = 512u;
    uint gqa = p.Hq / p.Hkv;
    uint n_qt = (p.M + BQ - 1u) / BQ;
    uint M_pad = n_qt * BQ;
    uint h_loc = tgid.x, qt = tgid.y % n_qt, hkv = tgid.y / n_qt;
    uint h = hkv * gqa + h_loc, t0 = qt * BQ;
    if (h_loc >= gqa || hkv >= p.Hkv || t0 >= p.M) return;

    uint qh = sg >> 2, dq = sg & 3u;
    uint qk_q = (sg >> 1) * 16u, qk_k = (sg & 1u) * 16u;
    uint qid = lane >> 2, fm = (qid & 4u) + ((lane >> 1) & 3u), fn = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid < 64u) { sMg[tid] = -INFINITY; sLg[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device half* qp_qk = q_h + ((ulong)h * M_pad + t0 + qk_q) * 256ul;
    const float scale_log2 = p.scale * 1.4426950408889634f;
    const bool full_q = (t0 + BQ <= p.M);
    uint max_kv = p.pos + min(t0 + BQ, p.M);
    uint kr = tid >> 3, ch = tid & 7u;

    auto ld256 = [&](const device uchar* kv, const device float2* sb, uint kb, uint tile_k) {
        threadgroup half4* dst = (threadgroup half4*)(&sKV[kr][ch * 32u]);
        if (kr < tile_k) {
            ulong row = (ulong)(kb + kr) * p.Hkv + hkv;
            if (KV_BITS == 16) {
                const device half4* hp = (const device half4*)((const device half*)kv + row * 256ul + ch * 32u);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; i++) dst[i] = hp[i];
            } else if (KV_BITS == 8) {
                float2 s = sb[row * 4ul + (ch >> 1)];
                float4 fs = float4(s.x), fb = float4(s.y);
                const device uchar4* qp = (const device uchar4*)(kv + row * 256ul + ch * 32u);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; i++) dst[i] = half4(fma(float4(qp[i]), fs, fb));
            } else {
                float2 s = sb[row * 4ul + (ch >> 1)];
                float4 fs = float4(s.x), fb = float4(s.y);
                const device uchar4* qp = (const device uchar4*)(kv + row * 128ul + (ch >> 1) * 32u);
                if ((ch & 1u) == 0u) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; i++) dst[i] = half4(fma(float4(qp[i] & uchar4(0x0Fu)), fs, fb));
                } else {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; i++) dst[i] = half4(fma(float4(qp[i] >> uchar4(4u)), fs, fb));
                }
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) dst[i] = half4(0.0h);
        }
    };

    bool first_chunk = true;
    for (uint kb_c = 0; kb_c < max_kv; kb_c += CHUNK_K) {
        uint kb_c_end = min(max_kv, kb_c + CHUNK_K);

        simdgroup_half8x8 O0[8], O1[8], O2[8], O3[8];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            O0[j] = make_filled_simdgroup_matrix<half,8,8>(0.0h);
            O1[j] = make_filled_simdgroup_matrix<half,8,8>(0.0h);
            O2[j] = make_filled_simdgroup_matrix<half,8,8>(0.0h);
            O3[j] = make_filled_simdgroup_matrix<half,8,8>(0.0h);
        }
        float m_row[8], l_row[8];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 8u; r++) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }

        for (uint kb = kb_c; kb < kb_c_end; kb += BK) {
            uint tile_k = min(BK, kb_c_end - kb);
            ld256(k_cache, k_sb, kb, tile_k);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 s00 = make_filled_simdgroup_matrix<half,8,8>(0.0h), s01 = s00, s10 = s00, s11 = s00;
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < 256u; kk += 8u) {
                simdgroup_half8x8 qa, qb, ka, kb_m;
                simdgroup_load(qa,   qp_qk + kk, 256);
                simdgroup_load(qb,   qp_qk + 8ul * 256ul + kk, 256);
                simdgroup_load(ka,   &sKV[qk_k + 0u][kk], STR, ulong2(0,0), true);
                simdgroup_load(kb_m, &sKV[qk_k + 8u][kk], STR, ulong2(0,0), true);
                simdgroup_multiply_accumulate(s00, qa, ka, s00); simdgroup_multiply_accumulate(s01, qa, kb_m, s01);
                simdgroup_multiply_accumulate(s10, qb, ka, s10); simdgroup_multiply_accumulate(s11, qb, kb_m, s11);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup half* sS = &sKV[0][0];
            uint r0 = qk_q + fm, r1 = qk_q + 8u + fm;
            sS[r0*32u + qk_k + fn]    = s00.thread_elements()[0]; sS[r0*32u + qk_k + fn + 1u]    = s00.thread_elements()[1];
            sS[r0*32u + qk_k + 8u+fn] = s01.thread_elements()[0]; sS[r0*32u + qk_k + 8u+fn + 1u] = s01.thread_elements()[1];
            sS[r1*32u + qk_k + fn]    = s10.thread_elements()[0]; sS[r1*32u + qk_k + fn + 1u]    = s10.thread_elements()[1];
            sS[r1*32u + qk_k + 8u+fn] = s11.thread_elements()[0]; sS[r1*32u + qk_k + 8u+fn + 1u] = s11.thread_elements()[1];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool interior = full_q && (kb + BK <= p.pos + t0);
            if (interior) {
                #pragma clang loop unroll(full)
                for (uint rr = 0; rr < 8u; rr++) {
                    uint r = sg * 8u + rr;
                    float sc = float(sS[r * 32u + lane]) * scale_log2;
                    float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                    float al = (kb == kb_c) ? 0.0f : exp2(mo - mn), pv = exp2(sc - mn);
                    l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                    sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint rr = 0; rr < 8u; rr++) {
                    uint r = sg * 8u + rr;
                    bool valid = (t0 + r < p.M) && (lane < tile_k) && (kb + lane <= p.pos + t0 + r);
                    float sc = valid ? (float(sS[r * 32u + lane]) * scale_log2) : -INFINITY;
                    float rm = simd_max(sc), mo = m_row[rr], mn = max(mo, rm);
                    float al = (mo == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mo - mn);
                    float pv = (valid && mn != -INFINITY) ? exp2(sc - mn) : 0.0f;
                    l_row[rr] = fma(l_row[rr], al, simd_sum(pv)); m_row[rr] = mn;
                    sP[r][lane] = half(pv); if (lane == 0u) sA[r] = half(al);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            ld256(v_cache, v_sb, kb, tile_k);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint rb = qh * 32u;
            half a0 = sA[rb+0u+fm], a1 = sA[rb+8u+fm], a2 = sA[rb+16u+fm], a3 = sA[rb+24u+fm];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                O0[j].thread_elements()[0] *= a0; O0[j].thread_elements()[1] *= a0;
                O1[j].thread_elements()[0] *= a1; O1[j].thread_elements()[1] *= a1;
                O2[j].thread_elements()[0] *= a2; O2[j].thread_elements()[1] *= a2;
                O3[j].thread_elements()[0] *= a3; O3[j].thread_elements()[1] *= a3;
            }
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 p0, p1, p2, p3;
                simdgroup_load(p0, &sP[rb + 0u][kk],  36);
                simdgroup_load(p1, &sP[rb + 8u][kk],  36);
                simdgroup_load(p2, &sP[rb + 16u][kk], 36);
                simdgroup_load(p3, &sP[rb + 24u][kk], 36);
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8u; j++) {
                    simdgroup_half8x8 vm; simdgroup_load(vm, &sKV[kk][dq * 64u + 8u * j], STR);
                    simdgroup_multiply_accumulate(O0[j], p0, vm, O0[j]);
                    simdgroup_multiply_accumulate(O1[j], p1, vm, O1[j]);
                    simdgroup_multiply_accumulate(O2[j], p2, vm, O2[j]);
                    simdgroup_multiply_accumulate(O3[j], p3, vm, O3[j]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 8u; rr++) {
                uint r = sg * 8u + rr;
                float mg = sMg[r], lg = sLg[r], mc = m_row[rr], lc = l_row[rr];
                float mn = max(mg, mc);
                float ag = (mg == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mg - mn);
                float ac = (mc == -INFINITY || mn == -INFINITY) ? 0.0f : exp2(mc - mn);
                sMg[r] = mn;
                sLg[r] = fma(lg, ag, lc * ac);
                sAg[r] = ag;
                sAc[r] = ac;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool is_last_chunk = (kb_c_end >= max_kv);
        uint rb = qh * 32u;
        for (uint q_sub = 0; q_sub < 4u; q_sub++) {
            uint r_loc = rb + q_sub * 8u + fm;
            uint gt = t0 + r_loc;
            if (gt < p.M) {
                ulong q_off = ((ulong)gt * p.Hq + h) * 256ul;
                device float* og = o_f32 + q_off;
                float ag = sAg[r_loc], ac = sAc[r_loc];
                if (is_last_chunk) {
                    float inv_l = (sLg[r_loc] > 0.0f) ? (1.0f / sLg[r_loc]) : 0.0f;
                    const device float* gp = gate + q_off;
                    device half* op = out + q_off;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8u; j++) {
                        uint c = dq * 64u + 8u * j + fn;
                        float cv0 = float((q_sub==0u)?O0[j].thread_elements()[0]:((q_sub==1u)?O1[j].thread_elements()[0]:((q_sub==2u)?O2[j].thread_elements()[0]:O3[j].thread_elements()[0])));
                        float cv1 = float((q_sub==0u)?O0[j].thread_elements()[1]:((q_sub==1u)?O1[j].thread_elements()[1]:((q_sub==2u)?O2[j].thread_elements()[1]:O3[j].thread_elements()[1])));
                        float v0 = first_chunk ? (cv0 * ac) : fma(og[c],      ag, cv0 * ac);
                        float v1 = first_chunk ? (cv1 * ac) : fma(og[c + 1u], ag, cv1 * ac);
                        op[c]      = half((v0 * inv_l) * sigmoid_f32(gp[c]));
                        op[c + 1u] = half((v1 * inv_l) * sigmoid_f32(gp[c + 1u]));
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8u; j++) {
                        uint c = dq * 64u + 8u * j + fn;
                        float cv0 = float((q_sub==0u)?O0[j].thread_elements()[0]:((q_sub==1u)?O1[j].thread_elements()[0]:((q_sub==2u)?O2[j].thread_elements()[0]:O3[j].thread_elements()[0])));
                        float cv1 = float((q_sub==0u)?O0[j].thread_elements()[1]:((q_sub==1u)?O1[j].thread_elements()[1]:((q_sub==2u)?O2[j].thread_elements()[1]:O3[j].thread_elements()[1])));
                        og[c]      = first_chunk ? (cv0 * ac) : fma(og[c],      ag, cv0 * ac);
                        og[c + 1u] = first_chunk ? (cv1 * ac) : fma(og[c + 1u], ag, cv1 * ac);
                    }
                }
            }
        }
        first_chunk = false;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen_gqa_prefill_flash_hier_f16_f16(
    const device half*  q_h   [[buffer(0)]], const device float* gate [[buffer(1)]],
    const device half*  kc    [[buffer(2)]], const device half*  vc   [[buffer(3)]],
    device half*        out   [[buffer(4)]], device float*       of32 [[buffer(5)]],
    constant FullAttnParams& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[64][36], sA[64];
    threadgroup float sMg[64], sLg[64], sAg[64], sAc[64];
    qwen_gqa_prefill_flash_hier_impl<16>(q_h, gate, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, out, of32, p, tgid, tid, lane, sg, sKV, sP, sA, sMg, sLg, sAg, sAc);
}

kernel void qwen_gqa_prefill_flash_hier_q8_f16(
    const device half*   q_h   [[buffer(0)]], const device float*  gate [[buffer(1)]],
    const device uchar*  kc    [[buffer(2)]], const device uchar*  vc   [[buffer(3)]],
    const device float2* ksb   [[buffer(4)]], const device float2* vsb  [[buffer(5)]],
    device half*         out   [[buffer(6)]], device float*        of32 [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[64][36], sA[64];
    threadgroup float sMg[64], sLg[64], sAg[64], sAc[64];
    qwen_gqa_prefill_flash_hier_impl<8>(q_h, gate, kc, vc, ksb, vsb, out, of32, p, tgid, tid, lane, sg, sKV, sP, sA, sMg, sLg, sAg, sAc);
}

kernel void qwen_gqa_prefill_flash_hier_q4_f16(
    const device half*   q_h   [[buffer(0)]], const device float*  gate [[buffer(1)]],
    const device uchar*  kc    [[buffer(2)]], const device uchar*  vc   [[buffer(3)]],
    const device float2* ksb   [[buffer(4)]], const device float2* vsb  [[buffer(5)]],
    device half*         out   [[buffer(6)]], device float*        of32 [[buffer(7)]],
    constant FullAttnParams& p [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sKV[32][264], sP[64][36], sA[64];
    threadgroup float sMg[64], sLg[64], sAg[64], sAc[64];
    qwen_gqa_prefill_flash_hier_impl<4>(q_h, gate, kc, vc, ksb, vsb, out, of32, p, tgid, tid, lane, sg, sKV, sP, sA, sMg, sLg, sAg, sAc);
}

// ============================================================================
// Quest Query-Aware Block-Sparse Decode Attention (Opt-in via --sparse-kv-budget N)
// - Maintains exact per-32-token page (k_min, k_max) bounds in half2 (~1% KV RAM)
// - Scores all middle pages using exact mathematical Quest upper bound over all 6 GQA heads x M tokens,
//   normalized in log-softmax space (score_h - max_h) so every retrieval head is scale-invariant!
// - Always keeps sinks [0..127] + recent [kv_len-2048..kv_len-1] + exact top-K middle pages!
// ============================================================================
struct QuestParams {
    uint kv_len;
    uint Hq;
    uint Hkv;
    uint M;
    uint start_page;
    uint end_page;
    uint total_pages;
    uint sink_pages;
    uint recent_start_page;
    uint budget_pages;
    uint max_active_pages;
    uint score_stride;
};

template<int KV_BITS>
inline void qwen_quest_build_bounds_impl(
    const device uchar*  kc,
    const device float2* ksb,
    device half2*        k_bounds,
    constant QuestParams& p,
    uint2 tgid, uint tid
) {
    uint hkv = tgid.x;
    uint pg  = p.start_page + tgid.y;
    if (hkv >= p.Hkv || pg >= p.end_page) return;
    uint d = tid; // 0..255
    float mn = INFINITY, mx = -INFINITY;
    uint kb = pg * 32u;
    #pragma clang loop unroll_count(4)
    for (uint k = 0u; k < 32u; k++) {
        ulong row = (ulong)(kb + k) * p.Hkv + hkv;
        float v;
        if (KV_BITS == 16) {
            v = float(((const device half*)kc)[row * 256ul + d]);
        } else if (KV_BITS == 8) {
            float2 sb = ksb[row * 4ul + (d >> 6)];
            v = fma(float(kc[row * 256ul + d]), sb.x, sb.y);
        } else {
            float2 sb = ksb[row * 4ul + (d >> 6)];
            uint grp = d >> 6, off = d & 63u;
            uchar b = kc[row * 128ul + grp * 32u + (off & 31u)];
            uint qv = (off < 32u) ? (b & 0x0Fu) : (b >> 4u);
            v = fma(float(qv), sb.x, sb.y);
        }
        mn = min(mn, v);
        mx = max(mx, v);
    }
    k_bounds[((ulong)pg * p.Hkv + hkv) * 256ul + d] = half2(half(mn), half(mx));
}

kernel void qwen_quest_build_bounds_f16(
    const device half* kc [[buffer(0)]], device half2* kb [[buffer(1)]],
    constant QuestParams& p [[buffer(2)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) { qwen_quest_build_bounds_impl<16>((const device uchar*)kc, nullptr, kb, p, tgid, tid); }

kernel void qwen_quest_build_bounds_q8(
    const device uchar* kc [[buffer(0)]], const device float2* ksb [[buffer(1)]], device half2* kb [[buffer(2)]],
    constant QuestParams& p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) { qwen_quest_build_bounds_impl<8>(kc, ksb, kb, p, tgid, tid); }

kernel void qwen_quest_build_bounds_q4(
    const device uchar* kc [[buffer(0)]], const device float2* ksb [[buffer(1)]], device half2* kb [[buffer(2)]],
    constant QuestParams& p [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) { qwen_quest_build_bounds_impl<4>(kc, ksb, kb, p, tgid, tid); }

kernel void qwen_quest_score_pages(
    const device float*  q_in        [[buffer(0)]], // [M, Hq, 256]
    const device half2*  k_bounds    [[buffer(1)]], // [max_pages, Hkv, 256]
    device float*        page_scores [[buffer(2)]], // [Hkv*7, score_stride]
    constant QuestParams& p          [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    uint hkv = tgid.x;
    if (hkv >= p.Hkv) return;

    threadgroup half sQlo[6][256];
    threadgroup half sQhi[6][256];
    for (uint hl = 0u; hl < 6u; hl++) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint t = 0u; t < p.M; t++) {
            float qv = q_in[((ulong)t * p.Hq + (hkv * 6u + hl)) * 256ul + tid];
            lo = min(lo, qv);
            hi = max(hi, qv);
        }
        sQlo[hl][tid] = half(lo);
        sQhi[hl][tid] = half(hi);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_mid = p.recent_start_page - p.sink_pages;
    for (uint idx = tgid.y * 8u + sg; idx < n_mid; idx += 16u * 8u) {
        uint pg = p.sink_pages + idx;
        const device half2* kb_ptr = k_bounds + ((ulong)pg * p.Hkv + hkv) * 256ul + lane * 8u;
        float kmn[8], kmx[8];
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; i++) {
            half2 b = kb_ptr[i];
            kmn[i] = float(b.x);
            kmx[i] = float(b.y);
        }
        #pragma clang loop unroll(full)
        for (uint hl = 0u; hl < 6u; hl++) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 8u; i++) {
                float ql = float(sQlo[hl][lane * 8u + i]);
                float qh = float(sQhi[hl][lane * 8u + i]);
                float p0 = max(qh * kmx[i], qh * kmn[i]);
                float p1 = max(ql * kmx[i], ql * kmn[i]);
                acc += max(p0, p1);
            }
            float s_hl = simd_sum(acc);
            if (lane == 0u) {
                page_scores[((ulong)hkv * 7u + hl) * p.score_stride + pg] = s_hl;
            }
        }
    }
}

kernel void qwen_quest_select_topk(
    device float*        page_scores   [[buffer(0)]], // [Hkv*7, score_stride]
    device uint*         active_pages  [[buffer(1)]], // [Hkv, max_active_pages]
    device uint*         active_counts [[buffer(2)]], // [Hkv]
    constant QuestParams& p            [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    uint hkv = tgid;
    if (hkv >= p.Hkv) return;

    uint pg_lo = p.sink_pages;
    uint pg_hi = p.recent_start_page;
    uint n_rec = p.total_pages - p.recent_start_page;
    uint K_mid = (p.budget_pages > p.sink_pages + n_rec) ? (p.budget_pages - p.sink_pages - n_rec) : 0u;
    uint n_mid = pg_hi - pg_lo;
    if (K_mid > n_mid) K_mid = n_mid;

    threadgroup float sMaxH[6];
    threadgroup float sRedF[8];
    threadgroup uint  sRedU[8];
    threadgroup float sMinC;
    threadgroup atomic_uint sBits[256];
    threadgroup atomic_uint sTaken;

    // 1. Per-head max over middle pages (for scale-invariant softmax-space normalization)
    for (uint hl = 0u; hl < 6u; hl++) {
        const device float* sp = page_scores + ((ulong)hkv * 7u + hl) * p.score_stride;
        float mx = -INFINITY;
        for (uint pg = pg_lo + tid; pg < pg_hi; pg += 256u) mx = max(mx, sp[pg]);
        mx = simd_max(mx);
        if (lane == 0u) sRedF[sg] = mx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = (lane < 8u) ? sRedF[lane] : -INFINITY;
            v = simd_max(v);
            if (lane == 0u) sMaxH[hl] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2. Combined scale-invariant score = max_{hl=0..5}(score_hl - max_hl) <= 0.0f
    device float* comb = page_scores + ((ulong)hkv * 7u + 6u) * p.score_stride;
    float mn_c = INFINITY;
    for (uint pg = pg_lo + tid; pg < pg_hi; pg += 256u) {
        float c = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint hl = 0u; hl < 6u; hl++) {
            float sc = page_scores[((ulong)hkv * 7u + hl) * p.score_stride + pg] - sMaxH[hl];
            c = max(c, sc);
        }
        comb[pg] = c;
        mn_c = min(mn_c, c);
    }
    mn_c = simd_min(mn_c);
    if (lane == 0u) sRedF[sg] = mn_c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < 8u) ? sRedF[lane] : INFINITY;
        v = simd_min(v);
        if (lane == 0u) sMinC = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Exact 20-step binary search for top-K threshold (maintains invariant count(comb >= lo) >= K_mid)
    float lo = sMinC;
    float hi = 0.0f;
    for (uint iter = 0u; iter < 20u; iter++) {
        float mid = 0.5f * (lo + hi);
        uint cnt = 0u;
        for (uint pg = pg_lo + tid; pg < pg_hi; pg += 256u) {
            if (comb[pg] >= mid) cnt++;
        }
        cnt = simd_sum(cnt);
        if (lane == 0u) sRedU[sg] = cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tot = 0u;
        if (sg == 0u) {
            uint v = (lane < 8u) ? sRedU[lane] : 0u;
            tot = simd_sum(v);
            if (lane == 0u) sRedU[0] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sRedU[0] >= K_mid) lo = mid; else hi = mid;
    }

    // 4. Mark sinks + recent + exact top-K_mid middle pages into 8192-bit SRAM bitmask
    atomic_store_explicit(&sBits[tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&sTaken, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.sink_pages) {
        atomic_fetch_or_explicit(&sBits[tid >> 5], 1u << (tid & 31u), memory_order_relaxed);
    }
    for (uint pg = p.recent_start_page + tid; pg < p.total_pages; pg += 256u) {
        atomic_fetch_or_explicit(&sBits[pg >> 5], 1u << (pg & 31u), memory_order_relaxed);
    }
    for (uint pg = pg_lo + tid; pg < pg_hi; pg += 256u) {
        if (comb[pg] > lo) {
            uint slot = atomic_fetch_add_explicit(&sTaken, 1u, memory_order_relaxed);
            if (slot < K_mid) {
                atomic_fetch_or_explicit(&sBits[pg >> 5], 1u << (pg & 31u), memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&sTaken, memory_order_relaxed) < K_mid) {
        for (uint pg = pg_lo + tid; pg < pg_hi; pg += 256u) {
            if (comb[pg] == lo) {
                if (atomic_load_explicit(&sTaken, memory_order_relaxed) >= K_mid) break;
                uint slot = atomic_fetch_add_explicit(&sTaken, 1u, memory_order_relaxed);
                if (slot < K_mid) {
                    atomic_fetch_or_explicit(&sBits[pg >> 5], 1u << (pg & 31u), memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Parallel prefix-sum stream compaction into strictly ascending active_pages[hkv][0..cnt-1]
    uint w = atomic_load_explicit(&sBits[tid], memory_order_relaxed);
    uint c = popcount(w);
    uint sg_ex = simd_prefix_exclusive_sum(c);
    uint sg_tot = simd_sum(c);
    if (lane == 0u) sRedU[sg] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint k = 0u; k < sg; k++) sg_off += sRedU[k];
    uint out_pos = sg_off + sg_ex;
    uint out_base = hkv * p.max_active_pages;
    uint cur_w = w;
    while (cur_w != 0u) {
        uint b = ctz(cur_w);
        if (out_pos < p.max_active_pages) active_pages[out_base + out_pos] = tid * 32u + b;
        out_pos++;
        cur_w &= (cur_w - 1u);
    }
    if (tid == 255u) {
        active_counts[hkv] = min(out_pos, p.max_active_pages);
    }
}

// Sparse GQA-6 AMX entry points (M=1, M=2, M=3..4, M=5..8 x f16/q8/q4 x dense/tree)
#define DEF_GQA6_M1_SP_F16(NAME, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], \
    constant FullAttnParams& p [[buffer(6)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8]; \
    qwen_gqa6_m1_splitk_p1_impl<16, IS_TREE, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sS, sP, sA, ap, ac); \
}
#define DEF_GQA6_M1_SP_Q(NAME, BITS, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device float* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]], \
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]], \
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]], \
    constant FullAttnParams& p [[buffer(8)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sS[4][8][32], sP[8][36], sA[8]; \
    qwen_gqa6_m1_splitk_p1_impl<BITS, IS_TREE, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sS, sP, sA, ap, ac); \
}
#define DEF_GQA6_M2_SP_F16(NAME, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], \
    constant FullAttnParams& p [[buffer(6)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16]; \
    qwen_gqa6_m2_splitk_p1_impl<16, IS_TREE, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sS, sP, sA, ap, ac); \
}
#define DEF_GQA6_M2_SP_Q(NAME, BITS, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]], \
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]], \
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]], \
    constant FullAttnParams& p [[buffer(8)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sS[4][16][32], sP[16][36], sA[16]; \
    qwen_gqa6_m2_splitk_p1_impl<BITS, IS_TREE, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sS, sP, sA, ap, ac); \
}
#define DEF_GQA6_M4_SP_F16(NAME, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], \
    constant FullAttnParams& p [[buffer(6)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sP[24][36], sA[24]; \
    qwen_gqa6_m4_splitk_p1_impl<16, IS_TREE, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sP, sA, ap, ac); \
}
#define DEF_GQA6_M4_SP_Q(NAME, BITS, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]], \
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]], \
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]], \
    constant FullAttnParams& p [[buffer(8)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV[32][264], sP[24][36], sA[24]; \
    qwen_gqa6_m4_splitk_p1_impl<BITS, IS_TREE, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV, sP, sA, ap, ac); \
}
#define DEF_GQA6_V_SP_F16(NAME, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]], \
    device float* po [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], \
    constant FullAttnParams& p [[buffer(6)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48]; \
    qwen_gqa6_verify_splitk_p1_impl<16, IS_TREE, true>(q, (const device uchar*)kc, (const device uchar*)vc, nullptr, nullptr, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV_raw, sP, sAlpha, ap, ac); \
}
#define DEF_GQA6_V_SP_Q(NAME, BITS, IS_TREE, TP_ARG, TP_PTR, AP_IDX, AC_IDX) \
kernel void NAME( \
    const device half* q [[buffer(0)]], const device uchar* kc [[buffer(1)]], const device uchar* vc [[buffer(2)]], \
    const device float2* ksb [[buffer(3)]], const device float2* vsb [[buffer(4)]], \
    device float* po [[buffer(5)]], device float* pm [[buffer(6)]], device float* pl [[buffer(7)]], \
    constant FullAttnParams& p [[buffer(8)]] TP_ARG, \
    const device uint* ap [[buffer(AP_IDX)]], const device uint* ac [[buffer(AC_IDX)]], \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half sKV_raw[4224], sP[48][36], sAlpha[48]; \
    qwen_gqa6_verify_splitk_p1_impl<BITS, IS_TREE, true>(q, kc, vc, ksb, vsb, po, pm, pl, p, TP_PTR, tgid, tid, lane, sg, sKV_raw, sP, sAlpha, ap, ac); \
}

#define QUEST_TP_F16 , constant TreeParams& tp [[buffer(7)]]
#define QUEST_TP_Q   , constant TreeParams& tp [[buffer(9)]]
#define QUEST_NO_TP

DEF_GQA6_M1_SP_F16(qwen_gqa6_m1_splitk_p1_f16_sparse,      false, QUEST_NO_TP,  nullptr, 7,  8)
DEF_GQA6_M1_SP_F16(qwen_gqa6_m1_splitk_p1_f16_tree_sparse, true,  QUEST_TP_F16, &tp,     8,  9)
DEF_GQA6_M1_SP_Q  (qwen_gqa6_m1_splitk_p1_q8_sparse,    8, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M1_SP_Q  (qwen_gqa6_m1_splitk_p1_q8_tree_sparse,8,true,  QUEST_TP_Q,   &tp,     10, 11)
DEF_GQA6_M1_SP_Q  (qwen_gqa6_m1_splitk_p1_q4_sparse,    4, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M1_SP_Q  (qwen_gqa6_m1_splitk_p1_q4_tree_sparse,4,true,  QUEST_TP_Q,   &tp,     10, 11)

DEF_GQA6_M2_SP_F16(qwen_gqa6_m2_splitk_p1_f16_sparse,      false, QUEST_NO_TP,  nullptr, 7,  8)
DEF_GQA6_M2_SP_F16(qwen_gqa6_m2_splitk_p1_f16_tree_sparse, true,  QUEST_TP_F16, &tp,     8,  9)
DEF_GQA6_M2_SP_Q  (qwen_gqa6_m2_splitk_p1_q8_sparse,    8, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M2_SP_Q  (qwen_gqa6_m2_splitk_p1_q8_tree_sparse,8,true,  QUEST_TP_Q,   &tp,     10, 11)
DEF_GQA6_M2_SP_Q  (qwen_gqa6_m2_splitk_p1_q4_sparse,    4, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M2_SP_Q  (qwen_gqa6_m2_splitk_p1_q4_tree_sparse,4,true,  QUEST_TP_Q,   &tp,     10, 11)

DEF_GQA6_M4_SP_F16(qwen_gqa6_m4_splitk_p1_f16_sparse,      false, QUEST_NO_TP,  nullptr, 7,  8)
DEF_GQA6_M4_SP_F16(qwen_gqa6_m4_splitk_p1_f16_tree_sparse, true,  QUEST_TP_F16, &tp,     8,  9)
DEF_GQA6_M4_SP_Q  (qwen_gqa6_m4_splitk_p1_q8_sparse,    8, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M4_SP_Q  (qwen_gqa6_m4_splitk_p1_q8_tree_sparse,8,true,  QUEST_TP_Q,   &tp,     10, 11)
DEF_GQA6_M4_SP_Q  (qwen_gqa6_m4_splitk_p1_q4_sparse,    4, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_M4_SP_Q  (qwen_gqa6_m4_splitk_p1_q4_tree_sparse,4,true,  QUEST_TP_Q,   &tp,     10, 11)

DEF_GQA6_V_SP_F16 (qwen_gqa6_verify_splitk_p1_f16_sparse,      false, QUEST_NO_TP,  nullptr, 7,  8)
DEF_GQA6_V_SP_F16 (qwen_gqa6_verify_splitk_p1_f16_tree_sparse, true,  QUEST_TP_F16, &tp,     8,  9)
DEF_GQA6_V_SP_Q   (qwen_gqa6_verify_splitk_p1_q8_sparse,    8, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_V_SP_Q   (qwen_gqa6_verify_splitk_p1_q8_tree_sparse,8,true,  QUEST_TP_Q,   &tp,     10, 11)
DEF_GQA6_V_SP_Q   (qwen_gqa6_verify_splitk_p1_q4_sparse,    4, false, QUEST_NO_TP,  nullptr, 9,  10)
DEF_GQA6_V_SP_Q   (qwen_gqa6_verify_splitk_p1_q4_tree_sparse,4,true,  QUEST_TP_Q,   &tp,     10, 11)
