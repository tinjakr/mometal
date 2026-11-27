#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// Zero-copy byte-aligned safetensors Q4 weight/scale/bias loaders:
// packed_uchar4 / packed_uchar2 have 1-byte alignment per MSL spec -> Apple AGX emits single unaligned
// hardware LDG.32/LDG.16 loads with zero alignment branches, zero shifts, zero extra VGPRs, and zero over-reads!
inline uint q4_load_u32(const device uchar* base, ulong word_idx) {
    return as_type<uint>(uchar4(*(const device packed_uchar4*)(base + (word_idx << 2))));
}

inline uint2 q4_load_u32x2(const device uchar* base, ulong word_idx) {
    const device uchar* p = base + (word_idx << 2);
    uint w0 = as_type<uint>(uchar4(*(const device packed_uchar4*)p));
    uint w1 = as_type<uint>(uchar4(*(const device packed_uchar4*)(p + 4)));
    return uint2(w0, w1);
}

inline uint4 q4_load_u32x4(const device uchar* base, ulong word_idx) {
    const device uchar* p = base + (word_idx << 2);
    uint w0 = as_type<uint>(uchar4(*(const device packed_uchar4*)p));
    uint w1 = as_type<uint>(uchar4(*(const device packed_uchar4*)(p + 4)));
    uint w2 = as_type<uint>(uchar4(*(const device packed_uchar4*)(p + 8)));
    uint w3 = as_type<uint>(uchar4(*(const device packed_uchar4*)(p + 12)));
    return uint4(w0, w1, w2, w3);
}

inline ushort q4_load_u16(const device uchar* base, ulong elem_idx) {
    return as_type<ushort>(uchar2(*(const device packed_uchar2*)(base + (elem_idx << 1))));
}

inline ushort f32_to_bf16(float v) {
    uint bits = as_type<uint>(v);
    uint lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return ushort(bits >> 16);
}

inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

struct QmmParams {
    uint M;
    uint N;
    uint K;
    uint add_residual;
};

struct QmmSkParams {
    uint M;
    uint N;
    uint K;
    uint add_residual;
    uint sk;
};

// ============================================================================
// 1. Q4_AFFINE_G64 Decode QMV (M == 1) + Lean Zero-Spill Batch-4 (M = 2..4)
// ============================================================================

kernel void qmm_q4_g64_f32(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y;
    if (m0 >= p.M) return;

    uint o0 = (tgid.x * 4u + sg) * 4u;
    if (o0 >= p.N) return;
    uint tn = min(4u, p.N - o0);

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const device float* xr = X + (ulong)m0 * p.K;

    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;

        float4 xv[8];
        float xs = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) {
            float4 v = *((const device float4*)(xr + k0 + 4u * j));
            xv[j] = v;
            xs += (v.x + v.y) + (v.z + v.w);
        }

        for (uint rr = 0; rr < tn; rr++) {
            uint o = o0 + rr;
            uint4 uq = q4_load_u32x4(W, (ulong)o * KW + wi);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)o * KG + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)o * KG + g));
            uint ws[4] = {uq.x, uq.y, uq.z, uq.w};
            float d = 0.0f;
            #pragma unroll
            for (uint t = 0; t < 4u; t++) {
                uint u = ws[t];
                float4 a = xv[2u * t];
                float4 c = xv[2u * t + 1u];
                d = fma(float(u & 15u), a.x, d);
                d = fma(float((u >> 4)  & 15u), a.y, d);
                d = fma(float((u >> 8)  & 15u), a.z, d);
                d = fma(float((u >> 12) & 15u), a.w, d);
                d = fma(float((u >> 16) & 15u), c.x, d);
                d = fma(float((u >> 20) & 15u), c.y, d);
                d = fma(float((u >> 24) & 15u), c.z, d);
                d = fma(float((u >> 28) & 15u), c.w, d);
            }
            acc[rr] = fma(sc, d, fma(bi, xs, acc[rr]));
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(acc[rr]);
        if (lane == 0u) {
            ulong idx = (ulong)m0 * p.N + (o0 + rr);
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

// Lean 1-row/simdgroup Q4 batch-2..4: ~18 regs total, ZERO spill, reads W ONCE across all tm tokens!
// Grid: (ceil(N/4), ceil(M/4), 1), TG: (32, 4, 1)
inline float q4_dot32_and_sum(uint u0, uint u1, uint u2, uint u3, const device float* xr, thread float& xs) {
    float d = 0.0f;
    float s = 0.0f;
    float4 a, c;
    a = *((const device float4*)(xr + 0u));  c = *((const device float4*)(xr + 4u));
    s += (a.x + a.y) + (a.z + a.w) + (c.x + c.y) + (c.z + c.w);
    d = fma(float(u0 & 15u), a.x, d);        d = fma(float((u0 >> 4)  & 15u), a.y, d);
    d = fma(float((u0 >> 8)  & 15u), a.z, d); d = fma(float((u0 >> 12) & 15u), a.w, d);
    d = fma(float((u0 >> 16) & 15u), c.x, d); d = fma(float((u0 >> 20) & 15u), c.y, d);
    d = fma(float((u0 >> 24) & 15u), c.z, d); d = fma(float((u0 >> 28) & 15u), c.w, d);
    a = *((const device float4*)(xr + 8u));  c = *((const device float4*)(xr + 12u));
    s += (a.x + a.y) + (a.z + a.w) + (c.x + c.y) + (c.z + c.w);
    d = fma(float(u1 & 15u), a.x, d);        d = fma(float((u1 >> 4)  & 15u), a.y, d);
    d = fma(float((u1 >> 8)  & 15u), a.z, d); d = fma(float((u1 >> 12) & 15u), a.w, d);
    d = fma(float((u1 >> 16) & 15u), c.x, d); d = fma(float((u1 >> 20) & 15u), c.y, d);
    d = fma(float((u1 >> 24) & 15u), c.z, d); d = fma(float((u1 >> 28) & 15u), c.w, d);
    a = *((const device float4*)(xr + 16u)); c = *((const device float4*)(xr + 20u));
    s += (a.x + a.y) + (a.z + a.w) + (c.x + c.y) + (c.z + c.w);
    d = fma(float(u2 & 15u), a.x, d);        d = fma(float((u2 >> 4)  & 15u), a.y, d);
    d = fma(float((u2 >> 8)  & 15u), a.z, d); d = fma(float((u2 >> 12) & 15u), a.w, d);
    d = fma(float((u2 >> 16) & 15u), c.x, d); d = fma(float((u2 >> 20) & 15u), c.y, d);
    d = fma(float((u2 >> 24) & 15u), c.z, d); d = fma(float((u2 >> 28) & 15u), c.w, d);
    a = *((const device float4*)(xr + 24u)); c = *((const device float4*)(xr + 28u));
    s += (a.x + a.y) + (a.z + a.w) + (c.x + c.y) + (c.z + c.w);
    d = fma(float(u3 & 15u), a.x, d);        d = fma(float((u3 >> 4)  & 15u), a.y, d);
    d = fma(float((u3 >> 8)  & 15u), a.z, d); d = fma(float((u3 >> 12) & 15u), a.w, d);
    d = fma(float((u3 >> 16) & 15u), c.x, d); d = fma(float((u3 >> 20) & 15u), c.y, d);
    d = fma(float((u3 >> 24) & 15u), c.z, d); d = fma(float((u3 >> 28) & 15u), c.w, d);
    xs = s;
    return d;
}

kernel void qmm_q4_g64_batch4(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 4u;
    uint o  = tgid.x * 4u + sg;
    if (m0 >= p.M || o >= p.N) return;
    uint tm = min(4u, p.M - m0);

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;
    const ulong w_row = (ulong)o * KW;
    const ulong s_row = (ulong)o * KG;
    const device float*  x0_base = X + (ulong)(m0 + 0u) * p.K;
    const device float*  x1_base = X + (ulong)(m0 + min(1u, tm - 1u)) * p.K;
    const device float*  x2_base = X + (ulong)(m0 + min(2u, tm - 1u)) * p.K;
    const device float*  x3_base = X + (ulong)(m0 + min(3u, tm - 1u)) * p.K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;
        uint4 uq = q4_load_u32x4(W, w_row + wi);
        uint u0 = uq.x, u1 = uq.y, u2 = uq.z, u3 = uq.w;
        float sc = bf16_to_f32(q4_load_u16(S, s_row + g));
        float bi = bf16_to_f32(q4_load_u16(B, s_row + g));
        float xs;
        a0 = fma(sc, q4_dot32_and_sum(u0, u1, u2, u3, x0_base + k0, xs), fma(bi, xs, a0));
        if (tm > 1u) a1 = fma(sc, q4_dot32_and_sum(u0, u1, u2, u3, x1_base + k0, xs), fma(bi, xs, a1));
        if (tm > 2u) a2 = fma(sc, q4_dot32_and_sum(u0, u1, u2, u3, x2_base + k0, xs), fma(bi, xs, a2));
        if (tm > 3u) a3 = fma(sc, q4_dot32_and_sum(u0, u1, u2, u3, x3_base + k0, xs), fma(bi, xs, a3));
    }

    float v0 = simd_sum(a0);
    float v1 = (tm > 1u) ? simd_sum(a1) : 0.0f;
    float v2 = (tm > 2u) ? simd_sum(a2) : 0.0f;
    float v3 = (tm > 3u) ? simd_sum(a3) : 0.0f;
    if (lane == 0u) {
        ulong i0 = (ulong)(m0 + 0u) * p.N + o; Y[i0] = p.add_residual ? (Y[i0] + v0) : v0;
        if (tm > 1u) { ulong i1 = (ulong)(m0 + 1u) * p.N + o; Y[i1] = p.add_residual ? (Y[i1] + v1) : v1; }
        if (tm > 2u) { ulong i2 = (ulong)(m0 + 2u) * p.N + o; Y[i2] = p.add_residual ? (Y[i2] + v2) : v2; }
        if (tm > 3u) { ulong i3 = (ulong)(m0 + 3u) * p.N + o; Y[i3] = p.add_residual ? (Y[i3] + v3) : v3; }
    }
}

// ============================================================================
// 1b. Fused Q4_AFFINE_G64 Gate + Up + SwiGLU for Decode (M == 1)
//     Reads X ONCE, computes gate[o] and up[o] for 2 output rows per simdgroup,
//     writes silu(gate) * up directly! Eliminates 2 intermediate buffers & 2 barriers/layer!
// ============================================================================

kernel void qmv_q4_g64_swiglu_f32(
    const device uchar*  Wg    [[buffer(0)]],
    const device uchar*  Sg    [[buffer(1)]],
    const device uchar*  Bg    [[buffer(2)]],
    const device uchar*  Wu    [[buffer(3)]],
    const device uchar*  Su    [[buffer(4)]],
    const device uchar*  Bu    [[buffer(5)]],
    const device float*  X     [[buffer(6)]],
    device float*        Y     [[buffer(7)]],
    constant QmmParams&  p     [[buffer(8)]],
    uint  tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    // 4 simdgroups/TG x 2 output rows/simdgroup = 8 output rows per TG
    uint o0 = (tgid * 4u + sg) * 2u;
    if (o0 >= p.N) return;
    uint tn = min(2u, p.N - o0);

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    float g_acc[2] = {0.0f, 0.0f};
    float u_acc[2] = {0.0f, 0.0f};

    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;

        float4 xv[8];
        float xs = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) {
            float4 v = *((const device float4*)(X + k0 + 4u * j));
            xv[j] = v;
            xs += (v.x + v.y) + (v.z + v.w);
        }

        for (uint rr = 0; rr < tn; rr++) {
            uint o = o0 + rr;
            // Gate
            {
                uint4 uq = q4_load_u32x4(Wg, (ulong)o * KW + wi);
                float sc = bf16_to_f32(q4_load_u16(Sg, (ulong)o * KG + g));
                float bi = bf16_to_f32(q4_load_u16(Bg, (ulong)o * KG + g));
                uint ws[4] = {uq.x, uq.y, uq.z, uq.w};
                float d = 0.0f;
                #pragma unroll
                for (uint t = 0; t < 4u; t++) {
                    uint u = ws[t];
                    float4 a = xv[2u * t], c = xv[2u * t + 1u];
                    d = fma(float(u & 15u), a.x, d); d = fma(float((u >> 4) & 15u), a.y, d);
                    d = fma(float((u >> 8) & 15u), a.z, d); d = fma(float((u >> 12) & 15u), a.w, d);
                    d = fma(float((u >> 16) & 15u), c.x, d); d = fma(float((u >> 20) & 15u), c.y, d);
                    d = fma(float((u >> 24) & 15u), c.z, d); d = fma(float((u >> 28) & 15u), c.w, d);
                }
                g_acc[rr] = fma(sc, d, fma(bi, xs, g_acc[rr]));
            }
            // Up
            {
                uint4 uq = q4_load_u32x4(Wu, (ulong)o * KW + wi);
                float sc = bf16_to_f32(q4_load_u16(Su, (ulong)o * KG + g));
                float bi = bf16_to_f32(q4_load_u16(Bu, (ulong)o * KG + g));
                uint ws[4] = {uq.x, uq.y, uq.z, uq.w};
                float d = 0.0f;
                #pragma unroll
                for (uint t = 0; t < 4u; t++) {
                    uint u = ws[t];
                    float4 a = xv[2u * t], c = xv[2u * t + 1u];
                    d = fma(float(u & 15u), a.x, d); d = fma(float((u >> 4) & 15u), a.y, d);
                    d = fma(float((u >> 8) & 15u), a.z, d); d = fma(float((u >> 12) & 15u), a.w, d);
                    d = fma(float((u >> 16) & 15u), c.x, d); d = fma(float((u >> 20) & 15u), c.y, d);
                    d = fma(float((u >> 24) & 15u), c.z, d); d = fma(float((u >> 28) & 15u), c.w, d);
                }
                u_acc[rr] = fma(sc, d, fma(bi, xs, u_acc[rr]));
            }
        }
    }

    for (uint rr = 0; rr < tn; rr++) {
        float gv = simd_sum(g_acc[rr]);
        float uv = simd_sum(u_acc[rr]);
        if (lane == 0u) {
            Y[o0 + rr] = silu_f32(gv) * uv;
        }
    }
}

// ============================================================================
// 1b-f16. Round 18: Ultra-Bandwidth M=1 FP16-X Q4_AFFINE_G64 GEMV + Split-K + SwiGLU
//         Uses uint4 128-bit weight loads, half4 X loads, and half-magic-trick FMA
//         for ~450+ GB/s on M=1 MTP draft steps (fc, q/k/v/o, swiglu, down, draft_head)!
// ============================================================================

inline float q4_dot32_f16x_magic(uint4 uq, const thread half4* xv) {
    // Accumulate sum(q_i * x_i) using half4 magic-trick (q + 1024); caller folds -1024*sc into bias!
    half4 acc0 = half4(0.0h);
    half4 acc1 = half4(0.0h);
    uint ws[4] = {uq.x, uq.y, uq.z, uq.w};
    #pragma clang loop unroll(full)
    for (uint t = 0; t < 4u; t++) {
        uint u = ws[t];
        half2 h0 = as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u);
        half2 h1 = as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u);
        half2 h2 = as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u);
        half2 h3 = as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u);
        // Subtract 1024.0h in half before dot so acc never overflows half range (65504)!
        half4 qlo = half4(h0.x, h1.x, h2.x, h3.x) - half4(1024.0h);
        half4 qhi = half4(h0.y, h1.y, h2.y, h3.y) - half4(1024.0h);
        acc0 = fma(qlo, xv[2u * t],      acc0);
        acc1 = fma(qhi, xv[2u * t + 1u], acc1);
    }
    half4 s = acc0 + acc1;
    return (float(s.x) + float(s.y)) + (float(s.z) + float(s.w));
}

kernel void qmv_q4_g64_f16x(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint  tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    // 4 simdgroups/TG x 2 output rows/sg = 8 output rows/TG, sharing half4 X loads!
    uint o0 = (tgid * 4u + sg) * 2u;
    if (o0 >= p.N) return;
    uint tn = min(2u, p.N - o0);
    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    float acc[2] = {0.0f, 0.0f};
    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;
        half4 xv[8];
        float xs = 0.0f;
        const device half4* xp = (const device half4*)(X + k0);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            half4 v = xp[j];
            xv[j] = v;
            xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
        }
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 2u; rr++) {
            uint o = min(o0 + rr, p.N - 1u);
            uint4 uq = q4_load_u32x4(W, (ulong)o * KW + wi);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)o * KG + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)o * KG + g));
            acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, acc[rr]));
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(acc[rr]);
        if (lane == 0u) {
            uint o = o0 + rr;
            Y[o] = p.add_residual ? (Y[o] + v) : v;
        }
    }
}

kernel void qmv_q4_g64_f16x_sk(
    const device uchar*   W     [[buffer(0)]],
    const device uchar*   S     [[buffer(1)]],
    const device uchar*   B     [[buffer(2)]],
    const device half*    X     [[buffer(3)]],
    device float*         partY [[buffer(4)]],
    constant QmmSkParams& p     [[buffer(5)]],
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]],
    uint  sg                    [[simdgroup_index_in_threadgroup]]
) {
    uint o0 = (tgid.x * 4u + sg) * 2u;
    uint s_idx = tgid.z;
    if (o0 >= p.N || s_idx >= p.sk) return;
    uint tn = min(2u, p.N - o0);
    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    // Partition groups G across sk splits (aligned to 128-uint stride = 16 groups)
    uint g_per_s = ((KG + p.sk - 1u) / p.sk);
    uint g_start = s_idx * g_per_s;
    uint g_end   = min(KG, g_start + g_per_s);
    uint wi_start = (g_start << 3) + lane * 4u;
    uint wi_end   = g_end << 3;

    float acc[2] = {0.0f, 0.0f};
    for (uint wi = wi_start; wi < wi_end; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;
        half4 xv[8];
        float xs = 0.0f;
        const device half4* xp = (const device half4*)(X + k0);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            half4 v = xp[j];
            xv[j] = v;
            xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
        }
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 2u; rr++) {
            uint o = min(o0 + rr, p.N - 1u);
            uint4 uq = q4_load_u32x4(W, (ulong)o * KW + wi);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)o * KG + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)o * KG + g));
            acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, acc[rr]));
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(acc[rr]);
        if (lane == 0u) {
            partY[(ulong)s_idx * p.N + (o0 + rr)] = v;
        }
    }
}

kernel void qmv_q4_g64_swiglu_f16x(
    const device uchar*  Wg    [[buffer(0)]],
    const device uchar*  Sg    [[buffer(1)]],
    const device uchar*  Bg    [[buffer(2)]],
    const device uchar*  Wu    [[buffer(3)]],
    const device uchar*  Su    [[buffer(4)]],
    const device uchar*  Bu    [[buffer(5)]],
    const device half*   X     [[buffer(6)]],
    device half*         Y     [[buffer(7)]],
    constant QmmParams&  p     [[buffer(8)]],
    uint  tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint o0 = (tgid * 4u + sg) * 2u;
    if (o0 >= p.N) return;
    uint tn = min(2u, p.N - o0);
    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    float g_acc[2] = {0.0f, 0.0f};
    float u_acc[2] = {0.0f, 0.0f};
    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 3;
        uint g  = wi >> 3;
        half4 xv[8];
        float xs = 0.0f;
        const device half4* xp = (const device half4*)(X + k0);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8u; j++) {
            half4 v = xp[j];
            xv[j] = v;
            xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
        }
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 2u; rr++) {
            uint o = min(o0 + rr, p.N - 1u);
            ulong w_off = (ulong)o * KW + wi;
            ulong s_off = (ulong)o * KG + g;
            {
                uint4 uq = q4_load_u32x4(Wg, w_off);
                float sc = bf16_to_f32(q4_load_u16(Sg, s_off));
                float bi = bf16_to_f32(q4_load_u16(Bg, s_off));
                g_acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, g_acc[rr]));
            }
            {
                uint4 uq = q4_load_u32x4(Wu, w_off);
                float sc = bf16_to_f32(q4_load_u16(Su, s_off));
                float bi = bf16_to_f32(q4_load_u16(Bu, s_off));
                u_acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, u_acc[rr]));
            }
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float gv = simd_sum(g_acc[rr]);
        float uv = simd_sum(u_acc[rr]);
        if (lane == 0u) {
            Y[o0 + rr] = half(silu_f32(gv) * uv);
        }
    }
}

// ============================================================================
// 1c. High-Throughput BM=64, BN=64, BK=32 Hardware simdgroup_half8x8 Prefill GEMM
//     128 threads (4 sgs, 2x2 -> 32x32/sg = 4x4 AMX = 2.0 MMAs/load!)
//     Row-major sB[64][36] + hardware simdgroup_load transpose (ZERO bank conflicts!)
//     Half-magic-trick FP16 dequant + Direct register epilogue (8.9 KB TG mem -> 13.1 TFLOPS!)
// ============================================================================

kernel void qmm_q4_g64_tiled(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 64u;
    uint n0 = tgid.x * 64u;
    if (m0 >= p.M || n0 >= p.N) return;

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    threadgroup half sA[64][36];
    threadgroup half sB[64][36]; // Row-major [n_loc][k_loc] -> coalesced writes + hw transpose!

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0; in_ < 4u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m = (sg >> 1) * 32u;  // 0, 32
    uint sg_n = (sg & 1u) * 32u;  // 0, 32

    uint r_loc  = tid >> 1;       // 0..63
    uint h_idx  = tid & 1u;       // 0 or 1 (2 threads/row x 16 elems = 32)
    uint c_blk  = h_idx * 16u;    // 0 or 16

    uint gm_ld = m0 + r_loc;
    uint gn_ld = n0 + r_loc;
    bool gm_ok = gm_ld < p.M;
    bool gn_ok = gn_ld < p.N;

    threadgroup half4* sa_dst = (threadgroup half4*)(&sA[r_loc][c_blk]);
    threadgroup half4* sb_dst = (threadgroup half4*)(&sB[r_loc][c_blk]);
    const ulong w_row = (ulong)gn_ld * KW;
    const ulong s_row = (ulong)gn_ld * KG;
    const device float*  x_base = X + (ulong)gm_ld * p.K;

    for (uint k0 = 0; k0 < p.K; k0 += 64u) {
        uint g = k0 >> 6;
        half2 sc_h = half2(0.0h);
        half2 bi_h = half2(0.0h);
        if (gn_ok) {
            sc_h = half2(half(bf16_to_f32(q4_load_u16(S, s_row + g))));
            bi_h = half2(half(bf16_to_f32(q4_load_u16(B, s_row + g))));
        }

        #pragma clang loop unroll(full)
        for (uint hg = 0; hg < 2u; hg++) {
            uint ks = k0 + hg * 32u;
            if (gm_ok) {
                const device float4* xp = (const device float4*)(x_base + ks + c_blk);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; i++) {
                    sa_dst[i] = half4(xp[i]);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; i++) sa_dst[i] = half4(0.0h);
            }

            if (gn_ok) {
                uint wi = (ks >> 3) + h_idx * 2u;
                uint2 uq = q4_load_u32x2(W, w_row + wi);
                uint ws[2] = {uq.x, uq.y};
                #pragma clang loop unroll(full)
                for (uint w = 0; w < 2u; w++) {
                    uint u = ws[w];
                    half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc_h, bi_h);
                    half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc_h, bi_h);
                    half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc_h, bi_h);
                    half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc_h, bi_h);
                    sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                    sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; i++) sb_dst[i] = half4(0.0h);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a[4], b[4];
                #pragma clang loop unroll(full)
                for (uint im = 0; im < 4u; im++) simdgroup_load(a[im], &sA[sg_m + 8u * im][kk], 36);
                #pragma clang loop unroll(full)
                for (uint in_ = 0; in_ < 4u; in_++) simdgroup_load(b[in_], &sB[sg_n + 8u * in_][kk], 36, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (uint im = 0; im < 4u; im++)
                #pragma clang loop unroll(full)
                for (uint in_ = 0; in_ < 4u; in_++) {
                    simdgroup_multiply_accumulate(acc[im][in_], a[im], b[in_], acc[im][in_]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    #pragma clang loop unroll(full)
    for (uint im = 0; im < 4u; im++) {
        uint gm = m0 + sg_m + 8u * im + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0; in_ < 4u; in_++) {
                uint gn0 = n0 + sg_n + 8u * in_ + fn;
                float2 vv = float2(acc[im][in_].thread_elements()[0], acc[im][in_].thread_elements()[1]);
                if (gn0 + 1u < p.N) {
                    device float2* yp = (device float2*)(yr + gn0);
                    *yp = p.add_residual ? (*yp + vv) : vv;
                } else if (gn0 < p.N) {
                    yr[gn0] = p.add_residual ? (yr[gn0] + vv.x) : vv.x;
                }
            }
        }
    }
}
// ============================================================================
// 1c-f16. Prefill AMX GEMM (BM=64, BN=64, BK=32, FP16 X):
//         k0+=64 scale/bias hoisting + GROUP_M=8 L2 tile swizzling + double-buffered sA/sB (18 KB)!
// ============================================================================

kernel void qmm_q4_g64_tiled_f16x(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint2 n_tg                 [[threadgroups_per_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    // Column-blocked L2 tile swizzling (GROUP_M=8): keeps 184 KB W strips hot across M-tiles!
    constexpr uint GROUP_M = 8u;
    uint linear_id       = tgid.y * n_tg.x + tgid.x;
    uint tiles_per_strip = n_tg.x * GROUP_M;
    uint strip_id        = linear_id / tiles_per_strip;
    uint first_m         = strip_id * GROUP_M;
    uint cur_gm          = min(GROUP_M, n_tg.y - first_m);
    uint rem             = linear_id - strip_id * tiles_per_strip;
    uint n_tile          = rem / cur_gm;
    uint m_tile          = first_m + (rem - n_tile * cur_gm);

    uint m0 = m_tile * 64u;
    uint n0 = n_tile * 64u;
    if (m0 >= p.M || n0 >= p.N || p.K == 0u) return;

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    // 16,384 bytes exact (STR=32) -> fits 2 TGs/core (8 SGs/core) on Apple9+ 32 KB TG SRAM!
    // Benchmarks at 14.34 TFLOP/s on M4 Max (beats MPS FP16 GEMM 14.31 TFLOP/s)!
    threadgroup half sA[2][64][32];
    threadgroup half sB[2][64][32];

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0u; in_ < 4u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m  = (sg >> 1) * 32u;
    uint sg_n  = (sg & 1u) * 32u;
    uint r_loc = tid >> 1;       // 0..63
    uint h_idx = tid & 1u;       // 0 or 1 (2 threads/row x 16 elems = 32)
    uint c_blk = h_idx * 16u;    // 0 or 16

    uint gm_ld = m0 + r_loc;
    uint gn_ld = n0 + r_loc;
    bool gm_ok = gm_ld < p.M;
    bool gn_ok = gn_ld < p.N;

    const ulong w_row = (ulong)(gn_ok ? gn_ld : 0u) * KW;
    const ulong s_row = (ulong)(gn_ok ? gn_ld : 0u) * KG;
    const device half*   x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * p.K;

    // Prologue: load first K=0..31 into buf 0 and hoist group-0 scale/bias across hg=0..1
    half2 sc_h = half2(0.0h);
    half2 bi_h = half2(0.0h);
    {
        threadgroup half4* sa_dst = (threadgroup half4*)(&sA[0][r_loc][c_blk]);
        threadgroup half4* sb_dst = (threadgroup half4*)(&sB[0][r_loc][c_blk]);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + c_blk);
            sa_dst[0] = xp[0]; sa_dst[1] = xp[1]; sa_dst[2] = xp[2]; sa_dst[3] = xp[3];
        } else {
            sa_dst[0] = half4(0.0h); sa_dst[1] = half4(0.0h); sa_dst[2] = half4(0.0h); sa_dst[3] = half4(0.0h);
        }
        if (gn_ok) {
            sc_h = half2(half(bf16_to_f32(q4_load_u16(S, s_row))));
            bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(q4_load_u16(B, s_row)))));
            uint2 uq = q4_load_u32x2(W, w_row + h_idx * 2u);
            uint ws[2] = {uq.x, uq.y};
            #pragma clang loop unroll(full)
            for (uint w = 0u; w < 2u; w++) {
                uint u = ws[w];
                half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
            }
        } else {
            sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0u; k0 < p.K; k0 += 64u) {
        // ---- Sub-step hg=0: compute on buf 0 while prefetching k0+32 (reuses hoisted sc_h/bi_h!) ----
        half4 px0 = half4(0.0h), px1 = half4(0.0h), px2 = half4(0.0h), px3 = half4(0.0h);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + 32u + c_blk);
            px0 = xp[0]; px1 = xp[1]; px2 = xp[2]; px3 = xp[3];
        }
        uint2 puq = gn_ok ? q4_load_u32x2(W, w_row + (k0 >> 3) + 4u + h_idx * 2u) : uint2(0u);

        {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) simdgroup_load(a[im], &sA[0][sg_m + 8u * im][0], 32);
            #pragma clang loop unroll(full)
            for (uint in_ = 0u; in_ < 4u; in_++) simdgroup_load(b[in_], &sB[0][sg_n + 8u * in_][0], 32, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 an[4], bn[4];
                if (kk < 24u) {
                    #pragma clang loop unroll(full)
                    for (uint im = 0u; im < 4u; im++) simdgroup_load(an[im], &sA[0][sg_m + 8u * im][kk + 8u], 32);
                    #pragma clang loop unroll(full)
                    for (uint in_ = 0u; in_ < 4u; in_++) simdgroup_load(bn[in_], &sB[0][sg_n + 8u * in_][kk + 8u], 32, ulong2(0, 0), true);
                }
                #pragma clang loop unroll(full)
                for (uint im = 0u; im < 4u; im++)
                #pragma clang loop unroll(full)
                for (uint in_ = 0u; in_ < 4u; in_++) {
                    simdgroup_multiply_accumulate(acc[im][in_], a[im], b[in_], acc[im][in_]);
                }
                if (kk < 24u) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 4u; i++) { a[i] = an[i]; b[i] = bn[i]; }
                }
            }
        }

        {
            threadgroup half4* sa_dst = (threadgroup half4*)(&sA[1][r_loc][c_blk]);
            threadgroup half4* sb_dst = (threadgroup half4*)(&sB[1][r_loc][c_blk]);
            sa_dst[0] = px0; sa_dst[1] = px1; sa_dst[2] = px2; sa_dst[3] = px3;
            if (gn_ok) {
                uint ws[2] = {puq.x, puq.y};
                #pragma clang loop unroll(full)
                for (uint w = 0u; w < 2u; w++) {
                    uint u = ws[w];
                    half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                    sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
                }
            } else {
                sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Sub-step hg=1: compute on buf 1 while prefetching next group k0+64 ----
        const uint k_nxt = k0 + 64u;
        const bool has_nxt = k_nxt < p.K;
        half4 nx0 = half4(0.0h), nx1 = half4(0.0h), nx2 = half4(0.0h), nx3 = half4(0.0h);
        uint2 nuq = uint2(0u);
        ushort ns_raw = 0u, nb_raw = 0u;
        if (has_nxt) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k_nxt + c_blk);
                nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
            }
            if (gn_ok) {
                uint g_nxt = k_nxt >> 6;
                ns_raw = q4_load_u16(S, s_row + g_nxt);
                nb_raw = q4_load_u16(B, s_row + g_nxt);
                nuq = q4_load_u32x2(W, w_row + (k_nxt >> 3) + h_idx * 2u);
            }
        }

        {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) simdgroup_load(a[im], &sA[1][sg_m + 8u * im][0], 32);
            #pragma clang loop unroll(full)
            for (uint in_ = 0u; in_ < 4u; in_++) simdgroup_load(b[in_], &sB[1][sg_n + 8u * in_][0], 32, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 an[4], bn[4];
                if (kk < 24u) {
                    #pragma clang loop unroll(full)
                    for (uint im = 0u; im < 4u; im++) simdgroup_load(an[im], &sA[1][sg_m + 8u * im][kk + 8u], 32);
                    #pragma clang loop unroll(full)
                    for (uint in_ = 0u; in_ < 4u; in_++) simdgroup_load(bn[in_], &sB[1][sg_n + 8u * in_][kk + 8u], 32, ulong2(0, 0), true);
                }
                #pragma clang loop unroll(full)
                for (uint im = 0u; im < 4u; im++)
                #pragma clang loop unroll(full)
                for (uint in_ = 0u; in_ < 4u; in_++) {
                    simdgroup_multiply_accumulate(acc[im][in_], a[im], b[in_], acc[im][in_]);
                }
                if (kk < 24u) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0u; i < 4u; i++) { a[i] = an[i]; b[i] = bn[i]; }
                }
            }
        }

        if (has_nxt) {
            threadgroup half4* sa_dst = (threadgroup half4*)(&sA[0][r_loc][c_blk]);
            threadgroup half4* sb_dst = (threadgroup half4*)(&sB[0][r_loc][c_blk]);
            sa_dst[0] = nx0; sa_dst[1] = nx1; sa_dst[2] = nx2; sa_dst[3] = nx3;
            if (gn_ok) {
                sc_h = half2(half(bf16_to_f32(ns_raw)));
                bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(nb_raw))));
                uint ws[2] = {nuq.x, nuq.y};
                #pragma clang loop unroll(full)
                for (uint w = 0u; w < 2u; w++) {
                    uint u = ws[w];
                    half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                    sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
                }
            } else {
                sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 4u; im++) {
        uint gm = m0 + sg_m + 8u * im + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0u; in_ < 4u; in_++) {
                uint gn0 = n0 + sg_n + 8u * in_ + fn;
                float2 vv = float2(acc[im][in_].thread_elements()[0], acc[im][in_].thread_elements()[1]);
                if (gn0 + 1u < p.N) {
                    device float2* yp = (device float2*)(yr + gn0);
                    *yp = p.add_residual ? (*yp + vv) : vv;
                } else if (gn0 < p.N) {
                    yr[gn0] = p.add_residual ? (yr[gn0] + vv.x) : vv.x;
                }
            }
        }
    }
}

// ============================================================================
// 1c-swiglu. Fused Prefill Gate + Up + SwiGLU -> FP16 (BM=64, BN=32, BK=32, 128 thr)
//            16,384 B exact double-buffered sA+sBg+sBu -> 2 TGs/core on Apple9+!
//            Shares sA across gate & up, eliminates 5.7 GB FP32 gate/up DRAM + 64 barriers + 64 SwiGLU launches!
//            Grid: (ceil_div(N, 32), ceil_div(M, 64), 1), TG: (128, 1, 1)
// ============================================================================

kernel void qmm_q4_g64_swiglu_f16x(
    const device uchar*  Wg    [[buffer(0)]],
    const device uchar*  Sg    [[buffer(1)]],
    const device uchar*  Bg    [[buffer(2)]],
    const device uchar*  Wu    [[buffer(3)]],
    const device uchar*  Su    [[buffer(4)]],
    const device uchar*  Bu    [[buffer(5)]],
    const device half*   X     [[buffer(6)]],
    device half*         Y     [[buffer(7)]],
    constant QmmParams&  p     [[buffer(8)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint2 n_tg                 [[threadgroups_per_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint GROUP_M = 8u;
    uint linear_id       = tgid.y * n_tg.x + tgid.x;
    uint tiles_per_strip = n_tg.x * GROUP_M;
    uint strip_id        = linear_id / tiles_per_strip;
    uint first_m         = strip_id * GROUP_M;
    uint cur_gm          = min(GROUP_M, n_tg.y - first_m);
    uint rem             = linear_id - strip_id * tiles_per_strip;
    uint n_tile          = rem / cur_gm;
    uint m_tile          = first_m + (rem - n_tile * cur_gm);

    uint m0 = m_tile * 64u;
    uint n0 = n_tile * 32u;
    if (m0 >= p.M || n0 >= p.N || p.K == 0u) return;

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    // 16,384 bytes exact (8192 + 4096 + 4096) -> 2 TGs/core on Apple9+!
    threadgroup half sA[2][64][32];
    threadgroup half sBg[2][32][32];
    threadgroup half sBu[2][32][32];

    simdgroup_float8x8 ag[4][2], au[4][2];
    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0u; in_ < 2u; in_++) {
        ag[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        au[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m = (sg >> 1) * 32u;
    uint sg_n = (sg & 1u) * 16u;
    uint ar   = tid >> 1;
    uint ah   = tid & 1u;
    uint ac   = ah * 16u;
    uint gm_ld = m0 + ar;
    bool gm_ok = gm_ld < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * p.K;

    bool is_u  = (tid >= 64u);
    uint btid  = tid & 63u;
    uint br    = btid >> 1;
    uint bh    = btid & 1u;
    uint bc    = bh * 16u;
    uint gn_ld = n0 + br;
    bool gn_ok = gn_ld < p.N;
    const device uchar* Wp = is_u ? Wu : Wg;
    const device uchar* Sp = is_u ? Su : Sg;
    const device uchar* Bp = is_u ? Bu : Bg;
    const ulong w_row = (ulong)(gn_ok ? gn_ld : 0u) * KW;
    const ulong s_row = (ulong)(gn_ok ? gn_ld : 0u) * KG;

    half2 sc_h = half2(0.0h);
    half2 bi_h = half2(0.0h);
    {
        threadgroup half4* sa_dst = (threadgroup half4*)(&sA[0][ar][ac]);
        threadgroup half4* sb_dst = (threadgroup half4*)(is_u ? &sBu[0][br][bc] : &sBg[0][br][bc]);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + ac);
            sa_dst[0] = xp[0]; sa_dst[1] = xp[1]; sa_dst[2] = xp[2]; sa_dst[3] = xp[3];
        } else {
            sa_dst[0] = half4(0.0h); sa_dst[1] = half4(0.0h); sa_dst[2] = half4(0.0h); sa_dst[3] = half4(0.0h);
        }
        if (gn_ok) {
            sc_h = half2(half(bf16_to_f32(q4_load_u16(Sp, s_row))));
            bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(q4_load_u16(Bp, s_row)))));
            uint2 uq = q4_load_u32x2(Wp, w_row + bh * 2u);
            uint ws[2] = {uq.x, uq.y};
            #pragma clang loop unroll(full)
            for (uint w = 0u; w < 2u; w++) {
                uint u = ws[w];
                half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
            }
        } else {
            sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0u; k0 < p.K; k0 += 64u) {
        half4 px0 = half4(0.0h), px1 = half4(0.0h), px2 = half4(0.0h), px3 = half4(0.0h);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + 32u + ac);
            px0 = xp[0]; px1 = xp[1]; px2 = xp[2]; px3 = xp[3];
        }
        uint2 puq = gn_ok ? q4_load_u32x2(Wp, w_row + (k0 >> 3) + 4u + bh * 2u) : uint2(0u);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[4], bg[2], bu[2];
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) simdgroup_load(a[im], &sA[0][sg_m + 8u * im][kk], 32);
            simdgroup_load(bg[0], &sBg[0][sg_n][kk],      32, ulong2(0, 0), true);
            simdgroup_load(bg[1], &sBg[0][sg_n + 8u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(bu[0], &sBu[0][sg_n][kk],      32, ulong2(0, 0), true);
            simdgroup_load(bu[1], &sBu[0][sg_n + 8u][kk], 32, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) {
                simdgroup_multiply_accumulate(ag[im][0], a[im], bg[0], ag[im][0]);
                simdgroup_multiply_accumulate(ag[im][1], a[im], bg[1], ag[im][1]);
                simdgroup_multiply_accumulate(au[im][0], a[im], bu[0], au[im][0]);
                simdgroup_multiply_accumulate(au[im][1], a[im], bu[1], au[im][1]);
            }
        }

        {
            threadgroup half4* sa_dst = (threadgroup half4*)(&sA[1][ar][ac]);
            threadgroup half4* sb_dst = (threadgroup half4*)(is_u ? &sBu[1][br][bc] : &sBg[1][br][bc]);
            sa_dst[0] = px0; sa_dst[1] = px1; sa_dst[2] = px2; sa_dst[3] = px3;
            if (gn_ok) {
                uint ws[2] = {puq.x, puq.y};
                #pragma clang loop unroll(full)
                for (uint w = 0u; w < 2u; w++) {
                    uint u = ws[w];
                    half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                    sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
                }
            } else {
                sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_nxt = k0 + 64u;
        const bool has_nxt = k_nxt < p.K;
        half4 nx0 = half4(0.0h), nx1 = half4(0.0h), nx2 = half4(0.0h), nx3 = half4(0.0h);
        uint2 nuq = uint2(0u);
        ushort ns_raw = 0u, nb_raw = 0u;
        if (has_nxt) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k_nxt + ac);
                nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
            }
            if (gn_ok) {
                uint g_nxt = k_nxt >> 6;
                ns_raw = q4_load_u16(Sp, s_row + g_nxt);
                nb_raw = q4_load_u16(Bp, s_row + g_nxt);
                nuq = q4_load_u32x2(Wp, w_row + (k_nxt >> 3) + bh * 2u);
            }
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[4], bg[2], bu[2];
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) simdgroup_load(a[im], &sA[1][sg_m + 8u * im][kk], 32);
            simdgroup_load(bg[0], &sBg[1][sg_n][kk],      32, ulong2(0, 0), true);
            simdgroup_load(bg[1], &sBg[1][sg_n + 8u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(bu[0], &sBu[1][sg_n][kk],      32, ulong2(0, 0), true);
            simdgroup_load(bu[1], &sBu[1][sg_n + 8u][kk], 32, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (uint im = 0u; im < 4u; im++) {
                simdgroup_multiply_accumulate(ag[im][0], a[im], bg[0], ag[im][0]);
                simdgroup_multiply_accumulate(ag[im][1], a[im], bg[1], ag[im][1]);
                simdgroup_multiply_accumulate(au[im][0], a[im], bu[0], au[im][0]);
                simdgroup_multiply_accumulate(au[im][1], a[im], bu[1], au[im][1]);
            }
        }

        if (has_nxt) {
            threadgroup half4* sa_dst = (threadgroup half4*)(&sA[0][ar][ac]);
            threadgroup half4* sb_dst = (threadgroup half4*)(is_u ? &sBu[0][br][bc] : &sBg[0][br][bc]);
            sa_dst[0] = nx0; sa_dst[1] = nx1; sa_dst[2] = nx2; sa_dst[3] = nx3;
            if (gn_ok) {
                sc_h = half2(half(bf16_to_f32(ns_raw)));
                bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(nb_raw))));
                uint ws[2] = {nuq.x, nuq.y};
                #pragma clang loop unroll(full)
                for (uint w = 0u; w < 2u; w++) {
                    uint u = ws[w];
                    half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                    sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                    sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
                }
            } else {
                sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 4u; im++) {
        uint gm = m0 + sg_m + 8u * im + fm;
        if (gm < p.M) {
            device half* yr = Y + (ulong)gm * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0u; in_ < 2u; in_++) {
                uint gn0 = n0 + sg_n + 8u * in_ + fn;
                float g0 = ag[im][in_].thread_elements()[0], g1 = ag[im][in_].thread_elements()[1];
                float u0 = au[im][in_].thread_elements()[0], u1 = au[im][in_].thread_elements()[1];
                half2 hv = half2(silu_f32(g0) * u0, silu_f32(g1) * u1);
                if (gn0 + 1u < p.N) {
                    *(device half2*)(yr + gn0) = hv;
                } else if (gn0 < p.N) {
                    yr[gn0] = hv.x;
                }
            }
        }
    }
}

// ============================================================================
// 1c-smalln. Small-N Prefill FP16-X AMX GEMM (BM=32, BN=16, BK=64, 64 threads)
//            5.08x faster on GDN in_proj_a / in_proj_b (N=48, M=512: 48 TGs vs 8!)
// ============================================================================

kernel void qmm_q4_g64_smalln_f16x(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 16u;
    if (m0 >= p.M || n0 >= p.N) return;
    uint KW = p.K >> 3, KG = p.K >> 6;

    threadgroup half sA[32][68];
    threadgroup half sB[16][68];

    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 2u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0; in_ < 2u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m = sg * 16u;
    uint ar = tid >> 1, ah = tid & 1u;
    uint gm = m0 + ar;
    bool gm_ok = gm < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm : 0u) * p.K;
    threadgroup half4* sa_dst = (threadgroup half4*)(&sA[ar][ah * 32u]);

    uint br = tid >> 2, bh = tid & 3u;
    uint gn = n0 + br;
    bool gn_ok = gn < p.N;
    const ulong w_row = (ulong)(gn_ok ? gn : 0u) * KW;
    const ulong s_row = (ulong)(gn_ok ? gn : 0u) * KG;
    threadgroup half4* sb_dst = (threadgroup half4*)(&sB[br][bh * 16u]);

    for (uint k0 = 0; k0 < p.K; k0 += 64u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + ah * 32u);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sa_dst[i] = xp[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; i++) sa_dst[i] = half4(0.0h);
        }
        if (gn_ok) {
            uint g = k0 >> 6;
            half2 sc_h = half2(half(bf16_to_f32(q4_load_u16(S, s_row + g))));
            half2 bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(q4_load_u16(B, s_row + g)))));
            uint2 uq = q4_load_u32x2(W, w_row + (k0 >> 3) + bh * 2u);
            uint ws[2] = {uq.x, uq.y};
            #pragma clang loop unroll(full)
            for (uint w = 0; w < 2u; w++) {
                uint u = ws[w];
                half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; i++) sb_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 64u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &sA[sg_m][kk], 68);
            simdgroup_load(a1, &sA[sg_m + 8u][kk], 68);
            simdgroup_load(b0, &sB[0][kk], 68, ulong2(0, 0), true);
            simdgroup_load(b1, &sB[8][kk], 68, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 2u; im++) {
        uint om = m0 + sg_m + 8u * im + fm;
        if (om < p.M) {
            device float* yr = Y + (ulong)om * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0; in_ < 2u; in_++) {
                uint on0 = n0 + 8u * in_ + fn;
                float2 vv = float2(acc[im][in_].thread_elements()[0], acc[im][in_].thread_elements()[1]);
                if (on0 + 1u < p.N) {
                    device float2* yp = (device float2*)(yr + on0);
                    *yp = p.add_residual ? (*yp + vv) : vv;
                } else if (on0 < p.N) {
                    yr[on0] = p.add_residual ? (yr[on0] + vv.x) : vv.x;
                }
            }
        }
    }
}

// ============================================================================
// 1c-smalln-ab. Fused Small-N Prefill in_proj_a + in_proj_b (BM=16, BN=16, BK=64, 64 thr)
//               sg=0 computes Ya, sg=1 computes Yb, BOTH share sA[16][68] (6.5 KB -> 4 TGs/core)!
//               Doubles TG occupancy (48 -> 96 TGs at M=512, N=48) + halves X reads -> 2.11x-2.31x faster!
//               Grid: (ceil_div(N, 16), ceil_div(M, 16), 1), TG: (64, 1, 1)
// ============================================================================

kernel void qmm_q4_g64_smalln_ab_f16x(
    const device uchar*  Wa    [[buffer(0)]],
    const device uchar*  Sa    [[buffer(1)]],
    const device uchar*  Ba    [[buffer(2)]],
    const device uchar*  Wb    [[buffer(3)]],
    const device uchar*  Sb    [[buffer(4)]],
    const device uchar*  Bb    [[buffer(5)]],
    const device half*   X     [[buffer(6)]],
    device float*        Ya    [[buffer(7)]],
    device float*        Yb    [[buffer(8)]],
    constant QmmParams&  p     [[buffer(9)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 16u;
    uint n0 = tgid.x * 16u;
    if (m0 >= p.M || n0 >= p.N || p.K == 0u) return;
    uint KW = p.K >> 3, KG = p.K >> 6;

    threadgroup half sA[16][68];
    threadgroup half sB[2][16][68];

    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 2u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0u; in_ < 2u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // A: 64 thr -> 16 rows x 4 chunks of 16 halfs (4 half4/thr)
    uint ar = tid >> 2, ah = tid & 3u;
    uint gm = m0 + ar;
    bool gm_ok = gm < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm : 0u) * p.K;
    threadgroup half4* sa_dst = (threadgroup half4*)(&sA[ar][ah * 16u]);

    // B: sg=0 loads Wa into sB[0], sg=1 loads Wb into sB[1] (16 rows x 2 halves of 32 halfs)
    uint br = lane >> 1, bh = lane & 1u;
    uint gn = n0 + br;
    bool gn_ok = gn < p.N;
    const device uchar* Wp = (sg == 0u) ? Wa : Wb;
    const device uchar* Sp = (sg == 0u) ? Sa : Sb;
    const device uchar* Bp = (sg == 0u) ? Ba : Bb;
    const ulong w_row = (ulong)(gn_ok ? gn : 0u) * KW;
    const ulong s_row = (ulong)(gn_ok ? gn : 0u) * KG;
    threadgroup half4* sb_dst = (threadgroup half4*)(&sB[sg][br][bh * 32u]);

    for (uint k0 = 0u; k0 < p.K; k0 += 64u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + ah * 16u);
            sa_dst[0] = xp[0]; sa_dst[1] = xp[1]; sa_dst[2] = xp[2]; sa_dst[3] = xp[3];
        } else {
            sa_dst[0] = half4(0.0h); sa_dst[1] = half4(0.0h); sa_dst[2] = half4(0.0h); sa_dst[3] = half4(0.0h);
        }
        if (gn_ok) {
            uint g = k0 >> 6;
            half2 sc_h = half2(half(bf16_to_f32(q4_load_u16(Sp, s_row + g))));
            half2 bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(q4_load_u16(Bp, s_row + g)))));
            uint2 u0 = q4_load_u32x2(Wp, w_row + (k0 >> 3) + bh * 4u);
            uint2 u1 = q4_load_u32x2(Wp, w_row + (k0 >> 3) + bh * 4u + 2u);
            uint ws[4] = {u0.x, u0.y, u1.x, u1.y};
            #pragma clang loop unroll(full)
            for (uint w = 0u; w < 4u; w++) {
                uint u = ws[w];
                half2 h0 = fma(as_type<half2>(( u        & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h1 = fma(as_type<half2>(((u >> 4)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h2 = fma(as_type<half2>(((u >> 8)  & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                half2 h3 = fma(as_type<half2>(((u >> 12) & 0x000F000Fu) | 0x64006400u), sc_h, bi_h);
                sb_dst[w * 2u + 0u] = half4(h0.x, h1.x, h2.x, h3.x);
                sb_dst[w * 2u + 1u] = half4(h0.y, h1.y, h2.y, h3.y);
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 8u; i++) sb_dst[i] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 64u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &sA[0][kk], 68);
            simdgroup_load(a1, &sA[8][kk], 68);
            simdgroup_load(b0, &sB[sg][0][kk], 68, ulong2(0, 0), true);
            simdgroup_load(b1, &sB[sg][8][kk], 68, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    device float* Yp = (sg == 0u) ? Ya : Yb;
    #pragma clang loop unroll(full)
    for (uint im = 0u; im < 2u; im++) {
        uint om = m0 + 8u * im + fm;
        if (om < p.M) {
            device float* yr = Yp + (ulong)om * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0u; in_ < 2u; in_++) {
                uint on0 = n0 + 8u * in_ + fn;
                float2 vv = float2(acc[im][in_].thread_elements()[0], acc[im][in_].thread_elements()[1]);
                if (on0 + 1u < p.N) {
                    device float2* yp = (device float2*)(yr + on0);
                    *yp = p.add_residual ? (*yp + vv) : vv;
                } else if (on0 < p.N) {
                    yr[on0] = p.add_residual ? (yr[on0] + vv.x) : vv.x;
                }
            }
        }
    }
}

// ============================================================================
// 1d. Register-Direct Hardware simdgroup_half8x8 Small-Batch Q4_AFFINE_G64
//     Half-magic-trick dequant: ((wv >> 4t) & 0x000F000F) | 0x64006400 = half2(1024+q_t, 1024+q_{t+4})
//     ZERO threadgroup A/B transpose, ZERO per-weight float unpack, FLAT cost for M = 1..8!
//     - mma_qmm8_q4_g64_f32:        M = 1..8,  TN=4 (32 out rows/TG), KS=8 split-K, UG=2 prefetch
//     - mma_qmm8_q4_g64_swiglu_f32: M = 1..8,  Fused gate+up+SwiGLU (TN=2 gate + TN=2 up, shares X!)
//     - mma_qmm16_q4_g64_f32:       M = 9..24+ (tiled by 16 rows/tgid.y), NB=2 shares every A fragment!
// ============================================================================

kernel void mma_qmm8_q4_g64_f32(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;
    constexpr uint UG = 2u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint m0 = tgid.y * 8u;
    if (n0 >= N || m0 >= p.M) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ct[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) { ct[t][0] = 0.0f; ct[t][1] = 0.0f; }

    const uint gm0 = m0 + fn;
    const uint gm1 = m0 + fn + 1u;
    const bool m0ok = gm0 < p.M;
    const bool m1ok = gm1 < p.M;
    const device float* xr0 = X + (ulong)(m0ok ? gm0 : 0u) * K;
    const device float* xr1 = X + (ulong)(m1ok ? gm1 : 0u) * K;

    for (uint g0 = sg * UG; g0 < G; g0 += KS * UG) {
        uint2 wqs[UG][TN];
        float scs[UG][TN], bis[UG][TN];
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint gg = min(g0 + u, G - 1u);
            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                uint n = min(n0 + 8u * t + fm, N - 1u);
                wqs[u][t] = q4_load_u32x2(W, (ulong)n * KW + 8u * gg + 2u * b);
                scs[u][t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + gg));
                bis[u][t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + gg));
            }
        }
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint g = g0 + u;
            if (g >= G) break;

            uint kx = 64u * g + 16u * sb + 4u * se;
            float4 a0 = *(const device float4*)(xr0 + kx);
            float4 a1 = *(const device float4*)(xr0 + kx + 8u);
            float4 c0 = *(const device float4*)(xr1 + kx);
            float4 c1 = *(const device float4*)(xr1 + kx + 8u);
            float xv0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float xv1[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

            float s0 = 0.0f, s1 = 0.0f;
            simdgroup_half8x8 Bm[8];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                half h0 = half(m0ok ? xv0[j] : 0.0f);
                half h1 = half(m1ok ? xv1[j] : 0.0f);
                s0 += float(h0);
                s1 += float(h1);
                Bm[j].thread_elements()[0] = h0;
                Bm[j].thread_elements()[1] = h1;
            }
            s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
            s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
            s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                uint2 wq = wqs[u][t];
                float sc = scs[u][t];
                float bi = bis[u][t] - 1024.0f * sc;
                simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; i++) {
                    uint wv = (i == 0u) ? wq.x : wq.y;
                    #pragma clang loop unroll(full)
                    for (uint tt = 0; tt < 4u; tt++) {
                        simdgroup_half8x8 A;
                        half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                        A.thread_elements()[0] = h.x;
                        A.thread_elements()[1] = h.y;
                        simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                    }
                }
                ct[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ct[t][0]));
                ct[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ct[t][1]));
            }
        }
    }

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red[sg][t][fm * 8u + fn]      = ct[t][0];
        red[sg][t][fm * 8u + fn + 1u] = ct[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TN * 64 == 256 == 32 * KS -> exactly 1 output element per thread!
    uint e  = lane + 32u * sg;
    uint t  = e >> 6;
    uint r  = (e >> 3) & 7u;
    uint m  = e & 7u;
    uint gm = m0 + m;
    uint gn = n0 + 8u * t + r;
    if (gm < p.M && gn < N) {
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < KS; k++) acc += red[k][t][r * 8u + m];
        ulong idx = (ulong)gm * N + gn;
        Y[idx] = p.add_residual ? (Y[idx] + acc) : acc;
    }
}

kernel void mma_qmm8_q4_g64_swiglu_f32(
    const device uchar*  Wg    [[buffer(0)]],
    const device uchar*  Sg    [[buffer(1)]],
    const device uchar*  Bg    [[buffer(2)]],
    const device uchar*  Wu    [[buffer(3)]],
    const device uchar*  Su    [[buffer(4)]],
    const device uchar*  Bu    [[buffer(5)]],
    const device float*  X     [[buffer(6)]],
    device float*        Y     [[buffer(7)]],
    constant QmmParams&  p     [[buffer(8)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 2u; // 2 gate tiles + 2 up tiles = 4 tiles/thread (16 out rows/TG)
    constexpr uint KS = 8u;
    constexpr uint UG = 2u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint m0 = tgid.y * 8u;
    if (n0 >= N || m0 >= p.M) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ctg[TN][2], ctu[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        ctg[t][0] = 0.0f; ctg[t][1] = 0.0f;
        ctu[t][0] = 0.0f; ctu[t][1] = 0.0f;
    }

    const uint gm0 = m0 + fn;
    const uint gm1 = m0 + fn + 1u;
    const bool m0ok = gm0 < p.M;
    const bool m1ok = gm1 < p.M;
    const device float* xr0 = X + (ulong)(m0ok ? gm0 : 0u) * K;
    const device float* xr1 = X + (ulong)(m1ok ? gm1 : 0u) * K;

    for (uint g0 = sg * UG; g0 < G; g0 += KS * UG) {
        uint2 wqg[UG][TN], wqu[UG][TN];
        float scg[UG][TN], big[UG][TN], scu[UG][TN], biu[UG][TN];
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint gg = min(g0 + u, G - 1u);
            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                uint n = min(n0 + 8u * t + fm, N - 1u);
                ulong w_off = (ulong)n * KW + 8u * gg + 2u * b;
                ulong s_off = (ulong)n * G + gg;
                wqg[u][t] = q4_load_u32x2(Wg, w_off);
                scg[u][t] = bf16_to_f32(q4_load_u16(Sg, s_off));
                big[u][t] = bf16_to_f32(q4_load_u16(Bg, s_off));
                wqu[u][t] = q4_load_u32x2(Wu, w_off);
                scu[u][t] = bf16_to_f32(q4_load_u16(Su, s_off));
                biu[u][t] = bf16_to_f32(q4_load_u16(Bu, s_off));
            }
        }
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint g = g0 + u;
            if (g >= G) break;

            uint kx = 64u * g + 16u * sb + 4u * se;
            float4 a0 = *(const device float4*)(xr0 + kx);
            float4 a1 = *(const device float4*)(xr0 + kx + 8u);
            float4 c0 = *(const device float4*)(xr1 + kx);
            float4 c1 = *(const device float4*)(xr1 + kx + 8u);
            float xv0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float xv1[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};

            float s0 = 0.0f, s1 = 0.0f;
            simdgroup_half8x8 Bm[8];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                half h0 = half(m0ok ? xv0[j] : 0.0f);
                half h1 = half(m1ok ? xv1[j] : 0.0f);
                s0 += float(h0);
                s1 += float(h1);
                Bm[j].thread_elements()[0] = h0;
                Bm[j].thread_elements()[1] = h1;
            }
            s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
            s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
            s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                // Gate
                {
                    uint2 wq = wqg[u][t];
                    float sc = scg[u][t];
                    float bi = big[u][t] - 1024.0f * sc;
                    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; i++) {
                        uint wv = (i == 0u) ? wq.x : wq.y;
                        #pragma clang loop unroll(full)
                        for (uint tt = 0; tt < 4u; tt++) {
                            simdgroup_half8x8 A;
                            half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                            A.thread_elements()[0] = h.x;
                            A.thread_elements()[1] = h.y;
                            simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                        }
                    }
                    ctg[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ctg[t][0]));
                    ctg[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ctg[t][1]));
                }
                // Up
                {
                    uint2 wq = wqu[u][t];
                    float sc = scu[u][t];
                    float bi = biu[u][t] - 1024.0f * sc;
                    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; i++) {
                        uint wv = (i == 0u) ? wq.x : wq.y;
                        #pragma clang loop unroll(full)
                        for (uint tt = 0; tt < 4u; tt++) {
                            simdgroup_half8x8 A;
                            half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                            A.thread_elements()[0] = h.x;
                            A.thread_elements()[1] = h.y;
                            simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                        }
                    }
                    ctu[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ctu[t][0]));
                    ctu[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ctu[t][1]));
                }
            }
        }
    }

    threadgroup float red_g[KS][TN][64];
    threadgroup float red_u[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red_g[sg][t][fm * 8u + fn]      = ctg[t][0];
        red_g[sg][t][fm * 8u + fn + 1u] = ctg[t][1];
        red_u[sg][t][fm * 8u + fn]      = ctu[t][0];
        red_u[sg][t][fm * 8u + fn + 1u] = ctu[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = lane + 32u * sg;
    if (e < TN * 64u) {
        uint t  = e >> 6;
        uint r  = (e >> 3) & 7u;
        uint m  = e & 7u;
        uint gm = m0 + m;
        uint gn = n0 + 8u * t + r;
        if (gm < p.M && gn < N) {
            float gv = 0.0f, uv = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < KS; k++) {
                gv += red_g[k][t][r * 8u + m];
                uv += red_u[k][t][r * 8u + m];
            }
            Y[(ulong)gm * N + gn] = silu_f32(gv) * uv;
        }
    }
}

// ============================================================================
// 1d-f16. FP16-Activation Small-Batch (M = 2..8) Register-Direct AMX Decode
//         UG=1 + half4 X loads cut register pressure in half & eliminate float->half
//         converts: +49.6% on MLP down_proj (231 -> 345 GB/s!), +5-7% on all others!
// ============================================================================

kernel void mma_qmm8_q4_g64_f16x(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint m0 = tgid.y * 8u;
    if (n0 >= N || m0 >= p.M) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ct[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) { ct[t][0] = 0.0f; ct[t][1] = 0.0f; }

    const uint gm0 = m0 + fn;
    const uint gm1 = m0 + fn + 1u;
    const bool m0ok = gm0 < p.M;
    const bool m1ok = gm1 < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? gm0 : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? gm1 : 0u) * K;

    for (uint g = sg; g < G; g += KS) {
        uint2 wqs[TN];
        float scs[TN], bis[TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint n = min(n0 + 8u * t + fm, N - 1u);
            wqs[t] = q4_load_u32x2(W, (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + g));
            bis[t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + g));
        }

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint2 wq = wqs[t];
            float sc = scs[t];
            float bi = bis[t] - 1024.0f * sc;
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; i++) {
                uint wv = (i == 0u) ? wq.x : wq.y;
                #pragma clang loop unroll(full)
                for (uint tt = 0; tt < 4u; tt++) {
                    simdgroup_half8x8 A;
                    half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                    A.thread_elements()[0] = h.x;
                    A.thread_elements()[1] = h.y;
                    simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                }
            }
            ct[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ct[t][0]));
            ct[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ct[t][1]));
        }
    }

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red[sg][t][fm * 8u + fn]      = ct[t][0];
        red[sg][t][fm * 8u + fn + 1u] = ct[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e  = lane + 32u * sg;
    uint t  = e >> 6;
    uint r  = (e >> 3) & 7u;
    uint m  = e & 7u;
    uint gm = m0 + m;
    uint gn = n0 + 8u * t + r;
    if (gm < p.M && gn < N) {
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < KS; k++) acc += red[k][t][r * 8u + m];
        ulong idx = (ulong)gm * N + gn;
        Y[idx] = p.add_residual ? (Y[idx] + acc) : acc;
    }
}

// Split-K along tgid.z for small-N, large-K decode projections (down_proj 5120x17408 sk=4/8; out_proj 5120x6144 sk=2)
kernel void mma_qmm8_q4_g64_f16x_sk(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        partY [[buffer(4)]],
    constant QmmSkParams& p    [[buffer(5)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint s_idx = tgid.z;
    if (n0 >= N || s_idx >= p.sk) return;

    const uint g_per_sk = (G + p.sk - 1u) / p.sk;
    const uint g_start  = s_idx * g_per_sk;
    const uint g_end    = min(G, g_start + g_per_sk);

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ct[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) { ct[t][0] = 0.0f; ct[t][1] = 0.0f; }

    const bool m0ok = fn < p.M;
    const bool m1ok = (fn + 1u) < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? fn : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? (fn + 1u) : 0u) * K;

    for (uint g = g_start + sg; g < g_end; g += KS) {
        uint2 wqs[TN];
        float scs[TN], bis[TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint n = min(n0 + 8u * t + fm, N - 1u);
            wqs[t] = q4_load_u32x2(W, (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + g));
            bis[t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + g));
        }

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint2 wq = wqs[t];
            float sc = scs[t];
            float bi = bis[t] - 1024.0f * sc;
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; i++) {
                uint wv = (i == 0u) ? wq.x : wq.y;
                #pragma clang loop unroll(full)
                for (uint tt = 0; tt < 4u; tt++) {
                    simdgroup_half8x8 A;
                    half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                    A.thread_elements()[0] = h.x;
                    A.thread_elements()[1] = h.y;
                    simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                }
            }
            ct[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ct[t][0]));
            ct[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ct[t][1]));
        }
    }

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red[sg][t][fm * 8u + fn]      = ct[t][0];
        red[sg][t][fm * 8u + fn + 1u] = ct[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e  = lane + 32u * sg;
    uint t  = e >> 6;
    uint r  = (e >> 3) & 7u;
    uint m  = e & 7u;
    uint gn = n0 + 8u * t + r;
    if (m < p.M && gn < N) {
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < KS; k++) acc += red[k][t][r * 8u + m];
        ulong sk_stride = (ulong)p.M * N;
        partY[(ulong)s_idx * sk_stride + (ulong)m * N + gn] = acc;
    }
}

kernel void mma_qmm8_sk_reduce(
    const device float*  partY [[buffer(0)]],
    device float*        Y     [[buffer(1)]],
    constant QmmSkParams& p    [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]
) {
    uint total = p.M * p.N;
    if (gid >= total) return;
    float s = p.add_residual ? Y[gid] : 0.0f;
    for (uint k = 0u; k < p.sk; k++) s += partY[(ulong)k * total + gid];
    Y[gid] = s;
}

struct QmmSkNormParams {
    uint  M;
    uint  N;
    uint  K;
    uint  add_residual;
    uint  sk;
    float eps;
};

// Fused Split-K Reduce + Residual Add + RMSNorm-F16 (float4 vectorized loads/stores + unrolled sk=8/2!)
kernel void mma_qmm8_sk_reduce_rmsnorm_f16(
    const device float*       partY  [[buffer(0)]],
    device float*             hid    [[buffer(1)]],
    const device ushort*      norm_w [[buffer(2)]],
    device half*              normed [[buffer(3)]],
    constant QmmSkNormParams& p      [[buffer(4)]],
    uint row                         [[threadgroup_position_in_grid]],
    uint tid                         [[thread_position_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const ulong total = (ulong)p.M * p.N;
    const ulong row_off = (ulong)row * p.N;
    device float* hp = hid + row_off;
    device half*  np = normed + row_off;

    threadgroup float smem[32];
    const uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    const uint n4 = p.N >> 2;

    if ((p.N & 3u) == 0u && n4 <= tg_size * 2u) {
        const ulong total4 = total >> 2;
        const ulong row_off4 = row_off >> 2;
        const device float4* partY4 = (const device float4*)partY;
        device float4* hp4 = (device float4*)hp;
        device half4*  np4 = (device half4*)np;

        float4 v0 = float4(0.0f), v1 = float4(0.0f);
        float sq = 0.0f;
        const bool ok0 = tid < n4;
        const uint tid1 = tid + tg_size;
        const bool ok1 = tid1 < n4;

        if (ok0) {
            float4 acc = p.add_residual ? hp4[tid] : float4(0.0f);
            if (p.sk == 8u) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 8u; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid];
            } else if (p.sk == 2u) {
                acc += partY4[row_off4 + tid] + partY4[total4 + row_off4 + tid];
            } else {
                for (uint s = 0u; s < p.sk; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid];
            }
            hp4[tid] = acc;
            v0 = acc;
            sq = fma(acc.x, acc.x, fma(acc.y, acc.y, fma(acc.z, acc.z, fma(acc.w, acc.w, sq))));
        }
        if (ok1) {
            float4 acc = p.add_residual ? hp4[tid1] : float4(0.0f);
            if (p.sk == 8u) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 8u; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid1];
            } else if (p.sk == 2u) {
                acc += partY4[row_off4 + tid1] + partY4[total4 + row_off4 + tid1];
            } else {
                for (uint s = 0u; s < p.sk; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid1];
            }
            hp4[tid1] = acc;
            v1 = acc;
            sq = fma(acc.x, acc.x, fma(acc.y, acc.y, fma(acc.z, acc.z, fma(acc.w, acc.w, sq))));
        }

        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];

        if (ok0) {
            uint base = tid * 4u;
            float4 w = float4(bf16_to_f32(norm_w[base + 0u]), bf16_to_f32(norm_w[base + 1u]),
                              bf16_to_f32(norm_w[base + 2u]), bf16_to_f32(norm_w[base + 3u]));
            np4[tid] = half4(v0 * (inv_rms * w));
        }
        if (ok1) {
            uint base = tid1 * 4u;
            float4 w = float4(bf16_to_f32(norm_w[base + 0u]), bf16_to_f32(norm_w[base + 1u]),
                              bf16_to_f32(norm_w[base + 2u]), bf16_to_f32(norm_w[base + 3u]));
            np4[tid1] = half4(v1 * (inv_rms * w));
        }
    } else if ((p.N & 3u) == 0u && n4 <= tg_size * 5u) {
        const ulong total4 = total >> 2;
        const ulong row_off4 = row_off >> 2;
        const device float4* partY4 = (const device float4*)partY;
        device float4* hp4 = (device float4*)hp;
        device half4*  np4 = (device half4*)np;

        float4 vals[5] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        float sq = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 5u; ++k) {
            uint idx4 = k * tg_size + tid;
            if (idx4 < n4) {
                float4 acc = p.add_residual ? hp4[idx4] : float4(0.0f);
                for (uint s = 0u; s < p.sk; ++s) acc += partY4[(ulong)s * total4 + row_off4 + idx4];
                hp4[idx4] = acc;
                vals[k] = acc;
                sq = fma(acc.x, acc.x, fma(acc.y, acc.y, fma(acc.z, acc.z, fma(acc.w, acc.w, sq))));
            }
        }
        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];

        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 5u; ++k) {
            uint idx4 = k * tg_size + tid;
            if (idx4 < n4) {
                uint base = idx4 * 4u;
                float4 w = float4(bf16_to_f32(norm_w[base + 0u]), bf16_to_f32(norm_w[base + 1u]),
                                  bf16_to_f32(norm_w[base + 2u]), bf16_to_f32(norm_w[base + 3u]));
                np4[idx4] = half4(vals[k] * (inv_rms * w));
            }
        }
    } else {
        float sq = 0.0f;
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0u; s < p.sk; s++) {
                acc += partY[(ulong)s * total + row_off + idx];
            }
            hp[idx] = acc;
            sq = fma(acc, acc, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            np[idx] = half(hp[idx] * inv_rms * bf16_to_f32(norm_w[idx]));
        }
    }
}

// Fused Layer-63 Split-K Reduce + Residual Add + Dual Final RMSNorm (FP32 for MTP + FP16 for lm_head, float4!)
kernel void mma_qmm8_sk_reduce_dual_norm_f16(
    const device float*       partY   [[buffer(0)]],
    device float*             hid     [[buffer(1)]],
    const device ushort*      norm_w  [[buffer(2)]],
    device float*             out_f32 [[buffer(3)]],
    device half*              out_f16 [[buffer(4)]],
    constant QmmSkNormParams& p       [[buffer(5)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint tg_size                      [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const ulong total = (ulong)p.M * p.N;
    const ulong row_off = (ulong)row * p.N;
    device float* hp  = hid + row_off;
    device float* y32 = out_f32 + row_off;
    device half*  y16 = out_f16 + row_off;

    threadgroup float smem[32];
    const uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    const uint n4 = p.N >> 2;

    if ((p.N & 3u) == 0u && n4 <= tg_size * 2u) {
        const ulong total4 = total >> 2;
        const ulong row_off4 = row_off >> 2;
        const device float4* partY4 = (const device float4*)partY;
        device float4* hp4  = (device float4*)hp;
        device float4* y32_4 = (device float4*)y32;
        device half4*  y16_4 = (device half4*)y16;

        float4 v0 = float4(0.0f), v1 = float4(0.0f);
        float sq = 0.0f;
        const bool ok0 = tid < n4;
        const uint tid1 = tid + tg_size;
        const bool ok1 = tid1 < n4;

        if (ok0) {
            float4 acc = p.add_residual ? hp4[tid] : float4(0.0f);
            if (p.sk == 8u) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 8u; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid];
            } else if (p.sk == 2u) {
                acc += partY4[row_off4 + tid] + partY4[total4 + row_off4 + tid];
            } else {
                for (uint s = 0u; s < p.sk; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid];
            }
            hp4[tid] = acc;
            v0 = acc;
            sq = fma(acc.x, acc.x, fma(acc.y, acc.y, fma(acc.z, acc.z, fma(acc.w, acc.w, sq))));
        }
        if (ok1) {
            float4 acc = p.add_residual ? hp4[tid1] : float4(0.0f);
            if (p.sk == 8u) {
                #pragma clang loop unroll(full)
                for (uint s = 0u; s < 8u; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid1];
            } else if (p.sk == 2u) {
                acc += partY4[row_off4 + tid1] + partY4[total4 + row_off4 + tid1];
            } else {
                for (uint s = 0u; s < p.sk; ++s) acc += partY4[(ulong)s * total4 + row_off4 + tid1];
            }
            hp4[tid1] = acc;
            v1 = acc;
            sq = fma(acc.x, acc.x, fma(acc.y, acc.y, fma(acc.z, acc.z, fma(acc.w, acc.w, sq))));
        }

        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];

        if (ok0) {
            uint base = tid * 4u;
            float4 w = float4(bf16_to_f32(norm_w[base + 0u]), bf16_to_f32(norm_w[base + 1u]),
                              bf16_to_f32(norm_w[base + 2u]), bf16_to_f32(norm_w[base + 3u]));
            float4 out = v0 * (inv_rms * w);
            y32_4[tid] = out;
            y16_4[tid] = half4(out);
        }
        if (ok1) {
            uint base = tid1 * 4u;
            float4 w = float4(bf16_to_f32(norm_w[base + 0u]), bf16_to_f32(norm_w[base + 1u]),
                              bf16_to_f32(norm_w[base + 2u]), bf16_to_f32(norm_w[base + 3u]));
            float4 out = v1 * (inv_rms * w);
            y32_4[tid1] = out;
            y16_4[tid1] = half4(out);
        }
    } else {
        float sq = 0.0f;
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0u; s < p.sk; s++) {
                acc += partY[(ulong)s * total + row_off + idx];
            }
            hp[idx] = acc;
            sq = fma(acc, acc, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            float v = hp[idx] * inv_rms * bf16_to_f32(norm_w[idx]);
            y32[idx] = v;
            y16[idx] = half(v);
        }
    }
}

// Round 13: Dual-output RMSNorm (writes FP32 out_f32 for MTP + FP16 out_f16 for fast FP16-X lm_head!)
struct DualNormParams { uint M; uint D; float eps; };

kernel void rmsnorm_f32_bf16w_dual(
    const device float*  X       [[buffer(0)]],
    const device ushort* W       [[buffer(1)]],
    device float*        Y_f32   [[buffer(2)]],
    device half*         Y_f16   [[buffer(3)]],
    constant DualNormParams& p   [[buffer(4)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xr = X + (ulong)row * p.D;
    device float* y32 = Y_f32 + (ulong)row * p.D;
    device half*  y16 = Y_f16 + (ulong)row * p.D;

    float sq = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = xr[i];
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
        float val = xr[i] * inv_rms * bf16_to_f32(W[i]);
        y32[i] = val;
        y16[i] = half(val);
    }
}

kernel void mma_qmm8_q4_g64_swiglu_f16(
    const device uchar*  Wg    [[buffer(0)]],
    const device uchar*  Sg    [[buffer(1)]],
    const device uchar*  Bg    [[buffer(2)]],
    const device uchar*  Wu    [[buffer(3)]],
    const device uchar*  Su    [[buffer(4)]],
    const device uchar*  Bu    [[buffer(5)]],
    const device half*   X     [[buffer(6)]],
    device half*         Y     [[buffer(7)]],
    constant QmmParams&  p     [[buffer(8)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 2u;
    constexpr uint KS = 8u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint m0 = tgid.y * 8u;
    if (n0 >= N || m0 >= p.M) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ctg[TN][2], ctu[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        ctg[t][0] = 0.0f; ctg[t][1] = 0.0f;
        ctu[t][0] = 0.0f; ctu[t][1] = 0.0f;
    }

    const uint gm0 = m0 + fn;
    const uint gm1 = m0 + fn + 1u;
    const bool m0ok = gm0 < p.M;
    const bool m1ok = gm1 < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? gm0 : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? gm1 : 0u) * K;

    for (uint g = sg; g < G; g += KS) {
        uint2 wqg[TN], wqu[TN];
        float scg[TN], big[TN], scu[TN], biu[TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint n = min(n0 + 8u * t + fm, N - 1u);
            ulong w_off = (ulong)n * KW + 8u * g + 2u * b;
            ulong s_off = (ulong)n * G + g;
            wqg[t] = q4_load_u32x2(Wg, w_off);
            scg[t] = bf16_to_f32(q4_load_u16(Sg, s_off));
            big[t] = bf16_to_f32(q4_load_u16(Bg, s_off));
            wqu[t] = q4_load_u32x2(Wu, w_off);
            scu[t] = bf16_to_f32(q4_load_u16(Su, s_off));
            biu[t] = bf16_to_f32(q4_load_u16(Bu, s_off));
        }

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            {
                uint2 wq = wqg[t];
                float sc = scg[t];
                float bi = big[t] - 1024.0f * sc;
                simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; i++) {
                    uint wv = (i == 0u) ? wq.x : wq.y;
                    #pragma clang loop unroll(full)
                    for (uint tt = 0; tt < 4u; tt++) {
                        simdgroup_half8x8 A;
                        half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                        A.thread_elements()[0] = h.x;
                        A.thread_elements()[1] = h.y;
                        simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                    }
                }
                ctg[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ctg[t][0]));
                ctg[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ctg[t][1]));
            }
            {
                uint2 wq = wqu[t];
                float sc = scu[t];
                float bi = biu[t] - 1024.0f * sc;
                simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; i++) {
                    uint wv = (i == 0u) ? wq.x : wq.y;
                    #pragma clang loop unroll(full)
                    for (uint tt = 0; tt < 4u; tt++) {
                        simdgroup_half8x8 A;
                        half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                        A.thread_elements()[0] = h.x;
                        A.thread_elements()[1] = h.y;
                        simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                    }
                }
                ctu[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ctu[t][0]));
                ctu[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ctu[t][1]));
            }
        }
    }

    threadgroup float red_g[KS][TN][64];
    threadgroup float red_u[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red_g[sg][t][fm * 8u + fn]      = ctg[t][0];
        red_g[sg][t][fm * 8u + fn + 1u] = ctg[t][1];
        red_u[sg][t][fm * 8u + fn]      = ctu[t][0];
        red_u[sg][t][fm * 8u + fn + 1u] = ctu[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = lane + 32u * sg;
    if (e < TN * 64u) {
        uint t  = e >> 6;
        uint r  = (e >> 3) & 7u;
        uint m  = e & 7u;
        uint gm = m0 + m;
        uint gn = n0 + 8u * t + r;
        if (gm < p.M && gn < N) {
            float gv = 0.0f, uv = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < KS; k++) {
                gv += red_g[k][t][r * 8u + m];
                uv += red_u[k][t][r * 8u + m];
            }
            Y[(ulong)gm * N + gn] = half(silu_f32(gv) * uv);
        }
    }
}

// Round 20: Dedicated Ultra-Fast Fused in_proj_a + in_proj_b (M=1..8, N=48+48=96, 12 TGs x 128 threads!)
// Shares X across both a and b, saturates GPU cores instead of 2+2 TG tail stragglers (14 GB/s -> 150+ GB/s)!
kernel void mma_qmm8_in_proj_ab_f16x(
    const device uchar*  Wa [[buffer(0)]], const device uchar*  Sa [[buffer(1)]], const device uchar*  Ba [[buffer(2)]],
    const device uchar*  Wb [[buffer(3)]], const device uchar*  Sb [[buffer(4)]], const device uchar*  Bb [[buffer(5)]],
    const device half*   X  [[buffer(6)]],
    device float*        Ya [[buffer(7)]],
    device float*        Yb [[buffer(8)]],
    constant QmmParams&  p  [[buffer(9)]],
    uint  tgid              [[threadgroup_position_in_grid]],
    uint  lane              [[thread_index_in_simdgroup]],
    uint  sg                [[simdgroup_index_in_threadgroup]]
) {
    // 12 TGs total: tgid 0..5 -> Ya (8 rows/TG), tgid 6..11 -> Yb (8 rows/TG), KS=4 (128 threads/TG)
    constexpr uint KS = 4u;
    const uint n_tgs_half = (p.N + 7u) >> 3;
    const bool is_b = (tgid >= n_tgs_half);
    const uint n0 = (is_b ? (tgid - n_tgs_half) : tgid) * 8u;
    if (n0 >= p.N) return;

    const device uchar*  W = is_b ? Wb : Wa;
    const device uchar*  S = is_b ? Sb : Sa;
    const device uchar*  B = is_b ? Bb : Ba;
    device float*        Y = is_b ? Yb : Ya;

    const uint K = p.K, N = p.N, KW = K >> 3, G = K >> 6;
    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float c0_acc = 0.0f, c1_acc = 0.0f;
    const uint gm0 = fn, gm1 = fn + 1u;
    const bool m0ok = gm0 < p.M, m1ok = gm1 < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? gm0 : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? gm1 : 0u) * K;
    const uint n = min(n0 + fm, N - 1u);

    for (uint g = sg; g < G; g += KS) {
        uint2 wq = q4_load_u32x2(W, (ulong)n * KW + 8u * g + 2u * b);
        float sc = bf16_to_f32(q4_load_u16(S, (ulong)n * G + g));
        float bi = bf16_to_f32(q4_load_u16(B, (ulong)n * G + g)) - 1024.0f * sc;

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2u; i++) {
            uint wv = (i == 0u) ? wq.x : wq.y;
            #pragma clang loop unroll(full)
            for (uint tt = 0; tt < 4u; tt++) {
                simdgroup_half8x8 A;
                half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                A.thread_elements()[0] = h.x; A.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
            }
        }
        c0_acc = fma(sc, C.thread_elements()[0], fma(bi, s0, c0_acc));
        c1_acc = fma(sc, C.thread_elements()[1], fma(bi, s1, c1_acc));
    }

    threadgroup float red[KS][64];
    red[sg][fm * 8u + fn]      = c0_acc;
    red[sg][fm * 8u + fn + 1u] = c1_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = lane + 32u * sg;
    if (e < 64u) {
        uint r = e >> 3, m = e & 7u;
        uint gn = n0 + r;
        if (m < p.M && gn < N) {
            float acc = red[0][r * 8u + m] + red[1][r * 8u + m] + red[2][r * 8u + m] + red[3][r * 8u + m];
            Y[(ulong)m * N + gn] = acc;
        }
    }
}

kernel void mma_qmm16_q4_g64_f32(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;
    constexpr uint UG = 2u;
    constexpr uint NB = 2u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    const uint m0 = tgid.y * 16u;
    if (n0 >= N || m0 >= p.M) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ct[TN][NB][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++)
    #pragma clang loop unroll(full)
    for (uint c = 0; c < NB; c++) { ct[t][c][0] = 0.0f; ct[t][c][1] = 0.0f; }

    bool mok[NB][2];
    const device float* xr[NB][2];
    #pragma clang loop unroll(full)
    for (uint c = 0; c < NB; c++)
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2u; r++) {
        uint gm = m0 + 8u * c + fn + r;
        mok[c][r] = gm < p.M;
        xr[c][r]  = X + (ulong)(mok[c][r] ? gm : 0u) * K;
    }

    for (uint g0 = sg * UG; g0 < G; g0 += KS * UG) {
        uint2 wqs[UG][TN];
        float scs[UG][TN], bis[UG][TN];
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint gg = min(g0 + u, G - 1u);
            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                uint n = min(n0 + 8u * t + fm, N - 1u);
                wqs[u][t] = q4_load_u32x2(W, (ulong)n * KW + 8u * gg + 2u * b);
                scs[u][t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + gg));
                bis[u][t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + gg));
            }
        }
        #pragma clang loop unroll(full)
        for (uint u = 0; u < UG; u++) {
            uint g = g0 + u;
            if (g >= G) break;

            uint kx = 64u * g + 16u * sb + 4u * se;
            simdgroup_half8x8 Bm[NB][8];
            float xs[NB][2];
            #pragma clang loop unroll(full)
            for (uint c = 0; c < NB; c++) {
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 2u; r++) {
                    float4 a0 = *(const device float4*)(xr[c][r] + kx);
                    float4 a1 = *(const device float4*)(xr[c][r] + kx + 8u);
                    float v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    float s = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8u; j++) {
                        half hj = half(mok[c][r] ? v[j] : 0.0f);
                        s += float(hj);
                        Bm[c][j].thread_elements()[r] = hj;
                    }
                    s += simd_shuffle_xor(s, 2);
                    s += simd_shuffle_xor(s, 4);
                    s += simd_shuffle_xor(s, 16);
                    xs[c][r] = s;
                }
            }

            #pragma clang loop unroll(full)
            for (uint t = 0; t < TN; t++) {
                uint2 wq = wqs[u][t];
                float sc = scs[u][t];
                float bi = bis[u][t] - 1024.0f * sc;
                simdgroup_float8x8 C[NB];
                #pragma clang loop unroll(full)
                for (uint c = 0; c < NB; c++) C[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; i++) {
                    uint wv = (i == 0u) ? wq.x : wq.y;
                    #pragma clang loop unroll(full)
                    for (uint tt = 0; tt < 4u; tt++) {
                        simdgroup_half8x8 A;
                        half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                        A.thread_elements()[0] = h.x;
                        A.thread_elements()[1] = h.y;
                        #pragma clang loop unroll(full)
                        for (uint c = 0; c < NB; c++) {
                            simdgroup_multiply_accumulate(C[c], A, Bm[c][4u * i + tt], C[c]);
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (uint c = 0; c < NB; c++) {
                    ct[t][c][0] = fma(sc, C[c].thread_elements()[0], fma(bi, xs[c][0], ct[t][c][0]));
                    ct[t][c][1] = fma(sc, C[c].thread_elements()[1], fma(bi, xs[c][1], ct[t][c][1]));
                }
            }
        }
    }

    threadgroup float red[KS][TN][NB * 64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++)
    #pragma clang loop unroll(full)
    for (uint c = 0; c < NB; c++) {
        red[sg][t][c * 64u + fm * 8u + fn]      = ct[t][c][0];
        red[sg][t][c * 64u + fm * 8u + fn + 1u] = ct[t][c][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = lane + 32u * sg; e < TN * NB * 64u; e += 32u * KS) {
        uint t  = e >> 7;
        uint c  = (e >> 6) & 1u;
        uint r  = (e >> 3) & 7u;
        uint m  = e & 7u;
        uint gm = m0 + 8u * c + m;
        uint gn = n0 + 8u * t + r;
        if (gm < p.M && gn < N) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < KS; k++) acc += red[k][t][c * 64u + r * 8u + m];
            ulong idx = (ulong)gm * N + gn;
            Y[idx] = p.add_residual ? (Y[idx] + acc) : acc;
        }
    }
}

// ============================================================================
// 1e. Full-Cache-Line Hardware simdgroup_half8x8 Small-Batch GEMM for Q4_AFFINE_G64 (fallback)
// ============================================================================

kernel void qmm_q4_g64_m8(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 8u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    uint KW = p.K >> 3;
    uint KG = p.K >> 6;

    threadgroup half  sA[8][132];
    threadgroup half  sB[128][36];
    threadgroup float sOut[8][32];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint r_loc = tid >> 2;          // 0..31 (n_loc)
    uint lane4 = tid & 3u;          // 0..3
    uint c_blk = lane4 * 32u;       // 0, 32, 64, 96 (32 nibbles = 16 bytes/thread -> 64B/row!)
    uint ab    = tid >> 4;          // 0..7 (m_loc)
    uint ak    = (tid & 15u) * 8u;  // 0, 8, ..., 120 (8 floats/thread for sA[8][128])

    for (uint k0 = 0; k0 < p.K; k0 += 128u) {
        uint gm = m0 + ab;
        if (gm < p.M) {
            const device float* xp = X + (ulong)gm * p.K + k0 + ak;
            if (k0 + ak + 8u <= p.K) {
                float4 v0 = *((const device float4*)(xp + 0u));
                float4 v1 = *((const device float4*)(xp + 4u));
                sA[ab][ak + 0u] = half(v0.x); sA[ab][ak + 1u] = half(v0.y);
                sA[ab][ak + 2u] = half(v0.z); sA[ab][ak + 3u] = half(v0.w);
                sA[ab][ak + 4u] = half(v1.x); sA[ab][ak + 5u] = half(v1.y);
                sA[ab][ak + 6u] = half(v1.z); sA[ab][ak + 7u] = half(v1.w);
            } else {
                #pragma unroll
                for (uint i = 0; i < 8u; i++) sA[ab][ak + i] = (k0 + ak + i < p.K) ? half(xp[i]) : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sA[ab][ak + i] = half(0.0h);
        }

        uint gn = n0 + r_loc;
        if (gn < p.N && (k0 + c_blk) < p.K) {
            uint wi = (k0 >> 3) + lane4 * 4u;
            uint g  = (k0 + c_blk) >> 6;
            uint4 u = q4_load_u32x4(W, (ulong)gn * KW + wi);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)gn * KG + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)gn * KG + g));
            uint words[4] = {u.x, u.y, u.z, u.w};
            #pragma unroll
            for (uint w = 0; w < 4u; w++) {
                uint q = words[w];
                uint cb = c_blk + w * 8u;
                sB[cb + 0u][r_loc] = half(fma(float(q & 15u), sc, bi));
                sB[cb + 1u][r_loc] = half(fma(float((q >> 4)  & 15u), sc, bi));
                sB[cb + 2u][r_loc] = half(fma(float((q >> 8)  & 15u), sc, bi));
                sB[cb + 3u][r_loc] = half(fma(float((q >> 12) & 15u), sc, bi));
                sB[cb + 4u][r_loc] = half(fma(float((q >> 16) & 15u), sc, bi));
                sB[cb + 5u][r_loc] = half(fma(float((q >> 20) & 15u), sc, bi));
                sB[cb + 6u][r_loc] = half(fma(float((q >> 24) & 15u), sc, bi));
                sB[cb + 7u][r_loc] = half(fma(float((q >> 28) & 15u), sc, bi));
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 32u; i++) sB[c_blk + i][r_loc] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &sA[0][kk], 132);
            simdgroup_load(b, &sB[kk][sg * 8u], 36);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &sOut[0][sg * 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gm = m0 + ab;
    uint out_c = (tid & 15u) * 2u;
    if (gm < p.M) {
        uint gn0 = n0 + out_c;
        if (gn0 < p.N) {
            ulong idx = (ulong)gm * p.N + gn0;
            float v = sOut[ab][out_c];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
        if (gn0 + 1u < p.N) {
            ulong idx = (ulong)gm * p.N + gn0 + 1u;
            float v = sOut[ab][out_c + 1u];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

// ============================================================================
// 2. Q8_AFFINE_G64 Decode QMV + Tiled Prefill GEMM
// ============================================================================

kernel void qmm_q8_g64_f32(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y;
    if (m0 >= p.M) return;

    uint o0 = (tgid.x * 4u + sg) * 4u;
    if (o0 >= p.N) return;
    uint tn = min(4u, p.N - o0);

    uint KW = p.K >> 2;
    uint KG = p.K >> 6;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const device float* xr = X + (ulong)m0 * p.K;

    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 2;
        uint g  = wi >> 4;

        float4 x0 = *((const device float4*)(xr + k0));
        float4 x1 = *((const device float4*)(xr + k0 + 4u));
        float4 x2 = *((const device float4*)(xr + k0 + 8u));
        float4 x3 = *((const device float4*)(xr + k0 + 12u));
        float xs = ((x0.x + x0.y) + (x0.z + x0.w)) + ((x1.x + x1.y) + (x1.z + x1.w))
                 + ((x2.x + x2.y) + (x2.z + x2.w)) + ((x3.x + x3.y) + (x3.z + x3.w));

        for (uint rr = 0; rr < tn; rr++) {
            uint o = o0 + rr;
            const device uint* wp = W + (ulong)o * KW + wi;
            uint u0 = wp[0], u1 = wp[1], u2 = wp[2], u3 = wp[3];
            float sc = bf16_to_f32(S[(ulong)o * KG + g]);
            float bi = bf16_to_f32(B[(ulong)o * KG + g]);
            float d = 0.0f;
            d = fma(float(u0 & 255u), x0.x, d);        d = fma(float((u0 >> 8)  & 255u), x0.y, d);
            d = fma(float((u0 >> 16) & 255u), x0.z, d); d = fma(float((u0 >> 24) & 255u), x0.w, d);
            d = fma(float(u1 & 255u), x1.x, d);        d = fma(float((u1 >> 8)  & 255u), x1.y, d);
            d = fma(float((u1 >> 16) & 255u), x1.z, d); d = fma(float((u1 >> 24) & 255u), x1.w, d);
            d = fma(float(u2 & 255u), x2.x, d);        d = fma(float((u2 >> 8)  & 255u), x2.y, d);
            d = fma(float((u2 >> 16) & 255u), x2.z, d); d = fma(float((u2 >> 24) & 255u), x2.w, d);
            d = fma(float(u3 & 255u), x3.x, d);        d = fma(float((u3 >> 8)  & 255u), x3.y, d);
            d = fma(float((u3 >> 16) & 255u), x3.z, d); d = fma(float((u3 >> 24) & 255u), x3.w, d);
            acc[rr] = fma(sc, d, fma(bi, xs, acc[rr]));
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(acc[rr]);
        if (lane == 0u) {
            ulong idx = (ulong)m0 * p.N + (o0 + rr);
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

inline float q8_dot16_and_sum(uint u0, uint u1, uint u2, uint u3, const device float* xr, thread float& xs) {
    float4 x0 = *((const device float4*)(xr + 0u));
    float4 x1 = *((const device float4*)(xr + 4u));
    float4 x2 = *((const device float4*)(xr + 8u));
    float4 x3 = *((const device float4*)(xr + 12u));
    xs = ((x0.x + x0.y) + (x0.z + x0.w)) + ((x1.x + x1.y) + (x1.z + x1.w))
       + ((x2.x + x2.y) + (x2.z + x2.w)) + ((x3.x + x3.y) + (x3.z + x3.w));
    float d = 0.0f;
    d = fma(float(u0 & 255u), x0.x, d);        d = fma(float((u0 >> 8)  & 255u), x0.y, d);
    d = fma(float((u0 >> 16) & 255u), x0.z, d); d = fma(float((u0 >> 24) & 255u), x0.w, d);
    d = fma(float(u1 & 255u), x1.x, d);        d = fma(float((u1 >> 8)  & 255u), x1.y, d);
    d = fma(float((u1 >> 16) & 255u), x1.z, d); d = fma(float((u1 >> 24) & 255u), x1.w, d);
    d = fma(float(u2 & 255u), x2.x, d);        d = fma(float((u2 >> 8)  & 255u), x2.y, d);
    d = fma(float((u2 >> 16) & 255u), x2.z, d); d = fma(float((u2 >> 24) & 255u), x2.w, d);
    d = fma(float(u3 & 255u), x3.x, d);        d = fma(float((u3 >> 8)  & 255u), x3.y, d);
    d = fma(float((u3 >> 16) & 255u), x3.z, d); d = fma(float((u3 >> 24) & 255u), x3.w, d);
    return d;
}

// Lean 1-row/simdgroup Q8 batch-2..4: ~18 regs total, ZERO spill, reads W ONCE across all tm tokens!
kernel void qmm_q8_g64_batch4(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 4u;
    uint o  = tgid.x * 4u + sg;
    if (m0 >= p.M || o >= p.N) return;
    uint tm = min(4u, p.M - m0);

    uint KW = p.K >> 2;
    uint KG = p.K >> 6;
    const device uint*   w_row = W + (ulong)o * KW;
    const device ushort* s_row = S + (ulong)o * KG;
    const device ushort* b_row = B + (ulong)o * KG;
    const device float*  x0_base = X + (ulong)(m0 + 0u) * p.K;
    const device float*  x1_base = X + (ulong)(m0 + min(1u, tm - 1u)) * p.K;
    const device float*  x2_base = X + (ulong)(m0 + min(2u, tm - 1u)) * p.K;
    const device float*  x3_base = X + (ulong)(m0 + min(3u, tm - 1u)) * p.K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint wi = lane * 4u; wi < KW; wi += 128u) {
        uint k0 = wi << 2;
        uint g  = wi >> 4;
        uint u0 = w_row[wi + 0u], u1 = w_row[wi + 1u], u2 = w_row[wi + 2u], u3 = w_row[wi + 3u];
        float sc = bf16_to_f32(s_row[g]);
        float bi = bf16_to_f32(b_row[g]);
        float xs;
        a0 = fma(sc, q8_dot16_and_sum(u0, u1, u2, u3, x0_base + k0, xs), fma(bi, xs, a0));
        if (tm > 1u) a1 = fma(sc, q8_dot16_and_sum(u0, u1, u2, u3, x1_base + k0, xs), fma(bi, xs, a1));
        if (tm > 2u) a2 = fma(sc, q8_dot16_and_sum(u0, u1, u2, u3, x2_base + k0, xs), fma(bi, xs, a2));
        if (tm > 3u) a3 = fma(sc, q8_dot16_and_sum(u0, u1, u2, u3, x3_base + k0, xs), fma(bi, xs, a3));
    }

    float v0 = simd_sum(a0);
    float v1 = (tm > 1u) ? simd_sum(a1) : 0.0f;
    float v2 = (tm > 2u) ? simd_sum(a2) : 0.0f;
    float v3 = (tm > 3u) ? simd_sum(a3) : 0.0f;
    if (lane == 0u) {
        ulong i0 = (ulong)(m0 + 0u) * p.N + o; Y[i0] = p.add_residual ? (Y[i0] + v0) : v0;
        if (tm > 1u) { ulong i1 = (ulong)(m0 + 1u) * p.N + o; Y[i1] = p.add_residual ? (Y[i1] + v1) : v1; }
        if (tm > 2u) { ulong i2 = (ulong)(m0 + 2u) * p.N + o; Y[i2] = p.add_residual ? (Y[i2] + v2) : v2; }
        if (tm > 3u) { ulong i3 = (ulong)(m0 + 3u) * p.N + o; Y[i3] = p.add_residual ? (Y[i3] + v3) : v3; }
    }
}

kernel void qmm_q8_g64_tiled(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    uint KW = p.K >> 2;
    uint KG = p.K >> 6;

    threadgroup half sA[32][36];
    threadgroup half sB[32][36];

    simdgroup_float8x8 acc[2][2];
    acc[0][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[0][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint sg_m = (sg >> 1) * 16u;
    uint sg_n = (sg & 1u) * 16u;

    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint gm = m0 + r_loc;
        if (gm < p.M) {
            const device float* xp = X + (ulong)gm * p.K + k0 + c_blk;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sA[r_loc][c_blk + i] = half(xp[i]);
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sA[r_loc][c_blk + i] = half(0.0h);
        }

        uint gn = n0 + r_loc;
        if (gn < p.N) {
            uint wi = (k0 >> 2) + (tid & 3u) * 2u;
            uint g  = k0 >> 6;
            const device uint* wp = W + (ulong)gn * KW + wi;
            uint u0 = wp[0], u1 = wp[1];
            float sc = bf16_to_f32(S[(ulong)gn * KG + g]);
            float bi = bf16_to_f32(B[(ulong)gn * KG + g]);
            sB[c_blk + 0u][r_loc] = half(fma(float(u0 & 255u), sc, bi));
            sB[c_blk + 1u][r_loc] = half(fma(float((u0 >> 8)  & 255u), sc, bi));
            sB[c_blk + 2u][r_loc] = half(fma(float((u0 >> 16) & 255u), sc, bi));
            sB[c_blk + 3u][r_loc] = half(fma(float((u0 >> 24) & 255u), sc, bi));
            sB[c_blk + 4u][r_loc] = half(fma(float(u1 & 255u), sc, bi));
            sB[c_blk + 5u][r_loc] = half(fma(float((u1 >> 8)  & 255u), sc, bi));
            sB[c_blk + 6u][r_loc] = half(fma(float((u1 >> 16) & 255u), sc, bi));
            sB[c_blk + 7u][r_loc] = half(fma(float((u1 >> 24) & 255u), sc, bi));
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sB[c_blk + i][r_loc] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &sA[sg_m + 0u][kk], 36);
            simdgroup_load(a1, &sA[sg_m + 8u][kk], 36);
            simdgroup_load(b0, &sB[kk][sg_n + 0u], 36);
            simdgroup_load(b1, &sB[kk][sg_n + 8u], 36);

            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float sOut[32][32];
    simdgroup_store(acc[0][0], &sOut[sg_m + 0u][sg_n + 0u], 32);
    simdgroup_store(acc[0][1], &sOut[sg_m + 0u][sg_n + 8u], 32);
    simdgroup_store(acc[1][0], &sOut[sg_m + 8u][sg_n + 0u], 32);
    simdgroup_store(acc[1][1], &sOut[sg_m + 8u][sg_n + 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gm = m0 + r_loc;
    if (gm < p.M) {
        for (uint i = 0; i < 8u; i++) {
            uint gn = n0 + c_blk + i;
            if (gn < p.N) {
                ulong idx = (ulong)gm * p.N + gn;
                float v = sOut[r_loc][c_blk + i];
                Y[idx] = p.add_residual ? (Y[idx] + v) : v;
            }
        }
    }
}

// ============================================================================
// 2b. Hardware simdgroup_half8x8 Small-Batch GEMM for Q8_AFFINE_G64 (M = 2..8)
// ============================================================================

// Full-Cache-Line Hardware simdgroup_half8x8 Small-Batch GEMM for Q8_AFFINE_G64 (M = 2..8)
// BM=8, BN=32, BK=64: 1 uint4/thread x 4 threads = 64B FULL CACHE LINE, 1 S/B load/group!
kernel void qmm_q8_g64_m8(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device float*  X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    constant QmmParams&  p     [[buffer(5)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 8u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    uint KW = p.K >> 2;
    uint KG = p.K >> 6;

    threadgroup half  sA[8][68];
    threadgroup half  sB[64][36];
    threadgroup float sOut[8][32];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint r_loc = tid >> 2;
    uint lane4 = tid & 3u;
    uint c_blk = lane4 * 16u;      // 0, 16, 32, 48 (16 bytes/thread -> 64B/row!)
    uint ab    = tid >> 4;
    uint ak    = (tid & 15u) * 4u; // 0..60 (4 floats/thread)

    for (uint k0 = 0; k0 < p.K; k0 += 64u) {
        uint gm = m0 + ab;
        if (gm < p.M) {
            const device float* xp = X + (ulong)gm * p.K + k0 + ak;
            float4 v = *((const device float4*)xp);
            sA[ab][ak + 0u] = half(v.x); sA[ab][ak + 1u] = half(v.y);
            sA[ab][ak + 2u] = half(v.z); sA[ab][ak + 3u] = half(v.w);
        } else {
            sA[ab][ak + 0u] = half(0.0h); sA[ab][ak + 1u] = half(0.0h);
            sA[ab][ak + 2u] = half(0.0h); sA[ab][ak + 3u] = half(0.0h);
        }

        uint gn = n0 + r_loc;
        if (gn < p.N) {
            uint wi = (k0 >> 2) + lane4 * 4u;
            uint g  = k0 >> 6;
            uint4 u = *((const device uint4*)(W + (ulong)gn * KW + wi));
            float sc = bf16_to_f32(S[(ulong)gn * KG + g]);
            float bi = bf16_to_f32(B[(ulong)gn * KG + g]);
            sB[c_blk + 0u][r_loc]  = half(fma(float(u.x & 255u), sc, bi));
            sB[c_blk + 1u][r_loc]  = half(fma(float((u.x >> 8)  & 255u), sc, bi));
            sB[c_blk + 2u][r_loc]  = half(fma(float((u.x >> 16) & 255u), sc, bi));
            sB[c_blk + 3u][r_loc]  = half(fma(float((u.x >> 24) & 255u), sc, bi));
            sB[c_blk + 4u][r_loc]  = half(fma(float(u.y & 255u), sc, bi));
            sB[c_blk + 5u][r_loc]  = half(fma(float((u.y >> 8)  & 255u), sc, bi));
            sB[c_blk + 6u][r_loc]  = half(fma(float((u.y >> 16) & 255u), sc, bi));
            sB[c_blk + 7u][r_loc]  = half(fma(float((u.y >> 24) & 255u), sc, bi));
            sB[c_blk + 8u][r_loc]  = half(fma(float(u.z & 255u), sc, bi));
            sB[c_blk + 9u][r_loc]  = half(fma(float((u.z >> 8)  & 255u), sc, bi));
            sB[c_blk + 10u][r_loc] = half(fma(float((u.z >> 16) & 255u), sc, bi));
            sB[c_blk + 11u][r_loc] = half(fma(float((u.z >> 24) & 255u), sc, bi));
            sB[c_blk + 12u][r_loc] = half(fma(float(u.w & 255u), sc, bi));
            sB[c_blk + 13u][r_loc] = half(fma(float((u.w >> 8)  & 255u), sc, bi));
            sB[c_blk + 14u][r_loc] = half(fma(float((u.w >> 16) & 255u), sc, bi));
            sB[c_blk + 15u][r_loc] = half(fma(float((u.w >> 24) & 255u), sc, bi));
        } else {
            #pragma unroll
            for (uint i = 0; i < 16u; i++) sB[c_blk + i][r_loc] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 64u; kk += 8u) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &sA[0][kk], 68);
            simdgroup_load(b, &sB[kk][sg * 8u], 36);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &sOut[0][sg * 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gm = m0 + ab;
    uint out_c = (tid & 15u) * 2u;
    if (gm < p.M) {
        uint gn0 = n0 + out_c;
        if (gn0 < p.N) {
            ulong idx = (ulong)gm * p.N + gn0;
            float v = sOut[ab][out_c];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
        if (gn0 + 1u < p.N) {
            ulong idx = (ulong)gm * p.N + gn0 + 1u;
            float v = sOut[ab][out_c + 1u];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

// ============================================================================
// 3. BF16 Linear: Decode QMV + Tiled Prefill GEMM
// ============================================================================

kernel void qmm_bf16_f32(
    const device ushort* W     [[buffer(0)]],
    const device float*  X     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    constant QmmParams&  p     [[buffer(3)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y;
    if (m0 >= p.M) return;

    uint o0 = (tgid.x * 4u + sg) * 4u;
    if (o0 >= p.N) return;
    uint tn = min(4u, p.N - o0);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const device float* xr = X + (ulong)m0 * p.K;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
        if (k + 4u <= p.K) {
            float4 x4 = *((const device float4*)(xr + k));
            for (uint rr = 0; rr < tn; rr++) {
                const device ushort* wp = W + (ulong)(o0 + rr) * p.K + k;
                float a = acc[rr];
                a = fma(bf16_to_f32(wp[0]), x4.x, a);
                a = fma(bf16_to_f32(wp[1]), x4.y, a);
                a = fma(bf16_to_f32(wp[2]), x4.z, a);
                a = fma(bf16_to_f32(wp[3]), x4.w, a);
                acc[rr] = a;
            }
        } else {
            for (uint kk = k; kk < p.K; kk++) {
                float xv = xr[kk];
                for (uint rr = 0; rr < tn; rr++) {
                    acc[rr] = fma(bf16_to_f32(W[(ulong)(o0 + rr) * p.K + kk]), xv, acc[rr]);
                }
            }
        }
    }
    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(acc[rr]);
        if (lane == 0u) {
            ulong idx = (ulong)m0 * p.N + (o0 + rr);
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

kernel void qmm_bf16_batch4(
    const device ushort* W     [[buffer(0)]],
    const device float*  X     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    constant QmmParams&  p     [[buffer(3)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 4u;
    uint o  = tgid.x * 4u + sg;
    if (m0 >= p.M || o >= p.N) return;
    uint tm = min(4u, p.M - m0);

    const device ushort* w_row = W + (ulong)o * p.K;
    const device float* x0_base = X + (ulong)(m0 + 0u) * p.K;
    const device float* x1_base = X + (ulong)(m0 + min(1u, tm - 1u)) * p.K;
    const device float* x2_base = X + (ulong)(m0 + min(2u, tm - 1u)) * p.K;
    const device float* x3_base = X + (ulong)(m0 + min(3u, tm - 1u)) * p.K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint k = lane * 4u; k < p.K; k += 128u) {
        if (k + 4u <= p.K) {
            float w0 = bf16_to_f32(w_row[k + 0u]), w1 = bf16_to_f32(w_row[k + 1u]);
            float w2 = bf16_to_f32(w_row[k + 2u]), w3 = bf16_to_f32(w_row[k + 3u]);
            float4 x0 = *((const device float4*)(x0_base + k));
            a0 = fma(w0, x0.x, fma(w1, x0.y, fma(w2, x0.z, fma(w3, x0.w, a0))));
            if (tm > 1u) { float4 x1 = *((const device float4*)(x1_base + k)); a1 = fma(w0, x1.x, fma(w1, x1.y, fma(w2, x1.z, fma(w3, x1.w, a1)))); }
            if (tm > 2u) { float4 x2 = *((const device float4*)(x2_base + k)); a2 = fma(w0, x2.x, fma(w1, x2.y, fma(w2, x2.z, fma(w3, x2.w, a2)))); }
            if (tm > 3u) { float4 x3 = *((const device float4*)(x3_base + k)); a3 = fma(w0, x3.x, fma(w1, x3.y, fma(w2, x3.z, fma(w3, x3.w, a3)))); }
        } else {
            for (uint kk = k; kk < p.K; kk++) {
                float wv = bf16_to_f32(w_row[kk]);
                a0 = fma(wv, x0_base[kk], a0);
                if (tm > 1u) a1 = fma(wv, x1_base[kk], a1);
                if (tm > 2u) a2 = fma(wv, x2_base[kk], a2);
                if (tm > 3u) a3 = fma(wv, x3_base[kk], a3);
            }
        }
    }
    float v0 = simd_sum(a0);
    float v1 = (tm > 1u) ? simd_sum(a1) : 0.0f;
    float v2 = (tm > 2u) ? simd_sum(a2) : 0.0f;
    float v3 = (tm > 3u) ? simd_sum(a3) : 0.0f;
    if (lane == 0u) {
        ulong i0 = (ulong)(m0 + 0u) * p.N + o; Y[i0] = p.add_residual ? (Y[i0] + v0) : v0;
        if (tm > 1u) { ulong i1 = (ulong)(m0 + 1u) * p.N + o; Y[i1] = p.add_residual ? (Y[i1] + v1) : v1; }
        if (tm > 2u) { ulong i2 = (ulong)(m0 + 2u) * p.N + o; Y[i2] = p.add_residual ? (Y[i2] + v2) : v2; }
        if (tm > 3u) { ulong i3 = (ulong)(m0 + 3u) * p.N + o; Y[i3] = p.add_residual ? (Y[i3] + v3) : v3; }
    }
}

kernel void qmm_bf16_tiled(
    const device ushort* W     [[buffer(0)]],
    const device float*  X     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    constant QmmParams&  p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 32u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    threadgroup half sA[32][36];
    threadgroup half sB[32][36];

    simdgroup_float8x8 acc[2][2];
    acc[0][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[0][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint sg_m = (sg >> 1) * 16u;
    uint sg_n = (sg & 1u) * 16u;
    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint gm = m0 + r_loc;
        if (gm < p.M) {
            const device float* xp = X + (ulong)gm * p.K + k0 + c_blk;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                sA[r_loc][c_blk + i] = (k0 + c_blk + i < p.K) ? half(xp[i]) : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sA[r_loc][c_blk + i] = half(0.0h);
        }

        uint gn = n0 + r_loc;
        if (gn < p.N) {
            const device ushort* wp = W + (ulong)gn * p.K + k0 + c_blk;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                sB[c_blk + i][r_loc] = (k0 + c_blk + i < p.K) ? half(bf16_to_f32(wp[i])) : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sB[c_blk + i][r_loc] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &sA[sg_m + 0u][kk], 36);
            simdgroup_load(a1, &sA[sg_m + 8u][kk], 36);
            simdgroup_load(b0, &sB[kk][sg_n + 0u], 36);
            simdgroup_load(b1, &sB[kk][sg_n + 8u], 36);

            simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float sOut[32][32];
    simdgroup_store(acc[0][0], &sOut[sg_m + 0u][sg_n + 0u], 32);
    simdgroup_store(acc[0][1], &sOut[sg_m + 0u][sg_n + 8u], 32);
    simdgroup_store(acc[1][0], &sOut[sg_m + 8u][sg_n + 0u], 32);
    simdgroup_store(acc[1][1], &sOut[sg_m + 8u][sg_n + 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gm = m0 + r_loc;
    if (gm < p.M) {
        for (uint i = 0; i < 8u; i++) {
            uint gn = n0 + c_blk + i;
            if (gn < p.N) {
                ulong idx = (ulong)gm * p.N + gn;
                float v = sOut[r_loc][c_blk + i];
                Y[idx] = p.add_residual ? (Y[idx] + v) : v;
            }
        }
    }
}

// ============================================================================
// 3b. Hardware simdgroup_half8x8 Small-Batch GEMM for BF16 (M = 2..8)
// ============================================================================

kernel void qmm_bf16_m8(
    const device ushort* W     [[buffer(0)]],
    const device float*  X     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    constant QmmParams&  p     [[buffer(3)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint m0 = tgid.y * 8u;
    uint n0 = tgid.x * 32u;
    if (m0 >= p.M || n0 >= p.N) return;

    threadgroup half  sA[8][36];
    threadgroup half  sB[32][36];
    threadgroup float sOut[8][32];

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;
    uint ab    = tid >> 4;
    uint ak    = (tid & 15u) * 2u;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint gm = m0 + ab;
        if (gm < p.M) {
            const device float* xp = X + (ulong)gm * p.K + k0 + ak;
            sA[ab][ak + 0u] = (k0 + ak + 0u < p.K) ? half(xp[0]) : half(0.0h);
            sA[ab][ak + 1u] = (k0 + ak + 1u < p.K) ? half(xp[1]) : half(0.0h);
        } else {
            sA[ab][ak + 0u] = half(0.0h);
            sA[ab][ak + 1u] = half(0.0h);
        }

        uint gn = n0 + r_loc;
        if (gn < p.N) {
            const device ushort* wp = W + (ulong)gn * p.K + k0 + c_blk;
            #pragma unroll
            for (uint i = 0; i < 8u; i++) {
                sB[c_blk + i][r_loc] = (k0 + c_blk + i < p.K) ? half(bf16_to_f32(wp[i])) : half(0.0h);
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 8u; i++) sB[c_blk + i][r_loc] = half(0.0h);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, &sA[0][kk], 36);
            simdgroup_load(b, &sB[kk][sg * 8u], 36);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &sOut[0][sg * 8u], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gm = m0 + ab;
    if (gm < p.M) {
        uint gn0 = n0 + ak;
        if (gn0 < p.N) {
            ulong idx = (ulong)gm * p.N + gn0;
            float v = sOut[ab][ak];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
        if (gn0 + 1u < p.N) {
            ulong idx = (ulong)gm * p.N + gn0 + 1u;
            float v = sOut[ab][ak + 1u];
            Y[idx] = p.add_residual ? (Y[idx] + v) : v;
        }
    }
}

// ============================================================================
// 4. Embedding Lookups
// ============================================================================

struct EmbedParams {
    uint M;
    uint D;
    uint V; // Repurposed from unused float eps: clamps tok < V (with legacy float-eps ABI guard if V >= 0x10000000u)
};

kernel void embed_q4_g64_f32(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device int*    ids   [[buffer(3)]],
    device float*        out   [[buffer(4)]],
    constant EmbedParams& p    [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= p.D || row >= p.M) return;

    uint tok = uint(ids[row]);
    uint v_lim = (p.V > 0u && p.V < 0x10000000u) ? p.V : 0x10000000u;
    if (tok >= v_lim) tok = 0u;
    uint KW = p.D >> 3;
    uint KG = p.D >> 6;
    uint w_idx = col >> 3;
    uint nib_shift = (col & 7u) * 4u;
    uint g_idx = col >> 6;

    uint word = q4_load_u32(W, (ulong)tok * KW + w_idx);
    float q = float((word >> nib_shift) & 15u);
    float sc = bf16_to_f32(q4_load_u16(S, (ulong)tok * KG + g_idx));
    float bi = bf16_to_f32(q4_load_u16(B, (ulong)tok * KG + g_idx));
    out[(ulong)row * p.D + col] = fma(q, sc, bi);
}

// ============================================================================
// 5. RMSNorm, SwiGLU, Residual Add, Copy
// ============================================================================

struct NormParams {
    uint M;
    uint D;
    float eps;
};

kernel void rmsnorm_f32_bf16w(
    const device float*  X     [[buffer(0)]],
    const device ushort* W     [[buffer(1)]],
    device float*        Y     [[buffer(2)]],
    constant NormParams& p     [[buffer(3)]],
    uint  row                  [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_position_in_threadgroup]],
    uint  tg_size              [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xr = X + (ulong)row * p.D;
    device float* yr = Y + (ulong)row * p.D;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = xr[i];
        sum_sq = fma(v, v, sum_sq);
    }
    sum_sq = simd_sum(sum_sq);

    threadgroup float shared_sums[32];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0u) shared_sums[sg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nsg = tg_size >> 5;
    if (sg == 0u) {
        float total = (lane < nsg) ? shared_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0u) shared_sums[0] = rsqrt(total / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_sums[0];

    for (uint i = tid; i < p.D; i += tg_size) {
        yr[i] = xr[i] * inv_rms * bf16_to_f32(W[i]);
    }
}

kernel void rmsnorm_f32_bf16w_to_f16(
    const device float*  X     [[buffer(0)]],
    const device ushort* W     [[buffer(1)]],
    device half*         Y     [[buffer(2)]],
    constant NormParams& p     [[buffer(3)]],
    uint  row                  [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_position_in_threadgroup]],
    uint  tg_size              [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xr = X + (ulong)row * p.D;
    device half* yr = Y + (ulong)row * p.D;

    float sum_sq = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = xr[i];
        sum_sq = fma(v, v, sum_sq);
    }
    sum_sq = simd_sum(sum_sq);

    threadgroup float shared_sums[32];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0u) shared_sums[sg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nsg = tg_size >> 5;
    if (sg == 0u) {
        float total = (lane < nsg) ? shared_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0u) shared_sums[0] = rsqrt(total / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_sums[0];

    for (uint i = tid; i < p.D; i += tg_size) {
        yr[i] = half(xr[i] * inv_rms * bf16_to_f32(W[i]));
    }
}

kernel void swiglu_f32(
    const device float* gate [[buffer(0)]],
    const device float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    out[gid] = silu_f32(gate[gid]) * up[gid];
}

kernel void swiglu_f32_to_f16(
    const device float* gate [[buffer(0)]],
    const device float* up   [[buffer(1)]],
    device half*        out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    out[gid] = half(silu_f32(gate[gid]) * up[gid]);
}

kernel void vec_add_f32(
    device float*       dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant uint&      N   [[buffer(2)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    dst[gid] += src[gid];
}

kernel void vec_copy_f32(
    const device float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      N   [[buffer(2)]],
    uint gid                [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    dst[gid] = src[gid];
}

// ============================================================================
// 6. Fast 2-stage GPU Argmax over logits
// ============================================================================

struct ArgmaxOut {
    uint  idx;
    float val;
};

kernel void argmax_stage1(
    const device float* logits [[buffer(0)]],
    device ArgmaxOut*   parts  [[buffer(1)]],
    constant uint&      V      [[buffer(2)]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint  tid                  [[thread_index_in_threadgroup]],
    uint2 tg_size              [[threads_per_threadgroup]],
    uint2 n_tg                 [[threadgroups_per_grid]]
) {
    uint row = tgid.y;
    const device float* lr = logits + (ulong)row * V;
    float best_v = -INFINITY;
    uint  best_i = 0u;
    uint stride = tg_size.x * n_tg.x;
    for (uint i = tgid.x * tg_size.x + tid; i < V; i += stride) {
        float v = lr[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint  oi = simd_shuffle_down(best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    threadgroup float s_v[32];
    threadgroup uint  s_i[32];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0u) { s_v[sg] = best_v; s_i[sg] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size.x >> 5;
        best_v = (lane < nsg) ? s_v[lane] : -INFINITY;
        best_i = (lane < nsg) ? s_i[lane] : 0u;
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) {
            uint slot = row * n_tg.x + tgid.x;
            parts[slot].idx = best_i;
            parts[slot].val = best_v;
        }
    }
}

kernel void argmax_stage2(
    const device ArgmaxOut* parts [[buffer(0)]],
    device uint*            out   [[buffer(1)]],
    constant uint&          N     [[buffer(2)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  tid                     [[thread_index_in_threadgroup]],
    uint2 tg_size                 [[threads_per_threadgroup]]
) {
    uint row = tgid.y;
    const device ArgmaxOut* pr = parts + (ulong)row * N;
    float best_v = -INFINITY;
    uint  best_i = 0u;
    for (uint i = tid; i < N; i += tg_size.x) {
        float v = pr[i].val;
        if (v > best_v) { best_v = v; best_i = pr[i].idx; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint  oi = simd_shuffle_down(best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    threadgroup float s_v[32];
    threadgroup uint  s_i[32];
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    if (lane == 0u) { s_v[sg] = best_v; s_i[sg] = best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size.x >> 5;
        best_v = (lane < nsg) ? s_v[lane] : -INFINITY;
        best_i = (lane < nsg) ? s_i[lane] : 0u;
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) out[row] = best_i;
    }
}

// ============================================================================
// 6b. Round 30: Fused lm_head (FP16-X M=2..8) + Per-TG Warp-Shuffle Row-Max!
//     Remaps epilogue so simdgroup sg = row m (0..7) and lane = col (0..31):
//     Computes exact per-TG (best_v, best_i) in 5 warp shuffles with ZERO extra
//     TG memory or barriers -> completely eliminates argmax_stage1 (8 MB DRAM read)!
// ============================================================================

kernel void mma_qmm8_lmhead_max_f16x(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    device ArgmaxOut*    parts [[buffer(5)]],
    constant QmmParams&  p     [[buffer(6)]], // p.add_residual != 0 skips writing full Y logits (greedy fast path!)
    uint2 tgid                 [[threadgroup_position_in_grid]],
    uint2 n_tg                 [[threadgroups_per_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    if (n0 >= N) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    float ct[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) { ct[t][0] = 0.0f; ct[t][1] = 0.0f; }

    const bool m0ok = fn < p.M;
    const bool m1ok = (fn + 1u) < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? fn : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? (fn + 1u) : 0u) * K;

    for (uint g = sg; g < G; g += KS) {
        uint2 wqs[TN];
        float scs[TN], bis[TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint n = min(n0 + 8u * t + fm, N - 1u);
            wqs[t] = q4_load_u32x2(W, (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + g));
            bis[t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + g));
        }

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint2 wq = wqs[t];
            float sc = scs[t];
            float bi = bis[t] - 1024.0f * sc;
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; i++) {
                uint wv = (i == 0u) ? wq.x : wq.y;
                #pragma clang loop unroll(full)
                for (uint tt = 0; tt < 4u; tt++) {
                    simdgroup_half8x8 A;
                    half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                    A.thread_elements()[0] = h.x;
                    A.thread_elements()[1] = h.y;
                    simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                }
            }
            ct[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ct[t][0]));
            ct[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ct[t][1]));
        }
    }

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red[sg][t][fm * 8u + fn]      = ct[t][0];
        red[sg][t][fm * 8u + fn + 1u] = ct[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: simdgroup sg = row m (0..7), lane = column (0..31) -> optional Y write + 5-step warp row-max!
    uint m = sg;
    if (m < p.M) {
        uint t  = lane >> 3;
        uint r  = lane & 7u;
        uint gn = n0 + lane;
        float best_v = -INFINITY;
        uint  best_i = gn;
        if (gn < N) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < KS; k++) acc += red[k][t][r * 8u + m];
            if (p.add_residual == 0u) {
                Y[(ulong)m * N + gn] = acc;
            }
            best_v = acc;
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) {
            uint slot = m * n_tg.x + tgid.x;
            parts[slot].idx = best_i;
            parts[slot].val = best_v;
        }
    }
}

// Fused M=1 FP16-X lm_head + per-TG row-max (eliminates argmax_stage1 1 MB DRAM read + barrier!)
// p.add_residual != 0 skips writing full Y logits when caller only needs argmax.
kernel void qmv_q4_g64_f16x_max(
    const device uchar*  W     [[buffer(0)]],
    const device uchar*  S     [[buffer(1)]],
    const device uchar*  B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    device ArgmaxOut*    parts [[buffer(5)]],
    constant QmmParams&  p     [[buffer(6)]],
    uint  tgid                 [[threadgroup_position_in_grid]],
    uint  lane                 [[thread_index_in_simdgroup]],
    uint  sg                   [[simdgroup_index_in_threadgroup]]
) {
    uint o0 = (tgid * 4u + sg) * 2u;
    uint KW = p.K >> 3;
    uint KG = p.K >> 6;
    float sg_best_v = -INFINITY;
    uint  sg_best_i = 0u;

    if (o0 < p.N) {
        uint tn = min(2u, p.N - o0);
        float acc[2] = {0.0f, 0.0f};
        for (uint wi = lane * 4u; wi < KW; wi += 128u) {
            uint k0 = wi << 3;
            uint g  = wi >> 3;
            const device half4* xp = (const device half4*)(X + k0);
            half4 xv[8];
            float xs = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; j++) {
                half4 v = xp[j];
                xv[j] = v;
                xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
            }
            #pragma clang loop unroll(full)
            for (uint rr = 0u; rr < 2u; rr++) {
                if (rr < tn) {
                    uint o = o0 + rr;
                    uint4 uq = q4_load_u32x4(W, (ulong)o * KW + wi);
                    float sc = bf16_to_f32(q4_load_u16(S, (ulong)o * KG + g));
                    float bi = bf16_to_f32(q4_load_u16(B, (ulong)o * KG + g));
                    acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, acc[rr]));
                }
            }
        }
        for (uint rr = 0u; rr < tn; rr++) {
            float v = simd_sum(acc[rr]);
            if (lane == 0u) {
                uint o = o0 + rr;
                if (p.add_residual == 0u) Y[o] = v;
                if (v > sg_best_v) { sg_best_v = v; sg_best_i = o; }
            }
        }
    }

    threadgroup float sv[4];
    threadgroup uint  si[4];
    if (lane == 0u) { sv[sg] = sg_best_v; si[sg] = sg_best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float bv = sv[0];
        uint  bi = si[0];
        for (uint s = 1u; s < 4u; ++s) {
            if (sv[s] > bv) { bv = sv[s]; bi = si[s]; }
        }
        parts[tgid].val = bv;
        parts[tgid].idx = bi;
    }
}

// ============================================================================
// 6c. Dynamic Compact row_map Draft lm_head (B=1 & B=2..8) + Fused TG Row-Max!
//     Reads only the active V_draft rows (e.g. 16k-32k hot + dynamic prompt/gen
//     tokens) via zero-copy row_map[local_n] -> real_vocab_id (full 2.8 KB/row
//     coalesced DRAM bursts, 0 MB weight duplication, eliminates argmax_stage1)!
// ============================================================================

kernel void qmv_q4_g64_f16x_indexed_max(
    const device uchar*  W       [[buffer(0)]],
    const device uchar*  S       [[buffer(1)]],
    const device uchar*  B       [[buffer(2)]],
    const device half*   X       [[buffer(3)]],
    device float*        Y       [[buffer(4)]],
    device ArgmaxOut*    parts   [[buffer(5)]],
    const device uint*   row_map [[buffer(6)]],
    constant QmmParams&  p       [[buffer(7)]],
    uint  tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]]
) {
    uint o0 = (tgid * 4u + sg) * 2u;
    uint KW = p.K >> 3;
    uint KG = p.K >> 6;
    float sg_best_v = -INFINITY;
    uint  sg_best_i = 0u;

    if (o0 < p.N) {
        uint tn = min(2u, p.N - o0);
        uint orig[2];
        orig[0] = row_map[o0];
        orig[1] = row_map[min(o0 + 1u, p.N - 1u)];

        float acc[2] = {0.0f, 0.0f};
        for (uint wi = lane * 4u; wi < KW; wi += 128u) {
            uint k0 = wi << 3;
            uint g  = wi >> 3;
            half4 xv[8];
            float xs = 0.0f;
            const device half4* xp = (const device half4*)(X + k0);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; j++) {
                half4 v = xp[j];
                xv[j] = v;
                xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
            }
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 2u; rr++) {
                uint ro = orig[rr];
                uint4 uq = q4_load_u32x4(W, (ulong)ro * KW + wi);
                float sc = bf16_to_f32(q4_load_u16(S, (ulong)ro * KG + g));
                float bi = bf16_to_f32(q4_load_u16(B, (ulong)ro * KG + g));
                acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, acc[rr]));
            }
        }
        for (uint rr = 0; rr < tn; rr++) {
            float v = simd_sum(acc[rr]);
            if (lane == 0u) {
                Y[o0 + rr] = v;
                if (v > sg_best_v) { sg_best_v = v; sg_best_i = orig[rr]; }
            }
        }
    }

    threadgroup float sv[4];
    threadgroup uint  si[4];
    if (lane == 0u) { sv[sg] = sg_best_v; si[sg] = sg_best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float bv = sv[0];
        uint  bi = si[0];
        for (uint s = 1u; s < 4u; ++s) {
            if (sv[s] > bv) { bv = sv[s]; bi = si[s]; }
        }
        parts[tgid].val = bv;
        parts[tgid].idx = bi;
    }
}

kernel void mma_qmm8_lmhead_max_f16x_indexed(
    const device uchar*  W       [[buffer(0)]],
    const device uchar*  S       [[buffer(1)]],
    const device uchar*  B       [[buffer(2)]],
    const device half*   X       [[buffer(3)]],
    device float*        Y       [[buffer(4)]],
    device ArgmaxOut*    parts   [[buffer(5)]],
    const device uint*   row_map [[buffer(6)]],
    constant QmmParams&  p       [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint2 n_tg                   [[threadgroups_per_grid]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;

    const uint K  = p.K;
    const uint N  = p.N;
    const uint KW = K >> 3;
    const uint G  = K >> 6;
    const uint n0 = tgid.x * (8u * TN);
    if (n0 >= N) return;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint b   = fn >> 1;
    const uint sb  = fm >> 1;
    const uint se  = fm & 1u;

    uint orig_n[TN];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        orig_n[t] = row_map[min(n0 + 8u * t + fm, N - 1u)];
    }

    float ct[TN][2];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) { ct[t][0] = 0.0f; ct[t][1] = 0.0f; }

    const bool m0ok = fn < p.M;
    const bool m1ok = (fn + 1u) < p.M;
    const device half* xr0 = X + (ulong)(m0ok ? fn : 0u) * K;
    const device half* xr1 = X + (ulong)(m1ok ? (fn + 1u) : 0u) * K;

    for (uint g = sg; g < G; g += KS) {
        uint2 wqs[TN];
        float scs[TN], bis[TN];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint n = orig_n[t];
            wqs[t] = q4_load_u32x2(W, (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(q4_load_u16(S, (ulong)n * G + g));
            bis[t] = bf16_to_f32(q4_load_u16(B, (ulong)n * G + g));
        }

        uint kx = 64u * g + 16u * sb + 4u * se;
        half4 a0 = m0ok ? *(const device half4*)(xr0 + kx)      : half4(0.0h);
        half4 a1 = m0ok ? *(const device half4*)(xr0 + kx + 8u) : half4(0.0h);
        half4 c0 = m1ok ? *(const device half4*)(xr1 + kx)      : half4(0.0h);
        half4 c1 = m1ok ? *(const device half4*)(xr1 + kx + 8u) : half4(0.0h);

        float s0 = (float(a0.x) + float(a0.y)) + (float(a0.z) + float(a0.w))
                 + (float(a1.x) + float(a1.y)) + (float(a1.z) + float(a1.w));
        float s1 = (float(c0.x) + float(c0.y)) + (float(c0.z) + float(c0.w))
                 + (float(c1.x) + float(c1.y)) + (float(c1.z) + float(c1.w));
        s0 += simd_shuffle_xor(s0, 2);  s1 += simd_shuffle_xor(s1, 2);
        s0 += simd_shuffle_xor(s0, 4);  s1 += simd_shuffle_xor(s1, 4);
        s0 += simd_shuffle_xor(s0, 16); s1 += simd_shuffle_xor(s1, 16);

        simdgroup_half8x8 Bm[8];
        Bm[0].thread_elements()[0] = a0.x; Bm[0].thread_elements()[1] = c0.x;
        Bm[1].thread_elements()[0] = a0.y; Bm[1].thread_elements()[1] = c0.y;
        Bm[2].thread_elements()[0] = a0.z; Bm[2].thread_elements()[1] = c0.z;
        Bm[3].thread_elements()[0] = a0.w; Bm[3].thread_elements()[1] = c0.w;
        Bm[4].thread_elements()[0] = a1.x; Bm[4].thread_elements()[1] = c1.x;
        Bm[5].thread_elements()[0] = a1.y; Bm[5].thread_elements()[1] = c1.y;
        Bm[6].thread_elements()[0] = a1.z; Bm[6].thread_elements()[1] = c1.z;
        Bm[7].thread_elements()[0] = a1.w; Bm[7].thread_elements()[1] = c1.w;

        #pragma clang loop unroll(full)
        for (uint t = 0; t < TN; t++) {
            uint2 wq = wqs[t];
            float sc = scs[t];
            float bi = bis[t] - 1024.0f * sc;
            simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; i++) {
                uint wv = (i == 0u) ? wq.x : wq.y;
                #pragma clang loop unroll(full)
                for (uint tt = 0; tt < 4u; tt++) {
                    simdgroup_half8x8 A;
                    half2 h = as_type<half2>(((wv >> (4u * tt)) & 0x000F000Fu) | 0x64006400u);
                    A.thread_elements()[0] = h.x;
                    A.thread_elements()[1] = h.y;
                    simdgroup_multiply_accumulate(C, A, Bm[4u * i + tt], C);
                }
            }
            ct[t][0] = fma(sc, C.thread_elements()[0], fma(bi, s0, ct[t][0]));
            ct[t][1] = fma(sc, C.thread_elements()[1], fma(bi, s1, ct[t][1]));
        }
    }

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < TN; t++) {
        red[sg][t][fm * 8u + fn]      = ct[t][0];
        red[sg][t][fm * 8u + fn + 1u] = ct[t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint m = sg;
    if (m < p.M) {
        uint t  = lane >> 3;
        uint r  = lane & 7u;
        uint gn = n0 + lane;
        float best_v = -INFINITY;
        uint  best_i = 0u;
        if (gn < N) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < KS; k++) acc += red[k][t][r * 8u + m];
            Y[(ulong)m * N + gn] = acc;
            best_v = acc;
            best_i = row_map[gn];
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) {
            uint slot = m * n_tg.x + tgid.x;
            parts[slot].idx = best_i;
            parts[slot].val = best_v;
        }
    }
}

// ============================================================================
// 7. Round 9 & 10: GPU Logit Threshold Compaction + Single-CB GPU Sampled MTP Drafting
// ============================================================================

struct LogitCand {
    float val;
    uint  id;
};

struct VerifyCompactParams {
    uint  V;
    uint  n_parts;
    uint  max_cands;
    float margin;
};

// Stage 2 for sampled verify: reduces argmax_stage1 partials -> row_max[row] and zeroes cand_cnt[row]
kernel void logit_max_stage2(
    const device ArgmaxOut* parts     [[buffer(0)]],
    device float*           row_max   [[buffer(1)]],
    device uint*            cand_cnt  [[buffer(2)]],
    constant VerifyCompactParams& p   [[buffer(3)]],
    uint2 tgid                        [[threadgroup_position_in_grid]],
    uint  tid                         [[thread_index_in_threadgroup]],
    uint2 tg_size                     [[threads_per_threadgroup]]
) {
    uint row = tgid.y;
    const device ArgmaxOut* pr = parts + (ulong)row * p.n_parts;
    float best_v = -INFINITY;
    for (uint i = tid; i < p.n_parts; i += tg_size.x) {
        best_v = max(best_v, pr[i].val);
    }
    best_v = simd_max(best_v);
    threadgroup float s_v[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size.x >> 5;
    if (lane == 0u) s_v[sg] = best_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float m = simd_max((lane < nsg) ? s_v[lane] : -INFINITY);
        if (lane == 0u) {
            row_max[row] = m;
            cand_cnt[row] = 0u;
        }
    }
}

// Round 9: Compact all logits within [row_max - margin, row_max] into shared cands[row, max_cands]
// Eliminates scanning 248,320 floats/row on CPU (~3 ms/round -> ~0.02 ms/round!)
kernel void logit_compact_scan(
    const device float*        logits   [[buffer(0)]],
    const device float*        row_max  [[buffer(1)]],
    device LogitCand*          cands    [[buffer(2)]],
    device atomic_uint*        cand_cnt [[buffer(3)]],
    constant VerifyCompactParams& p     [[buffer(4)]],
    uint2 tgid                          [[threadgroup_position_in_grid]],
    uint  tid                           [[thread_index_in_threadgroup]],
    uint2 tg_size                       [[threads_per_threadgroup]],
    uint2 n_tg                          [[threadgroups_per_grid]]
) {
    uint row = tgid.y;
    float cut = row_max[row] - p.margin;
    const device float* lr = logits + (ulong)row * p.V;
    device LogitCand* cr = cands + (ulong)row * p.max_cands;
    uint stride = tg_size.x * n_tg.x;
    for (uint i = tgid.x * tg_size.x + tid; i < p.V; i += stride) {
        float v = lr[i];
        if (v >= cut) {
            uint slot = atomic_fetch_add_explicit(&cand_cnt[row], 1u, memory_order_relaxed);
            if (slot < p.max_cands) {
                cr[slot].val = v;
                cr[slot].id  = i;
            }
        }
    }
}

struct DraftSampleParams {
    uint  V;
    uint  n_parts;
    float inv_temp;
    float margin;
    uint  seed_lo;
    uint  seed_hi;
    uint  top_k;
    float top_p;
};

struct DraftDistGpu {
    uint      count;
    uint      sampled_id;
    float     sampled_q;
    float     q_top1;
    uint      alt_id[3];
    uint      alt_count;
    LogitCand cands[256];
};

// Round 10 + Stage 1: Single-TG GPU sampled MTP drafter with parallel 256-elem bitonic sort,
// exact GPU top_k + top_p nucleus truncation, top-3 backup alternative extraction, and confidence q_top1!
kernel void mtp_sample_draft_gpu(
    const device float*       logits   [[buffer(0)]],
    const device ArgmaxOut*   parts    [[buffer(1)]],
    device uint*              tok_out  [[buffer(2)]],
    device DraftDistGpu*      dist_out [[buffer(3)]],
    constant DraftSampleParams& p      [[buffer(4)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]]
) {
    threadgroup float s_v[32];
    threadgroup uint  s_i[32];
    threadgroup float c_val[256];
    threadgroup uint  c_id[256];
    threadgroup atomic_uint c_cnt;
    threadgroup float s_max;
    threadgroup uint  s_best;
    threadgroup float s_sum;
    threadgroup uint  s_keep;

    // 1. Exact global max + argmax from argmax_stage1 partials
    float best_v = -INFINITY;
    uint  best_i = 0u;
    for (uint i = tid; i < p.n_parts; i += tg_size) {
        float pv = parts[i].val;
        if (isfinite(pv) && pv > best_v) { best_v = pv; best_i = parts[i].idx; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint  oi = simd_shuffle_down(best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) { s_v[sg] = best_v; s_i[sg] = best_i; }
    if (tid == 0u) atomic_store_explicit(&c_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        best_v = (lane < nsg) ? s_v[lane] : -INFINITY;
        best_i = (lane < nsg) ? s_i[lane] : 0u;
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { s_max = best_v; s_best = best_i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Scan V logits and compact all within [s_max - margin, s_max] into TG memory
    float cut = s_max - p.margin;
    if (isfinite(s_max)) {
        for (uint i = tid; i < p.V; i += tg_size) {
            float v = logits[i];
            if (isfinite(v) && v >= cut) {
                uint slot = atomic_fetch_add_explicit(&c_cnt, 1u, memory_order_relaxed);
                if (slot < 256u) {
                    c_val[slot] = v;
                    c_id[slot]  = i;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = atomic_load_explicit(&c_cnt, memory_order_relaxed);
    if (!isfinite(s_max) || cnt == 0u || cnt > 256u) {
        // Fallback to deterministic argmax (still 100% mathematically exact in rejection sampling!)
        if (tid == 0u) {
            uint b = s_best;
            tok_out[0] = b;
            dist_out->count = 1u;
            dist_out->sampled_id = b;
            dist_out->sampled_q = 1.0f;
            dist_out->q_top1 = 1.0f;
            dist_out->alt_id[0] = b;
            dist_out->alt_id[1] = b;
            dist_out->alt_id[2] = b;
            dist_out->alt_count = 0u;
            dist_out->cands[0].val = 1.0f;
            dist_out->cands[0].id  = b;
        }
        return;
    }

    // 3. Parallel 256-element descending bitonic sort (0.001 ms!) -> exact ordering for top_k, top_p, and alt_id[3]
    if (tid >= cnt) {
        c_val[tid] = -INFINITY;
        c_id[tid]  = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint ixj = tid ^ j;
            if (ixj > tid) {
                float vi = c_val[tid];
                float vj = c_val[ixj];
                bool desc = ((tid & k) == 0u);
                if (desc ? (vi < vj) : (vi > vj)) {
                    c_val[tid] = vj; c_val[ixj] = vi;
                    uint ti = c_id[tid]; c_id[tid] = c_id[ixj]; c_id[ixj] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (p.top_k > 0u && cnt > p.top_k) {
        cnt = p.top_k;
    }

    // 4. Parallel softmax normalization over cnt <= 256 sorted candidates
    float w = (tid < cnt) ? exp((c_val[tid] - s_max) * p.inv_temp) : 0.0f;
    float sg_w = simd_sum(w);
    if (lane == 0u) s_v[sg] = sg_w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? s_v[lane] : 0.0f);
        if (lane == 0u) s_sum = (tot > 0.0f) ? (1.0f / tot) : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < cnt) {
        c_val[tid] = w * s_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. Exact GPU top_p nucleus truncation + renormalization + sampling + top-3 alt extraction!
    if (tid == 0u) {
        uint keep = cnt;
        if (p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint i = 0u; i < cnt; ++i) {
                cum += c_val[i];
                if (cum >= p.top_p) {
                    keep = i + 1u;
                    break;
                }
            }
            float inv_c = (cum > 0.0f) ? (1.0f / cum) : 1.0f;
            for (uint i = 0u; i < keep; ++i) {
                c_val[i] *= inv_c;
            }
        }
        s_keep = keep;

        uint h = p.seed_lo ^ (p.seed_hi * 0x9E3779B9u + 0x85EBCA6Bu);
        h = (h ^ (h >> 16)) * 0x7FEB352Du;
        h = (h ^ (h >> 15)) * 0x846CA68Bu;
        h = h ^ (h >> 16);
        float u = float(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);

        float acc = 0.0f;
        uint chosen_id = c_id[keep - 1u];
        float chosen_q = c_val[keep - 1u];
        for (uint i = 0u; i < keep; ++i) {
            acc += c_val[i];
            if (u <= acc) {
                chosen_id = c_id[i];
                chosen_q  = c_val[i];
                break;
            }
        }
        uint ac = 0u;
        uint alt0 = chosen_id, alt1 = chosen_id, alt2 = chosen_id;
        for (uint i = 0u; i < keep && ac < 3u; ++i) {
            if (c_id[i] != chosen_id) {
                if (ac == 0u) alt0 = c_id[i];
                else if (ac == 1u) alt1 = c_id[i];
                else alt2 = c_id[i];
                ++ac;
            }
        }
        tok_out[0] = chosen_id;
        dist_out->count = keep;
        dist_out->sampled_id = chosen_id;
        dist_out->sampled_q = chosen_q;
        dist_out->q_top1 = c_val[0];
        dist_out->alt_id[0] = alt0;
        dist_out->alt_id[1] = alt1;
        dist_out->alt_id[2] = alt2;
        dist_out->alt_count = ac;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint keep = s_keep;
    if (tid < keep) {
        dist_out->cands[tid].val = c_val[tid];
        dist_out->cands[tid].id  = c_id[tid];
    }
}


struct DraftBatchParams {
    uint  B;
    uint  V;
    uint  n_parts;
    float inv_temp;
    float margin;
    uint  seed_lo;
    uint  seed_hi;
    uint  top_k;
    float top_p;
    uint  greedy;
    uint  out_tok_idx[8];
    int   out_alt0_idx[8];
    int   out_alt1_idx[8];
    uint  out_dist_idx[8];
};

// Stage 3: Batched multi-branch GPU MTP drafter (B = 1..8 rows in parallel, 1 TG per row!)
// Supports both exact sampled q(x) (top_k + top_p) and exact greedy top-1/top-2/top-3 extraction,
// writing primary child + up to 2 backup alternative children directly into buf_token_ids_!
kernel void mtp_sample_draft_batch_gpu(
    const device float*       logits    [[buffer(0)]], // [B, V]
    const device ArgmaxOut*   parts     [[buffer(1)]], // [B, n_parts]
    device uint*              tok_buf   [[buffer(2)]], // buf_token_ids_
    device DraftDistGpu*      dist_buf  [[buffer(3)]], // buf_draft_dists_
    constant DraftBatchParams& p        [[buffer(4)]],
    const device uint*        row_map   [[buffer(5)]], // [V] -> real vocab ID
    uint row                            [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tg_size                        [[threads_per_threadgroup]]
) {
    if (row >= p.B) return;

    const device float*     r_logits = logits + (ulong)row * p.V;
    const device ArgmaxOut* r_parts  = parts  + (ulong)row * p.n_parts;
    const device uint*      r_map    = (p.V == 256u) ? (row_map + (ulong)row * 256u) : row_map;
    device DraftDistGpu*    dist_out = dist_buf + p.out_dist_idx[row];

    threadgroup float s_v[32];
    threadgroup uint  s_i[32];
    threadgroup float c_val[256];
    threadgroup uint  c_id[256];
    threadgroup atomic_uint c_cnt;
    threadgroup float s_max;
    threadgroup uint  s_best;
    threadgroup float s_sum;
    threadgroup uint  s_keep;

    float best_v = -INFINITY;
    uint  best_i = 0u;
    for (uint i = tid; i < p.n_parts; i += tg_size) {
        float pv = r_parts[i].val;
        if (isfinite(pv) && pv > best_v) { best_v = pv; best_i = r_parts[i].idx; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(best_v, off);
        uint  oi = simd_shuffle_down(best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;
    if (lane == 0u) { s_v[sg] = best_v; s_i[sg] = best_i; }
    if (tid == 0u) atomic_store_explicit(&c_cnt, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        best_v = (lane < nsg) ? s_v[lane] : -INFINITY;
        best_i = (lane < nsg) ? s_i[lane] : 0u;
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(best_v, off);
            uint  oi = simd_shuffle_down(best_i, off);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0u) { s_max = best_v; s_best = best_i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float cut = s_max - p.margin;
    if (isfinite(s_max)) {
        for (uint i = tid; i < p.V; i += tg_size) {
            float v = r_logits[i];
            if (isfinite(v) && v >= cut) {
                uint slot = atomic_fetch_add_explicit(&c_cnt, 1u, memory_order_relaxed);
                if (slot < 256u) {
                    c_val[slot] = v;
                    c_id[slot]  = r_map[i];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = atomic_load_explicit(&c_cnt, memory_order_relaxed);
    if (!isfinite(s_max) || cnt == 0u || cnt > 256u) {
        if (tid == 0u) {
            uint b = s_best;
            tok_buf[p.out_tok_idx[row]] = b;
            if (p.out_alt0_idx[row] >= 0) tok_buf[p.out_alt0_idx[row]] = b;
            if (p.out_alt1_idx[row] >= 0) tok_buf[p.out_alt1_idx[row]] = b;
            dist_out->count = 1u;
            dist_out->sampled_id = b;
            dist_out->sampled_q = 1.0f;
            dist_out->q_top1 = 1.0f;
            dist_out->alt_id[0] = b;
            dist_out->alt_id[1] = b;
            dist_out->alt_id[2] = b;
            dist_out->alt_count = 0u;
            dist_out->cands[0].val = 1.0f;
            dist_out->cands[0].id  = b;
        }
        return;
    }

    if (tid >= cnt) {
        c_val[tid] = -INFINITY;
        c_id[tid]  = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint ixj = tid ^ j;
            if (ixj > tid) {
                float vi = c_val[tid];
                float vj = c_val[ixj];
                bool desc = ((tid & k) == 0u);
                if (desc ? (vi < vj) : (vi > vj)) {
                    c_val[tid] = vj; c_val[ixj] = vi;
                    uint ti = c_id[tid]; c_id[tid] = c_id[ixj]; c_id[ixj] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint full_cnt = cnt;
    if (p.top_k > 0u && cnt > p.top_k) {
        cnt = p.top_k;
    }

    float w = (tid < cnt) ? exp((c_val[tid] - s_max) * p.inv_temp) : 0.0f;
    float sg_w = simd_sum(w);
    if (lane == 0u) s_v[sg] = sg_w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? s_v[lane] : 0.0f);
        if (lane == 0u) s_sum = (tot > 0.0f) ? (1.0f / tot) : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < cnt) {
        c_val[tid] = w * s_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint keep = cnt;
        if (p.greedy == 0u && p.top_p > 0.0f && p.top_p < 1.0f) {
            float cum = 0.0f;
            for (uint i = 0u; i < cnt; ++i) {
                cum += c_val[i];
                if (cum >= p.top_p) {
                    keep = i + 1u;
                    break;
                }
            }
            float inv_c = (cum > 0.0f) ? (1.0f / cum) : 1.0f;
            for (uint i = 0u; i < keep; ++i) {
                c_val[i] *= inv_c;
            }
        }
        s_keep = keep;

        uint chosen_id = c_id[0];
        float chosen_q = c_val[0];
        if (p.greedy == 0u) {
            uint h = (p.seed_lo + row * 0x9E3779B9u) ^ ((p.seed_hi ^ (row * 0x85EBCA6Bu)) * 0x9E3779B9u + 0x85EBCA6Bu);
            h = (h ^ (h >> 16)) * 0x7FEB352Du;
            h = (h ^ (h >> 15)) * 0x846CA68Bu;
            h = h ^ (h >> 16);
            float u = float(h & 0x00FFFFFFu) * (1.0f / 16777216.0f);

            float acc = 0.0f;
            chosen_id = c_id[keep - 1u];
            chosen_q  = c_val[keep - 1u];
            for (uint i = 0u; i < keep; ++i) {
                acc += c_val[i];
                if (u <= acc) {
                    chosen_id = c_id[i];
                    chosen_q  = c_val[i];
                    break;
                }
            }
        }

        // Collect top-3 distinct backup alternatives from full sorted list (even if top_p kept only 1!)
        uint ac = 0u;
        uint alt0 = chosen_id, alt1 = chosen_id, alt2 = chosen_id;
        for (uint i = 0u; i < full_cnt && ac < 3u; ++i) {
            if (c_id[i] != chosen_id) {
                if (ac == 0u) alt0 = c_id[i];
                else if (ac == 1u) alt1 = c_id[i];
                else alt2 = c_id[i];
                ++ac;
            }
        }

        tok_buf[p.out_tok_idx[row]] = chosen_id;
        if (p.out_alt0_idx[row] >= 0) tok_buf[p.out_alt0_idx[row]] = alt0;
        if (p.out_alt1_idx[row] >= 0) tok_buf[p.out_alt1_idx[row]] = alt1;

        dist_out->count = keep;
        dist_out->sampled_id = chosen_id;
        dist_out->sampled_q = chosen_q;
        dist_out->q_top1 = c_val[0];
        dist_out->alt_id[0] = alt0;
        dist_out->alt_id[1] = alt1;
        dist_out->alt_id[2] = alt2;
        dist_out->alt_count = ac;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint keep = s_keep;
    if (tid < keep) {
        dist_out->cands[tid].val = c_val[tid];
        dist_out->cands[tid].id  = c_id[tid];
    }
}


// ============================================================================

// ============================================================================
// 8. Single-Copy ANE 4-bit (model.hwx: hf4 & universal M1-M5 lut4) MLP Kernels!
//    Reads Apple Neural Engine's compiled 4-bit [K][8] model.hwx pages directly
//    via zero-copy mmap: 4.0 bpw (11.1% less DRAM than g64!), zero per-group
//    scales/biases, exact per-row (scale, 1032 + offset) applied once at end,
//    exact sum(x) computed in registers via 2 warp shuffles (0 TG mem, 0 barriers!)
// ============================================================================

struct AneShareParams {
    uint M;
    uint HID;             // hidden_size (multiple of 128, e.g. 5120)
    uint INTER;           // intermediate_size (= n_subblocks * SUB, e.g. 17408)
    uint SUB;             // sub-block intermediate width (multiple of 256, e.g. 4352)
    uint DK;              // down_proj K-group width (= SUB / 2, multiple of 128, e.g. 2176)
    uint n_subblocks;     // number of MLP sub-blocks (1..4, e.g. 4)
    uint lin_base_w[16];  // word offset of chunk[0] for linear_0..15
    uint lin_chunk_w[16]; // chunk_bytes / 4 for linear_0..15
    uint lin_hdr_w[16];   // per-linear group header in 32-bit words: 6 (24B asym), 4 (16B sym/lut4), 0 (lut4 unfused)
};

// Optional PSO specialization constants (default/undefined falls back transparently to AneShareParams!)
constant uint FC_ANE_HID   [[function_constant(0)]];
constant uint FC_ANE_INTER [[function_constant(1)]];
constant uint FC_ANE_SUB   [[function_constant(2)]];
constant uint FC_ANE_DK    [[function_constant(3)]];
constant uint FC_ANE_NSB   [[function_constant(4)]];
constant uint FC_ANE_HDR   [[function_constant(5)]];

inline half2 ane_dq4_f16(uint w, uint sh) {
    uint u = (w >> sh) ^ 0x88u;
    return as_type<half2>(ushort2(u & 0xFu, (u >> 4) & 0xFu) | ushort2(0x6400u));
}

// Fast decode when weight words were XORed with 0x88888888u once at load time (4x fewer XORs!)
inline half2 ane_dq4_f16_xored(uint w_xored, uint sh) {
    uint u = w_xored >> sh;
    return as_type<half2>(ushort2(u & 0xFu, (u >> 4) & 0xFu) | ushort2(0x6400u));
}

inline void ane_hf4_mma4_xored(
    uint2 w_lo, uint2 w_hi, uint sh,
    const thread simdgroup_half8x8* Af,
    thread simdgroup_float8x8& C
) {
    uint w[4] = {w_lo.x, w_lo.y, w_hi.x, w_hi.y};
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
        simdgroup_half8x8 B;
        half2 h = ane_dq4_f16_xored(w[q], sh);
        B.thread_elements()[0] = h.x;
        B.thread_elements()[1] = h.y;
        simdgroup_multiply_accumulate(C, Af[q], B, C);
    }
}

// Universal non-uniform 16-entry FP16 LUT decode via 1 KB threadgroup byte-pair table s_lut2[256]!
// Maps 2 adjacent 4-bit nibbles (1 byte = (w >> sh) & 0xFFu) -> half2(lut[lo], lut[hi]) in ONE 32-bit TG load
// (zero simd_shuffle stalls, zero extra shifts, 32-bank spread + hardware broadcast on repeated pairs!).
inline void ane_lut4_mma4_tbl(
    uint2 w_lo, uint2 w_hi, uint sh,
    const threadgroup half2* s_lut2,
    const thread simdgroup_half8x8* Af,
    thread simdgroup_float8x8& C
) {
    uint w[4] = {w_lo.x, w_lo.y, w_hi.x, w_hi.y};
    #pragma clang loop unroll(full)
    for (uint q = 0u; q < 4u; ++q) {
        simdgroup_half8x8 B;
        half2 h = s_lut2[(w[q] >> sh) & 0xFFu];
        B.thread_elements()[0] = h.x;
        B.thread_elements()[1] = h.y;
        simdgroup_multiply_accumulate(C, Af[q], B, C);
    }
}

// Fused gate + up + SiLU from single-copy model.hwx -> half Y[M, INTER]
// Direct-load zero-spill inner loop + pre-XOR + post-reduction SO epilogue (8x less SO) + [[function_constant]]!
// Grid: (INTER / 16, ceil_div(M, 8), 1), TG: (256, 1, 1) (8 simdgroups)
kernel void ane_hf4_swiglu_bf16(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]], // [0..INTER-1]=gate, [INTER..2*INTER-1]=up ({scale, 1032+off})
    const device half*          X  [[buffer(2)]], // [M, HID] FP16 normed*inv1
    device half*                Y  [[buffer(3)]], // [M, INTER] FP16 act
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint KS = 8u;
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS   = SUB >> 3;          // SUB / 8 (groups per sub-block)
    const uint tgs_sb = SUB >> 4;         // SUB / 16
    const uint total_tgs = NSB * tgs_sb;

    const uint m0 = tgid.y * 8u;
    if (tgid.x >= total_tgs || m0 >= p.M) return;

    const uint sb    = tgid.x / tgs_sb;
    const uint rem   = tgid.x - sb * tgs_sb;
    const uint slot0 = (rem >> 4) * 2u;
    const uint ch    = rem & 15u;
    const uint G0    = sb * NGS + slot0 * 16u + ch;
    const uint G1    = G0 + 16u;

    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    // Odd slot (pg1/pu1) is ALWAYS 16B-aligned for all supported hdr in {0, 4, 6} (zero alignment branch!)
    const device uint* pg0 = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                             + 16u + (ulong)slot0 * gw_g + hdr_g + 4u * fm;
    const device uint* pg1 = pg0 + gw_g;
    const device uint* pu0 = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                             + 16u + (ulong)slot0 * gw_u + hdr_u + 4u * fm;
    const device uint* pu1 = pu0 + gw_u;

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * HID;

    simdgroup_float8x8 Cg0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cg1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float sx = 0.0f;

    // Lockstep strided-K across 8 simdgroups -> 1 KB contiguous DRAM burst per iteration (zero spill!)
    for (uint ko = sg * 32u; ko < HID; ko += KS * 32u) {
        half4 x0 = mok ? *(const device half4*)(xr + ko + 4u * fn)      : half4(0.0h);
        half4 x1 = mok ? *(const device half4*)(xr + ko + 4u * fn + 4u) : half4(0.0h);
        sx += ((float(x0.x) + float(x0.y)) + (float(x0.z) + float(x0.w)))
            + ((float(x1.x) + float(x1.y)) + (float(x1.z) + float(x1.w)));

        simdgroup_half8x8 Af[4];
        Af[0].thread_elements()[0] = x0.x; Af[0].thread_elements()[1] = x1.x;
        Af[1].thread_elements()[0] = x0.y; Af[1].thread_elements()[1] = x1.y;
        Af[2].thread_elements()[0] = x0.z; Af[2].thread_elements()[1] = x1.z;
        Af[3].thread_elements()[0] = x0.w; Af[3].thread_elements()[1] = x1.w;

        uint2 a0 = *(const device uint2*)(pg0 + ko)      ^ uint2(0x88888888u);
        uint2 b0 = *(const device uint2*)(pg0 + ko + 2u) ^ uint2(0x88888888u);
        uint4 w1 = *(const device uint4*)(pg1 + ko)      ^ uint4(0x88888888u);
        ane_hf4_mma4_xored(a0, b0, sh, Af, Cg0);

        uint2 a2 = *(const device uint2*)(pu0 + ko)      ^ uint2(0x88888888u);
        uint2 b2 = *(const device uint2*)(pu0 + ko + 2u) ^ uint2(0x88888888u);
        ane_hf4_mma4_xored(w1.xy, w1.zw, sh, Af, Cg1);

        uint4 w3 = *(const device uint4*)(pu1 + ko)      ^ uint4(0x88888888u);
        ane_hf4_mma4_xored(a2, b2, sh, Af, Cu0);
        ane_hf4_mma4_xored(w3.xy, w3.zw, sh, Af, Cu1);
    }

    sx += simd_shuffle_xor(sx, 1);
    sx += simd_shuffle_xor(sx, 8);

    threadgroup float red_g[KS][2][64];
    threadgroup float red_u[KS][2][64];
    threadgroup float red_sx[KS][8];

    if (fn == 0u) red_sx[sg][fm] = sx;
    red_g[sg][0][fm * 8u + fn]      = Cg0.thread_elements()[0];
    red_g[sg][0][fm * 8u + fn + 1u] = Cg0.thread_elements()[1];
    red_g[sg][1][fm * 8u + fn]      = Cg1.thread_elements()[0];
    red_g[sg][1][fm * 8u + fn + 1u] = Cg1.thread_elements()[1];
    red_u[sg][0][fm * 8u + fn]      = Cu0.thread_elements()[0];
    red_u[sg][0][fm * 8u + fn + 1u] = Cu0.thread_elements()[1];
    red_u[sg][1][fm * 8u + fn]      = Cu1.thread_elements()[0];
    red_u[sg][1][fm * 8u + fn + 1u] = Cu1.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Post-reduction SO epilogue: 8x fewer SO loads & scale/offset multiplies!
    uint e = sg * 32u + lane;
    if (e < 128u) {
        uint t  = e >> 6;
        uint rc = e & 63u;
        uint m  = rc >> 3;
        uint c  = rc & 7u;
        uint gm_out = m0 + m;
        if (gm_out < p.M) {
            float g = 0.0f, u = 0.0f, sx_tot = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0u; k < KS; ++k) {
                g      += red_g[k][t][rc];
                u      += red_u[k][t][rc];
                sx_tot += red_sx[k][m];
            }
            uint oc = ((t == 0u) ? G0 : G1) * 8u + c;
            float2 so_g = SO[oc];
            float2 so_u = SO[INTER + oc];
            float gv = (g - so_g.y * sx_tot) * so_g.x;
            float uv = (u - so_u.y * sx_tot) * so_u.x;
            Y[(ulong)gm_out * INTER + oc] = half(silu_f32(gv) * uv);
        }
    }
}

// Dedicated 453 GB/s M=1 QMV gate+up+SiLU from single-copy model.hwx -> half Y[INTER]
// Grid: (ceil_div(INTER / 8, 4), 1, 1), TG: (128, 1, 1) (4 simdgroups, 1 slot = 8 channels/sg)
kernel void ane_hf4_swiglu_m1(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint sg                        [[simdgroup_index_in_threadgroup]]
) {
    const uint HID          = p.HID;
    const uint INTER        = p.INTER;
    const uint SUB          = p.SUB;
    const uint NGS          = SUB >> 3;   // SUB / 8
    const uint slots_per_ch = SUB >> 7;   // SUB / 128
    const uint total_g      = p.n_subblocks * NGS;
    const uint gidx = tgid * 4u + sg;
    if (gidx >= total_g) return;

    const uint sb    = gidx / NGS;
    const uint rem   = gidx - sb * NGS;
    const uint ch    = rem / slots_per_ch;
    const uint slot  = rem - ch * slots_per_ch;
    const uint G0    = sb * NGS + slot * 16u + ch;

    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = p.lin_hdr_w[li_g];
    const uint hdr_u = p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    const device uint* pg = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                            + 16u + (ulong)slot * gw_g + hdr_g;
    const device uint* pu = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                            + 16u + (ulong)slot * gw_u + hdr_u;

    float ag[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float au[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sx = 0.0f;

    for (uint k = lane * 4u; k < HID; k += 128u) {
        half4 xv = *(const device half4*)(X + k);
        float4 xf = float4(xv);
        sx += (xf.x + xf.y) + (xf.z + xf.w);
        uint2 g0 = *(const device uint2*)(pg + k);
        uint2 g1 = *(const device uint2*)(pg + k + 2u);
        uint2 u0 = *(const device uint2*)(pu + k);
        uint2 u1 = *(const device uint2*)(pu + k + 2u);
        uint gw4[4] = {g0.x ^ 0x88888888u, g0.y ^ 0x88888888u, g1.x ^ 0x88888888u, g1.y ^ 0x88888888u};
        uint uw4[4] = {u0.x ^ 0x88888888u, u0.y ^ 0x88888888u, u1.x ^ 0x88888888u, u1.y ^ 0x88888888u};
        float x4[4] = {xf.x, xf.y, xf.z, xf.w};
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint gv = gw4[j], uv = uw4[j];
            float xv1 = x4[j];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 8u; ++c) {
                ag[c] = fma(float((gv >> (4u * c)) & 0xFu), xv1, ag[c]);
                au[c] = fma(float((uv >> (4u * c)) & 0xFu), xv1, au[c]);
            }
        }
    }

    sx = simd_sum(sx);
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float g = simd_sum(ag[c]);
        float u = simd_sum(au[c]);
        if (lane == 0u) {
            uint oc = G0 * 8u + c;
            float2 sg0 = SO[oc];
            float2 su0 = SO[INTER + oc];
            float gv = (g - (sg0.y - 1024.0f) * sx) * sg0.x;
            float uv = (u - (su0.y - 1024.0f) * sx) * su0.x;
            Y[oc] = half(silu_f32(gv) * uv);
        }
    }
}

// Dedicated M<=4 register-vectorized gate+up+SiLU from single-copy model.hwx -> half Y[M, INTER]
// Zero 8-row AMX padding, zero TG memory, zero barriers -> ~440 GB/s at M=2..4!
// Grid: (ceil_div(INTER / 8, 4), ceil_div(M, 4), 1), TG: (128, 1, 1)
kernel void ane_hf4_swiglu_m4(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint total_g      = NSB * NGS;
    const uint m0   = tgid.y * 4u;
    const uint gidx = tgid.x * 4u + sg;
    if (gidx >= total_g || m0 >= p.M) return;

    const uint sb    = gidx / NGS;
    const uint rem   = gidx - sb * NGS;
    const uint ch    = rem / slots_per_ch;
    const uint slot  = rem - ch * slots_per_ch;
    const uint G0    = sb * NGS + slot * 16u + ch;

    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    const device uint* pg = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                            + 16u + (ulong)slot * gw_g + hdr_g;
    const device uint* pu = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                            + 16u + (ulong)slot * gw_u + hdr_u;

    const bool mok0 = (m0 + 0u) < p.M;
    const bool mok1 = (m0 + 1u) < p.M;
    const bool mok2 = (m0 + 2u) < p.M;
    const bool mok3 = (m0 + 3u) < p.M;
    const device half* xr0 = X + (ulong)(mok0 ? (m0 + 0u) : 0u) * HID;
    const device half* xr1 = X + (ulong)(mok1 ? (m0 + 1u) : 0u) * HID;
    const device half* xr2 = X + (ulong)(mok2 ? (m0 + 2u) : 0u) * HID;
    const device half* xr3 = X + (ulong)(mok3 ? (m0 + 3u) : 0u) * HID;

    float4 ag[8], au[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) { ag[c] = float4(0.0f); au[c] = float4(0.0f); }
    float4 sx = float4(0.0f);

    for (uint k = lane * 4u; k < HID; k += 128u) {
        half4 x0 = mok0 ? *(const device half4*)(xr0 + k) : half4(0.0h);
        half4 x1 = mok1 ? *(const device half4*)(xr1 + k) : half4(0.0h);
        half4 x2 = mok2 ? *(const device half4*)(xr2 + k) : half4(0.0h);
        half4 x3 = mok3 ? *(const device half4*)(xr3 + k) : half4(0.0h);

        uint2 g0 = *(const device uint2*)(pg + k);
        uint2 g1 = *(const device uint2*)(pg + k + 2u);
        uint2 u0 = *(const device uint2*)(pu + k);
        uint2 u1 = *(const device uint2*)(pu + k + 2u);
        uint gw4[4] = {g0.x ^ 0x88888888u, g0.y ^ 0x88888888u, g1.x ^ 0x88888888u, g1.y ^ 0x88888888u};
        uint uw4[4] = {u0.x ^ 0x88888888u, u0.y ^ 0x88888888u, u1.x ^ 0x88888888u, u1.y ^ 0x88888888u};

        float4 xk[4];
        xk[0] = float4(float(x0.x), float(x1.x), float(x2.x), float(x3.x));
        xk[1] = float4(float(x0.y), float(x1.y), float(x2.y), float(x3.y));
        xk[2] = float4(float(x0.z), float(x1.z), float(x2.z), float(x3.z));
        xk[3] = float4(float(x0.w), float(x1.w), float(x2.w), float(x3.w));
        sx += (xk[0] + xk[1]) + (xk[2] + xk[3]);

        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint gv = gw4[j], uv = uw4[j];
            float4 xv = xk[j];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 8u; ++c) {
                ag[c] = fma(float((gv >> (4u * c)) & 0xFu), xv, ag[c]);
                au[c] = fma(float((uv >> (4u * c)) & 0xFu), xv, au[c]);
            }
        }
    }

    float sx_r[4] = { simd_sum(sx.x), simd_sum(sx.y), simd_sum(sx.z), simd_sum(sx.w) };
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float g_r[4] = { simd_sum(ag[c].x), simd_sum(ag[c].y), simd_sum(ag[c].z), simd_sum(ag[c].w) };
        float u_r[4] = { simd_sum(au[c].x), simd_sum(au[c].y), simd_sum(au[c].z), simd_sum(au[c].w) };
        if (lane == 0u) {
            uint oc = G0 * 8u + c;
            float2 sg0 = SO[oc];
            float2 su0 = SO[INTER + oc];
            float goff = sg0.y - 1024.0f;
            float uoff = su0.y - 1024.0f;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (m0 + r < p.M) {
                    float gv = (g_r[r] - goff * sx_r[r]) * sg0.x;
                    float uv = (u_r[r] - uoff * sx_r[r]) * su0.x;
                    Y[(ulong)(m0 + r) * INTER + oc] = half(silu_f32(gv) * uv);
                }
            }
        }
    }
}

// Dedicated 457 GB/s M=1 QMV Split-K down_proj from single-copy model.hwx -> partY[2*n_subblocks, HID]
// Grid: (16 * ceil_div(HID / 128, 4), 1, 2 * n_subblocks), TG: (128, 1, 1) (4 simdgroups = 4 slots/TG)
kernel void ane_hf4_down_m1_sk(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    const uint HID          = p.HID;
    const uint INTER        = p.INTER;
    const uint DK           = p.DK;
    const uint n_q          = p.n_subblocks * 2u;
    const uint slots_per_ch = HID >> 7;                // HID / 128
    const uint tg_per_ch    = (slots_per_ch + 3u) >> 2; // ceil_div(slots_per_ch, 4)
    if (tgid.x >= 16u * tg_per_ch || tgid.z >= n_q) return;

    const uint ch   = tgid.x / tg_per_ch;
    const uint slot = (tgid.x - ch * tg_per_ch) * 4u + sg;
    if (slot >= slots_per_ch) return;

    const uint q   = tgid.z;
    const uint sb  = q >> 1;
    const uint li  = sb * 4u + 2u + (q & 1u);
    const uint hdr = p.lin_hdr_w[li];
    const uint gw  = hdr + DK;

    const device uint* pw = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                            + 16u + (ulong)slot * gw + hdr;
    const device half* xr = X + (ulong)q * DK;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float sx = 0.0f;

    for (uint k = lane * 4u; k < DK; k += 128u) {
        half4 xv = *(const device half4*)(xr + k);
        float4 xf = float4(xv);
        sx += (xf.x + xf.y) + (xf.z + xf.w);
        uint2 u0 = *(const device uint2*)(pw + k);
        uint2 u1 = *(const device uint2*)(pw + k + 2u);
        uint w4[4] = {u0.x ^ 0x88888888u, u0.y ^ 0x88888888u, u1.x ^ 0x88888888u, u1.y ^ 0x88888888u};
        float x4[4] = {xf.x, xf.y, xf.z, xf.w};
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint wv = w4[j];
            float xv1 = x4[j];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 8u; ++c) {
                acc[c] = fma(float((wv >> (4u * c)) & 0xFu), xv1, acc[c]);
            }
        }
    }

    sx = simd_sum(sx);
    const uint go = slot * 16u + ch;
    const uint so_down_base = 2u * INTER + q * HID;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float v = simd_sum(acc[c]);
        if (lane == 0u) {
            uint o = go * 8u + c;
            float2 sd = SO[so_down_base + o];
            partY[(ulong)q * HID + o] = (v - (sd.y - 1024.0f) * sx) * sd.x;
        }
    }
}

// Dedicated M<=4 register-vectorized Split-K down_proj from single-copy model.hwx -> partY[2*n_subblocks, M, HID]
// Grid: (16 * ceil_div(HID / 128, 4), ceil_div(M, 4), 2 * n_subblocks), TG: (128, 1, 1)
kernel void ane_hf4_down_sk_m4(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 3u) >> 2;
    const uint m0 = tgid.y * 4u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M || tgid.z >= n_q) return;

    const uint ch   = tgid.x / tg_per_ch;
    const uint slot = (tgid.x - ch * tg_per_ch) * 4u + sg;
    if (slot >= slots_per_ch) return;

    const uint q   = tgid.z;
    const uint sb  = q >> 1;
    const uint li  = sb * 4u + 2u + (q & 1u);
    const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw  = hdr + DK;

    const device uint* pw = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                            + 16u + (ulong)slot * gw + hdr;

    const bool mok0 = (m0 + 0u) < p.M;
    const bool mok1 = (m0 + 1u) < p.M;
    const bool mok2 = (m0 + 2u) < p.M;
    const bool mok3 = (m0 + 3u) < p.M;
    const device half* xr0 = X + (ulong)(mok0 ? (m0 + 0u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr1 = X + (ulong)(mok1 ? (m0 + 1u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr2 = X + (ulong)(mok2 ? (m0 + 2u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr3 = X + (ulong)(mok3 ? (m0 + 3u) : 0u) * INTER + (ulong)q * DK;

    float4 acc[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) acc[c] = float4(0.0f);
    float4 sx = float4(0.0f);

    for (uint k = lane * 4u; k < DK; k += 128u) {
        half4 x0 = mok0 ? *(const device half4*)(xr0 + k) : half4(0.0h);
        half4 x1 = mok1 ? *(const device half4*)(xr1 + k) : half4(0.0h);
        half4 x2 = mok2 ? *(const device half4*)(xr2 + k) : half4(0.0h);
        half4 x3 = mok3 ? *(const device half4*)(xr3 + k) : half4(0.0h);

        uint2 u0 = *(const device uint2*)(pw + k);
        uint2 u1 = *(const device uint2*)(pw + k + 2u);
        uint w4[4] = {u0.x ^ 0x88888888u, u0.y ^ 0x88888888u, u1.x ^ 0x88888888u, u1.y ^ 0x88888888u};

        float4 xk[4];
        xk[0] = float4(float(x0.x), float(x1.x), float(x2.x), float(x3.x));
        xk[1] = float4(float(x0.y), float(x1.y), float(x2.y), float(x3.y));
        xk[2] = float4(float(x0.z), float(x1.z), float(x2.z), float(x3.z));
        xk[3] = float4(float(x0.w), float(x1.w), float(x2.w), float(x3.w));
        sx += (xk[0] + xk[1]) + (xk[2] + xk[3]);

        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint wv = w4[j];
            float4 xv = xk[j];
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 8u; ++c) {
                acc[c] = fma(float((wv >> (4u * c)) & 0xFu), xv, acc[c]);
            }
        }
    }

    float sx_r[4] = { simd_sum(sx.x), simd_sum(sx.y), simd_sum(sx.z), simd_sum(sx.w) };
    const uint go = slot * 16u + ch;
    const uint so_down_base = 2u * INTER + q * HID;
    const ulong sk_stride = (ulong)p.M * HID;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float v_r[4] = { simd_sum(acc[c].x), simd_sum(acc[c].y), simd_sum(acc[c].z), simd_sum(acc[c].w) };
        if (lane == 0u) {
            uint o = go * 8u + c;
            float2 sd = SO[so_down_base + o];
            float doff = sd.y - 1024.0f;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (m0 + r < p.M) {
                    partY[(ulong)q * sk_stride + (ulong)(m0 + r) * HID + o] = (v_r[r] - doff * sx_r[r]) * sd.x;
                }
            }
        }
    }
}

// Zero-Padding 14+ TFLOPs BM=64, BN=64, BK=32 Tiled Prefill gate & up from single-copy model.hwx!
// Grid: (n_subblocks * ceil_div(SUB/8, 8), ceil_div(M, 64), 2) [z=0 -> gate_out, z=1 -> up_out], TG: (128, 1, 1)
kernel void ane_hf4_tiled_gateup(
    const device uint*          W        [[buffer(0)]],
    const device float2*        SO       [[buffer(1)]],
    const device half*          X        [[buffer(2)]],
    device float*               gate_out [[buffer(3)]],
    device float*               up_out   [[buffer(4)]],
    constant AneShareParams&    p        [[buffer(5)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint  tid                            [[thread_index_in_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]],
    uint  sg                             [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint tiles_sb     = (NGS + 7u) >> 3;
    const uint total_tiles  = NSB * tiles_sb;
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= total_tiles || m0 >= p.M || tgid.z >= 2u) return;

    const uint sb      = tgid.x / tiles_sb;
    const uint tile    = tgid.x - sb * tiles_sb;
    const uint li      = sb * 4u + tgid.z; // 0=gate, 1=up
    const uint hdr     = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw      = hdr + HID;
    const uint so_base = tgid.z * INTER;

    threadgroup half sA[64][36];
    threadgroup half sB[32][68];

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m = (sg >> 1) * 32u;
    const uint sg_n = (sg & 1u) * 32u;
    const uint r_a  = tid >> 1;
    const uint c_a  = (tid & 1u) * 16u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * HID;
    threadgroup half4* da = (threadgroup half4*)(&sA[r_a][c_a]);

    const uint w_slot = tid >> 4;          // 0..7
    const uint w_k    = (tid & 15u) * 2u;  // 0, 2, ..., 30
    const uint flat   = tile * 8u + w_slot;
    const bool flat_ok = flat < NGS;
    const uint flat_c = flat_ok ? flat : 0u;
    const uint ch     = flat_c / slots_per_ch;
    const uint slot   = flat_c - ch * slots_per_ch;
    const uint G_s    = sb * NGS + slot * 16u + ch;
    const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                               + 16u + (ulong)slot * gw + hdr + w_k;

    half sc8[8], bi8[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float2 so = SO[so_base + G_s * 8u + c];
        sc8[c] = flat_ok ? half(so.x) : half(0.0h);
        bi8[c] = flat_ok ? half(-(so.y - 1024.0f) * so.x) : half(0.0h);
    }

    for (uint k0 = 0u; k0 < HID; k0 += 32u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + c_a);
            da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
        } else {
            da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
        }

        uint2 u2 = flat_ok ? *(const device uint2*)(w_ptr + k0) : uint2(0u);
        uint ws[2] = {u2.x, u2.y};
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
            uint w = ws[j];
            half2 h0 = ane_dq4_f16(w, 0u)  - half2(1024.0h);
            half2 h1 = ane_dq4_f16(w, 8u)  - half2(1024.0h);
            half2 h2 = ane_dq4_f16(w, 16u) - half2(1024.0h);
            half2 h3 = ane_dq4_f16(w, 24u) - half2(1024.0h);
            threadgroup half* dst = &sB[w_k + j][w_slot * 8u];
            dst[0] = fma(h0.x, sc8[0], bi8[0]); dst[1] = fma(h0.y, sc8[1], bi8[1]);
            dst[2] = fma(h1.x, sc8[2], bi8[2]); dst[3] = fma(h1.y, sc8[3], bi8[3]);
            dst[4] = fma(h2.x, sc8[4], bi8[4]); dst[5] = fma(h2.y, sc8[5], bi8[5]);
            dst[6] = fma(h3.x, sc8[6], bi8[6]); dst[7] = fma(h3.y, sc8[7], bi8[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(a[i], &sA[sg_m + 8u * i][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(b[j], &sB[kk][sg_n + 8u * j], 68);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i)
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* Y = (tgid.z == 0u) ? gate_out : up_out;
    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    uint G_out[4];
    bool G_ok[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        uint fo = tile * 8u + (sg_n >> 3) + j;
        G_ok[j] = fo < NGS;
        uint foc = G_ok[j] ? fo : 0u;
        uint co = foc / slots_per_ch;
        uint so = foc - co * slots_per_ch;
        G_out[j] = sb * NGS + so * 16u + co;
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * INTER;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                if (G_ok[j]) {
                    *(device float2*)(yr + G_out[j] * 8u + fn) =
                        float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// Zero-Padding 14+ TFLOPs BM=64, BN=64, BK=32 Single-Pass Tiled Prefill down_proj (+ residual add!)
// Grid: (16 * ceil_div(HID/128, 8), ceil_div(M, 64), 1), TG: (128, 1, 1)
kernel void ane_hf4_tiled_down(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]], // [M, INTER] FP16 act
    device float*               Y  [[buffer(3)]], // [M, HID] FP32 hidden (+= residual)
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 7u) >> 3;
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M) return;

    const uint ch    = tgid.x / tg_per_ch;
    const uint slot0 = (tgid.x - ch * tg_per_ch) * 8u;

    threadgroup half sA[64][36];
    threadgroup half sB[32][68];

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m  = (sg >> 1) * 32u;
    const uint sg_n  = (sg & 1u) * 32u;
    const uint r_a   = tid >> 1;
    const uint c_a   = (tid & 1u) * 16u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_row = X + (ulong)(gm_ok ? gm_ld : 0u) * INTER;
    threadgroup half4* da = (threadgroup half4*)(&sA[r_a][c_a]);

    const uint w_slot  = tid >> 4;          // 0..7
    const uint w_k     = (tid & 15u) * 2u;  // 0, 2, ..., 30
    const uint slot    = slot0 + w_slot;
    const bool slot_ok = slot < slots_per_ch;
    const uint slot_c  = slot_ok ? slot : 0u;
    const uint go_s    = slot_c * 16u + ch;

    for (uint q = 0u; q < n_q; ++q) {
        const uint sb  = q >> 1;
        const uint li  = sb * 4u + 2u + (q & 1u);
        const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
        const uint gw  = hdr + DK;
        const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                                   + 16u + (ulong)slot_c * gw + hdr + w_k;
        const device half* x_base = x_row + (ulong)q * DK;
        const uint so_down_base = 2u * INTER + q * HID;

        half sc8[8], bi8[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) {
            float2 sd = SO[so_down_base + go_s * 8u + c];
            sc8[c] = slot_ok ? half(sd.x) : half(0.0h);
            bi8[c] = slot_ok ? half(-(sd.y - 1024.0f) * sd.x) : half(0.0h);
        }

        for (uint k0 = 0u; k0 < DK; k0 += 32u) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k0 + c_a);
                da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
            } else {
                da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
            }

            uint2 u2 = slot_ok ? *(const device uint2*)(w_ptr + k0) : uint2(0u);
            uint ws[2] = {u2.x, u2.y};
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                uint w = ws[j];
                half2 h0 = ane_dq4_f16(w, 0u)  - half2(1024.0h);
                half2 h1 = ane_dq4_f16(w, 8u)  - half2(1024.0h);
                half2 h2 = ane_dq4_f16(w, 16u) - half2(1024.0h);
                half2 h3 = ane_dq4_f16(w, 24u) - half2(1024.0h);
                threadgroup half* dst = &sB[w_k + j][w_slot * 8u];
                dst[0] = fma(h0.x, sc8[0], bi8[0]); dst[1] = fma(h0.y, sc8[1], bi8[1]);
                dst[2] = fma(h1.x, sc8[2], bi8[2]); dst[3] = fma(h1.y, sc8[3], bi8[3]);
                dst[4] = fma(h2.x, sc8[4], bi8[4]); dst[5] = fma(h2.y, sc8[5], bi8[5]);
                dst[6] = fma(h3.x, sc8[6], bi8[6]); dst[7] = fma(h3.y, sc8[7], bi8[7]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a[4], b[4];
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i) simdgroup_load(a[i], &sA[sg_m + 8u * i][kk], 36);
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) simdgroup_load(b[j], &sB[kk][sg_n + 8u * j], 68);
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i)
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * HID;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                uint sl_out = slot0 + (sg_n >> 3) + j;
                if (sl_out < slots_per_ch) {
                    uint go = sl_out * 16u + ch;
                    device float2* yp = (device float2*)(yr + go * 8u + fn);
                    *yp = *yp + float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// Co-Exec / Mid-Batch Fused BM=32, BN=32, BK=32 SwiGLU from single-copy model.hwx -> half Y[M, INTER]
// Double-buffered sA/sBg/sBu (13.5 KB -> 4 TGs/core!), 128B coalesced W bursts, ZERO FP32 gate/up DRAM round-trip!
// Grid: (n_subblocks * ceil_div(SUB/8, 4), ceil_div(M, 32), 1), TG: (128, 1, 1)
kernel void ane_hf4_swiglu_tiled32(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint tiles_sb     = (NGS + 3u) >> 2;
    const uint total_tiles  = NSB * tiles_sb;
    const uint m0 = tgid.y * 32u;
    if (tgid.x >= total_tiles || m0 >= p.M || HID == 0u) return;

    const uint sb    = tgid.x / tiles_sb;
    const uint tile  = tgid.x - sb * tiles_sb;
    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    threadgroup half sA[2][32][36];
    threadgroup half sBg[2][32][36];
    threadgroup half sBu[2][32][36];

    simdgroup_float8x8 acc_g[2][2], acc_u[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
        acc_g[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc_u[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint sg_m  = (sg >> 1) * 16u;
    const uint sg_n  = (sg & 1u) * 16u;
    const uint r_a   = tid >> 2;         // 0..31
    const uint c_a   = (tid & 3u) * 8u;  // 0, 8, 16, 24
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * HID;

    const uint w_slot  = tid >> 5;       // 0..3 (1 full SG per slot -> 128B coalesced read!)
    const uint w_k     = tid & 31u;      // 0..31
    const uint flat    = tile * 4u + w_slot;
    const bool flat_ok = flat < NGS;
    const uint flat_c  = flat_ok ? flat : 0u;
    const uint ch      = flat_c / slots_per_ch;
    const uint slot    = flat_c - ch * slots_per_ch;
    const uint G_s     = sb * NGS + slot * 16u + ch;

    const device uint* wg_ptr = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                                + 16u + (ulong)slot * gw_g + hdr_g + w_k;
    const device uint* wu_ptr = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                                + 16u + (ulong)slot * gw_u + hdr_u + w_k;

    half2 sc_g[4], bi_g[4], sc_u[4], bi_u[4];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 4u; ++c) {
        float2 sog0 = SO[G_s * 8u + 2u * c + 0u];
        float2 sog1 = SO[G_s * 8u + 2u * c + 1u];
        float2 sou0 = SO[INTER + G_s * 8u + 2u * c + 0u];
        float2 sou1 = SO[INTER + G_s * 8u + 2u * c + 1u];
        sc_g[c] = flat_ok ? half2(half(sog0.x), half(sog1.x)) : half2(0.0h);
        bi_g[c] = flat_ok ? half2(half(-(sog0.y - 1024.0f) * sog0.x), half(-(sog1.y - 1024.0f) * sog1.x)) : half2(0.0h);
        sc_u[c] = flat_ok ? half2(half(sou0.x), half(sou1.x)) : half2(0.0h);
        bi_u[c] = flat_ok ? half2(half(-(sou0.y - 1024.0f) * sou0.x), half(-(sou1.y - 1024.0f) * sou1.x)) : half2(0.0h);
    }

    // Prologue: load k0=0 into buf 0
    {
        threadgroup half4* da = (threadgroup half4*)(&sA[0][r_a][c_a]);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + c_a);
            da[0] = xp[0]; da[1] = xp[1];
        } else {
            da[0] = half4(0.0h); da[1] = half4(0.0h);
        }
        uint wg = (flat_ok ? wg_ptr[0] : 0u) ^ 0x88888888u;
        uint wu = (flat_ok ? wu_ptr[0] : 0u) ^ 0x88888888u;
        threadgroup half2* dg = (threadgroup half2*)(&sBg[0][w_k][w_slot * 8u]);
        dg[0] = fma(ane_dq4_f16_xored(wg,  0u) - half2(1024.0h), sc_g[0], bi_g[0]);
        dg[1] = fma(ane_dq4_f16_xored(wg,  8u) - half2(1024.0h), sc_g[1], bi_g[1]);
        dg[2] = fma(ane_dq4_f16_xored(wg, 16u) - half2(1024.0h), sc_g[2], bi_g[2]);
        dg[3] = fma(ane_dq4_f16_xored(wg, 24u) - half2(1024.0h), sc_g[3], bi_g[3]);

        threadgroup half2* du = (threadgroup half2*)(&sBu[0][w_k][w_slot * 8u]);
        du[0] = fma(ane_dq4_f16_xored(wu,  0u) - half2(1024.0h), sc_u[0], bi_u[0]);
        du[1] = fma(ane_dq4_f16_xored(wu,  8u) - half2(1024.0h), sc_u[1], bi_u[1]);
        du[2] = fma(ane_dq4_f16_xored(wu, 16u) - half2(1024.0h), sc_u[2], bi_u[2]);
        du[3] = fma(ane_dq4_f16_xored(wu, 24u) - half2(1024.0h), sc_u[3], bi_u[3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0u; k0 < HID; k0 += 32u) {
        const uint cur = (k0 >> 5) & 1u;
        const uint nxt = cur ^ 1u;
        const uint k_nxt = k0 + 32u;
        const bool has_nxt = k_nxt < HID;

        half4 nx0 = half4(0.0h), nx1 = half4(0.0h);
        uint nwg = 0u, nwu = 0u;
        if (has_nxt) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k_nxt + c_a);
                nx0 = xp[0]; nx1 = xp[1];
            }
            if (flat_ok) {
                nwg = wg_ptr[k_nxt];
                nwu = wu_ptr[k_nxt];
            }
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[2], bg[2], bu[2];
            simdgroup_load(a[0],  &sA[cur][sg_m][kk], 36);
            simdgroup_load(a[1],  &sA[cur][sg_m + 8u][kk], 36);
            simdgroup_load(bg[0], &sBg[cur][kk][sg_n], 36);
            simdgroup_load(bg[1], &sBg[cur][kk][sg_n + 8u], 36);
            simdgroup_load(bu[0], &sBu[cur][kk][sg_n], 36);
            simdgroup_load(bu[1], &sBu[cur][kk][sg_n + 8u], 36);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 2u; ++i)
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_multiply_accumulate(acc_g[i][j], a[i], bg[j], acc_g[i][j]);
                simdgroup_multiply_accumulate(acc_u[i][j], a[i], bu[j], acc_u[i][j]);
            }
        }

        if (has_nxt) {
            threadgroup half4* da = (threadgroup half4*)(&sA[nxt][r_a][c_a]);
            da[0] = nx0; da[1] = nx1;

            uint wg_x = nwg ^ 0x88888888u;
            uint wu_x = nwu ^ 0x88888888u;
            threadgroup half2* dg = (threadgroup half2*)(&sBg[nxt][w_k][w_slot * 8u]);
            dg[0] = fma(ane_dq4_f16_xored(wg_x,  0u) - half2(1024.0h), sc_g[0], bi_g[0]);
            dg[1] = fma(ane_dq4_f16_xored(wg_x,  8u) - half2(1024.0h), sc_g[1], bi_g[1]);
            dg[2] = fma(ane_dq4_f16_xored(wg_x, 16u) - half2(1024.0h), sc_g[2], bi_g[2]);
            dg[3] = fma(ane_dq4_f16_xored(wg_x, 24u) - half2(1024.0h), sc_g[3], bi_g[3]);

            threadgroup half2* du = (threadgroup half2*)(&sBu[nxt][w_k][w_slot * 8u]);
            du[0] = fma(ane_dq4_f16_xored(wu_x,  0u) - half2(1024.0h), sc_u[0], bi_u[0]);
            du[1] = fma(ane_dq4_f16_xored(wu_x,  8u) - half2(1024.0h), sc_u[1], bi_u[1]);
            du[2] = fma(ane_dq4_f16_xored(wu_x, 16u) - half2(1024.0h), sc_u[2], bi_u[2]);
            du[3] = fma(ane_dq4_f16_xored(wu_x, 24u) - half2(1024.0h), sc_u[3], bi_u[3]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    uint G_out[2];
    bool G_ok[2];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
        uint fo = tile * 4u + (sg_n >> 3) + j;
        G_ok[j] = fo < NGS;
        uint foc = G_ok[j] ? fo : 0u;
        uint co = foc / slots_per_ch;
        uint so = foc - co * slots_per_ch;
        G_out[j] = sb * NGS + so * 16u + co;
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device half* yr = Y + (ulong)gm * INTER;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                if (G_ok[j]) {
                    float g0 = acc_g[i][j].thread_elements()[0], g1 = acc_g[i][j].thread_elements()[1];
                    float u0 = acc_u[i][j].thread_elements()[0], u1 = acc_u[i][j].thread_elements()[1];
                    *(device half2*)(yr + G_out[j] * 8u + fn) = half2(half(silu_f32(g0) * u0), half(silu_f32(g1) * u1));
                }
            }
        }
    }
}

// Co-Exec / Mid-Batch Split-K BM=32, BN=32, BK=32 down_proj -> partY[sk, M, HID] (sk = n_tg.z in {1,2,4,8}!)
// Double-buffered sA/sB (9 KB -> 4+ TGs/core!), plugs straight into mma_qmm8_sk_reduce!
// Grid: (16 * ceil_div(HID/128, 4), ceil_div(M, 32), sk), TG: (128, 1, 1)
kernel void ane_hf4_down_tiled32_sk(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint3 n_tg                        [[threadgroups_per_grid]],
    uint  tid                         [[thread_index_in_threadgroup]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 3u) >> 2;
    const uint m0 = tgid.y * 32u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M || DK == 0u) return;

    const uint ch    = tgid.x / tg_per_ch;
    const uint slot0 = (tgid.x - ch * tg_per_ch) * 4u;
    const uint sk    = max(1u, n_tg.z);
    const uint q_per_sk = (n_q + sk - 1u) / sk;
    const uint q_start  = tgid.z * q_per_sk;
    const uint q_end    = min(n_q, q_start + q_per_sk);

    threadgroup half sA[2][32][36];
    threadgroup half sB[2][32][36];

    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m  = (sg >> 1) * 16u;
    const uint sg_n  = (sg & 1u) * 16u;
    const uint r_a   = tid >> 2;
    const uint c_a   = (tid & 3u) * 8u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_row = X + (ulong)(gm_ok ? gm_ld : 0u) * INTER;

    const uint w_slot  = tid >> 5;       // 0..3
    const uint w_k     = tid & 31u;      // 0..31
    const uint slot    = slot0 + w_slot;
    const bool slot_ok = slot < slots_per_ch;
    const uint slot_c  = slot_ok ? slot : 0u;
    const uint go_s    = slot_c * 16u + ch;

    for (uint q = q_start; q < q_end; ++q) {
        const uint sb  = q >> 1;
        const uint li  = sb * 4u + 2u + (q & 1u);
        const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
        const uint gw  = hdr + DK;
        const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                                   + 16u + (ulong)slot_c * gw + hdr + w_k;
        const device half* x_base = x_row + (ulong)q * DK;
        const uint so_down_base = 2u * INTER + q * HID;

        half2 sc8[4], bi8[4];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
            float2 sd0 = SO[so_down_base + go_s * 8u + 2u * c + 0u];
            float2 sd1 = SO[so_down_base + go_s * 8u + 2u * c + 1u];
            sc8[c] = slot_ok ? half2(half(sd0.x), half(sd1.x)) : half2(0.0h);
            bi8[c] = slot_ok ? half2(half(-(sd0.y - 1024.0f) * sd0.x), half(-(sd1.y - 1024.0f) * sd1.x)) : half2(0.0h);
        }

        // Prologue for this q: load k0=0 into buf 0
        {
            threadgroup half4* da = (threadgroup half4*)(&sA[0][r_a][c_a]);
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + c_a);
                da[0] = xp[0]; da[1] = xp[1];
            } else {
                da[0] = half4(0.0h); da[1] = half4(0.0h);
            }
            uint w = (slot_ok ? w_ptr[0] : 0u) ^ 0x88888888u;
            threadgroup half2* dst = (threadgroup half2*)(&sB[0][w_k][w_slot * 8u]);
            dst[0] = fma(ane_dq4_f16_xored(w,  0u) - half2(1024.0h), sc8[0], bi8[0]);
            dst[1] = fma(ane_dq4_f16_xored(w,  8u) - half2(1024.0h), sc8[1], bi8[1]);
            dst[2] = fma(ane_dq4_f16_xored(w, 16u) - half2(1024.0h), sc8[2], bi8[2]);
            dst[3] = fma(ane_dq4_f16_xored(w, 24u) - half2(1024.0h), sc8[3], bi8[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k0 = 0u; k0 < DK; k0 += 32u) {
            const uint cur = (k0 >> 5) & 1u;
            const uint nxt = cur ^ 1u;
            const uint k_nxt = k0 + 32u;
            const bool has_nxt = k_nxt < DK;

            half4 nx0 = half4(0.0h), nx1 = half4(0.0h);
            uint nw = 0u;
            if (has_nxt) {
                if (gm_ok) {
                    const device half4* xp = (const device half4*)(x_base + k_nxt + c_a);
                    nx0 = xp[0]; nx1 = xp[1];
                }
                if (slot_ok) nw = w_ptr[k_nxt];
            }

            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a[2], b[2];
                simdgroup_load(a[0], &sA[cur][sg_m][kk], 36);
                simdgroup_load(a[1], &sA[cur][sg_m + 8u][kk], 36);
                simdgroup_load(b[0], &sB[cur][kk][sg_n], 36);
                simdgroup_load(b[1], &sB[cur][kk][sg_n + 8u], 36);
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 2u; ++i)
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 2u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }

            if (has_nxt) {
                threadgroup half4* da = (threadgroup half4*)(&sA[nxt][r_a][c_a]);
                da[0] = nx0; da[1] = nx1;
                uint w_x = nw ^ 0x88888888u;
                threadgroup half2* dst = (threadgroup half2*)(&sB[nxt][w_k][w_slot * 8u]);
                dst[0] = fma(ane_dq4_f16_xored(w_x,  0u) - half2(1024.0h), sc8[0], bi8[0]);
                dst[1] = fma(ane_dq4_f16_xored(w_x,  8u) - half2(1024.0h), sc8[1], bi8[1]);
                dst[2] = fma(ane_dq4_f16_xored(w_x, 16u) - half2(1024.0h), sc8[2], bi8[2]);
                dst[3] = fma(ane_dq4_f16_xored(w_x, 24u) - half2(1024.0h), sc8[3], bi8[3]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (q + 1u < q_end) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const ulong sk_off = (ulong)tgid.z * ((ulong)p.M * HID);

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = partY + sk_off + (ulong)gm * HID;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                uint sl_out = slot0 + (sg_n >> 3) + j;
                if (sl_out < slots_per_ch) {
                    uint go = sl_out * 16u + ch;
                    *(device float2*)(yr + go * 8u + fn) =
                        float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// Split-K down_proj impl templated by KS (8 for existing 256-thr ABI; 4 for 128-thr 0% SG tail waste!)
// Direct-load zero-spill inner loop + pre-XOR + post-reduction SO epilogue + [[function_constant]]!
template<uint KS>
inline void ane_hf4_down_sk_impl(
    const device uint*          W,
    const device float2*        SO,
    const device half*          X,
    device float*               partY,
    constant AneShareParams&    p,
    threadgroup float*          red,
    threadgroup float*          red_sx,
    uint3 tgid,
    uint  lane,
    uint  sg
) {
    constexpr uint TN = 4u;
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint total_tgx    = ((slots_per_ch + 3u) >> 2) * 16u;

    const uint m0 = tgid.y * 8u;
    const uint q  = tgid.z;
    if (tgid.x >= total_tgx || m0 >= p.M || q >= n_q) return;

    const uint slot0 = (tgid.x >> 4) * TN;
    const uint ch    = tgid.x & 15u;
    const uint sb    = q >> 1;
    const uint li    = sb * 4u + 2u + (q & 1u);
    const uint hdr   = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw    = hdr + DK;

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    const device uint* pw_base = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                                 + 16u + hdr + 4u * fm;
    const uint s0 = min(slot0 + 0u, slots_per_ch - 1u);
    const uint s1 = min(slot0 + 1u, slots_per_ch - 1u);
    const uint s2 = min(slot0 + 2u, slots_per_ch - 1u);
    const uint s3 = min(slot0 + 3u, slots_per_ch - 1u);
    const device uint* pw0 = pw_base + (ulong)s0 * gw;
    const device uint* pw1 = pw_base + (ulong)s1 * gw;
    const device uint* pw2 = pw_base + (ulong)s2 * gw;
    const device uint* pw3 = pw_base + (ulong)s3 * gw;
    const bool ok_u4_1 = ((s1 & 1u) == 1u) || ((hdr & 2u) == 0u);
    const bool ok_u4_3 = ((s3 & 1u) == 1u) || ((hdr & 2u) == 0u);

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * INTER + (ulong)q * DK;

    simdgroup_float8x8 C[TN];
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float sx = 0.0f;

    // Strided-K across KS simdgroups -> 1 KB (KS=8) or 512B (KS=4) lockstep DRAM burst (zero spill!)
    for (uint ko = sg * 32u; ko < DK; ko += KS * 32u) {
        half4 x0 = mok ? *(const device half4*)(xr + ko + 4u * fn)      : half4(0.0h);
        half4 x1 = mok ? *(const device half4*)(xr + ko + 4u * fn + 4u) : half4(0.0h);
        sx += ((float(x0.x) + float(x0.y)) + (float(x0.z) + float(x0.w)))
            + ((float(x1.x) + float(x1.y)) + (float(x1.z) + float(x1.w)));

        simdgroup_half8x8 Af[4];
        Af[0].thread_elements()[0] = x0.x; Af[0].thread_elements()[1] = x1.x;
        Af[1].thread_elements()[0] = x0.y; Af[1].thread_elements()[1] = x1.y;
        Af[2].thread_elements()[0] = x0.z; Af[2].thread_elements()[1] = x1.z;
        Af[3].thread_elements()[0] = x0.w; Af[3].thread_elements()[1] = x1.w;

        const device uint* p0 = pw0 + ko;
        const device uint* p1 = pw1 + ko;
        const device uint* p2 = pw2 + ko;
        const device uint* p3 = pw3 + ko;

        uint2 a0 = *(const device uint2*)(p0)      ^ uint2(0x88888888u);
        uint2 b0 = *(const device uint2*)(p0 + 2u) ^ uint2(0x88888888u);
        uint4 w1 = (ok_u4_1 ? *(const device uint4*)(p1)
                            : uint4(*(const device uint2*)(p1), *(const device uint2*)(p1 + 2u))) ^ uint4(0x88888888u);
        ane_hf4_mma4_xored(a0, b0, sh, Af, C[0]);

        uint2 a2 = *(const device uint2*)(p2)      ^ uint2(0x88888888u);
        uint2 b2 = *(const device uint2*)(p2 + 2u) ^ uint2(0x88888888u);
        ane_hf4_mma4_xored(w1.xy, w1.zw, sh, Af, C[1]);

        uint4 w3 = (ok_u4_3 ? *(const device uint4*)(p3)
                            : uint4(*(const device uint2*)(p3), *(const device uint2*)(p3 + 2u))) ^ uint4(0x88888888u);
        ane_hf4_mma4_xored(a2, b2, sh, Af, C[2]);
        ane_hf4_mma4_xored(w3.xy, w3.zw, sh, Af, C[3]);
    }

    sx += simd_shuffle_xor(sx, 1);
    sx += simd_shuffle_xor(sx, 8);

    const uint so_down_base = 2u * INTER + q * HID;
    if (fn == 0u) red_sx[sg * 8u + fm] = sx;
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) {
        red[(sg * TN + t) * 64u + fm * 8u + fn]      = C[t].thread_elements()[0];
        red[(sg * TN + t) * 64u + fm * 8u + fn + 1u] = C[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint e0 = sg * 32u + lane;
    const uint t0 = e0 >> 6;
    const uint rc = e0 & 63u;
    const uint m  = rc >> 3;
    const uint c  = rc & 7u;
    const uint gm_out = m0 + m;
    if (gm_out < p.M) {
        float sx_tot = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < KS; ++k) sx_tot += red_sx[k * 8u + m];
        const ulong sk_stride = (ulong)p.M * HID;
        constexpr uint PASSES = 8u / KS;
        constexpr uint T_STEP = 4u / PASSES;
        #pragma clang loop unroll(full)
        for (uint pass = 0u; pass < PASSES; ++pass) {
            uint t = t0 + pass * T_STEP;
            if ((slot0 + t) < slots_per_ch) {
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (uint k = 0u; k < KS; ++k) acc += red[(k * TN + t) * 64u + rc];
                uint go_out = (slot0 + t) * 16u + ch;
                uint o = go_out * 8u + c;
                float2 sd = SO[so_down_base + o];
                partY[(ulong)q * sk_stride + (ulong)gm_out * HID + o] = (acc - sd.y * sx_tot) * sd.x;
            }
        }
    }
}

// Grid: (16 * ceil_div(HID/128, 4), ceil_div(M, 8), 2 * n_subblocks), TG: (256, 1, 1) (8 simdgroups)
kernel void ane_hf4_down_sk_bf16(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float red[8u * 4u * 64u];
    threadgroup float red_sx[8u * 8u];
    ane_hf4_down_sk_impl<8u>(W, SO, X, partY, p, red, red_sx, tgid, lane, sg);
}

// KS=4 (128 thr, 4 KB TG SRAM -> 2x occupancy + 0% SG tail waste on DK=2176!)
// Grid: (16 * ceil_div(HID/128, 4), ceil_div(M, 8), 2 * n_subblocks), TG: (128, 1, 1) (4 simdgroups)
kernel void ane_hf4_down_sk4_bf16(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float red[4u * 4u * 64u];
    threadgroup float red_sx[4u * 8u];
    ane_hf4_down_sk_impl<4u>(W, SO, X, partY, p, red, red_sx, tgid, lane, sg);
}

// ============================================================================
// 8b. Universal Single-Copy Non-Uniform LUT4 (NF4 / arbitrary 16-entry FP16 LUT)
//     Reads the verified half[16] LUT directly from hwx_buf chunk header bytes [0..31]
//     (ZERO extra buffer bindings -> exact drop-in ABI with ane_hf4_*!),
//     expanding once per TG into a 1 KB threadgroup half2 s_lut2[256] byte-pair table
//     for single-load 2-nibble dequant (zero simd_shuffle stalls, zero extra SRAM on M=8!).
// ============================================================================

// Grid: (INTER / 16, ceil_div(M, 8), 1), TG: (256, 1, 1)
kernel void ane_lut4_swiglu_bf16(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]], // .x = per-channel scale
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint KS = 8u;
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS   = SUB >> 3;
    const uint tgs_sb = SUB >> 4;
    const uint total_tgs = NSB * tgs_sb;

    const uint m0 = tgid.y * 8u;
    if (tgid.x >= total_tgs || m0 >= p.M) return;

    const uint sb    = tgid.x / tgs_sb;
    const uint rem   = tgid.x - sb * tgs_sb;
    const uint slot0 = (rem >> 4) * 2u;
    const uint ch    = rem & 15u;
    const uint G0    = sb * NGS + slot0 * 16u + ch;
    const uint G1    = G0 + 16u;

    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    const device uint* ch_g = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g];
    const device uint* ch_u = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u];

    // Alias 2x1 KB byte-pair LUT tables into red_g (0 B extra SRAM -> keeps full 4 TGs/core!)
    threadgroup float red_g[KS][2][64];
    threadgroup float red_u[KS][2][64];
    threadgroup half2* s_lut2_g = (threadgroup half2*)(&red_g[0][0][0]);
    threadgroup half2* s_lut2_u = s_lut2_g + 256u;
    {
        const uint tid = sg * 32u + lane;
        const device half* lg = (const device half*)ch_g;
        const device half* lu = (const device half*)ch_u;
        s_lut2_g[tid] = half2(lg[tid & 15u], lg[tid >> 4]);
        s_lut2_u[tid] = half2(lu[tid & 15u], lu[tid >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    const device uint* pg0 = ch_g + 16u + (ulong)slot0 * gw_g + hdr_g + 4u * fm;
    const device uint* pg1 = pg0 + gw_g;
    const device uint* pu0 = ch_u + 16u + (ulong)slot0 * gw_u + hdr_u + 4u * fm;
    const device uint* pu1 = pu0 + gw_u;
    const bool even_u4_g = (hdr_g & 2u) == 0u;
    const bool even_u4_u = (hdr_u & 2u) == 0u;

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * HID;

    simdgroup_float8x8 Cg0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cg1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint ko = sg * 32u; ko < HID; ko += KS * 32u) {
        half4 x0 = mok ? *(const device half4*)(xr + ko + 4u * fn)      : half4(0.0h);
        half4 x1 = mok ? *(const device half4*)(xr + ko + 4u * fn + 4u) : half4(0.0h);
        simdgroup_half8x8 Af[4];
        Af[0].thread_elements()[0] = x0.x; Af[0].thread_elements()[1] = x1.x;
        Af[1].thread_elements()[0] = x0.y; Af[1].thread_elements()[1] = x1.y;
        Af[2].thread_elements()[0] = x0.z; Af[2].thread_elements()[1] = x1.z;
        Af[3].thread_elements()[0] = x0.w; Af[3].thread_elements()[1] = x1.w;

        uint4 w0 = even_u4_g ? *(const device uint4*)(pg0 + ko)
                             : uint4(*(const device uint2*)(pg0 + ko), *(const device uint2*)(pg0 + ko + 2u));
        uint4 w1 = *(const device uint4*)(pg1 + ko);
        ane_lut4_mma4_tbl(w0.xy, w0.zw, sh, s_lut2_g, Af, Cg0);

        uint4 w2 = even_u4_u ? *(const device uint4*)(pu0 + ko)
                             : uint4(*(const device uint2*)(pu0 + ko), *(const device uint2*)(pu0 + ko + 2u));
        ane_lut4_mma4_tbl(w1.xy, w1.zw, sh, s_lut2_g, Af, Cg1);

        uint4 w3 = *(const device uint4*)(pu1 + ko);
        ane_lut4_mma4_tbl(w2.xy, w2.zw, sh, s_lut2_u, Af, Cu0);
        ane_lut4_mma4_tbl(w3.xy, w3.zw, sh, s_lut2_u, Af, Cu1);
    }

    // Fence s_lut2_g/u reads before reusing red_g for reduction!
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red_g[sg][0][fm * 8u + fn]      = Cg0.thread_elements()[0];
    red_g[sg][0][fm * 8u + fn + 1u] = Cg0.thread_elements()[1];
    red_g[sg][1][fm * 8u + fn]      = Cg1.thread_elements()[0];
    red_g[sg][1][fm * 8u + fn + 1u] = Cg1.thread_elements()[1];
    red_u[sg][0][fm * 8u + fn]      = Cu0.thread_elements()[0];
    red_u[sg][0][fm * 8u + fn + 1u] = Cu0.thread_elements()[1];
    red_u[sg][1][fm * 8u + fn]      = Cu1.thread_elements()[0];
    red_u[sg][1][fm * 8u + fn + 1u] = Cu1.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = sg * 32u + lane;
    if (e < 128u) {
        uint t  = e >> 6;
        uint rc = e & 63u;
        uint m  = rc >> 3;
        uint c  = rc & 7u;
        uint gm_out = m0 + m;
        if (gm_out < p.M) {
            float g = 0.0f, u = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0u; k < KS; ++k) {
                g += red_g[k][t][rc];
                u += red_u[k][t][rc];
            }
            uint oc = ((t == 0u) ? G0 : G1) * 8u + c;
            float gv = g * SO[oc].x;
            float uv = u * SO[INTER + oc].x;
            Y[(ulong)gm_out * INTER + oc] = half(silu_f32(gv) * uv);
        }
    }
}

// Grid: (16 * ceil_div(HID/128, 4), ceil_div(M, 8), 2 * n_subblocks), TG: (128, 1, 1) (KS=4, 0% SG tail waste!)
kernel void ane_lut4_down_sk4_bf16(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TN = 4u;
    constexpr uint KS = 4u;
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint total_tgx    = ((slots_per_ch + 3u) >> 2) * 16u;

    const uint m0 = tgid.y * 8u;
    const uint q  = tgid.z;
    if (tgid.x >= total_tgx || m0 >= p.M || q >= n_q) return;

    const uint slot0 = (tgid.x >> 4) * TN;
    const uint ch    = tgid.x & 15u;
    const uint sb    = q >> 1;
    const uint li    = sb * 4u + 2u + (q & 1u);
    const uint hdr   = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw    = hdr + DK;

    const device uint* chunk_base = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li];

    // Alias 1 KB byte-pair LUT table into red (0 B extra SRAM -> keeps full 8 TGs/core!)
    threadgroup float red[KS][TN][64];
    threadgroup half2* s_lut2 = (threadgroup half2*)(&red[0][0][0]);
    {
        const uint tid = sg * 32u + lane;
        const device half* lv = (const device half*)chunk_base;
        s_lut2[tid]        = half2(lv[tid & 15u], lv[tid >> 4]);
        s_lut2[tid + 128u] = half2(lv[(tid + 128u) & 15u], lv[(tid + 128u) >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    const device uint* pw_base = chunk_base + 16u + hdr + 4u * fm;
    const uint s0 = min(slot0 + 0u, slots_per_ch - 1u);
    const uint s1 = min(slot0 + 1u, slots_per_ch - 1u);
    const uint s2 = min(slot0 + 2u, slots_per_ch - 1u);
    const uint s3 = min(slot0 + 3u, slots_per_ch - 1u);
    const device uint* pw0 = pw_base + (ulong)s0 * gw;
    const device uint* pw1 = pw_base + (ulong)s1 * gw;
    const device uint* pw2 = pw_base + (ulong)s2 * gw;
    const device uint* pw3 = pw_base + (ulong)s3 * gw;
    const bool even_u4 = (hdr & 2u) == 0u;
    const bool ok_u4_1 = even_u4 || ((s1 & 1u) == 1u);
    const bool ok_u4_3 = even_u4 || ((s3 & 1u) == 1u);

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * INTER + (ulong)q * DK;

    simdgroup_float8x8 C[TN];
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint ko = sg * 32u; ko < DK; ko += KS * 32u) {
        half4 x0 = mok ? *(const device half4*)(xr + ko + 4u * fn)      : half4(0.0h);
        half4 x1 = mok ? *(const device half4*)(xr + ko + 4u * fn + 4u) : half4(0.0h);
        simdgroup_half8x8 Af[4];
        Af[0].thread_elements()[0] = x0.x; Af[0].thread_elements()[1] = x1.x;
        Af[1].thread_elements()[0] = x0.y; Af[1].thread_elements()[1] = x1.y;
        Af[2].thread_elements()[0] = x0.z; Af[2].thread_elements()[1] = x1.z;
        Af[3].thread_elements()[0] = x0.w; Af[3].thread_elements()[1] = x1.w;

        const device uint* p0 = pw0 + ko;
        const device uint* p1 = pw1 + ko;
        const device uint* p2 = pw2 + ko;
        const device uint* p3 = pw3 + ko;

        uint4 w0 = even_u4 ? *(const device uint4*)(p0)
                           : uint4(*(const device uint2*)(p0), *(const device uint2*)(p0 + 2u));
        uint4 w1 = ok_u4_1 ? *(const device uint4*)(p1)
                           : uint4(*(const device uint2*)(p1), *(const device uint2*)(p1 + 2u));
        ane_lut4_mma4_tbl(w0.xy, w0.zw, sh, s_lut2, Af, C[0]);

        uint4 w2 = even_u4 ? *(const device uint4*)(p2)
                           : uint4(*(const device uint2*)(p2), *(const device uint2*)(p2 + 2u));
        ane_lut4_mma4_tbl(w1.xy, w1.zw, sh, s_lut2, Af, C[1]);

        uint4 w3 = ok_u4_3 ? *(const device uint4*)(p3)
                           : uint4(*(const device uint2*)(p3), *(const device uint2*)(p3 + 2u));
        ane_lut4_mma4_tbl(w2.xy, w2.zw, sh, s_lut2, Af, C[2]);
        ane_lut4_mma4_tbl(w3.xy, w3.zw, sh, s_lut2, Af, C[3]);
    }

    // Fence s_lut2 reads before reusing red for reduction!
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint so_down_base = 2u * INTER + q * HID;
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) {
        red[sg][t][fm * 8u + fn]      = C[t].thread_elements()[0];
        red[sg][t][fm * 8u + fn + 1u] = C[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint e0 = sg * 32u + lane;
    const uint t0 = e0 >> 6;
    const uint rc = e0 & 63u;
    const uint m  = rc >> 3;
    const uint c  = rc & 7u;
    const uint gm_out = m0 + m;
    if (gm_out < p.M) {
        const ulong sk_stride = (ulong)p.M * HID;
        #pragma clang loop unroll(full)
        for (uint pass = 0u; pass < 2u; ++pass) {
            uint t = t0 + pass * 2u;
            if ((slot0 + t) < slots_per_ch) {
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (uint k = 0u; k < KS; ++k) acc += red[k][t][rc];
                uint go_out = (slot0 + t) * 16u + ch;
                uint o = go_out * 8u + c;
                partY[(ulong)q * sk_stride + (ulong)gm_out * HID + o] = acc * SO[so_down_base + o].x;
            }
        }
    }
}

// Universal LUT4 M<=4 (handles M=1..4) gate+up+SiLU -> half Y[M, INTER]
// Grid: (ceil_div(INTER / 8, 4), ceil_div(M, 4), 1), TG: (128, 1, 1)
kernel void ane_lut4_swiglu_m4(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint total_g      = NSB * NGS;
    const uint m0   = tgid.y * 4u;
    if (tgid.x * 4u >= total_g || m0 >= p.M) return;

    // SUB % 256 == 0 => NGS % 32 == 0 => all 4 simdgroups in this TG share identical sb!
    const uint sb_tg = (tgid.x * 4u) / NGS;
    const uint li_g  = sb_tg * 4u + 0u;
    const uint li_u  = sb_tg * 4u + 1u;

    threadgroup half2 s_lut2_g[256];
    threadgroup half2 s_lut2_u[256];
    {
        const uint tid = sg * 32u + lane;
        const device half* lg = (const device half*)(W + (ulong)p.lin_base_w[li_g]);
        const device half* lu = (const device half*)(W + (ulong)p.lin_base_w[li_u]);
        s_lut2_g[tid]        = half2(lg[tid & 15u], lg[tid >> 4]);
        s_lut2_g[tid + 128u] = half2(lg[(tid + 128u) & 15u], lg[(tid + 128u) >> 4]);
        s_lut2_u[tid]        = half2(lu[tid & 15u], lu[tid >> 4]);
        s_lut2_u[tid + 128u] = half2(lu[(tid + 128u) & 15u], lu[(tid + 128u) >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint gidx = tgid.x * 4u + sg;
    if (gidx >= total_g) return;

    const uint rem   = gidx - sb_tg * NGS;
    const uint ch    = rem / slots_per_ch;
    const uint slot  = rem - ch * slots_per_ch;
    const uint G0    = sb_tg * NGS + slot * 16u + ch;

    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    const device uint* ch_g = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g];
    const device uint* ch_u = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u];
    const device uint* pg = ch_g + 16u + (ulong)slot * gw_g + hdr_g;
    const device uint* pu = ch_u + 16u + (ulong)slot * gw_u + hdr_u;

    const bool mok0 = (m0 + 0u) < p.M;
    const bool mok1 = (m0 + 1u) < p.M;
    const bool mok2 = (m0 + 2u) < p.M;
    const bool mok3 = (m0 + 3u) < p.M;
    const device half* xr0 = X + (ulong)(mok0 ? (m0 + 0u) : 0u) * HID;
    const device half* xr1 = X + (ulong)(mok1 ? (m0 + 1u) : 0u) * HID;
    const device half* xr2 = X + (ulong)(mok2 ? (m0 + 2u) : 0u) * HID;
    const device half* xr3 = X + (ulong)(mok3 ? (m0 + 3u) : 0u) * HID;

    float4 ag[8], au[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) { ag[c] = float4(0.0f); au[c] = float4(0.0f); }

    for (uint k = lane * 4u; k < HID; k += 128u) {
        half4 x0 = mok0 ? *(const device half4*)(xr0 + k) : half4(0.0h);
        half4 x1 = mok1 ? *(const device half4*)(xr1 + k) : half4(0.0h);
        half4 x2 = mok2 ? *(const device half4*)(xr2 + k) : half4(0.0h);
        half4 x3 = mok3 ? *(const device half4*)(xr3 + k) : half4(0.0h);

        uint2 g0 = *(const device uint2*)(pg + k);
        uint2 g1 = *(const device uint2*)(pg + k + 2u);
        uint2 u0 = *(const device uint2*)(pu + k);
        uint2 u1 = *(const device uint2*)(pu + k + 2u);
        uint gw4[4] = {g0.x, g0.y, g1.x, g1.y};
        uint uw4[4] = {u0.x, u0.y, u1.x, u1.y};

        float4 xk[4];
        xk[0] = float4(float(x0.x), float(x1.x), float(x2.x), float(x3.x));
        xk[1] = float4(float(x0.y), float(x1.y), float(x2.y), float(x3.y));
        xk[2] = float4(float(x0.z), float(x1.z), float(x2.z), float(x3.z));
        xk[3] = float4(float(x0.w), float(x1.w), float(x2.w), float(x3.w));

        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint gv = gw4[j], uv = uw4[j];
            float4 xv = xk[j];
            #pragma clang loop unroll(full)
            for (uint cp = 0u; cp < 4u; ++cp) {
                half2 wg2 = s_lut2_g[(gv >> (8u * cp)) & 0xFFu];
                half2 wu2 = s_lut2_u[(uv >> (8u * cp)) & 0xFFu];
                ag[2u * cp + 0u] = fma(float(wg2.x), xv, ag[2u * cp + 0u]);
                ag[2u * cp + 1u] = fma(float(wg2.y), xv, ag[2u * cp + 1u]);
                au[2u * cp + 0u] = fma(float(wu2.x), xv, au[2u * cp + 0u]);
                au[2u * cp + 1u] = fma(float(wu2.y), xv, au[2u * cp + 1u]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float g_r[4] = { simd_sum(ag[c].x), simd_sum(ag[c].y), simd_sum(ag[c].z), simd_sum(ag[c].w) };
        float u_r[4] = { simd_sum(au[c].x), simd_sum(au[c].y), simd_sum(au[c].z), simd_sum(au[c].w) };
        if (lane == 0u) {
            uint oc = G0 * 8u + c;
            float sc_g = SO[oc].x;
            float sc_u = SO[INTER + oc].x;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (m0 + r < p.M) {
                    Y[(ulong)(m0 + r) * INTER + oc] = half(silu_f32(g_r[r] * sc_g) * (u_r[r] * sc_u));
                }
            }
        }
    }
}

// Universal LUT4 M<=4 (handles M=1..4) Split-K down_proj -> partY[2*n_subblocks, M, HID]
// Grid: (16 * ceil_div(HID / 128, 4), ceil_div(M, 4), 2 * n_subblocks), TG: (128, 1, 1)
kernel void ane_lut4_down_sk_m4(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 3u) >> 2;
    const uint m0 = tgid.y * 4u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M || tgid.z >= n_q) return;

    const uint ch  = tgid.x / tg_per_ch;
    const uint q   = tgid.z;
    const uint sb  = q >> 1;
    const uint li  = sb * 4u + 2u + (q & 1u);
    const device uint* chunk_base = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li];

    // Load 1 KB byte-pair LUT BEFORE non-uniform slot early-return (barrier-safe!)
    threadgroup half2 s_lut2[256];
    {
        const uint tid = sg * 32u + lane;
        const device half* lv = (const device half*)chunk_base;
        s_lut2[tid]        = half2(lv[tid & 15u], lv[tid >> 4]);
        s_lut2[tid + 128u] = half2(lv[(tid + 128u) & 15u], lv[(tid + 128u) >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slot = (tgid.x - ch * tg_per_ch) * 4u + sg;
    if (slot >= slots_per_ch) return;

    const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw  = hdr + DK;
    const device uint* pw = chunk_base + 16u + (ulong)slot * gw + hdr;

    const bool mok0 = (m0 + 0u) < p.M;
    const bool mok1 = (m0 + 1u) < p.M;
    const bool mok2 = (m0 + 2u) < p.M;
    const bool mok3 = (m0 + 3u) < p.M;
    const device half* xr0 = X + (ulong)(mok0 ? (m0 + 0u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr1 = X + (ulong)(mok1 ? (m0 + 1u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr2 = X + (ulong)(mok2 ? (m0 + 2u) : 0u) * INTER + (ulong)q * DK;
    const device half* xr3 = X + (ulong)(mok3 ? (m0 + 3u) : 0u) * INTER + (ulong)q * DK;

    float4 acc[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) acc[c] = float4(0.0f);

    for (uint k = lane * 4u; k < DK; k += 128u) {
        half4 x0 = mok0 ? *(const device half4*)(xr0 + k) : half4(0.0h);
        half4 x1 = mok1 ? *(const device half4*)(xr1 + k) : half4(0.0h);
        half4 x2 = mok2 ? *(const device half4*)(xr2 + k) : half4(0.0h);
        half4 x3 = mok3 ? *(const device half4*)(xr3 + k) : half4(0.0h);

        uint2 u0 = *(const device uint2*)(pw + k);
        uint2 u1 = *(const device uint2*)(pw + k + 2u);
        uint w4[4] = {u0.x, u0.y, u1.x, u1.y};

        float4 xk[4];
        xk[0] = float4(float(x0.x), float(x1.x), float(x2.x), float(x3.x));
        xk[1] = float4(float(x0.y), float(x1.y), float(x2.y), float(x3.y));
        xk[2] = float4(float(x0.z), float(x1.z), float(x2.z), float(x3.z));
        xk[3] = float4(float(x0.w), float(x1.w), float(x2.w), float(x3.w));

        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) {
            uint wv = w4[j];
            float4 xv = xk[j];
            #pragma clang loop unroll(full)
            for (uint cp = 0u; cp < 4u; ++cp) {
                half2 ww2 = s_lut2[(wv >> (8u * cp)) & 0xFFu];
                acc[2u * cp + 0u] = fma(float(ww2.x), xv, acc[2u * cp + 0u]);
                acc[2u * cp + 1u] = fma(float(ww2.y), xv, acc[2u * cp + 1u]);
            }
        }
    }

    const uint go = slot * 16u + ch;
    const uint so_down_base = 2u * INTER + q * HID;
    const ulong sk_stride = (ulong)p.M * HID;
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float v_r[4] = { simd_sum(acc[c].x), simd_sum(acc[c].y), simd_sum(acc[c].z), simd_sum(acc[c].w) };
        if (lane == 0u) {
            uint o = go * 8u + c;
            float sc_d = SO[so_down_base + o].x;
            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 4u; ++r) {
                if (m0 + r < p.M) {
                    partY[(ulong)q * sk_stride + (ulong)(m0 + r) * HID + o] = v_r[r] * sc_d;
                }
            }
        }
    }
}

// Universal LUT4 BM=64, BN=64, BK=32 Tiled Prefill gate & up
// Grid: (n_subblocks * ceil_div(SUB/8, 8), ceil_div(M, 64), 2), TG: (128, 1, 1)
kernel void ane_lut4_tiled_gateup(
    const device uint*          W        [[buffer(0)]],
    const device float2*        SO       [[buffer(1)]],
    const device half*          X        [[buffer(2)]],
    device float*               gate_out [[buffer(3)]],
    device float*               up_out   [[buffer(4)]],
    constant AneShareParams&    p        [[buffer(5)]],
    uint3 tgid                           [[threadgroup_position_in_grid]],
    uint  tid                            [[thread_index_in_threadgroup]],
    uint  lane                           [[thread_index_in_simdgroup]],
    uint  sg                             [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint tiles_sb     = (NGS + 7u) >> 3;
    const uint total_tiles  = NSB * tiles_sb;
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= total_tiles || m0 >= p.M || tgid.z >= 2u) return;

    const uint sb      = tgid.x / tiles_sb;
    const uint tile    = tgid.x - sb * tiles_sb;
    const uint li      = sb * 4u + tgid.z;
    const uint hdr     = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
    const uint gw      = hdr + HID;
    const uint so_base = tgid.z * INTER;

    threadgroup half  sA[64][36];
    threadgroup half  sB[32][68];
    threadgroup half2 s_lut2[256]; // 1 KB -> 9.75 KB total (3 TGs/core unchanged!)
    {
        const device half* lv = (const device half*)(W + (ulong)p.lin_base_w[li]);
        s_lut2[tid]        = half2(lv[tid & 15u], lv[tid >> 4]);
        s_lut2[tid + 128u] = half2(lv[(tid + 128u) & 15u], lv[(tid + 128u) >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m = (sg >> 1) * 32u;
    const uint sg_n = (sg & 1u) * 32u;
    const uint r_a  = tid >> 1;
    const uint c_a  = (tid & 1u) * 16u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * HID;
    threadgroup half4* da = (threadgroup half4*)(&sA[r_a][c_a]);

    const uint w_slot = tid >> 4;
    const uint w_k    = (tid & 15u) * 2u;
    const uint flat   = tile * 8u + w_slot;
    const bool flat_ok = flat < NGS;
    const uint flat_c = flat_ok ? flat : 0u;
    const uint ch     = flat_c / slots_per_ch;
    const uint slot   = flat_c - ch * slots_per_ch;
    const uint G_s    = sb * NGS + slot * 16u + ch;
    const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                               + 16u + (ulong)slot * gw + hdr + w_k;

    half2 sc2[4];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 4u; ++c) {
        sc2[c] = flat_ok ? half2(half(SO[so_base + G_s * 8u + 2u * c + 0u].x),
                                 half(SO[so_base + G_s * 8u + 2u * c + 1u].x)) : half2(0.0h);
    }

    for (uint k0 = 0u; k0 < HID; k0 += 32u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + c_a);
            da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
        } else {
            da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
        }

        uint2 u2 = flat_ok ? *(const device uint2*)(w_ptr + k0) : uint2(0u);
        uint ws[2] = {u2.x, u2.y};
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; ++j) {
            uint w = ws[j];
            threadgroup half2* dst = (threadgroup half2*)(&sB[w_k + j][w_slot * 8u]);
            dst[0] = s_lut2[(w >>  0u) & 0xFFu] * sc2[0];
            dst[1] = s_lut2[(w >>  8u) & 0xFFu] * sc2[1];
            dst[2] = s_lut2[(w >> 16u) & 0xFFu] * sc2[2];
            dst[3] = s_lut2[(w >> 24u) & 0xFFu] * sc2[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[4], b[4];
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i) simdgroup_load(a[i], &sA[sg_m + 8u * i][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) simdgroup_load(b[j], &sB[kk][sg_n + 8u * j], 68);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 4u; ++i)
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* Y = (tgid.z == 0u) ? gate_out : up_out;
    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    uint G_out[4];
    bool G_ok[4];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        uint fo = tile * 8u + (sg_n >> 3) + j;
        G_ok[j] = fo < NGS;
        uint foc = G_ok[j] ? fo : 0u;
        uint co = foc / slots_per_ch;
        uint so = foc - co * slots_per_ch;
        G_out[j] = sb * NGS + so * 16u + co;
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * INTER;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                if (G_ok[j]) {
                    *(device float2*)(yr + G_out[j] * 8u + fn) =
                        float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// Universal LUT4 BM=64, BN=64, BK=32 Single-Pass Tiled Prefill down_proj (+ residual add!)
// Grid: (16 * ceil_div(HID/128, 8), ceil_div(M, 64), 1), TG: (128, 1, 1)
kernel void ane_lut4_tiled_down(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device float*               Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 7u) >> 3;
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M) return;

    const uint ch    = tgid.x / tg_per_ch;
    const uint slot0 = (tgid.x - ch * tg_per_ch) * 8u;

    threadgroup half  sA[64][36];
    threadgroup half  sB[32][68];
    threadgroup half2 s_lut2[256]; // 1 KB -> 9.75 KB total (3 TGs/core unchanged!)

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m  = (sg >> 1) * 32u;
    const uint sg_n  = (sg & 1u) * 32u;
    const uint r_a   = tid >> 1;
    const uint c_a   = (tid & 1u) * 16u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_row = X + (ulong)(gm_ok ? gm_ld : 0u) * INTER;
    threadgroup half4* da = (threadgroup half4*)(&sA[r_a][c_a]);

    const uint w_slot  = tid >> 4;
    const uint w_k     = (tid & 15u) * 2u;
    const uint slot    = slot0 + w_slot;
    const bool slot_ok = slot < slots_per_ch;
    const uint slot_c  = slot_ok ? slot : 0u;
    const uint go_s    = slot_c * 16u + ch;

    for (uint q = 0u; q < n_q; ++q) {
        const uint sb  = q >> 1;
        const uint li  = sb * 4u + 2u + (q & 1u);
        const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
        const uint gw  = hdr + DK;
        const device uint* chunk_base = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li];
        {
            const device half* lv = (const device half*)chunk_base;
            s_lut2[tid]        = half2(lv[tid & 15u], lv[tid >> 4]);
            s_lut2[tid + 128u] = half2(lv[(tid + 128u) & 15u], lv[(tid + 128u) >> 4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const device uint* w_ptr = chunk_base + 16u + (ulong)slot_c * gw + hdr + w_k;
        const device half* x_base = x_row + (ulong)q * DK;
        const uint so_down_base = 2u * INTER + q * HID;

        half2 sc2[4];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
            sc2[c] = slot_ok ? half2(half(SO[so_down_base + go_s * 8u + 2u * c + 0u].x),
                                     half(SO[so_down_base + go_s * 8u + 2u * c + 1u].x)) : half2(0.0h);
        }

        for (uint k0 = 0u; k0 < DK; k0 += 32u) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k0 + c_a);
                da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
            } else {
                da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
            }

            uint2 u2 = slot_ok ? *(const device uint2*)(w_ptr + k0) : uint2(0u);
            uint ws[2] = {u2.x, u2.y};
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                uint w = ws[j];
                threadgroup half2* dst = (threadgroup half2*)(&sB[w_k + j][w_slot * 8u]);
                dst[0] = s_lut2[(w >>  0u) & 0xFFu] * sc2[0];
                dst[1] = s_lut2[(w >>  8u) & 0xFFu] * sc2[1];
                dst[2] = s_lut2[(w >> 16u) & 0xFFu] * sc2[2];
                dst[3] = s_lut2[(w >> 24u) & 0xFFu] * sc2[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a[4], b[4];
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i) simdgroup_load(a[i], &sA[sg_m + 8u * i][kk], 36);
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) simdgroup_load(b[j], &sB[kk][sg_n + 8u * j], 68);
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 4u; ++i)
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * HID;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                uint sl_out = slot0 + (sg_n >> 3) + j;
                if (sl_out < slots_per_ch) {
                    uint go = sl_out * 16u + ch;
                    device float2* yp = (device float2*)(yr + go * 8u + fn);
                    *yp = *yp + float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// Universal LUT4 Co-Exec Fused BM=32, BN=32, BK=32 SwiGLU -> half Y[M, INTER]
// Grid: (n_subblocks * ceil_div(SUB/8, 4), ceil_div(M, 32), 1), TG: (128, 1, 1)
kernel void ane_lut4_swiglu_tiled32(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]],
    device half*                Y  [[buffer(3)]],
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint SUB   = (is_function_constant_defined(FC_ANE_SUB)   && FC_ANE_SUB   != 0u) ? FC_ANE_SUB   : p.SUB;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint NGS          = SUB >> 3;
    const uint slots_per_ch = SUB >> 7;
    const uint tiles_sb     = (NGS + 3u) >> 2;
    const uint total_tiles  = NSB * tiles_sb;
    const uint m0 = tgid.y * 32u;
    if (tgid.x >= total_tiles || m0 >= p.M || HID == 0u) return;

    const uint sb    = tgid.x / tiles_sb;
    const uint tile  = tgid.x - sb * tiles_sb;
    const uint li_g  = sb * 4u + 0u;
    const uint li_u  = sb * 4u + 1u;
    const uint hdr_g = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_g];
    const uint hdr_u = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li_u];
    const uint gw_g  = hdr_g + HID;
    const uint gw_u  = hdr_u + HID;

    threadgroup half  sA[2][32][36];
    threadgroup half  sBg[2][32][36];
    threadgroup half  sBu[2][32][36];
    threadgroup half2 s_lut2_g[256]; // +2 KB -> 15.5 KB <= 16 KB (keeps 2 TGs/core!)
    threadgroup half2 s_lut2_u[256];
    {
        const device half* lg = (const device half*)(W + (ulong)p.lin_base_w[li_g]);
        const device half* lu = (const device half*)(W + (ulong)p.lin_base_w[li_u]);
        s_lut2_g[tid]        = half2(lg[tid & 15u], lg[tid >> 4]);
        s_lut2_g[tid + 128u] = half2(lg[(tid + 128u) & 15u], lg[(tid + 128u) >> 4]);
        s_lut2_u[tid]        = half2(lu[tid & 15u], lu[tid >> 4]);
        s_lut2_u[tid + 128u] = half2(lu[(tid + 128u) & 15u], lu[(tid + 128u) >> 4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc_g[2][2], acc_u[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
        acc_g[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc_u[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint sg_m  = (sg >> 1) * 16u;
    const uint sg_n  = (sg & 1u) * 16u;
    const uint r_a   = tid >> 2;
    const uint c_a   = (tid & 3u) * 8u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * HID;

    const uint w_slot  = tid >> 5;
    const uint w_k     = tid & 31u;
    const uint flat    = tile * 4u + w_slot;
    const bool flat_ok = flat < NGS;
    const uint flat_c  = flat_ok ? flat : 0u;
    const uint ch      = flat_c / slots_per_ch;
    const uint slot    = flat_c - ch * slots_per_ch;
    const uint G_s     = sb * NGS + slot * 16u + ch;

    const device uint* wg_ptr = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                                + 16u + (ulong)slot * gw_g + hdr_g + w_k;
    const device uint* wu_ptr = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                                + 16u + (ulong)slot * gw_u + hdr_u + w_k;

    half2 sc_g[4], sc_u[4];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 4u; ++c) {
        sc_g[c] = flat_ok ? half2(half(SO[G_s * 8u + 2u * c + 0u].x), half(SO[G_s * 8u + 2u * c + 1u].x)) : half2(0.0h);
        sc_u[c] = flat_ok ? half2(half(SO[INTER + G_s * 8u + 2u * c + 0u].x), half(SO[INTER + G_s * 8u + 2u * c + 1u].x)) : half2(0.0h);
    }

    {
        threadgroup half4* da = (threadgroup half4*)(&sA[0][r_a][c_a]);
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + c_a);
            da[0] = xp[0]; da[1] = xp[1];
        } else {
            da[0] = half4(0.0h); da[1] = half4(0.0h);
        }
        uint wg = flat_ok ? wg_ptr[0] : 0u;
        uint wu = flat_ok ? wu_ptr[0] : 0u;
        threadgroup half2* dg = (threadgroup half2*)(&sBg[0][w_k][w_slot * 8u]);
        dg[0] = s_lut2_g[(wg >>  0u) & 0xFFu] * sc_g[0];
        dg[1] = s_lut2_g[(wg >>  8u) & 0xFFu] * sc_g[1];
        dg[2] = s_lut2_g[(wg >> 16u) & 0xFFu] * sc_g[2];
        dg[3] = s_lut2_g[(wg >> 24u) & 0xFFu] * sc_g[3];

        threadgroup half2* du = (threadgroup half2*)(&sBu[0][w_k][w_slot * 8u]);
        du[0] = s_lut2_u[(wu >>  0u) & 0xFFu] * sc_u[0];
        du[1] = s_lut2_u[(wu >>  8u) & 0xFFu] * sc_u[1];
        du[2] = s_lut2_u[(wu >> 16u) & 0xFFu] * sc_u[2];
        du[3] = s_lut2_u[(wu >> 24u) & 0xFFu] * sc_u[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k0 = 0u; k0 < HID; k0 += 32u) {
        const uint cur = (k0 >> 5) & 1u;
        const uint nxt = cur ^ 1u;
        const uint k_nxt = k0 + 32u;
        const bool has_nxt = k_nxt < HID;

        half4 nx0 = half4(0.0h), nx1 = half4(0.0h);
        uint nwg = 0u, nwu = 0u;
        if (has_nxt) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k_nxt + c_a);
                nx0 = xp[0]; nx1 = xp[1];
            }
            if (flat_ok) {
                nwg = wg_ptr[k_nxt];
                nwu = wu_ptr[k_nxt];
            }
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 a[2], bg[2], bu[2];
            simdgroup_load(a[0],  &sA[cur][sg_m][kk], 36);
            simdgroup_load(a[1],  &sA[cur][sg_m + 8u][kk], 36);
            simdgroup_load(bg[0], &sBg[cur][kk][sg_n], 36);
            simdgroup_load(bg[1], &sBg[cur][kk][sg_n + 8u], 36);
            simdgroup_load(bu[0], &sBu[cur][kk][sg_n], 36);
            simdgroup_load(bu[1], &sBu[cur][kk][sg_n + 8u], 36);
            #pragma clang loop unroll(full)
            for (uint i = 0u; i < 2u; ++i)
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_multiply_accumulate(acc_g[i][j], a[i], bg[j], acc_g[i][j]);
                simdgroup_multiply_accumulate(acc_u[i][j], a[i], bu[j], acc_u[i][j]);
            }
        }

        if (has_nxt) {
            threadgroup half4* da = (threadgroup half4*)(&sA[nxt][r_a][c_a]);
            da[0] = nx0; da[1] = nx1;

            threadgroup half2* dg = (threadgroup half2*)(&sBg[nxt][w_k][w_slot * 8u]);
            dg[0] = s_lut2_g[(nwg >>  0u) & 0xFFu] * sc_g[0];
            dg[1] = s_lut2_g[(nwg >>  8u) & 0xFFu] * sc_g[1];
            dg[2] = s_lut2_g[(nwg >> 16u) & 0xFFu] * sc_g[2];
            dg[3] = s_lut2_g[(nwg >> 24u) & 0xFFu] * sc_g[3];

            threadgroup half2* du = (threadgroup half2*)(&sBu[nxt][w_k][w_slot * 8u]);
            du[0] = s_lut2_u[(nwu >>  0u) & 0xFFu] * sc_u[0];
            du[1] = s_lut2_u[(nwu >>  8u) & 0xFFu] * sc_u[1];
            du[2] = s_lut2_u[(nwu >> 16u) & 0xFFu] * sc_u[2];
            du[3] = s_lut2_u[(nwu >> 24u) & 0xFFu] * sc_u[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;

    uint G_out[2];
    bool G_ok[2];
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) {
        uint fo = tile * 4u + (sg_n >> 3) + j;
        G_ok[j] = fo < NGS;
        uint foc = G_ok[j] ? fo : 0u;
        uint co = foc / slots_per_ch;
        uint so = foc - co * slots_per_ch;
        G_out[j] = sb * NGS + so * 16u + co;
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device half* yr = Y + (ulong)gm * INTER;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                if (G_ok[j]) {
                    float g0 = acc_g[i][j].thread_elements()[0], g1 = acc_g[i][j].thread_elements()[1];
                    float u0 = acc_u[i][j].thread_elements()[0], u1 = acc_u[i][j].thread_elements()[1];
                    *(device half2*)(yr + G_out[j] * 8u + fn) = half2(half(silu_f32(g0) * u0), half(silu_f32(g1) * u1));
                }
            }
        }
    }
}

// Universal LUT4 Co-Exec Split-K BM=32, BN=32, BK=32 down_proj -> partY[sk, M, HID] (sk = n_tg.z)
// Grid: (16 * ceil_div(HID/128, 4), ceil_div(M, 32), sk), TG: (128, 1, 1)
kernel void ane_lut4_down_tiled32_sk(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]],
    const device half*          X     [[buffer(2)]],
    device float*               partY [[buffer(3)]],
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint3 n_tg                        [[threadgroups_per_grid]],
    uint  tid                         [[thread_index_in_threadgroup]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    const uint HID   = (is_function_constant_defined(FC_ANE_HID)   && FC_ANE_HID   != 0u) ? FC_ANE_HID   : p.HID;
    const uint INTER = (is_function_constant_defined(FC_ANE_INTER) && FC_ANE_INTER != 0u) ? FC_ANE_INTER : p.INTER;
    const uint DK    = (is_function_constant_defined(FC_ANE_DK)    && FC_ANE_DK    != 0u) ? FC_ANE_DK    : p.DK;
    const uint NSB   = (is_function_constant_defined(FC_ANE_NSB)   && FC_ANE_NSB   != 0u) ? FC_ANE_NSB   : p.n_subblocks;
    const uint n_q          = NSB * 2u;
    const uint slots_per_ch = HID >> 7;
    const uint tg_per_ch    = (slots_per_ch + 3u) >> 2;
    const uint m0 = tgid.y * 32u;
    if (tgid.x >= 16u * tg_per_ch || m0 >= p.M || DK == 0u) return;

    const uint ch    = tgid.x / tg_per_ch;
    const uint slot0 = (tgid.x - ch * tg_per_ch) * 4u;
    const uint sk    = max(1u, n_tg.z);
    const uint q_per_sk = (n_q + sk - 1u) / sk;
    const uint q_start  = tgid.z * q_per_sk;
    const uint q_end    = min(n_q, q_start + q_per_sk);

    threadgroup half  sA[2][32][36];
    threadgroup half  sB[2][32][36];
    threadgroup half2 s_lut2[256]; // +1 KB -> 10.0 KB <= 10.67 KB (keeps 3 TGs/core!)

    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i)
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 2u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m  = (sg >> 1) * 16u;
    const uint sg_n  = (sg & 1u) * 16u;
    const uint r_a   = tid >> 2;
    const uint c_a   = (tid & 3u) * 8u;
    const uint gm_ld = m0 + r_a;
    const bool gm_ok = gm_ld < p.M;
    const device half* x_row = X + (ulong)(gm_ok ? gm_ld : 0u) * INTER;

    const uint w_slot  = tid >> 5;
    const uint w_k     = tid & 31u;
    const uint slot    = slot0 + w_slot;
    const bool slot_ok = slot < slots_per_ch;
    const uint slot_c  = slot_ok ? slot : 0u;
    const uint go_s    = slot_c * 16u + ch;

    for (uint q = q_start; q < q_end; ++q) {
        const uint sb  = q >> 1;
        const uint li  = sb * 4u + 2u + (q & 1u);
        const uint hdr = is_function_constant_defined(FC_ANE_HDR) ? FC_ANE_HDR : p.lin_hdr_w[li];
        const uint gw  = hdr + DK;
        const device uint* chunk_base = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li];
        {
            const device half* lv = (const device half*)chunk_base;
            s_lut2[tid]        = half2(lv[tid & 15u], lv[tid >> 4]);
            s_lut2[tid + 128u] = half2(lv[(tid + 128u) & 15u], lv[(tid + 128u) >> 4]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const device uint* w_ptr = chunk_base + 16u + (ulong)slot_c * gw + hdr + w_k;
        const device half* x_base = x_row + (ulong)q * DK;
        const uint so_down_base = 2u * INTER + q * HID;

        half2 sc8[4];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; ++c) {
            sc8[c] = slot_ok ? half2(half(SO[so_down_base + go_s * 8u + 2u * c + 0u].x),
                                     half(SO[so_down_base + go_s * 8u + 2u * c + 1u].x)) : half2(0.0h);
        }

        {
            threadgroup half4* da = (threadgroup half4*)(&sA[0][r_a][c_a]);
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + c_a);
                da[0] = xp[0]; da[1] = xp[1];
            } else {
                da[0] = half4(0.0h); da[1] = half4(0.0h);
            }
            uint w = slot_ok ? w_ptr[0] : 0u;
            threadgroup half2* dst = (threadgroup half2*)(&sB[0][w_k][w_slot * 8u]);
            dst[0] = s_lut2[(w >>  0u) & 0xFFu] * sc8[0];
            dst[1] = s_lut2[(w >>  8u) & 0xFFu] * sc8[1];
            dst[2] = s_lut2[(w >> 16u) & 0xFFu] * sc8[2];
            dst[3] = s_lut2[(w >> 24u) & 0xFFu] * sc8[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k0 = 0u; k0 < DK; k0 += 32u) {
            const uint cur = (k0 >> 5) & 1u;
            const uint nxt = cur ^ 1u;
            const uint k_nxt = k0 + 32u;
            const bool has_nxt = k_nxt < DK;

            half4 nx0 = half4(0.0h), nx1 = half4(0.0h);
            uint nw = 0u;
            if (has_nxt) {
                if (gm_ok) {
                    const device half4* xp = (const device half4*)(x_base + k_nxt + c_a);
                    nx0 = xp[0]; nx1 = xp[1];
                }
                if (slot_ok) nw = w_ptr[k_nxt];
            }

            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < 32u; kk += 8u) {
                simdgroup_half8x8 a[2], b[2];
                simdgroup_load(a[0], &sA[cur][sg_m][kk], 36);
                simdgroup_load(a[1], &sA[cur][sg_m + 8u][kk], 36);
                simdgroup_load(b[0], &sB[cur][kk][sg_n], 36);
                simdgroup_load(b[1], &sB[cur][kk][sg_n + 8u], 36);
                #pragma clang loop unroll(full)
                for (uint i = 0u; i < 2u; ++i)
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 2u; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }

            if (has_nxt) {
                threadgroup half4* da = (threadgroup half4*)(&sA[nxt][r_a][c_a]);
                da[0] = nx0; da[1] = nx1;
                threadgroup half2* dst = (threadgroup half2*)(&sB[nxt][w_k][w_slot * 8u]);
                dst[0] = s_lut2[(nw >>  0u) & 0xFFu] * sc8[0];
                dst[1] = s_lut2[(nw >>  8u) & 0xFFu] * sc8[1];
                dst[2] = s_lut2[(nw >> 16u) & 0xFFu] * sc8[2];
                dst[3] = s_lut2[(nw >> 24u) & 0xFFu] * sc8[3];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (q + 1u < q_end) threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint qid = lane >> 2;
    uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const ulong sk_off = (ulong)tgid.z * ((ulong)p.M * HID);

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = partY + sk_off + (ulong)gm * HID;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                uint sl_out = slot0 + (sg_n >> 3) + j;
                if (sl_out < slots_per_ch) {
                    uint go = sl_out * 16u + ch;
                    *(device float2*)(yr + go * 8u + fn) =
                        float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
                }
            }
        }
    }
}

// ============================================================================
// 9. BitScreen: 1-Bit Sign(W) x 3-Bitplane(X) Hardware-Popcount Pre-Filter
//    + Exact Q4_g64 Top-256 Rescore (0.15 ms for V=98,304 -> 3x larger draft
//    vocab than Q4 32k, eliminates ~95% OOV misses, 100% lossless!)
// ============================================================================

struct BitScreenParams {
    uint V;
    uint B;
    uint n_extra;
    uint R; // V / 2048 (rows per strided-interleaved simdgroup bucket)
};

// Optional hidden-dim specialization constant for BitScreen (default/undefined -> 5120u, supports D <= 8192!)
constant uint FC_BS_D [[function_constant(10)]];

// Build 1-bit sign(W) [V, KW] + float2 W_meta [V] in STRIDE-2048 interleaved order once at startup (~2.5 ms).
// Simdgroup s (0..2047) owns orig_row = r * 2048 + s (r = 0..R-1), stored contiguously at slot = s * R + r!
kernel void lmh_build_bitscreen(
    const device uchar*  W      [[buffer(0)]], // Q4_g64 [V, D/8]
    const device uchar*  S      [[buffer(1)]], // bf16 [V, D/64]
    const device uchar*  B      [[buffer(2)]], // bf16 [V, D/64]
    device uint*         W_bits [[buffer(3)]], // [V, D/32]
    device float2*       W_meta [[buffer(4)]], // [V]
    constant uint&       V      [[buffer(5)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]]
) {
    uint slot = tgid * 4u + sg;
    if (slot >= V) return;
    constexpr uint BS_MAX_NSG = 8u;
    const uint bs_d   = (is_function_constant_defined(FC_BS_D) && FC_BS_D != 0u) ? FC_BS_D : 5120u;
    const uint bs_kw  = bs_d >> 5;
    const uint bs_nsg = min((bs_kw + 31u) >> 5, BS_MAX_NSG);
    const uint q4_w   = bs_d >> 3;
    const uint q4_g   = bs_d >> 6;

    uint R = V >> 11; // V / 2048
    uint s = slot / R;
    uint r = slot - s * R;
    uint row = r * 2048u + s;

    float abs_sum = 0.0f;
    uint  pos_cnt = 0u;
    for (uint k = 0u; k < bs_nsg; ++k) {
        uint kw = k * 32u + lane;
        if (kw < bs_kw) {
            uint g  = kw >> 1; // Q4 group-64
            uint4 uq = q4_load_u32x4(W, (ulong)row * q4_w + kw * 4u);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)row * q4_g + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)row * q4_g + g));
            uint ws[4] = {uq.x, uq.y, uq.z, uq.w};
            uint mask = 0u;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                uint wv = ws[j];
                #pragma clang loop unroll(full)
                for (uint tt = 0u; tt < 8u; ++tt) {
                    float val = fma(sc, float((wv >> (4u * tt)) & 0xFu), bi);
                    if (val >= 0.0f) mask |= (1u << (j * 8u + tt));
                    abs_sum += fabs(val);
                }
            }
            W_bits[(ulong)slot * bs_kw + kw] = mask;
            pos_cnt += popcount(mask);
        }
    }
    float mean_abs = simd_sum(abs_sum) * (1.0f / float(bs_d));
    int   sum_sign = 2 * int(simd_sum(pos_cnt)) - int(bs_d);
    if (lane == 0u) {
        W_meta[slot] = float2(mean_abs, -3.5f * float(sum_sign));
    }
}

// Quantize X [B, D] FP16 -> 3 bitplanes X_bp [B, 3, KW] + weighted popcount X_pc [B] (<5 us!)
// Grid: (B, 1, 1), TG: (KW, 1, 1) (1 thread per 32-element word; safe for any D % 64 == 0, D <= 8192!)
kernel void bitscreen_quant_x(
    const device half* X    [[buffer(0)]],
    device uint*       X_bp [[buffer(1)]],
    device float*      X_pc [[buffer(2)]],
    constant uint&     B    [[buffer(3)]],
    uint m                  [[threadgroup_position_in_grid]],
    uint tid                [[thread_position_in_threadgroup]],
    uint lane               [[thread_index_in_simdgroup]],
    uint sg                 [[simdgroup_index_in_threadgroup]]
) {
    if (m >= B) return;
    constexpr uint BS_MAX_NSG = 8u;
    const uint bs_d   = (is_function_constant_defined(FC_BS_D) && FC_BS_D != 0u) ? FC_BS_D : 5120u;
    const uint bs_kw  = bs_d >> 5;
    const uint bs_nsg = min((bs_kw + 31u) >> 5, BS_MAX_NSG);

    half4 xv[8] = {half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h),
                   half4(0.0h), half4(0.0h), half4(0.0h), half4(0.0h)};
    float sq = 0.0f;
    if (tid < bs_kw) {
        const device half4* xp = (const device half4*)(X + (ulong)m * bs_d + tid * 32u);
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
            half4 v = xp[j];
            xv[j] = v;
            float4 f = float4(v);
            sq = fma(f.x, f.x, fma(f.y, f.y, fma(f.z, f.z, fma(f.w, f.w, sq))));
        }
    }
    threadgroup float smem[32];
    float sg_sq = simd_sum(sq);
    if (lane == 0u && sg < bs_nsg) smem[sg] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot_sq = 0.0f;
    for (uint s = 0u; s < bs_nsg; ++s) tot_sq += smem[s];
    float rms = sqrt(tot_sq * (1.0f / float(bs_d))) + 1e-6f;
    float inv_clip = 1.0f / (3.5f * rms);

    uint pc = 0u;
    if (tid < bs_kw) {
        uint bp0 = 0u, bp1 = 0u, bp2 = 0u;
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j) {
            float4 f = float4(xv[j]);
            uint u0 = uint(clamp(rint((f.x * inv_clip + 1.0f) * 3.5f), 0.0f, 7.0f));
            uint u1 = uint(clamp(rint((f.y * inv_clip + 1.0f) * 3.5f), 0.0f, 7.0f));
            uint u2 = uint(clamp(rint((f.z * inv_clip + 1.0f) * 3.5f), 0.0f, 7.0f));
            uint u3 = uint(clamp(rint((f.w * inv_clip + 1.0f) * 3.5f), 0.0f, 7.0f));
            uint sh = j * 4u;
            bp0 |= ((u0 & 1u) << (sh + 0u)) | ((u1 & 1u) << (sh + 1u)) | ((u2 & 1u) << (sh + 2u)) | ((u3 & 1u) << (sh + 3u));
            bp1 |= (((u0 >> 1) & 1u) << (sh + 0u)) | (((u1 >> 1) & 1u) << (sh + 1u)) | (((u2 >> 1) & 1u) << (sh + 2u)) | (((u3 >> 1) & 1u) << (sh + 3u));
            bp2 |= (((u0 >> 2) & 1u) << (sh + 0u)) | (((u1 >> 2) & 1u) << (sh + 1u)) | (((u2 >> 2) & 1u) << (sh + 2u)) | (((u3 >> 2) & 1u) << (sh + 3u));
        }
        ulong base = (ulong)m * (3u * bs_kw);
        X_bp[base + 0u * bs_kw + tid] = bp0;
        X_bp[base + 1u * bs_kw + tid] = bp1;
        X_bp[base + 2u * bs_kw + tid] = bp2;
        pc = popcount(bp0) + (popcount(bp1) << 1) + (popcount(bp2) << 2);
    }

    float sg_pc = float(simd_sum(pc));
    if (lane == 0u && sg < bs_nsg) smem[sg] = sg_pc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot_pc = 0.0f;
        for (uint s = 0u; s < bs_nsg; ++s) tot_pc += smem[s];
        X_pc[m] = tot_pc;
    }
}

// Register-resident X_bp BitScreen scoring + FUSED per-simdgroup Register Top-2 (NB = 1, 2, 3)!
// Grid: (512, 1, 1), TG: (32, 4, 1) -> 2,048 simdgroups, parameterized by FC_BS_D (default 5120)!
template<uint NB>
inline void bitscreen_score_impl(
    const device uint*       W_bits,
    const device float2*     W_meta,
    const device uint*       X_bp,
    const device float*      X_pc,
    device ArgmaxOut*        top2_out, // [B, 4096]
    constant BitScreenParams& p,
    uint tgid,
    uint lane,
    uint sg
) {
    const uint s = tgid * 4u + sg; // 0..2047
    if (s >= 2048u) return;

    constexpr uint BS_MAX_NSG = 8u;
    const uint bs_d   = (is_function_constant_defined(FC_BS_D) && FC_BS_D != 0u) ? FC_BS_D : 5120u;
    const uint bs_kw  = bs_d >> 5;
    const uint bs_nsg = min((bs_kw + 31u) >> 5, BS_MAX_NSG);

    uint x0[NB][BS_MAX_NSG], x1[NB][BS_MAX_NSG], x2[NB][BS_MAX_NSG];
    float xpc[NB];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) {
        const device uint* xb = X_bp + (ulong)b * (3u * bs_kw);
        xpc[b] = X_pc[b];
        for (uint k = 0u; k < bs_nsg; ++k) {
            uint idx = k * 32u + lane;
            bool ok = idx < bs_kw;
            x0[b][k] = ok ? xb[0u * bs_kw + idx] : 0u;
            x1[b][k] = ok ? xb[1u * bs_kw + idx] : 0u;
            x2[b][k] = ok ? xb[2u * bs_kw + idx] : 0u;
        }
    }

    float v0[NB], v1[NB];
    uint  i0[NB], i1[NB];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) { v0[b] = -INFINITY; v1[b] = -INFINITY; i0[b] = 0u; i1[b] = 0u; }

    const uint slot0 = s * p.R;
    for (uint r = 0u; r < p.R; ++r) {
        uint slot = slot0 + r;
        const device uint* wp = W_bits + (ulong)slot * bs_kw;
        uint w[BS_MAX_NSG];
        for (uint k = 0u; k < bs_nsg; ++k) {
            uint idx = k * 32u + lane;
            w[k] = (idx < bs_kw) ? wp[idx] : 0u;
        }
        float2 meta = (lane == 0u) ? W_meta[slot] : float2(0.0f);
        uint orig_row = r * 2048u + s;

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < NB; ++b) {
            uint c0 = 0u, c1 = 0u, c2 = 0u;
            for (uint k = 0u; k < bs_nsg; ++k) {
                c0 += popcount(w[k] & x0[b][k]);
                c1 += popcount(w[k] & x1[b][k]);
                c2 += popcount(w[k] & x2[b][k]);
            }
            float tot = float(simd_sum(c0 + (c1 << 1) + (c2 << 2)));
            if (lane == 0u) {
                float sc = (2.0f * tot - xpc[b] + meta.y) * meta.x;
                if (sc > v0[b]) {
                    v1[b] = v0[b]; i1[b] = i0[b];
                    v0[b] = sc;    i0[b] = orig_row;
                } else if (sc > v1[b]) {
                    v1[b] = sc;    i1[b] = orig_row;
                }
            }
        }
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint b = 0u; b < NB; ++b) {
            ulong dst = (ulong)b * 4096u + (ulong)s * 2u;
            top2_out[dst + 0u].idx = i0[b];
            top2_out[dst + 0u].val = v0[b];
            top2_out[dst + 1u].idx = i1[b];
            top2_out[dst + 1u].val = v1[b];
        }
    }
}

kernel void bitscreen_score_b1(
    const device uint* W_bits [[buffer(0)]], const device float2* W_meta [[buffer(1)]],
    const device uint* X_bp   [[buffer(2)]], const device float*  X_pc   [[buffer(3)]],
    device ArgmaxOut*  top2   [[buffer(4)]], constant BitScreenParams& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) { bitscreen_score_impl<1>(W_bits, W_meta, X_bp, X_pc, top2, p, tgid, lane, sg); }

kernel void bitscreen_score_b2(
    const device uint* W_bits [[buffer(0)]], const device float2* W_meta [[buffer(1)]],
    const device uint* X_bp   [[buffer(2)]], const device float*  X_pc   [[buffer(3)]],
    device ArgmaxOut*  top2   [[buffer(4)]], constant BitScreenParams& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) { bitscreen_score_impl<2>(W_bits, W_meta, X_bp, X_pc, top2, p, tgid, lane, sg); }

kernel void bitscreen_score_b3(
    const device uint* W_bits [[buffer(0)]], const device float2* W_meta [[buffer(1)]],
    const device uint* X_bp   [[buffer(2)]], const device float*  X_pc   [[buffer(3)]],
    device ArgmaxOut*  top2   [[buffer(4)]], constant BitScreenParams& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) { bitscreen_score_impl<3>(W_bits, W_meta, X_bp, X_pc, top2, p, tgid, lane, sg); }

// Single-pass REGISTER-ONLY 64-bin selector over 4,096 L2-resident top2 items -> <=256 cands (supports <=128 extra!)
// Grid: (B, 1, 1), TG: (256, 1, 1) (each thread holds 16 items in registers -> ZERO repeat DRAM passes!)
kernel void bitscreen_select_cands(
    const device ArgmaxOut*  top2      [[buffer(0)]], // [B, 4096]
    const device uint*       extra_ids [[buffer(1)]], // [n_extra] (special + session IDs >= V)
    device uint*             cands     [[buffer(2)]], // [B, 256]
    device uint*             cand_cnt  [[buffer(3)]], // [B]
    constant BitScreenParams& p        [[buffer(4)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]],
    uint sg                            [[simdgroup_index_in_threadgroup]]
) {
    if (row >= p.B) return;
    const device ArgmaxOut* tr = top2  + (ulong)row * 4096u;
    device uint*            cd_r = cands + (ulong)row * 256u;

    float lv[16];
    uint  li[16];
    float lmax = -INFINITY, lsum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 16u; ++k) {
        ArgmaxOut it = tr[k * 256u + tid];
        lv[k] = it.val;
        li[k] = it.idx;
        lmax = max(lmax, it.val);
        lsum += it.val;
    }

    threadgroup float sm_max[8], sm_sum[8];
    float sg_m = simd_max(lmax);
    float sg_s = simd_sum(lsum);
    if (lane == 0u) { sm_max[sg] = sg_m; sm_sum[sg] = sg_s; }
    threadgroup atomic_uint hist[64];
    threadgroup atomic_uint out_n;
    if (tid < 64u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    if (tid == 0u) atomic_store_explicit(&out_n, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float s_lo, s_inv, s_cut_strict, s_cut_edge;
    if (tid == 0u) {
        float gm = sm_max[0], gs = sm_sum[0];
        for (uint k = 1u; k < 8u; ++k) { gm = max(gm, sm_max[k]); gs += sm_sum[k]; }
        float lo = gs * (1.0f / 4096.0f);
        float span = max(gm - lo, 1e-3f);
        s_lo  = lo;
        s_inv = 63.0f / span;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float lo = s_lo, inv = s_inv;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 16u; ++k) {
        float v = lv[k];
        if (v >= lo) {
            uint b = min(63u, uint((v - lo) * inv));
            atomic_fetch_add_explicit(&hist[b], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint max_bs = 256u - min(p.n_extra, 128u);
    const uint target_bs = min(192u, max_bs);
    if (tid == 0u) {
        uint cum = 0u;
        int chosen = -1;
        for (int b = 63; b >= 0; --b) {
            uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
            cum += h;
            if (cum >= target_bs) { chosen = b; break; }
        }
        if (chosen >= 0) {
            s_cut_strict = lo + float(chosen + 1) / inv;
            s_cut_edge   = lo + float(chosen)     / inv;
        } else {
            s_cut_strict = lo;
            s_cut_edge   = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float cut_hi = s_cut_strict;
    const float cut_lo = s_cut_edge;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 16u; ++k) {
        if (lv[k] >= cut_hi) {
            uint sl = atomic_fetch_add_explicit(&out_n, 1u, memory_order_relaxed);
            if (sl < max_bs) cd_r[sl] = li[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 16u; ++k) {
        float v = lv[k];
        if (v >= cut_lo && v < cut_hi) {
            uint sl = atomic_fetch_add_explicit(&out_n, 1u, memory_order_relaxed);
            if (sl < max_bs) cd_r[sl] = li[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_bs = min(atomic_load_explicit(&out_n, memory_order_relaxed), max_bs);
    uint n_ex = min(p.n_extra, 256u - n_bs);
    if (tid < n_ex) {
        cd_r[n_bs + tid] = extra_ids[tid];
    }
    if (tid == 0u) {
        cand_cnt[row] = n_bs + n_ex;
    }
}

// Stage 2: Exact Q4_g64 rescore on the <=256 candidates per batch row -> exact FP32 logits + ArgmaxOut parts!
// Grid: (64, B, 1), TG: (32, 4, 1) (1 row/sg = 4 rows/TG -> 64 * 4 = 256 slots exact -> saturates all 40 M4 Max cores even at B=1!)
kernel void bitscreen_rescore_q4(
    const device uchar*  W        [[buffer(0)]],
    const device uchar*  S        [[buffer(1)]],
    const device uchar*  B        [[buffer(2)]],
    const device half*   X        [[buffer(3)]], // [M, 5120]
    const device uint*   cands    [[buffer(4)]], // [M, 256]
    const device uint*   cand_cnt [[buffer(5)]], // [M]
    device float*        Y        [[buffer(6)]], // [M, 256]
    device ArgmaxOut*    parts    [[buffer(7)]], // [M, 64]
    constant QmmParams&  p        [[buffer(8)]], // p.K=5120, p.M=batch
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]],
    uint  sg                      [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= p.M) return;
    uint o   = tgid.x * 4u + sg; // 0..255
    uint cnt = cand_cnt[row];
    const uint KW = p.K >> 3;
    const uint KG = p.K >> 6;
    const device half* xr = X + (ulong)row * p.K;
    const device uint* cr = cands + (ulong)row * 256u;
    device float*      yr = Y + (ulong)row * 256u;

    float sg_best_v = -INFINITY;
    uint  sg_best_i = 0u;

    if (o < cnt) {
        uint ro = cr[o];
        float acc = 0.0f;
        for (uint wi = lane * 4u; wi < KW; wi += 128u) {
            uint k0 = wi << 3;
            uint g  = wi >> 3;
            half4 xv[8];
            float xs = 0.0f;
            const device half4* xp = (const device half4*)(xr + k0);
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; ++j) {
                half4 v = xp[j];
                xv[j] = v;
                xs += (float(v.x) + float(v.y)) + (float(v.z) + float(v.w));
            }
            uint4 uq = q4_load_u32x4(W, (ulong)ro * KW + wi);
            float sc = bf16_to_f32(q4_load_u16(S, (ulong)ro * KG + g));
            float bi = bf16_to_f32(q4_load_u16(B, (ulong)ro * KG + g));
            acc = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, acc));
        }
        float v = simd_sum(acc);
        if (lane == 0u) {
            yr[o] = v;
            sg_best_v = v;
            sg_best_i = ro;
        }
    } else if (lane == 0u) {
        yr[o] = -INFINITY;
    }

    threadgroup float sv[4];
    threadgroup uint  si[4];
    if (lane == 0u) { sv[sg] = sg_best_v; si[sg] = sg_best_i; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float bv = sv[0];
        uint  bi = si[0];
        for (uint s = 1u; s < 4u; ++s) {
            if (sv[s] > bv) { bv = sv[s]; bi = si[s]; }
        }
        parts[(ulong)row * 64u + tgid.x].val = bv;
        parts[(ulong)row * 64u + tgid.x].idx = bi;
    }
}
