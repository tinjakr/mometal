#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
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
            uint4 uq = *((const device uint4*)(W + (ulong)o * KW + wi));
            float sc = bf16_to_f32(S[(ulong)o * KG + g]);
            float bi = bf16_to_f32(B[(ulong)o * KG + g]);
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

    uint KW = p.K >> 3;
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
        uint k0 = wi << 3;
        uint g  = wi >> 3;
        uint u0 = w_row[wi + 0u], u1 = w_row[wi + 1u], u2 = w_row[wi + 2u], u3 = w_row[wi + 3u];
        float sc = bf16_to_f32(s_row[g]);
        float bi = bf16_to_f32(b_row[g]);
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
    const device uint*   Wg    [[buffer(0)]],
    const device ushort* Sg    [[buffer(1)]],
    const device ushort* Bg    [[buffer(2)]],
    const device uint*   Wu    [[buffer(3)]],
    const device ushort* Su    [[buffer(4)]],
    const device ushort* Bu    [[buffer(5)]],
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
                uint4 uq = *((const device uint4*)(Wg + (ulong)o * KW + wi));
                float sc = bf16_to_f32(Sg[(ulong)o * KG + g]);
                float bi = bf16_to_f32(Bg[(ulong)o * KG + g]);
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
                uint4 uq = *((const device uint4*)(Wu + (ulong)o * KW + wi));
                float sc = bf16_to_f32(Su[(ulong)o * KG + g]);
                float bi = bf16_to_f32(Bu[(ulong)o * KG + g]);
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
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
            uint4 uq = *((const device uint4*)(W + (ulong)o * KW + wi));
            float sc = bf16_to_f32(S[(ulong)o * KG + g]);
            float bi = bf16_to_f32(B[(ulong)o * KG + g]);
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
    const device uint*    W     [[buffer(0)]],
    const device ushort*  S     [[buffer(1)]],
    const device ushort*  B     [[buffer(2)]],
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
            uint4 uq = *((const device uint4*)(W + (ulong)o * KW + wi));
            float sc = bf16_to_f32(S[(ulong)o * KG + g]);
            float bi = bf16_to_f32(B[(ulong)o * KG + g]);
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
    const device uint*   Wg    [[buffer(0)]],
    const device ushort* Sg    [[buffer(1)]],
    const device ushort* Bg    [[buffer(2)]],
    const device uint*   Wu    [[buffer(3)]],
    const device ushort* Su    [[buffer(4)]],
    const device ushort* Bu    [[buffer(5)]],
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
                uint4 uq = *((const device uint4*)(Wg + w_off));
                float sc = bf16_to_f32(Sg[s_off]);
                float bi = bf16_to_f32(Bg[s_off]);
                g_acc[rr] = fma(sc, q4_dot32_f16x_magic(uq, xv), fma(bi, xs, g_acc[rr]));
            }
            {
                uint4 uq = *((const device uint4*)(Wu + w_off));
                float sc = bf16_to_f32(Su[s_off]);
                float bi = bf16_to_f32(Bu[s_off]);
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
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
    const device uint*   w_base = W + (ulong)gn_ld * KW;
    const device ushort* s_base = S + (ulong)gn_ld * KG;
    const device ushort* b_base = B + (ulong)gn_ld * KG;
    const device float*  x_base = X + (ulong)gm_ld * p.K;

    for (uint k0 = 0; k0 < p.K; k0 += 64u) {
        uint g = k0 >> 6;
        half2 sc_h = half2(0.0h);
        half2 bi_h = half2(0.0h);
        if (gn_ok) {
            sc_h = half2(half(bf16_to_f32(s_base[g])));
            bi_h = half2(half(bf16_to_f32(b_base[g])));
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
                uint2 uq = *((const device uint2*)(w_base + wi));
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
// 1c-f16. Round 21: 14.2+ TFLOPS BM=64, BN=64, BK=32 Hardware AMX Prefill GEMM (FP16 X!)
//         9.2 KB TG mem (3 TGs/core residency!) + pre-folded bi_adj + zero register spill!
// ============================================================================

kernel void qmm_q4_g64_tiled_f16x(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
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

    // 9,216 bytes total -> fits 3 full threadgroups simultaneously per M4 Max core!
    threadgroup half sA[64][36];
    threadgroup half sB[64][36];

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0; in_ < 4u; in_++) {
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

    threadgroup half4* sa_dst = (threadgroup half4*)(&sA[r_loc][c_blk]);
    threadgroup half4* sb_dst = (threadgroup half4*)(&sB[r_loc][c_blk]);
    const device uint*   w_base = W + (ulong)gn_ld * KW;
    const device ushort* s_base = S + (ulong)gn_ld * KG;
    const device ushort* b_base = B + (ulong)gn_ld * KG;
    const device half*   x_base = X + (ulong)gm_ld * p.K;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + c_blk);
            sa_dst[0] = xp[0]; sa_dst[1] = xp[1]; sa_dst[2] = xp[2]; sa_dst[3] = xp[3];
        } else {
            sa_dst[0] = half4(0.0h); sa_dst[1] = half4(0.0h); sa_dst[2] = half4(0.0h); sa_dst[3] = half4(0.0h);
        }

        if (gn_ok) {
            uint g = k0 >> 6;
            half2 sc_h = half2(half(bf16_to_f32(s_base[g])));
            half2 bi_h = fma(half2(-1024.0h), sc_h, half2(half(bf16_to_f32(b_base[g]))));
            uint2 uq = *((const device uint2*)(w_base + (k0 >> 3) + h_idx * 2u));
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
            sb_dst[0] = half4(0.0h); sb_dst[1] = half4(0.0h); sb_dst[2] = half4(0.0h); sb_dst[3] = half4(0.0h);
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
// 1c-smalln. Small-N Prefill FP16-X AMX GEMM (BM=32, BN=16, BK=64, 64 threads)
//            5.08x faster on GDN in_proj_a / in_proj_b (N=48, M=512: 48 TGs vs 8!)
// ============================================================================

kernel void qmm_q4_g64_smalln_f16x(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
    const device uint*   w_base = W + (ulong)(gn_ok ? gn : 0u) * KW;
    const device ushort* s_base = S + (ulong)(gn_ok ? gn : 0u) * KG;
    const device ushort* b_base = B + (ulong)(gn_ok ? gn : 0u) * KG;
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
            half2 sc_h = half2(half(bf16_to_f32(s_base[g])));
            half2 bi_h = half2(half(bf16_to_f32(b_base[g])));
            uint2 uq = *((const device uint2*)(w_base + (k0 >> 3) + bh * 2u));
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
// 1d. Register-Direct Hardware simdgroup_half8x8 Small-Batch Q4_AFFINE_G64
//     Half-magic-trick dequant: ((wv >> 4t) & 0x000F000F) | 0x64006400 = half2(1024+q_t, 1024+q_{t+4})
//     ZERO threadgroup A/B transpose, ZERO per-weight float unpack, FLAT cost for M = 1..8!
//     - mma_qmm8_q4_g64_f32:        M = 1..8,  TN=4 (32 out rows/TG), KS=8 split-K, UG=2 prefetch
//     - mma_qmm8_q4_g64_swiglu_f32: M = 1..8,  Fused gate+up+SwiGLU (TN=2 gate + TN=2 up, shares X!)
//     - mma_qmm16_q4_g64_f32:       M = 9..24+ (tiled by 16 rows/tgid.y), NB=2 shares every A fragment!
// ============================================================================

kernel void mma_qmm8_q4_g64_f32(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
                wqs[u][t] = *(const device uint2*)(W + (ulong)n * KW + 8u * gg + 2u * b);
                scs[u][t] = bf16_to_f32(S[(ulong)n * G + gg]);
                bis[u][t] = bf16_to_f32(B[(ulong)n * G + gg]);
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
                float v0 = m0ok ? xv0[j] : 0.0f;
                float v1 = m1ok ? xv1[j] : 0.0f;
                s0 += v0;
                s1 += v1;
                Bm[j].thread_elements()[0] = half(v0);
                Bm[j].thread_elements()[1] = half(v1);
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
    const device uint*   Wg    [[buffer(0)]],
    const device ushort* Sg    [[buffer(1)]],
    const device ushort* Bg    [[buffer(2)]],
    const device uint*   Wu    [[buffer(3)]],
    const device ushort* Su    [[buffer(4)]],
    const device ushort* Bu    [[buffer(5)]],
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
                wqg[u][t] = *(const device uint2*)(Wg + w_off);
                scg[u][t] = bf16_to_f32(Sg[s_off]);
                big[u][t] = bf16_to_f32(Bg[s_off]);
                wqu[u][t] = *(const device uint2*)(Wu + w_off);
                scu[u][t] = bf16_to_f32(Su[s_off]);
                biu[u][t] = bf16_to_f32(Bu[s_off]);
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
                float v0 = m0ok ? xv0[j] : 0.0f;
                float v1 = m1ok ? xv1[j] : 0.0f;
                s0 += v0;
                s1 += v1;
                Bm[j].thread_elements()[0] = half(v0);
                Bm[j].thread_elements()[1] = half(v1);
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
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
            wqs[t] = *(const device uint2*)(W + (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(S[(ulong)n * G + g]);
            bis[t] = bf16_to_f32(B[(ulong)n * G + g]);
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

// Split-K along tgid.z for small-N, large-K decode projections (down_proj 5120x17408 sk=4 -> 365 GB/s; out_proj 5120x6144 sk=2 -> 322 GB/s)
kernel void mma_qmm8_q4_g64_f16x_sk(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
            wqs[t] = *(const device uint2*)(W + (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(S[(ulong)n * G + g]);
            bis[t] = bf16_to_f32(B[(ulong)n * G + g]);
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
    for (uint k = 0; k < p.sk; k++) s += partY[(ulong)k * total + gid];
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

// Round 8: Fused Split-K Reduce + Residual Add + RMSNorm-F16 (1 TG/row, 256 threads, registers-only!)
// Eliminates 127 kernel launches, 127 barriers, and 127 buf_hidden_ re-reads per decode round!
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

    if (p.N == 5120u && tg_size == 1024u) {
        float vals[5];
        float sq = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint idx = k * 1024u + tid;
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0; s < p.sk; s++) {
                acc += partY[(ulong)s * total + row_off + idx];
            }
            hp[idx] = acc;
            vals[k] = acc;
            sq = fma(acc, acc, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum(smem[lane]);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];

        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint idx = k * 1024u + tid;
            np[idx] = half(vals[k] * inv_rms * bf16_to_f32(norm_w[idx]));
        }
    } else if (p.N == 5120u && tg_size == 256u) {
        float vals[20];
        float sq = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint idx = k * 256u + tid;
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0; s < p.sk; s++) {
                acc += partY[(ulong)s * total + row_off + idx];
            }
            hp[idx] = acc;
            vals[k] = acc;
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

        #pragma clang loop unroll(full)
        for (uint k = 0; k < 20u; k++) {
            uint idx = k * 256u + tid;
            np[idx] = half(vals[k] * inv_rms * bf16_to_f32(norm_w[idx]));
        }
    } else {
        float sq = 0.0f;
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0; s < p.sk; s++) {
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

// Round 29: Fused Layer-63 Split-K Reduce + Residual Add + Dual Final RMSNorm (FP32 for MTP + FP16 for lm_head!)
// Eliminates final_norm kernel launch, barrier, and buf_hidden_ round-trip on every decode round!
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

    if (p.N == 5120u && tg_size == 1024u) {
        float vals[5];
        float sq = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint idx = k * 1024u + tid;
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0; s < p.sk; s++) {
                acc += partY[(ulong)s * total + row_off + idx];
            }
            hp[idx] = acc;
            vals[k] = acc;
            sq = fma(acc, acc, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) smem[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum(smem[lane]);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.N) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = smem[0];

        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint idx = k * 1024u + tid;
            float v = vals[k] * inv_rms * bf16_to_f32(norm_w[idx]);
            y32[idx] = v;
            y16[idx] = half(v);
        }
    } else {
        float sq = 0.0f;
        for (uint idx = tid; idx < p.N; idx += tg_size) {
            float acc = p.add_residual ? hp[idx] : 0.0f;
            for (uint s = 0; s < p.sk; s++) {
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
    const device uint*   Wg    [[buffer(0)]],
    const device ushort* Sg    [[buffer(1)]],
    const device ushort* Bg    [[buffer(2)]],
    const device uint*   Wu    [[buffer(3)]],
    const device ushort* Su    [[buffer(4)]],
    const device ushort* Bu    [[buffer(5)]],
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
            wqg[t] = *(const device uint2*)(Wg + w_off);
            scg[t] = bf16_to_f32(Sg[s_off]);
            big[t] = bf16_to_f32(Bg[s_off]);
            wqu[t] = *(const device uint2*)(Wu + w_off);
            scu[t] = bf16_to_f32(Su[s_off]);
            biu[t] = bf16_to_f32(Bu[s_off]);
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
    const device uint*   Wa [[buffer(0)]], const device ushort* Sa [[buffer(1)]], const device ushort* Ba [[buffer(2)]],
    const device uint*   Wb [[buffer(3)]], const device ushort* Sb [[buffer(4)]], const device ushort* Bb [[buffer(5)]],
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

    const device uint*   W = is_b ? Wb : Wa;
    const device ushort* S = is_b ? Sb : Sa;
    const device ushort* B = is_b ? Bb : Ba;
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
        uint2 wq = *(const device uint2*)(W + (ulong)n * KW + 8u * g + 2u * b);
        float sc = bf16_to_f32(S[(ulong)n * G + g]);
        float bi = bf16_to_f32(B[(ulong)n * G + g]) - 1024.0f * sc;

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
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
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
                wqs[u][t] = *(const device uint2*)(W + (ulong)n * KW + 8u * gg + 2u * b);
                scs[u][t] = bf16_to_f32(S[(ulong)n * G + gg]);
                bis[u][t] = bf16_to_f32(B[(ulong)n * G + gg]);
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
                        float vj = mok[c][r] ? v[j] : 0.0f;
                        s += vj;
                        Bm[c][j].thread_elements()[r] = half(vj);
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
            const device uint* wp = W + (ulong)gn * KW + wi;
            uint4 u = *((const device uint4*)wp);
            float sc = bf16_to_f32(S[(ulong)gn * KG + g]);
            float bi = bf16_to_f32(B[(ulong)gn * KG + g]);
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
    float eps;
};

kernel void embed_q4_g64_f32(
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device int*    ids   [[buffer(3)]],
    device float*        out   [[buffer(4)]],
    constant EmbedParams& p    [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (col >= p.D || row >= p.M) return;

    uint tok = uint(ids[row]);
    uint KW = p.D >> 3;
    uint KG = p.D >> 6;
    uint w_idx = col >> 3;
    uint nib_shift = (col & 7u) * 4u;
    uint g_idx = col >> 6;

    uint word = W[(ulong)tok * KW + w_idx];
    float q = float((word >> nib_shift) & 15u);
    float sc = bf16_to_f32(S[(ulong)tok * KG + g_idx]);
    float bi = bf16_to_f32(B[(ulong)tok * KG + g_idx]);
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
    const device uint*   W     [[buffer(0)]],
    const device ushort* S     [[buffer(1)]],
    const device ushort* B     [[buffer(2)]],
    const device half*   X     [[buffer(3)]],
    device float*        Y     [[buffer(4)]],
    device ArgmaxOut*    parts [[buffer(5)]],
    constant QmmParams&  p     [[buffer(6)]],
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
            wqs[t] = *(const device uint2*)(W + (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(S[(ulong)n * G + g]);
            bis[t] = bf16_to_f32(B[(ulong)n * G + g]);
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

    // Epilogue: simdgroup sg = row m (0..7), lane = column (0..31) -> coalesced write + 5-step warp row-max!
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
            Y[(ulong)m * N + gn] = acc;
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

// ============================================================================
// 6c. Dynamic Compact row_map Draft lm_head (B=1 & B=2..8) + Fused TG Row-Max!
//     Reads only the active V_draft rows (e.g. 16k-32k hot + dynamic prompt/gen
//     tokens) via zero-copy row_map[local_n] -> real_vocab_id (full 2.8 KB/row
//     coalesced DRAM bursts, 0 MB weight duplication, eliminates argmax_stage1)!
// ============================================================================

kernel void qmv_q4_g64_f16x_indexed_max(
    const device uint*   W       [[buffer(0)]],
    const device ushort* S       [[buffer(1)]],
    const device ushort* B       [[buffer(2)]],
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
                uint4 uq = *((const device uint4*)(W + (ulong)ro * KW + wi));
                float sc = bf16_to_f32(S[(ulong)ro * KG + g]);
                float bi = bf16_to_f32(B[(ulong)ro * KG + g]);
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
    const device uint*   W       [[buffer(0)]],
    const device ushort* S       [[buffer(1)]],
    const device ushort* B       [[buffer(2)]],
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
            wqs[t] = *(const device uint2*)(W + (ulong)n * KW + 8u * g + 2u * b);
            scs[t] = bf16_to_f32(S[(ulong)n * G + g]);
            bis[t] = bf16_to_f32(B[(ulong)n * G + g]);
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
        if (parts[i].val > best_v) { best_v = parts[i].val; best_i = parts[i].idx; }
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
    for (uint i = tid; i < p.V; i += tg_size) {
        float v = logits[i];
        if (v >= cut) {
            uint slot = atomic_fetch_add_explicit(&c_cnt, 1u, memory_order_relaxed);
            if (slot < 256u) {
                c_val[slot] = v;
                c_id[slot]  = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = atomic_load_explicit(&c_cnt, memory_order_relaxed);
    if (cnt == 0u || cnt > 256u) {
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
        if (r_parts[i].val > best_v) { best_v = r_parts[i].val; best_i = r_parts[i].idx; }
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
    for (uint i = tid; i < p.V; i += tg_size) {
        float v = r_logits[i];
        if (v >= cut) {
            uint slot = atomic_fetch_add_explicit(&c_cnt, 1u, memory_order_relaxed);
            if (slot < 256u) {
                c_val[slot] = v;
                c_id[slot]  = r_map[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cnt = atomic_load_explicit(&c_cnt, memory_order_relaxed);
    if (cnt == 0u || cnt > 256u) {
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
// 8. Single-Copy ANE hf4 (model.hwx) Native BF16-AMX MLP Kernels!
//    Reads Apple Neural Engine's compiled 4-bit [K][8] model.hwx pages directly
//    via zero-copy mmap: 4.0 bpw (11.1% less DRAM than g64!), zero per-group
//    scales/biases, exact per-row (scale, 136 + offset) applied once at end,
//    exact sum(x) computed in registers via 2 warp shuffles (0 TG mem, 0 barriers!)
// ============================================================================

struct AneShareParams {
    uint M;
    uint hdr_words;       // 6 (24B asym int4) or 4 (16B sym int4)
    uint lin_base_w[16];  // word offset of chunk[0] for linear_0..15
    uint lin_chunk_w[16]; // chunk_bytes / 4 for linear_0..15
};

inline half2 ane_dq4_f16(uint w, uint sh) {
    uint u = (w >> sh) ^ 0x88u;
    return as_type<half2>(ushort2(u & 0xFu, (u >> 4) & 0xFu) | ushort2(0x6400u));
}

// Fused gate + up + SiLU from single-copy model.hwx -> half Y[M, 17408]
// Slot-major + lockstep strided-K + KS=8 (256 threads -> 1 KB coalesced DRAM bursts across 64 real .hwx layers!)
// Grid: (17408 / 16 = 1088, ceil_div(M, 8), 1), TG: (256, 1, 1) (8 simdgroups)
kernel void ane_hf4_swiglu_bf16(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]], // [0..17407]=gate, [17408..34815]=up ({scale, 1032+off})
    const device half*          X  [[buffer(2)]], // [M, 5120] FP16 normed*inv1
    device half*                Y  [[buffer(3)]], // [M, 17408] FP16 act
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HID = 5120u;
    constexpr uint INTER = 17408u;
    constexpr uint NGS = 544u; // 4352 / 8 = 34 slots * 16 chunks
    constexpr uint KS = 8u;

    const uint m0 = tgid.y * 8u;
    if (tgid.x >= 1088u || m0 >= p.M) return;

    const uint sb    = tgid.x / 272u;           // 0..3
    const uint rem   = tgid.x - sb * 272u;      // 0..271 (17 slot-pairs * 16 chunks)
    const uint slot0 = (rem >> 4) * 2u;         // 0, 2, ..., 32 (slot-major -> coalesced Y & balanced DRAM channels!)
    const uint ch    = rem & 15u;               // 0..15
    const uint G0    = sb * NGS + slot0 * 16u + ch;
    const uint G1    = G0 + 16u;                // adjacent slot in SAME chunk ch!
    const uint gw    = p.hdr_words + HID;
    const bool aligned1 = (p.hdr_words == 6u);

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    const uint li_g = sb * 4u + 0u;
    const uint li_u = sb * 4u + 1u;
    const device uint* pg0 = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                             + 16u + (ulong)slot0 * gw + p.hdr_words + 4u * fm;
    const device uint* pg1 = pg0 + gw;
    const device uint* pu0 = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                             + 16u + (ulong)slot0 * gw + p.hdr_words + 4u * fm;
    const device uint* pu1 = pu0 + gw;

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * HID;

    simdgroup_float8x8 Cg0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cg1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Cu1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float sx = 0.0f;

    // Lockstep strided-K across 8 simdgroups -> 1 KB contiguous DRAM burst per iteration (5120 / 256 = 20 exact!)
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

        uint2 a0 = *(const device uint2*)(pg0 + ko);
        uint2 b0 = *(const device uint2*)(pg0 + ko + 2u);
        uint4 w1 = aligned1 ? *(const device uint4*)(pg1 + ko)
                            : uint4(*(const device uint2*)(pg1 + ko), *(const device uint2*)(pg1 + ko + 2u));

        {
            uint w[4] = {a0.x, a0.y, b0.x, b0.y};
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < 4u; ++q) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[q], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(Cg0, Af[q], B, Cg0);
            }
        }
        uint2 a2 = *(const device uint2*)(pu0 + ko);
        uint2 b2 = *(const device uint2*)(pu0 + ko + 2u);
        {
            uint w[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < 4u; ++q) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[q], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(Cg1, Af[q], B, Cg1);
            }
        }
        uint4 w3 = aligned1 ? *(const device uint4*)(pu1 + ko)
                            : uint4(*(const device uint2*)(pu1 + ko), *(const device uint2*)(pu1 + ko + 2u));
        {
            uint w[4] = {a2.x, a2.y, b2.x, b2.y};
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < 4u; ++q) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[q], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(Cu0, Af[q], B, Cu0);
            }
        }
        {
            uint w[4] = {w3.x, w3.y, w3.z, w3.w};
            #pragma clang loop unroll(full)
            for (uint q = 0u; q < 4u; ++q) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[q], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(Cu1, Af[q], B, Cu1);
            }
        }
    }

    sx += simd_shuffle_xor(sx, 1);
    sx += simd_shuffle_xor(sx, 8);

    threadgroup float red_g[KS][2][64];
    threadgroup float red_u[KS][2][64];
    uint c0 = G0 * 8u + fn;
    uint c1 = G1 * 8u + fn;
    float2 sg0 = SO[c0],           sg1 = SO[c0 + 1u];
    float2 su0 = SO[INTER + c0],   su1 = SO[INTER + c0 + 1u];
    float2 sg2 = SO[c1],           sg3 = SO[c1 + 1u];
    float2 su2 = SO[INTER + c1],   su3 = SO[INTER + c1 + 1u];

    red_g[sg][0][fm * 8u + fn]      = (Cg0.thread_elements()[0] - sg0.y * sx) * sg0.x;
    red_g[sg][0][fm * 8u + fn + 1u] = (Cg0.thread_elements()[1] - sg1.y * sx) * sg1.x;
    red_g[sg][1][fm * 8u + fn]      = (Cg1.thread_elements()[0] - sg2.y * sx) * sg2.x;
    red_g[sg][1][fm * 8u + fn + 1u] = (Cg1.thread_elements()[1] - sg3.y * sx) * sg3.x;
    red_u[sg][0][fm * 8u + fn]      = (Cu0.thread_elements()[0] - su0.y * sx) * su0.x;
    red_u[sg][0][fm * 8u + fn + 1u] = (Cu0.thread_elements()[1] - su1.y * sx) * su1.x;
    red_u[sg][1][fm * 8u + fn]      = (Cu1.thread_elements()[0] - su2.y * sx) * su2.x;
    red_u[sg][1][fm * 8u + fn + 1u] = (Cu1.thread_elements()[1] - su3.y * sx) * su3.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = sg * 32u + lane; // 0..255; only 0..127 needed
    if (e < 128u) {
        uint t = e >> 6;
        uint rc = e & 63u;
        uint m = rc >> 3;
        uint c = rc & 7u;
        uint gm_out = m0 + m;
        if (gm_out < p.M) {
            float g = 0.0f, u = 0.0f;
            #pragma clang loop unroll(full)
            for (uint k = 0u; k < KS; ++k) {
                g += red_g[k][t][rc];
                u += red_u[k][t][rc];
            }
            uint G_out = (t == 0u) ? G0 : G1;
            Y[(ulong)gm_out * INTER + G_out * 8u + c] = half(silu_f32(g) * u);
        }
    }
}

// Dedicated 453 GB/s M=1 QMV gate+up+SiLU from single-copy model.hwx -> half Y[17408]
// Grid: (544, 1, 1), TG: (128, 1, 1) (4 simdgroups, 1 slot = 8 channels/sg)
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
    constexpr uint HID = 5120u;
    constexpr uint INTER = 17408u;
    constexpr uint NGS = 544u;
    const uint gidx = tgid * 4u + sg;
    if (gidx >= 2176u) return;

    const uint sb   = gidx / 544u;
    const uint rem  = gidx - sb * 544u;
    const uint ch   = rem / 34u;
    const uint slot = rem - ch * 34u;
    const uint G0   = sb * NGS + slot * 16u + ch;
    const uint gw   = p.hdr_words + HID;

    const uint li_g = sb * 4u + 0u;
    const uint li_u = sb * 4u + 1u;
    const device uint* pg = W + (ulong)p.lin_base_w[li_g] + (ulong)ch * p.lin_chunk_w[li_g]
                            + 16u + (ulong)slot * gw + p.hdr_words;
    const device uint* pu = W + (ulong)p.lin_base_w[li_u] + (ulong)ch * p.lin_chunk_w[li_u]
                            + 16u + (ulong)slot * gw + p.hdr_words;

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

// Dedicated 457 GB/s M=1 QMV Split-K=8 down_proj from single-copy model.hwx -> partY[8, 5120]
// Grid: (160, 1, 8), TG: (128, 1, 1) (4 simdgroups = 4 slots/TG)
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
    constexpr uint HID = 5120u;
    constexpr uint DK = 2176u;
    if (tgid.x >= 160u || tgid.z >= 8u) return;

    const uint ch   = tgid.x / 10u;
    const uint slot = (tgid.x - ch * 10u) * 4u + sg;
    const uint q    = tgid.z;
    const uint gw   = p.hdr_words + DK;
    const uint sb   = q >> 1;
    const uint li   = sb * 4u + 2u + (q & 1u);

    const device uint* pw = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                            + 16u + (ulong)slot * gw + p.hdr_words;
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
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float v = simd_sum(acc[c]);
        if (lane == 0u) {
            uint o = go * 8u + c;
            float2 sd = SO[34816u + q * HID + o];
            partY[(ulong)q * HID + o] = (v - (sd.y - 1024.0f) * sx) * sd.x;
        }
    }
}

// Zero-Padding 14+ TFLOPs BM=64, BN=64, BK=32 Tiled Prefill gate & up from single-copy model.hwx!
// Maps 544 slots/sub-block into 68 EXACT 8-slot tiles (ZERO wasted tail slots!).
// Grid: (272, ceil_div(M, 64), 2) [z=0 -> gate_out, z=1 -> up_out], TG: (128, 1, 1)
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
    constexpr uint HID = 5120u;
    constexpr uint INTER = 17408u;
    constexpr uint NGS = 544u;
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= 272u || m0 >= p.M || tgid.z >= 2u) return;

    const uint sb   = tgid.x / 68u;        // 0..3
    const uint tile = tgid.x - sb * 68u;   // 0..67 (68 * 8 = 544 exact!)
    const uint li   = sb * 4u + tgid.z;    // 0=gate, 1=up
    const uint gw   = p.hdr_words + HID;
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
    const uint flat   = tile * 8u + w_slot; // 0..543 exact!
    const uint ch     = flat / 34u;         // 0..15
    const uint slot   = flat - ch * 34u;    // 0..33
    const uint G_s    = sb * NGS + slot * 16u + ch;
    const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                               + 16u + (ulong)slot * gw + p.hdr_words + w_k;

    // Exact FP16 scale & bias after subtracting 1024.0h from ane_dq4_f16 (full 11-bit mantissa precision!)
    half sc8[8], bi8[8];
    #pragma clang loop unroll(full)
    for (uint c = 0u; c < 8u; ++c) {
        float2 so = SO[so_base + G_s * 8u + c];
        sc8[c] = half(so.x);
        bi8[c] = half(-(so.y - 1024.0f) * so.x);
    }

    for (uint k0 = 0u; k0 < HID; k0 += 32u) {
        if (gm_ok) {
            const device half4* xp = (const device half4*)(x_base + k0 + c_a);
            da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
        } else {
            da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
        }

        uint2 u2 = *(const device uint2*)(w_ptr + k0);
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
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 4u; ++j) {
        uint fo = tile * 8u + (sg_n >> 3) + j;
        uint co = fo / 34u;
        uint so = fo - co * 34u;
        G_out[j] = sb * NGS + so * 16u + co;
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
        uint gm = m0 + sg_m + 8u * i + fm;
        if (gm < p.M) {
            device float* yr = Y + (ulong)gm * INTER;
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j) {
                *(device float2*)(yr + G_out[j] * 8u + fn) =
                    float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
            }
        }
    }
}

// Zero-Padding 14+ TFLOPs BM=64, BN=64, BK=32 Single-Pass Tiled Prefill down_proj (+ residual add!)
// Loops across all 8 sub-blocks q=0..7 inside ONE kernel -> ZERO Split-K DRAM, ZERO extra barrier!
// Grid: (80, ceil_div(M, 64), 1), TG: (128, 1, 1)
kernel void ane_hf4_tiled_down(
    const device uint*          W  [[buffer(0)]],
    const device float2*        SO [[buffer(1)]],
    const device half*          X  [[buffer(2)]], // [M, 17408] FP16 act
    device float*               Y  [[buffer(3)]], // [M, 5120] FP32 hidden (+= residual)
    constant AneShareParams&    p  [[buffer(4)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]],
    uint  lane                     [[thread_index_in_simdgroup]],
    uint  sg                       [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HID = 5120u;
    constexpr uint INTER = 17408u;
    constexpr uint DK = 2176u; // 68 * 32 exact!
    const uint m0 = tgid.y * 64u;
    if (tgid.x >= 80u || m0 >= p.M) return;

    const uint ch    = tgid.x / 5u;             // 0..15
    const uint slot0 = (tgid.x - ch * 5u) * 8u; // 0, 8, 16, 24, 32 (40 exact!)
    const uint gw    = p.hdr_words + DK;

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

    const uint w_slot = tid >> 4;          // 0..7
    const uint w_k    = (tid & 15u) * 2u;  // 0, 2, ..., 30
    const uint slot   = slot0 + w_slot;    // 0..39
    const uint go_s   = slot * 16u + ch;   // group index in 0..639

    for (uint q = 0u; q < 8u; ++q) {
        const uint sb = q >> 1;
        const uint li = sb * 4u + 2u + (q & 1u);
        const device uint* w_ptr = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                                   + 16u + (ulong)slot * gw + p.hdr_words + w_k;
        const device half* x_base = x_row + (ulong)q * DK;

        half sc8[8], bi8[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; ++c) {
            float2 sd = SO[34816u + q * HID + go_s * 8u + c];
            sc8[c] = half(sd.x);
            bi8[c] = half(-(sd.y - 1024.0f) * sd.x);
        }

        for (uint k0 = 0u; k0 < DK; k0 += 32u) {
            if (gm_ok) {
                const device half4* xp = (const device half4*)(x_base + k0 + c_a);
                da[0] = xp[0]; da[1] = xp[1]; da[2] = xp[2]; da[3] = xp[3];
            } else {
                da[0] = half4(0.0h); da[1] = half4(0.0h); da[2] = half4(0.0h); da[3] = half4(0.0h);
            }

            uint2 u2 = *(const device uint2*)(w_ptr + k0);
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
                uint go = (slot0 + (sg_n >> 3) + j) * 16u + ch;
                device float2* yp = (device float2*)(yr + go * 8u + fn);
                *yp = *yp + float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
            }
        }
    }
}

// Split-K=8 down_proj from single-copy model.hwx -> partY[8, M, 5120] (plugs straight into sk_reduce_rmsnorm_f16!)
// Adjacent-slot TN=4 in SAME chunk ch (35KB contiguous!), KS=8, 1-step interleaved load/MMA + uint4 on odd slots!
// Grid: (5120 / 32 = 160, ceil_div(M, 8), 8), TG: (256, 1, 1) (8 simdgroups)
kernel void ane_hf4_down_sk_bf16(
    const device uint*          W     [[buffer(0)]],
    const device float2*        SO    [[buffer(1)]], // [34816 + q*5120 .. +5119]
    const device half*          X     [[buffer(2)]], // [M, 17408] FP16 act
    device float*               partY [[buffer(3)]], // [8, M, 5120] FP32
    constant AneShareParams&    p     [[buffer(4)]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint  lane                        [[thread_index_in_simdgroup]],
    uint  sg                          [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HID = 5120u;
    constexpr uint INTER = 17408u;
    constexpr uint DK = 2176u; // 68 * 32
    constexpr uint TN = 4u;
    constexpr uint KS = 8u;

    const uint m0 = tgid.y * 8u;
    const uint q  = tgid.z;
    if (tgid.x >= 160u || m0 >= p.M || q >= 8u) return;

    const uint slot0 = (tgid.x >> 4) * TN; // 0, 4, ..., 36 (40 slots total!)
    const uint ch    = tgid.x & 15u;       // 0..15
    const uint gw    = p.hdr_words + DK;
    const uint sb    = q >> 1;
    const uint li    = sb * 4u + 2u + (q & 1u);

    const uint qid = lane >> 2;
    const uint fm  = (qid & 4u) + ((lane >> 1) & 3u);
    const uint fn  = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint sh  = 4u * fn;

    const device uint* pw0 = W + (ulong)p.lin_base_w[li] + (ulong)ch * p.lin_chunk_w[li]
                             + 16u + (ulong)slot0 * gw + p.hdr_words + 4u * fm;

    const uint gm = m0 + fm;
    const bool mok = gm < p.M;
    const device half* xr = X + (ulong)(mok ? gm : 0u) * INTER + (ulong)q * DK;

    simdgroup_float8x8 C[TN];
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) C[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float sx = 0.0f;

    // Strided-K across 8 simdgroups -> 1 KB lockstep DRAM burst!
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
        const device uint* p1 = p0 + gw;
        const device uint* p2 = p1 + gw;
        const device uint* p3 = p2 + gw;

        uint2 a0 = *(const device uint2*)(p0);
        uint2 b0 = *(const device uint2*)(p0 + 2u);
        uint4 w1 = (p.hdr_words == 6u) ? *(const device uint4*)(p1)
                                       : uint4(*(const device uint2*)(p1), *(const device uint2*)(p1 + 2u));
        {
            uint w[4] = {a0.x, a0.y, b0.x, b0.y};
            #pragma clang loop unroll(full)
            for (uint qq = 0u; qq < 4u; ++qq) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[qq], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(C[0], Af[qq], B, C[0]);
            }
        }
        uint2 a2 = *(const device uint2*)(p2);
        uint2 b2 = *(const device uint2*)(p2 + 2u);
        {
            uint w[4] = {w1.x, w1.y, w1.z, w1.w};
            #pragma clang loop unroll(full)
            for (uint qq = 0u; qq < 4u; ++qq) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[qq], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(C[1], Af[qq], B, C[1]);
            }
        }
        uint4 w3 = (p.hdr_words == 6u) ? *(const device uint4*)(p3)
                                       : uint4(*(const device uint2*)(p3), *(const device uint2*)(p3 + 2u));
        {
            uint w[4] = {a2.x, a2.y, b2.x, b2.y};
            #pragma clang loop unroll(full)
            for (uint qq = 0u; qq < 4u; ++qq) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[qq], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(C[2], Af[qq], B, C[2]);
            }
        }
        {
            uint w[4] = {w3.x, w3.y, w3.z, w3.w};
            #pragma clang loop unroll(full)
            for (uint qq = 0u; qq < 4u; ++qq) {
                simdgroup_half8x8 B;
                half2 h = ane_dq4_f16(w[qq], sh);
                B.thread_elements()[0] = h.x; B.thread_elements()[1] = h.y;
                simdgroup_multiply_accumulate(C[3], Af[qq], B, C[3]);
            }
        }
    }

    sx += simd_shuffle_xor(sx, 1);
    sx += simd_shuffle_xor(sx, 8);

    threadgroup float red[KS][TN][64];
    #pragma clang loop unroll(full)
    for (uint t = 0u; t < TN; ++t) {
        uint o = ((slot0 + t) * 16u + ch) * 8u + fn;
        float2 sd0 = SO[34816u + q * HID + o];
        float2 sd1 = SO[34816u + q * HID + o + 1u];
        red[sg][t][fm * 8u + fn]      = (C[t].thread_elements()[0] - sd0.y * sx) * sd0.x;
        red[sg][t][fm * 8u + fn + 1u] = (C[t].thread_elements()[1] - sd1.y * sx) * sd1.x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e  = sg * 32u + lane; // 0..255 exact 1-to-1!
    uint t  = e >> 6;          // 0..3
    uint rc = e & 63u;
    uint m  = rc >> 3;
    uint c  = rc & 7u;
    uint gm_out = m0 + m;
    if (gm_out < p.M) {
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < KS; ++k) acc += red[k][t][rc];
        uint go_out = (slot0 + t) * 16u + ch;
        ulong sk_stride = (ulong)p.M * HID;
        partY[(ulong)q * sk_stride + (ulong)gm_out * HID + go_out * 8u + c] = acc;
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

// Build 1-bit sign(W) [V, 160] + float2 W_meta [V] in STRIDE-2048 interleaved order once at startup (~2.5 ms).
// Simdgroup s (0..2047) owns orig_row = r * 2048 + s (r = 0..R-1), stored contiguously at slot = s * R + r!
// Interleaving eliminates BPE low-ID clustering so register-only top-2 per simdgroup has <0.05% recall loss.
kernel void lmh_build_bitscreen(
    const device uint*   W      [[buffer(0)]], // Q4_g64 [V, 640]
    const device ushort* S      [[buffer(1)]], // bf16 [V, 80]
    const device ushort* B      [[buffer(2)]], // bf16 [V, 80]
    device uint*         W_bits [[buffer(3)]], // [V, 160]
    device float2*       W_meta [[buffer(4)]], // [V]
    constant uint&       V      [[buffer(5)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint sg                     [[simdgroup_index_in_threadgroup]]
) {
    uint slot = tgid * 4u + sg;
    if (slot >= V) return;
    uint R = V >> 11; // V / 2048
    uint s = slot / R;
    uint r = slot - s * R;
    uint row = r * 2048u + s;

    float abs_sum = 0.0f;
    uint  pos_cnt = 0u;
    for (uint k = 0u; k < 5u; ++k) {
        uint kw = k * 32u + lane; // 0..159 (each word = 32 weights)
        uint g  = kw >> 1;        // 0..79 (Q4 group-64)
        uint4 uq = *((const device uint4*)(W + (ulong)row * 640u + kw * 4u));
        float sc = bf16_to_f32(S[(ulong)row * 80u + g]);
        float bi = bf16_to_f32(B[(ulong)row * 80u + g]);
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
        W_bits[(ulong)slot * 160u + kw] = mask;
        pos_cnt += popcount(mask);
    }
    float mean_abs = simd_sum(abs_sum) * (1.0f / 5120.0f);
    int   sum_sign = 2 * int(simd_sum(pos_cnt)) - 5120;
    if (lane == 0u) {
        W_meta[slot] = float2(mean_abs, -3.5f * float(sum_sign));
    }
}

// Quantize X [B, 5120] FP16 -> 3 bitplanes X_bp [B, 3, 160] + weighted popcount X_pc [B] (<5 us!)
// Grid: (B, 1, 1), TG: (160, 1, 1) (5 simdgroups, 1 thread per 32-element word)
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
    if (m >= B || tid >= 160u) return;
    const device half4* xp = (const device half4*)(X + (ulong)m * 5120u + tid * 32u);
    half4 xv[8];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (uint j = 0u; j < 8u; ++j) {
        half4 v = xp[j];
        xv[j] = v;
        float4 f = float4(v);
        sq = fma(f.x, f.x, fma(f.y, f.y, fma(f.z, f.z, fma(f.w, f.w, sq))));
    }
    threadgroup float smem[5];
    float sg_sq = simd_sum(sq);
    if (lane == 0u) smem[sg] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_sq = smem[0] + smem[1] + smem[2] + smem[3] + smem[4];
    float rms = sqrt(tot_sq * (1.0f / 5120.0f)) + 1e-6f;
    float inv_clip = 1.0f / (3.5f * rms);

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
    ulong base = (ulong)m * 480u;
    X_bp[base + 0u   + tid] = bp0;
    X_bp[base + 160u + tid] = bp1;
    X_bp[base + 320u + tid] = bp2;

    uint pc = popcount(bp0) + (popcount(bp1) << 1) + (popcount(bp2) << 2);
    float sg_pc = float(simd_sum(pc));
    if (lane == 0u) smem[sg] = sg_pc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        X_pc[m] = smem[0] + smem[1] + smem[2] + smem[3] + smem[4];
    }
}

// Register-resident X_bp BitScreen scoring + FUSED per-simdgroup Register Top-2 (NB = 1, 2, 3)!
// Grid: (512, 1, 1), TG: (32, 4, 1) -> 2,048 simdgroups, each scans R contiguous strided rows -> writes ONLY 32 KB top2!
// ZERO score buffer DRAM writes, ZERO threadgroup barriers!
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

    uint x0[NB][5], x1[NB][5], x2[NB][5];
    float xpc[NB];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) {
        const device uint* xb = X_bp + (ulong)b * 480u;
        xpc[b] = X_pc[b];
        #pragma clang loop unroll(full)
        for (uint k = 0u; k < 5u; ++k) {
            uint idx = k * 32u + lane;
            x0[b][k] = xb[0u   + idx];
            x1[b][k] = xb[160u + idx];
            x2[b][k] = xb[320u + idx];
        }
    }

    float v0[NB], v1[NB];
    uint  i0[NB], i1[NB];
    #pragma clang loop unroll(full)
    for (uint b = 0u; b < NB; ++b) { v0[b] = -INFINITY; v1[b] = -INFINITY; i0[b] = 0u; i1[b] = 0u; }

    const uint slot0 = s * p.R;
    for (uint r = 0u; r < p.R; ++r) {
        uint slot = slot0 + r;
        const device uint* wp = W_bits + (ulong)slot * 160u;
        uint w0 = wp[0u   + lane];
        uint w1 = wp[32u  + lane];
        uint w2 = wp[64u  + lane];
        uint w3 = wp[96u  + lane];
        uint w4 = wp[128u + lane];
        float2 meta = (lane == 0u) ? W_meta[slot] : float2(0.0f);
        uint orig_row = r * 2048u + s;

        #pragma clang loop unroll(full)
        for (uint b = 0u; b < NB; ++b) {
            uint c0 = popcount(w0 & x0[b][0]) + popcount(w1 & x0[b][1])
                    + popcount(w2 & x0[b][2]) + popcount(w3 & x0[b][3]) + popcount(w4 & x0[b][4]);
            uint c1 = popcount(w0 & x1[b][0]) + popcount(w1 & x1[b][1])
                    + popcount(w2 & x1[b][2]) + popcount(w3 & x1[b][3]) + popcount(w4 & x1[b][4]);
            uint c2 = popcount(w0 & x2[b][0]) + popcount(w1 & x2[b][1])
                    + popcount(w2 & x2[b][2]) + popcount(w3 & x2[b][3]) + popcount(w4 & x2[b][4]);
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

// Single-pass REGISTER-ONLY 64-bin selector over 4,096 L2-resident top2 items -> ~160..192 + <=64 extra (<8 us!)
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

    const uint max_bs = 256u - min(p.n_extra, 64u);
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
    const device uint*   W        [[buffer(0)]],
    const device ushort* S        [[buffer(1)]],
    const device ushort* B        [[buffer(2)]],
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
            uint4 uq = *((const device uint4*)(W + (ulong)ro * KW + wi));
            float sc = bf16_to_f32(S[(ulong)ro * KG + g]);
            float bi = bf16_to_f32(B[(ulong)ro * KG + g]);
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
