#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline ushort f32_to_bf16_rne(float v) {
    uint u = as_type<uint>(v);
    uint r = u + 0x7FFFu + ((u >> 16) & 1u);
    return ushort(r >> 16);
}

#if __METAL_VERSION__ >= 310
#define VIS_HAS_NATIVE_BF16 1
typedef bfloat  vis_bf16_t;
typedef bfloat2 vis_bf16_2;
inline float      vis_bf16_to_f32(vis_bf16_t v)             { return float(v); }
inline vis_bf16_t vis_f32_to_bf16(float v)                  { return bfloat(v); }
inline vis_bf16_2 vis_f32x2_to_bf16x2(float a, float b)     { return bfloat2(bfloat(a), bfloat(b)); }
#else
#define VIS_HAS_NATIVE_BF16 0
typedef ushort  vis_bf16_t;
typedef ushort2 vis_bf16_2;
inline float      vis_bf16_to_f32(vis_bf16_t v)             { return bf16_to_f32(v); }
inline vis_bf16_t vis_f32_to_bf16(float v)                  { return f32_to_bf16_rne(v); }
inline vis_bf16_2 vis_f32x2_to_bf16x2(float a, float b)     { return ushort2(f32_to_bf16_rne(a), f32_to_bf16_rne(b)); }
#endif

inline float gelu_tanh_f32(float v) {
    // Clamp inner to [-15, 15] so fast-math tanh never overflows exp(2*x) -> NaN!
    float inner = clamp(0.7978845608028654f * (v + 0.044715f * v * v * v), -15.0f, 15.0f);
    return 0.5f * v * (1.0f + tanh(inner));
}

inline float erff_approx(float x) {
    // Abramowitz & Stegun 7.1.26 (max error < 1.5e-7, exact for BF16/FP32!)
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    float r = 1.0f - poly * exp(-ax * ax);
    return (x >= 0.0f) ? r : -r;
}

inline float gelu_exact_f32(float v) {
    return 0.5f * v * (1.0f + erff_approx(v * 0.7071067811865475f));
}

// ============================================================================
// 1. Round 24: Native Apple M4 Hardware simdgroup_bfloat8x8 BF16 AMX GEMM!
//    ZERO weight conversion, ZERO activation conversion (pure bfloat4 copies!)
//    BM=64, BN=64, BK=32, 128 threads, sA[64][36]+sB[64][36]=9.2 KB (3 TGs/core!)
//    - ACT=0: FP32 output (+ optional bias + optional residual add)
//    - ACT=1: Fused bias + gelu_pytorch_tanh -> BF16 output (eliminates 27x [N,4304] DRAM round-trips!)
//    - ACT=2: Fused bias + exact erf GELU -> BF16 output (PatchMerger fc1!)
// ============================================================================

struct VisGemmParams {
    uint M;
    uint N;
    uint K;
    uint add_residual;
    uint has_bias;
};

#if VIS_HAS_NATIVE_BF16
template<int ACT, typename OutT>
inline void vis_bf16_gemm_native_impl(
    const device bfloat* W,
    const device bfloat* bias,
    const device bfloat* X,
    device OutT*         Y,
    constant VisGemmParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup bfloat (&sA)[64][36],
    threadgroup bfloat (&sB)[64][36]
) {
    uint m0 = tgid.y * 64u;
    uint n0 = tgid.x * 64u;
    if (m0 >= p.M || n0 >= p.N) return;

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0; in_ < 4u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m  = (sg >> 1) * 32u;
    uint sg_n  = (sg & 1u) * 32u;
    uint r_loc = tid >> 1;      // 0..63
    uint h_idx = tid & 1u;      // 0 or 1
    uint c_blk = h_idx * 16u;   // 0 or 16

    uint gm_ld = m0 + r_loc;
    uint gn_ld = n0 + r_loc;
    bool gm_ok = gm_ld < p.M;
    bool gn_ok = gn_ld < p.N;

    const device bfloat* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * p.K;
    const device bfloat* w_base = W + (ulong)(gn_ok ? gn_ld : 0u) * p.K;
    threadgroup bfloat4* sa_dst = (threadgroup bfloat4*)(&sA[r_loc][c_blk]);
    threadgroup bfloat4* sb_dst = (threadgroup bfloat4*)(&sB[r_loc][c_blk]);

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint k_rem = min(32u, p.K - k0);
        if (gm_ok && c_blk + 16u <= k_rem) {
            const device bfloat4* xp = (const device bfloat4*)(x_base + k0 + c_blk);
            sa_dst[0] = xp[0]; sa_dst[1] = xp[1]; sa_dst[2] = xp[2]; sa_dst[3] = xp[3];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; i++) {
                uint kc = c_blk + i;
                sA[r_loc][kc] = (gm_ok && kc < k_rem) ? x_base[k0 + kc] : bfloat(0.0f);
            }
        }

        if (gn_ok && c_blk + 16u <= k_rem) {
            const device bfloat4* wp = (const device bfloat4*)(w_base + k0 + c_blk);
            sb_dst[0] = wp[0]; sb_dst[1] = wp[1]; sb_dst[2] = wp[2]; sb_dst[3] = wp[3];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; i++) {
                uint kc = c_blk + i;
                sB[r_loc][kc] = (gn_ok && kc < k_rem) ? w_base[k0 + kc] : bfloat(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_bfloat8x8 a[4], b[4];
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
            device OutT* yr = Y + (ulong)gm * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0; in_ < 4u; in_++) {
                uint gn0 = n0 + sg_n + 8u * in_ + fn;
                float v0 = acc[im][in_].thread_elements()[0];
                float v1 = acc[im][in_].thread_elements()[1];
                if (p.has_bias) {
                    if (gn0 < p.N)      v0 += float(bias[gn0]);
                    if (gn0 + 1u < p.N) v1 += float(bias[gn0 + 1u]);
                }
                if (ACT == 1) {
                    v0 = gelu_tanh_f32(v0);
                    v1 = gelu_tanh_f32(v1);
                    if (gn0 + 1u < p.N) {
                        *((device bfloat2*)(yr + gn0)) = bfloat2(bfloat(v0), bfloat(v1));
                    } else if (gn0 < p.N) {
                        yr[gn0] = OutT(v0);
                    }
                } else if (ACT == 2) {
                    v0 = gelu_exact_f32(v0);
                    v1 = gelu_exact_f32(v1);
                    if (gn0 + 1u < p.N) {
                        *((device bfloat2*)(yr + gn0)) = bfloat2(bfloat(v0), bfloat(v1));
                    } else if (gn0 < p.N) {
                        yr[gn0] = OutT(v0);
                    }
                } else {
                    if (gn0 + 1u < p.N) {
                        device float2* yp = (device float2*)(yr + gn0);
                        float2 vv = float2(v0, v1);
                        *yp = p.add_residual ? (*yp + vv) : vv;
                    } else if (gn0 < p.N) {
                        yr[gn0] = p.add_residual ? (yr[gn0] + OutT(v0)) : OutT(v0);
                    }
                }
            }
        }
    }
}

kernel void vis_bf16_gemm_f16x(
    const device bfloat* W    [[buffer(0)]],
    const device bfloat* bias [[buffer(1)]],
    const device bfloat* X    [[buffer(2)]],
    device float*        Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup bfloat sA[64][36];
    threadgroup bfloat sB[64][36];
    vis_bf16_gemm_native_impl<0, float>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}

kernel void vis_bf16_gemm_gelu_tanh_f16x(
    const device bfloat* W    [[buffer(0)]],
    const device bfloat* bias [[buffer(1)]],
    const device bfloat* X    [[buffer(2)]],
    device bfloat*       Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup bfloat sA[64][36];
    threadgroup bfloat sB[64][36];
    vis_bf16_gemm_native_impl<1, bfloat>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}

kernel void vis_bf16_gemm_gelu_exact_f16x(
    const device bfloat* W    [[buffer(0)]],
    const device bfloat* bias [[buffer(1)]],
    const device bfloat* X    [[buffer(2)]],
    device bfloat*       Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup bfloat sA[64][36];
    threadgroup bfloat sB[64][36];
    vis_bf16_gemm_native_impl<2, bfloat>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}
#else
// Metal 3.0 fallback for M1/M2 (Apple7/Apple8): BF16 stored as ushort, AMX via simdgroup_half8x8
template<int ACT, typename OutT>
inline void vis_bf16_gemm_m30_impl(
    const device ushort* W,
    const device ushort* bias,
    const device ushort* X,
    device OutT*         Y,
    constant VisGemmParams& p,
    uint2 tgid, uint tid, uint lane, uint sg,
    threadgroup half (&sA)[64][36],
    threadgroup half (&sB)[64][36]
) {
    uint m0 = tgid.y * 64u;
    uint n0 = tgid.x * 64u;
    if (m0 >= p.M || n0 >= p.N) return;

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (uint im = 0; im < 4u; im++)
    #pragma clang loop unroll(full)
    for (uint in_ = 0; in_ < 4u; in_++) {
        acc[im][in_] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint sg_m  = (sg >> 1) * 32u;
    uint sg_n  = (sg & 1u) * 32u;
    uint r_loc = tid >> 1;
    uint h_idx = tid & 1u;
    uint c_blk = h_idx * 16u;

    uint gm_ld = m0 + r_loc;
    uint gn_ld = n0 + r_loc;
    bool gm_ok = gm_ld < p.M;
    bool gn_ok = gn_ld < p.N;

    const device ushort* x_base = X + (ulong)(gm_ok ? gm_ld : 0u) * p.K;
    const device ushort* w_base = W + (ulong)(gn_ok ? gn_ld : 0u) * p.K;

    for (uint k0 = 0; k0 < p.K; k0 += 32u) {
        uint k_rem = min(32u, p.K - k0);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 16u; i++) {
            uint kc = c_blk + i;
            sA[r_loc][kc] = (gm_ok && kc < k_rem) ? half(bf16_to_f32(x_base[k0 + kc])) : half(0.0h);
            sB[r_loc][kc] = (gn_ok && kc < k_rem) ? half(bf16_to_f32(w_base[k0 + kc])) : half(0.0h);
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
            device OutT* yr = Y + (ulong)gm * p.N;
            #pragma clang loop unroll(full)
            for (uint in_ = 0; in_ < 4u; in_++) {
                uint gn0 = n0 + sg_n + 8u * in_ + fn;
                float v0 = acc[im][in_].thread_elements()[0];
                float v1 = acc[im][in_].thread_elements()[1];
                if (p.has_bias) {
                    if (gn0 < p.N)      v0 += bf16_to_f32(bias[gn0]);
                    if (gn0 + 1u < p.N) v1 += bf16_to_f32(bias[gn0 + 1u]);
                }
                if (ACT == 1) {
                    v0 = gelu_tanh_f32(v0);
                    v1 = gelu_tanh_f32(v1);
                    if (gn0 + 1u < p.N) {
                        *((device ushort2*)(yr + gn0)) = ushort2(f32_to_bf16_rne(v0), f32_to_bf16_rne(v1));
                    } else if (gn0 < p.N) {
                        yr[gn0] = f32_to_bf16_rne(v0);
                    }
                } else if (ACT == 2) {
                    v0 = gelu_exact_f32(v0);
                    v1 = gelu_exact_f32(v1);
                    if (gn0 + 1u < p.N) {
                        *((device ushort2*)(yr + gn0)) = ushort2(f32_to_bf16_rne(v0), f32_to_bf16_rne(v1));
                    } else if (gn0 < p.N) {
                        yr[gn0] = f32_to_bf16_rne(v0);
                    }
                } else {
                    if (gn0 + 1u < p.N) {
                        device float2* yp = (device float2*)(yr + gn0);
                        float2 vv = float2(v0, v1);
                        *yp = p.add_residual ? (*yp + vv) : vv;
                    } else if (gn0 < p.N) {
                        yr[gn0] = p.add_residual ? (yr[gn0] + OutT(v0)) : OutT(v0);
                    }
                }
            }
        }
    }
}

kernel void vis_bf16_gemm_f16x(
    const device ushort* W    [[buffer(0)]],
    const device ushort* bias [[buffer(1)]],
    const device ushort* X    [[buffer(2)]],
    device float*        Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sA[64][36];
    threadgroup half sB[64][36];
    vis_bf16_gemm_m30_impl<0, float>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}

kernel void vis_bf16_gemm_gelu_tanh_f16x(
    const device ushort* W    [[buffer(0)]],
    const device ushort* bias [[buffer(1)]],
    const device ushort* X    [[buffer(2)]],
    device ushort*       Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sA[64][36];
    threadgroup half sB[64][36];
    vis_bf16_gemm_m30_impl<1, ushort>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}

kernel void vis_bf16_gemm_gelu_exact_f16x(
    const device ushort* W    [[buffer(0)]],
    const device ushort* bias [[buffer(1)]],
    const device ushort* X    [[buffer(2)]],
    device ushort*       Y    [[buffer(3)]],
    constant VisGemmParams& p [[buffer(4)]],
    uint2 tgid                [[threadgroup_position_in_grid]],
    uint  tid                 [[thread_index_in_threadgroup]],
    uint  lane                [[thread_index_in_simdgroup]],
    uint  sg                  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sA[64][36];
    threadgroup half sB[64][36];
    vis_bf16_gemm_m30_impl<2, ushort>(W, bias, X, Y, p, tgid, tid, lane, sg, sA, sB);
}
#endif

// ============================================================================
// 2. Round 25: Fused Bilinear 2D PosEmbed Add + Block-0 LayerNorm (FP32 -> BF16)
// ============================================================================

struct VisNormParams {
    uint  M;
    uint  D;
    float eps;
};

kernel void vis_add_pos_and_norm_f16(
    device float*            hid       [[buffer(0)]],
    const device vis_bf16_t* pos_embed [[buffer(1)]],
    const device int4*       pos_idx   [[buffer(2)]],
    const device float4*     pos_wt    [[buffer(3)]],
    const device vis_bf16_t* weight    [[buffer(4)]],
    const device vis_bf16_t* bias      [[buffer(5)]],
    device vis_bf16_t*       out       [[buffer(6)]],
    constant VisNormParams& p          [[buffer(7)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint tid                           [[thread_position_in_threadgroup]],
    uint tg_size                       [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    device float*      hp = hid + (ulong)row * p.D;
    device vis_bf16_t* op = out + (ulong)row * p.D;
    int4 idx = pos_idx[row];
    float4 wt = pos_wt[row];

    threadgroup float smem[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;

    if (p.D <= 1280u && tg_size == 256u) {
        float vals[5];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint d = k * 256u + tid;
            if (d < p.D) {
                float pe = vis_bf16_to_f32(pos_embed[(ulong)idx.x * p.D + d]) * wt.x
                         + vis_bf16_to_f32(pos_embed[(ulong)idx.y * p.D + d]) * wt.y
                         + vis_bf16_to_f32(pos_embed[(ulong)idx.z * p.D + d]) * wt.z
                         + vis_bf16_to_f32(pos_embed[(ulong)idx.w * p.D + d]) * wt.w;
                float v = hp[d] + pe;
                hp[d] = v;
                vals[k] = v;
                sum += v;
            }
        }
        sum = simd_sum(sum);
        if (lane == 0u) smem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = tot / float(p.D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = smem[0];

        float var = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint d = k * 256u + tid;
            if (d < p.D) {
                float diff = vals[k] - mean;
                var = fma(diff, diff, var);
            }
        }
        var = simd_sum(var);
        if (lane == 0u) smem[sg] = var;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_std = smem[0];

        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint d = k * 256u + tid;
            if (d < p.D) {
                op[d] = vis_f32_to_bf16((vals[k] - mean) * inv_std * vis_bf16_to_f32(weight[d]) + vis_bf16_to_f32(bias[d]));
            }
        }
    } else {
        float sum = 0.0f;
        for (uint d = tid; d < p.D; d += tg_size) {
            float pe = vis_bf16_to_f32(pos_embed[(ulong)idx.x * p.D + d]) * wt.x
                     + vis_bf16_to_f32(pos_embed[(ulong)idx.y * p.D + d]) * wt.y
                     + vis_bf16_to_f32(pos_embed[(ulong)idx.z * p.D + d]) * wt.z
                     + vis_bf16_to_f32(pos_embed[(ulong)idx.w * p.D + d]) * wt.w;
            float v = hp[d] + pe;
            hp[d] = v;
            sum += v;
        }
        sum = simd_sum(sum);
        if (lane == 0u) smem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = tot / float(p.D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = smem[0];

        float var = 0.0f;
        for (uint d = tid; d < p.D; d += tg_size) {
            float diff = hp[d] - mean;
            var = fma(diff, diff, var);
        }
        var = simd_sum(var);
        if (lane == 0u) smem[sg] = var;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_std = smem[0];

        for (uint d = tid; d < p.D; d += tg_size) {
            op[d] = vis_f32_to_bf16((hp[d] - mean) * inv_std * vis_bf16_to_f32(weight[d]) + vis_bf16_to_f32(bias[d]));
        }
    }
}

// ============================================================================
// 3. Round 25: Single-Pass Register-Cached LayerNorm (FP32 -> BF16)
// ============================================================================

kernel void vis_layernorm_f16(
    const device float*      x      [[buffer(0)]],
    const device vis_bf16_t* weight [[buffer(1)]],
    const device vis_bf16_t* bias   [[buffer(2)]],
    device vis_bf16_t*       out    [[buffer(3)]],
    constant VisNormParams& p       [[buffer(4)]],
    uint row                        [[threadgroup_position_in_grid]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xp = x + (ulong)row * p.D;
    device vis_bf16_t*  op = out + (ulong)row * p.D;

    threadgroup float smem[32];
    uint sg = tid >> 5, lane = tid & 31u, nsg = tg_size >> 5;

    if (p.D <= 1280u && tg_size == 256u) {
        float vals[5];
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint i = k * 256u + tid;
            if (i < p.D) {
                float v = xp[i];
                vals[k] = v;
                sum += v;
            }
        }
        sum = simd_sum(sum);
        if (lane == 0u) smem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = tot / float(p.D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = smem[0];

        float var = 0.0f;
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint i = k * 256u + tid;
            if (i < p.D) {
                float d = vals[k] - mean;
                var = fma(d, d, var);
            }
        }
        var = simd_sum(var);
        if (lane == 0u) smem[sg] = var;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_std = smem[0];

        #pragma clang loop unroll(full)
        for (uint k = 0; k < 5u; k++) {
            uint i = k * 256u + tid;
            if (i < p.D) {
                op[i] = vis_f32_to_bf16((vals[k] - mean) * inv_std * vis_bf16_to_f32(weight[i]) + vis_bf16_to_f32(bias[i]));
            }
        }
    } else {
        float sum = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) sum += xp[i];
        sum = simd_sum(sum);
        if (lane == 0u) smem[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = tot / float(p.D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean = smem[0];

        float var = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            float d = xp[i] - mean;
            var = fma(d, d, var);
        }
        var = simd_sum(var);
        if (lane == 0u) smem[sg] = var;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? smem[lane] : 0.0f);
            if (lane == 0u) smem[0] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_std = smem[0];

        for (uint i = tid; i < p.D; i += tg_size) {
            op[i] = vis_f32_to_bf16((xp[i] - mean) * inv_std * vis_bf16_to_f32(weight[i]) + vis_bf16_to_f32(bias[i]));
        }
    }
}

// ============================================================================
// 4. Round 25: 16x-Fewer-Trig Split QKV + 2D Vision RoPE -> Q, K, V FP16
// ============================================================================

struct VisAttnParams {
    uint  N;
    uint  H;
    uint  Dh;
    uint  seq_start;
    uint  seq_len;
    float scale;
};

kernel void vis_qkv_rope_split(
    const device float* qkv       [[buffer(0)]],
    const device float* rot_freqs [[buffer(1)]],
    device half*        q_out     [[buffer(2)]],
    device half*        k_out     [[buffer(3)]],
    device half*        v_out     [[buffer(4)]],
    constant VisAttnParams& p     [[buffer(5)]],
    uint t                        [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]]
) {
    uint half_d = p.Dh >> 1;
    if (t >= p.N || tid >= half_d) return;

    float freq = rot_freqs[(ulong)t * half_d + tid];
    float c = cos(freq);
    float s = sin(freq);

    ulong hd = (ulong)p.H * p.Dh;
    ulong qkv_base = (ulong)t * (3u * hd);

    // Round 25: Write Q, K, V in contiguous Head-Major [H, N, Dh] layout for 100% coalesced FlashAttention!
    for (uint h = 0; h < p.H; h++) {
        ulong h_off   = (ulong)h * p.Dh;
        ulong dst_off = ((ulong)h * p.N + t) * p.Dh;

        float q0 = qkv[qkv_base + h_off + tid];
        float q1 = qkv[qkv_base + h_off + half_d + tid];
        q_out[dst_off + tid]          = half(fma(q0, c, -q1 * s));
        q_out[dst_off + half_d + tid] = half(fma(q1, c,  q0 * s));

        float k0 = qkv[qkv_base + hd + h_off + tid];
        float k1 = qkv[qkv_base + hd + h_off + half_d + tid];
        k_out[dst_off + tid]          = half(fma(k0, c, -k1 * s));
        k_out[dst_off + half_d + tid] = half(fma(k1, c,  k0 * s));

        v_out[dst_off + tid]          = half(qkv[qkv_base + 2u * hd + h_off + tid]);
        v_out[dst_off + half_d + tid] = half(qkv[qkv_base + 2u * hd + h_off + half_d + tid]);
    }
}

// ============================================================================
// 5. Round 25: Exact-16KB Bank-Conflict-Free BQ=32, BK=32, D=72 Bidirectional AMX FA2!
//    sQ[32][76] + sKV[32][76] + sS[32][32] + sP[32][36] + sAlpha/sInvL = 16,384 B exact!
//    Outputs native BF16 directly for zero-conversion attn.proj bfloat8x8 GEMM!
// ============================================================================

kernel void vis_flash_attn_f16(
    const device half* q_in [[buffer(0)]], // [N, H, 72] FP16
    const device half* k_in [[buffer(1)]], // [N, H, 72] FP16
    const device half* v_in [[buffer(2)]], // [N, H, 72] FP16
    device vis_bf16_t* out  [[buffer(3)]], // [N, H, 72] BF16 -> feeds proj GEMM!
    constant VisAttnParams& p [[buffer(4)]],
    uint2 tgid              [[threadgroup_position_in_grid]],
    uint  tid               [[thread_index_in_threadgroup]],
    uint  lane              [[thread_index_in_simdgroup]],
    uint  sg                [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint D  = 72u;
    constexpr uint DS = 76u; // +4 padding eliminates shared memory bank conflicts!

    uint t0 = tgid.x * BQ;
    uint h  = tgid.y;
    if (h >= p.H || t0 >= p.seq_len) return;

    ulong out_stride = (ulong)p.H * D;
    ulong out_h_off  = (ulong)h * D;
    // Head-major contiguous pointers [H, N, 72] -> 100% cache-line & hardware prefetcher coalescing!
    const device half* q_head = q_in + ((ulong)h * p.N + p.seq_start) * D;
    const device half* k_head = k_in + ((ulong)h * p.N + p.seq_start) * D;
    const device half* v_head = v_in + ((ulong)h * p.N + p.seq_start) * D;

    threadgroup half  sQ[BQ][DS];  // 4,864 B
    threadgroup half  sKV[BK][DS]; // 4,864 B
    threadgroup float sS[BQ][BK];  // 4,096 B
    threadgroup half  sP[BQ][36];  // 2,304 B
    threadgroup float sAlpha[BQ];  //   128 B
    threadgroup float sInvL[BQ];   //   128 B -> Total 16,384 B EXACTLY (2 TGs/core!)

    uint r_ld = tid >> 2;          // 0..31
    uint c_ld = (tid & 3u) * 18u;  // 0, 18, 36, 54
    {
        uint lt = t0 + r_ld;
        threadgroup half2* dst2 = (threadgroup half2*)(&sQ[r_ld][c_ld]);
        if (lt < p.seq_len) {
            const device half2* qp2 = (const device half2*)(q_head + (ulong)lt * D + c_ld);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 9u; i++) dst2[i] = qp2[i];
        } else {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 9u; i++) dst2[i] = half2(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_m = sg * 8u;
    uint qid  = lane >> 2;
    uint fm   = (qid & 4u) + ((lane >> 1) & 3u);
    uint fn   = (qid & 2u) * 2u + (lane & 1u) * 2u;

    simdgroup_float8x8 O[9];
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 9u; j++) {
        O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float m_row[8], l_row[8];
    #pragma clang loop unroll(full)
    for (uint rr = 0; rr < 8u; rr++) { m_row[rr] = -INFINITY; l_row[rr] = 0.0f; }
    const float scale_log2 = p.scale * 1.4426950408889634f;

    for (uint kb = 0; kb < p.seq_len; kb += BK) {
        uint tile_k = min(BK, p.seq_len - kb);

        {
            threadgroup half2* dst2 = (threadgroup half2*)(&sKV[r_ld][c_ld]);
            if (r_ld < tile_k) {
                const device half2* kp2 = (const device half2*)(k_head + (ulong)(kb + r_ld) * D + c_ld);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 9u; i++) dst2[i] = kp2[i];
            } else {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 9u; i++) dst2[i] = half2(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sa0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 sa1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 sa2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 sa3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < D; kk += 8u) {
            simdgroup_half8x8 q0, q1, q2, q3, k0;
            simdgroup_load(q0, &sQ[0][kk],  DS);
            simdgroup_load(q1, &sQ[8][kk],  DS);
            simdgroup_load(q2, &sQ[16][kk], DS);
            simdgroup_load(q3, &sQ[24][kk], DS);
            simdgroup_load(k0, &sKV[sg * 8u][kk], DS, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sa0, q0, k0, sa0);
            simdgroup_multiply_accumulate(sa1, q1, k0, sa1);
            simdgroup_multiply_accumulate(sa2, q2, k0, sa2);
            simdgroup_multiply_accumulate(sa3, q3, k0, sa3);
        }

        uint sn = sg * 8u + fn;
        sS[fm][sn]            = sa0.thread_elements()[0];
        sS[fm][sn + 1u]       = sa0.thread_elements()[1];
        sS[8u + fm][sn]       = sa1.thread_elements()[0];
        sS[8u + fm][sn + 1u]  = sa1.thread_elements()[1];
        sS[16u + fm][sn]      = sa2.thread_elements()[0];
        sS[16u + fm][sn + 1u] = sa2.thread_elements()[1];
        sS[24u + fm][sn]      = sa3.thread_elements()[0];
        sS[24u + fm][sn + 1u] = sa3.thread_elements()[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool full_tile = (t0 + sg_m + 8u <= p.seq_len) && (tile_k == BK);
        if (full_tile) {
            #pragma clang loop unroll(full)
            for (uint rr = 0; rr < 8u; rr++) {
                uint r = sg_m + rr;
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
            for (uint rr = 0; rr < 8u; rr++) {
                uint r = sg_m + rr;
                bool valid = (t0 + r < p.seq_len) && (lane < tile_k);
                float score = valid ? (sS[r][lane] * scale_log2) : -INFINITY;
                float row_max = simd_max(score);
                float m_old = m_row[rr];
                float m_new = max(m_old, row_max);
                float alpha = (kb == 0u || m_old == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2(m_old - m_new);
                float p_val = valid ? exp2(score - m_new) : 0.0f;
                l_row[rr] = fma(l_row[rr], alpha, simd_sum(p_val));
                m_row[rr] = m_new;
                sP[r][lane] = half(p_val);
                if (lane == 0u) sAlpha[r] = alpha;
            }
        }

        {
            threadgroup half2* dst2 = (threadgroup half2*)(&sKV[r_ld][c_ld]);
            if (r_ld < tile_k) {
                const device half2* vp2 = (const device half2*)(v_head + (ulong)(kb + r_ld) * D + c_ld);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 9u; i++) dst2[i] = vp2[i];
            } else {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 9u; i++) dst2[i] = half2(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a_row = sAlpha[sg_m + fm];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 9u; j++) {
            O[j].thread_elements()[0] *= a_row;
            O[j].thread_elements()[1] *= a_row;
        }

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, &sP[sg_m][kk], 36);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 9u; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &sKV[kk][8u * j], DS);
                simdgroup_multiply_accumulate(O[j], pm, vm, O[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        #pragma clang loop unroll(full)
        for (uint rr = 0; rr < 8u; rr++) {
            sInvL[sg_m + rr] = (l_row[rr] > 0.0f) ? (1.0f / l_row[rr]) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lt = t0 + sg_m + fm;
    if (lt < p.seq_len) {
        float inv_l = sInvL[sg_m + fm];
        device vis_bf16_t* op = out + (ulong)(p.seq_start + lt) * out_stride + out_h_off;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 9u; j++) {
            uint c0 = 8u * j + fn;
            *((device vis_bf16_2*)(op + c0)) = vis_f32x2_to_bf16x2(O[j].thread_elements()[0] * inv_l,
                                                                   O[j].thread_elements()[1] * inv_l);
        }
    }
}
