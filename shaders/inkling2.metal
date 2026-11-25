#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline float sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Branchless bit-exact OCP FP4 E2M1 -> float (3 ALU ops, zero LUT loads!)
inline float fp4_e2m1_to_f32(uint c) {
    uint mag = c & 7u;
    uint body = (mag >= 2u) ? (0x3F000000u + (mag << 22)) : (mag * 0x3F000000u);
    return as_type<float>(((c & 8u) << 28) | body);
}

// Branchless bit-exact OCP FP8 E4M3FN -> float (zero ldexp() call!)
inline float fp8_e4m3_to_f32(uchar bits) {
    uint s = uint(bits & 0x80u) << 24;
    uint e = (bits >> 3) & 0x0Fu;
    uint m = bits & 0x07u;
    uint body = (e == 0u) ? as_type<uint>(float(m) * 0.001953125f)
                          : (((e + 120u) << 23) | (m << 20));
    return as_type<float>(s | body);
}

// ============================================================================
// 1. Embedding Lookup + RMSNorm (BF16 embed or Q4_AFFINE embed)
// ============================================================================

struct InkEmbedParams {
    uint  M;
    uint  D;
    float eps;
};

kernel void inkling_embed_bf16_norm(
    const device ushort* embed_w [[buffer(0)]],
    const device ushort* norm_w  [[buffer(1)]],
    const device int*    ids     [[buffer(2)]],
    device float*        out     [[buffer(3)]],
    constant InkEmbedParams& p   [[buffer(4)]],
    uint row                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_size                 [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    uint tok = uint(ids[row]);
    const device ushort* src = embed_w + (ulong)tok * p.D;
    device float* dst = out + (ulong)row * p.D;

    float sq = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = bf16_to_f32(src[i]);
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float s_sum[32];
    uint sg = tid >> 5, lane = tid & 31u;
    if (lane == 0u) s_sum[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size >> 5;
        float tot = (lane < nsg) ? s_sum[lane] : 0.0f;
        tot = simd_sum(tot);
        if (lane == 0u) s_sum[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = s_sum[0];

    for (uint i = tid; i < p.D; i += tg_size) {
        dst[i] = bf16_to_f32(src[i]) * inv_rms * bf16_to_f32(norm_w[i]);
    }
}

// ============================================================================
// 2. Block Attention Residual & Cross-Pass Message Read
//    closed:  [max_blocks, M, D] (slots 0..n_closed-1 valid)
//    partial: [M, D] (appended as slot n_closed, or unused if use_partial==0)
//    queries: [n_q, D] (f32)
//    out:     [n_q, M, D] (f32)
//    Each state s is unweighted-RMSNorm'd, dotted with query q, softmax'd over S,
//    then weighted sum of normalized states (attn_res_normalize_values=true).
// ============================================================================

struct AttnResParams {
    uint  M;
    uint  D;
    uint  n_closed;    // number of states in closed buffer
    uint  use_partial; // 1 = append partial as last state; 0 = only closed[0..n_closed-1]
    uint  n_q;         // 1, 3, or 4
    uint  closed_stride; // stride in floats between block slots (max_M * D)
    float eps;
    float scale;       // 1 / sqrt(D)
};

kernel void inkling_attn_res(
    const device float* closed  [[buffer(0)]],
    const device float* partial [[buffer(1)]],
    const device float* queries [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant AttnResParams& p   [[buffer(4)]],
    uint  row                   [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_position_in_threadgroup]],
    uint  tg_size               [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    uint n_states = p.n_closed + p.use_partial;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint nsg = tg_size >> 5;

    threadgroup float inv_rms_s[8];
    threadgroup float scratch[32];
    threadgroup float weights[4][8]; // [n_q][n_states]

    // 1) Compute inv_rms for each state s
    for (uint s = 0; s < n_states; s++) {
        const device float* sp = (s < p.n_closed)
            ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
            : (partial + (ulong)row * p.D);
        float sq = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            float v = sp[i];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) scratch[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = (lane < nsg) ? scratch[lane] : 0.0f;
            tot = simd_sum(tot);
            if (lane == 0u) inv_rms_s[s] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2) For each query qi, dot with normalized states and softmax
    for (uint qi = 0; qi < p.n_q; qi++) {
        const device float* qp = queries + (ulong)qi * p.D;
        for (uint s = 0; s < n_states; s++) {
            const device float* sp = (s < p.n_closed)
                ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
                : (partial + (ulong)row * p.D);
            float ir = inv_rms_s[s];
            float dot = 0.0f;
            for (uint i = tid; i < p.D; i += tg_size) {
                dot = fma(qp[i], sp[i] * ir, dot);
            }
            dot = simd_sum(dot);
            if (lane == 0u) scratch[sg] = dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u) {
                float tot = (lane < nsg) ? scratch[lane] : 0.0f;
                tot = simd_sum(tot);
                if (lane == 0u) weights[qi][s] = tot * p.scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            float m = -INFINITY;
            for (uint s = 0; s < n_states; s++) m = max(m, weights[qi][s]);
            float sum = 0.0f;
            for (uint s = 0; s < n_states; s++) {
                float e = exp(weights[qi][s] - m);
                weights[qi][s] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (uint s = 0; s < n_states; s++) weights[qi][s] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3) Output weighted sum of normalized states for each query qi
    for (uint qi = 0; qi < p.n_q; qi++) {
        device float* op = out + ((ulong)qi * p.M + row) * p.D;
        for (uint i = tid; i < p.D; i += tg_size) {
            float acc = 0.0f;
            for (uint s = 0; s < n_states; s++) {
                const device float* sp = (s < p.n_closed)
                    ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
                    : (partial + (ulong)row * p.D);
                acc = fma(weights[qi][s] * inv_rms_s[s], sp[i], acc);
            }
            op[i] = acc;
        }
    }
}

// Fused 1-query Block Attention Residual + Weighted RMSNorm (eliminates intermediate roundtrip + barrier!)
kernel void inkling_attn_res_norm1(
    const device float*  closed  [[buffer(0)]],
    const device float*  partial [[buffer(1)]],
    const device float*  query   [[buffer(2)]],
    const device ushort* norm_w  [[buffer(3)]],
    device float*        out     [[buffer(4)]],
    constant AttnResParams& p    [[buffer(5)]],
    uint  row                    [[threadgroup_position_in_grid]],
    uint  tid                    [[thread_position_in_threadgroup]],
    uint  tg_size                [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    uint n_states = p.n_closed + 1u;
    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint nsg = tg_size >> 5;

    threadgroup float inv_rms_s[8];
    threadgroup float scratch[32];
    threadgroup float weights[8];

    for (uint s = 0; s < n_states; s++) {
        const device float* sp = (s < p.n_closed)
            ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
            : (partial + (ulong)row * p.D);
        float sq = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            float v = sp[i];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) scratch[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? scratch[lane] : 0.0f);
            if (lane == 0u) inv_rms_s[s] = rsqrt(tot / float(p.D) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint s = 0; s < n_states; s++) {
        const device float* sp = (s < p.n_closed)
            ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
            : (partial + (ulong)row * p.D);
        float ir = inv_rms_s[s];
        float dot = 0.0f;
        for (uint i = tid; i < p.D; i += tg_size) {
            dot = fma(query[i], sp[i] * ir, dot);
        }
        dot = simd_sum(dot);
        if (lane == 0u) scratch[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float tot = simd_sum((lane < nsg) ? scratch[lane] : 0.0f);
            if (lane == 0u) weights[s] = tot * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float m = -INFINITY;
        for (uint s = 0; s < n_states; s++) m = max(m, weights[s]);
        float sum = 0.0f;
        for (uint s = 0; s < n_states; s++) {
            float e = exp(weights[s] - m);
            weights[s] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (uint s = 0; s < n_states; s++) weights[s] *= inv * inv_rms_s[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine + compute output RMS in registers (D=2304, tg_size=256 -> 9 floats/thread, zero DRAM roundtrip!)
    device float* op = out + (ulong)row * p.D;
    float local_v[16];
    float out_sq = 0.0f;
    uint idx_k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < n_states; s++) {
            const device float* sp = (s < p.n_closed)
                ? (closed + (ulong)s * p.closed_stride + (ulong)row * p.D)
                : (partial + (ulong)row * p.D);
            acc = fma(weights[s], sp[i], acc);
        }
        local_v[idx_k++] = acc;
        out_sq = fma(acc, acc, out_sq);
    }
    out_sq = simd_sum(out_sq);
    if (lane == 0u) scratch[sg] = out_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? scratch[lane] : 0.0f);
        if (lane == 0u) scratch[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_out = scratch[0];

    idx_k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        op[i] = local_v[idx_k++] * inv_out * bf16_to_f32(norm_w[i]);
    }
}

// ============================================================================
// 3. Fused Single-Kernel Loop Controller + Weighted RMSNorm!
//    Collapses ctrl_prep + down_proj(64x4608) + ctrl_apply + rmsnorm into ONE launch!
//    Uses 256 threads (8 simdgroups) per token row. Zero intermediate DRAM buffers!
// ============================================================================

struct CtrlParams {
    uint  M;
    uint  D;     // 2304
    uint  R;     // 64
    float eps;
};

kernel void inkling_controller_norm_fused(
    const device float*  x        [[buffer(0)]],
    const device float*  msg      [[buffer(1)]],
    const device ushort* down_w   [[buffer(2)]], // bf16 [64, 4608]
    const device ushort* forget_w [[buffer(3)]], // bf16 [2304, 64]
    const device ushort* gain_w   [[buffer(4)]], // bf16 [2304, 64]
    const device ushort* norm_w   [[buffer(5)]], // bf16 [2304]
    device float*        out      [[buffer(6)]], // f32 [M, 2304]
    constant CtrlParams& p        [[buffer(7)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xp = x + (ulong)row * p.D;
    const device float* mp = msg + (ulong)row * p.D;
    device float* op = out + (ulong)row * p.D;

    // 1) Stage normalized x and msg in threadgroup memory (half: 2304*2*2 = 9.2 KB!)
    threadgroup half s_nx[2304];
    threadgroup half s_nm[2304];
    threadgroup float s_state[64];
    threadgroup float s_red[32];

    uint sg = tid >> 5;
    uint lane = tid & 31u;
    uint nsg = tg_size >> 5; // 8 simdgroups

    float lx[12], lm[12];
    float sq_x = 0.0f, sq_m = 0.0f;
    uint cnt = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        float vx = xp[i], vm = mp[i];
        lx[cnt] = vx; lm[cnt] = vm; cnt++;
        sq_x = fma(vx, vx, sq_x);
        sq_m = fma(vm, vm, sq_m);
    }
    sq_x = simd_sum(sq_x);
    sq_m = simd_sum(sq_m);
    if (lane == 0u) { s_red[sg] = sq_x; s_red[16u + sg] = sq_m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tx = simd_sum((lane < nsg) ? s_red[lane] : 0.0f);
        float tm = simd_sum((lane < nsg) ? s_red[16u + lane] : 0.0f);
        if (lane == 0u) {
            s_red[0] = rsqrt(tx / float(p.D) + p.eps);
            s_red[1] = rsqrt(tm / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float irx = s_red[0], irm = s_red[1];

    cnt = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        s_nx[i] = half(lx[cnt] * irx);
        s_nm[i] = half(lm[cnt] * irm);
        cnt++;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) Compute state[64] = down_w[64, 4608] @ [s_nx, s_nm]: 8 simdgroups x 8 rows each
    uint twoD = 2u * p.D;
    for (uint r = sg; r < p.R; r += nsg) {
        const device ushort* wp = down_w + (ulong)r * twoD;
        float acc = 0.0f;
        for (uint k = lane * 4u; k < p.D; k += 128u) {
            acc = fma(bf16_to_f32(wp[k + 0u]), float(s_nx[k + 0u]), acc);
            acc = fma(bf16_to_f32(wp[k + 1u]), float(s_nx[k + 1u]), acc);
            acc = fma(bf16_to_f32(wp[k + 2u]), float(s_nx[k + 2u]), acc);
            acc = fma(bf16_to_f32(wp[k + 3u]), float(s_nx[k + 3u]), acc);
            const device ushort* wp2 = wp + p.D + k;
            acc = fma(bf16_to_f32(wp2[0]), float(s_nm[k + 0u]), acc);
            acc = fma(bf16_to_f32(wp2[1]), float(s_nm[k + 1u]), acc);
            acc = fma(bf16_to_f32(wp2[2]), float(s_nm[k + 2u]), acc);
            acc = fma(bf16_to_f32(wp2[3]), float(s_nm[k + 3u]), acc);
        }
        acc = simd_sum(acc);
        if (lane == 0u) s_state[r] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Compute gated output + RMSNorm in registers!
    float out_sq = 0.0f;
    cnt = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        const device ushort* fw = forget_w + (ulong)d * p.R;
        const device ushort* gw = gain_w + (ulong)d * p.R;
        float fa = 0.0f, ga = 0.0f;
        for (uint r = 0; r < p.R; r += 4u) {
            float st0 = s_state[r + 0u], st1 = s_state[r + 1u];
            float st2 = s_state[r + 2u], st3 = s_state[r + 3u];
            fa = fma(bf16_to_f32(fw[r + 0u]), st0, fma(bf16_to_f32(fw[r + 1u]), st1,
                 fma(bf16_to_f32(fw[r + 2u]), st2, fma(bf16_to_f32(fw[r + 3u]), st3, fa))));
            ga = fma(bf16_to_f32(gw[r + 0u]), st0, fma(bf16_to_f32(gw[r + 1u]), st1,
                 fma(bf16_to_f32(gw[r + 2u]), st2, fma(bf16_to_f32(gw[r + 3u]), st3, ga))));
        }
        float v = fma(1.0f - tanh(fa), lx[cnt], tanh(ga) * lm[cnt]);
        lx[cnt] = v;
        out_sq = fma(v, v, out_sq);
        cnt++;
    }
    out_sq = simd_sum(out_sq);
    if (lane == 0u) s_red[sg] = out_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum((lane < nsg) ? s_red[lane] : 0.0f);
        if (lane == 0u) s_red[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_out = s_red[0];

    cnt = 0u;
    for (uint d = tid; d < p.D; d += tg_size) {
        op[d] = lx[cnt++] * inv_out * bf16_to_f32(norm_w[d]);
    }
}

// Legacy split controller kernels kept for fallback/reference

kernel void inkling_controller_prep(
    const device float* x     [[buffer(0)]],
    const device float* msg   [[buffer(1)]],
    device float*       mixed [[buffer(2)]], // [M, 2*D]
    constant CtrlParams& p    [[buffer(3)]],
    uint row                  [[threadgroup_position_in_grid]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint tg_size              [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    const device float* xp = x + (ulong)row * p.D;
    const device float* mp = msg + (ulong)row * p.D;
    device float* outp = mixed + (ulong)row * (2u * p.D);

    float sq_x = 0.0f, sq_m = 0.0f;
    for (uint i = tid; i < p.D; i += tg_size) {
        float vx = xp[i], vm = mp[i];
        sq_x = fma(vx, vx, sq_x);
        sq_m = fma(vm, vm, sq_m);
    }
    sq_x = simd_sum(sq_x);
    sq_m = simd_sum(sq_m);

    threadgroup float sx[32], sm[32];
    uint sg = tid >> 5, lane = tid & 31u;
    if (lane == 0u) { sx[sg] = sq_x; sm[sg] = sq_m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size >> 5;
        float tx = simd_sum((lane < nsg) ? sx[lane] : 0.0f);
        float tm = simd_sum((lane < nsg) ? sm[lane] : 0.0f);
        if (lane == 0u) {
            sx[0] = rsqrt(tx / float(p.D) + p.eps);
            sm[0] = rsqrt(tm / float(p.D) + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float irx = sx[0], irm = sm[0];

    for (uint i = tid; i < p.D; i += tg_size) {
        outp[i]       = xp[i] * irx;
        outp[p.D + i] = mp[i] * irm;
    }
}

kernel void inkling_controller_apply(
    const device float*  x        [[buffer(0)]],
    const device float*  msg      [[buffer(1)]],
    const device float*  state    [[buffer(2)]], // [M, 64]
    const device ushort* forget_w [[buffer(3)]], // bf16 [D, 64]
    const device ushort* gain_w   [[buffer(4)]], // bf16 [D, 64]
    device float*        out      [[buffer(5)]], // [M, D]
    constant CtrlParams& p        [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint t = gid.y;
    if (d >= p.D || t >= p.M) return;

    const device float* st = state + (ulong)t * p.R;
    const device ushort* fw = forget_w + (ulong)d * p.R;
    const device ushort* gw = gain_w + (ulong)d * p.R;

    float f_acc = 0.0f;
    float g_acc = 0.0f;
    for (uint r = 0; r < p.R; r++) {
        float sv = st[r];
        f_acc = fma(bf16_to_f32(fw[r]), sv, f_acc);
        g_acc = fma(bf16_to_f32(gw[r]), sv, g_acc);
    }
    float fg = tanh(f_acc);
    float gn = tanh(g_acc);
    ulong idx = (ulong)t * p.D + d;
    out[idx] = fma(1.0f - fg, x[idx], gn * msg[idx]);
}

// ============================================================================
// 4. Short Causal Convolutions (Static & Dynamic, K=4) + Optional Residual Add
//    Static:  w is bf16 [C, 4] chronological (w[c,0]=t-3 .. w[c,3]=t), div = sqrt(5)
//    Dynamic: static_w is bf16 [4, C] lag-order (lag 0=t .. lag 3=t-3),
//             dyn_w is f32 [M, Hkv, 4] (lag 0..3), div = sqrt(9) = 3.0
// ============================================================================

struct SconvParams {
    uint  M;
    uint  C;
    uint  H;         // num heads (for dynamic conv)
    uint  has_add;   // 1 = add residual buffer to conv output
    float inv_div;   // 1/sqrt(5) for static, 1/3 for dynamic
    uint  save_tape; // 1 = write post-step [s0,s1,s2] per t into tape[t, 3, C]
};

kernel void inkling_sconv_static(
    const device float*  x          [[buffer(0)]],
    const device ushort* w          [[buffer(1)]],
    device float*        conv_state [[buffer(2)]], // [3, C]
    const device float*  add_buf    [[buffer(3)]],
    device float*        out        [[buffer(4)]],
    constant SconvParams& p         [[buffer(5)]],
    device float*        tape       [[buffer(6)]], // [M, 3, C] when save_tape != 0
    uint c                          [[thread_position_in_grid]]
) {
    if (c >= p.C) return;

    float w0 = bf16_to_f32(w[c * 4u + 0u]);
    float w1 = bf16_to_f32(w[c * 4u + 1u]);
    float w2 = bf16_to_f32(w[c * 4u + 2u]);
    float w3 = bf16_to_f32(w[c * 4u + 3u]);

    float s0 = conv_state[0u * p.C + c];
    float s1 = conv_state[1u * p.C + c];
    float s2 = conv_state[2u * p.C + c];

    for (uint t = 0; t < p.M; t++) {
        ulong idx = (ulong)t * p.C + c;
        float xv = x[idx];
        float conv = fma(s0, w0, fma(s1, w1, fma(s2, w2, xv * w3)));
        float res = (xv + conv) * p.inv_div;
        if (p.has_add) res += add_buf[idx];
        out[idx] = res;
        s0 = s1;
        s1 = s2;
        s2 = xv;
        if (p.save_tape) {
            ulong tb = (ulong)t * 3u * p.C + c;
            tape[tb + 0u * p.C] = s0;
            tape[tb + 1u * p.C] = s1;
            tape[tb + 2u * p.C] = s2;
        }
    }

    conv_state[0u * p.C + c] = s0;
    conv_state[1u * p.C + c] = s1;
    conv_state[2u * p.C + c] = s2;
}

kernel void inkling_sconv_dynamic(
    const device float*  x          [[buffer(0)]],
    const device ushort* static_w   [[buffer(1)]], // [4, C] (lag 0=current, 3=oldest)
    const device float*  dyn_w      [[buffer(2)]], // [M, H, 4] (lag 0..3)
    device float*        conv_state [[buffer(3)]], // [3, C] (0=t-3, 1=t-2, 2=t-1)
    device float*        out        [[buffer(4)]],
    constant SconvParams& p         [[buffer(5)]],
    device float*        tape       [[buffer(6)]], // [M, 3, C] when save_tape != 0
    uint c                          [[thread_position_in_grid]]
) {
    if (c >= p.C) return;

    uint head_dim = p.C / p.H;
    uint h = c / head_dim;

    float sw0 = bf16_to_f32(static_w[0u * p.C + c]); // lag 0 (current)
    float sw1 = bf16_to_f32(static_w[1u * p.C + c]); // lag 1 (s2)
    float sw2 = bf16_to_f32(static_w[2u * p.C + c]); // lag 2 (s1)
    float sw3 = bf16_to_f32(static_w[3u * p.C + c]); // lag 3 (s0)

    float s0 = conv_state[0u * p.C + c];
    float s1 = conv_state[1u * p.C + c];
    float s2 = conv_state[2u * p.C + c];

    for (uint t = 0; t < p.M; t++) {
        ulong idx = (ulong)t * p.C + c;
        float xv = x[idx];
        const device float* dw = dyn_w + ((ulong)t * p.H + h) * 4u;
        float conv = fma(xv, sw0 + dw[0],
                     fma(s2, sw1 + dw[1],
                     fma(s1, sw2 + dw[2],
                         s0 * (sw3 + dw[3]))));
        out[idx] = (xv + conv) * p.inv_div;
        s0 = s1;
        s1 = s2;
        s2 = xv;
        if (p.save_tape) {
            ulong tb = (ulong)t * 3u * p.C + c;
            tape[tb + 0u * p.C] = s0;
            tape[tb + 1u * p.C] = s1;
            tape[tb + 2u * p.C] = s2;
        }
    }

    conv_state[0u * p.C + c] = s0;
    conv_state[1u * p.C + c] = s1;
    conv_state[2u * p.C + c] = s2;
}

// ============================================================================
// 5. Head Prep (L2 Clip, Value Norm, Log Scale, KV Cache Write)
// ============================================================================

struct InkAttnParams {
    uint  M;
    uint  Hq;           // 24
    uint  Hkv;          // 8
    uint  D;            // 128
    uint  d_rel;        // 16
    uint  rel_extent;   // 1024
    uint  window;       // 1024 for sliding, 0 for global
    uint  pos;          // start pos of current batch
    uint  kv_len;       // total logical tokens seen after append (pos + M)
    uint  kv_cap;       // allocated capacity (1024 for sliding ring, max_seq for global)
    float qk_clip;      // 22.627417
    float rel_clip;     // 8.0
    float log_alpha;    // 0.1
    float log_floor;    // 8192.0
    float eps;          // 1e-6
    float scale;        // 1/sqrt(128)
    uint  num_splits;
};

kernel void inkling_q_rel_prep(
    const device float*  q_in      [[buffer(0)]],
    const device ushort* q_scale   [[buffer(1)]], // [Hq, D]
    const device float*  r_in      [[buffer(2)]],
    const device ushort* r_scale   [[buffer(3)]], // [Hq, d_rel]
    device float*        q_out     [[buffer(4)]],
    device float*        r_out     [[buffer(5)]],
    constant InkAttnParams& p      [[buffer(6)]],
    uint2 tgid                     [[threadgroup_position_in_grid]],
    uint  lane                     [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    float tau = 1.0f;
    if (p.window == 0u) {
        float ratio = max(float(p.pos + t + 1u) / p.log_floor, 1.0f);
        tau = 1.0f + p.log_alpha * log(ratio);
    }

    // Q: D = 128 -> 4 elements per lane
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device ushort* qsp = q_scale + (ulong)h * p.D;
    device float* qdst = q_out + q_off;

    float q0 = qp[lane]        * bf16_to_f32(qsp[lane]);
    float q1 = qp[lane + 32u]  * bf16_to_f32(qsp[lane + 32u]);
    float q2 = qp[lane + 64u]  * bf16_to_f32(qsp[lane + 64u]);
    float q3 = qp[lane + 96u]  * bf16_to_f32(qsp[lane + 96u]);
    float qsq = simd_sum(fma(q0, q0, fma(q1, q1, fma(q2, q2, q3 * q3))));
    float qmul = min(1.0f, p.qk_clip / (sqrt(qsq) + 1e-6f)) * tau;

    qdst[lane]        = q0 * qmul;
    qdst[lane + 32u]  = q1 * qmul;
    qdst[lane + 64u]  = q2 * qmul;
    qdst[lane + 96u]  = q3 * qmul;

    // Rel: d_rel = 16 -> lanes 0..15
    ulong r_off = ((ulong)t * p.Hq + h) * p.d_rel;
    float rv = (lane < p.d_rel)
        ? (r_in[r_off + lane] * bf16_to_f32(r_scale[(ulong)h * p.d_rel + lane]))
        : 0.0f;
    float rsq = simd_sum(rv * rv);
    if (lane < p.d_rel) {
        float rmul = min(1.0f, p.rel_clip / (sqrt(rsq) + 1e-6f)) * tau;
        r_out[r_off + lane] = rv * rmul;
    }
}

kernel void inkling_kv_prep_write(
    const device float* k_in    [[buffer(0)]],
    const device float* v_in    [[buffer(1)]],
    device half*        k_cache [[buffer(2)]], // [kv_cap, Hkv, D]
    device half*        v_cache [[buffer(3)]], // [kv_cap, Hkv, D]
    constant InkAttnParams& p   [[buffer(4)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    ulong src_off = ((ulong)t * p.Hkv + h) * p.D;
    const device float* kp = k_in + src_off;
    const device float* vp = v_in + src_off;

    uint seq_pos = p.pos + t;
    uint slot = (p.window > 0u) ? (seq_pos % p.kv_cap) : seq_pos;
    ulong dst_off = ((ulong)slot * p.Hkv + h) * p.D;
    device half* kdst = k_cache + dst_off;
    device half* vdst = v_cache + dst_off;

    // K: L2 clip (no learned scale)
    float k0 = kp[lane], k1 = kp[lane + 32u], k2 = kp[lane + 64u], k3 = kp[lane + 96u];
    float ksq = simd_sum(fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3))));
    float kmul = min(1.0f, p.qk_clip / (sqrt(ksq) + 1e-6f));
    kdst[lane]        = half(k0 * kmul);
    kdst[lane + 32u]  = half(k1 * kmul);
    kdst[lane + 64u]  = half(k2 * kmul);
    kdst[lane + 96u]  = half(k3 * kmul);

    // V: Unweighted RMSNorm (use_value_norm_no_gamma = true)
    float v0 = vp[lane], v1 = vp[lane + 32u], v2 = vp[lane + 64u], v3 = vp[lane + 96u];
    float vsq = simd_sum(fma(v0, v0, fma(v1, v1, fma(v2, v2, v3 * v3))));
    float vmul = rsqrt(vsq / float(p.D) + p.eps);
    vdst[lane]        = half(v0 * vmul);
    vdst[lane + 32u]  = half(v1 * vmul);
    vdst[lane + 64u]  = half(v2 * vmul);
    vdst[lane + 96u]  = half(v3 * vmul);
}

// ============================================================================
// 6. Relative Position Bias Profiles:
//    profiles[t, h, d] = sum_{r=0..15} rel[t, h, r] * rel_basis[r, d]
// ============================================================================

kernel void inkling_rel_profiles(
    const device float*  rel       [[buffer(0)]], // [M, Hq, 16]
    const device ushort* rel_basis [[buffer(1)]], // bf16 [16, 1024]
    device float*        profiles  [[buffer(2)]], // [M, Hq, 1024]
    constant InkAttnParams& p      [[buffer(3)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint d = gid.x; // 0..rel_extent-1
    uint h = gid.y; // 0..Hq-1
    uint t = gid.z; // 0..M-1
    if (d >= p.rel_extent || h >= p.Hq || t >= p.M) return;

    const device float* rp = rel + ((ulong)t * p.Hq + h) * p.d_rel;
    float acc = 0.0f;
    #pragma unroll
    for (uint r = 0; r < 16u; r++) {
        acc = fma(rp[r], bf16_to_f32(rel_basis[r * p.rel_extent + d]), acc);
    }
    profiles[((ulong)t * p.Hq + h) * p.rel_extent + d] = acc;
}

// ============================================================================
// 7. Flash GQA Attention with Relative Bias + Per-Token Sink + Sigmoid Output Gate
//    1 simdgroup (32 threads) per (h, t). D=128 -> 4 floats/lane in registers!
//    Sink trick: start online softmax at m_max = sink[t, h], l_sum = 1.0f, acc = 0!
// ============================================================================

kernel void inkling_rel_sink_attn_gated(
    const device float* q_in     [[buffer(0)]],
    const device float* profiles [[buffer(1)]], // [M, Hq, rel_extent]
    const device float* sink_in  [[buffer(2)]], // [M, Hq]
    const device float* gate_in  [[buffer(3)]], // [M, Hq, D]
    const device half*  k_cache  [[buffer(4)]], // [kv_cap, Hkv, D]
    const device half*  v_cache  [[buffer(5)]], // [kv_cap, Hkv, D]
    device float*       out      [[buffer(6)]], // [M, Hq, D]
    constant InkAttnParams& p    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    const device float* gp = gate_in + q_off;
    const device float* prof = profiles + ((ulong)t * p.Hq + h) * p.rel_extent;
    device float* op = out + q_off;

    float q0 = qp[lane], q1 = qp[lane + 32u], q2 = qp[lane + 64u], q3 = qp[lane + 96u];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    // Initialize online softmax with the query-dependent attention sink (value = 0)
    float m_max = sink_in[(ulong)t * p.Hq + h];
    float l_sum = 1.0f;

    uint q_pos = p.pos + t;
    uint k_start = (p.window > 0u && q_pos + 1u > p.window) ? (q_pos + 1u - p.window) : 0u;
    uint k_end   = q_pos + 1u;

    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    for (uint kpos = k_start; kpos < k_end; kpos++) {
        uint slot = (p.window > 0u) ? (kpos % p.kv_cap) : kpos;
        const device half* kp = k_cache + (ulong)slot * kv_stride + head_off;
        float dot = fma(q0, float(kp[lane]),
                    fma(q1, float(kp[lane + 32u]),
                    fma(q2, float(kp[lane + 64u]),
                        q3 * float(kp[lane + 96u]))));
        uint dist = q_pos - kpos;
        float bias = (dist < p.rel_extent) ? prof[dist] : 0.0f;
        float score = fma(simd_sum(dot), p.scale, bias);

        float m_new = max(m_max, score);
        float alpha = exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device half* vp = v_cache + (ulong)slot * kv_stride + head_off;
        a0 = fma(a0, alpha, beta * float(vp[lane]));
        a1 = fma(a1, alpha, beta * float(vp[lane + 32u]));
        a2 = fma(a2, alpha, beta * float(vp[lane + 64u]));
        a3 = fma(a3, alpha, beta * float(vp[lane + 96u]));
    }

    float inv_l = 1.0f / l_sum;
    op[lane]        = (a0 * inv_l) * sigmoid_f32(gp[lane]);
    op[lane + 32u]  = (a1 * inv_l) * sigmoid_f32(gp[lane + 32u]);
    op[lane + 64u]  = (a2 * inv_l) * sigmoid_f32(gp[lane + 64u]);
    op[lane + 96u]  = (a3 * inv_l) * sigmoid_f32(gp[lane + 96u]);
}

// ============================================================================
// 8. MoE Router + Top-8 Selection (FP32, entirely on GPU)
//    gate_w: bf16 [128, 2304], gate_bias: f32 [128]
//    top_idx: uint [M, 8], top_w: f32 [M, 8]
// ============================================================================

struct RouterParams {
    uint  M;
    uint  D;           // 2304
    uint  E;           // 128
    uint  K;           // 8
    float route_scale; // 8.0
};

kernel void inkling_moe_router_topk(
    const device float*  x         [[buffer(0)]],
    const device ushort* gate_w    [[buffer(1)]], // bf16 [128, D]
    const device float*  gate_bias [[buffer(2)]], // f32 [128]
    device uint*         top_idx   [[buffer(3)]], // [M, 8]
    device float*        top_w     [[buffer(4)]], // [M, 8]
    constant RouterParams& p       [[buffer(5)]],
    uint row                       [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    // Threadgroup = 128 threads (1 thread per expert, or 4 simdgroups each computing 32 experts)
    if (row >= p.M) return;

    threadgroup float scores[128];
    threadgroup float biased[128];

    if (tid < p.E) {
        const device float* xp = x + (ulong)row * p.D;
        const device ushort* wp = gate_w + (ulong)tid * p.D;
        float dot = 0.0f;
        for (uint i = 0; i < p.D; i++) {
            dot = fma(xp[i], bf16_to_f32(wp[i]), dot);
        }
        float sc = sigmoid_f32(dot);
        scores[tid] = sc;
        biased[tid] = sc + gate_bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float w_sum = 0.0f;
        uint chosen[8];
        float chosen_sc[8];
        for (uint k = 0; k < p.K; k++) {
            float best = -INFINITY;
            uint best_e = 0u;
            for (uint e = 0; e < p.E; e++) {
                float b = biased[e];
                if (b > best) { best = b; best_e = e; }
            }
            chosen[k] = best_e;
            chosen_sc[k] = scores[best_e];
            w_sum += chosen_sc[k];
            biased[best_e] = -INFINITY;
        }
        float inv = p.route_scale / (w_sum + 1e-20f);
        for (uint k = 0; k < p.K; k++) {
            top_idx[(ulong)row * p.K + k] = chosen[k];
            top_w[(ulong)row * p.K + k]   = chosen_sc[k] * inv;
        }
    }
}

// Fast top-8 selector given router_logits[M, 128] computed by parallel qmm_bf16_f32
kernel void inkling_router_topk_select(
    const device float* router_logits [[buffer(0)]],
    const device float* gate_bias     [[buffer(1)]],
    device uint*        top_idx       [[buffer(2)]],
    device float*       top_w         [[buffer(3)]],
    constant RouterParams& p          [[buffer(4)]],
    uint row                          [[threadgroup_position_in_grid]],
    uint tid                          [[thread_position_in_threadgroup]]
) {
    if (row >= p.M) return;
    threadgroup float scores[128];
    threadgroup float biased[128];
    if (tid < p.E) {
        float sc = sigmoid_f32(router_logits[(ulong)row * p.E + tid]);
        scores[tid] = sc;
        biased[tid] = sc + gate_bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float w_sum = 0.0f;
        uint chosen[8];
        float chosen_sc[8];
        for (uint k = 0; k < p.K; k++) {
            float best = -INFINITY;
            uint best_e = 0u;
            for (uint e = 0; e < p.E; e++) {
                float b = biased[e];
                if (b > best) { best = b; best_e = e; }
            }
            chosen[k] = best_e;
            chosen_sc[k] = scores[best_e];
            w_sum += chosen_sc[k];
            biased[best_e] = -INFINITY;
        }
        float inv = p.route_scale / (w_sum + 1e-20f);
        for (uint k = 0; k < p.K; k++) {
            top_idx[(ulong)row * p.K + k] = chosen[k];
            top_w[(ulong)row * p.K + k]   = chosen_sc[k] * inv;
        }
    }
}

// ============================================================================
// 9. NVFP4 Activation Rounding (exact match to Inkling native training format)
//    Given x[rows, width], computes rounded[rows, width] and outer[rows] = peak/1536
// ============================================================================

inline float ink_e4m3_round(float x) {
    float a = min(abs(x), 448.0f);
    float unit = ldexp(1.0f, max(-9, ilogb(max(a, 0.015625f)) - 3));
    return copysign(rint(a / unit) * unit, x);
}

inline float ink_e2m1_round(float x) {
    float a = abs(x);
    float y = (a > 5.0f) ? 6.0f : (a >= 3.5f) ? 4.0f : (a > 2.5f) ? 3.0f :
              (a >= 1.75f) ? 2.0f : (a > 1.25f) ? 1.5f : (a >= 0.75f) ? 1.0f :
              (a > 0.25f) ? 0.5f : 0.0f;
    return copysign(y, x);
}

struct ActQuantParams {
    uint rows;
    uint width;
};

kernel void inkling_nvfp4_quant_act(
    const device float* x       [[buffer(0)]],
    device float*       q_out   [[buffer(1)]],
    device float*       outer   [[buffer(2)]], // [rows]
    constant ActQuantParams& p  [[buffer(3)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    if (row >= p.rows) return;
    const device float* xr = x + (ulong)row * p.width;
    device float* qr = q_out + (ulong)row * p.width;

    // Find row peak
    float local_max = 0.0f;
    for (uint i = tid; i < p.width; i += tg_size) {
        local_max = max(local_max, abs(xr[i]));
    }
    local_max = simd_max(local_max);
    threadgroup float s_max[32];
    uint sg = tid >> 5, lane = tid & 31u;
    if (lane == 0u) s_max[sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size >> 5;
        float m = simd_max((lane < nsg) ? s_max[lane] : 0.0f);
        if (lane == 0u) {
            s_max[0] = m;
            outer[row] = m / 1536.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float peak = s_max[0];
    float out_sc = peak / 1536.0f;
    float encoder = (peak > 0.0f) ? (1536.0f / peak) : 0.0f;

    // Quantize in 16-element half-simdgroup blocks
    uint first = lane & 16u;
    for (uint i = tid; i < p.width; i += tg_size) {
        float v = xr[i];
        float top = abs(v);
        top = max(top, simd_shuffle_xor(top, 1));
        top = max(top, simd_shuffle_xor(top, 2));
        top = max(top, simd_shuffle_xor(top, 4));
        top = max(top, simd_shuffle_xor(top, 8));

        float s6 = ink_e4m3_round(top * (encoder / 6.0f));
        float s4 = ink_e4m3_round(top * (encoder / 4.0f));
        float a = (s6 == 0.0f) ? 0.0f : ink_e2m1_round(v / (s6 * out_sc)) * s6;
        float b = (s4 == 0.0f) ? 0.0f : ink_e2m1_round(v / (s4 * out_sc)) * s4;
        float ae = abs(a * out_sc - v);
        float be = abs(b * out_sc - v);
        ae += simd_shuffle_xor(ae, 1); be += simd_shuffle_xor(be, 1);
        ae += simd_shuffle_xor(ae, 2); be += simd_shuffle_xor(be, 2);
        ae += simd_shuffle_xor(ae, 4); be += simd_shuffle_xor(be, 4);
        ae += simd_shuffle_xor(ae, 8); be += simd_shuffle_xor(be, 8);
        qr[i] = (simd_shuffle(be, first) < simd_shuffle(ae, first)) ? b : a;
    }
}

// ============================================================================
// 10. NVFP4 MoE W13 + Fused Softcapped SwiGLU
//     w13_w:      U8 [128, 2*I=2304, D/2=1152] (interleaved gate=2j, up=2j+1!)
//     w13_scale:  U8 (F8_E4M3) [128, 2304, D/16=144]
//     w13_amax:   F32 [128]
//     w13_denom:  F32 [1]
//     Grid: (ceil(I / 8), M * K) where 1 simdgroup computes 2 pairs (gate+up) -> 4 sg/TG = 8 pairs
// ============================================================================

struct MoeNvfp4Params {
    uint  M;
    uint  D;        // 2304
    uint  I;        // 1152
    uint  K_top;    // 8
    float gate_cap; // 4.0
    float up_cap;   // 25.0
    uint  use_act_quant; // 1 = multiply by x_outer[row]; 0 = direct FP32 x
};

kernel void inkling_moe_nvfp4_w13_swiglu(
    const device uchar* w13_w     [[buffer(0)]],
    const device uchar* w13_scale [[buffer(1)]],
    const device float* w13_amax  [[buffer(2)]],
    const device float* w13_denom [[buffer(3)]],
    const device float* x_in      [[buffer(4)]], // [M, D] (quantized or raw f32)
    const device float* x_outer   [[buffer(5)]], // [M]
    const device uint*  top_idx   [[buffer(6)]], // [M, 8]
    device float*       mid_out   [[buffer(7)]], // [M, 8, I]
    constant MoeNvfp4Params& p    [[buffer(8)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]],
    uint  sg                      [[simdgroup_index_in_threadgroup]]
) {
    uint slot = tgid.y; // 0 .. M*K_top - 1
    if (slot >= p.M * p.K_top) return;
    uint row = slot / p.K_top;
    uint e   = top_idx[slot];

    // 1 simdgroup computes 4 intermediate units (j0..j0+3) = 8 rows of w13 -> 4 sg/TG = 16 units/TG
    uint j0 = (tgid.x * 4u + sg) * 4u;
    if (j0 >= p.I) return;
    uint tj = min(4u, p.I - j0);

    uint row_words = p.D >> 3;  // 288 uint32 words per row (1152 bytes / 4)
    uint row_grps  = p.D >> 4;  // 144 scale bytes per row
    const device uint* w13_u32 = (const device uint*)w13_w;
    ulong e_w_base = (ulong)e * (2u * p.I) * row_words;
    ulong e_s_base = (ulong)e * (2u * p.I) * row_grps;
    const device float* xp = x_in + (ulong)row * p.D;

    float g_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane processes 1 block of 16 weights (= 2 uint32 words, 1 FP8 scale) per iteration
    for (uint grp = lane; grp < row_grps; grp += 32u) {
        uint k0 = grp << 4;
        uint w0 = grp << 1;
        float4 x0 = *((const device float4*)(xp + k0));
        float4 x1 = *((const device float4*)(xp + k0 + 4u));
        float4 x2 = *((const device float4*)(xp + k0 + 8u));
        float4 x3 = *((const device float4*)(xp + k0 + 12u));

        for (uint jj = 0; jj < tj; jj++) {
            uint row_g = 2u * (j0 + jj);
            uint row_u = row_g + 1u;

            // Gate row
            {
                const device uint* wp = w13_u32 + e_w_base + (ulong)row_g * row_words + w0;
                uint q0 = wp[0], q1 = wp[1];
                float sc = fp8_e4m3_to_f32(w13_scale[e_s_base + (ulong)row_g * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp4_e2m1_to_f32(q0),        x0.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 4),   x0.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 8),   x0.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 12),  x0.w, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 16),  x1.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 20),  x1.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 24),  x1.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 28),  x1.w, d);
                d = fma(fp4_e2m1_to_f32(q1),        x2.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 4),   x2.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 8),   x2.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 12),  x2.w, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 16),  x3.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 20),  x3.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 24),  x3.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 28),  x3.w, d);
                g_acc[jj] = fma(d, sc, g_acc[jj]);
            }
            // Up row
            {
                const device uint* wp = w13_u32 + e_w_base + (ulong)row_u * row_words + w0;
                uint q0 = wp[0], q1 = wp[1];
                float sc = fp8_e4m3_to_f32(w13_scale[e_s_base + (ulong)row_u * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp4_e2m1_to_f32(q0),        x0.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 4),   x0.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 8),   x0.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 12),  x0.w, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 16),  x1.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 20),  x1.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 24),  x1.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 28),  x1.w, d);
                d = fma(fp4_e2m1_to_f32(q1),        x2.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 4),   x2.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 8),   x2.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 12),  x2.w, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 16),  x3.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 20),  x3.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 24),  x3.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 28),  x3.w, d);
                u_acc[jj] = fma(d, sc, u_acc[jj]);
            }
        }
    }

    float outer = (w13_amax[e] / w13_denom[0]) * (p.use_act_quant ? x_outer[row] : 1.0f);
    for (uint jj = 0; jj < tj; jj++) {
        float g = simd_sum(g_acc[jj]) * outer;
        float u = simd_sum(u_acc[jj]) * outer;
        if (lane == 0u) {
            float act = (p.gate_cap * tanh(g / p.gate_cap)) * sigmoid_f32(g)
                      * (p.up_cap   * tanh(u / p.up_cap));
            mid_out[(ulong)slot * p.I + (j0 + jj)] = act;
        }
    }
}

// ============================================================================
// 11. NVFP4 MoE W2 + Weighted Reduction across all K_top=8 experts (vectorized uint32 loads!)
// ============================================================================

kernel void inkling_moe_nvfp4_w2_combine(
    const device uchar* w2_w      [[buffer(0)]],
    const device uchar* w2_scale  [[buffer(1)]],
    const device float* w2_amax   [[buffer(2)]],
    const device float* w2_denom  [[buffer(3)]],
    const device float* mid_in    [[buffer(4)]], // [M, 8, I]
    const device float* mid_outer [[buffer(5)]], // [M, 8]
    const device uint*  top_idx   [[buffer(6)]], // [M, 8]
    const device float* top_w     [[buffer(7)]], // [M, 8]
    device float*       out       [[buffer(8)]], // [M, D]
    constant MoeNvfp4Params& p    [[buffer(9)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]],
    uint  sg                      [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= p.M) return;

    uint o0 = (tgid.x * 4u + sg) * 4u;
    if (o0 >= p.D) return;
    uint tn = min(4u, p.D - o0);

    uint row_words = p.I >> 3;  // 144 uint32 words per row
    uint row_grps  = p.I >> 4;  // 72 scale bytes per row
    const device uint* w2_u32 = (const device uint*)w2_w;
    float denom = w2_denom[0];

    float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = 0; k < p.K_top; k++) {
        uint slot = row * p.K_top + k;
        uint e    = top_idx[slot];
        float rw  = top_w[slot];
        float outer = (w2_amax[e] / denom) * (p.use_act_quant ? mid_outer[slot] : 1.0f);
        float eff_scale = rw * outer;

        ulong e_w_base = (ulong)e * p.D * row_words;
        ulong e_s_base = (ulong)e * p.D * row_grps;
        const device float* mp = mid_in + (ulong)slot * p.I;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint grp = lane; grp < row_grps; grp += 32u) {
            uint k0 = grp << 4;
            uint w0 = grp << 1;
            float4 m0 = *((const device float4*)(mp + k0));
            float4 m1 = *((const device float4*)(mp + k0 + 4u));
            float4 m2 = *((const device float4*)(mp + k0 + 8u));
            float4 m3 = *((const device float4*)(mp + k0 + 12u));

            for (uint rr = 0; rr < tn; rr++) {
                uint o = o0 + rr;
                const device uint* wp = w2_u32 + e_w_base + (ulong)o * row_words + w0;
                uint q0 = wp[0], q1 = wp[1];
                float sc = fp8_e4m3_to_f32(w2_scale[e_s_base + (ulong)o * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp4_e2m1_to_f32(q0),        m0.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 4),   m0.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 8),   m0.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 12),  m0.w, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 16),  m1.x, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 20),  m1.y, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 24),  m1.z, d);
                d = fma(fp4_e2m1_to_f32(q0 >> 28),  m1.w, d);
                d = fma(fp4_e2m1_to_f32(q1),        m2.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 4),   m2.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 8),   m2.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 12),  m2.w, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 16),  m3.x, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 20),  m3.y, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 24),  m3.z, d);
                d = fma(fp4_e2m1_to_f32(q1 >> 28),  m3.w, d);
                acc[rr] = fma(d, sc, acc[rr]);
            }
        }
        for (uint rr = 0; rr < tn; rr++) {
            total[rr] = fma(acc[rr], eff_scale, total[rr]);
        }
    }

    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(total[rr]);
        if (lane == 0u) {
            out[(ulong)row * p.D + (o0 + rr)] = v;
        }
    }
}

// ============================================================================
// 11b. MXFP8 MoE W13+SoftcapSwiGLU and W2+Combine (for layers 20..23 of official inkling-2-mini)
//      w13_w: F8_E4M3 [128, 2*I=2304, D=2304], w13_scale: U8 E8M0 [128, 2304, D/32=72]
//      w2_w:  F8_E4M3 [128, D=2304, I=1152],   w2_scale:  U8 E8M0 [128, 2304, I/32=36]
// ============================================================================

inline float e8m0_to_f32(uchar bits) {
    return (bits == 0u) ? ldexp(1.0f, -127) : as_type<float>(uint(bits) << 23);
}

kernel void inkling_mxfp8_quant_act(
    const device float* x      [[buffer(0)]],
    device float*       q_out  [[buffer(1)]],
    constant ActQuantParams& p [[buffer(2)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    if (row >= p.rows) return;
    const device float* xr = x + (ulong)row * p.width;
    device float* qr = q_out + (ulong)row * p.width;
    for (uint i = tid; i < p.width; i += tg_size) {
        float v = xr[i];
        float top = simd_max(abs(v));
        float sc = (top > 0.0f) ? exp2(ceil(log2(top / 448.0f))) : 1.0f;
        qr[i] = ink_e4m3_round(v / sc) * sc;
    }
}

kernel void inkling_moe_mxfp8_w13_swiglu(
    const device uchar* w13_w     [[buffer(0)]],
    const device uchar* w13_scale [[buffer(1)]],
    const device float* x_in      [[buffer(2)]],
    const device uint*  top_idx   [[buffer(3)]],
    device float*       mid_out   [[buffer(4)]],
    constant MoeNvfp4Params& p    [[buffer(5)]],
    uint2 tgid                    [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_simdgroup]],
    uint  sg                      [[simdgroup_index_in_threadgroup]]
) {
    uint slot = tgid.y;
    if (slot >= p.M * p.K_top) return;
    uint row = slot / p.K_top;
    uint e   = top_idx[slot];

    uint j0 = (tgid.x * 4u + sg) * 4u;
    if (j0 >= p.I) return;
    uint tj = min(4u, p.I - j0);

    uint row_words = p.D >> 2; // 576 uint32 words (4 fp8 bytes/word)
    uint row_grps  = p.D >> 5; // 72 E8M0 scale bytes (32 weights/block)
    const device uint* w13_u32 = (const device uint*)w13_w;
    ulong e_w_base = (ulong)e * (2u * p.I) * row_words;
    ulong e_s_base = (ulong)e * (2u * p.I) * row_grps;
    const device float* xp = x_in + (ulong)row * p.D;

    float g_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane reads 2 uint32 words = 8 fp8 weights (all inside one 32-weight group: grp = wi >> 3)
    for (uint wi = lane * 2u; wi < row_words; wi += 64u) {
        uint k0 = wi << 2;
        uint grp = wi >> 3;
        float4 x0 = *((const device float4*)(xp + k0));
        float4 x1 = *((const device float4*)(xp + k0 + 4u));

        for (uint jj = 0; jj < tj; jj++) {
            uint row_g = 2u * (j0 + jj);
            uint row_u = row_g + 1u;
            {
                const device uint* wp = w13_u32 + e_w_base + (ulong)row_g * row_words + wi;
                uint q0 = wp[0], q1 = wp[1];
                float sc = e8m0_to_f32(w13_scale[e_s_base + (ulong)row_g * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp8_e4m3_to_f32(uchar(q0)),        x0.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 8)),   x0.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 16)),  x0.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 24)),  x0.w, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1)),        x1.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 8)),   x1.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 16)),  x1.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 24)),  x1.w, d);
                g_acc[jj] = fma(d, sc, g_acc[jj]);
            }
            {
                const device uint* wp = w13_u32 + e_w_base + (ulong)row_u * row_words + wi;
                uint q0 = wp[0], q1 = wp[1];
                float sc = e8m0_to_f32(w13_scale[e_s_base + (ulong)row_u * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp8_e4m3_to_f32(uchar(q0)),        x0.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 8)),   x0.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 16)),  x0.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 24)),  x0.w, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1)),        x1.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 8)),   x1.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 16)),  x1.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 24)),  x1.w, d);
                u_acc[jj] = fma(d, sc, u_acc[jj]);
            }
        }
    }

    for (uint jj = 0; jj < tj; jj++) {
        float g = simd_sum(g_acc[jj]);
        float u = simd_sum(u_acc[jj]);
        if (lane == 0u) {
            float act = (p.gate_cap * tanh(g / p.gate_cap)) * sigmoid_f32(g)
                      * (p.up_cap   * tanh(u / p.up_cap));
            mid_out[(ulong)slot * p.I + (j0 + jj)] = act;
        }
    }
}

kernel void inkling_moe_mxfp8_w2_combine(
    const device uchar* w2_w     [[buffer(0)]],
    const device uchar* w2_scale [[buffer(1)]],
    const device float* mid_in   [[buffer(2)]],
    const device uint*  top_idx  [[buffer(3)]],
    const device float* top_w    [[buffer(4)]],
    device float*       out      [[buffer(5)]],
    constant MoeNvfp4Params& p   [[buffer(6)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]],
    uint  sg                     [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= p.M) return;

    uint o0 = (tgid.x * 4u + sg) * 4u;
    if (o0 >= p.D) return;
    uint tn = min(4u, p.D - o0);

    uint row_words = p.I >> 2; // 288 uint32 words
    uint row_grps  = p.I >> 5; // 36 scale bytes
    const device uint* w2_u32 = (const device uint*)w2_w;

    float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < p.K_top; k++) {
        uint slot = row * p.K_top + k;
        uint e    = top_idx[slot];
        float rw  = top_w[slot];

        ulong e_w_base = (ulong)e * p.D * row_words;
        ulong e_s_base = (ulong)e * p.D * row_grps;
        const device float* mp = mid_in + (ulong)slot * p.I;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint wi = lane * 2u; wi < row_words; wi += 64u) {
            uint k0 = wi << 2;
            uint grp = wi >> 3;
            float4 m0 = *((const device float4*)(mp + k0));
            float4 m1 = *((const device float4*)(mp + k0 + 4u));

            for (uint rr = 0; rr < tn; rr++) {
                uint o = o0 + rr;
                const device uint* wp = w2_u32 + e_w_base + (ulong)o * row_words + wi;
                uint q0 = wp[0], q1 = wp[1];
                float sc = e8m0_to_f32(w2_scale[e_s_base + (ulong)o * row_grps + grp]);
                float d = 0.0f;
                d = fma(fp8_e4m3_to_f32(uchar(q0)),        m0.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 8)),   m0.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 16)),  m0.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q0 >> 24)),  m0.w, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1)),        m1.x, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 8)),   m1.y, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 16)),  m1.z, d);
                d = fma(fp8_e4m3_to_f32(uchar(q1 >> 24)),  m1.w, d);
                acc[rr] = fma(d, sc, acc[rr]);
            }
        }
        for (uint rr = 0; rr < tn; rr++) total[rr] = fma(acc[rr], rw, total[rr]);
    }

    for (uint rr = 0; rr < tn; rr++) {
        float v = simd_sum(total[rr]);
        if (lane == 0u) out[(ulong)row * p.D + (o0 + rr)] = v;
    }
}

// ============================================================================
// 12. Final Logit Softcap: cap * tanh(logits / cap)
// ============================================================================

struct SoftcapParams {
    uint  N;
    float cap;
};

kernel void inkling_logit_softcap(
    device float*            logits [[buffer(0)]],
    constant SoftcapParams&  p      [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= p.N) return;
    logits[gid] = p.cap * tanh(logits[gid] / p.cap);
}

// ============================================================================
// 13. Expert-Grouped MoE Prefill (M >= 4): read each expert's weights ONCE
//     across up to 8 tokens assigned to that expert -> 4-6x less DRAM traffic!
// ============================================================================

struct MoeGroupParams {
    uint total_slots; // M * K_top
    uint E;           // 128
};

// 1 threadgroup of 128 threads: deterministic, atomic-free expert bucket sort
kernel void inkling_moe_group_sort(
    const device uint* top_idx      [[buffer(0)]],
    device uint*       exp_count    [[buffer(1)]], // [128]
    device uint*       exp_offset   [[buffer(2)]], // [128]
    device uint*       sorted_slots [[buffer(3)]], // [M * 8]
    constant MoeGroupParams& p      [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]]
) {
    threadgroup uint s_cnt[128];
    threadgroup uint s_off[128];

    uint cnt = 0u;
    if (tid < p.E) {
        for (uint s = 0; s < p.total_slots; s++) {
            if (top_idx[s] == tid) cnt++;
        }
        s_cnt[tid] = cnt;
        exp_count[tid] = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint running = 0u;
        for (uint e = 0; e < p.E; e++) {
            s_off[e] = running;
            exp_offset[e] = running;
            running += s_cnt[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < p.E && cnt > 0u) {
        uint pos = s_off[tid];
        for (uint s = 0; s < p.total_slots; s++) {
            if (top_idx[s] == tid) {
                sorted_slots[pos++] = s;
            }
        }
    }
}

// ============================================================================
// Hardware simdgroup_half8x8 AMX Tiled Expert-Grouped MoE Prefill!
// BM=8 tokens/expert, BN=32 outputs, BK=32 (128 threads = 4 simdgroups, 3.9 KB TG mem!)
// Grid: (ceil(2*I/32), E, 1) for w13, (ceil(D/32), E, 1) for w2; TG: (128, 1, 1)
// ============================================================================

kernel void inkling_moe_nvfp4_w13_grouped(
    const device uchar* w13_w        [[buffer(0)]],
    const device uchar* w13_scale    [[buffer(1)]],
    const device float* w13_amax     [[buffer(2)]],
    const device float* w13_denom    [[buffer(3)]],
    const device float* x_in         [[buffer(4)]],
    const device float* x_outer      [[buffer(5)]],
    const device uint*  exp_count    [[buffer(6)]],
    const device uint*  exp_offset   [[buffer(7)]],
    const device uint*  sorted_slots [[buffer(8)]],
    device float*       mid_out      [[buffer(9)]],
    constant MoeNvfp4Params& p       [[buffer(10)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg                         [[simdgroup_index_in_threadgroup]]
) {
    uint e = tgid.y;
    uint cnt = exp_count[e];
    if (cnt == 0u) return;

    uint n0 = tgid.x * 32u; // interleaved gate/up rows [n0 .. n0+31]
    uint twoI = 2u * p.I;
    if (n0 >= twoI) return;

    uint start = exp_offset[e];
    uint row_words = p.D >> 3;
    uint row_grps  = p.D >> 4;
    const device uint* w13_u32 = (const device uint*)w13_w;
    ulong e_w_base = (ulong)e * twoI * row_words;
    ulong e_s_base = (ulong)e * twoI * row_grps;
    float w_outer = w13_amax[e] / w13_denom[0];

    threadgroup half  sA[8][36];
    threadgroup half  sB[32][36];
    threadgroup float sOut[8][32];
    threadgroup uint  s_slots[8];
    threadgroup uint  s_rows[8];

    uint r_loc = tid >> 2;        // 0..31 (n_loc)
    uint c_blk = (tid & 3u) * 8u; // 0, 8, 16, 24 (k_loc)
    uint ab    = tid >> 4;        // 0..7 (token b)
    uint ak    = (tid & 15u) * 2u;// 0..30 (k_loc for sA)

    for (uint base = 0; base < cnt; base += 8u) {
        uint tb = min(8u, cnt - base);
        if (tid < tb) {
            uint sl = sorted_slots[start + base + tid];
            s_slots[tid] = sl;
            s_rows[tid]  = sl / p.K_top;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0; k0 < p.D; k0 += 32u) {
            if (ab < tb) {
                const device float* xp = x_in + (ulong)s_rows[ab] * p.D + k0 + ak;
                sA[ab][ak + 0u] = half(xp[0]);
                sA[ab][ak + 1u] = half(xp[1]);
            } else {
                sA[ab][ak + 0u] = half(0.0h);
                sA[ab][ak + 1u] = half(0.0h);
            }

            uint gn = n0 + r_loc;
            if (gn < twoI) {
                uint wi  = (k0 >> 3) + (tid & 3u);
                uint grp = (k0 + c_blk) >> 4;
                uint q   = w13_u32[e_w_base + (ulong)gn * row_words + wi];
                float sc = fp8_e4m3_to_f32(w13_scale[e_s_base + (ulong)gn * row_grps + grp]);
                sB[c_blk + 0u][r_loc] = half(fp4_e2m1_to_f32(q)        * sc);
                sB[c_blk + 1u][r_loc] = half(fp4_e2m1_to_f32(q >> 4)   * sc);
                sB[c_blk + 2u][r_loc] = half(fp4_e2m1_to_f32(q >> 8)   * sc);
                sB[c_blk + 3u][r_loc] = half(fp4_e2m1_to_f32(q >> 12)  * sc);
                sB[c_blk + 4u][r_loc] = half(fp4_e2m1_to_f32(q >> 16)  * sc);
                sB[c_blk + 5u][r_loc] = half(fp4_e2m1_to_f32(q >> 20)  * sc);
                sB[c_blk + 6u][r_loc] = half(fp4_e2m1_to_f32(q >> 24)  * sc);
                sB[c_blk + 7u][r_loc] = half(fp4_e2m1_to_f32(q >> 28)  * sc);
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

        // 128 threads = 8 tokens x 16 intermediate gate/up pairs!
        uint j_loc = tid & 15u;
        uint j_glob = (n0 >> 1) + j_loc;
        if (ab < tb && j_glob < p.I) {
            float outer = w_outer * (p.use_act_quant ? x_outer[s_rows[ab]] : 1.0f);
            float g = sOut[ab][2u * j_loc + 0u] * outer;
            float u = sOut[ab][2u * j_loc + 1u] * outer;
            mid_out[(ulong)s_slots[ab] * p.I + j_glob] =
                (p.gate_cap * tanh(g / p.gate_cap)) * sigmoid_f32(g) * (p.up_cap * tanh(u / p.up_cap));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void inkling_moe_nvfp4_w2_grouped(
    const device uchar* w2_w         [[buffer(0)]],
    const device uchar* w2_scale     [[buffer(1)]],
    const device float* w2_amax      [[buffer(2)]],
    const device float* w2_denom     [[buffer(3)]],
    const device float* mid_in       [[buffer(4)]],
    const device float* mid_outer    [[buffer(5)]],
    const device float* top_w        [[buffer(6)]],
    const device uint*  exp_count    [[buffer(7)]],
    const device uint*  exp_offset   [[buffer(8)]],
    const device uint*  sorted_slots [[buffer(9)]],
    device float*       slot_out     [[buffer(10)]],
    constant MoeNvfp4Params& p       [[buffer(11)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg                         [[simdgroup_index_in_threadgroup]]
) {
    uint e = tgid.y;
    uint cnt = exp_count[e];
    if (cnt == 0u) return;

    uint n0 = tgid.x * 32u;
    if (n0 >= p.D) return;

    uint start = exp_offset[e];
    uint row_words = p.I >> 3;
    uint row_grps  = p.I >> 4;
    const device uint* w2_u32 = (const device uint*)w2_w;
    ulong e_w_base = (ulong)e * p.D * row_words;
    ulong e_s_base = (ulong)e * p.D * row_grps;
    float w_outer = w2_amax[e] / w2_denom[0];

    threadgroup half  sA[8][36];
    threadgroup half  sB[32][36];
    threadgroup float sOut[8][32];
    threadgroup uint  s_slots[8];

    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;
    uint ab    = tid >> 4;
    uint ak    = (tid & 15u) * 2u;

    for (uint base = 0; base < cnt; base += 8u) {
        uint tb = min(8u, cnt - base);
        if (tid < tb) s_slots[tid] = sorted_slots[start + base + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0; k0 < p.I; k0 += 32u) {
            if (ab < tb) {
                const device float* mp = mid_in + (ulong)s_slots[ab] * p.I + k0 + ak;
                sA[ab][ak + 0u] = half(mp[0]);
                sA[ab][ak + 1u] = half(mp[1]);
            } else {
                sA[ab][ak + 0u] = half(0.0h);
                sA[ab][ak + 1u] = half(0.0h);
            }

            uint gn = n0 + r_loc;
            if (gn < p.D) {
                uint wi  = (k0 >> 3) + (tid & 3u);
                uint grp = (k0 + c_blk) >> 4;
                uint q   = w2_u32[e_w_base + (ulong)gn * row_words + wi];
                float sc = fp8_e4m3_to_f32(w2_scale[e_s_base + (ulong)gn * row_grps + grp]);
                sB[c_blk + 0u][r_loc] = half(fp4_e2m1_to_f32(q)        * sc);
                sB[c_blk + 1u][r_loc] = half(fp4_e2m1_to_f32(q >> 4)   * sc);
                sB[c_blk + 2u][r_loc] = half(fp4_e2m1_to_f32(q >> 8)   * sc);
                sB[c_blk + 3u][r_loc] = half(fp4_e2m1_to_f32(q >> 12)  * sc);
                sB[c_blk + 4u][r_loc] = half(fp4_e2m1_to_f32(q >> 16)  * sc);
                sB[c_blk + 5u][r_loc] = half(fp4_e2m1_to_f32(q >> 20)  * sc);
                sB[c_blk + 6u][r_loc] = half(fp4_e2m1_to_f32(q >> 24)  * sc);
                sB[c_blk + 7u][r_loc] = half(fp4_e2m1_to_f32(q >> 28)  * sc);
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

        if (ab < tb) {
            uint sl = s_slots[ab];
            float eff = top_w[sl] * w_outer * (p.use_act_quant ? mid_outer[sl] : 1.0f);
            if (n0 + ak + 0u < p.D) slot_out[(ulong)sl * p.D + n0 + ak + 0u] = sOut[ab][ak + 0u] * eff;
            if (n0 + ak + 1u < p.D) slot_out[(ulong)sl * p.D + n0 + ak + 1u] = sOut[ab][ak + 1u] * eff;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void inkling_moe_mxfp8_w13_grouped(
    const device uchar* w13_w        [[buffer(0)]],
    const device uchar* w13_scale    [[buffer(1)]],
    const device float* x_in         [[buffer(2)]],
    const device uint*  exp_count    [[buffer(3)]],
    const device uint*  exp_offset   [[buffer(4)]],
    const device uint*  sorted_slots [[buffer(5)]],
    device float*       mid_out      [[buffer(6)]],
    constant MoeNvfp4Params& p       [[buffer(7)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg                         [[simdgroup_index_in_threadgroup]]
) {
    uint e = tgid.y;
    uint cnt = exp_count[e];
    if (cnt == 0u) return;

    uint n0 = tgid.x * 32u;
    uint twoI = 2u * p.I;
    if (n0 >= twoI) return;

    uint start = exp_offset[e];
    uint row_words = p.D >> 2;
    uint row_grps  = p.D >> 5;
    const device uint* w13_u32 = (const device uint*)w13_w;
    ulong e_w_base = (ulong)e * twoI * row_words;
    ulong e_s_base = (ulong)e * twoI * row_grps;

    threadgroup half  sA[8][36];
    threadgroup half  sB[32][36];
    threadgroup float sOut[8][32];
    threadgroup uint  s_slots[8];
    threadgroup uint  s_rows[8];

    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;
    uint ab    = tid >> 4;
    uint ak    = (tid & 15u) * 2u;

    for (uint base = 0; base < cnt; base += 8u) {
        uint tb = min(8u, cnt - base);
        if (tid < tb) {
            uint sl = sorted_slots[start + base + tid];
            s_slots[tid] = sl;
            s_rows[tid]  = sl / p.K_top;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0; k0 < p.D; k0 += 32u) {
            if (ab < tb) {
                const device float* xp = x_in + (ulong)s_rows[ab] * p.D + k0 + ak;
                sA[ab][ak + 0u] = half(xp[0]);
                sA[ab][ak + 1u] = half(xp[1]);
            } else {
                sA[ab][ak + 0u] = half(0.0h);
                sA[ab][ak + 1u] = half(0.0h);
            }

            uint gn = n0 + r_loc;
            if (gn < twoI) {
                uint wi = (k0 >> 2) + (tid & 3u) * 2u;
                uint grp = k0 >> 5;
                const device uint* wp = w13_u32 + e_w_base + (ulong)gn * row_words + wi;
                uint q0 = wp[0], q1 = wp[1];
                float sc = e8m0_to_f32(w13_scale[e_s_base + (ulong)gn * row_grps + grp]);
                sB[c_blk + 0u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0))        * sc);
                sB[c_blk + 1u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 8))   * sc);
                sB[c_blk + 2u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 16))  * sc);
                sB[c_blk + 3u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 24))  * sc);
                sB[c_blk + 4u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1))        * sc);
                sB[c_blk + 5u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 8))   * sc);
                sB[c_blk + 6u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 16))  * sc);
                sB[c_blk + 7u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 24))  * sc);
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

        uint j_loc = tid & 15u;
        uint j_glob = (n0 >> 1) + j_loc;
        if (ab < tb && j_glob < p.I) {
            float g = sOut[ab][2u * j_loc + 0u];
            float u = sOut[ab][2u * j_loc + 1u];
            mid_out[(ulong)s_slots[ab] * p.I + j_glob] =
                (p.gate_cap * tanh(g / p.gate_cap)) * sigmoid_f32(g) * (p.up_cap * tanh(u / p.up_cap));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void inkling_moe_mxfp8_w2_grouped(
    const device uchar* w2_w         [[buffer(0)]],
    const device uchar* w2_scale     [[buffer(1)]],
    const device float* mid_in       [[buffer(2)]],
    const device float* top_w        [[buffer(3)]],
    const device uint*  exp_count    [[buffer(4)]],
    const device uint*  exp_offset   [[buffer(5)]],
    const device uint*  sorted_slots [[buffer(6)]],
    device float*       slot_out     [[buffer(7)]],
    constant MoeNvfp4Params& p       [[buffer(8)]],
    uint2 tgid                       [[threadgroup_position_in_grid]],
    uint  tid                        [[thread_index_in_threadgroup]],
    uint  sg                         [[simdgroup_index_in_threadgroup]]
) {
    uint e = tgid.y;
    uint cnt = exp_count[e];
    if (cnt == 0u) return;

    uint n0 = tgid.x * 32u;
    if (n0 >= p.D) return;

    uint start = exp_offset[e];
    uint row_words = p.I >> 2;
    uint row_grps  = p.I >> 5;
    const device uint* w2_u32 = (const device uint*)w2_w;
    ulong e_w_base = (ulong)e * p.D * row_words;
    ulong e_s_base = (ulong)e * p.D * row_grps;

    threadgroup half  sA[8][36];
    threadgroup half  sB[32][36];
    threadgroup float sOut[8][32];
    threadgroup uint  s_slots[8];

    uint r_loc = tid >> 2;
    uint c_blk = (tid & 3u) * 8u;
    uint ab    = tid >> 4;
    uint ak    = (tid & 15u) * 2u;

    for (uint base = 0; base < cnt; base += 8u) {
        uint tb = min(8u, cnt - base);
        if (tid < tb) s_slots[tid] = sorted_slots[start + base + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

        for (uint k0 = 0; k0 < p.I; k0 += 32u) {
            if (ab < tb) {
                const device float* mp = mid_in + (ulong)s_slots[ab] * p.I + k0 + ak;
                sA[ab][ak + 0u] = half(mp[0]);
                sA[ab][ak + 1u] = half(mp[1]);
            } else {
                sA[ab][ak + 0u] = half(0.0h);
                sA[ab][ak + 1u] = half(0.0h);
            }

            uint gn = n0 + r_loc;
            if (gn < p.D) {
                uint wi = (k0 >> 2) + (tid & 3u) * 2u;
                uint grp = k0 >> 5;
                const device uint* wp = w2_u32 + e_w_base + (ulong)gn * row_words + wi;
                uint q0 = wp[0], q1 = wp[1];
                float sc = e8m0_to_f32(w2_scale[e_s_base + (ulong)gn * row_grps + grp]);
                sB[c_blk + 0u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0))        * sc);
                sB[c_blk + 1u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 8))   * sc);
                sB[c_blk + 2u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 16))  * sc);
                sB[c_blk + 3u][r_loc] = half(fp8_e4m3_to_f32(uchar(q0 >> 24))  * sc);
                sB[c_blk + 4u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1))        * sc);
                sB[c_blk + 5u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 8))   * sc);
                sB[c_blk + 6u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 16))  * sc);
                sB[c_blk + 7u][r_loc] = half(fp8_e4m3_to_f32(uchar(q1 >> 24))  * sc);
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

        if (ab < tb) {
            uint sl = s_slots[ab];
            float rw = top_w[sl];
            if (n0 + ak + 0u < p.D) slot_out[(ulong)sl * p.D + n0 + ak + 0u] = sOut[ab][ak + 0u] * rw;
            if (n0 + ak + 1u < p.D) slot_out[(ulong)sl * p.D + n0 + ak + 1u] = sOut[ab][ak + 1u] * rw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void inkling_moe_reduce_slots(
    const device float* slot_out [[buffer(0)]], // [M * K_top, D]
    device float*       out      [[buffer(1)]], // [M, D]
    constant MoeNvfp4Params& p   [[buffer(2)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint row = gid.y;
    if (d >= p.D || row >= p.M) return;
    float sum = 0.0f;
    ulong base = (ulong)row * p.K_top * p.D + d;
    for (uint k = 0; k < p.K_top; k++) {
        sum += slot_out[base + (ulong)k * p.D];
    }
    out[(ulong)row * p.D + d] = sum;
}

// ============================================================================
// 14. Inkling 2 Mini DSpark MTP Drafter Kernels
// ============================================================================

struct MtpMixParams {
    uint  M_out;
    uint  D;
    uint  row_offset;
    float eps;
    float b0;
    float b1;
    float b2;
    float b3;
};

kernel void inkling_mtp_mix_norm(
    const device float*  cap0   [[buffer(0)]],
    const device float*  cap1   [[buffer(1)]],
    const device float*  cap2   [[buffer(2)]],
    const device float*  cap3   [[buffer(3)]],
    const device ushort* norm_w [[buffer(4)]],
    device float*        out    [[buffer(5)]],
    constant MtpMixParams& p    [[buffer(6)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    if (row >= p.M_out) return;
    ulong src_off = (ulong)(row + p.row_offset) * p.D;
    device float* op = out + (ulong)row * p.D;

    float local_v[16];
    float sq = 0.0f;
    uint k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = p.b0 * cap0[src_off + i] + p.b1 * cap1[src_off + i]
                + p.b2 * cap2[src_off + i] + p.b3 * cap3[src_off + i];
        local_v[k++] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float s_sum[32];
    uint sg = tid >> 5, lane = tid & 31u;
    if (lane == 0u) s_sum[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size >> 5;
        float tot = simd_sum((lane < nsg) ? s_sum[lane] : 0.0f);
        if (lane == 0u) s_sum[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = s_sum[0];

    k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        op[i] = local_v[k++] * inv_rms * bf16_to_f32(norm_w[i]);
    }
}

struct MtpAttnParams {
    uint  M;
    uint  Hq;         // 24
    uint  Hkv;        // 8
    uint  D;          // 128
    uint  pos;        // RoPE start pos (observe: start_pos, propose: ctx_len)
    uint  ctx_len;    // number of committed tokens in draft KV cache
    uint  kv_cap;     // 1024 for sliding, max_seq for global
    uint  window;     // 1024 for sliding, 0 for global
    float eps;
    float rope_theta; // 1e6
    float scale;      // 1/sqrt(128)
};

// Per-head K RMSNorm + full NeoX RoPE + FP16 write to draft context KV cache
kernel void inkling_mtp_k_rope_write(
    const device float*  k_in    [[buffer(0)]],
    const device float*  v_in    [[buffer(1)]],
    const device ushort* k_norm  [[buffer(2)]], // bf16 [D]
    device half*         k_cache [[buffer(3)]],
    device half*         v_cache [[buffer(4)]],
    constant MtpAttnParams& p    [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hkv || t >= p.M) return;

    ulong src_off = ((ulong)t * p.Hkv + h) * p.D;
    const device float* kp = k_in + src_off;
    const device float* vp = v_in + src_off;

    // 4 elems/lane for D=128: (lane, lane+64) and (lane+32, lane+96) are exact NeoX RoPE pairs!
    float k0 = kp[lane], k1 = kp[lane + 32u], k2 = kp[lane + 64u], k3 = kp[lane + 96u];
    float ksq = simd_sum(fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3))));
    float inv = rsqrt(ksq / float(p.D) + p.eps);
    k0 *= inv * bf16_to_f32(k_norm[lane]);
    k1 *= inv * bf16_to_f32(k_norm[lane + 32u]);
    k2 *= inv * bf16_to_f32(k_norm[lane + 64u]);
    k3 *= inv * bf16_to_f32(k_norm[lane + 96u]);

    float seq_f = float(p.pos + t);
    float ang0 = seq_f * pow(p.rope_theta, -2.0f * float(lane) / float(p.D));
    float ang1 = seq_f * pow(p.rope_theta, -2.0f * float(lane + 32u) / float(p.D));
    float c0 = cos(ang0), s0 = sin(ang0);
    float c1 = cos(ang1), s1 = sin(ang1);

    float rk0 = k0 * c0 - k2 * s0;
    float rk2 = k2 * c0 + k0 * s0;
    float rk1 = k1 * c1 - k3 * s1;
    float rk3 = k3 * c1 + k1 * s1;

    uint seq_pos = p.pos + t;
    uint slot = (p.window > 0u) ? (seq_pos % p.kv_cap) : seq_pos;
    ulong dst_off = ((ulong)slot * p.Hkv + h) * p.D;
    device half* kdst = k_cache + dst_off;
    device half* vdst = v_cache + dst_off;

    kdst[lane]        = half(rk0);
    kdst[lane + 32u]  = half(rk1);
    kdst[lane + 64u]  = half(rk2);
    kdst[lane + 96u]  = half(rk3);

    vdst[lane]        = half(vp[lane]);
    vdst[lane + 32u]  = half(vp[lane + 32u]);
    vdst[lane + 64u]  = half(vp[lane + 64u]);
    vdst[lane + 96u]  = half(vp[lane + 96u]);
}

// Single-launch Q/K RMSNorm + full NeoX RoPE + V half cast for M=8 draft block
// Grid: (Hq + Hkv, M, 1), TG: (32, 1, 1)
kernel void inkling_mtp_qkv_rope_prep(
    const device float*  q_in   [[buffer(0)]],
    const device float*  k_in   [[buffer(1)]],
    const device float*  v_in   [[buffer(2)]],
    const device ushort* q_norm [[buffer(3)]],
    const device ushort* k_norm [[buffer(4)]],
    device float*        q_out  [[buffer(5)]],
    device half*         k_self [[buffer(6)]],
    device half*         v_self [[buffer(7)]],
    constant MtpAttnParams& p   [[buffer(8)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint t = tgid.y;
    if (t >= p.M) return;

    float seq_f = float(p.pos + t);
    float ang0 = seq_f * pow(p.rope_theta, -2.0f * float(lane) / float(p.D));
    float ang1 = seq_f * pow(p.rope_theta, -2.0f * float(lane + 32u) / float(p.D));
    float c0 = cos(ang0), s0 = sin(ang0);
    float c1 = cos(ang1), s1 = sin(ang1);

    if (tgid.x < p.Hq) {
        uint h = tgid.x;
        ulong off = ((ulong)t * p.Hq + h) * p.D;
        const device float* qp = q_in + off;
        float q0 = qp[lane], q1 = qp[lane + 32u], q2 = qp[lane + 64u], q3 = qp[lane + 96u];
        float inv = rsqrt(simd_sum(fma(q0, q0, fma(q1, q1, fma(q2, q2, q3 * q3)))) / float(p.D) + p.eps);
        q0 *= inv * bf16_to_f32(q_norm[lane]);
        q1 *= inv * bf16_to_f32(q_norm[lane + 32u]);
        q2 *= inv * bf16_to_f32(q_norm[lane + 64u]);
        q3 *= inv * bf16_to_f32(q_norm[lane + 96u]);

        device float* qd = q_out + off;
        qd[lane]       = q0 * c0 - q2 * s0;
        qd[lane + 32u] = q1 * c1 - q3 * s1;
        qd[lane + 64u] = q2 * c0 + q0 * s0;
        qd[lane + 96u] = q3 * c1 + q1 * s1;
    } else {
        uint h = tgid.x - p.Hq;
        if (h >= p.Hkv) return;
        ulong off = ((ulong)t * p.Hkv + h) * p.D;
        const device float* kp = k_in + off;
        const device float* vp = v_in + off;
        float k0 = kp[lane], k1 = kp[lane + 32u], k2 = kp[lane + 64u], k3 = kp[lane + 96u];
        float inv = rsqrt(simd_sum(fma(k0, k0, fma(k1, k1, fma(k2, k2, k3 * k3)))) / float(p.D) + p.eps);
        k0 *= inv * bf16_to_f32(k_norm[lane]);
        k1 *= inv * bf16_to_f32(k_norm[lane + 32u]);
        k2 *= inv * bf16_to_f32(k_norm[lane + 64u]);
        k3 *= inv * bf16_to_f32(k_norm[lane + 96u]);

        device half* kd = k_self + off;
        device half* vd = v_self + off;
        kd[lane]       = half(k0 * c0 - k2 * s0);
        kd[lane + 32u] = half(k1 * c1 - k3 * s1);
        kd[lane + 64u] = half(k2 * c0 + k0 * s0);
        kd[lane + 96u] = half(k3 * c1 + k1 * s1);

        vd[lane]       = half(vp[lane]);
        vd[lane + 32u] = half(vp[lane + 32u]);
        vd[lane + 64u] = half(vp[lane + 64u]);
        vd[lane + 96u] = half(vp[lane + 96u]);
    }
}

// Unmasked bidirectional block-8 GQA attending over [ctx_kv, self_kv]
kernel void inkling_mtp_attn(
    const device float* q_in    [[buffer(0)]],
    const device half*  k_ctx   [[buffer(1)]],
    const device half*  v_ctx   [[buffer(2)]],
    const device half*  k_self  [[buffer(3)]],
    const device half*  v_self  [[buffer(4)]],
    device float*       out     [[buffer(5)]],
    constant MtpAttnParams& p   [[buffer(6)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  lane                  [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x;
    uint t = tgid.y;
    if (h >= p.Hq || t >= p.M) return;

    uint hkv = h / (p.Hq / p.Hkv);
    ulong q_off = ((ulong)t * p.Hq + h) * p.D;
    const device float* qp = q_in + q_off;
    device float* op = out + q_off;

    float q0 = qp[lane], q1 = qp[lane + 32u], q2 = qp[lane + 64u], q3 = qp[lane + 96u];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    ulong kv_stride = (ulong)p.Hkv * p.D;
    ulong head_off  = (ulong)hkv * p.D;

    uint k_start = (p.window > 0u && p.ctx_len > p.window) ? (p.ctx_len - p.window) : 0u;
    for (uint kpos = k_start; kpos < p.ctx_len; kpos++) {
        uint slot = (p.window > 0u) ? (kpos % p.kv_cap) : kpos;
        const device half* kp = k_ctx + (ulong)slot * kv_stride + head_off;
        float dot = fma(q0, float(kp[lane]),
                    fma(q1, float(kp[lane + 32u]),
                    fma(q2, float(kp[lane + 64u]),
                        q3 * float(kp[lane + 96u]))));
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device half* vp = v_ctx + (ulong)slot * kv_stride + head_off;
        a0 = fma(a0, alpha, beta * float(vp[lane]));
        a1 = fma(a1, alpha, beta * float(vp[lane + 32u]));
        a2 = fma(a2, alpha, beta * float(vp[lane + 64u]));
        a3 = fma(a3, alpha, beta * float(vp[lane + 96u]));
    }

    for (uint s = 0; s < p.M; s++) {
        const device half* kp = k_self + (ulong)s * kv_stride + head_off;
        float dot = fma(q0, float(kp[lane]),
                    fma(q1, float(kp[lane + 32u]),
                    fma(q2, float(kp[lane + 64u]),
                        q3 * float(kp[lane + 96u]))));
        float score = simd_sum(dot) * p.scale;
        float m_new = max(m_max, score);
        float alpha = (m_max == -INFINITY) ? 0.0f : exp(m_max - m_new);
        float beta  = exp(score - m_new);
        l_sum = fma(l_sum, alpha, beta);
        m_max = m_new;

        const device half* vp = v_self + (ulong)s * kv_stride + head_off;
        a0 = fma(a0, alpha, beta * float(vp[lane]));
        a1 = fma(a1, alpha, beta * float(vp[lane + 32u]));
        a2 = fma(a2, alpha, beta * float(vp[lane + 64u]));
        a3 = fma(a3, alpha, beta * float(vp[lane + 96u]));
    }

    float inv_l = 1.0f / l_sum;
    op[lane]        = a0 * inv_l;
    op[lane + 32u]  = a1 * inv_l;
    op[lane + 64u]  = a2 * inv_l;
    op[lane + 96u]  = a3 * inv_l;
}

struct MtpAddNormParams {
    uint  M;
    uint  D;
    uint  has_res;   // 1 = summed = a + b, 0 = summed = a
    uint  write_sum; // 1 = store summed into sum_out
    float eps;
};

kernel void inkling_mtp_add_norm(
    const device float*  a        [[buffer(0)]],
    const device float*  b        [[buffer(1)]],
    const device ushort* norm_w   [[buffer(2)]],
    device float*        sum_out  [[buffer(3)]],
    device float*        norm_out [[buffer(4)]],
    constant MtpAddNormParams& p  [[buffer(5)]],
    uint row                      [[threadgroup_position_in_grid]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    if (row >= p.M) return;
    ulong off = (ulong)row * p.D;
    float local_v[16];
    float sq = 0.0f;
    uint k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        float v = p.has_res ? (a[off + i] + b[off + i]) : a[off + i];
        if (p.write_sum) sum_out[off + i] = v;
        local_v[k++] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float s_sum[32];
    uint sg = tid >> 5, lane = tid & 31u;
    if (lane == 0u) s_sum[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        uint nsg = tg_size >> 5;
        float tot = simd_sum((lane < nsg) ? s_sum[lane] : 0.0f);
        if (lane == 0u) s_sum[0] = rsqrt(tot / float(p.D) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = s_sum[0];

    k = 0u;
    for (uint i = tid; i < p.D; i += tg_size) {
        norm_out[off + i] = local_v[k++] * inv_rms * bf16_to_f32(norm_w[i]);
    }
}

kernel void inkling_mtp_fill_masks(
    const device ushort* mask_embed [[buffer(0)]],
    device float*        h_block    [[buffer(1)]], // [8, D], rows 1..7 filled
    constant uint&       D          [[buffer(2)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint r = gid.y + 1u; // 1..7
    if (d >= D || r >= 8u) return;
    h_block[(ulong)r * D + d] = bf16_to_f32(mask_embed[d]);
}

struct MarkovPrepParams {
    uint rank;       // 256
    uint D;          // 2304
    uint step;       // 0..count-1
    uint prev_idx;   // index into draft_ids buffer (0 = anchor, 1 = draft[0], ...)
};

kernel void inkling_mtp_markov_prep(
    const device float*  state      [[buffer(0)]], // [rank]
    const device ushort* markov_w1  [[buffer(1)]], // bf16 [vocab, rank]
    const device uint*   tok_ring   [[buffer(2)]], // [anchor, draft0, draft1, ...]
    const device float*  h_final    [[buffer(3)]], // [block_size, D] (row step+1 is used!)
    const device float*  markov_ctx [[buffer(4)]], // [D]
    device float*        rnn_in     [[buffer(5)]], // [2*rank + 2*D = 5120]
    constant MarkovPrepParams& p    [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint total = 2u * p.rank + 2u * p.D;
    if (gid >= total) return;
    if (gid < p.rank) {
        rnn_in[gid] = (p.step == 0u) ? 0.0f : state[gid];
    } else if (gid < 2u * p.rank) {
        uint prev_tok = tok_ring[p.prev_idx];
        rnn_in[gid] = bf16_to_f32(markov_w1[(ulong)prev_tok * p.rank + (gid - p.rank)]);
    } else if (gid < 2u * p.rank + p.D) {
        rnn_in[gid] = h_final[(ulong)(p.step + 1u) * p.D + (gid - 2u * p.rank)];
    } else {
        rnn_in[gid] = markov_ctx[gid - (2u * p.rank + p.D)];
    }
}

kernel void inkling_mtp_markov_gate(
    const device float*  rnn_out  [[buffer(0)]], // [3 * rank]
    const device ushort* rnn_bias [[buffer(1)]], // bf16 [3 * rank]
    device float*        state    [[buffer(2)]], // [rank]
    device float*        signal   [[buffer(3)]], // [rank]
    constant uint&       rank     [[buffer(4)]],
    constant uint&       step     [[buffer(5)]],
    uint i                        [[thread_position_in_grid]]
) {
    if (i >= rank) return;
    float g = sigmoid_f32(rnn_out[i] + bf16_to_f32(rnn_bias[i]));
    float c = tanh(rnn_out[rank + i] + bf16_to_f32(rnn_bias[rank + i]));
    float o = tanh(rnn_out[2u * rank + i] + bf16_to_f32(rnn_bias[2u * rank + i]));
    float prev_s = (step == 0u) ? 0.0f : state[i];
    state[i]  = fma(g, prev_s, (1.0f - g) * c);
    signal[i] = o;
}

kernel void inkling_mtp_add_h_bias(
    const device float*  h_step   [[buffer(0)]], // [D]
    const device float*  proj_sig [[buffer(1)]], // [D]
    const device ushort* bias     [[buffer(2)]], // bf16 [D]
    device float*        eff_h    [[buffer(3)]], // [D]
    constant uint&       D        [[buffer(4)]],
    uint i                        [[thread_position_in_grid]]
) {
    if (i >= D) return;
    eff_h[i] = h_step[i] + proj_sig[i] + bf16_to_f32(bias[i]);
}
