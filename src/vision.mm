#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "vision.hpp"
#include "json.hpp"
#include "prefix_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mi {

namespace {

struct VisGemmParams { uint32_t M, N, K, add_residual, has_bias; };
struct VisPosParams  { uint32_t N, D; };
struct VisNormParams { uint32_t M, D; float eps; };
struct VisAttnParams { uint32_t N, H, Dh, seq_start, seq_len; float scale; };

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Decode standard Base64 string (ignores whitespace)
std::vector<uint8_t> decode_base64(const std::string& in) {
    static const int8_t T[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        int8_t d = T[c];
        if (d == -1) continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Exact Qwen3-VL smart_resize with long-edge cap (keeps screenshots crisp in <= 1,024 tokens!)
std::pair<int, int> smart_resize_qwen(int height, int width, int factor, int min_pixels, int max_pixels, int max_edge) {
    if (height <= 0 || width <= 0) throw std::runtime_error("Invalid image dimensions");
    double h = static_cast<double>(height);
    double w = static_cast<double>(width);
    if (max_edge > 0 && std::max(h, w) > static_cast<double>(max_edge)) {
        double s = static_cast<double>(max_edge) / std::max(h, w);
        h *= s;
        w *= s;
    }
    int h_bar = std::max(factor, static_cast<int>(std::llround(h / factor)) * factor);
    int w_bar = std::max(factor, static_cast<int>(std::llround(w / factor)) * factor);
    if (static_cast<int64_t>(h_bar) * w_bar > max_pixels) {
        double beta = std::sqrt((h * w) / static_cast<double>(max_pixels));
        h_bar = std::max(factor, static_cast<int>(std::floor(h / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::floor(w / beta / factor)) * factor);
    } else if (static_cast<int64_t>(h_bar) * w_bar < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (h * w));
        h_bar = std::max(factor, static_cast<int>(std::ceil(h * beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::ceil(w * beta / factor)) * factor);
    }
    return {h_bar, w_bar};
}

} // namespace

QwenVisionTower::QwenVisionTower(MetalContext& ctx, const SafetensorsStore& store, const std::string& model_dir)
    : ctx_(ctx) {
    std::string pfx;
    if (store.has("vision_tower.patch_embed.proj.weight")) pfx = "vision_tower.";
    else if (store.has("model.visual.patch_embed.proj.weight")) pfx = "model.visual.";
    else return; // Text-only checkpoint (no vision weights)

    Json cfg = parse_json(read_file_to_string(model_dir + "/config.json"));
    if (cfg.contains("vision_config") && cfg["vision_config"].is_object()) {
        const Json& vc = cfg["vision_config"];
        depth_               = static_cast<int>(vc["depth"].as_int(27));
        hidden_size_         = static_cast<int>(vc["hidden_size"].as_int(1152));
        intermediate_size_   = static_cast<int>(vc["intermediate_size"].as_int(4304));
        num_heads_           = static_cast<int>(vc["num_heads"].as_int(16));
        head_dim_            = hidden_size_ / num_heads_;
        out_hidden_size_     = static_cast<int>(vc["out_hidden_size"].as_int(5120));
        patch_size_          = static_cast<int>(vc["patch_size"].as_int(16));
        temporal_patch_size_ = static_cast<int>(vc["temporal_patch_size"].as_int(2));
        spatial_merge_size_  = static_cast<int>(vc["spatial_merge_size"].as_int(2));
        num_pos_embed_       = static_cast<int>(vc["num_position_embeddings"].as_int(2304));
        num_grid_per_side_   = static_cast<int>(std::round(std::sqrt(static_cast<double>(num_pos_embed_))));
    }

    std::string shader_dir = MOMETAL_SHADER_DIR;
    ctx_.load_shader_file("vision", shader_dir + "/vision.metal");
    pso_gemm_            = ctx_.pipeline("vision", "vis_bf16_gemm_f16x");
    pso_gemm_gelu_tanh_  = ctx_.pipeline("vision", "vis_bf16_gemm_gelu_tanh_f16x");
    pso_gemm_gelu_exact_ = ctx_.pipeline("vision", "vis_bf16_gemm_gelu_exact_f16x");
    pso_pos_add_norm_f16_ = ctx_.pipeline("vision", "vis_add_pos_and_norm_f16");
    pso_norm_f16_        = ctx_.pipeline("vision", "vis_layernorm_f16");
    pso_qkv_rope_        = ctx_.pipeline("vision", "vis_qkv_rope_split");
    pso_flash_attn_      = ctx_.pipeline("vision", "vis_flash_attn_f16");

    // Precompute exact u8 -> BF16 LUT (round-to-nearest-even) for native M4 bfloat8x8 PatchEmbed!
    for (int i = 0; i < 256; ++i) {
        float fv = static_cast<float>(i) * (1.0f / 127.5f) - 1.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &fv, 4);
        lut_u8_to_f16_[i] = static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }

    patch_w_     = store.get(pfx + "patch_embed.proj.weight");
    patch_b_     = store.get(pfx + "patch_embed.proj.bias");
    pos_embed_w_ = store.get(pfx + "pos_embed.weight");

    blocks_.resize(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        std::string bp = pfx + "blocks." + std::to_string(i) + ".";
        BlockWeights& bw = blocks_[i];
        bw.norm1_w = store.get(bp + "norm1.weight");
        bw.norm1_b = store.get(bp + "norm1.bias");
        bw.norm2_w = store.get(bp + "norm2.weight");
        bw.norm2_b = store.get(bp + "norm2.bias");
        bw.qkv_w   = store.get(bp + "attn.qkv.weight");
        bw.qkv_b   = store.get(bp + "attn.qkv.bias");
        bw.proj_w  = store.get(bp + "attn.proj.weight");
        bw.proj_b  = store.get(bp + "attn.proj.bias");
        bw.fc1_w   = store.get(bp + "mlp.linear_fc1.weight");
        bw.fc1_b   = store.get(bp + "mlp.linear_fc1.bias");
        bw.fc2_w   = store.get(bp + "mlp.linear_fc2.weight");
        bw.fc2_b   = store.get(bp + "mlp.linear_fc2.bias");
    }

    merger_norm_w_ = store.get(pfx + "merger.norm.weight");
    merger_norm_b_ = store.get(pfx + "merger.norm.bias");
    merger_fc1_w_  = store.get(pfx + "merger.linear_fc1.weight");
    merger_fc1_b_  = store.get(pfx + "merger.linear_fc1.bias");
    merger_fc2_w_  = store.get(pfx + "merger.linear_fc2.weight");
    merger_fc2_b_  = store.get(pfx + "merger.linear_fc2.bias");

    ready_ = true;
}

EncodedImage QwenVisionTower::encode_url(const std::string& url, int max_edge) {
    if (url.rfind("data:", 0) == 0) {
        size_t comma = url.find(',');
        if (comma == std::string::npos) throw std::runtime_error("Malformed data: image URL");
        std::vector<uint8_t> raw = decode_base64(url.substr(comma + 1));
        if (raw.empty()) throw std::runtime_error("Empty base64 image data");
        return encode_bytes(raw.data(), raw.size(), max_edge);
    }

    std::string path = url;
    if (path.rfind("file://", 0) == 0) path = path.substr(7);
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open image file: " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (raw.empty()) throw std::runtime_error("Empty image file: " + path);
    return encode_bytes(raw.data(), raw.size(), max_edge);
}

EncodedImage QwenVisionTower::encode_bytes(const uint8_t* data, size_t len, int max_edge) {
    if (!ready_) throw std::runtime_error("Vision tower not loaded for this model");

    uint64_t h = fnv1a_64(data, len);
    h = fnv1a_64(&max_edge, sizeof(max_edge), h);
    auto cit = cache_.find(h);
    if (cit != cache_.end()) {
        lru_order_.erase(cit->second.second);
        lru_order_.push_front(h);
        cit->second.second = lru_order_.begin();
        return cit->second.first;
    }

    // Decode compressed image (PNG / JPEG / WebP / HEIC / GIF) via macOS ImageIO + CoreGraphics
    int res_h = 0, res_w = 0;
    std::vector<uint8_t> rgba;
    @autoreleasepool {
        CFDataRef cf_data = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, data, static_cast<CFIndex>(len), kCFAllocatorNull);
        if (!cf_data) throw std::runtime_error("CFDataCreate failed for image");
        CGImageSourceRef src = CGImageSourceCreateWithData(cf_data, nullptr);
        CFRelease(cf_data);
        if (!src) throw std::runtime_error("CGImageSourceCreateWithData failed");
        CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
        CFRelease(src);
        if (!img) throw std::runtime_error("CGImageSourceCreateImageAtIndex failed");

        int orig_w = static_cast<int>(CGImageGetWidth(img));
        int orig_h = static_cast<int>(CGImageGetHeight(img));
        int factor = patch_size_ * spatial_merge_size_; // 32
        auto [th, tw] = smart_resize_qwen(orig_h, orig_w, factor, 65536, 1280 * 1280, max_edge);
        res_h = th;
        res_w = tw;

        rgba.resize(static_cast<size_t>(res_h) * res_w * 4u);
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef cg = CGBitmapContextCreate(
            rgba.data(), res_w, res_h, 8, static_cast<size_t>(res_w) * 4u, cs,
            static_cast<uint32_t>(kCGImageAlphaPremultipliedLast) | static_cast<uint32_t>(kCGBitmapByteOrder32Big)
        );
        CGColorSpaceRelease(cs);
        if (!cg) {
            CGImageRelease(img);
            throw std::runtime_error("CGBitmapContextCreate failed");
        }
        // Fill white background for transparent PNGs, then draw high-quality resized image
        CGContextSetRGBFillColor(cg, 1.0, 1.0, 1.0, 1.0);
        CGContextFillRect(cg, CGRectMake(0, 0, res_w, res_h));
        CGContextSetInterpolationQuality(cg, kCGInterpolationHigh);
        CGContextDrawImage(cg, CGRectMake(0, 0, res_w, res_h), img);
        CGContextRelease(cg);
        CGImageRelease(img);
    }

    // Patchify directly into [N_patches, tps=2, ps=16, ps=16, C=3] FP16 matching proj.weight [1152, 2, 16, 16, 3]!
    const int ps = patch_size_;
    const int ms = spatial_merge_size_;
    const int tps = temporal_patch_size_;
    const int gh = res_h / ps;
    const int gw = res_w / ps;
    const int mh_blocks = gh / ms;
    const int mw_blocks = gw / ms;
    const int n_patches = gh * gw;
    const int patch_dim = tps * ps * ps * 3; // 1536

    std::vector<uint16_t> patches_f16(static_cast<size_t>(n_patches) * patch_dim);
    const size_t frame_elems = static_cast<size_t>(ps) * ps * 3u;
    const size_t frame_bytes = frame_elems * sizeof(uint16_t);

    int p_idx = 0;
    for (int bh = 0; bh < mh_blocks; ++bh) {
        for (int bw = 0; bw < mw_blocks; ++bw) {
            for (int mh = 0; mh < ms; ++mh) {
                for (int mw = 0; mw < ms; ++mw) {
                    int y0 = (bh * ms + mh) * ps;
                    int x0 = (bw * ms + mw) * ps;
                    uint16_t* dst0 = patches_f16.data() + static_cast<size_t>(p_idx) * patch_dim;
                    uint16_t* dst1 = dst0 + frame_elems; // temporal duplicate (t=1 == t=0)
                    uint16_t* out_p = dst0;
                    for (int py = 0; py < ps; ++py) {
                        const uint8_t* row_p = rgba.data() + (static_cast<size_t>(y0 + py) * res_w + x0) * 4u;
                        for (int px = 0; px < ps; ++px) {
                            out_p[0] = lut_u8_to_f16_[row_p[0]];
                            out_p[1] = lut_u8_to_f16_[row_p[1]];
                            out_p[2] = lut_u8_to_f16_[row_p[2]];
                            out_p += 3;
                            row_p += 4;
                        }
                    }
                    std::memcpy(dst1, dst0, frame_bytes);
                    ++p_idx;
                }
            }
        }
    }

    EncodedImage enc_img = run_vit(patches_f16, gh, gw, h);

    // Insert into LRU cache
    if (cache_.size() >= MAX_CACHED_IMAGES) {
        uint64_t evict_k = lru_order_.back();
        lru_order_.pop_back();
        cache_.erase(evict_k);
    }
    lru_order_.push_front(h);
    cache_.emplace(h, std::make_pair(enc_img, lru_order_.begin()));
    return enc_img;
}

void QwenVisionTower::ensure_scratch(int N) {
    if (N <= scratch_cap_n_) return;
    int cap = std::max(N, 1024);
    const int ms = spatial_merge_size_;
    const int cap_merged = (cap + ms * ms - 1) / (ms * ms);
    const int patch_dim = temporal_patch_size_ * patch_size_ * patch_size_ * 3;
    const int half_d = head_dim_ / 2;

    size_t hid_bytes   = static_cast<size_t>(cap) * hidden_size_ * sizeof(float);
    size_t norm_bytes  = static_cast<size_t>(cap) * hidden_size_ * sizeof(uint16_t);
    size_t qkv_bytes   = static_cast<size_t>(cap) * 3 * hidden_size_ * sizeof(float);
    size_t q_bytes     = static_cast<size_t>(cap) * hidden_size_ * sizeof(uint16_t);
    size_t mlp16_bytes = static_cast<size_t>(cap) * intermediate_size_ * sizeof(uint16_t);
    size_t out_bytes   = static_cast<size_t>(cap_merged) * out_hidden_size_ * sizeof(float);

    b_patches_ = ctx_.alloc_shared(static_cast<size_t>(cap) * patch_dim * sizeof(uint16_t));
    b_pidx_    = ctx_.alloc_shared(static_cast<size_t>(cap) * 4u * sizeof(int32_t));
    b_pwt_     = ctx_.alloc_shared(static_cast<size_t>(cap) * 4u * sizeof(float));
    b_freqs_   = ctx_.alloc_shared(static_cast<size_t>(cap) * half_d * sizeof(float));
    b_hid_     = ctx_.alloc_private(hid_bytes);
    b_norm_    = ctx_.alloc_private(norm_bytes);
    b_qkv_     = ctx_.alloc_private(qkv_bytes);
    b_q_       = ctx_.alloc_private(q_bytes);
    b_k_       = ctx_.alloc_private(q_bytes);
    b_v_       = ctx_.alloc_private(q_bytes);
    b_aout_    = ctx_.alloc_private(std::max(q_bytes, mlp16_bytes));
    b_out_     = ctx_.alloc_shared(out_bytes);
    scratch_cap_n_ = cap;
}

EncodedImage QwenVisionTower::run_vit(const std::vector<uint16_t>& patches_f16, int gh, int gw, uint64_t hash) {
    const int N = gh * gw;
    const int ms = spatial_merge_size_;
    const int N_merged = N / (ms * ms);
    const int patch_dim = temporal_patch_size_ * patch_size_ * patch_size_ * 3; // 1536
    const int half_d = head_dim_ / 2; // 36
    const int quarter_d = half_d / 2; // 18

    // Precompute bilinear 2D pos_embed indices/weights + 2D Vision RoPE freqs in exact (bh, bw, mh, mw) order
    std::vector<int32_t> pos_idx(static_cast<size_t>(N) * 4);
    std::vector<float>   pos_wt(static_cast<size_t>(N) * 4);
    std::vector<float>   rot_freqs(static_cast<size_t>(N) * half_d);

    std::vector<float> inv_freq(quarter_d);
    for (int k = 0; k < quarter_d; ++k) {
        inv_freq[k] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * k) / static_cast<float>(half_d));
    }

    const int S = num_grid_per_side_; // 48
    int p_idx = 0;
    for (int bh = 0; bh < gh / ms; ++bh) {
        for (int bw = 0; bw < gw / ms; ++bw) {
            for (int mh = 0; mh < ms; ++mh) {
                for (int mw = 0; mw < ms; ++mw) {
                    int r = bh * ms + mh;
                    int c = bw * ms + mw;

                    float fh = (gh > 1) ? (static_cast<float>(r) * static_cast<float>(S - 1) / static_cast<float>(gh - 1)) : 0.0f;
                    float fw = (gw > 1) ? (static_cast<float>(c) * static_cast<float>(S - 1) / static_cast<float>(gw - 1)) : 0.0f;
                    int h0 = static_cast<int>(std::floor(fh));
                    int w0 = static_cast<int>(std::floor(fw));
                    int h1 = std::min(h0 + 1, S - 1);
                    int w1 = std::min(w0 + 1, S - 1);
                    float dh = fh - static_cast<float>(h0);
                    float dw = fw - static_cast<float>(w0);

                    pos_idx[p_idx * 4 + 0] = h0 * S + w0;
                    pos_idx[p_idx * 4 + 1] = h0 * S + w1;
                    pos_idx[p_idx * 4 + 2] = h1 * S + w0;
                    pos_idx[p_idx * 4 + 3] = h1 * S + w1;

                    pos_wt[p_idx * 4 + 0] = (1.0f - dh) * (1.0f - dw);
                    pos_wt[p_idx * 4 + 1] = (1.0f - dh) * dw;
                    pos_wt[p_idx * 4 + 2] = dh * (1.0f - dw);
                    pos_wt[p_idx * 4 + 3] = dh * dw;

                    float* rf = rot_freqs.data() + static_cast<size_t>(p_idx) * half_d;
                    for (int k = 0; k < quarter_d; ++k) {
                        rf[k]             = static_cast<float>(r) * inv_freq[k];
                        rf[quarter_d + k] = static_cast<float>(c) * inv_freq[k];
                    }
                    ++p_idx;
                }
            }
        }
    }

    EncodedImage result;
    result.content_hash = hash;
    result.grid_t = 1;
    result.grid_h = gh;
    result.grid_w = gw;
    result.embeddings.resize(static_cast<size_t>(N_merged) * out_hidden_size_);
    size_t out_bytes = result.embeddings.size() * sizeof(float);

    @autoreleasepool {
        ensure_scratch(N);
        std::memcpy([(id<MTLBuffer>)b_patches_ contents], patches_f16.data(), patches_f16.size() * sizeof(uint16_t));
        std::memcpy([(id<MTLBuffer>)b_pidx_ contents], pos_idx.data(), pos_idx.size() * sizeof(int32_t));
        std::memcpy([(id<MTLBuffer>)b_pwt_ contents], pos_wt.data(), pos_wt.size() * sizeof(float));
        std::memcpy([(id<MTLBuffer>)b_freqs_ contents], rot_freqs.data(), rot_freqs.size() * sizeof(float));

        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)ctx_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        auto barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

        auto gemm_pso = [&](id pso, const TensorView& w, const TensorView& b, id x_buf, id y_buf,
                            uint32_t M, uint32_t N_out, uint32_t K_in, bool add_res) {
            VisGemmParams gp{M, N_out, K_in, add_res ? 1u : 0u, b.valid() ? 1u : 0u};
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso];
            [enc setBuffer:(id<MTLBuffer>)w.buffer offset:w.offset atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)(b.valid() ? b.buffer : w.buffer) offset:(b.valid() ? b.offset : 0) atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)x_buf offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)y_buf offset:0 atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(N_out, 64u), ceil_div(M, 64u), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        };

        VisAttnParams ap{
            static_cast<uint32_t>(N), static_cast<uint32_t>(num_heads_), static_cast<uint32_t>(head_dim_),
            0u, static_cast<uint32_t>(N), 1.0f / std::sqrt(static_cast<float>(head_dim_))
        };
        VisNormParams np{static_cast<uint32_t>(N), static_cast<uint32_t>(hidden_size_), eps_};

        // 1. PatchEmbed Conv3d (as 9.2 KB BK=32 BF16 GEMM [N, 1536] -> [N, 1152])
        gemm_pso(pso_gemm_, patch_w_, patch_b_, b_patches_, b_hid_, static_cast<uint32_t>(N),
                 static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(patch_dim), false);
        barrier();

        // Round 25: Fuse 2D PosEmbed add + Block 0 norm1 into ONE single-pass register-cached kernel!
        {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_pos_add_norm_f16_];
            [enc setBuffer:(id<MTLBuffer>)b_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)pos_embed_w_.buffer offset:pos_embed_w_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)b_pidx_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)b_pwt_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)blocks_[0].norm1_w.buffer offset:blocks_[0].norm1_w.offset atIndex:4];
            [enc setBuffer:(id<MTLBuffer>)blocks_[0].norm1_b.buffer offset:blocks_[0].norm1_b.offset atIndex:5];
            [enc setBuffer:(id<MTLBuffer>)b_norm_ offset:0 atIndex:6];
            [enc setBytes:&np length:sizeof(np) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();
        }

        // 2. 27 ViT Blocks
        for (int i = 0; i < depth_; ++i) {
            const BlockWeights& bw = blocks_[i];

            // norm1 -> b_norm_ (FP16; skipped for i==0 because fused with pos_embed!)
            if (i > 0) {
                [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_norm_f16_];
                [enc setBuffer:(id<MTLBuffer>)b_hid_ offset:0 atIndex:0];
                [enc setBuffer:(id<MTLBuffer>)bw.norm1_w.buffer offset:bw.norm1_w.offset atIndex:1];
                [enc setBuffer:(id<MTLBuffer>)bw.norm1_b.buffer offset:bw.norm1_b.offset atIndex:2];
                [enc setBuffer:(id<MTLBuffer>)b_norm_ offset:0 atIndex:3];
                [enc setBytes:&np length:sizeof(np) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                barrier();
            }

            // QKV proj -> b_qkv_
            gemm_pso(pso_gemm_, bw.qkv_w, bw.qkv_b, b_norm_, b_qkv_, static_cast<uint32_t>(N),
                     static_cast<uint32_t>(3 * hidden_size_), static_cast<uint32_t>(hidden_size_), false);
            barrier();

            // Round 25: Split + 2D Vision RoPE (1 TG/token, 16x fewer cos/sin!) -> b_q_, b_k_, b_v_ (FP16)
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_qkv_rope_];
            [enc setBuffer:(id<MTLBuffer>)b_qkv_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)b_freqs_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)b_q_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)b_k_ offset:0 atIndex:3];
            [enc setBuffer:(id<MTLBuffer>)b_v_ offset:0 atIndex:4];
            [enc setBytes:&ap length:sizeof(ap) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            barrier();

            // Round 25: 2x-Bandwidth BQ=32, Exact-D=72 Bidirectional AMX FlashAttention -> b_aout_ (FP16)
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_flash_attn_];
            [enc setBuffer:(id<MTLBuffer>)b_q_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)b_k_ offset:0 atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)b_v_ offset:0 atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)b_aout_ offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ceil_div(N, 32u), num_heads_, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            barrier();

            // Attn out proj + residual -> b_hid_
            gemm_pso(pso_gemm_, bw.proj_w, bw.proj_b, b_aout_, b_hid_, static_cast<uint32_t>(N),
                     static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(hidden_size_), true);
            barrier();

            // norm2 -> b_norm_ (FP16, single-pass register-cached!)
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_norm_f16_];
            [enc setBuffer:(id<MTLBuffer>)b_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)bw.norm2_w.buffer offset:bw.norm2_w.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)bw.norm2_b.buffer offset:bw.norm2_b.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)b_norm_ offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            // Round 24: Fused MLP fc1 + bias + GELU(tanh) -> FP16 b_aout_ (zero FP32 [N,4304] round-trip!)
            gemm_pso(pso_gemm_gelu_tanh_, bw.fc1_w, bw.fc1_b, b_norm_, b_aout_, static_cast<uint32_t>(N),
                     static_cast<uint32_t>(intermediate_size_), static_cast<uint32_t>(hidden_size_), false);
            barrier();

            // MLP fc2 + residual -> b_hid_
            gemm_pso(pso_gemm_, bw.fc2_w, bw.fc2_b, b_aout_, b_hid_, static_cast<uint32_t>(N),
                     static_cast<uint32_t>(hidden_size_), static_cast<uint32_t>(intermediate_size_), true);
            barrier();
        }

        // 3. PatchMerger: LayerNorm(1152) -> reshape [N/4, 4608] -> Fused fc1+GELU(exact) -> fc2 [N/4, 5120]
        {
            [enc setComputePipelineState:(id<MTLComputePipelineState>)pso_norm_f16_];
            [enc setBuffer:(id<MTLBuffer>)b_hid_ offset:0 atIndex:0];
            [enc setBuffer:(id<MTLBuffer>)merger_norm_w_.buffer offset:merger_norm_w_.offset atIndex:1];
            [enc setBuffer:(id<MTLBuffer>)merger_norm_b_.buffer offset:merger_norm_b_.offset atIndex:2];
            [enc setBuffer:(id<MTLBuffer>)b_norm_ offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            barrier();

            uint32_t m_dim = static_cast<uint32_t>(hidden_size_ * ms * ms); // 4608
            gemm_pso(pso_gemm_gelu_exact_, merger_fc1_w_, merger_fc1_b_, b_norm_, b_aout_,
                     static_cast<uint32_t>(N_merged), m_dim, m_dim, false);
            barrier();

            gemm_pso(pso_gemm_, merger_fc2_w_, merger_fc2_b_, b_aout_, b_out_, static_cast<uint32_t>(N_merged),
                     static_cast<uint32_t>(out_hidden_size_), m_dim, false);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            throw std::runtime_error(std::string("Metal Vision Tower error: ") +
                                     [[cb.error localizedDescription] UTF8String]);
        }
        std::memcpy(result.embeddings.data(), [(id<MTLBuffer>)b_out_ contents], out_bytes);
    }

    return result;
}

} // namespace mi
