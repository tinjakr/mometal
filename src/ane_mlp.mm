#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "ane_mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <pthread/qos.h>
#include <stdexcept>
#include <thread>

namespace mi {

namespace {

constexpr OSType kOneComponent16Half = 0x4C303068; // 'L00h' FP16 IOSurface

// Minimal .npy loader for 1D float16 / float32 vectors (pieces_inv1_<tag>_L<L>.npy)
bool load_npy_f32(const std::string& path, std::vector<float>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    char magic[6];
    ifs.read(magic, 6);
    if (!ifs || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t ver[2];
    ifs.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hdr_len = 0;
    if (ver[0] == 1) {
        uint16_t l16 = 0;
        ifs.read(reinterpret_cast<char*>(&l16), 2);
        hdr_len = l16;
    } else {
        ifs.read(reinterpret_cast<char*>(&hdr_len), 4);
    }
    std::string hdr(hdr_len, '\0');
    ifs.read(hdr.data(), hdr_len);
    if (!ifs) return false;

    bool is_f16 = (hdr.find("<f2") != std::string::npos || hdr.find("|f2") != std::string::npos);
    bool is_f32 = (hdr.find("<f4") != std::string::npos || hdr.find("|f4") != std::string::npos);
    if (!is_f16 && !is_f32) return false;

    auto cur = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    size_t rem = static_cast<size_t>(ifs.tellg() - cur);
    ifs.seekg(cur, std::ios::beg);

    if (is_f32) {
        size_t n = rem / sizeof(float);
        out.resize(n);
        ifs.read(reinterpret_cast<char*>(out.data()), n * sizeof(float));
        return ifs.good();
    } else {
        size_t n = rem / sizeof(uint16_t);
        std::vector<uint16_t> raw(n);
        ifs.read(reinterpret_cast<char*>(raw.data()), n * sizeof(uint16_t));
        if (!ifs) return false;
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            __fp16 h;
            std::memcpy(&h, &raw[i], sizeof(uint16_t));
            out[i] = static_cast<float>(h);
        }
        return true;
    }
}

std::string detect_tag(const std::string& dir, int& out_pieces) {
    namespace fs = std::filesystem;
    // Priority order: full 4-bit (hf4), full 8-bit (hf8), halves 8-bit (hp8), halves 4-bit (hp4)
    const char* cands[] = {"hf4", "hf8", "hp8", "hp4", "hp4k4"};
    for (const char* t : cands) {
        std::string p0 = dir + "/L0_mlp_p0_" + t + ".mlmodelc";
        if (fs::is_directory(p0)) {
            std::string p1 = dir + "/L0_mlp_p1_" + t + ".mlmodelc";
            out_pieces = fs::is_directory(p1) ? 2 : 1;
            return t;
        }
    }
    // Fallback: scan directory for any L0_mlp_p0_*.mlmodelc
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string name = e.path().filename().string();
        const std::string pfx = "L0_mlp_p0_";
        const std::string sfx = ".mlmodelc";
        if (name.size() > pfx.size() + sfx.size() &&
            name.compare(0, pfx.size(), pfx) == 0 &&
            name.compare(name.size() - sfx.size(), sfx.size(), sfx) == 0) {
            std::string t = name.substr(pfx.size(), name.size() - pfx.size() - sfx.size());
            std::string p1 = dir + "/L0_mlp_p1_" + t + ".mlmodelc";
            out_pieces = fs::is_directory(p1) ? 2 : 1;
            return t;
        }
    }
    return "";
}

} // namespace

AneMlpEngine::IoSurfaceBinding AneMlpEngine::create_iosurface_io(int rows, int cols) {
    IoSurfaceBinding b{};
    CVPixelBufferRef pb = nullptr;
    NSDictionary* attrs = @{ (id)kCVPixelBufferIOSurfacePropertiesKey : @{} };
    CVReturn st = CVPixelBufferCreate(
        kCFAllocatorDefault,
        static_cast<size_t>(cols),
        static_cast<size_t>(rows),
        kOneComponent16Half,
        (__bridge CFDictionaryRef)attrs,
        &pb
    );
    if (st != kCVReturnSuccess || !pb) {
        throw std::runtime_error("CVPixelBufferCreate failed for ANE IOSurface (status=" + std::to_string(st) + ")");
    }
    b.pixel_buffer = pb;

    // Zero-initialize the IOSurface once
    CVPixelBufferLockBaseAddress(pb, 0);
    std::memset(CVPixelBufferGetBaseAddress(pb), 0, CVPixelBufferGetBytesPerRow(pb) * static_cast<size_t>(rows));
    CVPixelBufferUnlockBaseAddress(pb, 0);

    // CoreML MLMultiArray wrapping the exact same CVPixelBuffer / IOSurface
    NSArray<NSNumber*>* shape = @[@(rows), @(cols)];
    b.multi_array = [[MLMultiArray alloc] initWithPixelBuffer:pb shape:shape];
    if (!b.multi_array) {
        CVPixelBufferRelease(pb);
        throw std::runtime_error("MLMultiArray initWithPixelBuffer failed");
    }

    // Metal MTLTexture wrapping the EXACT SAME IOSurface (ZERO-COPY GPU <-> ANE!)
    IOSurfaceRef surf = CVPixelBufferGetIOSurface(pb);
    if (!surf) {
        CVPixelBufferRelease(pb);
        throw std::runtime_error("CVPixelBufferGetIOSurface returned null");
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                                  width:static_cast<NSUInteger>(cols)
                                                                                 height:static_cast<NSUInteger>(rows)
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tex = [(id<MTLDevice>)ctx_.device() newTextureWithDescriptor:td iosurface:surf plane:0];
    if (!tex) {
        CVPixelBufferRelease(pb);
        throw std::runtime_error("newTextureWithDescriptor:iosurface: failed");
    }
    b.mtl_texture = tex;
    return b;
}

void AneMlpEngine::destroy_iosurface_io(IoSurfaceBinding& b) {
    b.mtl_texture = nil;
    b.multi_array = nil;
    if (b.pixel_buffer) {
        CVPixelBufferRelease(static_cast<CVPixelBufferRef>(b.pixel_buffer));
        b.pixel_buffer = nullptr;
    }
}

AneMlpEngine::AneMlpEngine(MetalContext& ctx, const std::string& models_dir, int num_layers, const std::string& tag)
    : ctx_(ctx), num_layers_(num_layers) {
    auto t0 = std::chrono::steady_clock::now();

    if (!std::filesystem::is_directory(models_dir)) {
        throw std::runtime_error("ANE models directory not found: " + models_dir);
    }

    if (tag == "auto" || tag.empty()) {
        tag_ = detect_tag(models_dir, num_pieces_);
    } else {
        tag_ = tag;
        std::string p1 = models_dir + "/L0_mlp_p1_" + tag_ + ".mlmodelc";
        num_pieces_ = std::filesystem::is_directory(p1) ? 2 : 1;
    }
    if (tag_.empty()) {
        throw std::runtime_error("No L0_mlp_p0_*.mlmodelc found in " + models_dir);
    }

    // Create zero-copy IOSurface IO slots (2 halves x 2 sub-blocks = 4 input + 4*pieces output)
    for (int s = 0; s < MAX_SLOTS; ++s) {
        for (int sub = 0; sub < MAX_SUBS; ++sub) {
            in_io_[s][sub] = create_iosurface_io(block_rows_, PACKED_WIDTH);
            for (int k = 0; k < num_pieces_; ++k) {
                out_io_[s][sub][k] = create_iosurface_io(block_rows_, HIDDEN_WIDTH);
            }
        }
    }

    layers_.resize(static_cast<size_t>(num_layers_));

    // Load all 64 layer CoreML models in parallel across 8 threads
    std::atomic<int> next_layer{0};
    std::mutex err_mtx;
    std::string load_err;
    const int n_threads = std::min(8, num_layers_);

    auto loader = [&]() {
        @autoreleasepool {
            MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
            cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

            while (true) {
                int L = next_layer.fetch_add(1);
                if (L >= num_layers_) break;
                {
                    std::lock_guard<std::mutex> lk(err_mtx);
                    if (!load_err.empty()) break;
                }

                @autoreleasepool {
                    LayerData& ld = layers_[L];

                    // Optional inv1 SmoothQuant vector
                    std::string inv1_path = models_dir + "/pieces_inv1_" + tag_ + "_L" + std::to_string(L) + ".npy";
                    std::vector<float> inv1_vec;
                    if (load_npy_f32(inv1_path, inv1_vec) && static_cast<int>(inv1_vec.size()) == HIDDEN_WIDTH) {
                        bool all_ones = true;
                        for (float v : inv1_vec) {
                            if (std::fabs(v - 1.0f) > 1e-5f) { all_ones = false; break; }
                        }
                        if (!all_ones) {
                            id<MTLBuffer> buf = ctx_.alloc_shared(inv1_vec.size() * sizeof(float));
                            std::memcpy([buf contents], inv1_vec.data(), inv1_vec.size() * sizeof(float));
                            ld.inv1_buf = buf;
                        }
                    }

                    for (int k = 0; k < num_pieces_; ++k) {
                        std::string path = models_dir + "/L" + std::to_string(L) + "_mlp_p" +
                                           std::to_string(k) + "_" + tag_ + ".mlmodelc";
                        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
                        NSError* err = nil;
                        MLModel* m = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
                        if (!m) {
                            std::lock_guard<std::mutex> lk(err_mtx);
                            if (load_err.empty()) {
                                load_err = "Failed loading " + path + ": " +
                                           (err ? [[err localizedDescription] UTF8String] : "unknown error");
                            }
                            return;
                        }
                        ld.models[k] = m;

                        MLModelDescription* desc = m.modelDescription;
                        NSString* iname = desc.inputDescriptionsByName.allKeys.firstObject;
                        NSString* oname = desc.outputDescriptionsByName.allKeys.firstObject;

                        // Pre-bind providers & outputBackings for all (slot, sub) pairs (ZERO runtime allocs!)
                        for (int s = 0; s < MAX_SLOTS; ++s) {
                            for (int sub = 0; sub < MAX_SUBS; ++sub) {
                                MLFeatureValue* fv = [MLFeatureValue featureValueWithMultiArray:(MLMultiArray*)in_io_[s][sub].multi_array];
                                NSError* perr = nil;
                                MLDictionaryFeatureProvider* prov =
                                    [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{iname : fv} error:&perr];
                                MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
                                opts.outputBackings = @{oname : (MLMultiArray*)out_io_[s][sub][k].multi_array};
                                ld.calls[s][sub].pieces[k] = PieceCall{m, prov, opts};
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) threads.emplace_back(loader);
    for (auto& th : threads) th.join();

    if (!load_err.empty()) {
        for (int s = 0; s < MAX_SLOTS; ++s) {
            for (int sub = 0; sub < MAX_SUBS; ++sub) {
                destroy_iosurface_io(in_io_[s][sub]);
                for (int k = 0; k < num_pieces_; ++k) destroy_iosurface_io(out_io_[s][sub][k]);
            }
        }
        throw std::runtime_error(load_err);
    }

    // Warm up ALL 64 ANE layers once at startup (~0.35s) so the very first user request has ZERO cold-start E5/ANE lag!
    @autoreleasepool {
        for (int L = 0; L < num_layers_; ++L) {
            for (int k = 0; k < num_pieces_; ++k) {
                NSError* err = nil;
                const PieceCall& pc = layers_[L].calls[0][0].pieces[k];
                id<MLFeatureProvider> res = [(MLModel*)pc.model predictionFromFeatures:pc.provider options:pc.options error:&err];
                if (!res) {
                    throw std::runtime_error(std::string("ANE warmup prediction failed on L") + std::to_string(L) + ": " +
                                             (err ? [[err localizedDescription] UTF8String] : "unknown"));
                }
            }
        }
    }

    ready_ = true;
    worker_ = std::thread(&AneMlpEngine::worker_loop, this);

    auto t1 = std::chrono::steady_clock::now();
    load_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

AneMlpEngine::~AneMlpEngine() {
    if (worker_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_work_.notify_one();
        worker_.join();
    }
    layers_.clear();
    for (int s = 0; s < MAX_SLOTS; ++s) {
        for (int sub = 0; sub < MAX_SUBS; ++sub) {
            destroy_iosurface_io(in_io_[s][sub]);
            for (int k = 0; k < num_pieces_; ++k) destroy_iosurface_io(out_io_[s][sub][k]);
        }
    }
}

id AneMlpEngine::input_texture(int slot, int sub) const {
    return in_io_[slot][sub].mtl_texture;
}

id AneMlpEngine::output_texture(int slot, int sub, int piece) const {
    return out_io_[slot][sub][piece].mtl_texture;
}

id AneMlpEngine::inv1_buffer(int layer) const {
    return layers_[layer].inv1_buf;
}

void AneMlpEngine::submit(int layer, int slot, int n_subs) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        job_layer_ = layer;
        job_slot_  = slot;
        job_nsubs_ = n_subs;
        job_error_.clear();
        has_job_   = true;
    }
    cv_work_.notify_one();
}

void AneMlpEngine::wait() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [&]() { return !has_job_; });
    if (!job_error_.empty()) {
        std::string err = job_error_;
        job_error_.clear();
        throw std::runtime_error("ANE MLP prediction failed: " + err);
    }
}

void AneMlpEngine::worker_loop() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    while (true) {
        int L = 0, slot = 0, nsubs = 0;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [&]() { return stop_ || has_job_; });
            if (stop_ && !has_job_) return;
            L = job_layer_;
            slot = job_slot_;
            nsubs = job_nsubs_;
        }

        std::string err_str;
        @autoreleasepool {
            const LayerData& ld = layers_[L];
            if (nsubs > 1 && num_pieces_ == 1) {
                // Async 2-deep ANE submission: queues sub=1 onto ANE hardware ring before sub=0 finishes!
                dispatch_group_t grp = dispatch_group_create();
                std::string async_err;
                std::mutex async_mtx;
                std::string* async_err_p = &async_err;
                std::mutex* async_mtx_p = &async_mtx;
                for (int sub = 0; sub < nsubs; ++sub) {
                    const PieceCall& pc = ld.calls[slot][sub].pieces[0];
                    dispatch_group_enter(grp);
                    [(MLModel*)pc.model predictionFromFeatures:pc.provider options:pc.options
                        completionHandler:^(id<MLFeatureProvider> res, NSError* err) {
                            if (!res) {
                                std::lock_guard<std::mutex> lk(*async_mtx_p);
                                if (async_err_p->empty()) {
                                    *async_err_p = "L" + std::to_string(L) + " sub" + std::to_string(sub) + ": " +
                                                   (err ? [[err localizedDescription] UTF8String] : "unknown");
                                }
                            }
                            dispatch_group_leave(grp);
                        }];
                }
                dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);
                err_str = std::move(async_err);
            } else {
                for (int sub = 0; sub < nsubs && err_str.empty(); ++sub) {
                    for (int k = 0; k < num_pieces_; ++k) {
                        const PieceCall& pc = ld.calls[slot][sub].pieces[k];
                        NSError* err = nil;
                        id<MLFeatureProvider> res =
                            [(MLModel*)pc.model predictionFromFeatures:pc.provider options:pc.options error:&err];
                        if (!res) {
                            err_str = "L" + std::to_string(L) + " p" + std::to_string(k) + ": " +
                                      (err ? [[err localizedDescription] UTF8String] : "unknown");
                            break;
                        }
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            job_error_ = std::move(err_str);
            has_job_ = false;
        }
        cv_done_.notify_one();
    }
}

} // namespace mi
