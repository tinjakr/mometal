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
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <objc/message.h>
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
    if (!ifs) return false;
    uint32_t hdr_len = 0;
    if (ver[0] == 1) {
        uint16_t l16 = 0;
        ifs.read(reinterpret_cast<char*>(&l16), 2);
        hdr_len = l16;
    } else {
        ifs.read(reinterpret_cast<char*>(&hdr_len), 4);
    }
    if (!ifs || hdr_len > (1u << 20)) return false;
    std::string hdr(hdr_len, '\0');
    ifs.read(hdr.data(), hdr_len);
    if (!ifs) return false;

    bool is_f16 = (hdr.find("<f2") != std::string::npos || hdr.find("|f2") != std::string::npos);
    bool is_f32 = (hdr.find("<f4") != std::string::npos || hdr.find("|f4") != std::string::npos);
    if (!is_f16 && !is_f32) return false;

    auto cur = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    auto end = ifs.tellg();
    if (cur < 0 || end < cur) return false;
    size_t rem = static_cast<size_t>(end - cur);
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

std::string detect_tag(const std::string& dir, const std::string& ane_arch, int& out_pieces) {
    namespace fs = std::filesystem;
    // On M1/M2/M3 (h13/h14/h15), linear int4 (hf4) is promoted to INT8 by ANECompiler;
    // prefer universal 4-bit palettized (lut4) first on h13-h15, and native W4A8 (hf4) on M4+/h16+.
    const bool prefer_lut4 = (!ane_arch.empty() &&
                              (ane_arch.rfind("h13", 0) == 0 ||
                               ane_arch.rfind("h14", 0) == 0 ||
                               ane_arch.rfind("h15", 0) == 0));
    const char* cands_m4[] = {"hf4", "lut4"};
    const char* cands_m1[] = {"lut4", "hf4"};
    const char* const* cands = prefer_lut4 ? cands_m1 : cands_m4;
    for (int i = 0; i < 2; ++i) {
        const char* t = cands[i];
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

std::string AneMlpEngine::detect_ane_arch() {
    static std::string cached;
    static bool initialized = false;
    if (initialized) return cached;
    initialized = true;
    @autoreleasepool {
        void* handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_LAZY);
        if (!handle) return cached;
        Class cls = NSClassFromString(@"_ANEDeviceInfo");
        if (!cls) return cached;
        SEL sel = NSSelectorFromString(@"aneArchitectureType");
        if (![cls respondsToSelector:sel]) return cached;
        using Fn = NSString* (*)(id, SEL);
        NSString* arch = reinterpret_cast<Fn>(objc_msgSend)(cls, sel);
        if (arch && [arch isKindOfClass:[NSString class]]) {
            cached = std::string([arch UTF8String]);
        }
    }
    return cached;
}

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

void AneMlpEngine::cleanup_iosurfaces() noexcept {
    for (int s = 0; s < MAX_SLOTS; ++s) {
        for (int sub = 0; sub < MAX_SUBS; ++sub) {
            destroy_iosurface_io(in_io_[s][sub]);
            for (int k = 0; k < MAX_PIECES; ++k) {
                destroy_iosurface_io(out_io_[s][sub][k]);
            }
        }
    }
}

AneMlpEngine::AneMlpEngine(MetalContext& ctx, const std::string& models_dir, int num_layers,
                           int hidden_width, const std::string& tag, bool defer_warmup)
    : ctx_(ctx), num_layers_(num_layers), hidden_width_(hidden_width),
      packed_width_(((hidden_width + 64) / 64) * 64),
      ane_arch_(detect_ane_arch()) {
    auto t0 = std::chrono::steady_clock::now();

    for (int s = 0; s < MAX_SLOTS; ++s) {
        grp_[s] = dispatch_group_create();
        slot_rem_[s].store(0, std::memory_order_relaxed);
        slot_done_[s].store(true, std::memory_order_relaxed);
    }

    if (!std::filesystem::is_directory(models_dir)) {
        throw std::runtime_error("ANE models directory not found: " + models_dir);
    }

    if (tag == "auto" || tag.empty()) {
        tag_ = detect_tag(models_dir, ane_arch_, num_pieces_);
    } else {
        tag_ = tag;
        std::string p1 = models_dir + "/L0_mlp_p1_" + tag_ + ".mlmodelc";
        num_pieces_ = std::filesystem::is_directory(p1) ? 2 : 1;
    }
    if (tag_.empty()) {
        throw std::runtime_error("No L0_mlp_p0_*.mlmodelc found in " + models_dir);
    }

    // Create zero-copy IOSurface IO slots (2 halves x 2 sub-blocks = 4 input + 4*pieces output)
    try {
        for (int s = 0; s < MAX_SLOTS; ++s) {
            for (int sub = 0; sub < MAX_SUBS; ++sub) {
                in_io_[s][sub] = create_iosurface_io(block_rows_, packed_width_);
                for (int k = 0; k < num_pieces_; ++k) {
                    out_io_[s][sub][k] = create_iosurface_io(block_rows_, hidden_width_);
                }
            }
        }
    } catch (...) {
        cleanup_iosurfaces();
        throw;
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
                    try {
                        LayerData& ld = layers_[L];

                        // Optional inv1 SmoothQuant vector
                        std::string inv1_path = models_dir + "/pieces_inv1_" + tag_ + "_L" + std::to_string(L) + ".npy";
                        std::vector<float> inv1_vec;
                        if (load_npy_f32(inv1_path, inv1_vec) && static_cast<int>(inv1_vec.size()) == hidden_width_) {
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
                            if (!iname || !oname) {
                                std::lock_guard<std::mutex> lk(err_mtx);
                                if (load_err.empty()) {
                                    load_err = "Malformed ANE model (missing input/output description): " + path;
                                }
                                return;
                            }

                            // Pre-bind providers & outputBackings for all (slot, sub) pairs (ZERO runtime allocs!)
                            for (int s = 0; s < MAX_SLOTS; ++s) {
                                for (int sub = 0; sub < MAX_SUBS; ++sub) {
                                    MLFeatureValue* fv = [MLFeatureValue featureValueWithMultiArray:(MLMultiArray*)in_io_[s][sub].multi_array];
                                    NSError* perr = nil;
                                    MLDictionaryFeatureProvider* prov = fv
                                        ? [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{iname : fv} error:&perr]
                                        : nil;
                                    if (!fv || !prov) {
                                        std::lock_guard<std::mutex> lk(err_mtx);
                                        if (load_err.empty()) {
                                            load_err = "Failed creating CoreML feature provider for " + path + ": " +
                                                       (perr ? [[perr localizedDescription] UTF8String] : "nil provider");
                                        }
                                        return;
                                    }
                                    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
                                    opts.outputBackings = @{oname : (MLMultiArray*)out_io_[s][sub][k].multi_array};
                                    ld.calls[s][sub].pieces[k] = PieceCall{m, prov, opts};
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lk(err_mtx);
                        if (load_err.empty()) {
                            load_err = "Failed loading ANE layer L" + std::to_string(L) + ": " + e.what();
                        }
                        return;
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
        layers_.clear();
        cleanup_iosurfaces();
        throw std::runtime_error(load_err);
    }

    ready_ = true;

    if (!defer_warmup) {
        start_warmup_async();
        try {
            wait_warmup();
        } catch (...) {
            layers_.clear();
            cleanup_iosurfaces();
            throw;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    load_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

AneMlpEngine::~AneMlpEngine() {
    if (warmup_thread_.joinable()) {
        warmup_thread_.join();
    }
    warmup_started_ = false;
    drain_all_slots();
    for (int s = 0; s < MAX_SLOTS; ++s) {
        grp_[s] = nil;
    }
    layers_.clear();
    cleanup_iosurfaces();
}

void AneMlpEngine::run_warmup_4deep() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    std::string werr;
    std::mutex wmtx;
    std::string* werr_p = &werr;
    std::mutex* wmtx_p = &wmtx;

    // 4-deep async warmup cycling across all 4 (slot, sub) IOSurfaces:
    // keeps ANE hardware command ring saturated with ZERO IOSurface races!
    @autoreleasepool {
        constexpr int kLanes = MAX_SLOTS * MAX_SUBS; // 4
        dispatch_group_t lane_grp[kLanes];
        for (int i = 0; i < kLanes; ++i) lane_grp[i] = dispatch_group_create();

        for (int L = 0; L < num_layers_; ++L) {
            int lane = L & (kLanes - 1);
            int s = lane >> 1;
            int sub = lane & 1;
            dispatch_group_wait(lane_grp[lane], DISPATCH_TIME_FOREVER);
            {
                std::lock_guard<std::mutex> lk(wmtx);
                if (!werr.empty()) break;
            }
            dispatch_group_t grp = lane_grp[lane];
            for (int k = 0; k < num_pieces_; ++k) {
                const PieceCall& pc = layers_[L].calls[s][sub].pieces[k];
                dispatch_group_enter(grp);
                [(MLModel*)pc.model predictionFromFeatures:pc.provider options:pc.options
                    completionHandler:^(id<MLFeatureProvider> res, NSError* err) {
                        if (!res) {
                            std::lock_guard<std::mutex> lk(*wmtx_p);
                            if (werr_p->empty()) {
                                *werr_p = "ANE warmup prediction failed on L" + std::to_string(L) + ": " +
                                          (err ? [[err localizedDescription] UTF8String] : "unknown");
                            }
                        }
                        dispatch_group_leave(grp);
                    }];
            }
        }
        for (int i = 0; i < kLanes; ++i) {
            dispatch_group_wait(lane_grp[i], DISPATCH_TIME_FOREVER);
        }
    }
    warmup_err_ = std::move(werr);
}

void AneMlpEngine::start_warmup_async() {
    if (warmup_started_ || !ready_) return;
    warmup_err_.clear();
    warmup_started_ = true;
    warmup_thread_ = std::thread(&AneMlpEngine::run_warmup_4deep, this);
}

void AneMlpEngine::wait_warmup() {
    if (warmup_thread_.joinable()) {
        warmup_thread_.join();
    }
    warmup_started_ = false;
    if (!warmup_err_.empty()) {
        ready_ = false;
        std::string err = std::move(warmup_err_);
        warmup_err_.clear();
        throw std::runtime_error(err);
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

void AneMlpEngine::submit_slot(int layer, int slot, int n_subs) {
    if (warmup_started_) wait_warmup();
    const int total_calls = n_subs * num_pieces_;
    last_slot_ = slot;
    if (total_calls <= 0) {
        slot_done_[slot].store(true, std::memory_order_release);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(err_mtx_);
        slot_err_[slot].clear();
    }
    slot_rem_[slot].store(total_calls, std::memory_order_release);
    slot_done_[slot].store(false, std::memory_order_release);

    dispatch_group_t grp = (dispatch_group_t)grp_[slot];
    const LayerData& ld = layers_[layer];
    std::atomic<int>* rem_p = &slot_rem_[slot];
    std::atomic<bool>* done_p = &slot_done_[slot];
    std::mutex* emtx_p = &err_mtx_;
    std::string* serr_p = &slot_err_[slot];

    @autoreleasepool {
        for (int sub = 0; sub < n_subs; ++sub) {
            for (int k = 0; k < num_pieces_; ++k) {
                const PieceCall& pc = ld.calls[slot][sub].pieces[k];
                dispatch_group_enter(grp);
                [(MLModel*)pc.model predictionFromFeatures:pc.provider options:pc.options
                    completionHandler:^(id<MLFeatureProvider> res, NSError* err) {
                        if (!res) {
                            std::lock_guard<std::mutex> lk(*emtx_p);
                            if (serr_p->empty()) {
                                *serr_p = "L" + std::to_string(layer) + " sub" + std::to_string(sub) +
                                          " p" + std::to_string(k) + ": " +
                                          (err ? [[err localizedDescription] UTF8String] : "unknown");
                            }
                        }
                        if (rem_p->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            done_p->store(true, std::memory_order_release);
                        }
                        dispatch_group_leave(grp);
                    }];
            }
        }
    }
}

bool AneMlpEngine::is_slot_done(int slot) const {
    return slot_done_[slot].load(std::memory_order_acquire);
}

void AneMlpEngine::wait_slot(int slot) {
    dispatch_group_wait((dispatch_group_t)grp_[slot], DISPATCH_TIME_FOREVER);
    std::string err;
    {
        std::lock_guard<std::mutex> lk(err_mtx_);
        if (!slot_err_[slot].empty()) {
            err = std::move(slot_err_[slot]);
            slot_err_[slot].clear();
        }
    }
    if (!err.empty()) {
        throw std::runtime_error("ANE MLP prediction failed: " + err);
    }
}

void AneMlpEngine::drain_all_slots() noexcept {
    for (int s = 0; s < MAX_SLOTS; ++s) {
        if (grp_[s] != nil) {
            dispatch_group_wait((dispatch_group_t)grp_[s], DISPATCH_TIME_FOREVER);
        }
        try {
            std::lock_guard<std::mutex> lk(err_mtx_);
            slot_err_[s].clear();
        } catch (...) {
        }
        slot_rem_[s].store(0, std::memory_order_release);
        slot_done_[s].store(true, std::memory_order_release);
    }
}

void AneMlpEngine::submit(int layer, int slot, int n_subs) {
    submit_slot(layer, slot, n_subs);
}

void AneMlpEngine::wait() {
    wait_slot(last_slot_);
}

} // namespace mi
