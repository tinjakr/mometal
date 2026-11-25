#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ctx.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mach-o/dyld.h>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mi {

namespace {

std::string executable_dir() {
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    if (sz == 0) return "";
    std::string buf(sz, '\0');
    if (_NSGetExecutablePath(buf.data(), &sz) != 0) return "";
    std::error_code ec;
    fs::path p = fs::weakly_canonical(fs::path(buf.c_str()), ec);
    if (ec) p = fs::path(buf.c_str());
    return p.parent_path().string();
}

void estimate_chip_specs(const std::string& dev_name, bool apple9, bool apple8, bool apple7,
                         GpuTier& out_tier, int& out_cores, double& out_bw) {
    std::string lo = dev_name;
    for (char& c : lo) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    const bool is_ultra = (lo.find("ultra") != std::string::npos);
    const bool is_max   = (lo.find("max")   != std::string::npos);
    const bool is_pro   = (lo.find("pro")   != std::string::npos);
    const bool is_m5    = (lo.find("m5")    != std::string::npos);
    const bool is_m4    = (lo.find("m4")    != std::string::npos);
    const bool is_m3    = (lo.find("m3")    != std::string::npos);
    const bool is_m2    = (lo.find("m2")    != std::string::npos);
    const bool is_m1    = (lo.find("m1")    != std::string::npos);

    if (is_m5)        out_tier = GpuTier::APPLE10;
    else if (apple9)  out_tier = GpuTier::APPLE9;
    else if (apple8)  out_tier = GpuTier::APPLE8;
    else if (apple7)  out_tier = GpuTier::APPLE7;
    else              out_tier = GpuTier::UNKNOWN;

    if (is_m5) {
        if (is_ultra)    { out_cores = 80; out_bw = 1000.0; }
        else if (is_max) { out_cores = 40; out_bw = 600.0; }
        else if (is_pro) { out_cores = 20; out_bw = 300.0; }
        else             { out_cores = 10; out_bw = 153.0; }
    } else if (is_m4) {
        if (is_ultra)    { out_cores = 80; out_bw = 1000.0; }
        else if (is_max) { out_cores = 40; out_bw = 546.0; }
        else if (is_pro) { out_cores = 20; out_bw = 273.0; }
        else             { out_cores = 10; out_bw = 120.0; }
    } else if (is_m3) {
        if (is_ultra)    { out_cores = 80; out_bw = 819.0; }
        else if (is_max) { out_cores = 40; out_bw = 400.0; }
        else if (is_pro) { out_cores = 18; out_bw = 150.0; }
        else             { out_cores = 10; out_bw = 100.0; }
    } else if (is_m2) {
        if (is_ultra)    { out_cores = 76; out_bw = 800.0; }
        else if (is_max) { out_cores = 38; out_bw = 400.0; }
        else if (is_pro) { out_cores = 19; out_bw = 200.0; }
        else             { out_cores = 10; out_bw = 100.0; }
    } else if (is_m1) {
        if (is_ultra)    { out_cores = 64; out_bw = 800.0; }
        else if (is_max) { out_cores = 32; out_bw = 400.0; }
        else if (is_pro) { out_cores = 16; out_bw = 200.0; }
        else             { out_cores = 8;  out_bw = 68.0; }
    } else {
        // Conservative fallback for unrecognized Apple Silicon SKUs
        out_cores = is_ultra ? 64 : (is_max ? 32 : (is_pro ? 16 : 10));
        out_bw    = is_ultra ? 800.0 : (is_max ? 400.0 : (is_pro ? 200.0 : 100.0));
    }
}

} // namespace

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal GPU device available");
    }

    id<MTLDevice> dev = (id<MTLDevice>)device_;
    const bool apple7 = [dev supportsFamily:MTLGPUFamilyApple7];
    const bool apple8 = [dev supportsFamily:MTLGPUFamilyApple8];
    const bool apple9 = [dev supportsFamily:MTLGPUFamilyApple9];
    if (!apple7) {
        std::string name = device_name();
        throw std::runtime_error(
            "Unsupported Metal GPU '" + name + "': mometal requires Apple Silicon (MTLGPUFamilyApple7 / M1 or newer) "
            "with Metal 3.0 simdgroup_matrix support.");
    }
    supports_apple9_ = apple9;
    estimate_chip_specs(device_name(), apple9, apple8, apple7, gpu_tier_, est_gpu_cores_, est_bw_gbs_);

    queue_ = [dev newCommandQueue];
    if (!queue_) {
        throw std::runtime_error("Failed to create MTLCommandQueue");
    }
}

MetalContext::~MetalContext() {
    pipelines_.clear();
    libraries_.clear();
    queue_ = nil;
    device_ = nil;
}

std::string MetalContext::device_name() const {
    return std::string([[(id<MTLDevice>)device_ name] UTF8String]);
}

uint64_t MetalContext::recommended_max_working_set_size() const {
    return [(id<MTLDevice>)device_ recommendedMaxWorkingSetSize];
}

void MetalContext::load_shader_file(const std::string& tag, const std::string& path) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (libraries_.count(tag)) return;
    }

    id<MTLDevice> dev = (id<MTLDevice>)device_;
    bool use_m31 = false;
    if (tag == "vision" && supports_apple9_) {
        if (@available(macOS 14.0, *)) {
            use_m31 = true;
        }
    }

    // Fast path: load precompiled .metallib ONLY when present AND strictly newer than (or equal to) source .metal
    id<MTLLibrary> lib = nil;
    {
        std::error_code ec_src;
        auto src_mtime = fs::last_write_time(path, ec_src);
        if (!ec_src) {
            std::string lib_name = tag + (use_m31 ? "_m31.metallib" : "_m30.metallib");
            std::vector<std::string> cands;
            std::string exe_d = executable_dir();
            if (!exe_d.empty()) cands.push_back(exe_d + "/" + lib_name);
#ifdef MOMETAL_METALLIB_DIR
            std::string build_d = MOMETAL_METALLIB_DIR;
            if (!build_d.empty() && build_d != exe_d) {
                cands.push_back(build_d + "/" + lib_name);
            }
#endif
            for (const std::string& cand : cands) {
                std::error_code ec_c;
                if (!fs::is_regular_file(cand, ec_c) || ec_c) continue;
                auto lib_mtime = fs::last_write_time(cand, ec_c);
                if (ec_c || lib_mtime < src_mtime) continue; // Never load stale .metallib!
                NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cand.c_str()]];
                NSError* lerr = nil;
                lib = [dev newLibraryWithURL:url error:&lerr];
                if (lib) break;
            }
        }
    }

    if (!lib) {
        std::string src = read_file_to_string(path);
        NSString* ns_src = [NSString stringWithUTF8String:src.c_str()];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        // Metal 3.0 across M1-M5 for all LLM kernels; Metal 3.1 bfloat8x8 for vision on Apple9+ (M3/M4/M5)
        if (use_m31) {
            if (@available(macOS 14.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_1;
            } else {
                opts.languageVersion = MTLLanguageVersion3_0;
            }
        } else {
            opts.languageVersion = MTLLanguageVersion3_0;
        }

        NSError* err = nil;
        lib = [dev newLibraryWithSource:ns_src options:opts error:&err];
        if (!lib) {
            std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown error";
            throw std::runtime_error("Metal shader compile failed (" + path + "): " + msg);
        }
    }

    std::lock_guard<std::mutex> lk(mtx_);
    libraries_.emplace(tag, lib);
}

id MetalContext::pipeline(const std::string& tag, const std::string& func_name) {
    std::string key = tag + "::" + func_name;
    id<MTLLibrary> lib = nil;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;
        auto lit = libraries_.find(tag);
        if (lit == libraries_.end()) {
            throw std::runtime_error("Shader library not loaded: " + tag);
        }
        lib = lit->second;
    }

    NSString* ns_name = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:ns_name];
    if (!fn || fn.functionConstantsDictionary.count > 0) {
        // Apple Metal returns either nil or an unspecialized MTLFunction (non-empty
        // functionConstantsDictionary) from newFunctionWithName: when a kernel uses
        // [[function_constant]]; specialize with empty MTLFunctionConstantValues so defaults apply.
        MTLFunctionConstantValues* empty_fc = [[MTLFunctionConstantValues alloc] init];
        NSError* ferr = nil;
        fn = [lib newFunctionWithName:ns_name constantValues:empty_fc error:&ferr];
        if (!fn) {
            std::string msg = ferr ? (" (" + std::string([[ferr localizedDescription] UTF8String]) + ")") : "";
            throw std::runtime_error("Metal function not found: " + key + msg);
        }
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso = [(id<MTLDevice>)device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown error";
        throw std::runtime_error("Pipeline creation failed (" + key + "): " + msg);
    }

    std::lock_guard<std::mutex> lk(mtx_);
    return pipelines_.emplace(key, pso).first->second;
}

id MetalContext::pipeline_u32_constants(const std::string& tag, const std::string& func_name,
                                        const std::string& cache_key,
                                        const std::vector<std::pair<uint32_t, uint32_t>>& consts) {
    std::string key = tag + "::" + func_name + "#" + cache_key;
    id<MTLLibrary> lib = nil;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) return it->second;
        auto lit = libraries_.find(tag);
        if (lit == libraries_.end()) {
            throw std::runtime_error("Shader library not loaded: " + tag);
        }
        lib = lit->second;
    }

    NSString* ns_name = [NSString stringWithUTF8String:func_name.c_str()];
    MTLFunctionConstantValues* fc = [[MTLFunctionConstantValues alloc] init];
    for (const auto& kv : consts) {
        uint32_t val = kv.second;
        [fc setConstantValue:&val type:MTLDataTypeUInt atIndex:kv.first];
    }
    NSError* ferr = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:ns_name constantValues:fc error:&ferr];
    if (!fn) {
        std::string msg = ferr ? std::string([[ferr localizedDescription] UTF8String]) : "unknown error";
        throw std::runtime_error("Metal specialized function failed (" + key + "): " + msg);
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso = [(id<MTLDevice>)device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown error";
        throw std::runtime_error("Specialized pipeline creation failed (" + key + "): " + msg);
    }

    std::lock_guard<std::mutex> lk(mtx_);
    return pipelines_.emplace(key, pso).first->second;
}

id MetalContext::alloc_private(size_t bytes) {
    size_t sz = std::max<size_t>(bytes, 16);
    id<MTLBuffer> buf = [(id<MTLDevice>)device_ newBufferWithLength:sz options:MTLResourceStorageModePrivate];
    if (!buf) throw std::runtime_error("Failed to alloc private MTLBuffer of " + std::to_string(sz) + " bytes");
    return buf;
}

id MetalContext::alloc_shared(size_t bytes) {
    size_t sz = std::max<size_t>(bytes, 16);
    id<MTLBuffer> buf = [(id<MTLDevice>)device_ newBufferWithLength:sz options:MTLResourceStorageModeShared];
    if (!buf) throw std::runtime_error("Failed to alloc shared MTLBuffer of " + std::to_string(sz) + " bytes");
    std::memset([buf contents], 0, sz);
    return buf;
}

id MetalContext::alloc_shared_copy(const void* data, size_t bytes) {
    size_t sz = std::max<size_t>(bytes, 16);
    id<MTLBuffer> buf = [(id<MTLDevice>)device_ newBufferWithLength:sz options:MTLResourceStorageModeShared];
    if (!buf) throw std::runtime_error("Failed to alloc shared MTLBuffer of " + std::to_string(sz) + " bytes");
    std::memcpy([buf contents], data, bytes);
    return buf;
}

} // namespace mi
