#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_ctx.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace mi {

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("No Metal GPU device available");
    }
    queue_ = [device_ newCommandQueue];
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
    if (libraries_.count(tag)) return;

    std::string src = read_file_to_string(path);
    NSString* ns_src = [NSString stringWithUTF8String:src.c_str()];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_2;

    NSError* err = nil;
    id<MTLLibrary> lib = [(id<MTLDevice>)device_ newLibraryWithSource:ns_src options:opts error:&err];
    if (!lib) {
        std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown error";
        throw std::runtime_error("Metal shader compile failed (" + path + "): " + msg);
    }
    libraries_[tag] = lib;
}

id MetalContext::pipeline(const std::string& tag, const std::string& func_name) {
    std::string key = tag + "::" + func_name;
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;

    auto lit = libraries_.find(tag);
    if (lit == libraries_.end()) {
        throw std::runtime_error("Shader library not loaded: " + tag);
    }
    id<MTLLibrary> lib = lit->second;
    NSString* ns_name = [NSString stringWithUTF8String:func_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:ns_name];
    if (!fn) {
        throw std::runtime_error("Metal function not found: " + key);
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [(id<MTLDevice>)device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown error";
        throw std::runtime_error("Pipeline creation failed (" + key + "): " + msg);
    }
    pipelines_[key] = pso;
    return pso;
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
