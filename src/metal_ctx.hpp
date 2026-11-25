#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace mi {

enum class GpuTier : int {
    UNKNOWN = 0,
    APPLE7  = 7,  // M1 family
    APPLE8  = 8,  // M2 family
    APPLE9  = 9,  // M3 / M4 family
    APPLE10 = 10  // M5+ family
};

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    id device() const { return device_; }
    id queue() const { return queue_; }

    std::string device_name() const;
    uint64_t recommended_max_working_set_size() const;
    GpuTier gpu_tier() const { return gpu_tier_; }
    int estimated_gpu_cores() const { return est_gpu_cores_; }
    double estimated_bandwidth_gbs() const { return est_bw_gbs_; }
    bool supports_apple9() const { return supports_apple9_; }

    // Load precompiled .metallib if present & newer than source, else compile MSL with MTLMathModeFast
    void load_shader_file(const std::string& tag, const std::string& path);

    // Get or create compute pipeline state (falls back to empty MTLFunctionConstantValues when needed)
    id pipeline(const std::string& tag, const std::string& func_name);

    // Get or create specialized compute pipeline state with uint32 [[function_constant(index)]] values
    id pipeline_u32_constants(const std::string& tag, const std::string& func_name,
                              const std::string& cache_key,
                              const std::vector<std::pair<uint32_t, uint32_t>>& consts);

    // Allocate GPU buffers
    id alloc_private(size_t bytes);
    id alloc_shared(size_t bytes);
    id alloc_shared_copy(const void* data, size_t bytes);

private:
    id device_ = nil;
    id queue_ = nil;
    GpuTier gpu_tier_ = GpuTier::UNKNOWN;
    int est_gpu_cores_ = 16;
    double est_bw_gbs_ = 200.0;
    bool supports_apple9_ = false;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, id> libraries_;
    std::unordered_map<std::string, id> pipelines_;
};

} // namespace mi
