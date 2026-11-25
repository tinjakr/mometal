#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace mi {

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

    // Compile MSL shader file (runtime compilation with MTLMathModeFast)
    void load_shader_file(const std::string& tag, const std::string& path);

    // Get or create compute pipeline state
    id pipeline(const std::string& tag, const std::string& func_name);

    // Allocate GPU buffers
    id alloc_private(size_t bytes);
    id alloc_shared(size_t bytes);
    id alloc_shared_copy(const void* data, size_t bytes);

private:
    id device_ = nil;
    id queue_ = nil;
    std::unordered_map<std::string, id> libraries_;
    std::unordered_map<std::string, id> pipelines_;
};

} // namespace mi
