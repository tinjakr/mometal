#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace mi {

enum class DType {
    F32,
    F16,
    BF16,
    U32,
    I32,
    U8,
    F8_E4M3,
    UNKNOWN
};

struct TensorView {
    id buffer = nil;      // id<MTLBuffer> (shard buffer, zero-copy mmap)
    size_t offset = 0;    // byte offset inside shard buffer
    size_t nbytes = 0;
    const uint8_t* host_ptr = nullptr;
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;

    bool valid() const { return buffer != nil; }
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t s : shape) n *= s;
        return shape.empty() ? 0 : n;
    }
};

struct MappedSubRange {
    size_t offset = 0;
    size_t length = 0;
    id mtl_buffer = nil; // id<MTLBuffer>
};

struct MappedShard {
    std::string path;
    void* mmap_ptr = nullptr;
    size_t file_size = 0;
    size_t mapped_size = 0;
    id mtl_buffer = nil; // id<MTLBuffer> (whole-shard buffer when no skip_prefixes)
    std::vector<MappedSubRange> sub_ranges; // selective page-aligned Metal sub-buffers when skip_prefixes used
};

class SafetensorsStore {
public:
    SafetensorsStore() = default;
    ~SafetensorsStore();

    SafetensorsStore(const SafetensorsStore&) = delete;
    SafetensorsStore& operator=(const SafetensorsStore&) = delete;

    // Load a model directory (reads model.safetensors.index.json if present, else all *.safetensors)
    // Skips tensors whose names match any skip_prefix (NEVER faults or Metal-maps skipped tensor pages!)
    void load_dir(id device, const std::string& dir, const std::vector<std::string>& skip_prefixes = {},
                  bool prefetch_now = true);

    // Load a single .safetensors file (e.g., mtp.safetensors)
    void load_file(id device, const std::string& path, const std::string& key_prefix = "",
                   bool prefetch_now = true, const std::vector<std::string>& skip_prefixes = {});

    bool has(const std::string& name) const;
    const TensorView& get(const std::string& name) const;
    TensorView get_optional(const std::string& name) const;

    size_t total_mapped_bytes() const { return total_mapped_bytes_; }
    size_t tensor_count() const { return tensors_.size(); }

    // Advise OS kernel to asynchronously pre-read non-evicted mapped shards back into RAM
    void prefetch() const;

    // Release resident pages of an unused mmap'd tensor back to macOS (e.g. GPU MLP weights replaced by ANE_SHARE)
    void evict_tensor(const TensorView& tv);

private:
    std::vector<MappedShard> shards_;
    std::vector<id> aligned_copies_; // fallback only if a tensor offset is misaligned
    std::unordered_map<std::string, TensorView> tensors_;
    std::vector<std::pair<uintptr_t, uintptr_t>> evicted_ranges_;
    size_t total_mapped_bytes_ = 0;
};

DType parse_dtype(const std::string& s);
size_t dtype_size(DType d);
std::string read_file_to_string(const std::string& path);

} // namespace mi
