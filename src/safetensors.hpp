#pragma once

#include <cstddef>
#include <cstdint>
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
    std::vector<MappedSubRange> sub_ranges; // selective page-aligned Metal sub-buffers when filtering
};

class SafetensorsStore {
public:
    SafetensorsStore() = default;
    ~SafetensorsStore();

    SafetensorsStore(const SafetensorsStore&) = delete;
    SafetensorsStore& operator=(const SafetensorsStore&) = delete;

    // Load a model directory (reads model.safetensors.index.json if present, else all *.safetensors).
    // Parallelizes multi-shard mmap + Metal mapping; skips tensors matching any skip_prefix.
    void load_dir(id device, const std::string& dir, const std::vector<std::string>& skip_prefixes = {},
                  bool prefetch_now = true, bool allow_unaligned_q4 = false,
                  bool allow_unaligned_bf16 = false);

    // Load ONLY tensors matching at least one include_prefix (zero mmap/wiring of non-matching shards/tensors).
    void load_dir_only(id device, const std::string& dir, const std::vector<std::string>& include_prefixes,
                       bool prefetch_now = true, bool allow_unaligned_q4 = false,
                       bool allow_unaligned_bf16 = false);

    // Load a single .safetensors file (e.g., mtp.safetensors)
    void load_file(id device, const std::string& path, const std::string& key_prefix = "",
                   bool prefetch_now = true, const std::vector<std::string>& skip_prefixes = {},
                   bool allow_unaligned_q4 = false, bool allow_unaligned_bf16 = false);

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

// Zero-Metal-wiring CPU-only read-only mmap reader for Q4_g64 embedding tables (e.g. embed_tokens).
// Never creates an MTLBuffer or calls MADV_WILLNEED, so untouched vocab rows stay un-faulted & unwired!
class CpuMmapQ4Weight {
public:
    CpuMmapQ4Weight() = default;
    ~CpuMmapQ4Weight();

    CpuMmapQ4Weight(const CpuMmapQ4Weight&) = delete;
    CpuMmapQ4Weight& operator=(const CpuMmapQ4Weight&) = delete;

    // Open <prefix>.weight / .scales / .biases from model_dir (index.json or model.safetensors)
    bool open(const std::string& model_dir, const std::string& prefix);
    void close();

    bool valid() const { return w_ptr_ != nullptr; }
    uint32_t out_dim() const { return out_dim_; }
    uint32_t in_dim() const { return in_dim_; }

    // Dequantize M requested rows into contiguous float32 out_f32[M * in_dim] using ARM NEON
    void gather_f32(const int* token_ids, int M, float* out_f32) const;

private:
    struct RawMap {
        std::string path;
        void* ptr = nullptr;
        size_t size = 0;
    };
    std::vector<RawMap> maps_;
    const uint8_t*  w_ptr_ = nullptr;
    const uint8_t*  s_ptr_ = nullptr;
    const uint8_t*  b_ptr_ = nullptr;
    DType s_dtype_ = DType::BF16;
    DType b_dtype_ = DType::BF16;
    uint32_t out_dim_ = 0;
    uint32_t in_dim_ = 0;
    uint32_t groups_per_row_ = 0;
};

DType parse_dtype(const std::string& s);
size_t dtype_size(DType d);
std::string read_file_to_string(const std::string& path);

} // namespace mi
