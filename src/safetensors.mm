#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "safetensors.hpp"
#include "json.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace mi {

std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    auto sz = ifs.tellg();
    std::string out(static_cast<size_t>(sz), '\0');
    ifs.seekg(0);
    ifs.read(out.data(), sz);
    return out;
}

DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "U32") return DType::U32;
    if (s == "I32") return DType::I32;
    if (s == "U8") return DType::U8;
    if (s == "F8_E4M3") return DType::F8_E4M3;
    return DType::UNKNOWN;
}

size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32:
        case DType::U32:
        case DType::I32: return 4;
        case DType::F16:
        case DType::BF16: return 2;
        case DType::U8:
        case DType::F8_E4M3: return 1;
        default: return 1;
    }
}

SafetensorsStore::~SafetensorsStore() {
    aligned_copies_.clear();
    for (auto& sh : shards_) {
        sh.mtl_buffer = nil;
        for (auto& sr : sh.sub_ranges) sr.mtl_buffer = nil;
        sh.sub_ranges.clear();
        if (sh.mmap_ptr && sh.mmap_ptr != MAP_FAILED) {
            munmap(sh.mmap_ptr, sh.mapped_size);
        }
    }
    shards_.clear();
}

void SafetensorsStore::load_file(id device_id, const std::string& path, const std::string& key_prefix,
                                 bool prefetch_now, const std::vector<std::string>& skip_prefixes) {
    id<MTLDevice> device = device_id;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open safetensors: " + path);

    struct stat st{};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("fstat failed: " + path);
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    size_t page_size = static_cast<size_t>(getpagesize());
    size_t aligned_size = ((file_size + page_size - 1) / page_size) * page_size;

    // On macOS, mmap(file_size) maps the full page-aligned region, zero-filling the tail of the last page.
    void* ptr = mmap(nullptr, aligned_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + path);
    }

    // Parse safetensors header FIRST (touches only the ~20 KB JSON header!)
    uint64_t header_len = 0;
    std::memcpy(&header_len, ptr, sizeof(uint64_t));
    if (8 + header_len > file_size) {
        munmap(ptr, aligned_size);
        throw std::runtime_error("Corrupt safetensors header in " + path);
    }
    std::string_view hdr_sv(reinterpret_cast<const char*>(ptr) + 8, static_cast<size_t>(header_len));
    Json hdr = parse_json(hdr_sv);
    size_t data_base = 8 + static_cast<size_t>(header_len);

    auto should_skip = [&](const std::string& full_key) -> bool {
        for (const auto& p : skip_prefixes) {
            if (full_key.rfind(p, 0) == 0) return true;
        }
        return false;
    };

    if (!skip_prefixes.empty()) {
        // Selective Metal mapping: ONLY create newBufferWithBytesNoCopy for kept tensors' page ranges!
        // Skipped tensors (e.g., 9.63 GB of MLP weights replaced by ANE-Share) are NEVER faulted or GPU-pinned!
        struct PendingTensor {
            std::string full_key;
            DType dtype;
            std::vector<int64_t> shape;
            size_t raw_off;
            size_t nbytes;
        };
        std::vector<PendingTensor> kept;
        std::vector<std::pair<size_t, size_t>> ranges;
        for (const auto& [k, info] : hdr.as_object()) {
            if (k == "__metadata__") continue;
            std::string full_key = key_prefix.empty() ? k : (key_prefix + k);
            if (should_skip(full_key)) continue;
            const auto& offs = info["data_offsets"].as_array();
            size_t s = static_cast<size_t>(offs[0].as_int());
            size_t e = static_cast<size_t>(offs[1].as_int());
            size_t raw_off = data_base + s;
            size_t nbytes = e - s;
            PendingTensor pt;
            pt.full_key = std::move(full_key);
            pt.dtype = parse_dtype(info["dtype"].as_string());
            for (const auto& dim : info["shape"].as_array()) pt.shape.push_back(dim.as_int());
            pt.raw_off = raw_off;
            pt.nbytes = nbytes;
            kept.push_back(std::move(pt));
            if (nbytes > 0) {
                size_t p_lo = raw_off & ~(page_size - 1u);
                size_t p_hi = ((raw_off + nbytes + page_size - 1u) / page_size) * page_size;
                p_hi = std::min(p_hi, aligned_size);
                ranges.emplace_back(p_lo, p_hi);
            }
        }
        if (kept.empty()) {
            munmap(ptr, aligned_size);
            return;
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<size_t, size_t>> merged;
        for (const auto& r : ranges) {
            if (merged.empty() || r.first > merged.back().second) {
                merged.push_back(r);
            } else if (r.second > merged.back().second) {
                merged.back().second = r.second;
            }
        }

        MappedShard shard;
        shard.path = path;
        shard.mmap_ptr = ptr;
        shard.file_size = file_size;
        shard.mapped_size = aligned_size;
        shard.mtl_buffer = nil;

        for (const auto& [r_lo, r_hi] : merged) {
            size_t r_len = r_hi - r_lo;
            void* r_ptr = reinterpret_cast<uint8_t*>(ptr) + r_lo;
            if (prefetch_now) {
                madvise(r_ptr, r_len, MADV_WILLNEED);
            }
            id<MTLBuffer> sub_buf = [device newBufferWithBytesNoCopy:r_ptr
                                                              length:r_len
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
            if (!sub_buf) {
                for (auto& sr : shard.sub_ranges) sr.mtl_buffer = nil;
                munmap(ptr, aligned_size);
                throw std::runtime_error("Metal newBufferWithBytesNoCopy sub-range failed for " + path);
            }
            shard.sub_ranges.push_back(MappedSubRange{r_lo, r_len, sub_buf});
            total_mapped_bytes_ += r_len;
        }

        for (auto& pt : kept) {
            // Find covering sub_range via binary search
            size_t lo = 0, hi = shard.sub_ranges.size();
            while (lo + 1 < hi) {
                size_t mid = (lo + hi) >> 1;
                if (shard.sub_ranges[mid].offset <= pt.raw_off) lo = mid;
                else hi = mid;
            }
            const MappedSubRange& sr = shard.sub_ranges[lo];
            TensorView tv;
            tv.dtype = pt.dtype;
            tv.shape = std::move(pt.shape);
            tv.offset = pt.raw_off - sr.offset;
            tv.nbytes = pt.nbytes;
            tv.host_ptr = reinterpret_cast<const uint8_t*>(ptr) + pt.raw_off;
            tv.buffer = sr.mtl_buffer;

            size_t req_align = std::max<size_t>(4, dtype_size(tv.dtype));
            if (tv.offset % req_align != 0) {
                id<MTLBuffer> copy_buf = [device newBufferWithBytes:tv.host_ptr
                                                             length:std::max<size_t>(tv.nbytes, 16)
                                                            options:MTLResourceStorageModeShared];
                aligned_copies_.push_back(copy_buf);
                tv.buffer = copy_buf;
                tv.offset = 0;
                tv.host_ptr = reinterpret_cast<const uint8_t*>([copy_buf contents]);
            }
            tensors_[std::move(pt.full_key)] = std::move(tv);
        }
        shards_.push_back(std::move(shard));
        return;
    }

    if (prefetch_now) {
        madvise(ptr, file_size, MADV_WILLNEED);
    }

    // Zero-copy Metal buffer over the full mmap'd shard.
    // Note: NoCopy requires page-aligned pointer and page-aligned length.
    id<MTLBuffer> mtl_buf = [device newBufferWithBytesNoCopy:ptr
                                                      length:aligned_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!mtl_buf) {
        munmap(ptr, aligned_size);
        throw std::runtime_error("Metal newBufferWithBytesNoCopy failed for " + path +
                                 " (size=" + std::to_string(aligned_size) + ")");
    }

    MappedShard shard;
    shard.path = path;
    shard.mmap_ptr = ptr;
    shard.file_size = file_size;
    shard.mapped_size = aligned_size;
    shard.mtl_buffer = mtl_buf;
    shards_.push_back(shard);
    total_mapped_bytes_ += file_size;

    for (const auto& [k, info] : hdr.as_object()) {
        if (k == "__metadata__") continue;
        std::string full_key = key_prefix.empty() ? k : (key_prefix + k);
        TensorView tv;
        tv.dtype = parse_dtype(info["dtype"].as_string());
        for (const auto& dim : info["shape"].as_array()) {
            tv.shape.push_back(dim.as_int());
        }
        const auto& offs = info["data_offsets"].as_array();
        size_t s = static_cast<size_t>(offs[0].as_int());
        size_t e = static_cast<size_t>(offs[1].as_int());
        tv.offset = data_base + s;
        tv.nbytes = e - s;
        tv.host_ptr = reinterpret_cast<const uint8_t*>(ptr) + tv.offset;
        tv.buffer = mtl_buf;

        // Ensure 4-byte alignment for vectorized uint32/float loads in Metal kernels
        size_t req_align = std::max<size_t>(4, dtype_size(tv.dtype));
        if (tv.offset % req_align != 0) {
            id<MTLBuffer> copy_buf = [device newBufferWithBytes:tv.host_ptr
                                                         length:std::max<size_t>(tv.nbytes, 16)
                                                        options:MTLResourceStorageModeShared];
            aligned_copies_.push_back(copy_buf);
            tv.buffer = copy_buf;
            tv.offset = 0;
            tv.host_ptr = reinterpret_cast<const uint8_t*>([copy_buf contents]);
        }

        tensors_[full_key] = std::move(tv);
    }
}

void SafetensorsStore::load_dir(id device_id, const std::string& dir, const std::vector<std::string>& skip_prefixes,
                                bool prefetch_now) {
    std::string index_path = dir + "/model.safetensors.index.json";
    std::set<std::string> shard_files;
    if (fs::exists(index_path)) {
        Json idx = parse_json(read_file_to_string(index_path));
        for (const auto& [k, v] : idx["weight_map"].as_object()) {
            bool skip = false;
            for (const auto& p : skip_prefixes) {
                if (k.rfind(p, 0) == 0) { skip = true; break; }
            }
            if (!skip) {
                shard_files.insert(dir + "/" + v.as_string());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors") {
                std::string fname = entry.path().filename().string();
                if (fname.find("mtp") != std::string::npos) continue;
                shard_files.insert(entry.path().string());
            }
        }
    }

    if (shard_files.empty()) {
        throw std::runtime_error("No safetensors shards found in: " + dir);
    }

    for (const auto& sf : shard_files) {
        load_file(device_id, sf, "", prefetch_now, skip_prefixes);
    }
}

void SafetensorsStore::prefetch() const {
    for (const auto& sh : shards_) {
        if (!sh.mmap_ptr || sh.mmap_ptr == MAP_FAILED) continue;
        if (!sh.sub_ranges.empty()) {
            for (const auto& sr : sh.sub_ranges) {
                madvise(reinterpret_cast<uint8_t*>(sh.mmap_ptr) + sr.offset, sr.length, MADV_WILLNEED);
            }
        } else if (sh.file_size > 0) {
            madvise(sh.mmap_ptr, sh.file_size, MADV_WILLNEED);
        }
    }
}

void SafetensorsStore::evict_tensor(const TensorView& tv) {
    if (!tv.host_ptr || tv.nbytes == 0) return;
    const size_t ps = static_cast<size_t>(getpagesize());
    uintptr_t lo = reinterpret_cast<uintptr_t>(tv.host_ptr);
    uintptr_t hi = lo + tv.nbytes;
    // Only evict strictly interior full pages so adjacent non-evicted tensors stay untouched
    uintptr_t p_lo = (lo + ps - 1u) & ~(ps - 1u);
    uintptr_t p_hi = hi & ~(ps - 1u);
    if (p_hi > p_lo) {
        madvise(reinterpret_cast<void*>(p_lo), p_hi - p_lo, MADV_DONTNEED);
        evicted_ranges_.emplace_back(p_lo, p_hi);
    }
}

bool SafetensorsStore::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorView& SafetensorsStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Missing tensor: " + name);
    }
    return it->second;
}

TensorView SafetensorsStore::get_optional(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return TensorView{};
    return it->second;
}

} // namespace mi
