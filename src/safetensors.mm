#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "safetensors.hpp"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fs = std::filesystem;

namespace mi {

namespace {

struct ShardLoadResult {
    bool has_shard = false;
    MappedShard shard;
    std::vector<id> aligned_copies;
    std::vector<std::pair<std::string, TensorView>> tensors;
    size_t mapped_bytes = 0;
};

bool matches_any_prefix(const std::string& key, const std::vector<std::string>& prefixes) {
    for (const auto& p : prefixes) {
        if (key.rfind(p, 0) == 0) return true;
    }
    return false;
}

ShardLoadResult load_shard_isolated(id device_id, const std::string& path, const std::string& key_prefix,
                                    bool prefetch_now,
                                    const std::vector<std::string>& skip_prefixes,
                                    const std::vector<std::string>& include_prefixes) {
    ShardLoadResult out;
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

    void* ptr = mmap(nullptr, aligned_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for " + path);
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, ptr, sizeof(uint64_t));
    if (8 + header_len > file_size) {
        munmap(ptr, aligned_size);
        throw std::runtime_error("Corrupt safetensors header in " + path);
    }
    std::string_view hdr_sv(reinterpret_cast<const char*>(ptr) + 8, static_cast<size_t>(header_len));
    Json hdr = parse_json(hdr_sv);
    size_t data_base = 8 + static_cast<size_t>(header_len);

    const bool selective = !skip_prefixes.empty() || !include_prefixes.empty();
    auto keep_key = [&](const std::string& full_key) -> bool {
        if (!include_prefixes.empty() && !matches_any_prefix(full_key, include_prefixes)) return false;
        if (!skip_prefixes.empty() && matches_any_prefix(full_key, skip_prefixes)) return false;
        return true;
    };

    if (selective) {
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
            if (!keep_key(full_key)) continue;
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
            return out;
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

        out.shard.path = path;
        out.shard.mmap_ptr = ptr;
        out.shard.file_size = file_size;
        out.shard.mapped_size = aligned_size;
        out.shard.mtl_buffer = nil;

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
                for (auto& sr : out.shard.sub_ranges) sr.mtl_buffer = nil;
                munmap(ptr, aligned_size);
                throw std::runtime_error("Metal newBufferWithBytesNoCopy sub-range failed for " + path);
            }
            out.shard.sub_ranges.push_back(MappedSubRange{r_lo, r_len, sub_buf});
            out.mapped_bytes += r_len;
        }

        out.tensors.reserve(kept.size());
        for (auto& pt : kept) {
            size_t lo = 0, hi = out.shard.sub_ranges.size();
            while (lo + 1 < hi) {
                size_t mid = (lo + hi) >> 1;
                if (out.shard.sub_ranges[mid].offset <= pt.raw_off) lo = mid;
                else hi = mid;
            }
            const MappedSubRange& sr = out.shard.sub_ranges[lo];
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
                out.aligned_copies.push_back(copy_buf);
                tv.buffer = copy_buf;
                tv.offset = 0;
                tv.host_ptr = reinterpret_cast<const uint8_t*>([copy_buf contents]);
            }
            out.tensors.emplace_back(std::move(pt.full_key), std::move(tv));
        }
        out.has_shard = true;
        return out;
    }

    if (prefetch_now) {
        madvise(ptr, file_size, MADV_WILLNEED);
    }

    id<MTLBuffer> mtl_buf = [device newBufferWithBytesNoCopy:ptr
                                                      length:aligned_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!mtl_buf) {
        munmap(ptr, aligned_size);
        throw std::runtime_error("Metal newBufferWithBytesNoCopy failed for " + path +
                                 " (size=" + std::to_string(aligned_size) + ")");
    }

    out.shard.path = path;
    out.shard.mmap_ptr = ptr;
    out.shard.file_size = file_size;
    out.shard.mapped_size = aligned_size;
    out.shard.mtl_buffer = mtl_buf;
    out.mapped_bytes = file_size;

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

        size_t req_align = std::max<size_t>(4, dtype_size(tv.dtype));
        if (tv.offset % req_align != 0) {
            id<MTLBuffer> copy_buf = [device newBufferWithBytes:tv.host_ptr
                                                         length:std::max<size_t>(tv.nbytes, 16)
                                                        options:MTLResourceStorageModeShared];
            out.aligned_copies.push_back(copy_buf);
            tv.buffer = copy_buf;
            tv.offset = 0;
            tv.host_ptr = reinterpret_cast<const uint8_t*>([copy_buf contents]);
        }
        out.tensors.emplace_back(std::move(full_key), std::move(tv));
    }
    out.has_shard = true;
    return out;
}

inline float read_scalar_raw(const uint8_t* p, DType dt, size_t idx) {
    if (dt == DType::BF16) {
        uint16_t raw = 0;
        std::memcpy(&raw, p + idx * 2u, 2);
        uint32_t u = static_cast<uint32_t>(raw) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, 4);
        return f;
    } else if (dt == DType::F16) {
        __fp16 h;
        std::memcpy(&h, p + idx * 2u, 2);
        return static_cast<float>(h);
    } else {
        float f = 0.0f;
        std::memcpy(&f, p + idx * 4u, 4);
        return f;
    }
}

} // namespace

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
    ShardLoadResult r = load_shard_isolated(device_id, path, key_prefix, prefetch_now, skip_prefixes, {});
    if (!r.has_shard) return;
    shards_.push_back(std::move(r.shard));
    for (auto& c : r.aligned_copies) aligned_copies_.push_back(c);
    for (auto& kv : r.tensors) tensors_[std::move(kv.first)] = std::move(kv.second);
    total_mapped_bytes_ += r.mapped_bytes;
}

void SafetensorsStore::load_dir(id device_id, const std::string& dir, const std::vector<std::string>& skip_prefixes,
                                bool prefetch_now) {
    std::string index_path = dir + "/model.safetensors.index.json";
    std::set<std::string> shard_files;
    if (fs::exists(index_path)) {
        Json idx = parse_json(read_file_to_string(index_path));
        for (const auto& [k, v] : idx["weight_map"].as_object()) {
            if (!matches_any_prefix(k, skip_prefixes)) {
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

    std::vector<std::string> files(shard_files.begin(), shard_files.end());
    if (files.size() == 1u) {
        load_file(device_id, files[0], "", prefetch_now, skip_prefixes);
        return;
    }

    // Parallelize multi-shard mmap + madvise + Metal sub-buffer creation across shards!
    std::vector<ShardLoadResult> results(files.size());
    std::atomic<size_t> next_i{0};
    std::mutex exc_mtx;
    std::exception_ptr first_exc = nullptr;
    size_t n_thr = std::min<size_t>(4u, files.size());
    std::vector<std::thread> threads;
    threads.reserve(n_thr);

    for (size_t ti = 0; ti < n_thr; ++ti) {
        threads.emplace_back([&]() {
            @autoreleasepool {
                while (true) {
                    size_t idx = next_i.fetch_add(1);
                    if (idx >= files.size()) break;
                    try {
                        results[idx] = load_shard_isolated(device_id, files[idx], "", prefetch_now, skip_prefixes, {});
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(exc_mtx);
                        if (!first_exc) first_exc = std::current_exception();
                        break;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    if (first_exc) {
        for (auto& r : results) {
            if (r.has_shard) {
                r.shard.mtl_buffer = nil;
                for (auto& sr : r.shard.sub_ranges) sr.mtl_buffer = nil;
                if (r.shard.mmap_ptr && r.shard.mmap_ptr != MAP_FAILED) {
                    munmap(r.shard.mmap_ptr, r.shard.mapped_size);
                }
            }
        }
        std::rethrow_exception(first_exc);
    }

    for (auto& r : results) {
        if (!r.has_shard) continue;
        shards_.push_back(std::move(r.shard));
        for (auto& c : r.aligned_copies) aligned_copies_.push_back(c);
        for (auto& kv : r.tensors) tensors_[std::move(kv.first)] = std::move(kv.second);
        total_mapped_bytes_ += r.mapped_bytes;
    }
}

void SafetensorsStore::load_dir_only(id device_id, const std::string& dir,
                                     const std::vector<std::string>& include_prefixes,
                                     bool prefetch_now) {
    if (include_prefixes.empty()) return;
    std::string index_path = dir + "/model.safetensors.index.json";
    std::set<std::string> shard_files;
    if (fs::exists(index_path)) {
        Json idx = parse_json(read_file_to_string(index_path));
        for (const auto& [k, v] : idx["weight_map"].as_object()) {
            if (matches_any_prefix(k, include_prefixes)) {
                shard_files.insert(dir + "/" + v.as_string());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors") {
                shard_files.insert(entry.path().string());
            }
        }
    }

    for (const auto& sf : shard_files) {
        ShardLoadResult r = load_shard_isolated(device_id, sf, "", prefetch_now, {}, include_prefixes);
        if (!r.has_shard) continue;
        shards_.push_back(std::move(r.shard));
        for (auto& c : r.aligned_copies) aligned_copies_.push_back(c);
        for (auto& kv : r.tensors) tensors_[std::move(kv.first)] = std::move(kv.second);
        total_mapped_bytes_ += r.mapped_bytes;
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

// ---------------------------------------------------------------------------
// CpuMmapQ4Weight: Zero-Metal-wiring CPU-only Q4_g64 reader (for embed_tokens)
// ---------------------------------------------------------------------------
CpuMmapQ4Weight::~CpuMmapQ4Weight() {
    close();
}

void CpuMmapQ4Weight::close() {
    for (auto& m : maps_) {
        if (m.ptr && m.ptr != MAP_FAILED) {
            munmap(m.ptr, m.size);
        }
    }
    maps_.clear();
    w_ptr_ = nullptr;
    s_ptr_ = nullptr;
    b_ptr_ = nullptr;
    out_dim_ = 0;
    in_dim_ = 0;
    groups_per_row_ = 0;
}

bool CpuMmapQ4Weight::open(const std::string& model_dir, const std::string& prefix) {
    close();
    const std::string kw = prefix + ".weight";
    const std::string ks = prefix + ".scales";
    const std::string kb = prefix + ".biases";

    std::string fw, fs_p, fb;
    std::string index_path = model_dir + "/model.safetensors.index.json";
    if (fs::is_regular_file(index_path)) {
        Json idx = parse_json(read_file_to_string(index_path));
        const auto& wm = idx["weight_map"].as_object();
        auto iw = wm.find(kw), is = wm.find(ks), ib = wm.find(kb);
        if (iw == wm.end() || is == wm.end() || ib == wm.end()) return false;
        fw   = model_dir + "/" + iw->second.as_string();
        fs_p = model_dir + "/" + is->second.as_string();
        fb   = model_dir + "/" + ib->second.as_string();
    } else {
        std::string single = model_dir + "/model.safetensors";
        if (!fs::is_regular_file(single)) return false;
        fw = fs_p = fb = single;
    }

    auto get_map = [&](const std::string& p) -> const RawMap* {
        for (const auto& m : maps_) if (m.path == p) return &m;
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) return nullptr;
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 16) { ::close(fd); return nullptr; }
        size_t sz = static_cast<size_t>(st.st_size);
        void* ptr = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) ptr = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (ptr == MAP_FAILED) return nullptr;
        maps_.push_back(RawMap{p, ptr, sz});
        return &maps_.back();
    };

    const RawMap* mw = get_map(fw);
    const RawMap* ms = get_map(fs_p);
    const RawMap* mb = get_map(fb);
    if (!mw || !ms || !mb) { close(); return false; }

    auto locate = [](const RawMap* rm, const std::string& key,
                     DType& dt, std::vector<int64_t>& shape, const uint8_t*& ptr) -> bool {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(rm->ptr);
        uint64_t hlen = 0;
        std::memcpy(&hlen, base, 8);
        if (8u + hlen > rm->size) return false;
        Json hdr = parse_json(std::string_view(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(hlen)));
        const auto& obj = hdr.as_object();
        auto it = obj.find(key);
        if (it == obj.end()) return false;
        dt = parse_dtype(it->second["dtype"].as_string());
        shape.clear();
        for (const auto& d : it->second["shape"].as_array()) shape.push_back(d.as_int());
        size_t off0 = static_cast<size_t>(it->second["data_offsets"].as_array()[0].as_int());
        ptr = base + 8u + static_cast<size_t>(hlen) + off0;
        return true;
    };

    DType wdt = DType::UNKNOWN;
    std::vector<int64_t> wsh, ssh, bsh;
    const uint8_t* wp = nullptr;
    if (!locate(mw, kw, wdt, wsh, wp) || wdt != DType::U32 || wsh.size() != 2 ||
        !locate(ms, ks, s_dtype_, ssh, s_ptr_) || ssh.size() != 2 ||
        !locate(mb, kb, b_dtype_, bsh, b_ptr_) || bsh.size() != 2) {
        close();
        return false;
    }

    w_ptr_ = reinterpret_cast<const uint32_t*>(wp);
    out_dim_ = static_cast<uint32_t>(wsh[0]);
    in_dim_ = static_cast<uint32_t>(wsh[1] * 8);
    groups_per_row_ = static_cast<uint32_t>(ssh[1]);
    if (groups_per_row_ == 0 || (in_dim_ % groups_per_row_) != 0) {
        close();
        return false;
    }
    return true;
}

void CpuMmapQ4Weight::gather_f32(const int* token_ids, int M, float* out_f32) const {
    if (!w_ptr_ || M <= 0) return;
    const uint32_t words_per_row = in_dim_ >> 3;
    const uint32_t gsize = in_dim_ / groups_per_row_;
    const uint32_t words_per_g = gsize >> 3;

    for (int m = 0; m < M; ++m) {
        uint32_t row = (token_ids[m] >= 0 && static_cast<uint32_t>(token_ids[m]) < out_dim_)
            ? static_cast<uint32_t>(token_ids[m]) : 0u;
        const uint32_t* wr = w_ptr_ + static_cast<size_t>(row) * words_per_row;
        float* dst = out_f32 + static_cast<size_t>(m) * in_dim_;

        for (uint32_t g = 0; g < groups_per_row_; ++g) {
            size_t si = static_cast<size_t>(row) * groups_per_row_ + g;
            float sc = read_scalar_raw(s_ptr_, s_dtype_, si);
            float bi = read_scalar_raw(b_ptr_, b_dtype_, si);
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t vsc = vdupq_n_f32(sc);
            float32x4_t vbi = vdupq_n_f32(bi);
            const uint32_t* wg = wr + g * words_per_g;
            float* dg = dst + g * gsize;
            for (uint32_t wi = 0; wi < words_per_g; ++wi) {
                uint32_t word = wg[wi];
                uint32_t n_lo[4] = {
                    (word >> 0)  & 0xFu, (word >> 4)  & 0xFu,
                    (word >> 8)  & 0xFu, (word >> 12) & 0xFu
                };
                uint32_t n_hi[4] = {
                    (word >> 16) & 0xFu, (word >> 20) & 0xFu,
                    (word >> 24) & 0xFu, (word >> 28) & 0xFu
                };
                float32x4_t flo = vcvtq_f32_u32(vld1q_u32(n_lo));
                float32x4_t fhi = vcvtq_f32_u32(vld1q_u32(n_hi));
                vst1q_f32(dg + wi * 8u,      vfmaq_f32(vbi, flo, vsc));
                vst1q_f32(dg + wi * 8u + 4u, vfmaq_f32(vbi, fhi, vsc));
            }
#else
            const uint32_t* wg = wr + g * words_per_g;
            float* dg = dst + g * gsize;
            for (uint32_t wi = 0; wi < words_per_g; ++wi) {
                uint32_t word = wg[wi];
                for (uint32_t b = 0; b < 8u; ++b) {
                    dg[wi * 8u + b] = static_cast<float>((word >> (4u * b)) & 0xFu) * sc + bi;
                }
            }
#endif
        }
    }
}

} // namespace mi
