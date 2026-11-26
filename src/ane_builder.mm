#include "ane_builder.hpp"
#include "ane_mlp.hpp"
#include "json.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fs = std::filesystem;

namespace mi {

namespace {

// ---------------------------------------------------------------------------
// Minimal CoreML MIL BlobStorage v2 writer (weights/weight.bin), reusable across layers
// ---------------------------------------------------------------------------
class BlobWriter {
public:
    static constexpr uint32_t kDTypeFP16  = 1u;
    static constexpr uint32_t kDTypeInt4  = 8u;
    static constexpr uint32_t kDTypeUInt4 = 11u;

    BlobWriter() { reset(); }

    void reserve(size_t bytes) { buf_.reserve(bytes); }

    void reset() {
        buf_.resize(64u);
        std::memset(buf_.data(), 0, 64u);
        count_ = 0;
    }

    uint64_t add_blob(uint32_t mil_dtype, const void* data, size_t nbytes) {
        align64();
        uint64_t meta_off = buf_.size();
        uint64_t data_off = meta_off + 64u;
        buf_.resize(meta_off + 64u);
        std::memset(buf_.data() + meta_off, 0, 64u);

        uint32_t sentinel = 0xDEADBEEFu;
        uint64_t sz64 = static_cast<uint64_t>(nbytes);
        std::memcpy(buf_.data() + meta_off + 0,  &sentinel, 4);
        std::memcpy(buf_.data() + meta_off + 4,  &mil_dtype, 4);
        std::memcpy(buf_.data() + meta_off + 8,  &sz64, 8);
        std::memcpy(buf_.data() + meta_off + 16, &data_off, 8);

        if (nbytes > 0) {
            size_t old_sz = buf_.size();
            buf_.resize(old_sz + nbytes);
            std::memcpy(buf_.data() + old_sz, data, nbytes);
        }
        align64();
        ++count_;
        return meta_off;
    }

    bool write_to(const std::string& path) {
        uint32_t ver = 2u;
        std::memcpy(buf_.data() + 0, &count_, 4);
        std::memcpy(buf_.data() + 4, &ver, 4);
        int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return false;
        const uint8_t* p = buf_.data();
        size_t rem = buf_.size();
        while (rem > 0) {
            ssize_t n = ::write(fd, p, rem);
            if (n <= 0) { ::close(fd); return false; }
            p += static_cast<size_t>(n);
            rem -= static_cast<size_t>(n);
        }
        return ::close(fd) == 0;
    }

private:
    void align64() {
        size_t rem = buf_.size() & 63u;
        if (rem != 0u) {
            size_t pad = 64u - rem;
            size_t old_sz = buf_.size();
            buf_.resize(old_sz + pad);
            std::memset(buf_.data() + old_sz, 0, pad);
        }
    }
    std::vector<uint8_t> buf_;
    uint32_t count_ = 0;
};

bool write_bytes_fd(const std::string& path, const void* data, size_t nbytes) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    size_t rem = nbytes;
    while (rem > 0) {
        ssize_t n = ::write(fd, p, rem);
        if (n <= 0) { ::close(fd); return false; }
        p += static_cast<size_t>(n);
        rem -= static_cast<size_t>(n);
    }
    return ::close(fd) == 0;
}

// ---------------------------------------------------------------------------
// Minimal CoreML coremldata.bin writer (iOS18 / spec v9 MILProgram archive)
// ---------------------------------------------------------------------------
void pb_varint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<uint8_t>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

void pb_bytes_field(std::vector<uint8_t>& out, uint32_t field_num, const void* data, size_t len) {
    pb_varint(out, (static_cast<uint64_t>(field_num) << 3) | 2u);
    pb_varint(out, static_cast<uint64_t>(len));
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    out.insert(out.end(), p, p + len);
}

std::vector<uint8_t> pb_feature_fp16_2d(const std::string& name, int64_t d0, int64_t d1) {
    std::vector<uint8_t> shape_packed;
    pb_varint(shape_packed, static_cast<uint64_t>(d0));
    pb_varint(shape_packed, static_cast<uint64_t>(d1));

    std::vector<uint8_t> arr;
    pb_bytes_field(arr, 1u, shape_packed.data(), shape_packed.size());
    pb_varint(arr, (2u << 3) | 0u);
    pb_varint(arr, 65552u); // ArrayDataType::FLOAT16

    std::vector<uint8_t> ftype;
    pb_bytes_field(ftype, 5u, arr.data(), arr.size());

    std::vector<uint8_t> fdesc;
    pb_bytes_field(fdesc, 1u, name.data(), name.size());
    pb_bytes_field(fdesc, 3u, ftype.data(), ftype.size());
    return fdesc;
}

std::vector<uint8_t> make_coremldata_bin(int rows, int pw, int hid) {
    std::vector<uint8_t> proto;
    auto in_fd = pb_feature_fp16_2d("x", rows, pw);
    auto out_fd = pb_feature_fp16_2d("down_proj", rows, hid);
    pb_bytes_field(proto, 1u, in_fd.data(), in_fd.size());
    pb_bytes_field(proto, 10u, out_fd.data(), out_fd.size());

    std::vector<uint8_t> buf;
    buf.reserve(83u + proto.size() + 16u);
    auto push_u32 = [&](uint32_t v) {
        uint8_t b[4]; std::memcpy(b, &v, 4); buf.insert(buf.end(), b, b + 4);
    };
    auto push_u64 = [&](uint64_t v) {
        uint8_t b[8]; std::memcpy(b, &v, 8); buf.insert(buf.end(), b, b + 8);
    };
    auto push_zeros = [&](size_t n) { buf.insert(buf.end(), n, 0u); };

    push_u32(502u);
    push_u32(9u);
    push_zeros(20u);
    push_u64(7u);
    const char* gen = "generic";
    buf.insert(buf.end(), gen, gen + 7);
    push_u32(9u);
    push_zeros(28u);
    push_u64(proto.size());
    buf.insert(buf.end(), proto.begin(), proto.end());
    push_u32(502u);
    push_zeros(12u);
    return buf;
}

bool write_npy_f16(const std::string& path, const std::vector<uint16_t>& vals) {
    std::string dict = "{'descr': '<f2', 'fortran_order': False, 'shape': (" +
                       std::to_string(vals.size()) + ",), }";
    size_t total_hdr = 10u + dict.size() + 1u;
    size_t pad = (64u - (total_hdr & 63u)) & 63u;
    dict.append(pad, ' ');
    dict.push_back('\n');

    std::vector<uint8_t> out;
    out.reserve(10u + dict.size() + vals.size() * 2u);
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    out.insert(out.end(), magic, magic + 8);
    out.push_back(static_cast<uint8_t>(hlen & 0xFFu));
    out.push_back(static_cast<uint8_t>((hlen >> 8) & 0xFFu));
    out.insert(out.end(), dict.begin(), dict.end());
    const uint8_t* vp = reinterpret_cast<const uint8_t*>(vals.data());
    out.insert(out.end(), vp, vp + vals.size() * 2u);
    return write_bytes_fd(path, out.data(), out.size());
}

// ---------------------------------------------------------------------------
// Shared read-only CPU-only mmap safetensors shard pool (zero Metal GPU wiring!)
// ---------------------------------------------------------------------------
struct RawTensorMeta {
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    size_t data_off = 0;
    size_t nbytes = 0;
};

struct RawShard {
    std::string path;
    void* ptr = nullptr;
    size_t size = 0;
    std::unordered_map<std::string, RawTensorMeta> metas;

    ~RawShard() { close(); }
    void close() {
        if (ptr && ptr != MAP_FAILED) {
            munmap(ptr, size);
            ptr = nullptr;
            size = 0;
        }
        metas.clear();
        path.clear();
    }
    bool open_file(const std::string& p) {
        if (path == p && ptr) return true;
        close();
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 16) { ::close(fd); return false; }
        size_t fsz = static_cast<size_t>(st.st_size);
        void* m = mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) return false;
        madvise(m, fsz, MADV_SEQUENTIAL);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(m);
        uint64_t hlen = 0;
        std::memcpy(&hlen, base, 8);
        if (fsz < 8u || hlen > fsz - 8u) { munmap(m, fsz); return false; }
        std::unordered_map<std::string, RawTensorMeta> parsed_metas;
        try {
            std::string_view hsv(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(hlen));
            Json hdr = parse_json(hsv);
            size_t data_base = 8u + static_cast<size_t>(hlen);
            for (const auto& kv : hdr.as_object()) {
                if (kv.first == "__metadata__") continue;
                const Json& obj = kv.second;
                RawTensorMeta tm;
                tm.dtype = parse_dtype(obj["dtype"].as_string());
                for (const auto& s : obj["shape"].as_array()) {
                    int64_t dim = s.as_int();
                    if (dim < 0) { munmap(m, fsz); return false; }
                    tm.shape.push_back(dim);
                }
                const auto& offs = obj["data_offsets"].as_array();
                if (offs.size() != 2) { munmap(m, fsz); return false; }
                int64_t b0_i = offs[0].as_int();
                int64_t b1_i = offs[1].as_int();
                if (b0_i < 0 || b1_i < b0_i) { munmap(m, fsz); return false; }
                size_t b0 = static_cast<size_t>(b0_i);
                size_t b1 = static_cast<size_t>(b1_i);
                if (b1 > fsz - data_base) { munmap(m, fsz); return false; }
                tm.data_off = data_base + b0;
                tm.nbytes = b1 - b0;
                parsed_metas.emplace(kv.first, std::move(tm));
            }
        } catch (...) {
            munmap(m, fsz);
            return false;
        }
        metas = std::move(parsed_metas);
        path = p;
        ptr = m;
        size = fsz;
        return true;
    }
};

struct ShardPool {
    std::unordered_map<std::string, std::unique_ptr<RawShard>> shards;

    bool load_needed(const std::string& model_dir,
                     const std::unordered_map<std::string, std::string>& wmap,
                     const std::string& pfx,
                     const std::vector<int>& layers) {
        std::unordered_set<std::string> files_set;
        static const char* kProjs[3] = { "gate_proj", "up_proj", "down_proj" };
        static const char* kSufs[3]  = { ".weight", ".scales", ".biases" };
        for (int L : layers) {
            std::string pre = pfx + "layers." + std::to_string(L) + ".mlp.";
            for (const char* pr : kProjs) {
                for (const char* sf : kSufs) {
                    auto it = wmap.find(pre + pr + sf);
                    if (it != wmap.end()) files_set.insert(it->second);
                }
            }
        }
        std::vector<std::string> files(files_set.begin(), files_set.end());
        if (files.empty()) return false;

        std::vector<std::unique_ptr<RawShard>> opened(files.size());
        std::atomic<size_t> next_i{0};
        std::atomic<bool> ok{true};
        int n_thr = std::min<int>(4, static_cast<int>(files.size()));
        std::vector<std::thread> ths;
        for (int ti = 0; ti < n_thr; ++ti) {
            ths.emplace_back([&]() {
                while (true) {
                    size_t i = next_i.fetch_add(1);
                    if (i >= files.size()) break;
                    auto sh = std::make_unique<RawShard>();
                    if (!sh->open_file(model_dir + "/" + files[i])) {
                        ok.store(false);
                        break;
                    }
                    opened[i] = std::move(sh);
                }
            });
        }
        for (auto& th : ths) th.join();
        if (!ok.load()) return false;
        for (size_t i = 0; i < files.size(); ++i) {
            shards.emplace(files[i], std::move(opened[i]));
        }
        return true;
    }

    const RawShard* get(const std::string& fname) const {
        auto it = shards.find(fname);
        return (it == shards.end()) ? nullptr : it->second.get();
    }
};

inline float read_scalar(const uint8_t* p, DType dt, size_t idx) {
    if (dt == DType::F16) {
        __fp16 h;
        std::memcpy(&h, p + idx * 2u, 2);
        return static_cast<float>(h);
    } else if (dt == DType::BF16) {
        uint16_t raw = 0;
        std::memcpy(&raw, p + idx * 2u, 2);
        uint32_t u = static_cast<uint32_t>(raw) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, 4);
        return f;
    } else {
        float f = 0.0f;
        std::memcpy(&f, p + idx * 4u, 4);
        return f;
    }
}

inline uint16_t f32_to_f16_bits(float v) {
    __fp16 h = static_cast<__fp16>(v);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, 2);
    return bits;
}

inline float f16_bits_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<float>(h);
}

// Lightweight direct pointer descriptor to a safetensors weight matrix (ZERO FP32 matrix allocation!)
struct RawWeightRef {
    bool is_q4 = false;
    DType w_dtype = DType::UNKNOWN;
    DType s_dtype = DType::UNKNOWN;
    DType b_dtype = DType::UNKNOWN;
    const uint8_t* w_ptr = nullptr;
    const uint8_t* s_ptr = nullptr;
    const uint8_t* b_ptr = nullptr;
    uint32_t O = 0;
    uint32_t K = 0;
    uint32_t groups_per_row = 0;
    uint32_t gsize = 0;
    uint32_t bytes_per_row = 0;
};

bool resolve_weight_ref(const ShardPool& pool,
                        const std::unordered_map<std::string, std::string>& wmap,
                        const std::string& key_prefix,
                        uint32_t exp_O, uint32_t exp_K,
                        RawWeightRef& out) {
    out = RawWeightRef{};
    auto wit = wmap.find(key_prefix + ".weight");
    if (wit == wmap.end()) return false;
    const RawShard* wsh = pool.get(wit->second);
    if (!wsh) return false;
    auto wmit = wsh->metas.find(key_prefix + ".weight");
    if (wmit == wsh->metas.end()) return false;
    const RawTensorMeta& wm = wmit->second;
    if (wm.shape.size() != 2 || wm.shape[0] != static_cast<int64_t>(exp_O)) return false;

    out.O = exp_O;
    out.K = exp_K;
    out.w_dtype = wm.dtype;
    out.w_ptr = reinterpret_cast<const uint8_t*>(wsh->ptr) + wm.data_off;

    auto is_float_dtype = [](DType dt) {
        return dt == DType::F16 || dt == DType::BF16 || dt == DType::F32;
    };

    if (wm.dtype == DType::U32) {
        // Strictly validate 4-bit MLX U32 packing (8 nibbles per uint32 => shape[1] == exp_K / 8)
        if ((exp_K % 8u) != 0u || wm.shape[1] != static_cast<int64_t>(exp_K / 8u)) return false;
        const size_t exp_w_bytes = static_cast<size_t>(exp_O) * (exp_K >> 1);
        if (wm.nbytes != exp_w_bytes) return false;

        auto sit = wmap.find(key_prefix + ".scales");
        auto bit = wmap.find(key_prefix + ".biases");
        if (sit == wmap.end() || bit == wmap.end()) return false;
        const RawShard* ssh = pool.get(sit->second);
        const RawShard* bsh = pool.get(bit->second);
        if (!ssh || !bsh) return false;
        auto smit = ssh->metas.find(key_prefix + ".scales");
        auto bmit = bsh->metas.find(key_prefix + ".biases");
        if (smit == ssh->metas.end() || bmit == bsh->metas.end()) return false;
        const RawTensorMeta& sm = smit->second;
        const RawTensorMeta& bm = bmit->second;
        if (sm.shape.size() != 2 || sm.shape[0] != static_cast<int64_t>(exp_O) || sm.shape[1] <= 0) return false;
        if (bm.shape != sm.shape) return false;
        if (!is_float_dtype(sm.dtype) || !is_float_dtype(bm.dtype)) return false;

        out.is_q4 = true;
        out.s_dtype = sm.dtype;
        out.b_dtype = bm.dtype;
        out.s_ptr = reinterpret_cast<const uint8_t*>(ssh->ptr) + sm.data_off;
        out.b_ptr = reinterpret_cast<const uint8_t*>(bsh->ptr) + bm.data_off;
        out.groups_per_row = static_cast<uint32_t>(sm.shape[1]);
        out.gsize = exp_K / out.groups_per_row;
        out.bytes_per_row = exp_K >> 1;
        const size_t n_groups = static_cast<size_t>(exp_O) * out.groups_per_row;
        if (sm.nbytes < n_groups * dtype_size(sm.dtype)) return false;
        if (bm.nbytes < n_groups * dtype_size(bm.dtype)) return false;
        return (out.gsize > 0 && out.groups_per_row * out.gsize == exp_K);
    }
    if (!is_float_dtype(wm.dtype)) return false;
    if (wm.shape[1] != static_cast<int64_t>(exp_K)) return false;
    return wm.nbytes == static_cast<size_t>(exp_O) * exp_K * dtype_size(wm.dtype);
}

// Reusable tiny per-worker L1-resident scratch buffers (20 KB row_f32 + 10 KB row_f16 vs old 1.07 GB!)
struct WorkerScratch {
    std::vector<float> row_f32;
    std::vector<__fp16> row_f16;
    std::vector<float> aw1, s1, aw2, s2, tmp_med;
    std::vector<uint16_t> inv1_f16;
    std::vector<uint8_t> d_i4, o_i4;
    std::vector<uint16_t> s_f16;
    std::vector<uint16_t> z_sub, z_hid;
    BlobWriter bw;
};

// Pass 1 (zero FP32 matrix writes!): stream 4-bit safetensors directly into L1-resident aw[K] via pure NEON 32-wt unpack!
void accumulate_col_amax(const RawWeightRef& w, float* aw) {
    if (w.is_q4 && (w.gsize & 31u) == 0u) {
#if defined(__aarch64__) || defined(__ARM_NEON)
        const uint8x16_t v0f = vdupq_n_u8(0x0Fu);
        const uint32_t bytes_per_g = w.gsize >> 1;
        for (uint32_t r = 0; r < w.O; ++r) {
            const uint8_t* wr = w.w_ptr + static_cast<size_t>(r) * w.bytes_per_row;
            const size_t s_row = static_cast<size_t>(r) * w.groups_per_row;
            for (uint32_t g = 0; g < w.groups_per_row; ++g) {
                float sc = read_scalar(w.s_ptr, w.s_dtype, s_row + g);
                float bi = read_scalar(w.b_ptr, w.b_dtype, s_row + g);
                float32x4_t vsc = vdupq_n_f32(sc);
                float32x4_t vbi = vdupq_n_f32(bi);
                const uint8_t* gb = wr + static_cast<size_t>(g) * bytes_per_g;
                float* ag = aw + static_cast<size_t>(g) * w.gsize;
                for (uint32_t off = 0; off < w.gsize; off += 32u) {
                    uint8x16_t raw = vld1q_u8(gb + (off >> 1));
                    uint8x16_t lo_n = vandq_u8(raw, v0f);
                    uint8x16_t hi_n = vshrq_n_u8(raw, 4);
                    uint8x16_t n0_15  = vzip1q_u8(lo_n, hi_n);
                    uint8x16_t n16_31 = vzip2q_u8(lo_n, hi_n);

                    uint16x8_t u0 = vmovl_u8(vget_low_u8(n0_15));
                    uint16x8_t u1 = vmovl_u8(vget_high_u8(n0_15));
                    uint16x8_t u2 = vmovl_u8(vget_low_u8(n16_31));
                    uint16x8_t u3 = vmovl_u8(vget_high_u8(n16_31));

                    float32x4_t f0 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))),  vsc);
                    float32x4_t f1 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0))), vsc);
                    float32x4_t f2 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))),  vsc);
                    float32x4_t f3 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1))), vsc);
                    float32x4_t f4 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u2))),  vsc);
                    float32x4_t f5 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u2))), vsc);
                    float32x4_t f6 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u3))),  vsc);
                    float32x4_t f7 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u3))), vsc);

                    float* a = ag + off;
                    vst1q_f32(a +  0u, vmaxq_f32(vld1q_f32(a +  0u), vabsq_f32(f0)));
                    vst1q_f32(a +  4u, vmaxq_f32(vld1q_f32(a +  4u), vabsq_f32(f1)));
                    vst1q_f32(a +  8u, vmaxq_f32(vld1q_f32(a +  8u), vabsq_f32(f2)));
                    vst1q_f32(a + 12u, vmaxq_f32(vld1q_f32(a + 12u), vabsq_f32(f3)));
                    vst1q_f32(a + 16u, vmaxq_f32(vld1q_f32(a + 16u), vabsq_f32(f4)));
                    vst1q_f32(a + 20u, vmaxq_f32(vld1q_f32(a + 20u), vabsq_f32(f5)));
                    vst1q_f32(a + 24u, vmaxq_f32(vld1q_f32(a + 24u), vabsq_f32(f6)));
                    vst1q_f32(a + 28u, vmaxq_f32(vld1q_f32(a + 28u), vabsq_f32(f7)));
                }
            }
        }
        return;
#endif
    }
    if (w.is_q4) {
        for (uint32_t r = 0; r < w.O; ++r) {
            const uint8_t* wr = w.w_ptr + static_cast<size_t>(r) * w.bytes_per_row;
            const size_t s_row = static_cast<size_t>(r) * w.groups_per_row;
            for (uint32_t g = 0; g < w.groups_per_row; ++g) {
                float sc = read_scalar(w.s_ptr, w.s_dtype, s_row + g);
                float bi = read_scalar(w.b_ptr, w.b_dtype, s_row + g);
                uint32_t k0 = g * w.gsize;
                for (uint32_t j = 0; j < w.gsize; j += 2u) {
                    uint8_t b = wr[(k0 + j) >> 1];
                    float v0 = static_cast<float>(b & 0xFu) * sc + bi;
                    float v1 = static_cast<float>(b >> 4)   * sc + bi;
                    aw[k0 + j]     = std::max(aw[k0 + j],     std::fabs(v0));
                    aw[k0 + j + 1] = std::max(aw[k0 + j + 1], std::fabs(v1));
                }
            }
        }
    } else {
        for (uint32_t r = 0; r < w.O; ++r) {
            size_t base = static_cast<size_t>(r) * w.K;
            for (uint32_t k = 0; k < w.K; ++k) {
                aw[k] = std::max(aw[k], std::fabs(read_scalar(w.w_ptr, w.w_dtype, base + k)));
            }
        }
    }
}

// Pass 2: dequantize row `r` slice `[col0 .. col0+K)` into L1-resident `dst[K]`, fused with SmoothQuant scaling + min/max!
inline void dequant_row_slice_scaled(const RawWeightRef& w, uint32_t r, uint32_t col0, uint32_t K,
                                     const float* col_scale, float row_scale,
                                     float* dst, float& out_min, float& out_max) {
    if (w.is_q4 && (col0 % w.gsize) == 0u && (K % w.gsize) == 0u && (w.gsize & 31u) == 0u) {
#if defined(__aarch64__) || defined(__ARM_NEON)
        const uint8x16_t v0f = vdupq_n_u8(0x0Fu);
        const uint32_t g0 = col0 / w.gsize;
        const uint32_t ng = K / w.gsize;
        const uint32_t bytes_per_g = w.gsize >> 1;
        const uint8_t* wr = w.w_ptr + static_cast<size_t>(r) * w.bytes_per_row + static_cast<size_t>(g0) * bytes_per_g;
        const size_t s_base = static_cast<size_t>(r) * w.groups_per_row + g0;
        const float* cs_base = col_scale ? (col_scale + col0) : nullptr;

        float32x4_t vmn = vdupq_n_f32(1e38f);
        float32x4_t vmx = vdupq_n_f32(-1e38f);

        for (uint32_t gi = 0; gi < ng; ++gi) {
            float sc = read_scalar(w.s_ptr, w.s_dtype, s_base + gi) * row_scale;
            float bi = read_scalar(w.b_ptr, w.b_dtype, s_base + gi) * row_scale;
            float32x4_t vsc = vdupq_n_f32(sc);
            float32x4_t vbi = vdupq_n_f32(bi);
            const uint8_t* gb = wr + static_cast<size_t>(gi) * bytes_per_g;
            float* dg = dst + static_cast<size_t>(gi) * w.gsize;
            const float* cg = cs_base ? (cs_base + static_cast<size_t>(gi) * w.gsize) : nullptr;

            for (uint32_t off = 0; off < w.gsize; off += 32u) {
                uint8x16_t raw = vld1q_u8(gb + (off >> 1));
                uint8x16_t lo_n = vandq_u8(raw, v0f);
                uint8x16_t hi_n = vshrq_n_u8(raw, 4);
                uint8x16_t n0_15  = vzip1q_u8(lo_n, hi_n);
                uint8x16_t n16_31 = vzip2q_u8(lo_n, hi_n);

                uint16x8_t u0 = vmovl_u8(vget_low_u8(n0_15));
                uint16x8_t u1 = vmovl_u8(vget_high_u8(n0_15));
                uint16x8_t u2 = vmovl_u8(vget_low_u8(n16_31));
                uint16x8_t u3 = vmovl_u8(vget_high_u8(n16_31));

                float32x4_t f0 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0))),  vsc);
                float32x4_t f1 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0))), vsc);
                float32x4_t f2 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1))),  vsc);
                float32x4_t f3 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1))), vsc);
                float32x4_t f4 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u2))),  vsc);
                float32x4_t f5 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u2))), vsc);
                float32x4_t f6 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u3))),  vsc);
                float32x4_t f7 = vfmaq_f32(vbi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u3))), vsc);

                if (cg) {
                    const float* c = cg + off;
                    f0 = vmulq_f32(f0, vld1q_f32(c +  0u));
                    f1 = vmulq_f32(f1, vld1q_f32(c +  4u));
                    f2 = vmulq_f32(f2, vld1q_f32(c +  8u));
                    f3 = vmulq_f32(f3, vld1q_f32(c + 12u));
                    f4 = vmulq_f32(f4, vld1q_f32(c + 16u));
                    f5 = vmulq_f32(f5, vld1q_f32(c + 20u));
                    f6 = vmulq_f32(f6, vld1q_f32(c + 24u));
                    f7 = vmulq_f32(f7, vld1q_f32(c + 28u));
                }

                float* d = dg + off;
                vst1q_f32(d +  0u, f0); vst1q_f32(d +  4u, f1);
                vst1q_f32(d +  8u, f2); vst1q_f32(d + 12u, f3);
                vst1q_f32(d + 16u, f4); vst1q_f32(d + 20u, f5);
                vst1q_f32(d + 24u, f6); vst1q_f32(d + 28u, f7);

                float32x4_t mn01 = vminq_f32(vminq_f32(f0, f1), vminq_f32(f2, f3));
                float32x4_t mn23 = vminq_f32(vminq_f32(f4, f5), vminq_f32(f6, f7));
                float32x4_t mx01 = vmaxq_f32(vmaxq_f32(f0, f1), vmaxq_f32(f2, f3));
                float32x4_t mx23 = vmaxq_f32(vmaxq_f32(f4, f5), vmaxq_f32(f6, f7));
                vmn = vminq_f32(vmn, vminq_f32(mn01, mn23));
                vmx = vmaxq_f32(vmx, vmaxq_f32(mx01, mx23));
            }
        }
        out_min = vminvq_f32(vmn);
        out_max = vmaxvq_f32(vmx);
        return;
#endif
    }

    float mn = 1e38f, mx = -1e38f;
    if (w.is_q4) {
        const uint8_t* wr = w.w_ptr + static_cast<size_t>(r) * w.bytes_per_row;
        const size_t s_row = static_cast<size_t>(r) * w.groups_per_row;
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t ck = col0 + k;
            uint32_t g = ck / w.gsize;
            float sc = read_scalar(w.s_ptr, w.s_dtype, s_row + g);
            float bi = read_scalar(w.b_ptr, w.b_dtype, s_row + g);
            uint8_t b = wr[ck >> 1];
            uint32_t q = (ck & 1u) ? (b >> 4) : (b & 0xFu);
            float v = (static_cast<float>(q) * sc + bi) * row_scale;
            if (col_scale) v *= col_scale[ck];
            dst[k] = v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    } else {
        size_t base = static_cast<size_t>(r) * w.K + col0;
        for (uint32_t k = 0; k < K; ++k) {
            float v = read_scalar(w.w_ptr, w.w_dtype, base + k) * row_scale;
            if (col_scale) v *= col_scale[col0 + k];
            dst[k] = v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    out_min = mn;
    out_max = mx;
}

// ---------------------------------------------------------------------------
// Pre-normalize row FP32 -> L1-resident FP16 in [-1.0, +1.0] (10 KB for K=5120)
// ---------------------------------------------------------------------------
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
inline void build_norm_f16(const float* wr, uint32_t K, float inv_amax, __fp16* dst) {
    float32x4_t vinv = vdupq_n_f32(inv_amax);
    uint32_t k = 0;
    for (; k + 8u <= K; k += 8u) {
        float32x4_t w0 = vmulq_f32(vld1q_f32(wr + k), vinv);
        float32x4_t w1 = vmulq_f32(vld1q_f32(wr + k + 4u), vinv);
        float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(w0), w1);
        vst1q_f16(dst + k, h);
    }
    for (; k < K; ++k) {
        dst[k] = static_cast<__fp16>(wr[k] * inv_amax);
    }
}

// Ultra-fast 8-wide FP16 + FMLAL (FP32 accum) 2-candidate coarse evaluator for hf4/lut4!
inline void eval_hf4_err_2p_f16(const __fp16* rn, uint32_t K,
                                float sc0, float ratio0, float zp0,
                                float sc1, float ratio1, float zp1,
                                float& e0, float& e1) {
    float16x8_t vr0 = vdupq_n_f16(static_cast<float16_t>(ratio0));
    float16x8_t vr1 = vdupq_n_f16(static_cast<float16_t>(ratio1));
    float16x8_t vlo0 = vdupq_n_f16(static_cast<float16_t>(-8.0f - zp0));
    float16x8_t vhi0 = vdupq_n_f16(static_cast<float16_t>( 7.0f - zp0));
    float16x8_t vlo1 = vdupq_n_f16(static_cast<float16_t>(-8.0f - zp1));
    float16x8_t vhi1 = vdupq_n_f16(static_cast<float16_t>( 7.0f - zp1));
    float32x4_t a0l = vdupq_n_f32(0.0f), a0h = vdupq_n_f32(0.0f);
    float32x4_t a1l = vdupq_n_f32(0.0f), a1h = vdupq_n_f32(0.0f);
    uint32_t k = 0;
    for (; k + 8u <= K; k += 8u) {
        float16x8_t x = vld1q_f16(rn + k);
        float16x8_t wn0 = vmulq_f16(x, vr0);
        float16x8_t wn1 = vmulq_f16(x, vr1);
        float16x8_t q0 = vminq_f16(vmaxq_f16(vrndnq_f16(wn0), vlo0), vhi0);
        float16x8_t q1 = vminq_f16(vmaxq_f16(vrndnq_f16(wn1), vlo1), vhi1);
        float16x8_t d0 = vsubq_f16(wn0, q0);
        float16x8_t d1 = vsubq_f16(wn1, q1);
        a0l = vfmlalq_low_f16(a0l, d0, d0);
        a0h = vfmlalq_high_f16(a0h, d0, d0);
        a1l = vfmlalq_low_f16(a1l, d1, d1);
        a1h = vfmlalq_high_f16(a1h, d1, d1);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0l, a0h));
    float s1 = vaddvq_f32(vaddq_f32(a1l, a1h));
    for (; k < K; ++k) {
        float x = static_cast<float>(rn[k]);
        float wn0 = x * ratio0, wn1 = x * ratio1;
        float d0 = wn0 - std::clamp(std::nearbyintf(wn0), -8.0f - zp0, 7.0f - zp0);
        float d1 = wn1 - std::clamp(std::nearbyintf(wn1), -8.0f - zp1, 7.0f - zp1);
        s0 += d0 * d0;
        s1 += d1 * d1;
    }
    e0 = s0 * (sc0 * sc0);
    e1 = s1 * (sc1 * sc1);
}

// Ultra-fast 8-wide FP16 + FMLAL single-pass 3-zero-point evaluator!
inline void eval_hf4_err_3zp_f16(const __fp16* rn, uint32_t K, float sc, float ratio,
                                 float z0, float z1, float z2, float& e0, float& e1, float& e2) {
    float16x8_t vr = vdupq_n_f16(static_cast<float16_t>(ratio));
    float16x8_t vlo0 = vdupq_n_f16(static_cast<float16_t>(-8.0f - z0));
    float16x8_t vhi0 = vdupq_n_f16(static_cast<float16_t>( 7.0f - z0));
    float16x8_t vlo1 = vdupq_n_f16(static_cast<float16_t>(-8.0f - z1));
    float16x8_t vhi1 = vdupq_n_f16(static_cast<float16_t>( 7.0f - z1));
    float16x8_t vlo2 = vdupq_n_f16(static_cast<float16_t>(-8.0f - z2));
    float16x8_t vhi2 = vdupq_n_f16(static_cast<float16_t>( 7.0f - z2));
    float32x4_t a0l = vdupq_n_f32(0.0f), a0h = vdupq_n_f32(0.0f);
    float32x4_t a1l = vdupq_n_f32(0.0f), a1h = vdupq_n_f32(0.0f);
    float32x4_t a2l = vdupq_n_f32(0.0f), a2h = vdupq_n_f32(0.0f);
    uint32_t k = 0;
    for (; k + 8u <= K; k += 8u) {
        float16x8_t wn = vmulq_f16(vld1q_f16(rn + k), vr);
        float16x8_t r = vrndnq_f16(wn);
        float16x8_t d0 = vsubq_f16(wn, vminq_f16(vmaxq_f16(r, vlo0), vhi0));
        float16x8_t d1 = vsubq_f16(wn, vminq_f16(vmaxq_f16(r, vlo1), vhi1));
        float16x8_t d2 = vsubq_f16(wn, vminq_f16(vmaxq_f16(r, vlo2), vhi2));
        a0l = vfmlalq_low_f16(a0l, d0, d0); a0h = vfmlalq_high_f16(a0h, d0, d0);
        a1l = vfmlalq_low_f16(a1l, d1, d1); a1h = vfmlalq_high_f16(a1h, d1, d1);
        a2l = vfmlalq_low_f16(a2l, d2, d2); a2h = vfmlalq_high_f16(a2h, d2, d2);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0l, a0h));
    float s1 = vaddvq_f32(vaddq_f32(a1l, a1h));
    float s2 = vaddvq_f32(vaddq_f32(a2l, a2h));
    for (; k < K; ++k) {
        float wn = static_cast<float>(rn[k]) * ratio;
        float r = std::nearbyintf(wn);
        float d0 = wn - std::clamp(r, -8.0f - z0, 7.0f - z0);
        float d1 = wn - std::clamp(r, -8.0f - z1, 7.0f - z1);
        float d2 = wn - std::clamp(r, -8.0f - z2, 7.0f - z2);
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2;
    }
    float sc2 = sc * sc;
    e0 = s0 * sc2;
    e1 = s1 * sc2;
    e2 = s2 * sc2;
}
#endif

// Exact FP32 NEON fallback (simplified algebra: q - zp = clamp(round(w * inv_sc), -8 - zp, 7 - zp) + vfmsq_f32!)
inline float eval_hf4_err_1zp_f32(const float* wr, uint32_t K, float sc, float inv_sc, float zp) {
    float qlo = -8.0f - zp;
    float qhi =  7.0f - zp;
#if defined(__aarch64__) || defined(__ARM_NEON)
    float32x4_t vsc = vdupq_n_f32(sc);
    float32x4_t vinv = vdupq_n_f32(inv_sc);
    float32x4_t vqlo = vdupq_n_f32(qlo);
    float32x4_t vqhi = vdupq_n_f32(qhi);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    uint32_t k = 0;
    for (; k + 8u <= K; k += 8u) {
        float32x4_t w0 = vld1q_f32(wr + k);
        float32x4_t w1 = vld1q_f32(wr + k + 4u);
        float32x4_t q0 = vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(w0, vinv)), vqlo), vqhi);
        float32x4_t q1 = vminq_f32(vmaxq_f32(vrndnq_f32(vmulq_f32(w1, vinv)), vqlo), vqhi);
        float32x4_t d0 = vfmsq_f32(w0, q0, vsc);
        float32x4_t d1 = vfmsq_f32(w1, q1, vsc);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    float err = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < K; ++k) {
        float q = std::clamp(std::nearbyintf(wr[k] * inv_sc), qlo, qhi);
        float d = wr[k] - q * sc;
        err += d * d;
    }
    return err;
#else
    float err = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        float q = std::clamp(std::nearbyintf(wr[k] * inv_sc), qlo, qhi);
        float d = wr[k] - q * sc;
        err += d * d;
    }
    return err;
#endif
}

inline void eval_hf4_err_3zp_f32(const float* wr, uint32_t K, float sc, float inv_sc,
                                 float z0, float z1, float z2, float& e0, float& e1, float& e2) {
    float lo0 = -8.0f - z0, hi0 = 7.0f - z0;
    float lo1 = -8.0f - z1, hi1 = 7.0f - z1;
    float lo2 = -8.0f - z2, hi2 = 7.0f - z2;
#if defined(__aarch64__) || defined(__ARM_NEON)
    float32x4_t vsc = vdupq_n_f32(sc);
    float32x4_t vinv = vdupq_n_f32(inv_sc);
    float32x4_t vlo0 = vdupq_n_f32(lo0), vhi0 = vdupq_n_f32(hi0);
    float32x4_t vlo1 = vdupq_n_f32(lo1), vhi1 = vdupq_n_f32(hi1);
    float32x4_t vlo2 = vdupq_n_f32(lo2), vhi2 = vdupq_n_f32(hi2);
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f);
    uint32_t k = 0;
    for (; k + 4u <= K; k += 4u) {
        float32x4_t w = vld1q_f32(wr + k);
        float32x4_t r = vrndnq_f32(vmulq_f32(w, vinv));
        float32x4_t d0 = vfmsq_f32(w, vminq_f32(vmaxq_f32(r, vlo0), vhi0), vsc);
        float32x4_t d1 = vfmsq_f32(w, vminq_f32(vmaxq_f32(r, vlo1), vhi1), vsc);
        float32x4_t d2 = vfmsq_f32(w, vminq_f32(vmaxq_f32(r, vlo2), vhi2), vsc);
        a0 = vfmaq_f32(a0, d0, d0);
        a1 = vfmaq_f32(a1, d1, d1);
        a2 = vfmaq_f32(a2, d2, d2);
    }
    e0 = vaddvq_f32(a0); e1 = vaddvq_f32(a1); e2 = vaddvq_f32(a2);
    for (; k < K; ++k) {
        float r = std::nearbyintf(wr[k] * inv_sc);
        float d0 = wr[k] - std::clamp(r, lo0, hi0) * sc;
        float d1 = wr[k] - std::clamp(r, lo1, hi1) * sc;
        float d2 = wr[k] - std::clamp(r, lo2, hi2) * sc;
        e0 += d0 * d0; e1 += d1 * d1; e2 += d2 * d2;
    }
#else
    e0 = e1 = e2 = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        float r = std::nearbyintf(wr[k] * inv_sc);
        float d0 = wr[k] - std::clamp(r, lo0, hi0) * sc;
        float d1 = wr[k] - std::clamp(r, lo1, hi1) * sc;
        float d2 = wr[k] - std::clamp(r, lo2, hi2) * sc;
        e0 += d0 * d0; e1 += d1 * d1; e2 += d2 * d2;
    }
#endif
}

// Pure-NEON 16-weight -> 8-byte nibble packer (zero stack spills!)
inline void pack_int4_row_neon(const float* wr, uint32_t K, float inv_sc, float fzp, uint8_t* dst) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    float32x4_t vinv = vdupq_n_f32(inv_sc);
    float32x4_t vzp = vdupq_n_f32(fzp);
    float32x4_t vlo = vdupq_n_f32(-8.0f), vhi = vdupq_n_f32(7.0f);
    uint8x16_t v0f = vdupq_n_u8(0x0Fu);
    uint32_t k = 0;
    for (; k + 16u <= K; k += 16u) {
        int32x4_t q0 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vaddq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(wr + k +  0u), vinv)), vzp), vlo), vhi));
        int32x4_t q1 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vaddq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(wr + k +  4u), vinv)), vzp), vlo), vhi));
        int32x4_t q2 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vaddq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(wr + k +  8u), vinv)), vzp), vlo), vhi));
        int32x4_t q3 = vcvtq_s32_f32(vminq_f32(vmaxq_f32(vaddq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(wr + k + 12u), vinv)), vzp), vlo), vhi));
        int16x8_t s01 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
        int16x8_t s23 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
        uint8x16_t nib = vandq_u8(vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(s01), vmovn_s16(s23))), v0f);
        uint8x8_t even = vuzp1_u8(vget_low_u8(nib), vget_high_u8(nib));
        uint8x8_t odd  = vuzp2_u8(vget_low_u8(nib), vget_high_u8(nib));
        vst1_u8(dst + (k >> 1), vorr_u8(even, vshl_n_u8(odd, 4)));
    }
    for (; k < K; k += 2u) {
        int q0 = static_cast<int>(std::clamp(std::nearbyintf(wr[k]     * inv_sc) + fzp, -8.0f, 7.0f));
        int q1 = static_cast<int>(std::clamp(std::nearbyintf(wr[k + 1] * inv_sc) + fzp, -8.0f, 7.0f));
        dst[k >> 1] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
#else
    for (uint32_t k = 0; k < K; k += 2u) {
        int q0 = static_cast<int>(std::clamp(std::nearbyintf(wr[k]     * inv_sc) + fzp, -8.0f, 7.0f));
        int q1 = static_cast<int>(std::clamp(std::nearbyintf(wr[k + 1] * inv_sc) + fzp, -8.0f, 7.0f));
        dst[k >> 1] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
#endif
}

// Stream-quantize [O, K] -> hf4 asymmetric int4 directly from RawWeightRef (L1-resident row buffer!)
void quantize_slice_hf4_asym(const RawWeightRef& wref, uint32_t row0, uint32_t O,
                             uint32_t col0, uint32_t K,
                             const float* col_scale, const float* inv_row_scale,
                             bool mse_clip, WorkerScratch& ws) {
    ws.d_i4.resize(static_cast<size_t>(O) * (K / 2u));
    ws.o_i4.assign((O + 1u) / 2u, 0u);
    ws.s_f16.resize(O);
    ws.row_f32.resize(K);
    ws.row_f16.resize(K);

    static constexpr float kCoarseP[6] = { 0.55f, 0.64f, 0.73f, 0.82f, 0.91f, 1.00f };
    float* wr = ws.row_f32.data();
    __fp16* rn = ws.row_f16.data();
    (void)rn;

    for (uint32_t r = 0; r < O; ++r) {
        uint32_t gr = row0 + r;
        float rsc = inv_row_scale ? inv_row_scale[gr] : 1.0f;
        float wmin0 = 0.0f, wmax0 = 0.0f;
        dequant_row_slice_scaled(wref, gr, col0, K, col_scale, rsc, wr, wmin0, wmax0);

        float mid = 0.5f * (wmin0 + wmax0);
        float half = 0.5f * (wmax0 - wmin0);
        float amax0 = std::max(std::fabs(wmin0), std::fabs(wmax0));

        uint16_t best_sc_h = f32_to_f16_bits(std::max((wmax0 - wmin0) / 15.0f, 1e-6f));
        float best_sc = std::max(f16_bits_to_f32(best_sc_h), 1e-6f);
        int best_zp = static_cast<int>(std::clamp(std::nearbyintf(-wmin0 / best_sc) - 8.0f, -8.0f, 7.0f));

        if (mse_clip && half > 1e-7f) {
            float best_err = 1e38f;
            float best_p = 1.0f;

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
            const bool use_f16 = (amax0 > 1e-6f && (amax0 / std::max((half * 2.0f * 0.50f) / 15.0f, 1e-6f)) < 2048.0f);
            if (use_f16) {
                build_norm_f16(wr, K, 1.0f / amax0, rn);
                // Stage 1: 6 coarse steps evaluated in 3 fused 2-candidate FP16+FMLAL passes!
                for (int ci = 0; ci < 6; ci += 2) {
                    float p0 = kCoarseP[ci], p1 = kCoarseP[ci + 1];
                    float lo0 = mid - half * p0, hi0 = mid + half * p0;
                    float lo1 = mid - half * p1, hi1 = mid + half * p1;
                    uint16_t sch0 = f32_to_f16_bits(std::max((hi0 - lo0) / 15.0f, 1e-6f));
                    uint16_t sch1 = f32_to_f16_bits(std::max((hi1 - lo1) / 15.0f, 1e-6f));
                    float sc0 = std::max(f16_bits_to_f32(sch0), 1e-6f);
                    float sc1 = std::max(f16_bits_to_f32(sch1), 1e-6f);
                    int zp0 = static_cast<int>(std::clamp(std::nearbyintf(-lo0 / sc0) - 8.0f, -8.0f, 7.0f));
                    int zp1 = static_cast<int>(std::clamp(std::nearbyintf(-lo1 / sc1) - 8.0f, -8.0f, 7.0f));
                    float e0 = 0.0f, e1 = 0.0f;
                    eval_hf4_err_2p_f16(rn, K, sc0, amax0 / sc0, static_cast<float>(zp0),
                                        sc1, amax0 / sc1, static_cast<float>(zp1), e0, e1);
                    if (e0 < best_err) { best_err = e0; best_p = p0; best_sc_h = sch0; best_sc = sc0; best_zp = zp0; }
                    if (e1 < best_err) { best_err = e1; best_p = p1; best_sc_h = sch1; best_sc = sc1; best_zp = zp1; }
                }
                // Stage 2: up to 3 fine steps around best_p testing {zp0-1, zp0, zp0+1} in ONE FP16+FMLAL pass each!
                float fine_p[3] = {
                    std::clamp(best_p - 0.045f, 0.50f, 1.00f),
                    best_p,
                    std::clamp(best_p + 0.045f, 0.50f, 1.00f)
                };
                uint16_t seen_sch[3] = {};
                int n_seen = 0;
                for (int fi = 0; fi < 3; ++fi) {
                    float p = fine_p[fi];
                    float lo = mid - half * p;
                    float hi = mid + half * p;
                    uint16_t sc_h = f32_to_f16_bits(std::max((hi - lo) / 15.0f, 1e-6f));
                    bool dup = false;
                    for (int s = 0; s < n_seen; ++s) if (seen_sch[s] == sc_h) { dup = true; break; }
                    if (dup) continue;
                    seen_sch[n_seen++] = sc_h;
                    float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                    int zp0 = static_cast<int>(std::clamp(std::nearbyintf(-lo / sc) - 8.0f, -8.0f, 7.0f));
                    int z_cands[3] = { std::max(-8, zp0 - 1), zp0, std::min(7, zp0 + 1) };
                    float errs[3];
                    eval_hf4_err_3zp_f16(rn, K, sc, amax0 / sc,
                                         static_cast<float>(z_cands[0]),
                                         static_cast<float>(z_cands[1]),
                                         static_cast<float>(z_cands[2]),
                                         errs[0], errs[1], errs[2]);
                    for (int zi = 0; zi < 3; ++zi) {
                        if (errs[zi] < best_err) {
                            best_err = errs[zi];
                            best_sc_h = sc_h;
                            best_sc = sc;
                            best_zp = z_cands[zi];
                        }
                    }
                }
            } else
#endif
            {
                for (int ci = 0; ci < 6; ++ci) {
                    float p = kCoarseP[ci];
                    float lo = mid - half * p;
                    float hi = mid + half * p;
                    uint16_t sc_h = f32_to_f16_bits(std::max((hi - lo) / 15.0f, 1e-6f));
                    float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                    float inv_sc = 1.0f / sc;
                    int zp = static_cast<int>(std::clamp(std::nearbyintf(-lo * inv_sc) - 8.0f, -8.0f, 7.0f));
                    float err = eval_hf4_err_1zp_f32(wr, K, sc, inv_sc, static_cast<float>(zp));
                    if (err < best_err) {
                        best_err = err; best_p = p; best_sc_h = sc_h; best_sc = sc; best_zp = zp;
                    }
                }
                float fine_p[3] = {
                    std::clamp(best_p - 0.045f, 0.50f, 1.00f),
                    best_p,
                    std::clamp(best_p + 0.045f, 0.50f, 1.00f)
                };
                uint16_t seen_sch[3] = {};
                int n_seen = 0;
                for (int fi = 0; fi < 3; ++fi) {
                    float p = fine_p[fi];
                    float lo = mid - half * p;
                    float hi = mid + half * p;
                    uint16_t sc_h = f32_to_f16_bits(std::max((hi - lo) / 15.0f, 1e-6f));
                    bool dup = false;
                    for (int s = 0; s < n_seen; ++s) if (seen_sch[s] == sc_h) { dup = true; break; }
                    if (dup) continue;
                    seen_sch[n_seen++] = sc_h;
                    float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                    float inv_sc = 1.0f / sc;
                    int zp0 = static_cast<int>(std::clamp(std::nearbyintf(-lo * inv_sc) - 8.0f, -8.0f, 7.0f));
                    int z_cands[3] = { std::max(-8, zp0 - 1), zp0, std::min(7, zp0 + 1) };
                    float errs[3];
                    eval_hf4_err_3zp_f32(wr, K, sc, inv_sc,
                                         static_cast<float>(z_cands[0]),
                                         static_cast<float>(z_cands[1]),
                                         static_cast<float>(z_cands[2]),
                                         errs[0], errs[1], errs[2]);
                    for (int zi = 0; zi < 3; ++zi) {
                        if (errs[zi] < best_err) {
                            best_err = errs[zi]; best_sc_h = sc_h; best_sc = sc; best_zp = z_cands[zi];
                        }
                    }
                }
            }
        }

        ws.s_f16[r] = best_sc_h;
        uint8_t zp_nib = static_cast<uint8_t>(best_zp & 0xF);
        if ((r & 1u) == 0u) ws.o_i4[r >> 1] = zp_nib;
        else                ws.o_i4[r >> 1] |= static_cast<uint8_t>(zp_nib << 4);

        uint8_t* dst = ws.d_i4.data() + static_cast<size_t>(r) * (K / 2u);
        pack_int4_row_neon(wr, K, 1.0f / best_sc, static_cast<float>(best_zp), dst);
    }
}

// Stream-quantize [O, K] -> universal M1-M5 lut4 (two's-complement uint4 indices + fp16 per-channel scales)
void quantize_slice_lut4_sym(const RawWeightRef& wref, uint32_t row0, uint32_t O,
                             uint32_t col0, uint32_t K,
                             const float* col_scale, const float* inv_row_scale,
                             bool mse_clip, WorkerScratch& ws) {
    ws.d_i4.resize(static_cast<size_t>(O) * (K / 2u));
    ws.s_f16.resize(O);
    ws.row_f32.resize(K);
    ws.row_f16.resize(K);

    static constexpr float kCoarseP[6] = { 0.55f, 0.64f, 0.73f, 0.82f, 0.91f, 1.00f };
    float* wr = ws.row_f32.data();
    __fp16* rn = ws.row_f16.data();
    (void)rn;

    for (uint32_t r = 0; r < O; ++r) {
        uint32_t gr = row0 + r;
        float rsc = inv_row_scale ? inv_row_scale[gr] : 1.0f;
        float wmin0 = 0.0f, wmax0 = 0.0f;
        dequant_row_slice_scaled(wref, gr, col0, K, col_scale, rsc, wr, wmin0, wmax0);
        float amax0 = std::max(1e-6f, std::max(std::fabs(wmin0), std::fabs(wmax0)));

        uint16_t best_sc_h = f32_to_f16_bits(std::max(amax0 / 7.5f, 1e-6f));
        float best_sc = std::max(f16_bits_to_f32(best_sc_h), 1e-6f);

        if (mse_clip && amax0 > 1e-6f) {
            float best_err = 1e38f;
            float best_p = 1.0f;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
            build_norm_f16(wr, K, 1.0f / amax0, rn);
            for (int ci = 0; ci < 6; ci += 2) {
                float p0 = kCoarseP[ci], p1 = kCoarseP[ci + 1];
                uint16_t sch0 = f32_to_f16_bits(std::max((amax0 * p0) / 7.5f, 1e-6f));
                uint16_t sch1 = f32_to_f16_bits(std::max((amax0 * p1) / 7.5f, 1e-6f));
                float sc0 = std::max(f16_bits_to_f32(sch0), 1e-6f);
                float sc1 = std::max(f16_bits_to_f32(sch1), 1e-6f);
                float e0 = 0.0f, e1 = 0.0f;
                eval_hf4_err_2p_f16(rn, K, sc0, amax0 / sc0, 0.0f, sc1, amax0 / sc1, 0.0f, e0, e1);
                if (e0 < best_err) { best_err = e0; best_p = p0; best_sc_h = sch0; best_sc = sc0; }
                if (e1 < best_err) { best_err = e1; best_p = p1; best_sc_h = sch1; best_sc = sc1; }
            }
            float fp0 = std::clamp(best_p - 0.045f, 0.50f, 1.00f);
            float fp1 = std::clamp(best_p + 0.045f, 0.50f, 1.00f);
            uint16_t sch0 = f32_to_f16_bits(std::max((amax0 * fp0) / 7.5f, 1e-6f));
            uint16_t sch1 = f32_to_f16_bits(std::max((amax0 * fp1) / 7.5f, 1e-6f));
            float sc0 = std::max(f16_bits_to_f32(sch0), 1e-6f);
            float sc1 = std::max(f16_bits_to_f32(sch1), 1e-6f);
            float e0 = 0.0f, e1 = 0.0f;
            eval_hf4_err_2p_f16(rn, K, sc0, amax0 / sc0, 0.0f, sc1, amax0 / sc1, 0.0f, e0, e1);
            if (e0 < best_err) { best_err = e0; best_sc_h = sch0; best_sc = sc0; }
            if (e1 < best_err) { best_err = e1; best_sc_h = sch1; best_sc = sc1; }
#else
            for (int ci = 0; ci < 6; ++ci) {
                float p = kCoarseP[ci];
                uint16_t sc_h = f32_to_f16_bits(std::max((amax0 * p) / 7.5f, 1e-6f));
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_hf4_err_1zp_f32(wr, K, sc, 1.0f / sc, 0.0f);
                if (err < best_err) { best_err = err; best_p = p; best_sc_h = sc_h; best_sc = sc; }
            }
            float fine_p[2] = { std::clamp(best_p - 0.045f, 0.50f, 1.00f), std::clamp(best_p + 0.045f, 0.50f, 1.00f) };
            for (int fi = 0; fi < 2; ++fi) {
                uint16_t sc_h = f32_to_f16_bits(std::max((amax0 * fine_p[fi]) / 7.5f, 1e-6f));
                if (sc_h == best_sc_h) continue;
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_hf4_err_1zp_f32(wr, K, sc, 1.0f / sc, 0.0f);
                if (err < best_err) { best_err = err; best_sc_h = sc_h; best_sc = sc; }
            }
#endif
        }

        ws.s_f16[r] = best_sc_h;
        uint8_t* dst = ws.d_i4.data() + static_cast<size_t>(r) * (K / 2u);
        pack_int4_row_neon(wr, K, 1.0f / best_sc, 0.0f, dst);
    }
}

// ---------------------------------------------------------------------------
// Hardware-verified non-uniform NF4 16-entry FP16 LUT + 8-wide ARM64 NEON binary-tree + vqtbl quantizer!
// ---------------------------------------------------------------------------
struct Nf4Tables {
    uint16_t lut_f16[16];
    float    lut_f32[16];
    float    mid_f32[15];
    __fp16   mid_h[15];
    uint16_t lv2_f16[8]; // [mid[1], mid[5], mid[9], mid[13], 0, 0, 0, 0]
    uint16_t lv3_f16[8]; // [mid[0], mid[2], mid[4], mid[6], mid[8], mid[10], mid[12], mid[14]]

    Nf4Tables() {
        static constexpr float kRaw[16] = {
            -1.0f, -0.6961928f, -0.52507305f, -0.39491749f,
            -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
             0.0795803f,  0.1609302f,  0.2461123f,  0.33791524f,
             0.44070983f, 0.562617f,   0.72295684f, 1.0f
        };
        for (int i = 0; i < 16; ++i) {
            lut_f16[i] = f32_to_f16_bits(kRaw[i]);
            lut_f32[i] = f16_bits_to_f32(lut_f16[i]);
        }
        uint16_t mid_bits[15];
        for (int i = 0; i < 15; ++i) {
            mid_f32[i] = 0.5f * (lut_f32[i] + lut_f32[i + 1]);
            mid_h[i] = static_cast<__fp16>(mid_f32[i]);
            mid_bits[i] = f32_to_f16_bits(mid_f32[i]);
        }
        lv2_f16[0] = mid_bits[1];  lv2_f16[1] = mid_bits[5];
        lv2_f16[2] = mid_bits[9];  lv2_f16[3] = mid_bits[13];
        lv2_f16[4] = 0u; lv2_f16[5] = 0u; lv2_f16[6] = 0u; lv2_f16[7] = 0u;
        for (int i = 0; i < 8; ++i) lv3_f16[i] = mid_bits[i * 2];
    }
};

const Nf4Tables& nf4_tables() {
    static const Nf4Tables tbl;
    return tbl;
}

inline uint32_t nf4_nearest_scalar(float wn, const float* m) {
    uint32_t lo = 0u, hi = 15u;
    while (lo < hi) {
        uint32_t md = (lo + hi) >> 1;
        if (wn >= m[md]) lo = md + 1u;
        else hi = md;
    }
    return lo;
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
// Branchless 4-level binary search for 8 FP16 values simultaneously (100% NEON registers, zero spills!).
// Returns uint16x8_t where each 16-bit lane holds (2*idx) | ((2*idx) << 8) for idx in [0..15].
inline uint16x8_t nf4_classify8_f16(float16x8_t wn,
                                    float16x8_t vm7, float16x8_t vm3, float16x8_t vm11,
                                    uint8x16_t lv2_tbl, uint8x16_t lv3_tbl,
                                    uint16x8_t v0100, uint16x8_t v0202, uint16x8_t v0404) {
    uint16x8_t c0 = vcgeq_f16(wn, vm7);
    float16x8_t m1 = vbslq_f16(c0, vm11, vm3);
    uint16x8_t c1 = vcgeq_f16(wn, m1);

    uint16x8_t idx = vorrq_u16(vandq_u16(c0, v0404), vandq_u16(c1, v0202));
    float16x8_t m2 = vreinterpretq_f16_u8(vqtbl1q_u8(lv2_tbl, vreinterpretq_u8_u16(vorrq_u16(idx, v0100))));
    uint16x8_t c2 = vcgeq_f16(wn, m2);

    idx = vorrq_u16(vshlq_n_u16(idx, 1), vandq_u16(c2, v0202));
    float16x8_t m3 = vreinterpretq_f16_u8(vqtbl1q_u8(lv3_tbl, vreinterpretq_u8_u16(vorrq_u16(idx, v0100))));
    uint16x8_t c3 = vcgeq_f16(wn, m3);

    return vorrq_u16(vshlq_n_u16(idx, 1), vandq_u16(c3, v0202));
}

// 100% register-resident 2x-unrolled (16 weights/iter) FP16 + vqtbl2q_u8 + FMLAL NF4 MSE evaluator!
inline float eval_nf4_err_f16(const __fp16* rn, uint32_t K, float sc, float ratio, const Nf4Tables& tbl) {
    float16x8_t vr = vdupq_n_f16(static_cast<float16_t>(ratio));
    float16x8_t vm7  = vdupq_n_f16(tbl.mid_h[7]);
    float16x8_t vm3  = vdupq_n_f16(tbl.mid_h[3]);
    float16x8_t vm11 = vdupq_n_f16(tbl.mid_h[11]);
    uint8x16_t lv2_tbl = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lv2_f16));
    uint8x16_t lv3_tbl = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lv3_f16));
    uint8x16x2_t lut_tbl2;
    lut_tbl2.val[0] = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lut_f16));
    lut_tbl2.val[1] = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lut_f16 + 8));
    uint16x8_t v0100 = vdupq_n_u16(0x0100u);
    uint16x8_t v0202 = vdupq_n_u16(0x0202u);
    uint16x8_t v0404 = vdupq_n_u16(0x0404u);

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    uint32_t k = 0;
    for (; k + 16u <= K; k += 16u) {
        float16x8_t wn0 = vmulq_f16(vld1q_f16(rn + k), vr);
        float16x8_t wn1 = vmulq_f16(vld1q_f16(rn + k + 8u), vr);
        uint16x8_t idx0 = nf4_classify8_f16(wn0, vm7, vm3, vm11, lv2_tbl, lv3_tbl, v0100, v0202, v0404);
        uint16x8_t idx1 = nf4_classify8_f16(wn1, vm7, vm3, vm11, lv2_tbl, lv3_tbl, v0100, v0202, v0404);
        float16x8_t q0 = vreinterpretq_f16_u8(vqtbl2q_u8(lut_tbl2, vreinterpretq_u8_u16(vorrq_u16(idx0, v0100))));
        float16x8_t q1 = vreinterpretq_f16_u8(vqtbl2q_u8(lut_tbl2, vreinterpretq_u8_u16(vorrq_u16(idx1, v0100))));
        float16x8_t d0 = vsubq_f16(wn0, q0);
        float16x8_t d1 = vsubq_f16(wn1, q1);
        acc0 = vfmlalq_low_f16(acc0, d0, d0);
        acc1 = vfmlalq_high_f16(acc1, d0, d0);
        acc2 = vfmlalq_low_f16(acc2, d1, d1);
        acc3 = vfmlalq_high_f16(acc3, d1, d1);
    }
    for (; k + 8u <= K; k += 8u) {
        float16x8_t wn0 = vmulq_f16(vld1q_f16(rn + k), vr);
        uint16x8_t idx0 = nf4_classify8_f16(wn0, vm7, vm3, vm11, lv2_tbl, lv3_tbl, v0100, v0202, v0404);
        float16x8_t q0 = vreinterpretq_f16_u8(vqtbl2q_u8(lut_tbl2, vreinterpretq_u8_u16(vorrq_u16(idx0, v0100))));
        float16x8_t d0 = vsubq_f16(wn0, q0);
        acc0 = vfmlalq_low_f16(acc0, d0, d0);
        acc1 = vfmlalq_high_f16(acc1, d0, d0);
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; k < K; ++k) {
        float wn = static_cast<float>(rn[k]) * ratio;
        uint32_t ix = nf4_nearest_scalar(wn, tbl.mid_f32);
        float d = wn - tbl.lut_f32[ix];
        s += d * d;
    }
    return s * (sc * sc);
}

// 100% register-resident 16-weight -> 8-byte NF4 packer (zero stack spills!)
inline void pack_nf4_row_neon(const float* wr, uint32_t K, float inv_sc, const Nf4Tables& tbl, uint8_t* dst) {
    float32x4_t vinv = vdupq_n_f32(inv_sc);
    float16x8_t vm7  = vdupq_n_f16(tbl.mid_h[7]);
    float16x8_t vm3  = vdupq_n_f16(tbl.mid_h[3]);
    float16x8_t vm11 = vdupq_n_f16(tbl.mid_h[11]);
    uint8x16_t lv2_tbl = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lv2_f16));
    uint8x16_t lv3_tbl = vld1q_u8(reinterpret_cast<const uint8_t*>(tbl.lv3_f16));
    uint16x8_t v0100 = vdupq_n_u16(0x0100u);
    uint16x8_t v0202 = vdupq_n_u16(0x0202u);
    uint16x8_t v0404 = vdupq_n_u16(0x0404u);

    uint32_t k = 0;
    for (; k + 16u <= K; k += 16u) {
        float16x8_t wn0 = vcvt_high_f16_f32(
            vcvt_f16_f32(vmulq_f32(vld1q_f32(wr + k +  0u), vinv)),
            vmulq_f32(vld1q_f32(wr + k +  4u), vinv));
        float16x8_t wn1 = vcvt_high_f16_f32(
            vcvt_f16_f32(vmulq_f32(vld1q_f32(wr + k +  8u), vinv)),
            vmulq_f32(vld1q_f32(wr + k + 12u), vinv));
        uint16x8_t idx0 = nf4_classify8_f16(wn0, vm7, vm3, vm11, lv2_tbl, lv3_tbl, v0100, v0202, v0404);
        uint16x8_t idx1 = nf4_classify8_f16(wn1, vm7, vm3, vm11, lv2_tbl, lv3_tbl, v0100, v0202, v0404);
        // idx_bits low byte = 2*idx (even, bit 8 = 0) -> shrn #1 gives exact idx in [0..15]!
        uint8x8_t b0 = vshrn_n_u16(idx0, 1);
        uint8x8_t b1 = vshrn_n_u16(idx1, 1);
        uint8x8_t even = vuzp1_u8(b0, b1);
        uint8x8_t odd  = vuzp2_u8(b0, b1);
        vst1_u8(dst + (k >> 1), vorr_u8(even, vshl_n_u8(odd, 4)));
    }
    for (; k < K; k += 2u) {
        uint32_t q0 = nf4_nearest_scalar(wr[k]     * inv_sc, tbl.mid_f32);
        uint32_t q1 = nf4_nearest_scalar(wr[k + 1] * inv_sc, tbl.mid_f32);
        dst[k >> 1] = static_cast<uint8_t>((q0 & 0xFu) | ((q1 & 0xFu) << 4));
    }
}
#else
inline float eval_nf4_err_scalar(const float* wr, uint32_t K, float sc, float inv_sc, const Nf4Tables& tbl) {
    float err = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t ix = nf4_nearest_scalar(wr[k] * inv_sc, tbl.mid_f32);
        float d = wr[k] - tbl.lut_f32[ix] * sc;
        err += d * d;
    }
    return err;
}
#endif

// Stream-quantize [O, K] -> hardware-verified non-uniform NF4 lut4_a8!
void quantize_slice_nf4(const RawWeightRef& wref, uint32_t row0, uint32_t O,
                        uint32_t col0, uint32_t K,
                        const float* col_scale, const float* inv_row_scale,
                        bool mse_clip, WorkerScratch& ws) {
    const Nf4Tables& tbl = nf4_tables();
    ws.d_i4.resize(static_cast<size_t>(O) * (K / 2u));
    ws.s_f16.resize(O);
    ws.row_f32.resize(K);
    ws.row_f16.resize(K);

    static constexpr float kCoarseP[6] = { 0.60f, 0.68f, 0.76f, 0.84f, 0.92f, 1.00f };
    float* wr = ws.row_f32.data();
    __fp16* rn = ws.row_f16.data();
    (void)rn;

    for (uint32_t r = 0; r < O; ++r) {
        uint32_t gr = row0 + r;
        float rsc = inv_row_scale ? inv_row_scale[gr] : 1.0f;
        float wmin0 = 0.0f, wmax0 = 0.0f;
        dequant_row_slice_scaled(wref, gr, col0, K, col_scale, rsc, wr, wmin0, wmax0);
        float amax0 = std::max(1e-6f, std::max(std::fabs(wmin0), std::fabs(wmax0)));

        uint16_t best_sc_h = f32_to_f16_bits(std::max(amax0, 1e-6f));
        float best_sc = std::max(f16_bits_to_f32(best_sc_h), 1e-6f);

        if (mse_clip && amax0 > 1e-6f) {
            float best_err = 1e38f;
            float best_p = 1.0f;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
            build_norm_f16(wr, K, 1.0f / amax0, rn);
            uint16_t last_sch = 0xFFFFu;
            for (int ci = 0; ci < 6; ++ci) {
                float p = kCoarseP[ci];
                uint16_t sc_h = f32_to_f16_bits(std::max(amax0 * p, 1e-6f));
                if (sc_h == last_sch) continue;
                last_sch = sc_h;
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_nf4_err_f16(rn, K, sc, amax0 / sc, tbl);
                if (err < best_err) { best_err = err; best_p = p; best_sc_h = sc_h; best_sc = sc; }
            }
            float fine_p[2] = { std::clamp(best_p - 0.04f, 0.55f, 1.00f), std::clamp(best_p + 0.04f, 0.55f, 1.00f) };
            for (int fi = 0; fi < 2; ++fi) {
                uint16_t sc_h = f32_to_f16_bits(std::max(amax0 * fine_p[fi], 1e-6f));
                if (sc_h == best_sc_h) continue;
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_nf4_err_f16(rn, K, sc, amax0 / sc, tbl);
                if (err < best_err) { best_err = err; best_sc_h = sc_h; best_sc = sc; }
            }
#else
            for (int ci = 0; ci < 6; ++ci) {
                float p = kCoarseP[ci];
                uint16_t sc_h = f32_to_f16_bits(std::max(amax0 * p, 1e-6f));
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_nf4_err_scalar(wr, K, sc, 1.0f / sc, tbl);
                if (err < best_err) { best_err = err; best_p = p; best_sc_h = sc_h; best_sc = sc; }
            }
            float fine_p[2] = { std::clamp(best_p - 0.04f, 0.55f, 1.00f), std::clamp(best_p + 0.04f, 0.55f, 1.00f) };
            for (int fi = 0; fi < 2; ++fi) {
                uint16_t sc_h = f32_to_f16_bits(std::max(amax0 * fine_p[fi], 1e-6f));
                if (sc_h == best_sc_h) continue;
                float sc = std::max(f16_bits_to_f32(sc_h), 1e-6f);
                float err = eval_nf4_err_scalar(wr, K, sc, 1.0f / sc, tbl);
                if (err < best_err) { best_err = err; best_sc_h = sc_h; best_sc = sc; }
            }
#endif
        }

        ws.s_f16[r] = best_sc_h;
        uint8_t* dst = ws.d_i4.data() + static_cast<size_t>(r) * (K / 2u);
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_FML)
        pack_nf4_row_neon(wr, K, 1.0f / best_sc, tbl, dst);
#else
        float inv_sc = 1.0f / best_sc;
        for (uint32_t k = 0; k < K; k += 2u) {
            uint32_t q0 = nf4_nearest_scalar(wr[k]     * inv_sc, tbl.mid_f32);
            uint32_t q1 = nf4_nearest_scalar(wr[k + 1] * inv_sc, tbl.mid_f32);
            dst[k >> 1] = static_cast<uint8_t>((q0 & 0xFu) | ((q1 & 0xFu) << 4));
        }
#endif
    }
}

uint32_t choose_subblocks(uint32_t inter) {
    for (uint32_t n : {4u, 2u, 1u}) {
        if ((inter % n) == 0u) {
            uint32_t sub = inter / n;
            if ((sub % 256u) == 0u && ((sub / 2u) % 128u) == 0u) return n;
        }
    }
    return 0u;
}

std::vector<int> parse_layers_spec(const std::string& spec, int n_layers) {
    std::vector<int> out;
    if (spec.empty() || spec == "all") {
        out.reserve(static_cast<size_t>(n_layers));
        for (int i = 0; i < n_layers; ++i) out.push_back(i);
        return out;
    }
    auto parse_nonneg = [&](const std::string& tok) -> int {
        if (tok.empty()) throw std::runtime_error("Invalid --layers spec: \"" + spec + "\"");
        int64_t val = 0;
        for (char c : tok) {
            if (c < '0' || c > '9') {
                throw std::runtime_error("Invalid --layers spec: \"" + spec + "\"");
            }
            val = val * 10 + (c - '0');
            if (val > 1000000) {
                throw std::runtime_error("Invalid --layers spec (out of range): \"" + spec + "\"");
            }
        }
        return static_cast<int>(val);
    };
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        if (dash != std::string::npos) {
            if (part.find('-', dash + 1) != std::string::npos) {
                throw std::runtime_error("Invalid --layers range: \"" + part + "\"");
            }
            int a = parse_nonneg(part.substr(0, dash));
            int b = parse_nonneg(part.substr(dash + 1));
            if (a > b) {
                throw std::runtime_error("Invalid --layers range (start > end): \"" + part + "\"");
            }
            for (int i = a; i <= b; ++i) if (i >= 0 && i < n_layers) out.push_back(i);
        } else {
            int v = parse_nonneg(part);
            if (v >= 0 && v < n_layers) out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

float vec_median(const std::vector<float>& v, std::vector<float>& tmp) {
    tmp = v;
    size_t mid = tmp.size() / 2u;
    std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(mid), tmp.end());
    return std::max(tmp[mid], 1e-5f);
}

// Emit complete, valid iOS18 MIL program + weight.bin + coremldata.bin for 1 MLP layer!
// Uses zero-copy 2-pass L1-resident row streaming (20 KB/worker vs old 1.07 GB/worker!)
bool build_one_layer(const ShardPool& pool,
                     const std::unordered_map<std::string, std::string>& wmap,
                     const std::string& pfx,
                     const std::vector<uint8_t>& coremldata_bytes,
                     WorkerScratch& ws,
                     const std::string& out_dir,
                     const std::string& fmt,
                     const std::string& tag,
                     int L,
                     uint32_t HID, uint32_t INTER, uint32_t n_sb,
                     int rows, float alpha, float max_sq_scale, bool mse_clip) {
    const uint32_t PW  = ((HID + 64u) / 64u) * 64u;
    const uint32_t SUB = INTER / n_sb;
    const uint32_t DK  = SUB / 2u;

    std::string name = "L" + std::to_string(L) + "_mlp_p0_" + tag;
    std::string mlc_path = out_dir + "/" + name + ".mlmodelc";
    std::string inv1_path = out_dir + "/pieces_inv1_" + tag + "_L" + std::to_string(L) + ".npy";

    std::string pre = pfx + "layers." + std::to_string(L) + ".mlp.";
    RawWeightRef ref_g, ref_u, ref_d;
    if (!resolve_weight_ref(pool, wmap, pre + "gate_proj", INTER, HID, ref_g)) return false;
    if (!resolve_weight_ref(pool, wmap, pre + "up_proj",   INTER, HID, ref_u)) return false;
    if (!resolve_weight_ref(pool, wmap, pre + "down_proj", HID, INTER, ref_d)) return false;

    // Pass 1: Gentle clamped median-normalized weight-only SmoothQuant directly from 4-bit safetensors (0 DRAM writes!)
    const float exp_w = -(1.0f - alpha);
    const float s_hi = std::max(1.0f, max_sq_scale);
    const float s_lo = 1.0f / s_hi;
    const bool do_sq = (std::fabs(exp_w) > 1e-4f && s_hi > 1.001f);

    ws.inv1_f16.resize(HID);
    const float* s1_ptr = nullptr;
    const float* s2_ptr = nullptr;
    const float* inv_s2_ptr = nullptr;

    if (do_sq) {
        ws.aw1.assign(HID, 1e-5f);
        accumulate_col_amax(ref_g, ws.aw1.data());
        accumulate_col_amax(ref_u, ws.aw1.data());
        float med1 = vec_median(ws.aw1, ws.tmp_med);
        float inv_med1 = 1.0f / med1;
        ws.s1.resize(HID);
        for (uint32_t k = 0; k < HID; ++k) {
            ws.s1[k] = std::clamp(std::pow(ws.aw1[k] * inv_med1, exp_w), s_lo, s_hi);
            ws.inv1_f16[k] = f32_to_f16_bits(1.0f / ws.s1[k]);
        }

        ws.aw2.assign(INTER, 1e-5f);
        accumulate_col_amax(ref_d, ws.aw2.data());
        float med2 = vec_median(ws.aw2, ws.tmp_med);
        float inv_med2 = 1.0f / med2;
        ws.s2.resize(INTER);
        // Reuse ws.aw2 to store inv_s2[INTER] = 1.0f / s2[r] for up_proj row scaling!
        for (uint32_t k = 0; k < INTER; ++k) {
            float s2k = std::clamp(std::pow(ws.aw2[k] * inv_med2, exp_w), s_lo, s_hi);
            ws.s2[k] = s2k;
            ws.aw2[k] = 1.0f / s2k;
        }
        s1_ptr = ws.s1.data();
        s2_ptr = ws.s2.data();
        inv_s2_ptr = ws.aw2.data();
    } else {
        uint16_t one_h = f32_to_f16_bits(1.0f);
        std::fill(ws.inv1_f16.begin(), ws.inv1_f16.end(), one_h);
    }

    // Pass 2: Row-streaming quantization directly into BlobWriter!
    ws.bw.reset();
    std::ostringstream mil;
    const bool is_nf4  = (fmt == "nf4");
    const bool is_lut4 = (fmt == "lut4" || is_nf4);

    ws.z_sub.assign(SUB, 0u);
    ws.z_hid.assign(HID, 0u);
    uint64_t bias_sub_off = ws.bw.add_blob(BlobWriter::kDTypeFP16, ws.z_sub.data(), ws.z_sub.size() * 2u);
    uint64_t bias_hid_off = ws.bw.add_blob(BlobWriter::kDTypeFP16, ws.z_hid.data(), ws.z_hid.size() * 2u);

    uint16_t lut_i4[16];
    for (int i = 0; i < 16; ++i) lut_i4[i] = f32_to_f16_bits(static_cast<float>(i < 8 ? i : i - 16));
    const uint16_t* active_lut = is_nf4 ? nf4_tables().lut_f16 : lut_i4;

    mil << "program(1.3)\n"
        << "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3600.16.1\"}, {\"coremlc-version\", \"3600.25.2\"}})]\n"
        << "{\n"
        << "    func main<ios18>(tensor<fp16, [" << rows << ", " << PW << "]> x) {\n"
        << "            tensor<int32, [2]> slice_by_size_0_begin_0 = const()[name = string(\"slice_by_size_0_begin_0\"), val = tensor<int32, [2]>([0, 0])];\n"
        << "            tensor<int32, [2]> slice_by_size_0_size_0 = const()[name = string(\"slice_by_size_0_size_0\"), val = tensor<int32, [2]>([" << rows << ", " << HID << "])];\n"
        << "            tensor<fp16, [" << rows << ", " << HID << "]> slice_by_size_0 = slice_by_size(begin = slice_by_size_0_begin_0, size = slice_by_size_0_size_0, x = x)[name = string(\"slice_by_size_0\")];\n"
        << "            tensor<int32, [2]> slice_by_size_1_begin_0 = const()[name = string(\"slice_by_size_1_begin_0\"), val = tensor<int32, [2]>([0, " << HID << "])];\n"
        << "            tensor<int32, [2]> slice_by_size_1_size_0 = const()[name = string(\"slice_by_size_1_size_0\"), val = tensor<int32, [2]>([" << rows << ", 1])];\n"
        << "            tensor<fp16, [" << rows << ", 1]> slice_by_size_1 = slice_by_size(begin = slice_by_size_1_begin_0, size = slice_by_size_1_size_0, x = x)[name = string(\"slice_by_size_1\")];\n"
        << "            fp16 q8_scale = const()[name = string(\"q8_scale\"), val = fp16(0x1.02p-7)];\n"
        << "            string q8_dtype = const()[name = string(\"q8_dtype\"), val = string(\"int8\")];\n"
        << "            fp16 eps_c = const()[name = string(\"eps_c\"), val = fp16(0x1.a38p-14)];\n"
        << "            tensor<int32, [1]> rmax_axes = const()[name = string(\"rmax_axes\"), val = tensor<int32, [1]>([-1])];\n"
        << "            bool rmax_kd = const()[name = string(\"rmax_kd\"), val = bool(true)];\n"
        << "            tensor<int32, [2]> split_sizes = const()[name = string(\"split_sizes\"), val = tensor<int32, [2]>([" << DK << ", " << DK << "])];\n"
        << "            int32 split_axis = const()[name = string(\"split_axis\"), val = int32(-1)];\n"
        << "            tensor<fp16, [" << SUB << "]> bias_sub = const()[name = string(\"bias_sub\"), val = tensor<fp16, [" << SUB << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << bias_sub_off << ")))];\n"
        << "            tensor<fp16, [" << HID << "]> bias_hid = const()[name = string(\"bias_hid\"), val = tensor<fp16, [" << HID << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << bias_hid_off << ")))];\n"
        << "            tensor<int8, [" << rows << ", " << HID << "]> x_q8 = quantize(input = slice_by_size_0, output_dtype = q8_dtype, scale = q8_scale)[name = string(\"x_q8\")];\n"
        << "            tensor<fp16, [" << rows << ", " << HID << "]> x_dq = dequantize(input = x_q8, scale = q8_scale)[name = string(\"x_dq\")];\n";

    auto emit_linear = [&](uint32_t li, const RawWeightRef& wref, uint32_t r0, uint32_t O,
                           uint32_t c0, uint32_t K, const float* col_sc, const float* inv_row_sc,
                           const std::string& x_var, const std::string& r_var,
                           const std::string& bias_var, const std::string& out_var) {
        if (is_lut4) {
            if (is_nf4) {
                quantize_slice_nf4(wref, r0, O, c0, K, col_sc, inv_row_sc, mse_clip, ws);
            } else {
                quantize_slice_lut4_sym(wref, r0, O, c0, K, col_sc, inv_row_sc, mse_clip, ws);
            }
            uint64_t ioff = ws.bw.add_blob(BlobWriter::kDTypeUInt4, ws.d_i4.data(), ws.d_i4.size());
            uint64_t loff = ws.bw.add_blob(BlobWriter::kDTypeFP16, active_lut, 32u);
            uint64_t soff = ws.bw.add_blob(BlobWriter::kDTypeFP16, ws.s_f16.data(), ws.s_f16.size() * 2u);
            mil << "            tensor<fp16, [" << O << ", " << K << "]> linear_" << li << "_weight_lut4 = constexpr_lut_to_dense("
                << "indices = tensor<uint4, [" << O << ", " << K << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << ioff << "))), "
                << "lut = tensor<fp16, [1, 1, 16, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << loff << "))))"
                << "[name = string(\"linear_" << li << "_weight_lut4\")];\n";
            mil << "            tensor<fp16, [1, " << O << "]> linear_" << li << "_scale = const()"
                << "[name = string(\"linear_" << li << "_scale\"), val = tensor<fp16, [1, " << O << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << soff << ")))];\n";
            mil << "            tensor<fp16, [" << rows << ", " << O << "]> linear_" << li
                << " = linear(bias = " << bias_var << ", weight = linear_" << li << "_weight_lut4, x = " << x_var << ")[name = string(\"linear_" << li << "\")];\n";
            mil << "            tensor<fp16, [" << rows << ", " << O << "]> linear_" << li << "_sc = mul(x = linear_" << li << ", y = linear_" << li << "_scale)[name = string(\"linear_" << li << "_sc\")];\n";
            mil << "            tensor<fp16, [" << rows << ", " << O << "]> " << out_var << " = mul(x = linear_" << li << "_sc, y = " << r_var << ")[name = string(\"" << out_var << "\")];\n";
        } else {
            quantize_slice_hf4_asym(wref, r0, O, c0, K, col_sc, inv_row_sc, mse_clip, ws);
            uint64_t doff = ws.bw.add_blob(BlobWriter::kDTypeInt4, ws.d_i4.data(), ws.d_i4.size());
            uint64_t soff = ws.bw.add_blob(BlobWriter::kDTypeFP16, ws.s_f16.data(), ws.s_f16.size() * 2u);
            uint64_t ooff = ws.bw.add_blob(BlobWriter::kDTypeInt4, ws.o_i4.data(), ws.o_i4.size());
            mil << "            tensor<fp16, [" << O << ", " << K << "]> linear_" << li << "_weight_0_quantized = constexpr_blockwise_shift_scale("
                << "data = tensor<int4, [" << O << ", " << K << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << doff << "))), "
                << "offset = tensor<int4, [" << O << ", 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << ooff << "))), "
                << "scale = tensor<fp16, [" << O << ", 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(" << soff << "))))"
                << "[name = string(\"linear_" << li << "_weight_0_quantized\")];\n";
            mil << "            tensor<fp16, [" << rows << ", " << O << "]> linear_" << li
                << " = linear(bias = " << bias_var << ", weight = linear_" << li << "_weight_0_quantized, x = " << x_var << ")[name = string(\"linear_" << li << "\")];\n";
            mil << "            tensor<fp16, [" << rows << ", " << O << "]> " << out_var << " = mul(x = linear_" << li << ", y = " << r_var << ")[name = string(\"" << out_var << "\")];\n";
        }
    };

    std::string acc_var;
    for (uint32_t sb = 0; sb < n_sb; ++sb) {
        uint32_t o = sb * SUB;
        std::string sbs = std::to_string(sb);
        std::string g_var = "g_" + sbs;
        std::string u_var = "u_" + sbs;
        emit_linear(sb * 4u + 0u, ref_g, o, SUB, 0u, HID, s1_ptr, nullptr,    "x_dq", "slice_by_size_1", "bias_sub", g_var);
        emit_linear(sb * 4u + 1u, ref_u, o, SUB, 0u, HID, s1_ptr, inv_s2_ptr, "x_dq", "slice_by_size_1", "bias_sub", u_var);

        mil << "            tensor<fp16, [" << rows << ", " << SUB << "]> silu_" << sbs << " = silu(x = " << g_var << ")[name = string(\"silu_" << sbs << "\")];\n"
            << "            tensor<fp16, [" << rows << ", " << SUB << "]> act_" << sbs << " = mul(x = silu_" << sbs << ", y = " << u_var << ")[name = string(\"act_" << sbs << "\")];\n"
            << "            tensor<fp16, [" << rows << ", " << DK << "]> sp_" << sbs << "_0, tensor<fp16, [" << rows << ", " << DK << "]> sp_" << sbs << "_1 = split(axis = split_axis, split_sizes = split_sizes, x = act_" << sbs << ")[name = string(\"split_" << sbs << "\")];\n";

        std::string y_part[2];
        for (uint32_t k = 0; k < 2u; ++k) {
            std::string sk = sbs + "_" + std::to_string(k);
            std::string sp = "sp_" + sk;
            mil << "            tensor<fp16, [" << rows << ", " << DK << "]> abs_" << sk << " = abs(x = " << sp << ")[name = string(\"abs_" << sk << "\")];\n"
                << "            tensor<fp16, [" << rows << ", 1]> rmax_" << sk << " = reduce_max(axes = rmax_axes, keep_dims = rmax_kd, x = abs_" << sk << ")[name = string(\"rmax_" << sk << "\")];\n"
                << "            tensor<fp16, [" << rows << ", 1]> r_" << sk << " = add(x = rmax_" << sk << ", y = eps_c)[name = string(\"r_" << sk << "\")];\n"
                << "            tensor<fp16, [" << rows << ", " << DK << "]> n_" << sk << " = real_div(x = " << sp << ", y = r_" << sk << ")[name = string(\"n_" << sk << "\")];\n"
                << "            tensor<int8, [" << rows << ", " << DK << "]> pq_" << sk << " = quantize(input = n_" << sk << ", output_dtype = q8_dtype, scale = q8_scale)[name = string(\"pq_" << sk << "\")];\n"
                << "            tensor<fp16, [" << rows << ", " << DK << "]> pdq_" << sk << " = dequantize(input = pq_" << sk << ", scale = q8_scale)[name = string(\"pdq_" << sk << "\")];\n";
            y_part[k] = "yk_" + sk;
            emit_linear(sb * 4u + 2u + k, ref_d, 0u, HID, o + k * DK, DK, s2_ptr, nullptr, "pdq_" + sk, "r_" + sk, "bias_hid", y_part[k]);
        }
        std::string sb_sum = "sbsum_" + sbs;
        mil << "            tensor<fp16, [" << rows << ", " << HID << "]> " << sb_sum << " = add(x = " << y_part[0] << ", y = " << y_part[1] << ")[name = string(\"" << sb_sum << "\")];\n";
        if (acc_var.empty()) {
            acc_var = sb_sum;
        } else {
            std::string next_acc = "acc_" + sbs;
            mil << "            tensor<fp16, [" << rows << ", " << HID << "]> " << next_acc << " = add(x = " << acc_var << ", y = " << sb_sum << ")[name = string(\"" << next_acc << "\")];\n";
            acc_var = next_acc;
        }
    }

    mil << "            tensor<fp16, [" << rows << ", " << HID << "]> down_proj = identity(x = " << acc_var << ")[name = string(\"down_proj\")];\n"
        << "        } -> (down_proj);\n"
        << "}\n";

    std::string pid_suf = ".tmp." + std::to_string(getpid());
    std::string tmp_mlc = mlc_path + pid_suf;
    std::string tmp_inv1 = inv1_path + pid_suf;
    std::error_code ec;
    fs::remove_all(tmp_mlc, ec);
    fs::remove(tmp_inv1, ec);
    fs::create_directories(tmp_mlc + "/weights", ec);
    if (ec) return false;

    std::string mstr = mil.str();
    bool ok = true;
    ok = ok && write_bytes_fd(tmp_mlc + "/coremldata.bin", coremldata_bytes.data(), coremldata_bytes.size());
    ok = ok && write_bytes_fd(tmp_mlc + "/model.mil", mstr.data(), mstr.size());
    ok = ok && ws.bw.write_to(tmp_mlc + "/weights/weight.bin");
    ok = ok && write_npy_f16(tmp_inv1, ws.inv1_f16);

    if (!ok) {
        fs::remove_all(tmp_mlc, ec);
        fs::remove(tmp_inv1, ec);
        return false;
    }

    fs::remove_all(mlc_path, ec);
    fs::rename(tmp_mlc, mlc_path, ec);
    if (ec) {
        std::error_code ec_rm;
        fs::remove_all(tmp_mlc, ec_rm);
        fs::remove(tmp_inv1, ec_rm);
        return false;
    }
    fs::rename(tmp_inv1, inv1_path, ec);
    if (ec) {
        std::error_code ec_rm;
        fs::remove(tmp_inv1, ec_rm);
        return false;
    }
    return true;
}

} // namespace

std::string resolve_ane_build_format(const std::string& format) {
    if (format == "hf4" || format == "lut4" || format == "nf4") return format;
    std::string arch = AneMlpEngine::detect_ane_arch();
    if (!arch.empty() &&
        (arch.rfind("h13", 0) == 0 || arch.rfind("h14", 0) == 0 || arch.rfind("h15", 0) == 0)) {
        return "lut4";
    }
    return "hf4";
}

std::string default_ane_cache_dir(const std::string& model_dir, const std::string& fmt) {
    fs::path mp = fs::weakly_canonical(fs::path(model_dir));
    std::string base = mp.filename().string();
    if (base.empty()) base = "model";
    const char* home = std::getenv("HOME");
    std::string root = home ? (std::string(home) + "/.cache/mometal/ane") : "/tmp/mometal-ane";
    return root + "/" + base + "-" + fmt;
}

AneBuildResult build_ane_mlps_native(const AneBuildOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();
    AneBuildResult res;

    try {
        if (!fs::is_directory(opts.model_dir)) {
            res.error = "Model directory not found: " + opts.model_dir;
            return res;
        }
        if (opts.rows <= 0) {
            res.error = "Invalid ANE rows: " + std::to_string(opts.rows);
            return res;
        }

        std::string fmt = resolve_ane_build_format(opts.format);
        std::string tag = (opts.tag.empty() || opts.tag == "auto") ? fmt : opts.tag;
        std::string out_dir = opts.out_dir.empty() ? default_ane_cache_dir(opts.model_dir, fmt) : opts.out_dir;

        res.format = fmt;
        res.tag = tag;
        res.out_dir = out_dir;

        Json cfg = parse_json(read_file_to_string(opts.model_dir + "/config.json"));
        const Json& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;
        const Json* qcfg = cfg.contains("quantization") ? &cfg["quantization"]
                         : (tc.contains("quantization") ? &tc["quantization"] : nullptr);
        if (qcfg && qcfg->is_object() && qcfg->contains("bits")) {
            int qbits = static_cast<int>((*qcfg)["bits"].as_int(4));
            if (qbits != 4) {
                res.error = "Unsupported quantized model bits=" + std::to_string(qbits) +
                            " for ANE-Share builder (only 4-bit MLX or unquantized F16/BF16/F32 supported)";
                return res;
            }
        }
        uint32_t HID   = static_cast<uint32_t>(tc["hidden_size"].as_int(5120));
        uint32_t INTER = static_cast<uint32_t>(tc["intermediate_size"].as_int(17408));
        int n_layers   = static_cast<int>(tc["num_hidden_layers"].as_int(64));
        if (n_layers <= 0) {
            res.error = "Invalid num_hidden_layers=" + std::to_string(n_layers);
            return res;
        }

        uint32_t n_sb = choose_subblocks(INTER);
        if ((HID % 128u) != 0u || n_sb == 0u) {
            res.error = "Unsupported MLP dims for ANE-Share: HID=" + std::to_string(HID) +
                        " INTER=" + std::to_string(INTER);
            return res;
        }

        std::unordered_map<std::string, std::string> wmap;
        std::string idx_path = opts.model_dir + "/model.safetensors.index.json";
        if (fs::is_regular_file(idx_path)) {
            Json idx = parse_json(read_file_to_string(idx_path));
            for (const auto& kv : idx["weight_map"].as_object()) {
                wmap.emplace(kv.first, kv.second.as_string());
            }
        } else {
            std::string single = opts.model_dir + "/model.safetensors";
            RawShard sh;
            if (!sh.open_file(single)) {
                res.error = "Could not open model.safetensors or model.safetensors.index.json in " + opts.model_dir;
                return res;
            }
            for (const auto& kv : sh.metas) wmap.emplace(kv.first, "model.safetensors");
        }

        std::string pfx = "language_model.model.";
        if (!wmap.count(pfx + "layers.0.mlp.gate_proj.weight")) {
            if (wmap.count("model.language_model.layers.0.mlp.gate_proj.weight")) {
                pfx = "model.language_model.";
            } else {
                pfx = "model.";
            }
        }

        std::vector<int> layers = parse_layers_spec(opts.layers_spec, n_layers);
        res.total_layers = static_cast<int>(layers.size());
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            res.error = "Failed to create output directory: " + out_dir;
            return res;
        }

        std::vector<int> todo;
        int cached = 0;
        for (int L : layers) {
            std::string mlc = out_dir + "/L" + std::to_string(L) + "_mlp_p0_" + tag + ".mlmodelc";
            std::string inv = out_dir + "/pieces_inv1_" + tag + "_L" + std::to_string(L) + ".npy";
            if (fs::is_directory(mlc) &&
                fs::is_regular_file(mlc + "/model.mil") &&
                fs::is_regular_file(mlc + "/weights/weight.bin") &&
                fs::is_regular_file(mlc + "/coremldata.bin") &&
                fs::is_regular_file(inv)) {
                ++cached;
            } else {
                todo.push_back(L);
            }
        }
        res.cached_layers = cached;

        if (todo.empty()) {
            res.ok = true;
            res.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return res;
        }

        // Invalidate any stale .hwx_manifest_<tag> before emitting new layer bundles
        fs::remove(out_dir + "/.hwx_manifest_" + tag, ec);

        // Pre-open all needed safetensors shards ONCE in a shared read-only pool!
        ShardPool pool;
        if (!pool.load_needed(opts.model_dir, wmap, pfx, todo)) {
            res.error = "Failed opening safetensors shards in " + opts.model_dir;
            return res;
        }

        const uint32_t PW = ((HID + 64u) / 64u) * 64u;
        const std::vector<uint8_t> coremldata_bytes = make_coremldata_bin(opts.rows, static_cast<int>(PW), static_cast<int>(HID));

        int hw_thr = static_cast<int>(std::thread::hardware_concurrency());
        int req_workers = (opts.workers > 0) ? opts.workers : std::min<int>(12, std::max<int>(4, hw_thr));
        int n_workers = std::max(1, std::min<int>(req_workers, static_cast<int>(todo.size())));

        std::string arch = AneMlpEngine::detect_ane_arch();
        std::fprintf(stderr,
                     "[ane-build] Native C++ NEON streaming builder | ANE=%s | format=%s | tag=%s | dims=(%ux%u, %u sb) | workers=%d | building %zu/%d layers -> %s\n",
                     arch.empty() ? "unknown" : arch.c_str(), fmt.c_str(), tag.c_str(),
                     HID, INTER, n_sb, n_workers, todo.size(), res.total_layers, out_dir.c_str());

        // Dynamic work-stealing queue (eliminates per-worker tail stragglers!)
        std::atomic<size_t> next_todo{0};
        std::atomic<int> done_cnt{0};
        std::atomic<bool> failed{false};
        std::mutex err_mtx;
        std::string err_msg;
        const size_t est_blob_bytes = (3ull * HID * INTER) / 2ull + 2097152ull;

        std::vector<std::thread> threads;
        for (int wi = 0; wi < n_workers; ++wi) {
            threads.emplace_back([&]() {
                WorkerScratch ws;
                ws.bw.reserve(est_blob_bytes);
                while (!failed.load()) {
                    size_t ti = next_todo.fetch_add(1);
                    if (ti >= todo.size()) break;
                    int L = todo[ti];
                    bool ok = build_one_layer(
                        pool, wmap, pfx, coremldata_bytes, ws, out_dir, fmt, tag,
                        L, HID, INTER, n_sb, opts.rows, opts.alpha, opts.max_sq_scale, opts.mse_clip
                    );
                    if (!ok) {
                        failed.store(true);
                        std::lock_guard<std::mutex> lk(err_mtx);
                        if (err_msg.empty()) err_msg = "Failed building ANE MLP layer L" + std::to_string(L);
                        break;
                    }
                    int c = done_cnt.fetch_add(1) + 1;
                    if (c == static_cast<int>(todo.size()) || (c % 8) == 0) {
                        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        std::fprintf(stderr, "[ane-build] %d/%zu layers (%.1fs)\n", c, todo.size(), dt);
                    }
                }
            });
        }
        for (auto& th : threads) th.join();

        res.built_layers = done_cnt.load();
        res.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (failed.load()) {
            res.error = err_msg;
            return res;
        }
        std::fprintf(stderr, "[ane-build] Built %d layers (+%d cached) in %.2fs -> %s\n",
                     res.built_layers, res.cached_layers, res.elapsed_sec, out_dir.c_str());
        res.ok = true;
        return res;
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
        res.elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }
}

} // namespace mi
