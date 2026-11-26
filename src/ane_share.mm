#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ane_share.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mach-o/dyld.h>
#include <mutex>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace mi {

namespace {

constexpr uint32_t LUT4_BYTES = 64u;

struct KernelTensorInfo {
    uint64_t chunk_off[16] = {};
    uint64_t chunk_bytes = 0;
};

std::string current_proc_basename() {
    uint32_t sz = 0;
    _NSGetExecutablePath(nullptr, &sz);
    if (sz == 0) return "";
    std::string buf(sz, '\0');
    if (_NSGetExecutablePath(buf.data(), &sz) != 0) return "";
    return fs::path(buf.c_str()).filename().string();
}

std::string macos_build_version() {
    char buf[64] = {};
    size_t sz = sizeof(buf);
    if (sysctlbyname("kern.osversion", buf, &sz, nullptr, 0) == 0 && sz > 0) {
        return std::string(buf);
    }
    return "";
}

// Parse Mach-O 64-bit ANE model.hwx header (magic 0xbeefface) -> all 16-chunk __KERN* weight tensors
bool parse_hwx_kernel_tensors(std::ifstream& f, size_t file_size, std::vector<KernelTensorInfo>& out) {
    if (file_size < 32) return false;
    std::vector<uint8_t> hdr(std::min<size_t>(file_size, 65536u));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    size_t len = static_cast<size_t>(f.gcount());
    if (len < 32) return false;

    uint32_t magic = 0;
    std::memcpy(&magic, hdr.data(), 4);
    if (magic != 0xbeeffaceu) return false;

    uint32_t ncmds = 0;
    std::memcpy(&ncmds, hdr.data() + 16, 4);

    struct SectInfo {
        std::string seg;
        uint64_t addr = 0;
        uint64_t size = 0;
        uint32_t offset = 0;
    };
    struct SymInfo {
        std::string name;
        uint8_t nsect = 0;
        uint64_t nval = 0;
    };

    std::vector<SectInfo> sects;
    std::vector<SymInfo> syms;
    uint32_t symoff = 0, nsyms = 0, stroff = 0, strsize = 0;
    bool has_symtab = false;

    size_t off = 32;
    for (uint32_t c = 0; c < ncmds && off + 8 <= len; ++c) {
        uint32_t cmd = 0, cs = 0;
        std::memcpy(&cmd, hdr.data() + off, 4);
        std::memcpy(&cs,  hdr.data() + off + 4, 4);
        if (cs < 8 || off + cs > len) break;

        if (cmd == 0x19u && cs >= 72) { // LC_SEGMENT_64
            char segname[17] = {};
            std::memcpy(segname, hdr.data() + off + 8, 16);
            uint32_t nsects = 0;
            std::memcpy(&nsects, hdr.data() + off + 64, 4);
            size_t so = off + 72;
            for (uint32_t s = 0; s < nsects && so + 80 <= off + cs; ++s) {
                SectInfo si;
                si.seg = segname;
                std::memcpy(&si.addr,   hdr.data() + so + 32, 8);
                std::memcpy(&si.size,   hdr.data() + so + 40, 8);
                std::memcpy(&si.offset, hdr.data() + so + 48, 4);
                sects.push_back(std::move(si));
                so += 80;
            }
        } else if (cmd == 0x2u && cs >= 24) { // LC_SYMTAB
            std::memcpy(&symoff,  hdr.data() + off + 8,  4);
            std::memcpy(&nsyms,   hdr.data() + off + 12, 4);
            std::memcpy(&stroff,  hdr.data() + off + 16, 4);
            std::memcpy(&strsize, hdr.data() + off + 20, 4);
            has_symtab = true;
        }
        off += cs;
    }

    if (!has_symtab) return false;
    size_t sym_end = static_cast<size_t>(symoff) + static_cast<size_t>(nsyms) * 16u;
    size_t str_end = static_cast<size_t>(stroff) + static_cast<size_t>(strsize);
    size_t need_end = std::max(sym_end, str_end);
    if (need_end > file_size) return false;
    if (need_end > hdr.size()) {
        hdr.resize(need_end);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(hdr.data()), static_cast<std::streamsize>(need_end));
        if (static_cast<size_t>(f.gcount()) < need_end) return false;
    }

    const char* strs = reinterpret_cast<const char*>(hdr.data() + stroff);
    syms.reserve(nsyms);
    for (uint32_t i = 0; i < nsyms; ++i) {
        const uint8_t* sp = hdr.data() + symoff + static_cast<size_t>(i) * 16u;
        uint32_t strx = 0;
        uint8_t nsect = sp[5];
        uint64_t nval = 0;
        std::memcpy(&strx, sp, 4);
        std::memcpy(&nval, sp + 8, 8);
        if (strx < strsize) {
            SymInfo sym;
            sym.name = std::string(strs + strx, strnlen(strs + strx, strsize - strx));
            sym.nsect = nsect;
            sym.nval = nval;
            syms.push_back(std::move(sym));
        }
    }

    // Iterate ALL __KERN* sections (supports multi-segment __KERN_0 + __KERN_1 layouts)
    out.clear();
    for (int ki = 0; ki < static_cast<int>(sects.size()); ++ki) {
        if (sects[ki].seg.rfind("__KERN", 0) != 0) continue;
        const SectInfo& ks = sects[ki];

        std::vector<uint64_t> addrs;
        std::unordered_map<std::string, std::array<std::pair<bool, uint64_t>, 16>> fam;
        for (const auto& sym : syms) {
            if (sym.nsect != static_cast<uint8_t>(ki + 1)) continue;
            addrs.push_back(sym.nval);
            if (sym.name.size() >= 70 && sym.name[0] == 'K') {
                size_t pos = sym.name.find("_ne_");
                if (pos == 65) {
                    bool all_digits = true;
                    uint64_t idx_u = 0;
                    for (size_t di = 69; di < sym.name.size(); ++di) {
                        unsigned char ch = static_cast<unsigned char>(sym.name[di]);
                        if (!std::isdigit(ch)) { all_digits = false; break; }
                        idx_u = idx_u * 10ull + static_cast<uint64_t>(ch - '0');
                        if (idx_u >= 16ull) break;
                    }
                    if (all_digits && idx_u < 16ull) {
                        std::string hash = sym.name.substr(1, 64);
                        fam[hash][static_cast<size_t>(idx_u)] = {true, sym.nval};
                    }
                }
            }
        }
        addrs.push_back(ks.addr + ks.size);
        std::sort(addrs.begin(), addrs.end());
        addrs.erase(std::unique(addrs.begin(), addrs.end()), addrs.end());

        std::unordered_map<uint64_t, uint64_t> nxt_addr;
        for (size_t i = 0; i + 1 < addrs.size(); ++i) nxt_addr[addrs[i]] = addrs[i + 1];

        for (const auto& kv : fam) {
            bool complete = true;
            for (int c = 0; c < 16; ++c) {
                if (!kv.second[c].first) { complete = false; break; }
            }
            if (!complete) continue;
            KernelTensorInfo kti{};
            uint64_t a0 = kv.second[0].second;
            auto itn = nxt_addr.find(a0);
            if (itn == nxt_addr.end()) continue;
            kti.chunk_bytes = itn->second - a0;
            bool uniform = true;
            for (int c = 0; c < 16; ++c) {
                kti.chunk_off[c] = (kv.second[c].second - ks.addr) + ks.offset;
                if (c > 0 && kti.chunk_off[c] != kti.chunk_off[0] + static_cast<uint64_t>(c) * kti.chunk_bytes) {
                    uniform = false;
                }
            }
            if (uniform) out.push_back(kti);
        }
    }
    std::sort(out.begin(), out.end(), [](const KernelTensorInfo& a, const KernelTensorInfo& b) {
        return a.chunk_off[0] < b.chunk_off[0];
    });
    return !out.empty();
}

struct SrcLinear {
    uint32_t O = 0, K = 0;
    uint64_t w_off = 0;
    uint64_t o_off = 0;   // 0 if symmetric / lut4
    uint64_t s_off = 0;
    uint64_t lut_off = 0; // non-zero for lut4 / nf4
    bool has_offset = false;
    bool is_lut4 = false;
    bool is_nonuniform_lut = false;
    std::array<uint16_t, 16> lut_f16{}; // exact 16-entry FP16 LUT bits when is_lut4
    std::array<uint32_t, 8> fp0{};      // first 8 k-words of group 0 (rows 0..7)
};

bool read_blob_data_offset(std::ifstream& wf, uint64_t meta_off, uint64_t& out_data_off) {
    uint8_t hdr[24];
    wf.seekg(static_cast<std::streamoff>(meta_off));
    wf.read(reinterpret_cast<char*>(hdr), 24);
    if (!wf.good()) return false;
    std::memcpy(&out_data_off, hdr + 16, 8);
    return true;
}

bool pack_group_words(std::ifstream& wf, uint64_t w_data_off, uint32_t K,
                      uint32_t gi, uint32_t k0, uint32_t* out_words, uint32_t n_words) {
    uint32_t row_bytes = K >> 1;
    uint32_t byte_k0 = k0 >> 1;
    uint32_t n_bytes = (n_words + 1u) >> 1;
    uint8_t rbuf_stack[8 * 8];
    std::vector<uint8_t> rbuf_heap;
    uint8_t* rbuf = rbuf_stack;
    if (8u * n_bytes > sizeof(rbuf_stack)) {
        rbuf_heap.resize(8u * n_bytes);
        rbuf = rbuf_heap.data();
    }
    for (uint32_t j = 0; j < 8u; ++j) {
        uint64_t pos = w_data_off + static_cast<uint64_t>(gi * 8u + j) * row_bytes + byte_k0;
        wf.seekg(static_cast<std::streamoff>(pos));
        wf.read(reinterpret_cast<char*>(rbuf + j * n_bytes), n_bytes);
        if (!wf.good()) return false;
    }
    for (uint32_t k = 0; k < n_words; ++k) {
        uint32_t w = 0u;
        for (uint32_t j = 0; j < 8u; ++j) {
            uint8_t b = rbuf[j * n_bytes + (k >> 1)];
            uint32_t nib = ((k & 1u) == 0u) ? (b & 0xFu) : ((b >> 4) & 0xFu);
            w |= (nib << (4u * j));
        }
        out_words[k] = w;
    }
    return true;
}

struct ParsedMlpSource {
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t sub_size = 0;
    uint32_t dk_size = 0;
    uint32_t n_subblocks = 0;
    uint32_t n_linears = 0;
    std::array<SrcLinear, 16> lins{};
    std::vector<float> scales;
    std::vector<float> offs;
    bool is_asym = false;
    bool is_lut4 = false;
    bool is_nonuniform_lut = false;
};

bool read_and_classify_lut4(std::ifstream& wf, uint64_t lut_data_off,
                            std::array<uint16_t, 16>& out_lut_f16, bool& out_nonuniform) {
    wf.seekg(static_cast<std::streamoff>(lut_data_off));
    wf.read(reinterpret_cast<char*>(out_lut_f16.data()), 32);
    if (!wf.good()) return false;
    static constexpr float kTwosComp[16] = {
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f
    };
    bool is_twos = true;
    for (int i = 0; i < 16; ++i) {
        __fp16 h;
        std::memcpy(&h, &out_lut_f16[i], 2);
        if (!std::isfinite(static_cast<float>(h))) return false;
        if (std::fabs(static_cast<float>(h) - kTwosComp[i]) > 1e-3f) is_twos = false;
    }
    out_nonuniform = !is_twos;
    return true;
}

// Fast string_view integer helpers (replaces slow std::regex in parse_source_4bit -> ~50x faster!)
bool parse_u64_at(std::string_view sv, size_t pos, uint64_t& val, size_t* end_pos = nullptr) {
    if (pos >= sv.size() || !std::isdigit(static_cast<unsigned char>(sv[pos]))) return false;
    uint64_t v = 0;
    size_t i = pos;
    while (i < sv.size() && std::isdigit(static_cast<unsigned char>(sv[i]))) {
        v = v * 10ull + static_cast<uint64_t>(sv[i] - '0');
        ++i;
    }
    val = v;
    if (end_pos) *end_pos = i;
    return true;
}

bool parse_blob_offset_after(std::string_view line, std::string_view anchor, uint64_t& out_off) {
    size_t p = line.find(anchor);
    if (p == std::string_view::npos) return false;
    size_t q = line.find("offset = uint64(", p + anchor.size());
    if (q == std::string_view::npos) return false;
    return parse_u64_at(line, q + 16u, out_off);
}

bool parse_tensor_2d_shape(std::string_view line, uint32_t& d0, uint32_t& d1) {
    size_t p = line.find("tensor<fp16, [");
    if (p == std::string_view::npos) return false;
    uint64_t v0 = 0, v1 = 0;
    size_t e0 = 0;
    if (!parse_u64_at(line, p + 14u, v0, &e0)) return false;
    size_t c = line.find(',', e0);
    if (c == std::string_view::npos) return false;
    size_t p1 = c + 1u;
    while (p1 < line.size() && line[p1] == ' ') ++p1;
    if (!parse_u64_at(line, p1, v1)) return false;
    d0 = static_cast<uint32_t>(v0);
    d1 = static_cast<uint32_t>(v1);
    return true;
}

bool parse_source_4bit(const std::string& mlmodelc_dir,
                       uint32_t expected_hid, uint32_t expected_inter,
                       ParsedMlpSource& out) {
    std::ifstream mfs(mlmodelc_dir + "/model.mil", std::ios::binary | std::ios::ate);
    if (!mfs) return false;
    auto msz = mfs.tellg();
    if (msz <= 0) return false;
    std::string mil(static_cast<size_t>(msz), '\0');
    mfs.seekg(0);
    mfs.read(mil.data(), msz);
    if (!mfs) return false;

    bool seen[16] = {};
    int max_li = -1;
    int count = 0;
    out = ParsedMlpSource{};

    struct PendingLut {
        uint32_t O = 0, K = 0;
        uint64_t w_off = 0, lut_off = 0;
    };
    std::unordered_map<int, PendingLut> lut_map;
    std::unordered_map<int, std::pair<uint32_t, uint64_t>> scale_map;

    std::string_view sv(mil);
    size_t pos = 0;
    while (pos < sv.size()) {
        size_t nl = sv.find('\n', pos);
        std::string_view line = (nl == std::string_view::npos) ? sv.substr(pos) : sv.substr(pos, nl - pos);
        pos = (nl == std::string_view::npos) ? sv.size() : (nl + 1u);

        if (line.find("constexpr_blockwise_shift_scale(") != std::string_view::npos) {
            uint32_t O = 0, K = 0;
            if (!parse_tensor_2d_shape(line, O, K)) continue;
            size_t eq = line.find(" = constexpr_blockwise_shift_scale(");
            if (eq == std::string_view::npos) continue;
            std::string_view lhs = line.substr(0, eq);

            uint64_t li_u = 0;
            bool got_li = false;
            size_t lp = lhs.find("linear_");
            if (lp != std::string_view::npos) {
                got_li = parse_u64_at(lhs, lp + 7u, li_u);
            } else {
                lp = lhs.find("constexpr_blockwise_shift_scale_");
                if (lp != std::string_view::npos) got_li = parse_u64_at(lhs, lp + 32u, li_u);
            }
            if (!got_li || li_u >= 16ull) continue;

            uint64_t w_off = 0, s_off = 0, o_off = 0;
            if (!parse_blob_offset_after(line, "data = tensor<int4,", w_off)) continue;
            if (!parse_blob_offset_after(line, "scale = tensor<fp16,", s_off)) continue;
            bool has_off = parse_blob_offset_after(line, "offset = tensor<int4,", o_off);

            int li = static_cast<int>(li_u);
            SrcLinear& sl = out.lins[li];
            sl.O = O;
            sl.K = K;
            sl.w_off = w_off;
            sl.o_off = has_off ? o_off : 0ull;
            sl.s_off = s_off;
            sl.has_offset = has_off;
            sl.is_lut4 = false;
            if (!seen[li]) { seen[li] = true; ++count; max_li = std::max(max_li, li); }
            if (has_off) out.is_asym = true;
        } else if (line.find("constexpr_lut_to_dense(") != std::string_view::npos) {
            uint32_t O = 0, K = 0;
            if (!parse_tensor_2d_shape(line, O, K)) continue;
            size_t eq = line.find(" = constexpr_lut_to_dense(");
            if (eq == std::string_view::npos) continue;
            std::string_view lhs = line.substr(0, eq);
            size_t lp = lhs.find("linear_");
            uint64_t li_u = 0;
            if (lp == std::string_view::npos || !parse_u64_at(lhs, lp + 7u, li_u) || li_u >= 16ull) continue;

            uint64_t w_off = 0, lut_off = 0;
            if (!parse_blob_offset_after(line, "indices = tensor<uint4,", w_off)) continue;
            if (!parse_blob_offset_after(line, "lut = tensor<fp16,", lut_off)) continue;
            lut_map[static_cast<int>(li_u)] = PendingLut{O, K, w_off, lut_off};
        } else if (line.find("_scale = const()") != std::string_view::npos &&
                   line.find("BLOBFILE(") != std::string_view::npos) {
            uint32_t d0 = 0, d1 = 0;
            if (!parse_tensor_2d_shape(line, d0, d1) || d0 != 1u) continue;
            size_t lp = line.find("linear_");
            uint64_t li_u = 0;
            if (lp == std::string_view::npos || !parse_u64_at(line, lp + 7u, li_u) || li_u >= 16ull) continue;
            uint64_t s_off = 0;
            if (!parse_blob_offset_after(line, "BLOBFILE(", s_off)) continue;
            scale_map[static_cast<int>(li_u)] = {d1, s_off};
        }
    }

    if (count == 0 && !lut_map.empty()) {
        for (const auto& kv : lut_map) {
            int li = kv.first;
            auto sit = scale_map.find(li);
            if (sit == scale_map.end() || sit->second.first != kv.second.O) return false;
            SrcLinear& sl = out.lins[li];
            sl.O = kv.second.O;
            sl.K = kv.second.K;
            sl.w_off = kv.second.w_off;
            sl.lut_off = kv.second.lut_off;
            sl.s_off = sit->second.second;
            sl.has_offset = false;
            sl.is_lut4 = true;
            if (!seen[li]) { seen[li] = true; ++count; max_li = std::max(max_li, li); }
            out.is_lut4 = true;
        }
    }

    if (count == 0 || (count % 4) != 0 || max_li != count - 1) return false;
    for (int i = 0; i < count; ++i) if (!seen[i]) return false;

    uint32_t n_sb = static_cast<uint32_t>(count / 4);
    if (n_sb < 1u || n_sb > 4u) return false;

    uint32_t sub = out.lins[0].O;
    uint32_t hid = out.lins[0].K;
    uint32_t dk  = out.lins[2].K;
    uint32_t inter = n_sb * sub;

    if (hid != expected_hid || inter != expected_inter) return false;
    if ((hid % 128u) != 0u || (sub % 256u) != 0u || dk * 2u != sub || (dk % 128u) != 0u) return false;
    for (uint32_t sb = 0; sb < n_sb; ++sb) {
        if (out.lins[sb * 4 + 0].O != sub || out.lins[sb * 4 + 0].K != hid) return false;
        if (out.lins[sb * 4 + 1].O != sub || out.lins[sb * 4 + 1].K != hid) return false;
        if (out.lins[sb * 4 + 2].O != hid || out.lins[sb * 4 + 2].K != dk)  return false;
        if (out.lins[sb * 4 + 3].O != hid || out.lins[sb * 4 + 3].K != dk)  return false;
    }

    std::ifstream wf(mlmodelc_dir + "/weights/weight.bin", std::ios::binary);
    if (!wf) return false;

    out.hidden_size = hid;
    out.intermediate_size = inter;
    out.sub_size = sub;
    out.dk_size = dk;
    out.n_subblocks = n_sb;
    out.n_linears = static_cast<uint32_t>(count);

    size_t total_so = 2ull * inter + 2ull * n_sb * hid;
    out.scales.assign(total_so, 0.0f);
    out.offs.assign(total_so, 1032.0f);

    for (uint32_t li = 0; li < out.n_linears; ++li) {
        SrcLinear& sl = out.lins[li];
        if (!read_blob_data_offset(wf, sl.w_off, sl.w_off)) return false;
        if (!read_blob_data_offset(wf, sl.s_off, sl.s_off)) return false;
        if (sl.has_offset && !read_blob_data_offset(wf, sl.o_off, sl.o_off)) return false;
        if (sl.is_lut4) {
            if (!read_blob_data_offset(wf, sl.lut_off, sl.lut_off)) return false;
            if (!read_and_classify_lut4(wf, sl.lut_off, sl.lut_f16, sl.is_nonuniform_lut)) return false;
            if (sl.is_nonuniform_lut) out.is_nonuniform_lut = true;
        }

        if (!pack_group_words(wf, sl.w_off, sl.K, 0u, 0u, sl.fp0.data(), 8u)) return false;

        uint32_t sb = li >> 2;
        uint32_t role = li & 3u;
        size_t dst_base = 0;
        if (role == 0u)      dst_base = static_cast<size_t>(sb) * sub;
        else if (role == 1u) dst_base = static_cast<size_t>(inter) + static_cast<size_t>(sb) * sub;
        else                 dst_base = 2ull * inter + static_cast<size_t>(sb * 2u + (role - 2u)) * hid;

        std::vector<uint16_t> s_raw(sl.O);
        wf.seekg(static_cast<std::streamoff>(sl.s_off));
        wf.read(reinterpret_cast<char*>(s_raw.data()), static_cast<std::streamsize>(sl.O * sizeof(uint16_t)));
        if (!wf.good()) return false;
        for (uint32_t i = 0; i < sl.O; ++i) {
            __fp16 h;
            std::memcpy(&h, &s_raw[i], 2);
            out.scales[dst_base + i] = static_cast<float>(h);
        }
        if (sl.is_lut4 && sl.is_nonuniform_lut) {
            for (uint32_t i = 0; i < sl.O; ++i) out.offs[dst_base + i] = 0.0f;
        } else if (sl.has_offset) {
            std::vector<uint8_t> o_raw((sl.O + 1u) >> 1);
            wf.seekg(static_cast<std::streamoff>(sl.o_off));
            wf.read(reinterpret_cast<char*>(o_raw.data()), static_cast<std::streamsize>(o_raw.size()));
            if (!wf.good()) return false;
            for (uint32_t i = 0; i < sl.O; ++i) {
                uint8_t b = o_raw[i >> 1];
                int nib = ((i & 1u) == 0u) ? (b & 0xF) : ((b >> 4) & 0xF);
                if (nib > 7) nib -= 16;
                out.offs[dst_base + i] = 1032.0f + static_cast<float>(nib);
            }
        }
    }
    return wf.good();
}

uint64_t expected_chunk_bytes(uint32_t O, uint32_t K, uint32_t hdr_bytes) {
    uint64_t raw = LUT4_BYTES + static_cast<uint64_t>(O / 128u) * (hdr_bytes + 4ull * K);
    return ((raw + 63ull) / 64ull) * 64ull;
}

// Verify ALL N/N linears bit-exact (group 0 + mid + last group + 32B LUT check for NF4) and bind zero-copy MTLBuffer
bool verify_and_bind_hwx(id<MTLDevice> device, MetalContext& ctx, size_t page_sz,
                         const std::string& mdir, const ParsedMlpSource& src,
                         const std::string& hwx_path, const std::vector<KernelTensorInfo>& kts,
                         AneShareLayer& dst) {
    int fd = open(hwx_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    size_t fsz = static_cast<size_t>(st.st_size);
    size_t aligned_sz = ((fsz + page_sz - 1) / page_sz) * page_sz;
    void* ptr = mmap(nullptr, aligned_sz, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) ptr = mmap(nullptr, aligned_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return false;

    const uint8_t* hbytes = reinterpret_cast<const uint8_t*>(ptr);
    uint32_t base_w[16] = {}, chunk_w[16] = {}, hdr_w[16] = {};
    bool all_ok = true;
    std::ifstream wf(mdir + "/weights/weight.bin", std::ios::binary);
    if (!wf) {
        munmap(ptr, aligned_sz);
        return false;
    }

    const std::array<uint32_t, 3> cand_hdrs = src.is_lut4
        ? std::array<uint32_t, 3>{16u, 0u, 24u}
        : (src.is_asym ? std::array<uint32_t, 3>{24u, 16u, 0u}
                       : std::array<uint32_t, 3>{16u, 0u, 24u});

    for (uint32_t li = 0; li < src.n_linears; ++li) {
        const SrcLinear& sl = src.lins[li];
        bool lin_ok = false;
        for (const auto& kt : kts) {
            for (uint32_t hb : cand_hdrs) {
                if (kt.chunk_bytes != expected_chunk_bytes(sl.O, sl.K, hb)) continue;
                uint64_t w0_off = kt.chunk_off[0] + LUT4_BYTES + hb;
                if (w0_off + 32 > fsz) continue;
                if (std::memcmp(hbytes + w0_off, sl.fp0.data(), 32) != 0) continue;

                // For non-uniform LUT4 (e.g. NF4), require bit-exact 32B FP16 LUT match on ALL 16 chunks!
                if (sl.is_lut4 && sl.is_nonuniform_lut) {
                    bool lut_ok = true;
                    for (int ch = 0; ch < 16; ++ch) {
                        if (kt.chunk_off[ch] + 32 > fsz ||
                            std::memcmp(hbytes + kt.chunk_off[ch], sl.lut_f16.data(), 32) != 0) {
                            lut_ok = false;
                            break;
                        }
                    }
                    if (!lut_ok) continue;
                }

                // Bit-exact spot verification on mid-group & last-group!
                uint32_t g_bytes = hb + 4u * sl.K;
                uint32_t n_groups = sl.O / 8u;
                uint32_t check_gis[2] = { (n_groups / 2u), (n_groups - 1u) };
                bool spot_ok = true;
                for (uint32_t cgi : check_gis) {
                    uint32_t ch = cgi & 15u;
                    uint32_t slot = cgi >> 4;
                    uint32_t ref_w[8];
                    if (!pack_group_words(wf, sl.w_off, sl.K, cgi, sl.K - 8u, ref_w, 8u)) {
                        spot_ok = false;
                        break;
                    }
                    uint64_t hoff = kt.chunk_off[ch] + LUT4_BYTES + static_cast<uint64_t>(slot) * g_bytes
                                    + hb + static_cast<uint64_t>(sl.K - 8u) * 4u;
                    if (hoff + 32 > fsz || std::memcmp(hbytes + hoff, ref_w, 32) != 0) {
                        spot_ok = false;
                        break;
                    }
                }
                if (spot_ok) {
                    base_w[li]  = static_cast<uint32_t>(kt.chunk_off[0] / 4u);
                    chunk_w[li] = static_cast<uint32_t>(kt.chunk_bytes / 4u);
                    hdr_w[li]   = hb / 4u;
                    lin_ok = true;
                    break;
                }
            }
            if (lin_ok) break;
        }
        if (!lin_ok) { all_ok = false; break; }
    }

    if (!all_ok) {
        munmap(ptr, aligned_sz);
        return false;
    }

    id<MTLBuffer> mbuf = [device newBufferWithBytesNoCopy:ptr
                                                   length:aligned_sz
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!mbuf) {
        munmap(ptr, aligned_sz);
        return false;
    }

    size_t n_so = src.scales.size();
    id<MTLBuffer> so_buf = ctx.alloc_shared(n_so * 2 * sizeof(float));
    float* so_ptr = reinterpret_cast<float*>([so_buf contents]);
    for (size_t i = 0; i < n_so; ++i) {
        so_ptr[2 * i + 0] = src.scales[i];
        so_ptr[2 * i + 1] = src.offs[i];
    }

    id<MTLBuffer> lut_buf = nil;
    if (src.is_lut4) {
        lut_buf = ctx.alloc_shared(16u * 16u * sizeof(uint16_t));
        uint16_t* lp = reinterpret_cast<uint16_t*>([lut_buf contents]);
        for (uint32_t li = 0; li < src.n_linears; ++li) {
            std::memcpy(lp + li * 16u, src.lins[li].lut_f16.data(), 32);
        }
    }

    dst.hwx_buf = mbuf;
    dst.mmap_ptr = ptr;
    dst.mapped_size = aligned_sz;
    dst.buf_scales_offs = so_buf;
    dst.buf_lut16 = lut_buf;
    dst.is_lut4 = src.is_lut4;
    dst.is_nonuniform_lut = src.is_nonuniform_lut;
    dst.hidden_size = src.hidden_size;
    dst.intermediate_size = src.intermediate_size;
    dst.sub_size = src.sub_size;
    dst.dk_size = src.dk_size;
    dst.n_subblocks = src.n_subblocks;
    std::memcpy(dst.lin_base_w, base_w, sizeof(base_w));
    std::memcpy(dst.lin_chunk_w, chunk_w, sizeof(chunk_w));
    std::memcpy(dst.lin_hdr_w, hdr_w, sizeof(hdr_w));
    dst.ready = true;
    return true;
}

void scan_hwx_under_dir(const fs::path& dir, uint64_t min_hwx_sz, uint64_t max_hwx_sz,
                        uint64_t min_i8_sz, uint64_t max_i8_sz,
                        std::unordered_set<std::string>& seen_paths,
                        std::vector<std::string>& out_paths,
                        std::vector<uint64_t>& out_sizes,
                        bool& saw_double_size) {
    std::error_code ec_rec;
    for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec_rec);
         it != fs::recursive_directory_iterator(); it.increment(ec_rec)) {
        if (ec_rec) { ec_rec.clear(); continue; }
        std::error_code ec_f;
        if (!it->is_regular_file(ec_f) || ec_f) continue;
        if (it->path().filename() != "model.hwx") continue;
        auto sz = it->file_size(ec_f);
        if (ec_f) continue;
        if (sz >= min_hwx_sz && sz <= max_hwx_sz) {
            std::string p = it->path().string();
            if (seen_paths.insert(p).second) {
                out_paths.push_back(std::move(p));
                out_sizes.push_back(static_cast<uint64_t>(sz));
            }
        } else if (sz >= min_i8_sz && sz <= max_i8_sz) {
            saw_double_size = true;
        }
    }
}

} // namespace

AneShareManager::AneShareManager(MetalContext& ctx, const std::string& ane_models_dir,
                                 const std::string& tag, int num_layers,
                                 uint32_t hidden_size, uint32_t intermediate_size) {
    auto t0 = std::chrono::steady_clock::now();
    layers_.resize(static_cast<size_t>(num_layers));
    if (num_layers <= 0) return;

    // Probe L0 source MIL first to determine exact sub-block dimensions and format (hf4 / lut4 / nf4-lut4)
    ParsedMlpSource l0_probe;
    std::string l0_dir = ane_models_dir + "/L0_mlp_p0_" + tag + ".mlmodelc";
    if (!parse_source_4bit(l0_dir, hidden_size, intermediate_size, l0_probe)) {
        std::fprintf(stderr,
                     "[ane-share] %s is not a supported single-copy 4-bit MLP (expected hf4 int4, lut4, or nf4)\n",
                     l0_dir.c_str());
        return;
    }
    is_lut4_ = l0_probe.is_lut4;
    is_nonuniform_lut_ = l0_probe.is_nonuniform_lut;
    format_name_ = l0_probe.is_lut4
        ? (l0_probe.is_nonuniform_lut ? "nf4-lut4" : "lut4")
        : (l0_probe.is_asym ? "hf4-asym" : "hf4-sym");

    id<MTLDevice> device = ctx.device();
    size_t page_sz = static_cast<size_t>(getpagesize());
    std::vector<std::string> matched_hwx_paths(static_cast<size_t>(num_layers));
    std::vector<uint64_t> matched_hwx_sizes(static_cast<size_t>(num_layers), 0ull);
    std::vector<ParsedMlpSource> parsed_srcs(static_cast<size_t>(num_layers));
    std::vector<bool> parsed_ok(static_cast<size_t>(num_layers), false);

    // Pre-parse all N source MILs in parallel (no regex -> ~2-4 ms total for 64 layers!)
    {
        parsed_srcs[0] = std::move(l0_probe);
        parsed_ok[0] = true;
        std::atomic<int> next_L{1};
        int n_thr = std::min<int>(8, std::max<int>(1, num_layers - 1));
        std::vector<std::thread> ths;
        for (int ti = 0; ti < n_thr; ++ti) {
            ths.emplace_back([&]() {
                while (true) {
                    int L = next_L.fetch_add(1);
                    if (L >= num_layers) break;
                    std::string mdir = ane_models_dir + "/L" + std::to_string(L) + "_mlp_p0_" + tag + ".mlmodelc";
                    bool ok = parse_source_4bit(mdir, hidden_size, intermediate_size, parsed_srcs[L]);
                    if (ok && (parsed_srcs[L].is_lut4 != is_lut4_ ||
                               parsed_srcs[L].is_nonuniform_lut != is_nonuniform_lut_)) {
                        ok = false;
                    }
                    parsed_ok[L] = ok;
                }
            });
        }
        for (auto& th : ths) th.join();
    }

    const uint32_t probe_sub = parsed_srcs[0].sub_size;
    const uint32_t probe_hid = parsed_srcs[0].hidden_size;
    const uint32_t min_kts   = parsed_srcs[0].n_linears;

    // -----------------------------------------------------------------------
    // TIER 1: O(1) .hwx_manifest cache (0 directory scans; ALWAYS bit-exact verified!)
    // -----------------------------------------------------------------------
    const std::string manifest_path = ane_models_dir + "/.hwx_manifest_" + tag;
    bool manifest_hit_all = false;
    {
        std::vector<std::pair<int, std::string>> hints;
        std::ifstream mf(manifest_path);
        if (mf) {
            std::string hdr_line;
            if (std::getline(mf, hdr_line)) {
                std::string magic, mfmt;
                uint32_t mhid = 0, minter = 0;
                int mnl = 0;
                std::istringstream iss(hdr_line);
                if ((iss >> magic >> mhid >> minter >> mnl >> mfmt) &&
                    magic == "MIHWX1" && mhid == hidden_size && minter == intermediate_size &&
                    mnl == num_layers && mfmt == format_name_) {
                    std::vector<bool> seen_hint(static_cast<size_t>(num_layers), false);
                    std::string line;
                    while (std::getline(mf, line)) {
                        size_t t1 = line.find('\t');
                        size_t t2 = (t1 == std::string::npos) ? std::string::npos : line.find('\t', t1 + 1);
                        if (t1 == std::string::npos || t2 == std::string::npos) continue;
                        uint64_t L_u = 0;
                        size_t l_end = 0;
                        std::string_view l_sv(line.data(), t1);
                        if (!parse_u64_at(l_sv, 0, L_u, &l_end) || l_end != l_sv.size() ||
                            L_u >= static_cast<uint64_t>(num_layers)) {
                            continue;
                        }
                        int L = static_cast<int>(L_u);
                        std::string hp = line.substr(t2 + 1);
                        if (!hp.empty() && !seen_hint[static_cast<size_t>(L)]) {
                            seen_hint[static_cast<size_t>(L)] = true;
                            hints.emplace_back(L, std::move(hp));
                        }
                    }
                }
            }
        }

        if (!hints.empty()) {
            std::atomic<size_t> next_i{0};
            std::atomic<int> ok_cnt{0};
            int n_thr = std::min<int>(8, static_cast<int>(hints.size()));
            std::vector<std::thread> ths;
            for (int ti = 0; ti < n_thr; ++ti) {
                ths.emplace_back([&]() {
                    @autoreleasepool {
                        while (true) {
                            size_t idx = next_i.fetch_add(1);
                            if (idx >= hints.size()) break;
                            int L = hints[idx].first;
                            const std::string& hp = hints[idx].second;
                            if (!parsed_ok[L] || layers_[L].ready) continue;
                            struct stat st{};
                            if (stat(hp.c_str(), &st) != 0) continue;
                            size_t fsz = static_cast<size_t>(st.st_size);
                            std::ifstream f(hp, std::ios::binary);
                            if (!f) continue;
                            std::vector<KernelTensorInfo> kts;
                            if (!parse_hwx_kernel_tensors(f, fsz, kts) || kts.size() < min_kts) continue;
                            std::string mdir = ane_models_dir + "/L" + std::to_string(L) + "_mlp_p0_" + tag + ".mlmodelc";
                            if (verify_and_bind_hwx(device, ctx, page_sz, mdir, parsed_srcs[L], hp, kts, layers_[L])) {
                                matched_hwx_paths[L] = hp;
                                matched_hwx_sizes[L] = fsz;
                                ok_cnt.fetch_add(1);
                            }
                        }
                    }
                });
            }
            for (auto& th : ths) th.join();
            shared_count_ = ok_cnt.load();
            manifest_hit_all = (shared_count_ == num_layers);
        }
    }

    // -----------------------------------------------------------------------
    // TIER 2 (own-process owner dir) & TIER 3 (full ModelAssetsCache fallback)
    // -----------------------------------------------------------------------
    if (!manifest_hit_all) {
        const uint64_t raw_4bit_bytes = (3ull * hidden_size * intermediate_size) / 2ull;
        const uint64_t min_hwx_sz = std::max<uint64_t>(65536ull, (raw_4bit_bytes * 85ull) / 100ull);
        const uint64_t max_hwx_sz = (raw_4bit_bytes * 135ull) / 100ull + 2097152ull;
        const uint64_t min_i8_sz  = (raw_4bit_bytes * 175ull) / 100ull;
        const uint64_t max_i8_sz  = (raw_4bit_bytes * 235ull) / 100ull + 4194304ull;

        const std::string aned_root = "/Library/Caches/com.apple.aned";
        std::unordered_set<std::string> seen_paths;
        bool saw_double_size_hwx = false;

        auto run_tier = [&](const std::vector<std::string>& hwx_paths, const std::vector<uint64_t>& hwx_sizes) {
            if (hwx_paths.empty()) return;
            struct HwxIndexEntry {
                std::string path;
                uint64_t size = 0;
                std::vector<KernelTensorInfo> kts;
            };
            std::vector<HwxIndexEntry> hwx_entries(hwx_paths.size());
            std::unordered_map<uint64_t, std::vector<size_t>> fp_to_hwx;
            std::mutex map_mtx;

            {
                std::atomic<size_t> next_idx{0};
                int n_thr = std::min<int>(8, static_cast<int>(hwx_paths.size()));
                std::vector<std::thread> ths;
                for (int ti = 0; ti < n_thr; ++ti) {
                    ths.emplace_back([&]() {
                        while (true) {
                            size_t idx = next_idx.fetch_add(1);
                            if (idx >= hwx_paths.size()) break;
                            std::ifstream f(hwx_paths[idx], std::ios::binary);
                            if (!f) continue;
                            std::vector<KernelTensorInfo> kts;
                            if (!parse_hwx_kernel_tensors(f, static_cast<size_t>(hwx_sizes[idx]), kts) ||
                                kts.size() < min_kts) {
                                continue;
                            }
                            hwx_entries[idx].path = hwx_paths[idx];
                            hwx_entries[idx].size = hwx_sizes[idx];
                            hwx_entries[idx].kts = std::move(kts);

                            std::vector<uint64_t> keys;
                            for (const auto& kt : hwx_entries[idx].kts) {
                                // Check all valid header sizes without breaking early (handles sub=256 where 16B & 24B align identically)
                                for (uint32_t hdr_b : {24u, 16u, 0u}) {
                                    if (kt.chunk_bytes == expected_chunk_bytes(probe_sub, probe_hid, hdr_b)) {
                                        uint64_t fp64 = 0;
                                        f.seekg(static_cast<std::streamoff>(kt.chunk_off[0] + LUT4_BYTES + hdr_b));
                                        f.read(reinterpret_cast<char*>(&fp64), 8);
                                        if (f.good()) keys.push_back(fp64);
                                    }
                                }
                            }
                            std::lock_guard<std::mutex> lk(map_mtx);
                            for (uint64_t k : keys) fp_to_hwx[k].push_back(idx);
                        }
                    });
                }
                for (auto& th : ths) th.join();
            }

            std::vector<int> unmatched;
            for (int L = 0; L < num_layers; ++L) {
                if (!layers_[L].ready && parsed_ok[L]) unmatched.push_back(L);
            }
            if (unmatched.empty()) return;

            std::atomic<size_t> next_u{0};
            std::atomic<int> newly_matched{0};
            int n_thr = std::min<int>(8, static_cast<int>(unmatched.size()));
            std::vector<std::thread> workers;
            for (int ti = 0; ti < n_thr; ++ti) {
                workers.emplace_back([&]() {
                    @autoreleasepool {
                        while (true) {
                            size_t ui = next_u.fetch_add(1);
                            if (ui >= unmatched.size()) break;
                            int L = unmatched[ui];
                            const ParsedMlpSource& src = parsed_srcs[L];

                            uint64_t fp0_key = 0;
                            std::memcpy(&fp0_key, src.lins[0].fp0.data(), 8);
                            auto it_cands = fp_to_hwx.find(fp0_key);
                            if (it_cands == fp_to_hwx.end()) continue;

                            std::string mdir = ane_models_dir + "/L" + std::to_string(L) + "_mlp_p0_" + tag + ".mlmodelc";
                            for (size_t hi : it_cands->second) {
                                const auto& entry = hwx_entries[hi];
                                if (verify_and_bind_hwx(device, ctx, page_sz, mdir, src, entry.path, entry.kts, layers_[L])) {
                                    matched_hwx_paths[L] = entry.path;
                                    matched_hwx_sizes[L] = entry.size;
                                    newly_matched.fetch_add(1);
                                    break;
                                }
                            }
                        }
                    }
                });
            }
            for (auto& th : workers) th.join();
            shared_count_ += newly_matched.load();
        };

        // Tier 2: Scan ONLY own-process ModelAssetsCache/<proc>_* first (64 files vs 681+!)
        const bool can_read_aned = (access(aned_root.c_str(), R_OK | X_OK) == 0);
        if (!can_read_aned) {
            std::fprintf(stderr,
                         "[ane-share] Cannot read %s (%s). Single-copy ANE-Share requires SIP disabled + read ACL:\n"
                         "[ane-share]   sudo chmod -R +a \"user:$USER allow list,search,readattr,readextattr,readsecurity,file_inherit,directory_inherit\" %s\n"
                         "[ane-share] (GPU-only fallback works out of the box with SIP enabled.)\n",
                         aned_root.c_str(), std::strerror(errno), aned_root.c_str());
        }
        std::string proc = current_proc_basename();
        std::string osver = macos_build_version();
        std::error_code ec_root;
        if (can_read_aned && !proc.empty() && fs::exists(aned_root, ec_root) && !ec_root) {
            std::vector<std::string> own_paths;
            std::vector<uint64_t> own_sizes;
            std::vector<fs::path> ver_dirs;
            std::error_code ec_os;
            if (!osver.empty() && fs::is_directory(fs::path(aned_root) / osver / "ModelAssetsCache", ec_os) && !ec_os) {
                ver_dirs.push_back(fs::path(aned_root) / osver / "ModelAssetsCache");
            }
            std::error_code ec_v;
            for (auto vit = fs::directory_iterator(aned_root, fs::directory_options::skip_permission_denied, ec_v);
                 !ec_v && vit != fs::directory_iterator(); vit.increment(ec_v)) {
                fs::path mac = vit->path() / "ModelAssetsCache";
                if (!ver_dirs.empty() && mac == ver_dirs[0]) continue;
                std::error_code ec_m;
                if (fs::is_directory(mac, ec_m) && !ec_m) ver_dirs.push_back(mac);
            }
            std::string proc_pfx = proc + "_";
            for (const auto& mac : ver_dirs) {
                std::error_code ec_o;
                for (auto oit = fs::directory_iterator(mac, fs::directory_options::skip_permission_denied, ec_o);
                     !ec_o && oit != fs::directory_iterator(); oit.increment(ec_o)) {
                    std::error_code ec_od;
                    if (!oit->is_directory(ec_od) || ec_od) continue;
                    std::string oname = oit->path().filename().string();
                    if (oname.rfind(proc_pfx, 0) == 0 || oname == proc) {
                        scan_hwx_under_dir(oit->path(), min_hwx_sz, max_hwx_sz, min_i8_sz, max_i8_sz,
                                           seen_paths, own_paths, own_sizes, saw_double_size_hwx);
                    }
                }
            }
            run_tier(own_paths, own_sizes);
        }

        // Tier 3: Full fallback scan across ALL ModelAssetsCache owner dirs if any layers remain unmatched
        if (can_read_aned && shared_count_ < num_layers && fs::exists(aned_root, ec_root) && !ec_root) {
            std::vector<std::string> all_paths;
            std::vector<uint64_t> all_sizes;
            std::error_code ec_ver;
            for (auto vit = fs::directory_iterator(aned_root, fs::directory_options::skip_permission_denied, ec_ver);
                 !ec_ver && vit != fs::directory_iterator(); vit.increment(ec_ver)) {
                std::error_code ec_vdir;
                if (!vit->is_directory(ec_vdir) || ec_vdir) continue;
                fs::path mac = vit->path() / "ModelAssetsCache";
                std::error_code ec_mdir;
                if (!fs::is_directory(mac, ec_mdir) || ec_mdir) continue;

                std::error_code ec_own;
                for (auto oit = fs::directory_iterator(mac, fs::directory_options::skip_permission_denied, ec_own);
                     !ec_own && oit != fs::directory_iterator(); oit.increment(ec_own)) {
                    std::error_code ec_odir;
                    if (!oit->is_directory(ec_odir) || ec_odir) continue;
                    scan_hwx_under_dir(oit->path(), min_hwx_sz, max_hwx_sz, min_i8_sz, max_i8_sz,
                                       seen_paths, all_paths, all_sizes, saw_double_size_hwx);
                }
            }
            if (all_paths.empty() && shared_count_ == 0) {
                std::fprintf(stderr, "[ane-share] No readable 4-bit model.hwx files (%.1f-%.1f MB) found in %s\n",
                             min_hwx_sz / (1024.0 * 1024.0), max_hwx_sz / (1024.0 * 1024.0), aned_root.c_str());
                if (saw_double_size_hwx && !is_lut4_) {
                    std::fprintf(stderr,
                                 "[ane-share] Hint: found 2x-size INT8-promoted model.hwx (expected for hf4 on M1/M2/M3). "
                                 "Rebuild ANE MLPs with `--format lut4` or `--format nf4` for native 4-bit M1-M5 sharing.\n");
                }
            } else {
                run_tier(all_paths, all_sizes);
            }
        }

        // Persist atomic .hwx_manifest on complete match so next startup skips all directory scans!
        if (shared_count_ == num_layers) {
            std::string tmp_mf = manifest_path + ".tmp." + std::to_string(getpid());
            std::ofstream ofs(tmp_mf, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs << "MIHWX1 " << hidden_size << " " << intermediate_size << " "
                    << num_layers << " " << format_name_ << "\n";
                for (int L = 0; L < num_layers; ++L) {
                    ofs << L << "\t" << matched_hwx_sizes[L] << "\t" << matched_hwx_paths[L] << "\n";
                }
                ofs.close();
                if (ofs.good()) {
                    std::error_code ec_rn;
                    fs::rename(tmp_mf, manifest_path, ec_rn);
                    if (ec_rn) {
                        std::error_code ec_rm;
                        fs::remove(tmp_mf, ec_rm);
                    }
                } else {
                    std::error_code ec_rm;
                    fs::remove(tmp_mf, ec_rm);
                }
            }
        }
    }

    load_time_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[ane-share] Verified %d/%d %s MLP layers (single-copy model.hwx%s) in %.1f ms\n",
                 shared_count_, num_layers, format_name_.c_str(),
                 manifest_hit_all ? ", manifest" : "", load_time_ms_);
}

AneShareManager::~AneShareManager() {
    for (auto& l : layers_) {
        l.hwx_buf = nil;
        l.buf_scales_offs = nil;
        l.buf_folded_norm = nil;
        l.buf_lut16 = nil;
        if (l.mmap_ptr && l.mmap_ptr != MAP_FAILED) {
            munmap(l.mmap_ptr, l.mapped_size);
            l.mmap_ptr = nullptr;
        }
    }
}

} // namespace mi
