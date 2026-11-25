#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ane_share.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace fs = std::filesystem;

namespace mi {

namespace {

constexpr uint32_t HID = 5120u;
constexpr uint32_t SUB = 4352u;
constexpr uint32_t DK  = 2176u;
constexpr uint32_t LUT4_BYTES = 64u;

struct KernelTensorInfo {
    uint64_t chunk_off[16] = {};
    uint64_t chunk_bytes = 0;
};

// Parse Mach-O 64-bit ANE model.hwx header (magic 0xbeefface) -> 16-chunk __KERN weight tensors
bool parse_hwx_kernel_tensors(const uint8_t* buf, size_t len, std::vector<KernelTensorInfo>& out) {
    if (len < 32) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, buf, 4);
    if (magic != 0xbeeffaceu) return false;

    uint32_t ncmds = 0;
    std::memcpy(&ncmds, buf + 16, 4);

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
    size_t off = 32;
    for (uint32_t c = 0; c < ncmds && off + 8 <= len; ++c) {
        uint32_t cmd = 0, cs = 0;
        std::memcpy(&cmd, buf + off, 4);
        std::memcpy(&cs,  buf + off + 4, 4);
        if (cs < 8 || off + cs > len) break;

        if (cmd == 0x19u && cs >= 72) { // LC_SEGMENT_64
            char segname[17] = {};
            std::memcpy(segname, buf + off + 8, 16);
            uint32_t nsects = 0;
            std::memcpy(&nsects, buf + off + 64, 4);
            size_t so = off + 72;
            for (uint32_t s = 0; s < nsects && so + 80 <= off + cs; ++s) {
                SectInfo si;
                si.seg = segname;
                std::memcpy(&si.addr,   buf + so + 32, 8);
                std::memcpy(&si.size,   buf + so + 40, 8);
                std::memcpy(&si.offset, buf + so + 48, 4);
                sects.push_back(std::move(si));
                so += 80;
            }
        } else if (cmd == 0x2u && cs >= 24) { // LC_SYMTAB
            uint32_t symoff = 0, nsyms = 0, stroff = 0, strsize = 0;
            std::memcpy(&symoff,  buf + off + 8,  4);
            std::memcpy(&nsyms,   buf + off + 12, 4);
            std::memcpy(&stroff,  buf + off + 16, 4);
            std::memcpy(&strsize, buf + off + 20, 4);
            if (static_cast<size_t>(symoff) + static_cast<size_t>(nsyms) * 16 <= len &&
                static_cast<size_t>(stroff) + static_cast<size_t>(strsize) <= len) {
                const char* strs = reinterpret_cast<const char*>(buf + stroff);
                for (uint32_t i = 0; i < nsyms; ++i) {
                    const uint8_t* sp = buf + symoff + static_cast<size_t>(i) * 16;
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
            }
        }
        off += cs;
    }

    int ki = -1;
    for (int i = 0; i < static_cast<int>(sects.size()); ++i) {
        if (sects[i].seg.rfind("__KERN", 0) == 0) { ki = i; break; }
    }
    if (ki < 0) return false;
    const SectInfo& ks = sects[ki];

    std::vector<uint64_t> addrs;
    std::unordered_map<std::string, std::array<std::pair<bool, uint64_t>, 16>> fam;
    for (const auto& sym : syms) {
        if (sym.nsect != static_cast<uint8_t>(ki + 1)) continue;
        addrs.push_back(sym.nval);
        // Match K<64-hex>_ne_<0..15>
        if (sym.name.size() >= 70 && sym.name[0] == 'K') {
            size_t pos = sym.name.find("_ne_");
            if (pos == 65) {
                int idx = std::atoi(sym.name.c_str() + 69);
                if (idx >= 0 && idx < 16) {
                    std::string hash = sym.name.substr(1, 64);
                    fam[hash][idx] = {true, sym.nval};
                }
            }
        }
    }
    addrs.push_back(ks.addr + ks.size);
    std::sort(addrs.begin(), addrs.end());
    addrs.erase(std::unique(addrs.begin(), addrs.end()), addrs.end());

    std::unordered_map<uint64_t, uint64_t> nxt_addr;
    for (size_t i = 0; i + 1 < addrs.size(); ++i) nxt_addr[addrs[i]] = addrs[i + 1];

    out.clear();
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
    std::sort(out.begin(), out.end(), [](const KernelTensorInfo& a, const KernelTensorInfo& b) {
        return a.chunk_off[0] < b.chunk_off[0];
    });
    return !out.empty();
}

struct SrcLinear {
    uint32_t O = 0, K = 0;
    uint64_t w_off = 0;
    uint64_t o_off = 0; // 0 if symmetric
    uint64_t s_off = 0;
    bool has_offset = false;
    std::array<uint32_t, 8> fp0{}; // first 8 k-words of group 0 (rows 0..7)
};

static uint64_t read_blob_data_offset(std::ifstream& wf, uint64_t meta_off) {
    uint8_t hdr[24];
    wf.seekg(static_cast<std::streamoff>(meta_off));
    wf.read(reinterpret_cast<char*>(hdr), 24);
    uint64_t data_off = 0;
    std::memcpy(&data_off, hdr + 16, 8);
    return data_off;
}

// Pack 8 consecutive k-words for group gi (rows 8*gi .. 8*gi+7) starting at input k0 from weight.bin
static void pack_group_words(std::ifstream& wf, uint64_t w_data_off, uint32_t K,
                             uint32_t gi, uint32_t k0, uint32_t* out_words, uint32_t n_words) {
    uint32_t row_bytes = K >> 1;
    std::vector<uint8_t> rbuf(8u * ((n_words + 1u) >> 1));
    uint32_t byte_k0 = k0 >> 1;
    uint32_t n_bytes = (n_words + 1u) >> 1;
    for (uint32_t j = 0; j < 8u; ++j) {
        uint64_t pos = w_data_off + static_cast<uint64_t>(gi * 8u + j) * row_bytes + byte_k0;
        wf.seekg(static_cast<std::streamoff>(pos));
        wf.read(reinterpret_cast<char*>(rbuf.data() + j * n_bytes), n_bytes);
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
}

static bool parse_source_hf4(const std::string& mlmodelc_dir, std::array<SrcLinear, 16>& out_lins,
                             std::vector<float>& out_scales, std::vector<float>& out_offs, bool& out_asym) {
    std::ifstream mfs(mlmodelc_dir + "/model.mil");
    if (!mfs) return false;
    std::string mil((std::istreambuf_iterator<char>(mfs)), std::istreambuf_iterator<char>());

    // Match asymmetric: data + offset + scale
    static const std::regex re_asym(
        R"(tensor<fp16, \[(\d+), (\d+)\]> linear_(\d+)_weight_0_quantized = constexpr_blockwise_shift_scale\(data = tensor<int4, \[\d+, \d+\]>\(BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)\), offset = tensor<int4, \[\d+, 1\]>\(BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)\), scale = tensor<fp16, \[\d+, 1\]>\(BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)\)\))"
    );
    // Fallback symmetric: data + scale (no offset)
    static const std::regex re_sym(
        R"(tensor<fp16, \[(\d+), (\d+)\]> (?:linear_(\d+)_weight_0_quantized|constexpr_blockwise_shift_scale_(\d+)) = constexpr_blockwise_shift_scale\(data = tensor<int4, \[\d+, \d+\]>\(BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)\), scale = tensor<fp16, \[\d+, 1\]>\(BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)\)\))"
    );

    bool seen[16] = {};
    int count = 0;
    out_asym = false;
    for (std::sregex_iterator it(mil.begin(), mil.end(), re_asym), end; it != end; ++it) {
        int li = std::stoi((*it)[3].str());
        if (li < 0 || li >= 16) continue;
        SrcLinear& sl = out_lins[li];
        sl.O = static_cast<uint32_t>(std::stoul((*it)[1].str()));
        sl.K = static_cast<uint32_t>(std::stoul((*it)[2].str()));
        sl.w_off = std::stoull((*it)[4].str());
        sl.o_off = std::stoull((*it)[5].str());
        sl.s_off = std::stoull((*it)[6].str());
        sl.has_offset = true;
        if (!seen[li]) { seen[li] = true; ++count; }
        out_asym = true;
    }
    if (count < 16) {
        // Try symmetric layout (e.g. rowq4)
        for (std::sregex_iterator it(mil.begin(), mil.end(), re_sym), end; it != end; ++it) {
            std::string s_li = (*it)[3].matched ? (*it)[3].str() : (*it)[4].str();
            int li = std::stoi(s_li);
            if (li < 0 || li >= 16) continue;
            SrcLinear& sl = out_lins[li];
            sl.O = static_cast<uint32_t>(std::stoul((*it)[1].str()));
            sl.K = static_cast<uint32_t>(std::stoul((*it)[2].str()));
            sl.w_off = std::stoull((*it)[5].str());
            sl.s_off = std::stoull((*it)[6].str());
            sl.has_offset = false;
            if (!seen[li]) { seen[li] = true; ++count; }
        }
    }
    if (count != 16) return false;

    // Validate expected Qwen 3.8 full-MLP shapes: 4 sub-blocks of [gate(4352,5120), up(4352,5120), down0(5120,2176), down1(5120,2176)]
    for (int sb = 0; sb < 4; ++sb) {
        if (out_lins[sb * 4 + 0].O != SUB || out_lins[sb * 4 + 0].K != HID) return false;
        if (out_lins[sb * 4 + 1].O != SUB || out_lins[sb * 4 + 1].K != HID) return false;
        if (out_lins[sb * 4 + 2].O != HID || out_lins[sb * 4 + 2].K != DK)  return false;
        if (out_lins[sb * 4 + 3].O != HID || out_lins[sb * 4 + 3].K != DK)  return false;
    }

    std::ifstream wf(mlmodelc_dir + "/weights/weight.bin", std::ios::binary);
    if (!wf) return false;

    // Total scales/offsets: gate[17408] + up[17408] + down[8 * 5120] = 75776
    out_scales.assign(17408u + 17408u + 8u * 5120u, 0.0f);
    out_offs.assign(17408u + 17408u + 8u * 5120u, 1032.0f);

    for (int li = 0; li < 16; ++li) {
        SrcLinear& sl = out_lins[li];
        sl.w_off = read_blob_data_offset(wf, sl.w_off);
        sl.s_off = read_blob_data_offset(wf, sl.s_off);
        if (sl.has_offset) sl.o_off = read_blob_data_offset(wf, sl.o_off);

        // Compute 8-word fingerprint of group 0
        pack_group_words(wf, sl.w_off, sl.K, 0u, 0u, sl.fp0.data(), 8u);

        // Determine flat destination offset in scales/offs buffer
        int sb = li >> 2;
        int role = li & 3;
        size_t dst_base = 0;
        if (role == 0)      dst_base = static_cast<size_t>(sb) * SUB;
        else if (role == 1) dst_base = 17408u + static_cast<size_t>(sb) * SUB;
        else                dst_base = 34816u + static_cast<size_t>(sb * 2 + (role - 2)) * HID;

        std::vector<uint16_t> s_raw(sl.O);
        wf.seekg(static_cast<std::streamoff>(sl.s_off));
        wf.read(reinterpret_cast<char*>(s_raw.data()), sl.O * sizeof(uint16_t));
        for (uint32_t i = 0; i < sl.O; ++i) {
            __fp16 h;
            std::memcpy(&h, &s_raw[i], 2);
            out_scales[dst_base + i] = static_cast<float>(h);
        }
        if (sl.has_offset) {
            std::vector<uint8_t> o_raw((sl.O + 1u) >> 1);
            wf.seekg(static_cast<std::streamoff>(sl.o_off));
            wf.read(reinterpret_cast<char*>(o_raw.data()), o_raw.size());
            for (uint32_t i = 0; i < sl.O; ++i) {
                uint8_t b = o_raw[i >> 1];
                int nib = ((i & 1u) == 0u) ? (b & 0xF) : ((b >> 4) & 0xF);
                if (nib > 7) nib -= 16;
                out_offs[dst_base + i] = 1032.0f + static_cast<float>(nib);
            }
        }
    }
    return wf.good();
}

static uint64_t expected_chunk_bytes(uint32_t O, uint32_t K, uint32_t hdr_bytes) {
    uint64_t raw = LUT4_BYTES + static_cast<uint64_t>(O / 128u) * (hdr_bytes + 4ull * K);
    return ((raw + 63ull) / 64ull) * 64ull;
}

} // namespace

AneShareManager::AneShareManager(MetalContext& ctx, const std::string& ane_models_dir,
                                 const std::string& tag, int num_layers, int share_count) {
    auto t0 = std::chrono::steady_clock::now();
    layers_.resize(static_cast<size_t>(num_layers));
    if (share_count <= 0 || tag != "hf4") return;

    // Select which layers to share:
    // - share_count >= num_layers -> all layers [0 .. num_layers-1]
    // - share_count < num_layers  -> protect sensitive first/last layers symmetrically!
    std::vector<int> target_layers;
    int clamped = std::min(share_count, num_layers);
    int protect_each = (num_layers - clamped) / 2;
    for (int L = protect_each; L < protect_each + clamped; ++L) target_layers.push_back(L);

    // 1. Discover candidate >100 MB model.hwx files in /Library/Caches/com.apple.aned
    std::vector<std::string> hwx_paths;
    const std::string aned_root = "/Library/Caches/com.apple.aned";
    std::error_code ec;
    if (fs::exists(aned_root, ec)) {
        for (const auto& ver_entry : fs::directory_iterator(aned_root, ec)) {
            if (!ver_entry.is_directory(ec)) continue;
            fs::path mac = ver_entry.path() / "ModelAssetsCache";
            if (!fs::is_directory(mac, ec)) continue;
            const char* owners[] = {"mometal_unsigned", "metal-infer_unsigned", "Python_unsigned"};
            for (const char* own : owners) {
                fs::path odir = mac / own;
                if (!fs::is_directory(odir, ec)) continue;
                for (auto it = fs::recursive_directory_iterator(odir, fs::directory_options::skip_permission_denied, ec);
                     it != fs::recursive_directory_iterator(); it.increment(ec)) {
                    if (ec) { ec.clear(); continue; }
                    if (!it->is_regular_file(ec)) continue;
                    if (it->path().filename() != "model.hwx") continue;
                    auto sz = it->file_size(ec);
                    if (!ec && sz > 100000000u && sz < 200000000u) {
                        hwx_paths.push_back(it->path().string());
                    }
                }
            }
        }
    }
    if (hwx_paths.empty()) {
        std::fprintf(stderr, "[ane-share] No readable hf4 model.hwx files found in %s\n", aned_root.c_str());
        return;
    }

    // 2. Index candidate hwx files by their gate-sized tensor fp0s (read ONLY 64 KB header + 16x 32B fingerprints!)
    struct HwxIndexEntry {
        std::string path;
        std::vector<KernelTensorInfo> kts;
    };
    std::vector<HwxIndexEntry> hwx_entries(hwx_paths.size());
    std::unordered_map<uint64_t, std::vector<size_t>> fp_to_hwx;
    std::mutex map_mtx;

    {
        std::atomic<size_t> next_idx{0};
        int n_thr = std::min<int>(8, static_cast<int>(hwx_paths.size()));
        std::vector<std::thread> threads;
        for (int ti = 0; ti < n_thr; ++ti) {
            threads.emplace_back([&]() {
                std::vector<uint8_t> hdr(65536);
                while (true) {
                    size_t idx = next_idx.fetch_add(1);
                    if (idx >= hwx_paths.size()) break;
                    std::ifstream f(hwx_paths[idx], std::ios::binary);
                    if (!f) continue;
                    f.read(reinterpret_cast<char*>(hdr.data()), hdr.size());
                    size_t got = static_cast<size_t>(f.gcount());
                    std::vector<KernelTensorInfo> kts;
                    if (!parse_hwx_kernel_tensors(hdr.data(), got, kts) || kts.size() < 16) continue;
                    hwx_entries[idx].path = hwx_paths[idx];
                    hwx_entries[idx].kts = std::move(kts);

                    std::vector<uint64_t> keys;
                    for (const auto& kt : hwx_entries[idx].kts) {
                        for (uint32_t hdr_b : {24u, 16u}) {
                            if (kt.chunk_bytes == expected_chunk_bytes(SUB, HID, hdr_b)) {
                                uint64_t fp64 = 0;
                                f.seekg(static_cast<std::streamoff>(kt.chunk_off[0] + LUT4_BYTES + hdr_b));
                                f.read(reinterpret_cast<char*>(&fp64), 8);
                                if (f.good()) keys.push_back(fp64);
                                break;
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lk(map_mtx);
                    for (uint64_t k : keys) fp_to_hwx[k].push_back(idx);
                }
            });
        }
        for (auto& th : threads) th.join();
    }

    // 3. Match, verify bit-exact, and zero-copy mmap each target layer in parallel!
    std::atomic<size_t> next_t{0};
    std::atomic<int> matched_layers{0};
    int n_thr = std::min<int>(8, static_cast<int>(target_layers.size()));
    std::vector<std::thread> workers;
    id<MTLDevice> device = ctx.device();
    size_t page_sz = static_cast<size_t>(getpagesize());

    for (int ti = 0; ti < n_thr; ++ti) {
        workers.emplace_back([&]() {
            @autoreleasepool {
                while (true) {
                    size_t ti_idx = next_t.fetch_add(1);
                    if (ti_idx >= target_layers.size()) break;
                    int L = target_layers[ti_idx];
                    std::string mdir = ane_models_dir + "/L" + std::to_string(L) + "_mlp_p0_" + tag + ".mlmodelc";
                    std::array<SrcLinear, 16> src_lins;
                    std::vector<float> scales, offs;
                    bool is_asym = false;
                    if (!parse_source_hf4(mdir, src_lins, scales, offs, is_asym)) continue;

                    uint32_t hdr_bytes = is_asym ? 24u : 16u;
                    uint32_t hdr_words = hdr_bytes / 4u;
                    uint64_t fp0_key = 0;
                    std::memcpy(&fp0_key, src_lins[0].fp0.data(), 8);
                    auto it_cands = fp_to_hwx.find(fp0_key);
                    if (it_cands == fp_to_hwx.end()) continue;

                    // Test candidate hwx files for full 16/16 linear match + multi-group bit-exact check
                    for (size_t hi : it_cands->second) {
                        const auto& entry = hwx_entries[hi];
                        int fd = open(entry.path.c_str(), O_RDONLY);
                        if (fd < 0) continue;
                        struct stat st{};
                        if (fstat(fd, &st) != 0) { close(fd); continue; }
                        size_t fsz = static_cast<size_t>(st.st_size);
                        size_t aligned_sz = ((fsz + page_sz - 1) / page_sz) * page_sz;
                        void* ptr = mmap(nullptr, aligned_sz, PROT_READ, MAP_SHARED, fd, 0);
                        if (ptr == MAP_FAILED) ptr = mmap(nullptr, aligned_sz, PROT_READ, MAP_PRIVATE, fd, 0);
                        close(fd);
                        if (ptr == MAP_FAILED) continue;

                        const uint8_t* hbytes = reinterpret_cast<const uint8_t*>(ptr);
                        uint32_t base_w[16] = {}, chunk_w[16] = {};
                        bool all_ok = true;
                        std::ifstream wf(mdir + "/weights/weight.bin", std::ios::binary);

                        for (int li = 0; li < 16; ++li) {
                            const SrcLinear& sl = src_lins[li];
                            uint64_t exp_cb = expected_chunk_bytes(sl.O, sl.K, hdr_bytes);
                            bool lin_ok = false;
                            for (const auto& kt : entry.kts) {
                                if (kt.chunk_bytes != exp_cb) continue;
                                uint64_t w0_off = kt.chunk_off[0] + LUT4_BYTES + hdr_bytes;
                                if (w0_off + 32 > fsz) continue;
                                if (std::memcmp(hbytes + w0_off, sl.fp0.data(), 32) != 0) continue;

                                // Bit-exact spot verification on chunk 7 mid-group & chunk 15 last-group!
                                uint32_t g_bytes = hdr_bytes + 4u * sl.K;
                                uint32_t n_groups = sl.O / 8u;
                                uint32_t check_gis[2] = { (n_groups / 2u), (n_groups - 1u) };
                                bool spot_ok = true;
                                for (uint32_t cgi : check_gis) {
                                    uint32_t ch = cgi & 15u;
                                    uint32_t slot = cgi >> 4;
                                    uint32_t ref_w[8];
                                    pack_group_words(wf, sl.w_off, sl.K, cgi, sl.K - 8u, ref_w, 8u);
                                    uint64_t hoff = kt.chunk_off[ch] + LUT4_BYTES + static_cast<uint64_t>(slot) * g_bytes
                                                    + hdr_bytes + static_cast<uint64_t>(sl.K - 8u) * 4u;
                                    if (hoff + 32 > fsz || std::memcmp(hbytes + hoff, ref_w, 32) != 0) {
                                        spot_ok = false;
                                        break;
                                    }
                                }
                                if (spot_ok) {
                                    base_w[li] = static_cast<uint32_t>(kt.chunk_off[0] / 4u);
                                    chunk_w[li] = static_cast<uint32_t>(kt.chunk_bytes / 4u);
                                    lin_ok = true;
                                    break;
                                }
                            }
                            if (!lin_ok) { all_ok = false; break; }
                        }

                        if (!all_ok) {
                            munmap(ptr, aligned_sz);
                            continue;
                        }

                        id<MTLBuffer> mbuf = [device newBufferWithBytesNoCopy:ptr
                                                                       length:aligned_sz
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
                        if (!mbuf) {
                            munmap(ptr, aligned_sz);
                            continue;
                        }

                        // Pack {scale, 136.0f + offset} as float2[75776]
                        size_t n_so = scales.size();
                        id<MTLBuffer> so_buf = ctx.alloc_shared(n_so * 2 * sizeof(float));
                        float* so_ptr = reinterpret_cast<float*>([so_buf contents]);
                        for (size_t i = 0; i < n_so; ++i) {
                            so_ptr[2 * i + 0] = scales[i];
                            so_ptr[2 * i + 1] = offs[i];
                        }

                        AneShareLayer& dst = layers_[L];
                        dst.hwx_buf = mbuf;
                        dst.mmap_ptr = ptr;
                        dst.mapped_size = aligned_sz;
                        dst.buf_scales_offs = so_buf;
                        dst.hdr_words = hdr_words;
                        std::memcpy(dst.lin_base_w, base_w, sizeof(base_w));
                        std::memcpy(dst.lin_chunk_w, chunk_w, sizeof(chunk_w));
                        dst.ready = true;
                        matched_layers.fetch_add(1);
                        break;
                    }
                }
            }
        });
    }
    for (auto& th : workers) th.join();

    shared_count_ = matched_layers.load();
    load_time_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[ane-share] Shared %d/%d hf4 MLP layers (single-copy model.hwx) in %.1f ms\n",
                 shared_count_, clamped, load_time_ms_);
}

AneShareManager::~AneShareManager() {
    for (auto& l : layers_) {
        l.hwx_buf = nil;
        l.buf_scales_offs = nil;
        l.buf_folded_norm = nil;
        if (l.mmap_ptr && l.mmap_ptr != MAP_FAILED) {
            munmap(l.mmap_ptr, l.mapped_size);
            l.mmap_ptr = nullptr;
        }
    }
}

} // namespace mi
