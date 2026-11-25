#pragma once

#include "model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mi {

struct PrefixCacheHeader {
    char magic[8];        // "MICACHE1" (text-only) or "MICACHE2" (multimodal: rope_delta + image_hashes)
    uint64_t model_hash;  // FNV-1a hash of arch + model_dir + quant_desc + layers
    uint32_t pos;         // number of cached tokens
    uint32_t reserved;    // bit 31 = has_mtp, bits 0..30 = mtp_pos
    uint64_t state_bytes; // payload bytes following token_ids (+ MultimodalCacheMeta & hashes for MICACHE2)
};

struct MultimodalCacheMeta {
    int32_t  rope_delta;
    uint32_t num_images;
};

inline uint64_t fnv1a_64(const void* data, size_t len, uint64_t h = 14695981039346656037ull) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline uint64_t compute_model_hash(const ModelInfo& info) {
    // Strip decode-only Quest sparse budget suffix so Dense and Quest share identical KV/GDN checkpoints
    std::string qd = info.quant_desc;
    size_t qp = qd.find(" + Quest(");
    if (qp != std::string::npos) {
        size_t qe = qd.find(')', qp);
        if (qe != std::string::npos) qd.erase(qp, qe - qp + 1);
    }
    uint64_t h = fnv1a_64(info.arch.data(), info.arch.size());
    h = fnv1a_64(info.model_dir.data(), info.model_dir.size(), h);
    h = fnv1a_64(qd.data(), qd.size(), h);
    h = fnv1a_64(&info.num_layers, sizeof(info.num_layers), h);
    h = fnv1a_64(&info.num_visits, sizeof(info.num_visits), h);
    return h;
}

inline std::string token_prefix_filename(uint64_t model_hash, const std::vector<int>& tokens,
                                         const std::vector<uint64_t>& image_hashes = {}) {
    uint64_t h = fnv1a_64(&model_hash, sizeof(model_hash));
    h = fnv1a_64(tokens.data(), tokens.size() * sizeof(int), h);
    if (!image_hashes.empty()) {
        h = fnv1a_64(image_hashes.data(), image_hashes.size() * sizeof(uint64_t), h);
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%016llx_%zu.micache",
                  static_cast<unsigned long long>(h), tokens.size());
    return std::string(buf);
}

class PrefixCacheManager {
public:
    explicit PrefixCacheManager(std::string dir) : dir_(std::move(dir)) {
        if (!dir_.empty()) {
            std::filesystem::create_directories(dir_);
        }
    }

    bool enabled() const { return !dir_.empty(); }

    // Find the longest cached prefix (text or multimodal!) strictly shorter than prompt_ids.size()
    int try_restore(Model& model, const std::vector<int>& prompt_ids, std::vector<int>& restored_tokens,
                    const std::vector<uint64_t>& req_image_hashes = {},
                    std::vector<uint64_t>* restored_image_hashes = nullptr) const {
        if (restored_image_hashes) restored_image_hashes->clear();
        if (dir_.empty() || prompt_ids.size() < 2) return 0;

        uint64_t want_hash = compute_model_hash(model.info());
        std::string best_path;
        size_t best_len = 0;

        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir_, ec)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".micache") continue;
            std::ifstream ifs(entry.path(), std::ios::binary);
            if (!ifs) continue;
            PrefixCacheHeader hdr{};
            ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            if (!ifs) continue;
            bool is_v1 = (std::memcmp(hdr.magic, "MICACHE1", 8) == 0);
            bool is_v2 = (std::memcmp(hdr.magic, "MICACHE2", 8) == 0);
            if (!is_v1 && !is_v2) continue;
            if (hdr.model_hash != want_hash) continue;
            if (hdr.pos == 0 || hdr.pos >= prompt_ids.size() || hdr.pos <= best_len) continue;

            std::vector<int> cached_ids(hdr.pos);
            ifs.read(reinterpret_cast<char*>(cached_ids.data()), hdr.pos * sizeof(int));
            if (!ifs) continue;
            if (std::memcmp(cached_ids.data(), prompt_ids.data(), hdr.pos * sizeof(int)) != 0) continue;

            if (is_v2) {
                MultimodalCacheMeta meta{};
                ifs.read(reinterpret_cast<char*>(&meta), sizeof(meta));
                if (!ifs || meta.num_images > req_image_hashes.size()) continue;
                if (meta.num_images > 0) {
                    std::vector<uint64_t> cached_imgs(meta.num_images);
                    ifs.read(reinterpret_cast<char*>(cached_imgs.data()), meta.num_images * sizeof(uint64_t));
                    if (!ifs) continue;
                    if (!std::equal(cached_imgs.begin(), cached_imgs.end(), req_image_hashes.begin())) continue;
                }
            }

            best_len = hdr.pos;
            best_path = entry.path().string();
        }

        if (best_path.empty()) return 0;
        std::vector<uint64_t> loaded_imgs;
        if (model.load_state_from_file(best_path, restored_tokens, loaded_imgs)) {
            if (restored_image_hashes) *restored_image_hashes = std::move(loaded_imgs);
            return static_cast<int>(restored_tokens.size());
        }
        return 0;
    }

    bool save(Model& model, const std::vector<int>& tokens,
              const std::vector<uint64_t>& image_hashes = {}) const {
        if (dir_.empty() || tokens.empty()) return false;
        uint64_t mhash = compute_model_hash(model.info());
        std::string path = dir_ + "/" + token_prefix_filename(mhash, tokens, image_hashes);
        if (std::filesystem::exists(path)) {
            std::error_code ec;
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
            return true;
        }
        bool ok = model.save_state_to_file(path, tokens, image_hashes);
        if (ok) evict_lru(6ull * 1024 * 1024 * 1024); // Cap disk cache at 6 GB (oldest evicted first)
        return ok;
    }

private:
    void evict_lru(uint64_t max_bytes) const {
        std::error_code ec;
        struct Entry { std::filesystem::path p; std::filesystem::file_time_type mtime; uint64_t sz; };
        std::vector<Entry> items;
        uint64_t total = 0;
        for (const auto& e : std::filesystem::directory_iterator(dir_, ec)) {
            if (!e.is_regular_file() || e.path().extension() != ".micache") continue;
            uint64_t sz = e.file_size(ec);
            auto mt = e.last_write_time(ec);
            items.push_back({e.path(), mt, sz});
            total += sz;
        }
        if (total <= max_bytes || items.size() <= 2) return;
        std::sort(items.begin(), items.end(), [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
        for (size_t i = 0; i + 2 < items.size() && total > max_bytes; ++i) {
            std::filesystem::remove(items[i].p, ec);
            total -= items[i].sz;
        }
    }

    std::string dir_;
};

} // namespace mi
