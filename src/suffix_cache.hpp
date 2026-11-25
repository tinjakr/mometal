#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mi {

struct SuffixBranch {
    int cont[7] = {};
    int cont_len = 0;
    int match_len = 0;
    int count = 0;
    bool from_session = false;
};

struct SuffixResult {
    SuffixBranch b[2];
    int n_branches = 0;
};

// Fast O(1)-insert, <0.015 ms-lookup multi-length (L=3..16) 2-branch Suffix Index.
// Maintains two arenas:
//   1. Session arena (active prompt + generation + cross-turn history, bounded ring)
//   2. Static repo/corpus arena (indexed once via --suffix-dir, persists across turns)
// Uses -1 sentinels between turns/files so n-grams never cross document boundaries.
class SuffixCache {
public:
    static constexpr uint32_t SESS_MASK = 262143u;
    static constexpr uint32_t REPO_MASK = 262143u;
    static constexpr size_t MAX_SESSION_TOKENS = 524288u;
    static constexpr size_t KEEP_SESSION_TOKENS = 131072u;
    static constexpr size_t MAX_REPO_TOKENS = 2000000u;

    struct Slot4 {
        int32_t p[4] = {-1, -1, -1, -1};
    };

    SuffixCache() : sess_tbl_(SESS_MASK + 1u), repo_tbl_(REPO_MASK + 1u) {
        sess_tokens_.reserve(16384);
    }

    static inline uint32_t hash3(int a, int b, int c) {
        uint64_t h = static_cast<uint64_t>(static_cast<uint32_t>(a)) * 0x9E3779B97F4A7C15ull;
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(b)) * 0xBF58476D1CE4E5B9ull;
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(c)) * 0x94D049BB133111EBull;
        h ^= (h >> 33);
        return static_cast<uint32_t>(h);
    }

    void clear_session() {
        sess_tokens_.clear();
        std::fill(sess_tbl_.begin(), sess_tbl_.end(), Slot4{});
    }

    // Insert turn separator so n-grams never span unrelated prompts while retaining cross-turn memory
    void begin_turn() {
        if (!sess_tokens_.empty() && sess_tokens_.back() != -1) {
            sess_tokens_.push_back(-1);
        }
        if (sess_tokens_.size() >= MAX_SESSION_TOKENS) {
            compact_session();
        }
    }

    void push_session_token(int tok) {
        if (tok < 0) { begin_turn(); return; }
        if (sess_tokens_.size() >= MAX_SESSION_TOKENS) {
            compact_session();
        }
        int32_t idx = static_cast<int32_t>(sess_tokens_.size());
        sess_tokens_.push_back(tok);
        if (idx >= 2) {
            int t0 = sess_tokens_[idx - 2];
            int t1 = sess_tokens_[idx - 1];
            if (t0 >= 0 && t1 >= 0) {
                Slot4& sl = sess_tbl_[hash3(t0, t1, tok) & SESS_MASK];
                sl.p[3] = sl.p[2];
                sl.p[2] = sl.p[1];
                sl.p[1] = sl.p[0];
                sl.p[0] = idx;
            }
        }
    }

    void add_repo_doc(const int* tokens, int n_tokens) {
        if (!tokens || n_tokens < 4) return;
        if (repo_tokens_.size() >= MAX_REPO_TOKENS) return;
        if (!repo_tokens_.empty() && repo_tokens_.back() != -1) {
            repo_tokens_.push_back(-1);
        }
        size_t budget = MAX_REPO_TOKENS - repo_tokens_.size();
        int take = static_cast<int>(std::min<size_t>(static_cast<size_t>(n_tokens), budget));
        for (int i = 0; i < take; ++i) {
            int tok = tokens[i];
            int32_t idx = static_cast<int32_t>(repo_tokens_.size());
            repo_tokens_.push_back(tok);
            if (tok >= 0 && idx >= 2) {
                int t0 = repo_tokens_[idx - 2];
                int t1 = repo_tokens_[idx - 1];
                if (t0 >= 0 && t1 >= 0) {
                    Slot4& sl = repo_tbl_[hash3(t0, t1, tok) & REPO_MASK];
                    sl.p[3] = sl.p[2];
                    sl.p[2] = sl.p[1];
                    sl.p[1] = sl.p[0];
                    sl.p[0] = idx;
                }
            }
        }
    }

    size_t repo_token_count() const { return repo_tokens_.size(); }

    // Lookup longest matching suffixes ending with (..., hist[H-2], hist[H-1], anchor_tok).
    // Returns up to 2 distinct highest-scoring continuation branches (max 7 tokens each).
    SuffixResult lookup(const std::vector<int>& hist, int anchor_tok, int prefer_first = -1, int max_draft = 7) const {
        SuffixResult res{};
        const int H = static_cast<int>(hist.size());
        if (H < 2 || anchor_tok < 0 || max_draft <= 0) return res;
        const int s1 = hist[H - 1];
        const int s2 = hist[H - 2];
        if (s1 < 0 || s2 < 0) return res;

        const uint32_t h3 = hash3(s2, s1, anchor_tok);
        const int lim_draft = std::min(max_draft, 7);

        struct Cand {
            int next0 = -1;
            int cont[7] = {};
            int cont_len = 0;
            int best_len = 0;
            int count = 0;
            bool from_session = false;
            int score = -1;
        };
        Cand cands[8];
        int nc = 0;

        auto probe_arena = [&](const std::vector<int32_t>& arena, const Slot4& sl, bool is_sess) {
            const int N = static_cast<int>(arena.size());
            const int32_t* a = arena.data();
            for (int k = 0; k < 4; ++k) {
                int32_t p = sl.p[k];
                if (p < 2 || p >= N || (!is_sess && p + 1 >= N)) continue;
                if (a[p] != anchor_tok || a[p - 1] != s1 || a[p - 2] != s2) continue;

                // Measure exact longest suffix match length (3..16)
                int mlen = 3;
                int max_back = std::min({16, static_cast<int>(p) + 1, H + 1});
                while (mlen < max_back) {
                    int tok_a = a[p - mlen];
                    int tok_h = hist[H - mlen];
                    if (tok_a < 0 || tok_a != tok_h) break;
                    ++mlen;
                }

                // Extract continuation up to lim_draft tokens (stopping at sentinel -1)
                int clen = 0;
                int cbuf[7];
                if (is_sess && p >= N - 16 && (N - 1 - p) < lim_draft) {
                    // Exact periodic tail loop (handles periods 1..15 including single-token loops!)
                    int period = N - p;
                    bool ok = true;
                    for (int j = 0; j < lim_draft; ++j) {
                        int src = p + 1 + (j % period);
                        int nt = (src == N) ? anchor_tok : a[src];
                        if (nt < 0) { ok = false; break; }
                        cbuf[j] = nt;
                    }
                    if (ok) clen = lim_draft;
                } else {
                    for (int j = 0; j < lim_draft && (p + 1 + j) < N; ++j) {
                        int nt = a[p + 1 + j];
                        if (nt < 0) break;
                        cbuf[clen++] = nt;
                    }
                }
                if (clen == 0) continue;
                int next0 = cbuf[0];

                // Merge into candidate list by first continuation token next0
                int slot = -1;
                for (int i = 0; i < nc; ++i) {
                    if (cands[i].next0 == next0) { slot = i; break; }
                }
                if (slot < 0 && nc < 8) {
                    slot = nc++;
                    cands[slot].next0 = next0;
                }
                if (slot >= 0) {
                    Cand& c = cands[slot];
                    c.count++;
                    bool better = (mlen > c.best_len) ||
                                  (mlen == c.best_len && is_sess && !c.from_session) ||
                                  (mlen == c.best_len && is_sess == c.from_session && clen > c.cont_len);
                    if (better) {
                        c.best_len = mlen;
                        c.cont_len = clen;
                        std::memcpy(c.cont, cbuf, static_cast<size_t>(clen) * sizeof(int));
                        if (is_sess) c.from_session = true;
                    } else if (is_sess) {
                        c.from_session = true;
                    }
                }
            }
        };

        probe_arena(sess_tokens_, sess_tbl_[h3 & SESS_MASK], true);
        if (!repo_tokens_.empty()) {
            probe_arena(repo_tokens_, repo_tbl_[h3 & REPO_MASK], false);
        }
        if (nc == 0) return res;

        for (int i = 0; i < nc; ++i) {
            Cand& c = cands[i];
            int sc = c.best_len * 6 + (c.from_session ? 4 : 0) + std::min(c.count, 4) * 2 + (c.cont_len >= 4 ? 2 : 0);
            if (prefer_first >= 0 && c.next0 == prefer_first) sc += 20;
            c.score = sc;
        }

        // Pick top-2 distinct branches
        int best0 = 0;
        for (int i = 1; i < nc; ++i) {
            if (cands[i].score > cands[best0].score) best0 = i;
        }
        auto fill_branch = [](SuffixBranch& dst, const Cand& src) {
            dst.cont_len = src.cont_len;
            dst.match_len = src.best_len;
            dst.count = src.count;
            dst.from_session = src.from_session;
            std::memcpy(dst.cont, src.cont, static_cast<size_t>(src.cont_len) * sizeof(int));
        };
        fill_branch(res.b[0], cands[best0]);
        res.n_branches = 1;

        int best1 = -1;
        for (int i = 0; i < nc; ++i) {
            if (i == best0) continue;
            if (best1 < 0 || cands[i].score > cands[best1].score) best1 = i;
        }
        if (best1 >= 0) {
            fill_branch(res.b[1], cands[best1]);
            res.n_branches = 2;
        }
        return res;
    }

private:
    void compact_session() {
        size_t keep = std::min(KEEP_SESSION_TOKENS, sess_tokens_.size());
        size_t drop = sess_tokens_.size() - keep;
        sess_tokens_.erase(sess_tokens_.begin(), sess_tokens_.begin() + static_cast<std::ptrdiff_t>(drop));
        std::fill(sess_tbl_.begin(), sess_tbl_.end(), Slot4{});
        for (int32_t i = 2; i < static_cast<int32_t>(sess_tokens_.size()); ++i) {
            int t0 = sess_tokens_[i - 2], t1 = sess_tokens_[i - 1], t2 = sess_tokens_[i];
            if (t0 >= 0 && t1 >= 0 && t2 >= 0) {
                Slot4& sl = sess_tbl_[hash3(t0, t1, t2) & SESS_MASK];
                sl.p[3] = sl.p[2]; sl.p[2] = sl.p[1]; sl.p[1] = sl.p[0]; sl.p[0] = i;
            }
        }
    }

    std::vector<int32_t> sess_tokens_;
    std::vector<int32_t> repo_tokens_;
    std::vector<Slot4> sess_tbl_;
    std::vector<Slot4> repo_tbl_;
};

} // namespace mi
