#import <Foundation/Foundation.h>

#include "tokenizer.hpp"
#include "json.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace mi {

static void append_cp_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

Tokenizer::~Tokenizer() {
    if (regex_handle_) {
        CFRelease(regex_handle_);
        regex_handle_ = nullptr;
    }
}

void Tokenizer::init_byte_level_tables() {
    bool is_direct[256] = {};
    for (int b = 33; b <= 126; ++b) is_direct[b] = true;
    for (int b = 161; b <= 172; ++b) is_direct[b] = true;
    for (int b = 174; b <= 255; ++b) is_direct[b] = true;

    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = is_direct[b] ? static_cast<uint32_t>(b) : (256u + n++);
        std::string u8;
        append_cp_utf8(u8, cp);
        byte_to_utf8_[b] = u8;
        utf8_to_byte_[u8] = static_cast<uint8_t>(b);
    }
}

void Tokenizer::load(const std::string& model_dir, const std::string& arch) {
    arch_ = arch;
    init_byte_level_tables();

    std::string tok_path = model_dir + "/tokenizer.json";
    Json doc = parse_json(read_file_to_string(tok_path));

    // Normalizer
    if (doc.contains("normalizer") && doc["normalizer"].is_object()) {
        if (doc["normalizer"]["type"].as_string() == "NFC") {
            use_nfc_ = true;
        }
    }

    // Pre-tokenizer regex
    std::string pattern;
    if (doc.contains("pre_tokenizer") && doc["pre_tokenizer"].is_object()) {
        const auto& pt = doc["pre_tokenizer"];
        if (pt["type"].as_string() == "Sequence" && pt.contains("pretokenizers")) {
            for (const auto& sub : pt["pretokenizers"].as_array()) {
                if (sub["type"].as_string() == "Split" && sub["pattern"].contains("Regex")) {
                    pattern = sub["pattern"]["Regex"].as_string();
                    break;
                }
            }
        } else if (pt["type"].as_string() == "Split" && pt["pattern"].contains("Regex")) {
            pattern = pt["pattern"]["Regex"].as_string();
        }
    }
    if (!pattern.empty()) {
        NSString* ns_pat = [NSString stringWithUTF8String:pattern.c_str()];
        NSError* err = nil;
        NSRegularExpression* re = [NSRegularExpression regularExpressionWithPattern:ns_pat
                                                                            options:0
                                                                              error:&err];
        if (!re) {
            throw std::runtime_error("Failed to compile tokenizer regex: " + pattern);
        }
        regex_handle_ = (void*)CFBridgingRetain(re);
    }

    // Vocab
    const auto& vocab_obj = doc["model"]["vocab"].as_object();
    int max_id = 0;
    for (const auto& [tok, id_j] : vocab_obj) {
        int id = static_cast<int>(id_j.as_int());
        if (id > max_id) max_id = id;
    }
    if (doc.contains("added_tokens") && doc["added_tokens"].is_array()) {
        for (const auto& at : doc["added_tokens"].as_array()) {
            int id = static_cast<int>(at["id"].as_int());
            if (id > max_id) max_id = id;
        }
    }

    id_to_token_.assign(static_cast<size_t>(max_id + 1), "");
    token_to_id_.reserve(vocab_obj.size() * 2);
    for (const auto& [tok, id_j] : vocab_obj) {
        int id = static_cast<int>(id_j.as_int());
        token_to_id_[tok] = id;
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
            id_to_token_[id] = tok;
        }
    }

    added_to_id_.clear();
    added_tokens_sorted_.clear();
    std::memset(added_first_byte_, 0, sizeof(added_first_byte_));
    special_ids_.clear();

    if (doc.contains("added_tokens") && doc["added_tokens"].is_array()) {
        for (const auto& at : doc["added_tokens"].as_array()) {
            int id = static_cast<int>(at["id"].as_int());
            std::string content = at["content"].as_string();
            bool is_sp = at.contains("special") && at["special"].as_bool(false);
            if (!content.empty()) {
                if (added_to_id_.find(content) == added_to_id_.end()) {
                    added_tokens_sorted_.push_back(content);
                }
                added_to_id_[content] = id;
                added_first_byte_[static_cast<unsigned char>(content[0])] = true;
                token_to_id_[content] = id;
            }
            if (is_sp) {
                special_ids_.insert(id);
            }
            if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
                id_to_token_[id] = content;
            }
        }
    }
    std::sort(added_tokens_sorted_.begin(), added_tokens_sorted_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    // Merges (supports both ["a", "b"] pair arrays and "a b" strings)
    const auto& merges = doc["model"]["merges"].as_array();
    merge_ranks_.reserve(merges.size() * 2);
    int rank = 0;
    for (const auto& m : merges) {
        std::string a, b;
        if (m.is_array() && m.as_array().size() == 2) {
            a = m[0].as_string();
            b = m[1].as_string();
        } else if (m.is_string()) {
            const std::string& s = m.as_string();
            size_t sp = s.find(' ');
            if (sp == std::string::npos) { ++rank; continue; }
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        }
        auto ia = token_to_id_.find(a);
        auto ib = token_to_id_.find(b);
        if (ia != token_to_id_.end() && ib != token_to_id_.end()) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ia->second)) << 32) |
                           static_cast<uint32_t>(ib->second);
            merge_ranks_.emplace(key, rank);
        }
        ++rank;
    }
}

bool Tokenizer::is_special(int id) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return false;
    return special_ids_.find(id) != special_ids_.end();
}

std::vector<int> Tokenizer::bpe_encode_piece(const std::string& raw_bytes) const {
    // Convert raw bytes to ByteLevel unicode strings and look up initial token IDs
    std::vector<int> ids;
    ids.reserve(raw_bytes.size());
    for (unsigned char b : raw_bytes) {
        auto it = token_to_id_.find(byte_to_utf8_[b]);
        if (it != token_to_id_.end()) ids.push_back(it->second);
    }
    if (ids.size() <= 1) return ids;

    // Iterative lowest-rank BPE merge
    while (ids.size() > 1) {
        int best_rank = INT_MAX;
        size_t best_pos = SIZE_MAX;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) << 32) |
                           static_cast<uint32_t>(ids[i + 1]);
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos == SIZE_MAX) break;

        std::string merged = id_to_token_[ids[best_pos]] + id_to_token_[ids[best_pos + 1]];
        auto mit = token_to_id_.find(merged);
        if (mit == token_to_id_.end()) break;
        int merged_id = mit->second;

        // Merge all non-overlapping occurrences of the winning pair
        int left_id = ids[best_pos];
        int right_id = ids[best_pos + 1];
        std::vector<int> next;
        next.reserve(ids.size() - 1);
        for (size_t i = 0; i < ids.size();) {
            if (i + 1 < ids.size() && ids[i] == left_id && ids[i + 1] == right_id) {
                next.push_back(merged_id);
                i += 2;
            } else {
                next.push_back(ids[i]);
                i += 1;
            }
        }
        ids.swap(next);
    }
    return ids;
}

static std::string nsstring_to_utf8(NSString* s) {
    if (!s || s.length == 0) return {};
    NSData* d = [s dataUsingEncoding:NSUTF8StringEncoding];
    if (!d || d.length == 0) return {};
    return std::string(static_cast<const char*>(d.bytes), d.length);
}

// Length of valid RFC 3629 UTF-8 sequence starting at ptr[0], or 0 if invalid.
static size_t valid_utf8_seq_len(const unsigned char* p, size_t avail) {
    if (avail == 0) return 0;
    unsigned char b0 = p[0];
    if (b0 < 0x80u) return 1;
    size_t need = 0;
    unsigned char b1_lo = 0x80u, b1_hi = 0xBFu;
    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        need = 2;
    } else if (b0 == 0xE0u) {
        need = 3; b1_lo = 0xA0u; b1_hi = 0xBFu;
    } else if ((b0 >= 0xE1u && b0 <= 0xECu) || (b0 >= 0xEEu && b0 <= 0xEFu)) {
        need = 3;
    } else if (b0 == 0xEDu) {
        need = 3; b1_lo = 0x80u; b1_hi = 0x9Fu;
    } else if (b0 == 0xF0u) {
        need = 4; b1_lo = 0x90u; b1_hi = 0xBFu;
    } else if (b0 >= 0xF1u && b0 <= 0xF3u) {
        need = 4;
    } else if (b0 == 0xF4u) {
        need = 4; b1_lo = 0x80u; b1_hi = 0x8Fu;
    } else {
        return 0;
    }
    if (avail < need) return 0;
    for (size_t k = 1; k < need; ++k) {
        unsigned char bk = p[k];
        unsigned char lo = (k == 1) ? b1_lo : 0x80u;
        unsigned char hi = (k == 1) ? b1_hi : 0xBFu;
        if (bk < lo || bk > hi) return 0;
    }
    return need;
}

void Tokenizer::encode_ordinary(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;

    auto encode_ns = [&](NSString* ns) {
        if (use_nfc_) {
            ns = [ns precomposedStringWithCanonicalMapping];
        }
        if (!regex_handle_) {
            std::string norm = nsstring_to_utf8(ns);
            if (!norm.empty()) {
                auto piece_ids = bpe_encode_piece(norm);
                out.insert(out.end(), piece_ids.begin(), piece_ids.end());
            }
            return;
        }

        NSRegularExpression* re = (__bridge NSRegularExpression*)regex_handle_;
        NSArray<NSTextCheckingResult*>* matches = [re matchesInString:ns
                                                              options:0
                                                                range:NSMakeRange(0, ns.length)];
        // Preserve Split(Isolated) unmatched gaps as well as matched pieces, with embedded NUL safety
        NSUInteger cursor = 0;
        for (NSTextCheckingResult* m in matches) {
            if (m.range.location > cursor) {
                std::string gap_u8 = nsstring_to_utf8([ns substringWithRange:NSMakeRange(cursor, m.range.location - cursor)]);
                if (!gap_u8.empty()) {
                    auto piece_ids = bpe_encode_piece(gap_u8);
                    out.insert(out.end(), piece_ids.begin(), piece_ids.end());
                }
            }
            if (m.range.length > 0) {
                std::string sub_u8 = nsstring_to_utf8([ns substringWithRange:m.range]);
                if (!sub_u8.empty()) {
                    auto piece_ids = bpe_encode_piece(sub_u8);
                    out.insert(out.end(), piece_ids.begin(), piece_ids.end());
                }
            }
            cursor = m.range.location + m.range.length;
        }
        if (cursor < ns.length) {
            std::string tail_u8 = nsstring_to_utf8([ns substringWithRange:NSMakeRange(cursor, ns.length - cursor)]);
            if (!tail_u8.empty()) {
                auto piece_ids = bpe_encode_piece(tail_u8);
                out.insert(out.end(), piece_ids.begin(), piece_ids.end());
            }
        }
    };

    @autoreleasepool {
        NSString* ns = [[NSString alloc] initWithBytes:text.data()
                                                length:text.size()
                                              encoding:NSUTF8StringEncoding];
        if (ns) {
            encode_ns(ns);
            return;
        }

        // Input contains invalid UTF-8 bytes: split into maximal valid UTF-8 runs (with full NFC+regex)
        // and invalid raw byte runs (direct ByteLevel BPE) so surrounding valid text is never corrupted.
        const unsigned char* raw = reinterpret_cast<const unsigned char*>(text.data());
        size_t i = 0, n = text.size();
        while (i < n) {
            size_t run_start = i;
            while (i < n) {
                size_t sl = valid_utf8_seq_len(raw + i, n - i);
                if (sl == 0) break;
                i += sl;
            }
            if (i > run_start) {
                NSString* sub_ns = [[NSString alloc] initWithBytes:text.data() + run_start
                                                            length:i - run_start
                                                          encoding:NSUTF8StringEncoding];
                if (sub_ns) {
                    encode_ns(sub_ns);
                } else {
                    auto piece_ids = bpe_encode_piece(text.substr(run_start, i - run_start));
                    out.insert(out.end(), piece_ids.begin(), piece_ids.end());
                }
            }
            size_t bad_start = i;
            while (i < n && valid_utf8_seq_len(raw + i, n - i) == 0) {
                ++i;
            }
            if (i > bad_start) {
                auto piece_ids = bpe_encode_piece(text.substr(bad_start, i - bad_start));
                out.insert(out.end(), piece_ids.begin(), piece_ids.end());
            }
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> out;
    if (!parse_special || added_tokens_sorted_.empty()) {
        encode_ordinary(text, out);
        return out;
    }

    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < text.size()) {
        bool matched = false;
        if (added_first_byte_[static_cast<unsigned char>(text[pos])]) {
            for (const auto& sp : added_tokens_sorted_) {
                if (pos + sp.size() <= text.size() &&
                    text.compare(pos, sp.size(), sp) == 0) {
                    if (pos > plain_start) {
                        encode_ordinary(text.substr(plain_start, pos - plain_start), out);
                    }
                    out.push_back(added_to_id_.at(sp));
                    pos += sp.size();
                    plain_start = pos;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) ++pos;
    }
    if (plain_start < text.size()) {
        encode_ordinary(text.substr(plain_start), out);
    }
    return out;
}

std::string Tokenizer::decode_token(int id, bool skip_special) const {
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return "";
    const std::string& tok = id_to_token_[id];
    if (special_ids_.find(id) != special_ids_.end()) {
        return skip_special ? "" : tok;
    }
    if (added_to_id_.find(tok) != added_to_id_.end()) {
        // Non-special added token (e.g. <think>, <tool_call>, <|fim_prefix|>): keep verbatim even when skip_special=true
        return tok;
    }

    // Reverse ByteLevel UTF-8 chars back to raw bytes
    std::string out;
    out.reserve(tok.size());
    size_t i = 0;
    while (i < tok.size()) {
        unsigned char c = static_cast<unsigned char>(tok[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > tok.size()) len = 1;
        std::string ch = tok.substr(i, len);
        auto it = utf8_to_byte_.find(ch);
        if (it != utf8_to_byte_.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out += ch;
        }
        i += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) out += decode_token(id, skip_special);
    return out;
}

static std::string trim_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string trim_leading_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    return s.substr(a);
}

static std::string trim_trailing_ws(const std::string& s) {
    size_t b = s.size();
    while (b > 0 && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(0, b);
}

static std::vector<ContentPart> trimmed_message_parts(const ChatMessage& m) {
    if (m.parts.empty()) {
        std::string t = trim_ws(m.content);
        if (t.empty()) return {};
        return {ContentPart{false, std::move(t), 0}};
    }
    std::vector<ContentPart> out = m.parts;
    // Trim leading whitespace up to the first image or non-empty text part
    for (auto& p : out) {
        if (p.is_image) break;
        p.text = trim_leading_ws(p.text);
        if (!p.text.empty()) break;
    }
    // Trim trailing whitespace back to the last image or non-empty text part
    for (auto it = out.rbegin(); it != out.rend(); ++it) {
        if (it->is_image) break;
        it->text = trim_trailing_ws(it->text);
        if (!it->text.empty()) break;
    }
    std::vector<ContentPart> filtered;
    filtered.reserve(out.size());
    for (auto& p : out) {
        if (p.is_image || !p.text.empty()) filtered.push_back(std::move(p));
    }
    return filtered;
}

static std::string flatten_message_parts_str(const ChatMessage& m) {
    if (m.parts.empty()) return trim_ws(m.content);
    auto parts = trimmed_message_parts(m);
    std::string s;
    for (const auto& p : parts) {
        if (p.is_image) {
            s += "<|vision_start|>";
            for (int k = 0; k < p.image_tokens; ++k) s += "<|image_pad|>";
            s += "<|vision_end|>";
        } else {
            s += p.text;
        }
    }
    return s;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    const std::string& reasoning_effort,
    const std::vector<std::string>& tools_json
) const {
    std::string out;
    if (arch_ == "inkling2") {
        // Inkling 2 Mini native chat template (matches chat_template.jinja)
        static const std::unordered_map<std::string, std::string> eff_map = {
            {"none", "0"}, {"off", "0"}, {"minimal", "0.1"}, {"low", "0.2"},
            {"medium", "0.7"}, {"high", "0.9"}, {"xhigh", "0.99"}, {"max", "0.99"}
        };
        std::string eff_val = "0";
        auto eit = eff_map.find(reasoning_effort);
        if (eit != eff_map.end()) eff_val = eit->second;
        else if (!reasoning_effort.empty()) eff_val = reasoning_effort;

        // 1. Optional tool declarations
        if (!tools_json.empty()) {
            Json::Array specs;
            for (const auto& tj : tools_json) {
                try {
                    Json t = parse_json(tj);
                    const Json& fn = t.contains("function") ? t["function"] : t;
                    Json::Object spec;
                    spec["description"] = fn.contains("description") && fn["description"].is_string()
                        ? fn["description"].as_string() : "";
                    spec["name"] = fn.contains("name") ? fn["name"].as_string() : "";
                    spec["parameters"] = fn.contains("parameters") ? fn["parameters"] : Json(Json::Object{});
                    spec["type"] = t.contains("type") && t["type"].is_string() ? t["type"].as_string() : "function";
                    specs.push_back(Json(std::move(spec)));
                } catch (...) {}
            }
            out += "<|message_system|>tool_declare<|content_xml|>" + Json(std::move(specs)).dump() + "<|end_message|>";
        }

        auto emit_effort = [&]() {
            out += "<|message_system|><|content_text|>Thinking effort level: " + eff_val + "<|end_message|>";
        };

        bool effort_emitted = false;
        for (size_t idx = 0; idx < messages.size(); ++idx) {
            const auto& m = messages[idx];
            if (!effort_emitted && m.role != "system") {
                emit_effort();
                effort_emitted = true;
            }
            std::string role_tok = "<|message_user|>";
            if (m.role == "system") role_tok = "<|message_system|>";
            else if (m.role == "assistant") role_tok = "<|message_model|>";
            else if (m.role == "tool") role_tok = "<|message_tool|>";

            std::string hdr = role_tok;
            if (m.role == "tool") {
                if (!m.name.empty()) {
                    hdr += m.name;
                } else if (!m.tool_call_id.empty()) {
                    bool found_tc = false;
                    for (const auto& prev : messages) {
                        if (prev.role == "assistant") {
                            for (const auto& tc : prev.tool_calls) {
                                if (tc.id == m.tool_call_id && !tc.name.empty()) {
                                    hdr += tc.name;
                                    found_tc = true;
                                    break;
                                }
                            }
                            if (found_tc) break;
                        }
                    }
                }
            } else if (m.role == "assistant" && !m.name.empty()) {
                hdr += m.name;
            }

            if (m.role == "assistant" && !m.reasoning_content.empty()) {
                out += hdr + "<|content_thinking|>" + m.reasoning_content + "<|end_message|>";
            }
            std::string content_str = m.parts.empty() ? m.content : flatten_message_parts_str(m);
            if (!content_str.empty() || !(m.role == "assistant" && (!m.tool_calls.empty() || !m.reasoning_content.empty()))) {
                out += hdr + "<|content_text|>" + content_str + "<|end_message|>";
            }
            if (m.role == "assistant" && !m.tool_calls.empty()) {
                for (const auto& tc : m.tool_calls) {
                    Json::Object invoke;
                    invoke["name"] = tc.name;
                    try {
                        invoke["args"] = tc.arguments_json.empty() ? Json(Json::Object{}) : parse_json(tc.arguments_json);
                    } catch (...) {
                        invoke["args"] = Json(Json::Object{});
                    }
                    out += "<|message_model|>" + tc.name + "<|content_invoke_tool_json|>" +
                           Json(std::move(invoke)).dump() + "<|end_message|>";
                }
            }
            if (m.role == "assistant") {
                out += "<|content_model_end_sampling|>";
            }
        }
        if (!effort_emitted) emit_effort();

        if (add_generation_prompt) {
            out += "<|message_model|>";
            if (eff_val == "0") {
                out += "<|content_text|>";
            } else {
                out += "<|content_thinking|>";
            }
        }
        return out;
    }

    // Qwen 3.5 / 3.8 ChatML template (exact match to chat_template.jinja including XML tool calls + reasoning!)
    bool enable_thinking = !(reasoning_effort == "none" || reasoning_effort == "off");
    std::string reasoning_instr;
    if (enable_thinking) {
        if (reasoning_effort == "xhigh" || reasoning_effort == "high" || reasoning_effort == "max" || reasoning_effort.empty()) {
            reasoning_instr = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        } else if (reasoning_effort == "low" || reasoning_effort == "minimal") {
            reasoning_instr = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
        }
        // "medium": thinking enabled with no extra instruction
    }

    // Merge consecutive leading system/developer messages (e.g. [system, developer, user]) into the opening system block;
    // preserve any mid-conversation system/developer messages in place instead of dropping them!
    size_t lead_sys_end = 0;
    std::string sys_lead;
    while (lead_sys_end < messages.size() && messages[lead_sys_end].role == "system") {
        std::string sc = flatten_message_parts_str(messages[lead_sys_end]);
        if (!sc.empty()) {
            if (!sys_lead.empty()) sys_lead += "\n\n";
            sys_lead += sc;
        }
        ++lead_sys_end;
    }
    bool has_sys0 = (lead_sys_end > 0);

    if (!tools_json.empty()) {
        out += "<|im_start|>system\n";
        if (!reasoning_instr.empty()) out += reasoning_instr + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& tj : tools_json) {
            out += "\n" + tj;
        }
        out += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
               "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
               "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
               "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
               "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n"
               "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
               "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
               "</IMPORTANT>";
        if (!sys_lead.empty()) out += "\n\n" + sys_lead;
        out += "<|im_end|>\n";
    } else if (has_sys0) {
        if (!sys_lead.empty()) {
            out += "<|im_start|>system\n";
            if (!reasoning_instr.empty()) out += reasoning_instr + "\n\n";
            out += sys_lead + "<|im_end|>\n";
        } else if (!reasoning_instr.empty()) {
            out += "<|im_start|>system\n" + reasoning_instr + "<|im_end|>\n";
        }
    } else if (!reasoning_instr.empty()) {
        out += "<|im_start|>system\n" + reasoning_instr + "<|im_end|>\n";
    }

    for (size_t i = lead_sys_end; i < messages.size(); ++i) {
        const auto& m = messages[i];
        std::string content = flatten_message_parts_str(m);
        if (m.role == "system") {
            if (!content.empty()) {
                out += "<|im_start|>system\n" + content + "<|im_end|>\n";
            }
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string rc = trim_ws(m.reasoning_content);
            out += "<|im_start|>assistant\n";
            out += "<think>\n" + rc + "\n</think>\n\n" + content;
            if (!m.tool_calls.empty()) {
                for (size_t j = 0; j < m.tool_calls.size(); ++j) {
                    const auto& tc = m.tool_calls[j];
                    if (j == 0) {
                        out += content.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=";
                    } else {
                        out += "\n<tool_call>\n<function=";
                    }
                    out += tc.name + ">\n";
                    if (!tc.arguments_json.empty()) {
                        try {
                            Json args = parse_json(tc.arguments_json);
                            if (args.is_object()) {
                                std::vector<std::string> keys;
                                keys.reserve(args.as_object().size());
                                for (const auto& [k, _] : args.as_object()) keys.push_back(k);
                                std::sort(keys.begin(), keys.end());
                                for (const auto& k : keys) {
                                    const Json& v = args[k];
                                    out += "<parameter=" + k + ">\n";
                                    out += v.is_string() ? v.as_string() : v.dump();
                                    out += "\n</parameter>\n";
                                }
                            }
                        } catch (...) {}
                    }
                    out += "</function>\n</tool_call>";
                }
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            bool prev_is_tool = (i > lead_sys_end && messages[i - 1].role == "tool");
            bool next_is_tool = (i + 1 < messages.size() && messages[i + 1].role == "tool");
            if (!prev_is_tool) out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (!next_is_tool) out += "<|im_end|>\n";
        }
    }

    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!enable_thinking) {
            out += "<think>\n\n</think>\n\n";
        } else {
            out += "<think>\n";
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode_chat_template(
    const std::vector<ChatMessage>& messages,
    bool add_generation_prompt,
    const std::string& reasoning_effort,
    const std::vector<std::string>& tools_json,
    size_t* out_system_prefix_len
) const {
    if (out_system_prefix_len) *out_system_prefix_len = 0;

    std::vector<int> out;
    std::string ord_buf;

    auto flush_ord = [&]() {
        if (!ord_buf.empty()) {
            encode_ordinary(ord_buf, out);
            ord_buf.clear();
        }
    };
    auto push_ord = [&](std::string_view sv) {
        if (!sv.empty()) ord_buf.append(sv.data(), sv.size());
    };
    auto push_added = [&](const std::string& tag) {
        flush_ord();
        auto it = added_to_id_.find(tag);
        if (it != added_to_id_.end()) {
            out.push_back(it->second);
        } else {
            auto ids = encode(tag, true);
            out.insert(out.end(), ids.begin(), ids.end());
        }
    };
    auto push_image = [&](int n_tokens) {
        flush_ord();
        push_added("<|vision_start|>");
        auto pit = added_to_id_.find("<|image_pad|>");
        if (pit != added_to_id_.end()) {
            out.insert(out.end(), static_cast<size_t>(std::max(0, n_tokens)), pit->second);
        } else {
            for (int k = 0; k < n_tokens; ++k) push_added("<|image_pad|>");
        }
        push_added("<|vision_end|>");
    };
    auto push_parts = [&](const std::vector<ContentPart>& parts) {
        for (const auto& p : parts) {
            if (p.is_image) push_image(p.image_tokens);
            else push_ord(p.text);
        }
    };

    if (arch_ == "inkling2") {
        static const std::unordered_map<std::string, std::string> eff_map = {
            {"none", "0"}, {"off", "0"}, {"minimal", "0.1"}, {"low", "0.2"},
            {"medium", "0.7"}, {"high", "0.9"}, {"xhigh", "0.99"}, {"max", "0.99"}
        };
        std::string eff_val = "0";
        auto eit = eff_map.find(reasoning_effort);
        if (eit != eff_map.end()) eff_val = eit->second;
        else if (!reasoning_effort.empty()) eff_val = reasoning_effort;

        bool has_leading_sys = (!tools_json.empty() || (!messages.empty() && messages[0].role == "system"));
        bool sys_has_img = false;

        if (!tools_json.empty()) {
            Json::Array specs;
            for (const auto& tj : tools_json) {
                try {
                    Json t = parse_json(tj);
                    const Json& fn = t.contains("function") ? t["function"] : t;
                    Json::Object spec;
                    spec["description"] = fn.contains("description") && fn["description"].is_string()
                        ? fn["description"].as_string() : "";
                    spec["name"] = fn.contains("name") ? fn["name"].as_string() : "";
                    spec["parameters"] = fn.contains("parameters") ? fn["parameters"] : Json(Json::Object{});
                    spec["type"] = t.contains("type") && t["type"].is_string() ? t["type"].as_string() : "function";
                    specs.push_back(Json(std::move(spec)));
                } catch (...) {}
            }
            push_added("<|message_system|>");
            push_ord("tool_declare");
            push_added("<|content_xml|>");
            push_ord(Json(std::move(specs)).dump());
            push_added("<|end_message|>");
        }

        auto emit_effort = [&]() {
            push_added("<|message_system|>");
            push_added("<|content_text|>");
            push_ord("Thinking effort level: ");
            push_ord(eff_val);
            push_added("<|end_message|>");
        };

        bool effort_emitted = false;
        for (size_t idx = 0; idx < messages.size(); ++idx) {
            const auto& m = messages[idx];
            if (!effort_emitted && m.role != "system") {
                emit_effort();
                effort_emitted = true;
                if (out_system_prefix_len && has_leading_sys && !sys_has_img) {
                    flush_ord();
                    *out_system_prefix_len = out.size();
                }
            }
            const char* role_tok = "<|message_user|>";
            if (m.role == "system") role_tok = "<|message_system|>";
            else if (m.role == "assistant") role_tok = "<|message_model|>";
            else if (m.role == "tool") role_tok = "<|message_tool|>";

            std::string role_suffix;
            if (m.role == "tool") {
                if (!m.name.empty()) {
                    role_suffix = m.name;
                } else if (!m.tool_call_id.empty()) {
                    bool found_tc = false;
                    for (const auto& prev : messages) {
                        if (prev.role == "assistant") {
                            for (const auto& tc : prev.tool_calls) {
                                if (tc.id == m.tool_call_id && !tc.name.empty()) {
                                    role_suffix = tc.name;
                                    found_tc = true;
                                    break;
                                }
                            }
                            if (found_tc) break;
                        }
                    }
                }
            } else if (m.role == "assistant" && !m.name.empty()) {
                role_suffix = m.name;
            }

            if (m.role == "assistant" && !m.reasoning_content.empty()) {
                push_added(role_tok);
                push_ord(role_suffix);
                push_added("<|content_thinking|>");
                push_ord(m.reasoning_content);
                push_added("<|end_message|>");
            }

            bool has_content = !m.parts.empty() || !m.content.empty();
            if (has_content || !(m.role == "assistant" && (!m.tool_calls.empty() || !m.reasoning_content.empty()))) {
                push_added(role_tok);
                push_ord(role_suffix);
                push_added("<|content_text|>");
                if (!m.parts.empty()) {
                    for (const auto& p : m.parts) {
                        if (p.is_image) {
                            if (!effort_emitted) sys_has_img = true;
                            push_image(p.image_tokens);
                        } else {
                            push_ord(p.text);
                        }
                    }
                } else {
                    push_ord(m.content);
                }
                push_added("<|end_message|>");
            }

            if (m.role == "assistant" && !m.tool_calls.empty()) {
                for (const auto& tc : m.tool_calls) {
                    Json::Object invoke;
                    invoke["name"] = tc.name;
                    try {
                        invoke["args"] = tc.arguments_json.empty() ? Json(Json::Object{}) : parse_json(tc.arguments_json);
                    } catch (...) {
                        invoke["args"] = Json(Json::Object{});
                    }
                    push_added("<|message_model|>");
                    push_ord(tc.name);
                    push_added("<|content_invoke_tool_json|>");
                    push_ord(Json(std::move(invoke)).dump());
                    push_added("<|end_message|>");
                }
            }
            if (m.role == "assistant") {
                push_added("<|content_model_end_sampling|>");
            }
        }
        if (!effort_emitted) {
            emit_effort();
            if (out_system_prefix_len && has_leading_sys && !sys_has_img) {
                flush_ord();
                *out_system_prefix_len = out.size();
            }
        }
        if (add_generation_prompt) {
            push_added("<|message_model|>");
            push_added(eff_val == "0" ? "<|content_text|>" : "<|content_thinking|>");
        }
        flush_ord();
        return out;
    }

    // Qwen 3.5 / 3.8 ChatML piecewise tokenization
    bool enable_thinking = !(reasoning_effort == "none" || reasoning_effort == "off");
    std::string reasoning_instr;
    if (enable_thinking) {
        if (reasoning_effort == "xhigh" || reasoning_effort == "high" || reasoning_effort == "max" || reasoning_effort.empty()) {
            reasoning_instr = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        } else if (reasoning_effort == "low" || reasoning_effort == "minimal") {
            reasoning_instr = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
        }
    }

    size_t lead_sys_end = 0;
    std::vector<ContentPart> sys_lead_parts;
    bool sys_has_img = false;
    while (lead_sys_end < messages.size() && messages[lead_sys_end].role == "system") {
        auto mp = trimmed_message_parts(messages[lead_sys_end]);
        if (!mp.empty()) {
            if (!sys_lead_parts.empty()) sys_lead_parts.push_back(ContentPart{false, "\n\n", 0});
            for (auto& p : mp) {
                if (p.is_image) sys_has_img = true;
                sys_lead_parts.push_back(std::move(p));
            }
        }
        ++lead_sys_end;
    }
    bool has_sys0 = (lead_sys_end > 0);
    bool emitted_lead_sys = false;

    if (!tools_json.empty()) {
        emitted_lead_sys = true;
        push_added("<|im_start|>");
        push_ord("system\n");
        if (!reasoning_instr.empty()) {
            push_ord(reasoning_instr);
            push_ord("\n\n");
        }
        push_ord("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (const auto& tj : tools_json) {
            push_ord("\n");
            push_ord(tj);
        }
        push_ord("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
        push_added("<tool_call>");
        push_ord("\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                 "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
                 "</parameter>\n</function>\n");
        push_added("</tool_call>");
        push_ord("\n\n<IMPORTANT>\nReminder:\n"
                 "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within ");
        push_added("<tool_call>");
        push_added("</tool_call>");
        push_ord(" XML tags\n"
                 "- Required parameters MUST be specified\n"
                 "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
                 "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
                 "</IMPORTANT>");
        if (!sys_lead_parts.empty()) {
            push_ord("\n\n");
            push_parts(sys_lead_parts);
        }
        push_added("<|im_end|>");
        push_ord("\n");
    } else if (has_sys0) {
        if (!sys_lead_parts.empty()) {
            emitted_lead_sys = true;
            push_added("<|im_start|>");
            push_ord("system\n");
            if (!reasoning_instr.empty()) {
                push_ord(reasoning_instr);
                push_ord("\n\n");
            }
            push_parts(sys_lead_parts);
            push_added("<|im_end|>");
            push_ord("\n");
        } else if (!reasoning_instr.empty()) {
            emitted_lead_sys = true;
            push_added("<|im_start|>");
            push_ord("system\n");
            push_ord(reasoning_instr);
            push_added("<|im_end|>");
            push_ord("\n");
        }
    } else if (!reasoning_instr.empty()) {
        emitted_lead_sys = true;
        push_added("<|im_start|>");
        push_ord("system\n");
        push_ord(reasoning_instr);
        push_added("<|im_end|>");
        push_ord("\n");
    }

    if (out_system_prefix_len && emitted_lead_sys && !sys_has_img) {
        flush_ord();
        *out_system_prefix_len = out.size();
    }

    for (size_t i = lead_sys_end; i < messages.size(); ++i) {
        const auto& m = messages[i];
        auto parts = trimmed_message_parts(m);
        if (m.role == "system") {
            if (!parts.empty()) {
                push_added("<|im_start|>");
                push_ord("system\n");
                push_parts(parts);
                push_added("<|im_end|>");
                push_ord("\n");
            }
        } else if (m.role == "user") {
            push_added("<|im_start|>");
            push_ord("user\n");
            push_parts(parts);
            push_added("<|im_end|>");
            push_ord("\n");
        } else if (m.role == "assistant") {
            std::string rc = trim_ws(m.reasoning_content);
            push_added("<|im_start|>");
            push_ord("assistant\n");
            push_added("<think>");
            if (rc.empty()) {
                push_ord("\n\n");
            } else {
                push_ord("\n");
                push_ord(rc);
                push_ord("\n");
            }
            push_added("</think>");
            push_ord("\n\n");
            push_parts(parts);
            if (!m.tool_calls.empty()) {
                for (size_t j = 0; j < m.tool_calls.size(); ++j) {
                    const auto& tc = m.tool_calls[j];
                    if (j == 0) {
                        if (!parts.empty()) push_ord("\n\n");
                    } else {
                        push_ord("\n");
                    }
                    push_added("<tool_call>");
                    push_ord("\n<function=");
                    push_ord(tc.name);
                    push_ord(">\n");
                    if (!tc.arguments_json.empty()) {
                        try {
                            Json args = parse_json(tc.arguments_json);
                            if (args.is_object()) {
                                std::vector<std::string> keys;
                                keys.reserve(args.as_object().size());
                                for (const auto& [k, _] : args.as_object()) keys.push_back(k);
                                std::sort(keys.begin(), keys.end());
                                for (const auto& k : keys) {
                                    const Json& v = args[k];
                                    push_ord("<parameter=");
                                    push_ord(k);
                                    push_ord(">\n");
                                    push_ord(v.is_string() ? v.as_string() : v.dump());
                                    push_ord("\n</parameter>\n");
                                }
                            }
                        } catch (...) {}
                    }
                    push_ord("</function>\n");
                    push_added("</tool_call>");
                }
            }
            push_added("<|im_end|>");
            push_ord("\n");
        } else if (m.role == "tool") {
            bool prev_is_tool = (i > lead_sys_end && messages[i - 1].role == "tool");
            bool next_is_tool = (i + 1 < messages.size() && messages[i + 1].role == "tool");
            if (!prev_is_tool) {
                push_added("<|im_start|>");
                push_ord("user");
            }
            push_ord("\n");
            push_added("<tool_response>");
            push_ord("\n");
            push_parts(parts);
            push_ord("\n");
            push_added("</tool_response>");
            if (!next_is_tool) {
                push_added("<|im_end|>");
                push_ord("\n");
            }
        }
    }

    if (add_generation_prompt) {
        push_added("<|im_start|>");
        push_ord("assistant\n");
        push_added("<think>");
        if (!enable_thinking) {
            push_ord("\n\n");
            push_added("</think>");
            push_ord("\n\n");
        } else {
            push_ord("\n");
        }
    }

    flush_ord();
    return out;
}

} // namespace mi
