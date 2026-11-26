#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mi {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json; // JSON object string
};

struct ContentPart {
    bool is_image = false;
    std::string text;     // untrusted text when !is_image (encoded with parse_special=false)
    int image_tokens = 0; // exact <|image_pad|> count when is_image
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::string name;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
    std::vector<ContentPart> parts; // optional structured multimodal parts; if empty, uses `content`
};

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    void load(const std::string& model_dir, const std::string& arch);

    std::vector<int> encode(const std::string& text, bool parse_special = true) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
    std::string decode_token(int id, bool skip_special = false) const;

    // Format messages using native Qwen3.8 or Inkling 2 Mini chat template (with full tool + reasoning support)
    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true,
        const std::string& reasoning_effort = "none",
        const std::vector<std::string>& tools_json = {}
    ) const;

    // Safe piecewise chat-template tokenization:
    // - template control tags use parse_special=true
    // - untrusted message/tool/reasoning text uses parse_special=false (blocks <|im_end|>/<|image_pad|> injection)
    // - ContentPart{is_image=true} inserts exact <|vision_start|> + N*<|image_pad|> + <|vision_end|> token IDs
    // - Optional out_system_prefix_len receives the exact token count of the leading pure-text system+tools block
    std::vector<int> encode_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt = true,
        const std::string& reasoning_effort = "none",
        const std::vector<std::string>& tools_json = {},
        size_t* out_system_prefix_len = nullptr
    ) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool is_special(int id) const;

private:
    std::string arch_;
    bool use_nfc_ = false;
    void* regex_handle_ = nullptr; // NSRegularExpression* (retained via CFBridgingRetain)

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    // All added_tokens (both special:true and special:false) matched during encode(..., parse_special=true)
    std::unordered_map<std::string, int> added_to_id_;
    std::vector<std::string> added_tokens_sorted_; // longest first
    bool added_first_byte_[256] = {};
    // Only added_tokens with "special": true (stripped when skip_special=true; reported by is_special())
    std::unordered_set<int> special_ids_;
    std::unordered_map<uint64_t, int> merge_ranks_;  // (id_a << 32) | id_b -> rank

    // ByteLevel GPT-2 byte <-> unicode char tables
    std::string byte_to_utf8_[256];
    std::unordered_map<std::string, uint8_t> utf8_to_byte_;

    void init_byte_level_tables();
    std::vector<int> bpe_encode_piece(const std::string& raw_bytes) const;
    void encode_ordinary(const std::string& text, std::vector<int>& out) const;
};

} // namespace mi
