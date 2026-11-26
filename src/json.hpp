#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mi {

struct Json {
    using Object = std::unordered_map<std::string, Json>;
    using Array = std::vector<Json>;
    using Value = std::variant<std::nullptr_t, bool, double, int64_t, std::string, Array, Object>;

    Value val = nullptr;

    Json() = default;
    Json(std::nullptr_t) : val(nullptr) {}
    Json(bool b) : val(b) {}
    Json(int64_t i) : val(i) {}
    Json(int i) : val(int64_t(i)) {}
    Json(double d) : val(d) {}
    Json(std::string s) : val(std::move(s)) {}
    Json(const char* s) : val(std::string(s)) {}
    Json(Array a) : val(std::move(a)) {}
    Json(Object o) : val(std::move(o)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(val); }
    bool is_bool() const { return std::holds_alternative<bool>(val); }
    bool is_int() const { return std::holds_alternative<int64_t>(val); }
    bool is_double() const { return std::holds_alternative<double>(val); }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return std::holds_alternative<std::string>(val); }
    bool is_array() const { return std::holds_alternative<Array>(val); }
    bool is_object() const { return std::holds_alternative<Object>(val); }

    bool as_bool(bool def = false) const {
        if (auto* p = std::get_if<bool>(&val)) return *p;
        return def;
    }
    int64_t as_int(int64_t def = 0) const {
        if (auto* p = std::get_if<int64_t>(&val)) return *p;
        if (auto* p = std::get_if<double>(&val)) {
            if (!std::isfinite(*p) || *p < -9223372036854775808.0 || *p >= 9223372036854775808.0) return def;
            return static_cast<int64_t>(*p);
        }
        return def;
    }
    double as_double(double def = 0.0) const {
        if (auto* p = std::get_if<double>(&val)) return *p;
        if (auto* p = std::get_if<int64_t>(&val)) return static_cast<double>(*p);
        return def;
    }
    const std::string& as_string() const {
        static const std::string empty;
        if (auto* p = std::get_if<std::string>(&val)) return *p;
        return empty;
    }
    const Array& as_array() const {
        static const Array empty;
        if (auto* p = std::get_if<Array>(&val)) return *p;
        return empty;
    }
    const Object& as_object() const {
        static const Object empty;
        if (auto* p = std::get_if<Object>(&val)) return *p;
        return empty;
    }

    bool contains(const std::string& k) const {
        if (auto* o = std::get_if<Object>(&val)) return o->find(k) != o->end();
        return false;
    }
    const Json& operator[](const std::string& k) const {
        static const Json null_json;
        if (auto* o = std::get_if<Object>(&val)) {
            auto it = o->find(k);
            if (it != o->end()) return it->second;
        }
        return null_json;
    }
    const Json& operator[](size_t idx) const {
        static const Json null_json;
        if (auto* a = std::get_if<Array>(&val)) {
            if (idx < a->size()) return (*a)[idx];
        }
        return null_json;
    }

    static std::string escape_string(std::string_view s) {
        std::string out;
        out.reserve(s.size() + 2);
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
            }
        }
        out.push_back('"');
        return out;
    }

    std::string dump() const {
        if (is_null()) return "null";
        if (is_bool()) return as_bool() ? "true" : "false";
        if (is_int()) return std::to_string(as_int());
        if (is_double()) {
            double d = as_double();
            if (!std::isfinite(d)) return "null";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g", d);
            return buf;
        }
        if (is_string()) return escape_string(as_string());
        if (is_array()) {
            std::string out = "[";
            const auto& a = as_array();
            for (size_t i = 0; i < a.size(); ++i) {
                if (i) out.push_back(',');
                out += a[i].dump();
            }
            out.push_back(']');
            return out;
        }
        std::string out = "{";
        const auto& o = as_object();
        std::vector<const std::string*> keys;
        keys.reserve(o.size());
        for (const auto& [k, _] : o) keys.push_back(&k);
        std::sort(keys.begin(), keys.end(), [](const std::string* a, const std::string* b) { return *a < *b; });
        for (size_t i = 0; i < keys.size(); ++i) {
            if (i) out.push_back(',');
            out += escape_string(*keys[i]);
            out.push_back(':');
            out += o.at(*keys[i]).dump();
        }
        out.push_back('}');
        return out;
    }
};

class JsonParser {
public:
    static constexpr int MAX_DEPTH = 128;

    explicit JsonParser(std::string_view src) : data_(src.data()), end_(src.data() + src.size()), p_(src.data()) {}

    Json parse() {
        skip_ws();
        Json r = parse_value();
        skip_ws();
        if (p_ != end_) throw std::runtime_error("JSON: trailing characters after value");
        return r;
    }

private:
    const char* data_;
    const char* end_;
    const char* p_;
    int depth_ = 0;

    struct DepthGuard {
        int& depth;
        explicit DepthGuard(int& d) : depth(d) {
            if (++depth > MAX_DEPTH) throw std::runtime_error("JSON: maximum nesting depth exceeded");
        }
        ~DepthGuard() { --depth; }
    };

    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    uint32_t parse_hex4() {
        if (end_ - p_ < 4) throw std::runtime_error("JSON: truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
            else throw std::runtime_error("JSON: bad hex escape");
        }
        return v;
    }

    std::string parse_string() {
        if (p_ >= end_ || *p_ != '"') throw std::runtime_error("JSON: expected '\"'");
        ++p_;
        std::string out;
        const char* seg_start = p_;
        while (p_ < end_) {
            unsigned char c = static_cast<unsigned char>(*p_);
            if (c == '"') {
                if (p_ > seg_start) out.append(seg_start, p_ - seg_start);
                ++p_;
                return out;
            }
            if (c == '\\') {
                if (p_ > seg_start) out.append(seg_start, p_ - seg_start);
                ++p_;
                if (p_ >= end_) throw std::runtime_error("JSON: truncated escape");
                char esc = *p_++;
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                                const char* saved_p = p_;
                                p_ += 2;
                                uint32_t low = parse_hex4();
                                if (low >= 0xDC00 && low <= 0xDFFF) {
                                    cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                                } else {
                                    // Lone high surrogate followed by non-low \uXXXX: rewind so next escape isn't dropped
                                    p_ = saved_p;
                                    cp = 0xFFFD;
                                }
                            } else {
                                cp = 0xFFFD;
                            }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            cp = 0xFFFD;
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        out.push_back(esc);
                        break;
                }
                seg_start = p_;
            } else {
                ++p_;
            }
        }
        throw std::runtime_error("JSON: unterminated string");
    }

    Json parse_number() {
        const char* start = p_;
        bool is_float = false;
        if (p_ < end_ && *p_ == '-') ++p_;
        if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
            throw std::runtime_error("JSON: invalid number");
        }
        if (*p_ == '0') {
            ++p_;
            if (p_ < end_ && *p_ >= '0' && *p_ <= '9') {
                throw std::runtime_error("JSON: leading zero in number");
            }
        } else {
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (p_ < end_ && *p_ == '.') {
            is_float = true;
            ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
                throw std::runtime_error("JSON: invalid fractional number");
            }
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            is_float = true;
            ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            if (p_ >= end_ || *p_ < '0' || *p_ > '9') {
                throw std::runtime_error("JSON: invalid exponent in number");
            }
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        std::string tmp(start, static_cast<size_t>(p_ - start));
        if (is_float) return Json(std::strtod(tmp.c_str(), nullptr));
        errno = 0;
        char* ep = nullptr;
        long long v = std::strtoll(tmp.c_str(), &ep, 10);
        if (errno == ERANGE || ep != tmp.c_str() + tmp.size()) {
            return Json(std::strtod(tmp.c_str(), nullptr));
        }
        return Json(static_cast<int64_t>(v));
    }

    Json parse_array() {
        DepthGuard guard(depth_);
        ++p_; // '['
        Json::Array arr;
        skip_ws();
        if (p_ < end_ && *p_ == ']') { ++p_; return Json(std::move(arr)); }
        while (p_ < end_) {
            skip_ws();
            arr.push_back(parse_value());
            skip_ws();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == ']') { ++p_; return Json(std::move(arr)); }
            throw std::runtime_error("JSON: expected ',' or ']'");
        }
        throw std::runtime_error("JSON: unterminated array");
    }

    Json parse_object() {
        DepthGuard guard(depth_);
        ++p_; // '{'
        Json::Object obj;
        skip_ws();
        if (p_ < end_ && *p_ == '}') { ++p_; return Json(std::move(obj)); }
        while (p_ < end_) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (p_ >= end_ || *p_ != ':') throw std::runtime_error("JSON: expected ':'");
            ++p_;
            skip_ws();
            obj.insert_or_assign(std::move(key), parse_value());
            skip_ws();
            if (p_ < end_ && *p_ == ',') { ++p_; continue; }
            if (p_ < end_ && *p_ == '}') { ++p_; return Json(std::move(obj)); }
            throw std::runtime_error("JSON: expected ',' or '}'");
        }
        throw std::runtime_error("JSON: unterminated object");
    }

    Json parse_value() {
        skip_ws();
        if (p_ >= end_) throw std::runtime_error("JSON: unexpected EOF");
        char c = *p_;
        if (c == '"') return Json(parse_string());
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' && end_ - p_ >= 4 && std::memcmp(p_, "true", 4) == 0) { p_ += 4; return Json(true); }
        if (c == 'f' && end_ - p_ >= 5 && std::memcmp(p_, "false", 5) == 0) { p_ += 5; return Json(false); }
        if (c == 'n' && end_ - p_ >= 4 && std::memcmp(p_, "null", 4) == 0) { p_ += 4; return Json(nullptr); }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw std::runtime_error(std::string("JSON: unexpected char '") + c + "'");
    }
};

inline Json parse_json(std::string_view src) {
    return JsonParser(src).parse();
}

} // namespace mi
