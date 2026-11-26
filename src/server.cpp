#include "server.hpp"
#include "json.hpp"
#include "prefix_cache.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace mi {

namespace {

constexpr size_t MAX_HEADER_BYTES = 256ull * 1024ull;
constexpr size_t MAX_BODY_BYTES   = 64ull * 1024ull * 1024ull;
constexpr size_t MAX_TRAILER_BYTES = 64ull * 1024ull;
constexpr int    MAX_ACTIVE_CONNS = 32;

ssize_t recv_some(int fd, char* buf, size_t cap) {
    while (true) {
        ssize_t n = recv(fd, buf, cap, 0);
        if (n < 0 && errno == EINTR) continue;
        return n;
    }
}

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, data + sent, len - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool send_str(int fd, const std::string& s) {
    return send_all(fd, s.data(), s.size());
}

void send_json_response(int fd, int status, const std::string& status_text, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    send_str(fd, oss.str());
}

std::string trim_ows(std::string_view sv) {
    size_t a = 0;
    while (a < sv.size() && (sv[a] == ' ' || sv[a] == '\t')) ++a;
    size_t b = sv.size();
    while (b > a && (sv[b - 1] == ' ' || sv[b - 1] == '\t')) --b;
    return std::string(sv.substr(a, b - a));
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool header_has_token_ci(const std::string& val, const char* needle) {
    std::string lo = to_lower_ascii(val);
    size_t pos = 0;
    while (pos < lo.size()) {
        size_t comma = lo.find(',', pos);
        std::string_view part = (comma == std::string::npos)
            ? std::string_view(lo).substr(pos)
            : std::string_view(lo).substr(pos, comma - pos);
        if (trim_ows(part) == needle) return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

bool parse_strict_u64_dec(const std::string& s, size_t& out) {
    if (s.empty()) return false;
    for (unsigned char c : s) {
        if (c < '0' || c > '9') return false;
    }
    errno = 0;
    char* ep = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &ep, 10);
    if (errno == ERANGE || ep != s.c_str() + s.size()) return false;
    out = static_cast<size_t>(v);
    return true;
}

bool parse_strict_chunk_hex(std::string_view line, size_t& out) {
    // Strip optional chunk extension ";..."
    size_t semi = line.find(';');
    if (semi != std::string_view::npos) line = line.substr(0, semi);
    std::string s = trim_ows(line);
    if (s.empty()) return false;
    for (unsigned char c : s) {
        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!ok) return false;
    }
    errno = 0;
    char* ep = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &ep, 16);
    if (errno == ERANGE || ep != s.c_str() + s.size()) return false;
    out = static_cast<size_t>(v);
    return true;
}

bool read_http_request(int fd, std::string& method, std::string& path, std::string& body,
                       int& err_status, std::string& err_msg) {
    err_status = 0;
    err_msg.clear();
    method.clear();
    path.clear();
    body.clear();

    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = recv_some(fd, tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
        hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos && buf.size() > MAX_HEADER_BYTES) {
            err_status = 431;
            err_msg = "Request header fields too large";
            return false;
        }
    }
    if (hdr_end > MAX_HEADER_BYTES) {
        err_status = 431;
        err_msg = "Request header fields too large";
        return false;
    }

    std::string_view hdrs(buf.data(), hdr_end);
    size_t first_nl = hdrs.find("\r\n");
    std::string_view req_line = (first_nl == std::string_view::npos) ? hdrs : hdrs.substr(0, first_nl);
    {
        std::istringstream iss{std::string(req_line)};
        iss >> method >> path;
    }
    if (method.empty() || path.empty()) {
        err_status = 400;
        err_msg = "Malformed HTTP request line";
        return false;
    }

    bool has_content_len = false;
    size_t content_len = 0;
    bool chunked = false;
    bool expect_100 = false;

    if (first_nl != std::string_view::npos) {
        size_t pos = first_nl + 2;
        while (pos < hdrs.size()) {
            size_t nl = hdrs.find("\r\n", pos);
            std::string_view line = (nl == std::string_view::npos)
                ? hdrs.substr(pos) : hdrs.substr(pos, nl - pos);
            size_t colon = line.find(':');
            if (colon != std::string_view::npos) {
                std::string name = to_lower_ascii(trim_ows(line.substr(0, colon)));
                std::string val  = trim_ows(line.substr(colon + 1));
                if (name == "content-length") {
                    size_t parsed_cl = 0;
                    if (!parse_strict_u64_dec(val, parsed_cl)) {
                        err_status = 400;
                        err_msg = "Invalid Content-Length header";
                        return false;
                    }
                    if (has_content_len && parsed_cl != content_len) {
                        err_status = 400;
                        err_msg = "Conflicting Content-Length headers";
                        return false;
                    }
                    if (parsed_cl > MAX_BODY_BYTES) {
                        err_status = 413;
                        err_msg = "Request body too large";
                        return false;
                    }
                    has_content_len = true;
                    content_len = parsed_cl;
                } else if (name == "transfer-encoding") {
                    if (header_has_token_ci(val, "chunked")) {
                        chunked = true;
                    }
                } else if (name == "expect") {
                    if (to_lower_ascii(val) == "100-continue") {
                        expect_100 = true;
                    }
                }
            }
            if (nl == std::string_view::npos) break;
            pos = nl + 2;
        }
    }

    if (expect_100) {
        if (!send_str(fd, "HTTP/1.1 100 Continue\r\n\r\n")) return false;
    }

    body = buf.substr(hdr_end + 4);
    if (chunked) {
        std::string raw = std::move(body);
        body.clear();
        size_t pos = 0;
        while (true) {
            size_t line_end = std::string::npos;
            while ((line_end = raw.find("\r\n", pos)) == std::string::npos) {
                if (raw.size() - pos > 4096) {
                    err_status = 400;
                    err_msg = "Malformed chunk header";
                    return false;
                }
                ssize_t n = recv_some(fd, tmp, sizeof(tmp));
                if (n <= 0) return false;
                raw.append(tmp, static_cast<size_t>(n));
            }
            size_t chunk_sz = 0;
            if (!parse_strict_chunk_hex(std::string_view(raw.data() + pos, line_end - pos), chunk_sz)) {
                err_status = 400;
                err_msg = "Invalid chunk size";
                return false;
            }
            pos = line_end + 2;
            if (chunk_sz == 0) {
                // Drain optional trailer headers and terminating CRLF so macOS closes with clean TCP FIN (not RST)
                size_t trailer_bytes = 0;
                while (true) {
                    size_t tl_end = std::string::npos;
                    while ((tl_end = raw.find("\r\n", pos)) == std::string::npos) {
                        if (trailer_bytes + (raw.size() - pos) > MAX_TRAILER_BYTES) {
                            err_status = 400;
                            err_msg = "Chunked trailer too large";
                            return false;
                        }
                        ssize_t n = recv_some(fd, tmp, sizeof(tmp));
                        if (n <= 0) return false;
                        raw.append(tmp, static_cast<size_t>(n));
                    }
                    if (tl_end == pos) {
                        pos += 2; // Final empty line "\r\n"
                        break;
                    }
                    trailer_bytes += (tl_end - pos) + 2;
                    if (trailer_bytes > MAX_TRAILER_BYTES) {
                        err_status = 400;
                        err_msg = "Chunked trailer too large";
                        return false;
                    }
                    pos = tl_end + 2;
                }
                return true;
            }
            if (chunk_sz > MAX_BODY_BYTES || body.size() > MAX_BODY_BYTES - chunk_sz) {
                err_status = 413;
                err_msg = "Chunked request body too large";
                return false;
            }
            const size_t need = chunk_sz + 2;
            while (raw.size() - pos < need) {
                ssize_t n = recv_some(fd, tmp, sizeof(tmp));
                if (n <= 0) return false;
                raw.append(tmp, static_cast<size_t>(n));
            }
            if (raw[pos + chunk_sz] != '\r' || raw[pos + chunk_sz + 1] != '\n') {
                err_status = 400;
                err_msg = "Missing CRLF after chunk payload";
                return false;
            }
            body.append(raw, pos, chunk_sz);
            pos += need;
            if (pos >= 65536) {
                raw.erase(0, pos);
                pos = 0;
            }
        }
    }

    if (content_len > MAX_BODY_BYTES) {
        err_status = 413;
        err_msg = "Request body too large";
        return false;
    }
    while (body.size() < content_len) {
        ssize_t n = recv_some(fd, tmp, sizeof(tmp));
        if (n <= 0) return false;
        body.append(tmp, static_cast<size_t>(n));
        if (body.size() > MAX_BODY_BYTES) {
            err_status = 413;
            err_msg = "Request body too large";
            return false;
        }
    }
    if (body.size() > content_len) body.resize(content_len);
    return true;
}

bool is_eos(int tok, const std::vector<int>& eos_ids) {
    for (int e : eos_ids) if (tok == e) return true;
    return false;
}

// Tool parameter schema type map: tool_name -> (param_name -> type string)
using ToolSchemaMap = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

ToolSchemaMap extract_tool_schemas(const Json& req) {
    ToolSchemaMap out;
    if (!req.contains("tools") || !req["tools"].is_array()) return out;
    for (const auto& t : req["tools"].as_array()) {
        const Json& fn = t.contains("function") ? t["function"] : t;
        if (!fn.contains("name") || !fn["name"].is_string()) continue;
        std::string name = fn["name"].as_string();
        if (fn.contains("parameters") && fn["parameters"].is_object() &&
            fn["parameters"].contains("properties") && fn["parameters"]["properties"].is_object()) {
            for (const auto& [pk, pv] : fn["parameters"]["properties"].as_object()) {
                if (pv.is_object() && pv.contains("type") && pv["type"].is_string()) {
                    out[name][pk] = pv["type"].as_string();
                }
            }
        }
    }
    return out;
}

// Find the real closing </parameter> tag (avoiding premature match if value contains literal "</parameter>")
size_t find_parameter_close(const std::string& body, size_t from) {
    size_t first_hit = std::string::npos;
    size_t search = from;
    while (true) {
        size_t pos = body.find("</parameter>", search);
        if (pos == std::string::npos) return first_hit;
        if (first_hit == std::string::npos) first_hit = pos;
        size_t after = pos + 12;
        while (after < body.size() && (body[after] == ' ' || body[after] == '\t' || body[after] == '\n' || body[after] == '\r')) {
            ++after;
        }
        if (after >= body.size() ||
            body.compare(after, 11, "<parameter=") == 0 ||
            body.compare(after, 11, "</function>") == 0) {
            return pos;
        }
        search = pos + 12;
    }
}

// Parse Qwen XML tool call block:
//   <function=NAME>\n<parameter=K1>\nV1\n</parameter>\n...</function>
// Also accepts JSON fallback: {"name": "...", "arguments": {...}}
bool parse_qwen_tool_call(const std::string& body, const ToolSchemaMap& schemas,
                          std::string& out_name, std::string& out_args_json) {
    size_t fn_pos = body.find("<function=");
    if (fn_pos != std::string::npos) {
        size_t fn_end = body.find('>', fn_pos + 10);
        if (fn_end == std::string::npos) return false;
        out_name = body.substr(fn_pos + 10, fn_end - (fn_pos + 10));
        while (!out_name.empty() && std::isspace(static_cast<unsigned char>(out_name.back()))) out_name.pop_back();
        while (!out_name.empty() && std::isspace(static_cast<unsigned char>(out_name.front()))) out_name.erase(out_name.begin());

        Json::Object args_obj;
        auto sit = schemas.find(out_name);
        size_t p = fn_end + 1;
        while (true) {
            size_t ps = body.find("<parameter=", p);
            if (ps == std::string::npos) break;
            size_t pe = body.find('>', ps + 11);
            if (pe == std::string::npos) break;
            std::string pname = body.substr(ps + 11, pe - (ps + 11));
            size_t vend = find_parameter_close(body, pe + 1);
            if (vend == std::string::npos) break;
            std::string raw_val = body.substr(pe + 1, vend - (pe + 1));
            if (!raw_val.empty() && raw_val.front() == '\n') raw_val.erase(raw_val.begin());
            if (!raw_val.empty() && raw_val.back() == '\n') raw_val.pop_back();

            std::string expected_type = "string";
            if (sit != schemas.end()) {
                auto pit = sit->second.find(pname);
                if (pit != sit->second.end()) expected_type = pit->second;
            }

            if (expected_type != "string") {
                try {
                    args_obj[pname] = parse_json(raw_val);
                } catch (...) {
                    args_obj[pname] = raw_val;
                }
            } else {
                args_obj[pname] = raw_val;
            }
            p = vend + 12;
        }
        out_args_json = Json(std::move(args_obj)).dump();
        return !out_name.empty();
    }

    // JSON fallback: {"name": "...", "arguments": {...}}
    try {
        Json j = parse_json(body);
        if (j.is_object() && j.contains("name") && j["name"].is_string()) {
            out_name = j["name"].as_string();
            if (j.contains("arguments")) {
                out_args_json = j["arguments"].is_string() ? j["arguments"].as_string() : j["arguments"].dump();
            } else if (j.contains("args")) {
                out_args_json = j["args"].is_string() ? j["args"].as_string() : j["args"].dump();
            } else {
                out_args_json = "{}";
            }
            return true;
        }
    } catch (...) {}
    return false;
}

// Check longest suffix of `s` that is a prefix of `tag` (so we never leak partial XML tags / stop seqs while streaming)
size_t partial_tag_overlap(const std::string& s, const char* tag) {
    size_t tlen = std::strlen(tag);
    if (tlen <= 1 || s.empty()) return 0;
    size_t max_k = std::min(s.size(), tlen - 1);
    for (size_t k = max_k; k > 0; --k) {
        if (s.compare(s.size() - k, k, tag, k) == 0) return k;
    }
    return 0;
}

// Drain all complete, RFC 3629-valid UTF-8 code points from `buf` into `out`.
void drain_valid_utf8(std::string& buf, std::string& out, bool flush_all) {
    static constexpr const char kRepl[4] = "\xEF\xBF\xBD";
    const size_t n = buf.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char b0 = static_cast<unsigned char>(buf[i]);
        if (b0 < 0x80u) {
            out.push_back(static_cast<char>(b0));
            ++i;
            continue;
        }
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
            out.append(kRepl, 3);
            ++i;
            continue;
        }

        const size_t avail = std::min(need, n - i);
        size_t bad_at = 0;
        for (size_t k = 1; k < avail; ++k) {
            const unsigned char bk = static_cast<unsigned char>(buf[i + k]);
            const unsigned char lo = (k == 1) ? b1_lo : 0x80u;
            const unsigned char hi = (k == 1) ? b1_hi : 0xBFu;
            if (bk < lo || bk > hi) {
                bad_at = k;
                break;
            }
        }
        if (bad_at > 0) {
            out.append(kRepl, 3);
            i += bad_at;
            continue;
        }
        if (avail < need) {
            if (flush_all) {
                out.append(kRepl, 3);
                i = n;
            }
            break;
        }
        out.append(buf, i, need);
        i += need;
    }
    if (i == n) buf.clear();
    else if (i > 0) buf.erase(0, i);
}

std::string sanitize_utf8(const std::string& s) {
    std::string copy = s;
    std::string out;
    out.reserve(s.size());
    drain_valid_utf8(copy, out, true);
    return out;
}

struct ServerState {
    Model& model;
    const Tokenizer& tok;
    const ServerConfig& cfg;
    std::string model_id;
    PrefixCacheManager disk_cache;
    std::mutex infer_mtx;
    std::vector<int> state_tokens;
    std::vector<uint64_t> state_image_hashes;
    std::atomic<int> active_conns{0};
    std::atomic<int> cached_position{0};
    std::atomic<int64_t> last_req_ns{0};

    ServerState(Model& m, const Tokenizer& t, const ServerConfig& c, std::string mid)
        : model(m), tok(t), cfg(c), model_id(std::move(mid)), disk_cache(c.cache_dir) {
        last_req_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                          std::memory_order_relaxed);
    }
};

struct PendingImageRef {
    size_t msg_idx = 0;
    size_t part_idx = 0;
    std::string url;
};

void handle_connection(ServerState& st, int fd) {
    struct ConnGuard {
        ServerState& st;
        int& fd;
        ~ConnGuard() {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
            st.active_conns.fetch_sub(1, std::memory_order_acq_rel);
        }
    } conn_guard{st, fd};

    std::string method, path, body, err_msg;
    int err_status = 0;
    if (!read_http_request(fd, method, path, body, err_status, err_msg)) {
        if (err_status != 0) {
            Json::Object eo;
            eo["error"] = err_msg;
            const char* stxt = (err_status == 413) ? "Payload Too Large"
                             : (err_status == 431) ? "Request Header Fields Too Large"
                             : "Bad Request";
            send_json_response(fd, err_status, stxt, Json(std::move(eo)).dump());
        }
        return;
    }

    st.last_req_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                         std::memory_order_relaxed);

    // Strip optional ?query string from route path
    std::string route = path;
    size_t qm = route.find('?');
    if (qm != std::string::npos) route.resize(qm);

    if (method == "OPTIONS") {
        std::string resp =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key, Accept, HTTP-Referer, X-Title\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Connection: close\r\n\r\n";
        send_str(fd, resp);
        return;
    }

    if (route == "/health") {
        Json::Object obj;
        obj["status"] = "ok";
        obj["arch"] = st.model.info().arch;
        obj["model"] = st.model_id;
        obj["position"] = st.cached_position.load(std::memory_order_relaxed);
        send_json_response(fd, 200, "OK", Json(std::move(obj)).dump());
        return;
    }

    if (route == "/v1/models") {
        Json::Object m;
        m["id"] = st.model_id;
        m["object"] = "model";
        m["owned_by"] = "mometal";
        Json::Array data;
        data.push_back(Json(std::move(m)));
        Json::Object resp;
        resp["object"] = "list";
        resp["data"] = std::move(data);
        send_json_response(fd, 200, "OK", Json(std::move(resp)).dump());
        return;
    }

    if (method != "POST" || (route != "/v1/chat/completions" && route != "/v1/completions")) {
        send_json_response(fd, 404, "Not Found", "{\"error\":\"not found\"}");
        return;
    }

    // Parse and validate request JSON OUTSIDE infer_mtx so malformed requests fail fast with 400
    Json req;
    try {
        req = parse_json(body);
    } catch (const std::exception& ex) {
        Json::Object err;
        err["error"] = std::string("Invalid JSON: ") + ex.what();
        send_json_response(fd, 400, "Bad Request", Json(std::move(err)).dump());
        return;
    }
    if (!req.is_object()) {
        send_json_response(fd, 400, "Bad Request", "{\"error\":\"JSON body must be an object\"}");
        return;
    }

    const bool is_chat = (route == "/v1/chat/completions");
    const bool stream = req["stream"].as_bool(false);
    bool include_usage = true;
    if (req.contains("stream_options") && req["stream_options"].is_object() &&
        req["stream_options"].contains("include_usage")) {
        include_usage = req["stream_options"]["include_usage"].as_bool(true);
    }

    int max_tokens = 4096;
    if (req.contains("max_tokens") && req["max_tokens"].is_number()) {
        max_tokens = static_cast<int>(req["max_tokens"].as_int(4096));
    } else if (req.contains("max_completion_tokens") && req["max_completion_tokens"].is_number()) {
        max_tokens = static_cast<int>(req["max_completion_tokens"].as_int(4096));
    }
    if (max_tokens < 0) {
        send_json_response(fd, 400, "Bad Request", "{\"error\":\"max_tokens must be >= 0\"}");
        return;
    }

    SamplerConfig sc;
    sc.temperature      = static_cast<float>(req["temperature"].as_double(0.7));
    sc.top_k            = static_cast<int>(req["top_k"].as_int(20));
    sc.top_p            = static_cast<float>(req["top_p"].as_double(0.95));
    sc.presence_penalty = static_cast<float>(req["presence_penalty"].as_double(0.0));
    sc.repeat_penalty   = static_cast<float>(req["repetition_penalty"].as_double(1.0));
    if (req.contains("seed") && req["seed"].is_number()) {
        sc.seed = static_cast<uint64_t>(req["seed"].as_int());
    } else {
        sc.seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    // Parse OpenAI `stop` sequences (string or array of strings)
    std::vector<std::string> stop_seqs;
    if (req.contains("stop") && !req["stop"].is_null()) {
        if (req["stop"].is_string()) {
            const auto& s = req["stop"].as_string();
            if (!s.empty()) stop_seqs.push_back(s);
        } else if (req["stop"].is_array()) {
            for (const auto& sv : req["stop"].as_array()) {
                if (!sv.is_string()) {
                    send_json_response(fd, 400, "Bad Request", "{\"error\":\"stop array elements must be strings\"}");
                    return;
                }
                if (!sv.as_string().empty() && stop_seqs.size() < 16) {
                    stop_seqs.push_back(sv.as_string());
                }
            }
        } else {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"stop must be a string or array of strings\"}");
            return;
        }
    }

    // Resolve reasoning / thinking configuration across OpenAI / Qwen / OpenRouter / Baseten formats
    std::string effort = "none";
    if (req.contains("reasoning_effort") && req["reasoning_effort"].is_string()) {
        effort = req["reasoning_effort"].as_string();
    } else if (req.contains("reasoning") && req["reasoning"].is_object() &&
               req["reasoning"].contains("effort") && req["reasoning"]["effort"].is_string()) {
        effort = req["reasoning"]["effort"].as_string();
    }
    if (req.contains("enable_thinking") && req["enable_thinking"].is_bool()) {
        bool et = req["enable_thinking"].as_bool();
        if (!et) effort = "none";
        else if (effort == "none") effort = "medium";
    }
    if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object()) {
        const auto& ctk = req["chat_template_kwargs"];
        if (ctk.contains("enable_thinking") && ctk["enable_thinking"].is_bool()) {
            if (!ctk["enable_thinking"].as_bool()) effort = "none";
            else if (effort == "none") effort = "medium";
        }
        if (ctk.contains("reasoning_effort") && ctk["reasoning_effort"].is_string()) {
            effort = ctk["reasoning_effort"].as_string();
        }
    }
    const bool thinking_enabled = !(effort == "none" || effort == "off");

    std::vector<std::string> tools_json;
    if (req.contains("tools") && req["tools"].is_array()) {
        for (const auto& t : req["tools"].as_array()) {
            tools_json.push_back(t.dump());
        }
    }
    ToolSchemaMap tool_schemas = extract_tool_schemas(req);

    std::vector<ChatMessage> msgs;
    std::vector<PendingImageRef> pending_images;
    std::vector<int> completion_prompt_ids;

    if (is_chat) {
        if (!req.contains("messages") || !req["messages"].is_array() || req["messages"].as_array().empty()) {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"messages must be a non-empty array\"}");
            return;
        }
        for (const auto& jm : req["messages"].as_array()) {
            if (!jm.is_object() || !jm.contains("role") || !jm["role"].is_string()) {
                send_json_response(fd, 400, "Bad Request", "{\"error\":\"each message must be an object with string role\"}");
                return;
            }
            ChatMessage cm;
            cm.role = jm["role"].as_string();
            if (cm.role == "developer") cm.role = "system";
            if (jm.contains("name") && jm["name"].is_string()) cm.name = jm["name"].as_string();
            if (jm.contains("tool_call_id") && jm["tool_call_id"].is_string()) cm.tool_call_id = jm["tool_call_id"].as_string();
            if (jm.contains("reasoning_content") && jm["reasoning_content"].is_string()) {
                cm.reasoning_content = jm["reasoning_content"].as_string();
            }
            if (jm["content"].is_string()) {
                cm.content = jm["content"].as_string();
            } else if (jm["content"].is_array()) {
                for (const auto& part : jm["content"].as_array()) {
                    if (part.is_string()) {
                        cm.parts.push_back(ContentPart{false, part.as_string(), 0});
                    } else if (part.is_object()) {
                        std::string ptype = (part.contains("type") && part["type"].is_string())
                            ? part["type"].as_string() : "";
                        std::string img_url;
                        if (ptype == "image_url" && part.contains("image_url")) {
                            if (part["image_url"].is_string()) {
                                img_url = part["image_url"].as_string();
                            } else if (part["image_url"].is_object() &&
                                       part["image_url"].contains("url") &&
                                       part["image_url"]["url"].is_string()) {
                                img_url = part["image_url"]["url"].as_string();
                            }
                        } else if (ptype == "image") {
                            if (part.contains("source") && part["source"].is_object()) {
                                const auto& src = part["source"];
                                if (src.contains("type") && src["type"].is_string() &&
                                    src["type"].as_string() == "base64" &&
                                    src.contains("data") && src["data"].is_string()) {
                                    std::string mt = (src.contains("media_type") && src["media_type"].is_string())
                                        ? src["media_type"].as_string() : "image/png";
                                    img_url = "data:" + mt + ";base64," + src["data"].as_string();
                                } else if (src.contains("url") && src["url"].is_string()) {
                                    img_url = src["url"].as_string();
                                }
                            } else if (part.contains("data") && part["data"].is_string()) {
                                std::string mt = (part.contains("mimeType") && part["mimeType"].is_string())
                                    ? part["mimeType"].as_string() : "image/png";
                                img_url = "data:" + mt + ";base64," + part["data"].as_string();
                            } else if (part.contains("image_url") && part["image_url"].is_string()) {
                                img_url = part["image_url"].as_string();
                            }
                        }

                        if (!img_url.empty()) {
                            size_t pidx = cm.parts.size();
                            cm.parts.push_back(ContentPart{true, {}, 0});
                            pending_images.push_back(PendingImageRef{msgs.size(), pidx, std::move(img_url)});
                        } else if (part.contains("text") && part["text"].is_string()) {
                            cm.parts.push_back(ContentPart{false, part["text"].as_string(), 0});
                        }
                    }
                }
            }
            if (jm.contains("tool_calls") && jm["tool_calls"].is_array()) {
                for (const auto& jtc : jm["tool_calls"].as_array()) {
                    ToolCall tc;
                    if (jtc.contains("id") && jtc["id"].is_string()) tc.id = jtc["id"].as_string();
                    const Json& fn = jtc.contains("function") ? jtc["function"] : jtc;
                    if (fn.contains("name") && fn["name"].is_string()) tc.name = fn["name"].as_string();
                    if (fn.contains("arguments")) {
                        tc.arguments_json = fn["arguments"].is_string()
                            ? fn["arguments"].as_string()
                            : fn["arguments"].dump();
                    }
                    cm.tool_calls.push_back(std::move(tc));
                }
            }
            msgs.push_back(std::move(cm));
        }
    } else {
        if (!req.contains("prompt")) {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt is required\"}");
            return;
        }
        const Json& jp = req["prompt"];
        if (jp.is_string()) {
            completion_prompt_ids = st.tok.encode(jp.as_string(), true);
        } else if (jp.is_array() && !jp.as_array().empty()) {
            const auto& arr = jp.as_array();
            if (arr[0].is_string()) {
                std::string joined;
                for (const auto& el : arr) {
                    if (!el.is_string()) {
                        send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt array must be homogeneous strings or token IDs\"}");
                        return;
                    }
                    joined += el.as_string();
                }
                completion_prompt_ids = st.tok.encode(joined, true);
            } else if (arr[0].is_int()) {
                const int V = st.model.info().vocab_size;
                completion_prompt_ids.reserve(arr.size());
                for (const auto& el : arr) {
                    if (!el.is_int()) {
                        send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt array must be homogeneous strings or token IDs\"}");
                        return;
                    }
                    int64_t tid = el.as_int();
                    if (tid < 0 || tid >= V) {
                        send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt token ID out of range\"}");
                        return;
                    }
                    completion_prompt_ids.push_back(static_cast<int>(tid));
                }
            } else {
                send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt must be a string or array\"}");
                return;
            }
        } else {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"prompt must be a non-empty string or array\"}");
            return;
        }
        if (completion_prompt_ids.empty()) {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"empty prompt\"}");
            return;
        }
    }

    // Serialize all GPU/Metal/vision/state access across concurrent HTTP connections
    std::lock_guard<std::mutex> infer_lk(st.infer_mtx);
    bool headers_sent = false;
    bool model_dirty = false;

    try {
        std::vector<EncodedImage> req_images;
        std::vector<uint64_t> req_image_hashes;
        double vis_ms = 0.0;

        if (!pending_images.empty()) {
            if (!st.model.has_vision()) {
                send_json_response(fd, 400, "Bad Request", "{\"error\":\"Model does not support image inputs\"}");
                return;
            }
            req_images.reserve(pending_images.size());
            req_image_hashes.reserve(pending_images.size());
            for (const auto& ref : pending_images) {
                auto tv0 = std::chrono::steady_clock::now();
                EncodedImage enc_img = st.model.encode_image_url(ref.url, 1280);
                vis_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tv0).count();
                msgs[ref.msg_idx].parts[ref.part_idx].image_tokens = enc_img.num_tokens();
                req_image_hashes.push_back(enc_img.content_hash);
                req_images.push_back(std::move(enc_img));
            }
        }

        size_t sys_prefix_len = 0;
        std::vector<int> prompt_ids;
        if (is_chat) {
            prompt_ids = st.tok.encode_chat_template(msgs, true, effort, tools_json, &sys_prefix_len);
        } else {
            prompt_ids = std::move(completion_prompt_ids);
        }
        if (prompt_ids.empty()) {
            send_json_response(fd, 400, "Bad Request", "{\"error\":\"empty prompt\"}");
            return;
        }
        if (prompt_ids.size() > static_cast<size_t>(st.model.info().max_seq_len)) {
            Json::Object eo;
            eo["error"] = "Prompt length (" + std::to_string(prompt_ids.size()) +
                          ") exceeds max_seq_len (" + std::to_string(st.model.info().max_seq_len) + ")";
            send_json_response(fd, 400, "Bad Request", Json(std::move(eo)).dump());
            return;
        }

        // Append-only prefix reuse check (require exact sync between state_tokens and model.position()!)
        bool can_reuse = !st.state_tokens.empty() &&
                         st.state_tokens.size() == static_cast<size_t>(st.model.position()) &&
                         prompt_ids.size() > st.state_tokens.size() &&
                         std::equal(st.state_tokens.begin(), st.state_tokens.end(), prompt_ids.begin()) &&
                         req_image_hashes.size() >= st.state_image_hashes.size() &&
                         std::equal(st.state_image_hashes.begin(), st.state_image_hashes.end(), req_image_hashes.begin());
        int start_idx = 0;
        size_t reused_images = 0;
        double restore_ms = 0.0;

        model_dirty = true;
        if (can_reuse) {
            start_idx = static_cast<int>(st.state_tokens.size());
            reused_images = st.state_image_hashes.size();
        } else {
            st.model.reset();
            st.state_tokens.clear();
            st.state_image_hashes.clear();
            st.cached_position.store(0, std::memory_order_relaxed);
            if (st.disk_cache.enabled()) {
                auto tr0 = std::chrono::steady_clock::now();
                std::vector<int> restored;
                std::vector<uint64_t> restored_imgs;
                int n_rest = st.disk_cache.try_restore(st.model, prompt_ids, restored,
                                                       req_image_hashes, &restored_imgs);
                if (n_rest > 0) {
                    auto tr1 = std::chrono::steady_clock::now();
                    restore_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                    st.state_tokens = std::move(restored);
                    st.state_image_hashes = std::move(restored_imgs);
                    reused_images = st.state_image_hashes.size();
                    start_idx = n_rest;
                    st.cached_position.store(st.model.position(), std::memory_order_relaxed);
                }
            }
        }

        std::vector<EncodedImage> new_images;
        if (reused_images < req_images.size()) {
            new_images.reserve(req_images.size() - reused_images);
            for (size_t i = reused_images; i < req_images.size(); ++i) {
                new_images.push_back(std::move(req_images[i]));
            }
        }

        Sampler sampler(sc);
        for (int id : prompt_ids) sampler.observe(id);

        auto t_req = std::chrono::steady_clock::now();
        StepStats pf_stats{};
        int argmax_tok = 0;
        const float* logits = nullptr;

        // Checkpoint ONLY a genuine leading pure-text system/developer+tools block (never single-turn user prompts)
        int sys_ckpt_len = 0;
        if (st.disk_cache.enabled() && is_chat && prompt_ids.size() >= 192 &&
            sys_prefix_len >= 128 && sys_prefix_len < prompt_ids.size() &&
            static_cast<int>(sys_prefix_len) > start_idx) {
            sys_ckpt_len = static_cast<int>(sys_prefix_len);
        }

        auto run_tail_forward = [&](const int* toks, int n, int* out_am, StepStats* stats) -> const float* {
            if (!new_images.empty()) {
                return st.model.forward_multimodal(toks, n, new_images, out_am, stats);
            }
            return st.model.forward(toks, n, out_am, stats);
        };

        const size_t new_pf_toks = prompt_ids.size() - static_cast<size_t>(start_idx);
        std::fprintf(stderr, "[req] prefill prompt=%zu (cached=%d, new=%zu, tools=%zu, stream=%d)...\n",
                     prompt_ids.size(), start_idx, new_pf_toks, tools_json.size(), stream ? 1 : 0);
        std::fflush(stderr);

        if (sys_ckpt_len > start_idx) {
            StepStats s1{};
            st.model.forward(prompt_ids.data() + start_idx, sys_ckpt_len - start_idx, nullptr, &s1);
            std::vector<int> sys_toks(prompt_ids.begin(), prompt_ids.begin() + sys_ckpt_len);
            st.disk_cache.save(st.model, sys_toks);
            StepStats s2{};
            logits = run_tail_forward(
                prompt_ids.data() + sys_ckpt_len,
                static_cast<int>(prompt_ids.size()) - sys_ckpt_len,
                sampler.is_greedy() ? &argmax_tok : nullptr,
                &s2
            );
            pf_stats.gpu_ms = s1.gpu_ms + s2.gpu_ms;
        } else {
            logits = run_tail_forward(
                prompt_ids.data() + start_idx,
                static_cast<int>(prompt_ids.size()) - start_idx,
                sampler.is_greedy() ? &argmax_tok : nullptr,
                &pf_stats
            );
        }
        st.state_tokens = prompt_ids;
        st.state_image_hashes = std::move(req_image_hashes);
        st.cached_position.store(st.model.position(), std::memory_order_relaxed);

        // Clamp max_tokens to remaining KV capacity so decode never overflows max_seq_len mid-stream
        const int remaining_ctx = std::max(0, st.model.info().max_seq_len - st.model.position());
        const int eff_max_tokens = std::min(max_tokens, remaining_ctx);

        int next_tok = 0;
        if (eff_max_tokens > 0) {
            next_tok = sampler.is_greedy() ? argmax_tok : sampler.sample(logits, st.model.info().vocab_size);
        }
        auto t_first = std::chrono::steady_clock::now();
        double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_req).count() + restore_ms + vis_ms;

        std::string req_id = "chatcmpl-" + std::to_string(sc.seed & 0xFFFFFFFFu);
        int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        bool conn_ok = true;
        if (stream) {
            std::string sse_hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
            if (!send_str(fd, sse_hdr)) {
                conn_ok = false;
            } else {
                headers_sent = true;
            }
        }

        // Streaming & non-streaming state machine for reasoning_content + content + tool_calls + stop sequences!
        enum class OutMode { THINKING, CONTENT, TOOL_CALL };
        const bool is_inkling = (st.model.info().arch == "inkling2");
        OutMode mode = (is_chat && thinking_enabled) ? OutMode::THINKING : OutMode::CONTENT;
        std::string pending;
        std::string tool_buf;
        std::string utf8_reasoning_buf;
        std::string utf8_content_buf;
        std::string stop_buf;
        std::string full_reasoning;
        std::string full_content;
        std::vector<ToolCall> emitted_tools;
        bool strip_post_think_nl = false;
        bool hit_stop = false;
        std::string finish_reason = "length";

        auto send_sse_delta = [&](Json::Object delta) -> bool {
            if (!stream) return true;
            if (!conn_ok) return false;
            Json::Object ch;
            ch["index"] = 0;
            if (is_chat) ch["delta"] = std::move(delta);
            else ch["text"] = delta.count("content") ? delta["content"].as_string() : "";
            ch["finish_reason"] = nullptr;
            Json::Array choices;
            choices.push_back(Json(std::move(ch)));
            Json::Object chunk;
            chunk["id"] = req_id;
            chunk["object"] = is_chat ? "chat.completion.chunk" : "text_completion";
            chunk["created"] = created;
            chunk["model"] = st.model_id;
            chunk["choices"] = std::move(choices);
            if (!send_str(fd, "data: " + Json(std::move(chunk)).dump() + "\n\n")) {
                conn_ok = false;
                return false;
            }
            return true;
        };

        // Emit standard initial SSE assistant role chunk for chat completions
        if (stream && is_chat && conn_ok) {
            Json::Object init_d;
            init_d["role"] = "assistant";
            send_sse_delta(std::move(init_d));
        }

        auto emit_reasoning = [&](const std::string& s, bool flush_utf8 = false) {
            if (!conn_ok) return;
            if (!s.empty()) utf8_reasoning_buf += s;
            if (utf8_reasoning_buf.empty()) return;
            std::string ready;
            ready.reserve(utf8_reasoning_buf.size());
            drain_valid_utf8(utf8_reasoning_buf, ready, flush_utf8);
            if (ready.empty()) return;
            full_reasoning += ready;
            Json::Object d;
            d["reasoning_content"] = ready;
            send_sse_delta(std::move(d));
        };

        auto emit_content_final = [&](const std::string& s) {
            if (!conn_ok || s.empty()) return;
            full_content += s;
            Json::Object d;
            d["content"] = s;
            send_sse_delta(std::move(d));
        };

        auto emit_content = [&](std::string s, bool flush_utf8 = false) {
            if (!conn_ok || hit_stop) return;
            if (!s.empty()) {
                if (strip_post_think_nl) {
                    size_t p = 0;
                    while (p < s.size() && (s[p] == '\n' || s[p] == '\r')) ++p;
                    if (p < s.size()) {
                        s.erase(0, p);
                        strip_post_think_nl = false;
                    } else {
                        s.clear();
                    }
                }
                utf8_content_buf += s;
            }
            std::string ready;
            if (!utf8_content_buf.empty()) {
                ready.reserve(utf8_content_buf.size());
                drain_valid_utf8(utf8_content_buf, ready, flush_utf8);
            }
            if (stop_seqs.empty()) {
                emit_content_final(ready);
                return;
            }

            if (!ready.empty()) stop_buf += ready;
            if (stop_buf.empty()) return;

            size_t earliest = std::string::npos;
            for (const auto& seq : stop_seqs) {
                size_t p = stop_buf.find(seq);
                if (p != std::string::npos && p < earliest) earliest = p;
            }
            if (earliest != std::string::npos) {
                if (earliest > 0) emit_content_final(stop_buf.substr(0, earliest));
                stop_buf.clear();
                hit_stop = true;
                finish_reason = "stop";
                return;
            }
            if (flush_utf8) {
                emit_content_final(stop_buf);
                stop_buf.clear();
                return;
            }
            size_t hold = 0;
            for (const auto& seq : stop_seqs) {
                hold = std::max(hold, partial_tag_overlap(stop_buf, seq.c_str()));
            }
            if (stop_buf.size() > hold) {
                size_t emit_len = stop_buf.size() - hold;
                emit_content_final(stop_buf.substr(0, emit_len));
                stop_buf.erase(0, emit_len);
            }
        };

        auto emit_tool_call = [&](const std::string& raw_name, const std::string& raw_args_json) {
            if (!conn_ok) return;
            std::string name = sanitize_utf8(raw_name);
            std::string args_json = sanitize_utf8(raw_args_json);
            int tc_idx = static_cast<int>(emitted_tools.size());
            std::string tc_id = "call_" + std::to_string((sc.seed & 0xFFFFFFu) + tc_idx);
            emitted_tools.push_back(ToolCall{tc_id, name, args_json});
            if (stream) {
                Json::Object fn;
                fn["name"] = name;
                fn["arguments"] = args_json;
                Json::Object tc;
                tc["index"] = tc_idx;
                tc["id"] = tc_id;
                tc["type"] = "function";
                tc["function"] = std::move(fn);
                Json::Array arr;
                arr.push_back(Json(std::move(tc)));
                Json::Object d;
                d["tool_calls"] = std::move(arr);
                send_sse_delta(std::move(d));
            }
        };

        auto pump_qwen_pending = [&](bool flush_all) {
            while (!pending.empty() && conn_ok && !hit_stop) {
                if (mode == OutMode::THINKING) {
                    size_t pos = pending.find("</think>");
                    if (pos != std::string::npos) {
                        emit_reasoning(pending.substr(0, pos), true);
                        pending.erase(0, pos + 8);
                        mode = OutMode::CONTENT;
                        strip_post_think_nl = true;
                        continue;
                    }
                    size_t hold = flush_all ? 0 : partial_tag_overlap(pending, "</think>");
                    if (pending.size() > hold) {
                        emit_reasoning(pending.substr(0, pending.size() - hold), flush_all);
                        pending.erase(0, pending.size() - hold);
                    } else if (flush_all) {
                        emit_reasoning("", true);
                    }
                    break;
                } else if (mode == OutMode::CONTENT) {
                    size_t pos = pending.find("<tool_call>");
                    if (pos != std::string::npos) {
                        emit_content(pending.substr(0, pos), true);
                        if (hit_stop) break;
                        pending.erase(0, pos + 11);
                        tool_buf.clear();
                        mode = OutMode::TOOL_CALL;
                        continue;
                    }
                    size_t hold = flush_all ? 0 : partial_tag_overlap(pending, "<tool_call>");
                    if (pending.size() > hold) {
                        emit_content(pending.substr(0, pending.size() - hold), flush_all);
                        pending.erase(0, pending.size() - hold);
                    } else if (flush_all) {
                        emit_content("", true);
                    }
                    break;
                } else { // TOOL_CALL
                    size_t pos = pending.find("</tool_call>");
                    if (pos != std::string::npos) {
                        tool_buf += pending.substr(0, pos);
                        pending.erase(0, pos + 12);
                        std::string fn_name, args_json;
                        if (parse_qwen_tool_call(tool_buf, tool_schemas, fn_name, args_json)) {
                            emit_tool_call(fn_name, args_json);
                        } else {
                            emit_content("<tool_call>" + tool_buf + "</tool_call>");
                        }
                        tool_buf.clear();
                        mode = OutMode::CONTENT;
                        continue;
                    }
                    size_t hold = flush_all ? 0 : partial_tag_overlap(pending, "</tool_call>");
                    if (pending.size() > hold) {
                        tool_buf += pending.substr(0, pending.size() - hold);
                        pending.erase(0, pending.size() - hold);
                    }
                    break;
                }
            }
            // Flush buffered incomplete tool call at EOS / max_tokens even when `pending` was already drained!
            if (flush_all && mode == OutMode::TOOL_CALL && !tool_buf.empty() && conn_ok && !hit_stop) {
                std::string fn_name, args_json;
                if (parse_qwen_tool_call(tool_buf, tool_schemas, fn_name, args_json)) {
                    emit_tool_call(fn_name, args_json);
                } else {
                    emit_content("<tool_call>" + tool_buf, true);
                }
                tool_buf.clear();
                mode = OutMode::CONTENT;
            }
        };

        int gen_count = 0;
        int mtp_acc = 0, mtp_dft = 0, pld_hits = 0;
        auto t_dec0 = std::chrono::steady_clock::now();

        auto emit_tok = [&](int tok_id, bool& was_emitted) -> bool {
            was_emitted = false;
            if (is_eos(tok_id, st.model.info().eos_token_ids)) {
                finish_reason = "stop";
                return false;
            }
            ++gen_count;
            was_emitted = true;

            if (is_inkling && is_chat) {
                std::string raw_sp = st.tok.decode_token(tok_id, false);
                if (raw_sp == "<|content_model_end_sampling|>") {
                    --gen_count;
                    was_emitted = false;
                    finish_reason = "stop";
                    return false;
                }
                if (raw_sp == "<|content_thinking|>") { emit_content("", true); mode = OutMode::THINKING; return conn_ok && !hit_stop && (gen_count < eff_max_tokens); }
                if (raw_sp == "<|content_text|>") { emit_reasoning("", true); mode = OutMode::CONTENT; return conn_ok && !hit_stop && (gen_count < eff_max_tokens); }
                if (raw_sp == "<|content_invoke_tool_json|>") {
                    emit_reasoning("", true);
                    emit_content("", true);
                    tool_buf.clear();
                    mode = OutMode::TOOL_CALL;
                    return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
                }
                if (raw_sp == "<|end_message|>") {
                    if (mode == OutMode::TOOL_CALL && !tool_buf.empty()) {
                        std::string fn_name, args_json;
                        if (parse_qwen_tool_call(tool_buf, tool_schemas, fn_name, args_json)) {
                            emit_tool_call(fn_name, args_json);
                        }
                        tool_buf.clear();
                    }
                    mode = OutMode::CONTENT;
                    return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
                }
                if (st.tok.is_special(tok_id)) return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
                std::string piece = st.tok.decode_token(tok_id, true);
                if (mode == OutMode::THINKING) emit_reasoning(piece);
                else if (mode == OutMode::TOOL_CALL) tool_buf += piece;
                else emit_content(piece);
                return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
            }

            // Keep structural XML added tokens (</think>, <tool_call>, </tool_call>) for Qwen's state machine!
            std::string piece = st.tok.decode_token(tok_id, !is_chat);
            if (!piece.empty()) {
                if (!is_chat) {
                    emit_content(piece);
                } else {
                    if (piece == "<|im_start|>" || piece == "<|im_end|>" || piece == "<|endoftext|>") {
                        return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
                    }
                    pending += piece;
                    pump_qwen_pending(false);
                }
            }
            return conn_ok && !hit_stop && (gen_count < eff_max_tokens);
        };

        bool first_emitted = false;
        bool keep_going = (conn_ok && eff_max_tokens > 0) ? emit_tok(next_tok, first_emitted) : false;
        std::vector<int> mtp_batch;
        while (keep_going) {
            if (st.model.position() >= st.model.info().max_seq_len) {
                finish_reason = "length";
                break;
            }
            sampler.observe(next_tok);
            if (sampler.is_greedy() && st.model.has_mtp()) {
                mtp_batch.clear();
                StepStats ms{};
                st.model.forward_mtp_greedy(next_tok, mtp_batch, &ms);
                mtp_acc += ms.mtp_accepted;
                mtp_dft += ms.mtp_drafted;
                pld_hits += ms.pld_hits;
                st.state_tokens.push_back(next_tok);
                for (size_t j = 0; j < mtp_batch.size(); ++j) {
                    int t = mtp_batch[j];
                    if (j + 1 < mtp_batch.size()) sampler.observe(t);
                    bool emitted_j = false;
                    keep_going = emit_tok(t, emitted_j);
                    if (emitted_j && !hit_stop && j + 1 < mtp_batch.size()) {
                        st.state_tokens.push_back(t);
                    }
                    if (!keep_going) break;
                }
                next_tok = mtp_batch.back();
            } else if (!sampler.is_greedy() && st.model.supports_mtp_sampled()) {
                mtp_batch.clear();
                StepStats ms{};
                st.model.forward_mtp_sampled(next_tok, sampler, mtp_batch, &ms);
                mtp_acc += ms.mtp_accepted;
                mtp_dft += ms.mtp_drafted;
                pld_hits += ms.pld_hits;
                st.state_tokens.push_back(next_tok);
                for (size_t j = 0; j < mtp_batch.size(); ++j) {
                    bool emitted_j = false;
                    keep_going = emit_tok(mtp_batch[j], emitted_j);
                    if (emitted_j && !hit_stop && j + 1 < mtp_batch.size()) {
                        st.state_tokens.push_back(mtp_batch[j]);
                    }
                    if (!keep_going) break;
                }
                next_tok = mtp_batch.back();
            } else if (sampler.is_greedy()) {
                st.state_tokens.push_back(next_tok);
                next_tok = st.model.forward_greedy(next_tok);
                bool emitted_n = false;
                keep_going = emit_tok(next_tok, emitted_n);
            } else {
                st.state_tokens.push_back(next_tok);
                const float* lg = st.model.forward(&next_tok, 1);
                next_tok = sampler.sample(lg, st.model.info().vocab_size);
                bool emitted_n = false;
                keep_going = emit_tok(next_tok, emitted_n);
            }
        }

        // If a speculative batch advanced model.position() past the emitted tokens (e.g. mid-batch EOS/stop/max_tokens),
        // invalidate state_tokens so we neither reuse nor persist mismatched cache state.
        if (static_cast<int>(st.state_tokens.size()) != st.model.position()) {
            st.state_tokens.clear();
            st.state_image_hashes.clear();
        }
        st.cached_position.store(st.model.position(), std::memory_order_relaxed);

        if (!hit_stop) {
            if (is_chat && !is_inkling) {
                pump_qwen_pending(true);
            } else if (is_chat && is_inkling && mode == OutMode::TOOL_CALL && !tool_buf.empty() && conn_ok) {
                std::string fn_name, args_json;
                if (parse_qwen_tool_call(tool_buf, tool_schemas, fn_name, args_json)) {
                    emit_tool_call(fn_name, args_json);
                } else {
                    emit_content(tool_buf, true);
                }
                tool_buf.clear();
                mode = OutMode::CONTENT;
            }
            emit_reasoning("", true);
            emit_content("", true);
        } else {
            emit_reasoning("", true);
        }

        if (!emitted_tools.empty() && finish_reason == "stop") {
            finish_reason = "tool_calls";
        }

        auto t_dec1 = std::chrono::steady_clock::now();
        double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
        double dec_tps = (gen_count > 1 && dec_s > 0.0) ? ((gen_count - 1) / dec_s) : 0.0;
        std::fprintf(stderr, "[req] prompt=%zu (cached=%d, rest=%.1fms, vis=%.1fms, imgs=%zu) gen=%d tools=%zu ttft=%.1fms decode=%.1f tok/s",
                     prompt_ids.size(), start_idx, restore_ms, vis_ms, st.state_image_hashes.size(),
                     gen_count, emitted_tools.size(), ttft_ms, dec_tps);
        if (mtp_dft > 0) {
            std::fprintf(stderr, " mtp=%d/%d(%.0f%%,pld=%d)", mtp_acc, mtp_dft, 100.0 * mtp_acc / mtp_dft, pld_hits);
        }
        std::fprintf(stderr, "\n");

        Json::Object usage;
        usage["prompt_tokens"] = static_cast<int64_t>(prompt_ids.size());
        usage["completion_tokens"] = static_cast<int64_t>(gen_count);
        usage["total_tokens"] = static_cast<int64_t>(prompt_ids.size() + gen_count);
        if (start_idx > 0) {
            Json::Object pd;
            pd["cached_tokens"] = static_cast<int64_t>(start_idx);
            usage["prompt_tokens_details"] = std::move(pd);
        }

        if (stream) {
            if (conn_ok) {
                Json::Object ch;
                ch["index"] = 0;
                if (is_chat) ch["delta"] = Json::Object{};
                else ch["text"] = "";
                ch["finish_reason"] = finish_reason;
                Json::Array choices;
                choices.push_back(Json(std::move(ch)));
                Json::Object end_chunk;
                end_chunk["id"] = req_id;
                end_chunk["object"] = is_chat ? "chat.completion.chunk" : "text_completion";
                end_chunk["created"] = created;
                end_chunk["model"] = st.model_id;
                end_chunk["choices"] = std::move(choices);
                send_str(fd, "data: " + Json(std::move(end_chunk)).dump() + "\n\n");

                if (include_usage) {
                    Json::Object usage_chunk;
                    usage_chunk["id"] = req_id;
                    usage_chunk["object"] = is_chat ? "chat.completion.chunk" : "text_completion";
                    usage_chunk["created"] = created;
                    usage_chunk["model"] = st.model_id;
                    usage_chunk["choices"] = Json::Array{};
                    usage_chunk["usage"] = std::move(usage);
                    send_str(fd, "data: " + Json(std::move(usage_chunk)).dump() + "\n\n");
                }
                send_str(fd, "data: [DONE]\n\n");
            }
        } else {
            Json::Object ch;
            ch["index"] = 0;
            ch["finish_reason"] = finish_reason;
            if (is_chat) {
                Json::Object msg;
                msg["role"] = "assistant";
                msg["content"] = (full_content.empty() && !emitted_tools.empty()) ? Json(nullptr) : Json(full_content);
                if (!full_reasoning.empty()) msg["reasoning_content"] = full_reasoning;
                if (!emitted_tools.empty()) {
                    Json::Array tcs;
                    for (const auto& tc : emitted_tools) {
                        Json::Object fn;
                        fn["name"] = tc.name;
                        fn["arguments"] = tc.arguments_json;
                        Json::Object jtc;
                        jtc["id"] = tc.id;
                        jtc["type"] = "function";
                        jtc["function"] = std::move(fn);
                        tcs.push_back(Json(std::move(jtc)));
                    }
                    msg["tool_calls"] = std::move(tcs);
                }
                ch["message"] = std::move(msg);
            } else {
                ch["text"] = full_content;
            }
            Json::Array choices;
            choices.push_back(Json(std::move(ch)));

            Json::Object resp;
            resp["id"] = req_id;
            resp["object"] = is_chat ? "chat.completion" : "text_completion";
            resp["created"] = created;
            resp["model"] = st.model_id;
            resp["choices"] = std::move(choices);
            resp["usage"] = std::move(usage);
            send_json_response(fd, 200, "OK", Json(std::move(resp)).dump());
        }
        model_dirty = false;
    } catch (const std::exception& ex) {
        if (model_dirty) {
            st.model.reset();
            st.state_tokens.clear();
            st.state_image_hashes.clear();
            st.cached_position.store(0, std::memory_order_relaxed);
        }
        if (headers_sent) {
            Json::Object err_obj;
            err_obj["message"] = std::string(ex.what());
            err_obj["type"] = "server_error";
            Json::Object evt;
            evt["error"] = std::move(err_obj);
            send_str(fd, "data: " + Json(std::move(evt)).dump() + "\n\n");
            send_str(fd, "data: [DONE]\n\n");
        } else {
            Json::Object err;
            err["error"] = std::string(ex.what());
            send_json_response(fd, 500, "Internal Server Error", Json(std::move(err)).dump());
        }
        return;
    }

    // Close client socket BEFORE background disk checkpoint serialization
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
    if (st.disk_cache.enabled() && st.state_tokens.size() >= 256 &&
        static_cast<int>(st.state_tokens.size()) == st.model.position()) {
        st.disk_cache.save(st.model, st.state_tokens, st.state_image_hashes);
    }
}

} // namespace

void run_http_server(Model& model, const Tokenizer& tok, const ServerConfig& cfg) {
    signal(SIGPIPE, SIG_IGN);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) throw std::runtime_error("socket() failed");

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) <= 0) {
        close(srv);
        throw std::runtime_error("Invalid host IP: " + cfg.host);
    }
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(srv);
        throw std::runtime_error("bind() failed on " + cfg.host + ":" + std::to_string(cfg.port));
    }
    if (listen(srv, 64) < 0) {
        close(srv);
        throw std::runtime_error("listen() failed");
    }

    // Pre-fault & wire all mmap GPU weights once at startup
    {
        auto tw0 = std::chrono::steady_clock::now();
        model.forward_greedy(100);
        model.reset();
        double w_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
        std::fprintf(stderr, "[mometal] GPU warmup complete in %.1f ms\n", w_ms);
    }

    std::string model_id = cfg.model_id.empty() ? model.info().model_dir : cfg.model_id;
    std::fprintf(stderr, "[mometal] Serving %s (%s) on http://%s:%d\n",
                 model_id.c_str(), model.info().arch.c_str(), cfg.host.c_str(), cfg.port);

    ServerState st(model, tok, cfg, model_id);
    if (st.disk_cache.enabled()) {
        std::fprintf(stderr, "[mometal] Disk prefix cache enabled: %s\n", cfg.cache_dir.c_str());
    }

    while (true) {
        pollfd pfd{};
        pfd.fd = srv;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 15000);
        if (pr == 0) {
            int64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            int64_t idle_s = (now_ns - st.last_req_ns.load(std::memory_order_relaxed)) / 1000000000ll;
            if (idle_s < 900) {
                std::unique_lock<std::mutex> lk(st.infer_mtx, std::try_to_lock);
                if (lk.owns_lock()) {
                    st.model.keep_warm();
                }
            }
            continue;
        }
        if (pr < 0) continue;

        sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int fd = accept(srv, reinterpret_cast<sockaddr*>(&cli), &cli_len);
        if (fd < 0) continue;

        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
#ifdef SO_NOSIGPIPE
        int nosig = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosig, sizeof(nosig));
#endif
        timeval rtv{15, 0};
        timeval stv{30, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));

        if (st.active_conns.fetch_add(1, std::memory_order_acq_rel) >= MAX_ACTIVE_CONNS) {
            st.active_conns.fetch_sub(1, std::memory_order_acq_rel);
            send_json_response(fd, 503, "Service Unavailable", "{\"error\":\"too many active connections\"}");
            close(fd);
            continue;
        }

        std::thread([&st, fd]() {
            handle_connection(st, fd);
        }).detach();
    }
}

} // namespace mi
