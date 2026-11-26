#include "server.hpp"
#include "json.hpp"
#include "prefix_cache.hpp"
#include "sampler.hpp"

#include <arpa/inet.h>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace mi {

namespace {

bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, data + sent, len - sent, 0);
        if (n <= 0) return false;
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

bool read_http_request(int fd, std::string& method, std::string& path, std::string& body) {
    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
        hdr_end = buf.find("\r\n\r\n");
        if (buf.size() > 32 * 1024 * 1024) return false;
    }

    std::string hdrs = buf.substr(0, hdr_end);
    size_t first_nl = hdrs.find("\r\n");
    std::string req_line = hdrs.substr(0, first_nl);
    {
        std::istringstream iss(req_line);
        iss >> method >> path;
    }

    size_t content_len = 0;
    bool chunked = false;
    std::string lower_hdrs = hdrs;
    for (char& c : lower_hdrs) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    size_t cl_pos = lower_hdrs.find("content-length:");
    if (cl_pos != std::string::npos) {
        content_len = std::strtoull(hdrs.c_str() + cl_pos + 15, nullptr, 10);
    }
    if (lower_hdrs.find("transfer-encoding: chunked") != std::string::npos) {
        chunked = true;
    }

    body = buf.substr(hdr_end + 4);
    if (chunked) {
        std::string raw = std::move(body);
        body.clear();
        size_t pos = 0;
        while (true) {
            while (raw.find("\r\n", pos) == std::string::npos) {
                ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) return !body.empty();
                raw.append(tmp, static_cast<size_t>(n));
            }
            size_t line_end = raw.find("\r\n", pos);
            size_t chunk_sz = std::strtoull(raw.substr(pos, line_end - pos).c_str(), nullptr, 16);
            pos = line_end + 2;
            if (chunk_sz == 0) break;
            while (raw.size() < pos + chunk_sz + 2) {
                ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
                if (n <= 0) break;
                raw.append(tmp, static_cast<size_t>(n));
            }
            body.append(raw, pos, std::min(chunk_sz, raw.size() - pos));
            pos += chunk_sz + 2;
        }
        return true;
    }

    while (body.size() < content_len) {
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        body.append(tmp, static_cast<size_t>(n));
    }
    if (body.size() > content_len) body.resize(content_len);
    return true;
}

bool is_eos(int tok, const std::vector<int>& eos_ids) {
    for (int e : eos_ids) if (tok == e) return true;
    return false;
}

std::string trim_newlines(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == '\n' || s[a] == '\r')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return s.substr(a, b - a);
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
        // Trim whitespace from function name
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
            size_t vend = body.find("</parameter>", pe + 1);
            if (vend == std::string::npos) break;
            std::string raw_val = body.substr(pe + 1, vend - (pe + 1));
            // Strip exactly one leading and one trailing newline added by the XML format
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

// Check longest suffix of `s` that is a prefix of `tag` (so we never leak partial XML tags while streaming)
size_t partial_tag_overlap(const std::string& s, const char* tag) {
    size_t tlen = std::strlen(tag);
    size_t max_k = std::min(s.size(), tlen - 1);
    for (size_t k = max_k; k > 0; --k) {
        if (s.compare(s.size() - k, k, tag, k) == 0) return k;
    }
    return 0;
}

// Drain all complete, RFC 3629-valid UTF-8 code points from `buf` into `out`.
// When `flush_all == false`, any valid incomplete trailing 2/3/4-byte prefix (e.g. a multi-byte
// UTF-8 char split across ByteLevel BPE tokens) stays buffered in `buf` for the next token.
// Any genuinely invalid byte (or incomplete tail when `flush_all == true`) is replaced with
// U+FFFD ("\xEF\xBF\xBD") so emitted SSE JSON frames are guaranteed 100% valid UTF-8.
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
            need = 3; b1_lo = 0x80u; b1_hi = 0x9Fu; // reject UTF-16 surrogates U+D800..U+DFFF
        } else if (b0 == 0xF0u) {
            need = 4; b1_lo = 0x90u; b1_hi = 0xBFu;
        } else if (b0 >= 0xF1u && b0 <= 0xF3u) {
            need = 4;
        } else if (b0 == 0xF4u) {
            need = 4; b1_lo = 0x80u; b1_hi = 0x8Fu; // cap at U+10FFFF
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
            break; // keep valid incomplete prefix in buf for next token
        }
        out.append(buf, i, need);
        i += need;
    }
    if (i == n) buf.clear();
    else if (i > 0) buf.erase(0, i);
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
    if (listen(srv, 16) < 0) {
        close(srv);
        throw std::runtime_error("listen() failed");
    }

    // Pre-fault & wire all mmap GPU weights (Attention + GDN + MLP + MTP) into Metal page tables once at startup
    // so the very first user request has ZERO page-fault lag on either ANE prefill or GPU decode!
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

    // Track exact token sequence & image hashes currently in GPU KV/recurrent state for append-only prefix reuse
    std::vector<int> state_tokens;
    std::vector<uint64_t> state_image_hashes;
    PrefixCacheManager disk_cache(cfg.cache_dir);
    if (disk_cache.enabled()) {
        std::fprintf(stderr, "[mometal] Disk prefix cache enabled: %s\n", cfg.cache_dir.c_str());
    }

    auto last_req_time = std::chrono::steady_clock::now();
    while (true) {
        // Round 11: poll() with 15s timeout -> lightweight GPU+ANE keep-warm while interactively active (<15 min idle)
        pollfd pfd{};
        pfd.fd = srv;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 15000);
        if (pr == 0) {
            auto idle_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - last_req_time).count();
            if (idle_s < 900) {
                model.keep_warm();
            }
            continue;
        }
        if (pr < 0) continue;

        sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        int fd = accept(srv, reinterpret_cast<sockaddr*>(&cli), &cli_len);
        if (fd < 0) continue;
        // Round 34: Disable Nagle's algorithm so every SSE token chunk flushes with sub-ms latency!
        int nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        last_req_time = std::chrono::steady_clock::now();

        std::string method, path, body;
        if (!read_http_request(fd, method, path, body)) {
            close(fd);
            continue;
        }

        if (method == "OPTIONS") {
            std::string resp =
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                "Connection: close\r\n\r\n";
            send_str(fd, resp);
            close(fd);
            continue;
        }

        if (path == "/health") {
            Json::Object obj;
            obj["status"] = "ok";
            obj["arch"] = model.info().arch;
            obj["model"] = model_id;
            obj["position"] = model.position();
            send_json_response(fd, 200, "OK", Json(std::move(obj)).dump());
            close(fd);
            continue;
        }

        if (path == "/v1/models") {
            Json::Object m;
            m["id"] = model_id;
            m["object"] = "model";
            m["owned_by"] = "mometal";
            Json::Array data;
            data.push_back(Json(std::move(m)));
            Json::Object resp;
            resp["object"] = "list";
            resp["data"] = std::move(data);
            send_json_response(fd, 200, "OK", Json(std::move(resp)).dump());
            close(fd);
            continue;
        }

        if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions")) {
            try {
                Json req = parse_json(body);
                bool is_chat = (path == "/v1/chat/completions");
                bool stream = req["stream"].as_bool(false);
                bool include_usage = true;
                if (req.contains("stream_options") && req["stream_options"].is_object() &&
                    req["stream_options"].contains("include_usage")) {
                    include_usage = req["stream_options"]["include_usage"].as_bool(true);
                }

                int max_tokens = static_cast<int>(req["max_tokens"].as_int(
                    req.contains("max_completion_tokens") ? req["max_completion_tokens"].as_int(4096) : 4096));
                SamplerConfig sc;
                sc.temperature      = static_cast<float>(req["temperature"].as_double(0.7));
                sc.top_k            = static_cast<int>(req["top_k"].as_int(20));
                sc.top_p            = static_cast<float>(req["top_p"].as_double(0.95));
                sc.presence_penalty = static_cast<float>(req["presence_penalty"].as_double(0.0));
                sc.repeat_penalty   = static_cast<float>(req["repetition_penalty"].as_double(1.0));
                sc.seed             = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());

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
                bool thinking_enabled = !(effort == "none" || effort == "off");

                // Extract OpenAI tools
                std::vector<std::string> tools_json;
                if (req.contains("tools") && req["tools"].is_array()) {
                    for (const auto& t : req["tools"].as_array()) {
                        tools_json.push_back(t.dump());
                    }
                }
                ToolSchemaMap tool_schemas = extract_tool_schemas(req);

                std::string prompt;
                std::vector<EncodedImage> req_images;
                std::vector<uint64_t> req_image_hashes;
                double vis_ms = 0.0;
                if (is_chat) {
                    std::vector<ChatMessage> msgs;
                    for (const auto& jm : req["messages"].as_array()) {
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
                                    cm.content += part.as_string();
                                } else if (part.is_object()) {
                                    std::string ptype = part.contains("type") && part["type"].is_string()
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
                                        if (!model.has_vision()) {
                                            throw std::runtime_error("Model does not support image inputs");
                                        }
                                        auto tv0 = std::chrono::steady_clock::now();
                                        EncodedImage enc_img = model.encode_image_url(img_url, 1280);
                                        vis_ms += std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - tv0).count();
                                        cm.content += "<|vision_start|>";
                                        int n_vis = enc_img.num_tokens();
                                        cm.content.reserve(cm.content.size() + static_cast<size_t>(n_vis) * 13 + 16);
                                        for (int k = 0; k < n_vis; ++k) {
                                            cm.content += "<|image_pad|>";
                                        }
                                        cm.content += "<|vision_end|>";
                                        req_image_hashes.push_back(enc_img.content_hash);
                                        req_images.push_back(std::move(enc_img));
                                    } else if (part.contains("text") && part["text"].is_string()) {
                                        cm.content += part["text"].as_string();
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
                    prompt = tok.apply_chat_template(msgs, true, effort, tools_json);
                } else {
                    prompt = req["prompt"].as_string();
                }

                std::vector<int> prompt_ids = tok.encode(prompt, true);
                if (prompt_ids.empty()) {
                    send_json_response(fd, 400, "Bad Request", "{\"error\":\"empty prompt\"}");
                    close(fd);
                    continue;
                }

                // Append-only prefix reuse check (in-memory first including exact image hashes, then disk text prefix cache)
                bool can_reuse = !state_tokens.empty() &&
                                 prompt_ids.size() > state_tokens.size() &&
                                 std::equal(state_tokens.begin(), state_tokens.end(), prompt_ids.begin()) &&
                                 req_image_hashes.size() >= state_image_hashes.size() &&
                                 std::equal(state_image_hashes.begin(), state_image_hashes.end(), req_image_hashes.begin());
                int start_idx = 0;
                size_t reused_images = 0;
                double restore_ms = 0.0;
                if (can_reuse) {
                    start_idx = static_cast<int>(state_tokens.size());
                    reused_images = state_image_hashes.size();
                } else {
                    model.reset();
                    state_tokens.clear();
                    state_image_hashes.clear();
                    if (disk_cache.enabled()) {
                        auto tr0 = std::chrono::steady_clock::now();
                        std::vector<int> restored;
                        std::vector<uint64_t> restored_imgs;
                        int n_rest = disk_cache.try_restore(model, prompt_ids, restored,
                                                            req_image_hashes, &restored_imgs);
                        if (n_rest > 0) {
                            auto tr1 = std::chrono::steady_clock::now();
                            restore_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                            state_tokens = std::move(restored);
                            state_image_hashes = std::move(restored_imgs);
                            reused_images = state_image_hashes.size();
                            start_idx = n_rest;
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

                // Automatically checkpoint the reusable pure-text system+tools prefix ONCE so brand-new pi sessions
                // or rewound conversations (even multimodal ones!) restore system+tools in ~80 ms!
                int sys_ckpt_len = 0;
                if (disk_cache.enabled() && is_chat && prompt_ids.size() >= 192) {
                    const char* sys_end_tag = (model.info().arch == "inkling2") ? "<|end_message|>" : "<|im_end|>\n";
                    size_t sys_pos = prompt.find(sys_end_tag);
                    if (sys_pos != std::string::npos) {
                        std::string sys_prefix = prompt.substr(0, sys_pos + std::strlen(sys_end_tag));
                        std::vector<int> sys_ids = tok.encode(sys_prefix, true);
                        bool has_img_in_sys = false;
                        int img_tid = model.image_token_id();
                        if (img_tid >= 0) {
                            for (int tk : sys_ids) { if (tk == img_tid) { has_img_in_sys = true; break; } }
                        }
                        if (!has_img_in_sys && sys_ids.size() >= 128 && sys_ids.size() < prompt_ids.size() &&
                            static_cast<int>(sys_ids.size()) > start_idx &&
                            std::equal(sys_ids.begin(), sys_ids.end(), prompt_ids.begin())) {
                            sys_ckpt_len = static_cast<int>(sys_ids.size());
                        }
                    }
                }

                auto run_tail_forward = [&](const int* toks, int n, int* out_am, StepStats* st) -> const float* {
                    if (!new_images.empty()) {
                        return model.forward_multimodal(toks, n, new_images, out_am, st);
                    }
                    return model.forward(toks, n, out_am, st);
                };

                if (sys_ckpt_len > start_idx) {
                    StepStats s1{};
                    model.forward(prompt_ids.data() + start_idx, sys_ckpt_len - start_idx, nullptr, &s1);
                    std::vector<int> sys_toks(prompt_ids.begin(), prompt_ids.begin() + sys_ckpt_len);
                    disk_cache.save(model, sys_toks);
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
                state_tokens = prompt_ids;
                state_image_hashes = std::move(req_image_hashes);

                int next_tok = sampler.is_greedy() ? argmax_tok : sampler.sample(logits, model.info().vocab_size);
                auto t_first = std::chrono::steady_clock::now();
                double ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_req).count() + restore_ms + vis_ms;

                std::string req_id = "chatcmpl-" + std::to_string(sc.seed & 0xFFFFFFFFu);
                int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                if (stream) {
                    std::string sse_hdr =
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n\r\n";
                    send_str(fd, sse_hdr);
                }

                // Streaming & non-streaming state machine for reasoning_content + content + tool_calls!
                enum class OutMode { THINKING, CONTENT, TOOL_CALL };
                const bool is_inkling = (model.info().arch == "inkling2");
                OutMode mode = (is_chat && thinking_enabled) ? OutMode::THINKING : OutMode::CONTENT;
                std::string pending;
                std::string tool_buf;
                std::string utf8_reasoning_buf;
                std::string utf8_content_buf;
                std::string full_reasoning;
                std::string full_content;
                std::vector<ToolCall> emitted_tools;
                bool strip_post_think_nl = false;
                bool conn_ok = true;

                auto send_sse_delta = [&](Json::Object delta) -> bool {
                    if (!stream) return true;
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
                    chunk["model"] = model_id;
                    chunk["choices"] = std::move(choices);
                    if (!send_str(fd, "data: " + Json(std::move(chunk)).dump() + "\n\n")) {
                        conn_ok = false;
                        return false;
                    }
                    return true;
                };

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

                auto emit_content = [&](std::string s, bool flush_utf8 = false) {
                    if (!conn_ok) return;
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
                    if (utf8_content_buf.empty()) return;
                    std::string ready;
                    ready.reserve(utf8_content_buf.size());
                    drain_valid_utf8(utf8_content_buf, ready, flush_utf8);
                    if (ready.empty()) return;
                    full_content += ready;
                    Json::Object d;
                    d["content"] = ready;
                    send_sse_delta(std::move(d));
                };

                auto emit_tool_call = [&](const std::string& name, const std::string& args_json) {
                    if (!conn_ok) return;
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
                    while (!pending.empty() && conn_ok) {
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
                                    // Malformed tool block: emit verbatim as content so nothing is lost
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
                            if (flush_all && !tool_buf.empty()) {
                                std::string fn_name, args_json;
                                if (parse_qwen_tool_call(tool_buf, tool_schemas, fn_name, args_json)) {
                                    emit_tool_call(fn_name, args_json);
                                } else {
                                    emit_content("<tool_call>" + tool_buf);
                                }
                                tool_buf.clear();
                            }
                            break;
                        }
                    }
                };

                int gen_count = 0;
                int mtp_acc = 0, mtp_dft = 0, pld_hits = 0;
                std::string finish_reason = "length";
                auto t_dec0 = std::chrono::steady_clock::now();

                auto emit_tok = [&](int tok_id) -> bool {
                    if (is_eos(tok_id, model.info().eos_token_ids)) {
                        finish_reason = "stop";
                        return false;
                    }
                    ++gen_count;

                    if (is_inkling && is_chat) {
                        std::string raw_sp = tok.decode_token(tok_id, false);
                        if (raw_sp == "<|content_model_end_sampling|>") {
                            finish_reason = "stop";
                            return false;
                        }
                        if (raw_sp == "<|content_thinking|>") { emit_content("", true); mode = OutMode::THINKING; return gen_count < max_tokens; }
                        if (raw_sp == "<|content_text|>") { emit_reasoning("", true); mode = OutMode::CONTENT; return gen_count < max_tokens; }
                        if (raw_sp == "<|content_invoke_tool_json|>") {
                            emit_reasoning("", true);
                            emit_content("", true);
                            tool_buf.clear();
                            mode = OutMode::TOOL_CALL;
                            return gen_count < max_tokens;
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
                            return gen_count < max_tokens;
                        }
                        if (tok.is_special(tok_id)) return gen_count < max_tokens;
                        std::string piece = tok.decode_token(tok_id, true);
                        if (mode == OutMode::THINKING) emit_reasoning(piece);
                        else if (mode == OutMode::TOOL_CALL) tool_buf += piece;
                        else emit_content(piece);
                        return conn_ok && (gen_count < max_tokens);
                    }

                    // Keep structural XML special tokens (</think>, <tool_call>, </tool_call>) for Qwen's state machine!
                    std::string piece = tok.decode_token(tok_id, !is_chat);
                    if (!piece.empty()) {
                        if (!is_chat) {
                            emit_content(piece);
                        } else {
                            // Ignore stray ChatML control tokens if any appear
                            if (piece == "<|im_start|>" || piece == "<|im_end|>" || piece == "<|endoftext|>") {
                                return conn_ok && (gen_count < max_tokens);
                            }
                            pending += piece;
                            pump_qwen_pending(false);
                        }
                    }
                    return conn_ok && (gen_count < max_tokens);
                };

                bool keep_going = emit_tok(next_tok);
                std::vector<int> mtp_batch;
                while (keep_going) {
                    sampler.observe(next_tok);
                    if (sampler.is_greedy() && model.has_mtp()) {
                        mtp_batch.clear();
                        StepStats ms{};
                        model.forward_mtp_greedy(next_tok, mtp_batch, &ms);
                        mtp_acc += ms.mtp_accepted;
                        mtp_dft += ms.mtp_drafted;
                        pld_hits += ms.pld_hits;
                        state_tokens.push_back(next_tok);
                        for (size_t j = 0; j + 1 < mtp_batch.size(); ++j) state_tokens.push_back(mtp_batch[j]);
                        for (size_t j = 0; j < mtp_batch.size(); ++j) {
                            int t = mtp_batch[j];
                            if (j + 1 < mtp_batch.size()) sampler.observe(t);
                            keep_going = emit_tok(t);
                            if (!keep_going) break;
                        }
                        next_tok = mtp_batch.back();
                    } else if (!sampler.is_greedy() && model.supports_mtp_sampled()) {
                        mtp_batch.clear();
                        StepStats ms{};
                        model.forward_mtp_sampled(next_tok, sampler, mtp_batch, &ms);
                        mtp_acc += ms.mtp_accepted;
                        mtp_dft += ms.mtp_drafted;
                        pld_hits += ms.pld_hits;
                        state_tokens.push_back(next_tok);
                        for (size_t j = 0; j + 1 < mtp_batch.size(); ++j) state_tokens.push_back(mtp_batch[j]);
                        for (size_t j = 0; j < mtp_batch.size(); ++j) {
                            keep_going = emit_tok(mtp_batch[j]);
                            if (!keep_going) break;
                        }
                        next_tok = mtp_batch.back();
                    } else if (sampler.is_greedy()) {
                        state_tokens.push_back(next_tok);
                        next_tok = model.forward_greedy(next_tok);
                        keep_going = emit_tok(next_tok);
                    } else {
                        state_tokens.push_back(next_tok);
                        const float* lg = model.forward(&next_tok, 1);
                        next_tok = sampler.sample(lg, model.info().vocab_size);
                        keep_going = emit_tok(next_tok);
                    }
                }

                if (is_chat && !is_inkling) pump_qwen_pending(true);
                emit_reasoning("", true);
                emit_content("", true);
                if (!emitted_tools.empty() && finish_reason == "stop") {
                    finish_reason = "tool_calls";
                }

                auto t_dec1 = std::chrono::steady_clock::now();
                double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
                double dec_tps = (gen_count > 1 && dec_s > 0.0) ? ((gen_count - 1) / dec_s) : 0.0;
                std::fprintf(stderr, "[req] prompt=%zu (cached=%d, rest=%.1fms, vis=%.1fms, imgs=%zu) gen=%d tools=%zu ttft=%.1fms decode=%.1f tok/s",
                             prompt_ids.size(), start_idx, restore_ms, vis_ms, state_image_hashes.size(),
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
                    Json::Object ch;
                    ch["index"] = 0;
                    ch["delta"] = Json::Object{};
                    ch["finish_reason"] = finish_reason;
                    Json::Array choices;
                    choices.push_back(Json(std::move(ch)));
                    Json::Object end_chunk;
                    end_chunk["id"] = req_id;
                    end_chunk["object"] = is_chat ? "chat.completion.chunk" : "text_completion";
                    end_chunk["created"] = created;
                    end_chunk["model"] = model_id;
                    end_chunk["choices"] = std::move(choices);
                    if (include_usage) end_chunk["usage"] = std::move(usage);
                    send_str(fd, "data: " + Json(std::move(end_chunk)).dump() + "\n\n");
                    send_str(fd, "data: [DONE]\n\n");
                } else {
                    Json::Object ch;
                    ch["index"] = 0;
                    ch["finish_reason"] = finish_reason;
                    if (is_chat) {
                        Json::Object msg;
                        msg["role"] = "assistant";
                        msg["content"] = full_content.empty() && !emitted_tools.empty() ? Json(nullptr) : Json(full_content);
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
                    resp["model"] = model_id;
                    resp["choices"] = std::move(choices);
                    resp["usage"] = std::move(usage);
                    send_json_response(fd, 200, "OK", Json(std::move(resp)).dump());
                }
            } catch (const std::exception& ex) {
                Json::Object err;
                err["error"] = std::string(ex.what());
                send_json_response(fd, 500, "Internal Server Error", Json(std::move(err)).dump());
            }
            close(fd);
            // Round 11 + 32: Persist full text OR multimodal conversation checkpoint AFTER closing client socket!
            if (disk_cache.enabled() && state_tokens.size() >= 256 &&
                static_cast<int>(state_tokens.size()) == model.position()) {
                disk_cache.save(model, state_tokens, state_image_hashes);
            }
            continue;
        }

        send_json_response(fd, 404, "Not Found", "{\"error\":\"not found\"}");
        close(fd);
    }
}

} // namespace mi
