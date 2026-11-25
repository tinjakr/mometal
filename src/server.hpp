#pragma once

#include "model.hpp"
#include "tokenizer.hpp"

#include <string>

namespace mi {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8089;
    std::string model_id;
    std::string cache_dir;
};

void run_http_server(Model& model, const Tokenizer& tok, const ServerConfig& cfg);

} // namespace mi
