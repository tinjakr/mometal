#import <Foundation/Foundation.h>

#include "ane_builder.hpp"
#include "inkling2.hpp"
#include "json.hpp"
#include "metal_ctx.hpp"
#include "prefix_cache.hpp"
#include "qwen38.hpp"
#include "safetensors.hpp"
#include "sampler.hpp"
#include "server.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <malloc/malloc.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mi;

static void print_usage() {
    std::fprintf(stderr,
        "mometal - Zero-overhead C++20/Metal/ANE inference engine for Apple Silicon\n\n"
        "Usage:\n"
        "  mometal run       --model <dir> --prompt <text> [options]\n"
        "  mometal chat      --model <dir> [options]\n"
        "  mometal bench     --model <dir> [--prompt-tokens N] [--gen-tokens N] [options]\n"
        "  mometal verify    --model <dir> --tokens 1,2,3 [--top 10]\n"
        "  mometal serve     --model <dir> [--host 127.0.0.1] [--port 8089] [options]\n"
        "  mometal build-ane --model <dir> [--out <dir>] [--format auto|hf4|lut4|nf4] [--layers all|0-63] [--workers N]\n\n"
        "Options:\n"
        "  --max-tokens N       Max tokens to generate (default: 256)\n"
        "  --max-seq N          KV cache capacity for full-attention layers (default: 8192)\n"
        "  --kv-bits N          Full-attention KV cache precision: 16|8|4 (default: 8)\n"
        "  --sparse-kv-budget N Optional Quest query-aware block-sparse decode budget (0 = exact dense, e.g. 8192 or 12288)\n"
        "  --temp T             Sampling temperature, 0 = greedy (default: 0.0)\n"
        "  --top-k K            Top-k cutoff (default: 20)\n"
        "  --top-p P            Nucleus top-p (default: 0.95)\n"
        "  --presence-penalty P Presence penalty (default: 0.0)\n"
        "  --effort LEVEL       Reasoning effort: none|minimal|low|medium|high|xhigh (default: none)\n"
        "  --system TEXT        System prompt\n"
        "  --raw                Skip chat template, tokenize prompt directly\n"
        "  --mtp MODE           MTP speculative decoding: auto|off|<path> (default: auto)\n"
        "  --mtp-depth N        Number of speculative draft tokens per round (default: auto [5 Qwen, 3 Inkling])\n"
        "  --dflash DIR         DFlash 2 block-diffusion speculative drafter directory (e.g. Qwen3.8-27B-DFlash2)\n"
        "  --no-pld             Disable 4-gram prompt-lookup copy drafting\n"
        "  --ane-models DIR     Single-copy ANE-Share MLP directory: auto|off|<dir> (default: auto; falls back to GPU-only)\n"
        "  --ane-tag TAG        ANE model tag (e.g. hf4, lut4, nf4; default: auto per chip gen)\n"
        "  --ane-share          Enable single-copy ANE-Share (auto-builds .mlmodelc natively in C++ if missing!)\n"
        "  --cache-dir DIR      Enable disk-backed prefix/state cache in DIR\n"
        "  --no-nvfp4-act       Inkling: skip NVFP4 activation rounding (faster, slight diff)\n"
    );
}

static std::unique_ptr<Model> load_model(
    MetalContext& ctx, const std::string& dir, int max_seq, bool exact_nvfp4_act,
    const std::string& mtp_dir, int mtp_depth, bool enable_pld,
    const std::string& ane_models_dir, const std::string& ane_tag, int kv_bits, bool kv_bits_explicit,
    int ane_share_layers, int sparse_kv_budget = 0, const std::string& dflash_dir = ""
) {
    Json cfg = parse_json(read_file_to_string(dir + "/config.json"));
    std::string mtype = cfg["model_type"].as_string();
    if (mtype == "qwen3_5" || mtype == "qwen3_5_text") {
        int q_depth = (mtp_depth > 0) ? mtp_depth : 5;
        return std::make_unique<Qwen38Model>(ctx, dir, max_seq, mtp_dir, q_depth, enable_pld, ane_models_dir, ane_tag, kv_bits, ane_share_layers, sparse_kv_budget, dflash_dir);
    }
    if (mtype == "inkling2_loop" || mtype == "inkling2_loop_text") {
        std::vector<std::string> ignored;
        if (kv_bits_explicit && kv_bits != 16) ignored.push_back("--kv-bits");
        if (!dflash_dir.empty() && dflash_dir != "off" && dflash_dir != "none") ignored.push_back("--dflash");
        if (ane_share_layers > 0) ignored.push_back("--ane-share");
        if (ane_models_dir != "auto" && ane_models_dir != "off" && ane_models_dir != "none") ignored.push_back("--ane-models");
        if (sparse_kv_budget > 0) ignored.push_back("--sparse-kv-budget");
        if (!ignored.empty()) {
            std::fprintf(stderr, "[mometal] Warning: Inkling2 ignores Qwen-only options:");
            for (const auto& f : ignored) std::fprintf(stderr, " %s", f.c_str());
            std::fprintf(stderr, "\n");
        }
        int i_depth = (mtp_depth > 0) ? mtp_depth : 3;
        return std::make_unique<Inkling2Model>(ctx, dir, max_seq, exact_nvfp4_act, mtp_dir, i_depth);
    }
    throw std::runtime_error("Unsupported model_type in config.json: " + mtype);
}

static bool is_eos_token(int tok, const std::vector<int>& eos_ids) {
    for (int e : eos_ids) if (tok == e) return true;
    return false;
}

static std::vector<int> parse_csv_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        print_usage();
        return 0;
    }

    std::string model_dir;
    std::string prompt;
    std::string system_prompt;
    std::string effort = "none";
    std::string host = "127.0.0.1";
    std::string tokens_csv;
    std::string tokens_file;
    std::string save_ref_path;
    std::string eval_ref_path;
    std::string mtp_dir = "auto";
    std::string dflash_dir;
    std::string ane_models_dir = "auto";
    std::string ane_tag = "auto";
    std::string ane_out_dir;
    std::string ane_format = "auto";
    std::string ane_layers_spec = "all";
    float ane_alpha = 0.85f;
    int ane_workers = 0; // 0 = auto min(12, hardware_concurrency)
    bool ane_mse_clip = true;
    std::string cache_dir;
    std::vector<std::string> suffix_dirs;
    int mtp_depth = 0; // 0 = auto per arch (5 Qwen, 3 Inkling)
    int port = 8089;
    int max_tokens = 256;
    int max_seq = 8192;
    int kv_bits = 8;
    bool kv_bits_explicit = false;
    int ane_share_layers = 0;
    int sparse_kv_budget = 0;
    int prompt_tokens_bench = 128;
    int gen_tokens_bench = 64;
    int top_verify = 10;
    int eval_prefix_len = 256;
    bool raw_prompt = false;
    bool exact_nvfp4_act = true;
    bool enable_pld = true;
    SamplerConfig sc;

    try {
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
                return argv[++i];
            };
            if (a == "--model" || a == "-m") model_dir = next();
            else if (a == "--prompt" || a == "-p") prompt = next();
            else if (a == "--system") system_prompt = next();
            else if (a == "--max-tokens" || a == "-n") max_tokens = std::stoi(next());
            else if (a == "--max-seq") max_seq = std::stoi(next());
            else if (a == "--kv-bits") { kv_bits = std::stoi(next()); kv_bits_explicit = true; }
            else if (a == "--temp") sc.temperature = std::stof(next());
            else if (a == "--top-k") sc.top_k = std::stoi(next());
            else if (a == "--top-p") sc.top_p = std::stof(next());
            else if (a == "--presence-penalty") sc.presence_penalty = std::stof(next());
            else if (a == "--seed") sc.seed = std::stoull(next());
            else if (a == "--effort") effort = next();
            else if (a == "--host") host = next();
            else if (a == "--port") port = std::stoi(next());
            else if (a == "--tokens") tokens_csv = next();
            else if (a == "--tokens-file") tokens_file = next();
            else if (a == "--prefix-len") eval_prefix_len = std::stoi(next());
            else if (a == "--save-ref") save_ref_path = next();
            else if (a == "--eval-ref") eval_ref_path = next();
            else if (a == "--top") top_verify = std::stoi(next());
            else if (a == "--prompt-tokens") prompt_tokens_bench = std::stoi(next());
            else if (a == "--gen-tokens") gen_tokens_bench = std::stoi(next());
            else if (a == "--mtp") mtp_dir = next();
            else if (a == "--mtp-depth") mtp_depth = std::stoi(next());
            else if (a == "--dflash") dflash_dir = next();
            else if (a == "--no-pld") enable_pld = false;
            else if (a == "--ane-models") ane_models_dir = next();
            else if (a == "--ane-tag") ane_tag = next();
            else if (a == "--out") ane_out_dir = next();
            else if (a == "--format") ane_format = next();
            else if (a == "--layers") ane_layers_spec = next();
            else if (a == "--workers") ane_workers = std::stoi(next());
            else if (a == "--alpha") ane_alpha = std::stof(next());
            else if (a == "--no-mse-clip") ane_mse_clip = false;
            else if (a == "--ane-share") ane_share_layers = 64;
            else if (a == "--ane-share-layers") {
                (void)next();
                std::fprintf(stderr, "[mometal] Note: --ane-share-layers is deprecated; ANE uses strict all-layer single-copy sharing (else GPU-only fallback).\n");
            }
            else if (a == "--sparse-kv-budget") sparse_kv_budget = std::stoi(next());
            else if (a == "--cache-dir") cache_dir = next();
            else if (a == "--suffix-dir") suffix_dirs.push_back(next());
            else if (a == "--raw") raw_prompt = true;
            else if (a == "--no-nvfp4-act") exact_nvfp4_act = false;
            else {
                std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
                return 1;
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Argument error: %s\n", ex.what());
        return 1;
    }

    if (model_dir.empty()) {
        std::fprintf(stderr, "Error: --model <dir> is required.\n");
        return 1;
    }
    if (max_seq <= 0) {
        std::fprintf(stderr, "Error: --max-seq must be > 0 (got %d).\n", max_seq);
        return 1;
    }
    if (max_tokens < 0) {
        std::fprintf(stderr, "Error: --max-tokens must be >= 0 (got %d).\n", max_tokens);
        return 1;
    }
    if (kv_bits != 16 && kv_bits != 8 && kv_bits != 4) {
        std::fprintf(stderr, "Error: --kv-bits must be 16, 8, or 4 (got %d).\n", kv_bits);
        return 1;
    }
    if (port < 1 || port > 65535) {
        std::fprintf(stderr, "Error: --port must be in 1..65535 (got %d).\n", port);
        return 1;
    }
    if (top_verify <= 0) {
        std::fprintf(stderr, "Error: --top must be > 0 (got %d).\n", top_verify);
        return 1;
    }
    if (prompt_tokens_bench <= 0 || gen_tokens_bench <= 0) {
        std::fprintf(stderr, "Error: --prompt-tokens and --gen-tokens must be > 0.\n");
        return 1;
    }
    if (eval_prefix_len <= 0) {
        std::fprintf(stderr, "Error: --prefix-len must be > 0 (got %d).\n", eval_prefix_len);
        return 1;
    }
    if (ane_workers < 0) {
        std::fprintf(stderr, "Error: --workers must be >= 0 (got %d).\n", ane_workers);
        return 1;
    }
    if (sparse_kv_budget < 0) {
        std::fprintf(stderr, "Error: --sparse-kv-budget must be >= 0 (got %d).\n", sparse_kv_budget);
        return 1;
    }

    if (cmd == "build-ane") {
        @autoreleasepool {
            AneBuildOptions bopt;
            bopt.model_dir = model_dir;
            bopt.out_dir = ane_out_dir;
            bopt.format = ane_format;
            bopt.tag = ane_tag;
            bopt.layers_spec = ane_layers_spec;
            bopt.alpha = ane_alpha;
            bopt.mse_clip = ane_mse_clip;
            bopt.workers = ane_workers;
            AneBuildResult r = build_ane_mlps_native(bopt);
            if (!r.ok) {
                std::fprintf(stderr, "Error: %s\n", r.error.c_str());
                return 1;
            }
            return 0;
        }
    }

    try {
        // Bounded @autoreleasepool for model initialization so all transient shader/load Obj-C objects drain before serving!
        std::unique_ptr<MetalContext> ctx_ptr;
        std::unique_ptr<Model> model;
        @autoreleasepool {
            ctx_ptr = std::make_unique<MetalContext>();
            model = load_model(*ctx_ptr, model_dir, max_seq, exact_nvfp4_act, mtp_dir, mtp_depth, enable_pld,
                               ane_models_dir, ane_tag, kv_bits, kv_bits_explicit, ane_share_layers, sparse_kv_budget, dflash_dir);
        }
        malloc_zone_pressure_relief(nullptr, 0);
        MetalContext& ctx = *ctx_ptr;
        const ModelInfo& info = model->info();

        std::fprintf(stderr,
            "[mometal] GPU: %s (Apple%d | ~%d cores | ~%.0f GB/s | max working set %.1f GB)\n"
            "[mometal] Model: %s | %s | layers=%d visits=%d | weights=%.2f GB (mmap) | cache=%.2f GB | load=%.1f ms\n",
            ctx.device_name().c_str(),
            static_cast<int>(ctx.gpu_tier()),
            ctx.estimated_gpu_cores(),
            ctx.estimated_bandwidth_gbs(),
            ctx.recommended_max_working_set_size() / (1024.0 * 1024.0 * 1024.0),
            info.arch.c_str(), info.quant_desc.c_str(),
            info.num_layers, info.num_visits,
            info.mapped_weight_bytes / (1024.0 * 1024.0 * 1024.0),
            info.kv_cache_bytes / (1024.0 * 1024.0 * 1024.0),
            info.load_time_ms);

        if (cmd == "verify") {
            @autoreleasepool {
                std::vector<int> ids = parse_csv_ints(tokens_csv);
                if (ids.empty()) {
                    std::fprintf(stderr, "Error: --tokens 1,2,3 required for verify\n");
                    return 1;
                }
                for (int id : ids) {
                    if (id < 0 || id >= info.vocab_size) {
                        std::fprintf(stderr, "Error: token ID %d out of range [0, %d)\n", id, info.vocab_size);
                        return 1;
                    }
                }
                StepStats st{};
                int gpu_argmax = -1;
                const float* logits = model->forward(ids.data(), static_cast<int>(ids.size()), &gpu_argmax, &st);
                int V = info.vocab_size;

                std::vector<std::pair<float, int>> top;
                top.reserve(V);
                for (int i = 0; i < V; ++i) top.emplace_back(logits[i], i);
                int k = std::min(top_verify, V);
                std::partial_sort(top.begin(), top.begin() + k, top.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });

                std::printf("{\"arch\":\"%s\",\"gpu_ms\":%.3f,\"gpu_argmax\":%d,\"top\":[",
                            info.arch.c_str(), st.gpu_ms, gpu_argmax);
                for (int i = 0; i < k; ++i) {
                    if (i) std::printf(",");
                    std::printf("[%d,%.6f]", top[i].second, top[i].first);
                }
                std::printf("]}\n");
                return 0;
            }
        }

        if (cmd == "eval-ppl") {
            @autoreleasepool {
                if (tokens_file.empty()) {
                    std::fprintf(stderr, "Error: --tokens-file <path> required for eval-ppl\n");
                    return 1;
                }
                std::ifstream tf(tokens_file);
                if (!tf) throw std::runtime_error("Cannot open tokens file: " + tokens_file);
                std::vector<std::vector<int>> passages;
                std::string line;
                while (std::getline(tf, line)) {
                    auto ids = parse_csv_ints(line);
                    for (int id : ids) {
                        if (id < 0 || id >= info.vocab_size) {
                            throw std::runtime_error("Token ID " + std::to_string(id) +
                                                     " out of range [0, " + std::to_string(info.vocab_size) + ") in " + tokens_file);
                        }
                    }
                    if (static_cast<int>(ids.size()) > eval_prefix_len + 1) passages.push_back(std::move(ids));
                }
                constexpr int TOPK = 256;
                struct RefEntry {
                    int32_t top5[5];
                    float p_tail;
                    int32_t ids[TOPK];
                    float probs[TOPK];
                };
                std::ofstream ref_out;
                if (!save_ref_path.empty()) {
                    ref_out.open(save_ref_path, std::ios::binary);
                    if (!ref_out) throw std::runtime_error("Cannot open --save-ref: " + save_ref_path);
                }
                std::ifstream ref_in;
                if (!eval_ref_path.empty()) {
                    ref_in.open(eval_ref_path, std::ios::binary);
                    if (!ref_in) throw std::runtime_error("Cannot open --eval-ref: " + eval_ref_path);
                }

                const int V = info.vocab_size;
                std::vector<std::pair<float, int>> scratch(V);
                double total_nll = 0.0;
                double total_kl = 0.0;
                std::vector<double> kl_vals;
                int64_t total_scored = 0;
                int64_t top1_match = 0;
                int64_t top5_match = 0;
                int64_t argmax_in_ref5 = 0;

                auto score_logits = [&](const float* logits, int target_tok) {
                    float max_l = logits[0];
                    int argmax_id = 0;
                    for (int v = 1; v < V; ++v) {
                        if (logits[v] > max_l) { max_l = logits[v]; argmax_id = v; }
                    }
                    double sum_exp = 0.0;
                    for (int v = 0; v < V; ++v) sum_exp += std::exp(static_cast<double>(logits[v] - max_l));
                    double lse = static_cast<double>(max_l) + std::log(sum_exp);
                    double nll = lse - static_cast<double>(logits[target_tok]);
                    total_nll += nll;
                    ++total_scored;

                    if (ref_out.is_open()) {
                        for (int v = 0; v < V; ++v) scratch[v] = {logits[v], v};
                        std::partial_sort(scratch.begin(), scratch.begin() + TOPK, scratch.end(),
                                          [](const auto& a, const auto& b) { return a.first > b.first; });
                        RefEntry e{};
                        for (int i = 0; i < 5; ++i) e.top5[i] = scratch[i].second;
                        double p_sum = 0.0;
                        for (int i = 0; i < TOPK; ++i) {
                            e.ids[i] = scratch[i].second;
                            double p = std::exp(static_cast<double>(scratch[i].first) - lse);
                            e.probs[i] = static_cast<float>(p);
                            p_sum += p;
                        }
                        e.p_tail = static_cast<float>(std::max(1e-12, 1.0 - p_sum));
                        ref_out.write(reinterpret_cast<const char*>(&e), sizeof(e));
                    }
                    if (ref_in.is_open()) {
                        RefEntry r{};
                        if (!ref_in.read(reinterpret_cast<char*>(&r), sizeof(r))) {
                            throw std::runtime_error("Ref file ended early");
                        }
                        for (int i = 0; i < 5; ++i) {
                            if (r.top5[i] < 0 || r.top5[i] >= V) {
                                throw std::runtime_error("Ref file top5 token ID out of range for model vocab_size");
                            }
                        }
                        for (int i = 0; i < TOPK; ++i) {
                            if (r.ids[i] < 0 || r.ids[i] >= V) {
                                throw std::runtime_error("Ref file candidate token ID out of range for model vocab_size");
                            }
                        }
                        if (argmax_id == r.top5[0]) ++top1_match;
                        for (int i = 0; i < 5; ++i) {
                            if (argmax_id == r.top5[i]) { ++argmax_in_ref5; break; }
                        }
                        int better = 0;
                        float r0_l = logits[r.top5[0]];
                        for (int v = 0; v < V; ++v) if (logits[v] > r0_l) { if (++better >= 5) break; }
                        if (better < 5) ++top5_match;

                        double kl = 0.0;
                        double q_sum = 0.0;
                        for (int i = 0; i < TOPK; ++i) {
                            double p = static_cast<double>(r.probs[i]);
                            double log_q = static_cast<double>(logits[r.ids[i]]) - lse;
                            q_sum += std::exp(log_q);
                            if (p > 1e-12) kl += p * (std::log(p) - log_q);
                        }
                        double q_tail = std::max(1e-12, 1.0 - q_sum);
                        double p_tail = static_cast<double>(r.p_tail);
                        if (p_tail > 1e-12) kl += p_tail * (std::log(p_tail) - std::log(q_tail));
                        kl = std::max(0.0, kl);
                        total_kl += kl;
                        kl_vals.push_back(kl);
                    }
                };

                auto t0 = std::chrono::steady_clock::now();
                for (size_t pi = 0; pi < passages.size(); ++pi) {
                    @autoreleasepool {
                        const auto& ids = passages[pi];
                        model->reset();
                        const float* lg = model->forward(ids.data(), eval_prefix_len);
                        score_logits(lg, ids[eval_prefix_len]);
                        for (size_t pos = static_cast<size_t>(eval_prefix_len); pos + 1 < ids.size(); ++pos) {
                            int cur = ids[pos];
                            lg = model->forward(&cur, 1);
                            score_logits(lg, ids[pos + 1]);
                        }
                        std::fprintf(stderr, "[eval-ppl] passage %zu/%zu done | running ppl=%.4f\n",
                                     pi + 1, passages.size(), std::exp(total_nll / std::max<int64_t>(1, total_scored)));
                    }
                }
                auto t1 = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(t1 - t0).count();
                double ppl = std::exp(total_nll / std::max<int64_t>(1, total_scored));
                double mean_kl = kl_vals.empty() ? 0.0 : (total_kl / kl_vals.size());
                double med_kl = 0.0;
                if (!kl_vals.empty()) {
                    std::sort(kl_vals.begin(), kl_vals.end());
                    med_kl = kl_vals[kl_vals.size() / 2];
                }
                std::printf(
                    "{\"passages\":%zu,\"scored_tokens\":%lld,\"mean_nll\":%.6f,\"ppl\":%.4f,\"elapsed_s\":%.2f",
                    passages.size(), static_cast<long long>(total_scored),
                    total_nll / std::max<int64_t>(1, total_scored), ppl, dt);
                if (!kl_vals.empty()) {
                    std::printf(
                        ",\"mean_kl\":%.6f,\"median_kl\":%.6f,\"top1_agree_pct\":%.2f,\"ref_top1_in_top5_pct\":%.2f,\"argmax_in_ref_top5_pct\":%.2f",
                        mean_kl, med_kl,
                        100.0 * top1_match / total_scored,
                        100.0 * top5_match / total_scored,
                        100.0 * argmax_in_ref5 / total_scored);
                }
                std::printf("}\n");
                return 0;
            }
        }

        if (cmd == "profile") {
            @autoreleasepool {
                model->profile_ops();
                return 0;
            }
        }

        if (cmd == "bench") {
            @autoreleasepool {
                std::vector<int> fake_prompt(static_cast<size_t>(prompt_tokens_bench), 100);
                model->forward_greedy(100);
                model->reset();

                StepStats pf{};
                int tok_id = 0;
                model->forward(fake_prompt.data(), static_cast<int>(fake_prompt.size()), &tok_id, &pf);
                double pf_tps = fake_prompt.size() / (pf.gpu_ms / 1000.0);

                auto t0 = std::chrono::steady_clock::now();
                double min_ms = 1e9, max_ms = 0.0;
                int emitted = 0, rounds = 0, total_acc = 0, total_dft = 0;
                if (model->has_mtp()) {
                    std::vector<int> batch;
                    while (emitted < gen_tokens_bench) {
                        batch.clear();
                        StepStats ds{};
                        model->forward_mtp_greedy(tok_id, batch, &ds);
                        tok_id = batch.back();
                        emitted += static_cast<int>(batch.size());
                        total_acc += ds.mtp_accepted;
                        total_dft += ds.mtp_drafted;
                        ++rounds;
                        min_ms = std::min(min_ms, ds.gpu_ms);
                        max_ms = std::max(max_ms, ds.gpu_ms);
                    }
                } else {
                    for (int i = 0; i < gen_tokens_bench; ++i) {
                        StepStats ds{};
                        tok_id = model->forward_greedy(tok_id, &ds);
                        ++emitted;
                        ++rounds;
                        min_ms = std::min(min_ms, ds.gpu_ms);
                        max_ms = std::max(max_ms, ds.gpu_ms);
                    }
                }
                auto t1 = std::chrono::steady_clock::now();
                double dec_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double dec_tps = emitted / (dec_ms / 1000.0);

                std::printf(
                    "Benchmark (%s, %s)\n"
                    "  Memory:  weights=%.2f GB (mmap) | cache+MTP=%.2f GB\n"
                    "  Load:    %.1f ms\n"
                    "  Prefill: %d tok in %.1f ms (%.1f tok/s)\n"
                    "  Decode:  %d tok in %.1f ms (%.2f tok/s | avg %.2f ms/tok | min %.2f ms/rnd | max %.2f ms/rnd)\n",
                    info.arch.c_str(), info.quant_desc.c_str(),
                    info.mapped_weight_bytes / (1024.0 * 1024.0 * 1024.0),
                    info.kv_cache_bytes / (1024.0 * 1024.0 * 1024.0),
                    info.load_time_ms,
                    prompt_tokens_bench, pf.gpu_ms, pf_tps,
                    emitted, dec_ms, dec_tps, dec_ms / emitted, min_ms, max_ms
                );
                if (total_dft > 0) {
                    std::printf(
                        "  MTP:     accepted=%d/%d (%.1f%%) | %.2f tok/round (%d rounds)\n",
                        total_acc, total_dft, 100.0 * total_acc / total_dft,
                        static_cast<double>(emitted) / rounds, rounds
                    );
                }

                if (!cache_dir.empty() && prompt_tokens_bench >= 2) {
                    PrefixCacheManager pcm(cache_dir);
                    model->reset();
                    std::vector<int> prefix(fake_prompt.begin(), fake_prompt.end() - 1);
                    model->forward(prefix.data(), static_cast<int>(prefix.size()));
                    auto ts0 = std::chrono::steady_clock::now();
                    pcm.save(*model, prefix);
                    auto ts1 = std::chrono::steady_clock::now();
                    double save_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

                    model->reset();
                    std::vector<int> restored;
                    auto tr0 = std::chrono::steady_clock::now();
                    int n_rest = pcm.try_restore(*model, fake_prompt, restored);
                    auto tr1 = std::chrono::steady_clock::now();
                    double rest_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                    StepStats tail_st{};
                    int tail_tok = 0;
                    model->forward(fake_prompt.data() + n_rest, static_cast<int>(fake_prompt.size()) - n_rest, &tail_tok, &tail_st);
                    std::printf(
                        "  Cache:   saved %zu tok in %.1f ms | restored %d tok in %.1f ms + 1-tok %.1f ms = %.1f ms cached TTFT\n",
                        prefix.size(), save_ms, n_rest, rest_ms, tail_st.gpu_ms, rest_ms + tail_st.gpu_ms
                    );
                }
                return 0;
            }
        }

        // Commands requiring tokenizer
        Tokenizer tok;
        @autoreleasepool {
            auto tok_t0 = std::chrono::steady_clock::now();
            tok.load(model_dir, info.arch);
            auto tok_t1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[mometal] Tokenizer loaded (%d tokens) in %.1f ms\n",
                         tok.vocab_size(), std::chrono::duration<double, std::milli>(tok_t1 - tok_t0).count());

            if (!suffix_dirs.empty()) {
                auto st0 = std::chrono::steady_clock::now();
                size_t n_files = 0, n_toks = 0;
                auto is_code_ext = [](const std::string& ext) -> bool {
                    static const char* exts[] = {
                        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".mm", ".m", ".metal",
                        ".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".sh", ".zsh",
                        ".md", ".json", ".toml", ".yaml", ".yml", ".cmake", ".txt", ".swift", ".zig"
                    };
                    for (const char* e : exts) if (ext == e) return true;
                    return false;
                };
                namespace fs = std::filesystem;
                for (const auto& dir : suffix_dirs) {
                    std::error_code ec;
                    if (!fs::exists(dir, ec)) continue;
                    for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
                         it != fs::recursive_directory_iterator(); it.increment(ec)) {
                        if (ec) { ec.clear(); continue; }
                        const auto& p = it->path();
                        std::string fname = p.filename().string();
                        if (it->is_directory(ec)) {
                            if (fname == ".git" || fname == "build" || fname == "node_modules" ||
                                fname == ".venv" || fname == "__pycache__" || fname == "target" || fname == "dist") {
                                it.disable_recursion_pending();
                            }
                            continue;
                        }
                        if (!it->is_regular_file(ec)) continue;
                        auto fsz = it->file_size(ec);
                        if (ec || fsz == 0 || fsz > 262144u) continue;
                        if (!is_code_ext(p.extension().string()) && fname != "CMakeLists.txt" && fname != "Makefile") continue;
                        std::ifstream ifs(p, std::ios::binary);
                        if (!ifs) continue;
                        std::string content(fsz, '\0');
                        ifs.read(content.data(), static_cast<std::streamsize>(fsz));
                        auto ids = tok.encode(content, false);
                        if (ids.size() >= 4) {
                            model->add_suffix_corpus(ids.data(), static_cast<int>(ids.size()));
                            ++n_files;
                            n_toks += ids.size();
                            if (n_toks >= 2000000u) break;
                        }
                    }
                }
                double s_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st0).count();
                std::fprintf(stderr, "[mometal] Indexed suffix corpus: %zu files, %zu tokens in %.1f ms\n",
                             n_files, n_toks, s_ms);
            }
        }
        malloc_zone_pressure_relief(nullptr, 0);

        if (cmd == "serve") {
            ServerConfig srv_cfg;
            srv_cfg.host = host;
            srv_cfg.port = port;
            srv_cfg.model_id = model_dir;
            srv_cfg.cache_dir = cache_dir;
            run_http_server(*model, tok, srv_cfg);
            return 0;
        }

        if (cmd == "run") {
            @autoreleasepool {
                std::vector<int> ids;
                if (!raw_prompt) {
                    std::vector<ChatMessage> msgs;
                    if (!system_prompt.empty()) msgs.push_back(ChatMessage{"system", system_prompt, {}, {}, {}, {}, {}});
                    msgs.push_back(ChatMessage{"user", prompt, {}, {}, {}, {}, {}});
                    ids = tok.encode_chat_template(msgs, true, effort);
                } else {
                    ids = tok.encode(prompt, true);
                }
                if (ids.empty()) {
                    std::fprintf(stderr, "Empty prompt after tokenization.\n");
                    return 1;
                }

                PrefixCacheManager disk_cache(cache_dir);
                int start_idx = 0;
                double restore_ms = 0.0;
                if (disk_cache.enabled()) {
                    auto tr0 = std::chrono::steady_clock::now();
                    std::vector<int> restored;
                    start_idx = disk_cache.try_restore(*model, ids, restored);
                    auto tr1 = std::chrono::steady_clock::now();
                    if (start_idx > 0) restore_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
                }

                Sampler sampler(sc);
                for (int id : ids) sampler.observe(id);

                StepStats pf{};
                int argmax_tok = 0;
                const float* logits = nullptr;
                if (disk_cache.enabled() && ids.size() >= 32 && start_idx < static_cast<int>(ids.size()) - 1) {
                    int save_len = static_cast<int>(ids.size()) - 1;
                    model->forward(ids.data() + start_idx, save_len - start_idx, nullptr, &pf);
                    std::vector<int> prefix_toks(ids.begin(), ids.begin() + save_len);
                    disk_cache.save(*model, prefix_toks);
                    StepStats tail{};
                    logits = model->forward(ids.data() + save_len, 1, sampler.is_greedy() ? &argmax_tok : nullptr, &tail);
                    pf.gpu_ms += tail.gpu_ms;
                } else {
                    logits = model->forward(
                        ids.data() + start_idx, static_cast<int>(ids.size()) - start_idx,
                        sampler.is_greedy() ? &argmax_tok : nullptr, &pf
                    );
                }
                int next_tok = sampler.is_greedy() ? argmax_tok : sampler.sample(logits, info.vocab_size);

                int generated = 0;
                int mtp_acc = 0, mtp_dft = 0, mtp_rounds = 0, pld_hits = 0;
                auto t_dec0 = std::chrono::steady_clock::now();

                auto emit_tok = [&](int t) -> bool {
                    if (is_eos_token(t, info.eos_token_ids)) return false;
                    std::string piece = tok.decode_token(t, true);
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                    ++generated;
                    return generated < max_tokens;
                };

                bool keep = (max_tokens > 0) ? emit_tok(next_tok) : false;
                std::vector<int> batch;
                while (keep) {
                    if (model->position() >= info.max_seq_len) break;
                    sampler.observe(next_tok);
                    if (sampler.is_greedy() && model->has_mtp()) {
                        batch.clear();
                        StepStats ms{};
                        model->forward_mtp_greedy(next_tok, batch, &ms);
                        mtp_acc += ms.mtp_accepted;
                        mtp_dft += ms.mtp_drafted;
                        pld_hits += ms.pld_hits;
                        ++mtp_rounds;
                        for (size_t j = 0; j < batch.size(); ++j) {
                            if (j + 1 < batch.size()) sampler.observe(batch[j]);
                            keep = emit_tok(batch[j]);
                            if (!keep) break;
                        }
                        next_tok = batch.back();
                    } else if (!sampler.is_greedy() && model->supports_mtp_sampled()) {
                        batch.clear();
                        StepStats ms{};
                        model->forward_mtp_sampled(next_tok, sampler, batch, &ms);
                        mtp_acc += ms.mtp_accepted;
                        mtp_dft += ms.mtp_drafted;
                        pld_hits += ms.pld_hits;
                        ++mtp_rounds;
                        for (size_t j = 0; j < batch.size(); ++j) {
                            keep = emit_tok(batch[j]);
                            if (!keep) break;
                        }
                        next_tok = batch.back();
                    } else if (sampler.is_greedy()) {
                        next_tok = model->forward_greedy(next_tok);
                        keep = emit_tok(next_tok);
                    } else {
                        const float* lg = model->forward(&next_tok, 1);
                        next_tok = sampler.sample(lg, info.vocab_size);
                        keep = emit_tok(next_tok);
                    }
                }
                auto t_dec1 = std::chrono::steady_clock::now();
                double dec_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
                std::printf("\n");
                std::fprintf(stderr,
                    "\n[stats] prefill: %zu tok (cached=%d, rest=%.1fms) in %.1f ms (%.1f tok/s) | decode: %d tok in %.1f ms (%.2f tok/s)",
                    ids.size(), start_idx, restore_ms, pf.gpu_ms,
                    (pf.gpu_ms > 0.0) ? ((ids.size() - start_idx) / (pf.gpu_ms / 1000.0)) : 0.0,
                    generated, dec_s * 1000.0, (generated > 1 && dec_s > 0.0) ? ((generated - 1) / dec_s) : 0.0);
                if (mtp_dft > 0) {
                    std::fprintf(stderr, " | mtp: %d/%d (%.1f%%, %.2f tok/rnd, pld=%d)",
                                 mtp_acc, mtp_dft, 100.0 * mtp_acc / mtp_dft,
                                 static_cast<double>(generated - 1) / std::max(1, mtp_rounds), pld_hits);
                }
                std::fprintf(stderr, "\n");
                return 0;
            }
        }

        if (cmd == "chat") {
            std::vector<ChatMessage> history;
            if (!system_prompt.empty()) history.push_back(ChatMessage{"system", system_prompt, {}, {}, {}, {}, {}});
            std::vector<int> cached_ids;
            PrefixCacheManager disk_cache(cache_dir);
            Sampler sampler(sc);

            std::fprintf(stderr, "Interactive chat started. Type /reset to clear context, /quit to exit.\n");
            std::string line;
            while (true) {
                std::fprintf(stderr, "\n> ");
                if (!std::getline(std::cin, line)) break;
                if (line == "/quit" || line == "/exit") break;
                if (line == "/reset") {
                    history.clear();
                    if (!system_prompt.empty()) history.push_back(ChatMessage{"system", system_prompt, {}, {}, {}, {}, {}});
                    cached_ids.clear();
                    model->reset();
                    sampler.reset(sc.seed);
                    std::fprintf(stderr, "[context cleared]\n");
                    continue;
                }
                if (line.empty()) continue;

                @autoreleasepool {
                    history.push_back(ChatMessage{"user", line, {}, {}, {}, {}, {}});
                    std::vector<int> ids = tok.encode_chat_template(history, true, effort);

                    int start_idx = 0;
                    if (!cached_ids.empty() &&
                        cached_ids.size() == static_cast<size_t>(model->position()) &&
                        ids.size() > cached_ids.size() &&
                        std::equal(cached_ids.begin(), cached_ids.end(), ids.begin())) {
                        start_idx = static_cast<int>(cached_ids.size());
                    } else {
                        model->reset();
                        cached_ids.clear();
                        if (disk_cache.enabled()) {
                            std::vector<int> restored;
                            int n_rest = disk_cache.try_restore(*model, ids, restored);
                            if (n_rest > 0) {
                                cached_ids = std::move(restored);
                                start_idx = n_rest;
                            }
                        }
                    }

                    StepStats pf{};
                    int argmax_tok = 0;
                    const float* logits = model->forward(
                        ids.data() + start_idx, static_cast<int>(ids.size()) - start_idx,
                        sampler.is_greedy() ? &argmax_tok : nullptr, &pf
                    );
                    cached_ids = ids;
                    int next_tok = sampler.is_greedy() ? argmax_tok : sampler.sample(logits, info.vocab_size);

                    std::string reply;
                    int gen = 0;
                    int mtp_acc = 0, mtp_dft = 0;
                    auto t0 = std::chrono::steady_clock::now();

                    auto emit_chat = [&](int t, bool& was_emitted) -> bool {
                        was_emitted = false;
                        if (is_eos_token(t, info.eos_token_ids)) return false;
                        std::string piece = tok.decode_token(t, true);
                        reply += piece;
                        std::fwrite(piece.data(), 1, piece.size(), stdout);
                        std::fflush(stdout);
                        ++gen;
                        was_emitted = true;
                        return gen < max_tokens;
                    };

                    bool first_emitted = false;
                    bool keep = (max_tokens > 0) ? emit_chat(next_tok, first_emitted) : false;
                    std::vector<int> batch;
                    const int spec_max_batch = std::max(8, model->mtp_depth() + 1);
                    while (keep) {
                        if (model->position() >= info.max_seq_len) break;
                        sampler.observe(next_tok);
                        const bool allow_spec = (max_tokens - gen) >= spec_max_batch;
                        if (allow_spec && sampler.is_greedy() && model->has_mtp()) {
                            batch.clear();
                            StepStats ms{};
                            model->forward_mtp_greedy(next_tok, batch, &ms);
                            mtp_acc += ms.mtp_accepted;
                            mtp_dft += ms.mtp_drafted;
                            cached_ids.push_back(next_tok);
                            for (size_t j = 0; j < batch.size(); ++j) {
                                if (j + 1 < batch.size()) sampler.observe(batch[j]);
                                bool emitted_j = false;
                                keep = emit_chat(batch[j], emitted_j);
                                if (emitted_j && j + 1 < batch.size()) cached_ids.push_back(batch[j]);
                                if (!keep) break;
                            }
                            next_tok = batch.back();
                        } else if (allow_spec && !sampler.is_greedy() && model->supports_mtp_sampled()) {
                            batch.clear();
                            StepStats ms{};
                            model->forward_mtp_sampled(next_tok, sampler, batch, &ms);
                            mtp_acc += ms.mtp_accepted;
                            mtp_dft += ms.mtp_drafted;
                            cached_ids.push_back(next_tok);
                            for (size_t j = 0; j < batch.size(); ++j) {
                                bool emitted_j = false;
                                keep = emit_chat(batch[j], emitted_j);
                                if (emitted_j && j + 1 < batch.size()) cached_ids.push_back(batch[j]);
                                if (!keep) break;
                            }
                            next_tok = batch.back();
                        } else if (sampler.is_greedy()) {
                            cached_ids.push_back(next_tok);
                            next_tok = model->forward_greedy(next_tok);
                            bool emitted_n = false;
                            keep = emit_chat(next_tok, emitted_n);
                        } else {
                            cached_ids.push_back(next_tok);
                            const float* lg = model->forward(&next_tok, 1);
                            next_tok = sampler.sample(lg, info.vocab_size);
                            bool emitted_n = false;
                            keep = emit_chat(next_tok, emitted_n);
                        }
                    }
                    if (static_cast<int>(cached_ids.size()) != model->position()) {
                        cached_ids.clear();
                    }
                    if (disk_cache.enabled() && cached_ids.size() >= 64 &&
                        static_cast<int>(cached_ids.size()) == model->position()) {
                        disk_cache.save(*model, cached_ids);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double>(t1 - t0).count();
                    std::printf("\n");
                    std::fprintf(stderr, "[prefill +%d tok in %.0fms | %d tok @ %.1f tok/s",
                                 static_cast<int>(ids.size()) - start_idx, pf.gpu_ms, gen,
                                 (gen > 1 && dt > 0) ? ((gen - 1) / dt) : 0.0);
                    if (mtp_dft > 0) {
                        std::fprintf(stderr, " | mtp %d/%d", mtp_acc, mtp_dft);
                    }
                    std::fprintf(stderr, "]\n");
                    history.push_back(ChatMessage{"assistant", reply, {}, {}, {}, {}, {}});
                }
            }
            return 0;
        }

        std::fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
        print_usage();
        return 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal: %s\n", ex.what());
        return 1;
    }
}
