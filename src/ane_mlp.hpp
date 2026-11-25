#pragma once

#include "metal_ctx.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mi {

// Native C++/Objective-C++ CoreML + Zero-Copy IOSurface ANE MLP engine for Qwen 3.8 27B.
// Eliminates 100% of Python / PyObjC / NumPy / CPU memcpy overhead:
//   - CVPixelBuffer (IOSurface-backed, 0x4C303068 FP16) bound directly to MTLTexture (R16Float)
//   - Metal kernel writes [x'/r | r | 0] FP16 straight into the ANE input IOSurface
//   - Dedicated C++ worker thread executes [MLModel predictionFromFeatures:options:error:] on ANE
//   - Metal kernel reads down_proj FP16 straight from the ANE output IOSurface and adds to hidden!
class AneMlpEngine {
public:
    static constexpr int MAX_SLOTS = 2;       // Half A (slot 0) and Half B (slot 1)
    static constexpr int MAX_SUBS  = 2;       // 2 x 256 = 512 tokens/half (1024/chunk -> stays inside M4 Max 48MB L2!)
    static constexpr int MAX_PIECES = 2;      // 1 (full hf4) or 2 (halves p0+p1)
    static constexpr int PACKED_WIDTH = 5184; // 5120 normed/r + 1 scale r + 63 zero pad
    static constexpr int HIDDEN_WIDTH = 5120;

    AneMlpEngine(MetalContext& ctx, const std::string& models_dir, int num_layers,
                 const std::string& tag = "auto");
    ~AneMlpEngine();

    AneMlpEngine(const AneMlpEngine&) = delete;
    AneMlpEngine& operator=(const AneMlpEngine&) = delete;

    bool ready() const { return ready_; }
    int num_layers() const { return num_layers_; }
    int num_pieces() const { return num_pieces_; }
    int block_rows() const { return block_rows_; } // typically 256
    double load_time_ms() const { return load_time_ms_; }
    const std::string& tag() const { return tag_; }

    // Zero-copy Metal textures backed by the ANE IOSurfaces
    id input_texture(int slot, int sub) const;
    id output_texture(int slot, int sub, int piece = 0) const;

    // Optional SmoothQuant inv1 buffer (f32[5120]) for layer L; nil if inv1 == 1.0 everywhere
    id inv1_buffer(int layer) const;

    // Async execution on the dedicated ANE worker thread (runs n_subs consecutive 256-row calls)
    void submit(int layer, int slot, int n_subs);
    void wait();

private:
    struct IoSurfaceBinding {
        void* pixel_buffer = nullptr; // CVPixelBufferRef
        id multi_array = nil;         // MLMultiArray*
        id mtl_texture = nil;         // id<MTLTexture> (zero-copy R16Float on same IOSurface!)
    };

    struct PieceCall {
        id model = nil;    // MLModel*
        id provider = nil; // MLDictionaryFeatureProvider* (per slot & sub)
        id options = nil;  // MLPredictionOptions* (outputBackings -> out_surf)
    };

    struct LayerSlotSub {
        PieceCall pieces[MAX_PIECES];
    };

    struct LayerData {
        id models[MAX_PIECES] = {nil, nil};
        LayerSlotSub calls[MAX_SLOTS][MAX_SUBS];
        id inv1_buf = nil; // f32[5120] or nil if all 1.0
    };

    MetalContext& ctx_;
    int num_layers_ = 0;
    int num_pieces_ = 1;
    int block_rows_ = 256;
    std::string tag_;
    bool ready_ = false;
    double load_time_ms_ = 0.0;

    // Shared zero-copy IOSurface IO buffers: [slot][sub]
    IoSurfaceBinding in_io_[MAX_SLOTS][MAX_SUBS];
    IoSurfaceBinding out_io_[MAX_SLOTS][MAX_SUBS][MAX_PIECES];
    std::vector<LayerData> layers_;

    // Dedicated ANE worker thread with minimal-latency mutex/cv handshake
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    bool has_job_ = false;
    int job_layer_ = 0;
    int job_slot_ = 0;
    int job_nsubs_ = 1;
    std::string job_error_;

    IoSurfaceBinding create_iosurface_io(int rows, int cols);
    void destroy_iosurface_io(IoSurfaceBinding& b);
    void worker_loop();
};

} // namespace mi
