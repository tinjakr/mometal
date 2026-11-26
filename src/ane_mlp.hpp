#pragma once

#include "metal_ctx.hpp"

#include <atomic>
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
    static constexpr int MAX_SUBS  = 2;       // 2 x 256 = 512 tokens/half
    static constexpr int MAX_PIECES = 2;      // 1 (full 4-bit MLP) or 2 (halves)

    // Detect Apple Neural Engine architecture string (e.g. "h13g" M1, "h14g" M2, "h15g" M3, "h16g" M4, "h17g" M5)
    static std::string detect_ane_arch();

    AneMlpEngine(MetalContext& ctx, const std::string& models_dir, int num_layers,
                 int hidden_width = 5120, const std::string& tag = "auto",
                 bool defer_warmup = false);
    ~AneMlpEngine();

    AneMlpEngine(const AneMlpEngine&) = delete;
    AneMlpEngine& operator=(const AneMlpEngine&) = delete;

    // 4-deep async warmup across all 4 (slot, sub) IOSurfaces (can overlap with safetensors mmap!)
    void start_warmup_async();
    void wait_warmup();

    bool ready() const { return ready_; }
    int num_layers() const { return num_layers_; }
    int num_pieces() const { return num_pieces_; }
    int block_rows() const { return block_rows_; } // typically 256
    int hidden_width() const { return hidden_width_; }
    int packed_width() const { return packed_width_; }
    double load_time_ms() const { return load_time_ms_; }
    const std::string& tag() const { return tag_; }
    const std::string& ane_arch() const { return ane_arch_; }

    // Zero-copy Metal textures backed by the ANE IOSurfaces
    id input_texture(int slot, int sub) const;
    id output_texture(int slot, int sub, int piece = 0) const;

    // Optional SmoothQuant inv1 buffer (f32[5120]) for layer L; nil if inv1 == 1.0 everywhere
    id inv1_buffer(int layer) const;

    // Cross-slot async ANE hardware-ring submission (zero worker thread, zero mutex/CV!)
    void submit_slot(int layer, int slot, int n_subs);
    bool is_slot_done(int slot) const;
    void wait_slot(int slot);
    void drain_all_slots() noexcept;

    // Convenience wrappers operating on the most recently submitted slot
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
    int hidden_width_ = 5120;
    int packed_width_ = 5184;
    std::string tag_;
    std::string ane_arch_;
    bool ready_ = false;
    double load_time_ms_ = 0.0;

    // Shared zero-copy IOSurface IO buffers: [slot][sub]
    IoSurfaceBinding in_io_[MAX_SLOTS][MAX_SUBS];
    IoSurfaceBinding out_io_[MAX_SLOTS][MAX_SUBS][MAX_PIECES];
    std::vector<LayerData> layers_;

    // Per-slot persistent CoreML async completion tracking (zero worker thread, zero per-call allocs!)
    id grp_[MAX_SLOTS] = {nil, nil}; // dispatch_group_t under ARC
    std::atomic<int> slot_rem_[MAX_SLOTS];
    std::atomic<bool> slot_done_[MAX_SLOTS];
    std::mutex err_mtx_;
    std::string slot_err_[MAX_SLOTS];
    int last_slot_ = 0;

    // 4-deep async startup warmup thread
    std::thread warmup_thread_;
    bool warmup_started_ = false;
    std::string warmup_err_;

    IoSurfaceBinding create_iosurface_io(int rows, int cols);
    void destroy_iosurface_io(IoSurfaceBinding& b);
    void cleanup_iosurfaces() noexcept;
    void run_warmup_4deep();
};

} // namespace mi
