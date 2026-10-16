#pragma once
#include "../core/types.cuh"
#include <string>
#include <vector>

static constexpr int CORR_LEVELS = 16;
static constexpr int CORR_POINTS = 16;   // points per level = block-averaging factor
static constexpr int STRESS_COMPONENTS = 6;

// Multiple-tau correlator (Ramirez et al., J. Chem. Phys. 133, 154103 (2010))
// for the six stress components, run entirely on the device.
//
// Level 0 receives every sample; every CORR_POINTS samples of level l are
// averaged into one sample of level l+1. At level l, lag k corresponds to
// k * CORR_POINTS^l sampling intervals. Level 0 reports lags 0..P-1; higher
// levels report lags 1..P-1 only (lag 0 of block averages is not C(0), and
// shorter lags are resolved by the finer level).
struct MultipleTauCorrelator {
    float*  d_stress_buf = nullptr;   // [levels][points][6] ring buffers
    double* d_corr = nullptr;         // [levels][points][6] accumulated products
    int*    d_counts = nullptr;       // [levels][points] number of products
    int*    d_insert_idx = nullptr;   // [levels] next ring-buffer slot
    int*    d_fill = nullptr;         // [levels] samples held (saturates at points)
    double* d_block_accum = nullptr;  // [levels][6] running block sums
    int*    d_block_count = nullptr;  // [levels]
    char*   d_snapshot = nullptr;     // packed copy of all of the above

    struct Row {
        long long lag_samples;        // lag in sampling intervals
        double c[STRESS_COMPONENTS];  // <s(t) s(t+lag)>, order xx xy xz yy yz zz
        long long count;
    };

    void allocate();
    void free();
    void reset();

    // Device-side copy of the full correlator state (for rollback).
    void save_snapshot(cudaStream_t stream = 0);
    void restore_snapshot(cudaStream_t stream = 0);

    // d_stress_6: device pointer to one sample (xx xy xz yy yz zz)
    void push_sample(const float* d_stress_6, cudaStream_t stream = 0);

    // Raw accumulators (normalise by counts); kept for tests.
    void get_results(float* h_corr, int* h_counts, int* h_total_levels);

    // Normalised correlation function, one row per distinct lag.
    std::vector<Row> results() const;

    // Write "lag_steps C_xx C_xy C_xz C_yy C_yz C_zz count" text output.
    void write(const std::string& path, int sample_interval_steps) const;
};
