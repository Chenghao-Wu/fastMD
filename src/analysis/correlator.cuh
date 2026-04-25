#pragma once
#include "../core/types.cuh"

static constexpr int CORR_LEVELS = 16;
static constexpr int CORR_POINTS = 16;
static constexpr int STRESS_COMPONENTS = 6;

struct MultipleTauCorrelator {
    float* d_stress_buf;
    float* d_corr;
    int*   d_counts;
    int*   d_insert_idx;
    float* d_block_accum;
    int*   d_block_count;

    void allocate();
    void free();
    void reset();

    void push_sample(const float* d_stress_6, cudaStream_t stream = 0);

    void get_results(float* h_corr, int* h_counts, int* h_total_levels);
};
