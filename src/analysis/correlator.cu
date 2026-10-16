#include "correlator.cuh"
#include <cstdio>
#include <stdexcept>

namespace {
constexpr int kBuf = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
}

void MultipleTauCorrelator::allocate() {
    CUDA_CHECK(cudaMalloc(&d_stress_buf, kBuf * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr,       kBuf * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counts,     CORR_LEVELS * CORR_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_insert_idx, CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fill,       CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_accum, CORR_LEVELS * STRESS_COMPONENTS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_count, CORR_LEVELS * sizeof(int)));
    reset();
}

void MultipleTauCorrelator::free() {
    for (void* p : {(void*)d_stress_buf, (void*)d_corr, (void*)d_counts, (void*)d_insert_idx,
                    (void*)d_fill, (void*)d_block_accum, (void*)d_block_count})
        if (p) CUDA_CHECK(cudaFree(p));
    *this = MultipleTauCorrelator{};
}

void MultipleTauCorrelator::reset() {
    CUDA_CHECK(cudaMemset(d_stress_buf, 0, kBuf * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_corr, 0, kBuf * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_counts, 0, CORR_LEVELS * CORR_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_insert_idx, 0, CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fill, 0, CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_block_accum, 0, CORR_LEVELS * STRESS_COMPONENTS * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_block_count, 0, CORR_LEVELS * sizeof(int)));
}

// Single thread: at most CORR_LEVELS * CORR_POINTS * 6 products per sample.
__global__ void correlator_push_kernel(
    float* __restrict__ stress_buf,
    double* __restrict__ corr,
    int* __restrict__ counts,
    int* __restrict__ insert_idx,
    int* __restrict__ fill,
    double* __restrict__ block_accum,
    int* __restrict__ block_count,
    const float* __restrict__ new_sample)
{
    double sample[STRESS_COMPONENTS];
    for (int c = 0; c < STRESS_COMPONENTS; c++) sample[c] = new_sample[c];

    for (int level = 0; level < CORR_LEVELS; level++) {
        int idx = insert_idx[level];
        int base = level * CORR_POINTS * STRESS_COMPONENTS;
        for (int c = 0; c < STRESS_COMPONENTS; c++)
            stress_buf[base + idx * STRESS_COMPONENTS + c] = (float)sample[c];

        // Occupancy is tracked separately from the ring index: after the
        // buffer wraps, all CORR_POINTS lags remain available.
        int n_stored = min(fill[level] + 1, CORR_POINTS);
        fill[level] = n_stored;
        for (int lag = 0; lag < n_stored; lag++) {
            int older = (idx - lag + CORR_POINTS) % CORR_POINTS;
            for (int c = 0; c < STRESS_COMPONENTS; c++)
                corr[base + lag * STRESS_COMPONENTS + c] +=
                    sample[c] * stress_buf[base + older * STRESS_COMPONENTS + c];
            counts[level * CORR_POINTS + lag]++;
        }
        insert_idx[level] = (idx + 1) % CORR_POINTS;

        // Block-average into the next level.
        int ba = level * STRESS_COMPONENTS;
        for (int c = 0; c < STRESS_COMPONENTS; c++) block_accum[ba + c] += sample[c];
        if (++block_count[level] < CORR_POINTS) break;
        for (int c = 0; c < STRESS_COMPONENTS; c++) {
            sample[c] = block_accum[ba + c] / CORR_POINTS;
            block_accum[ba + c] = 0.0;
        }
        block_count[level] = 0;
    }
}

void MultipleTauCorrelator::push_sample(const float* d_stress_6, cudaStream_t stream) {
    correlator_push_kernel<<<1, 1, 0, stream>>>(
        d_stress_buf, d_corr, d_counts, d_insert_idx, d_fill,
        d_block_accum, d_block_count, d_stress_6);
    CUDA_CHECK(cudaGetLastError());
}

void MultipleTauCorrelator::get_results(float* h_corr, int* h_counts, int* h_total_levels) {
    std::vector<double> c(kBuf);
    CUDA_CHECK(cudaMemcpy(c.data(), d_corr, kBuf * sizeof(double), cudaMemcpyDeviceToHost));
    for (int k = 0; k < kBuf; k++) h_corr[k] = (float)c[k];
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, CORR_LEVELS * CORR_POINTS * sizeof(int),
                          cudaMemcpyDeviceToHost));
    *h_total_levels = CORR_LEVELS;
}

std::vector<MultipleTauCorrelator::Row> MultipleTauCorrelator::results() const {
    std::vector<double> c(kBuf);
    std::vector<int> n(CORR_LEVELS * CORR_POINTS);
    CUDA_CHECK(cudaMemcpy(c.data(), d_corr, kBuf * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n.data(), d_counts, n.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Row> rows;
    long long scale = 1;   // sampling intervals per sample at this level
    for (int level = 0; level < CORR_LEVELS; level++) {
        for (int lag = (level == 0 ? 0 : 1); lag < CORR_POINTS; lag++) {
            int cnt = n[level * CORR_POINTS + lag];
            if (cnt == 0) continue;
            Row r;
            r.lag_samples = (long long)lag * scale;
            for (int k = 0; k < STRESS_COMPONENTS; k++)
                r.c[k] = c[(level * CORR_POINTS + lag) * STRESS_COMPONENTS + k] / cnt;
            r.count = cnt;
            rows.push_back(r);
        }
        scale *= CORR_POINTS;
    }
    return rows;
}

void MultipleTauCorrelator::write(const std::string& path, int sample_interval_steps) const {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot open " + path);
    fprintf(fp, "# lag_steps  C_xx  C_xy  C_xz  C_yy  C_yz  C_zz  count\n");
    for (const Row& r : results()) {
        fprintf(fp, "%lld", r.lag_samples * sample_interval_steps);
        for (int k = 0; k < STRESS_COMPONENTS; k++) fprintf(fp, " %.10e", r.c[k]);
        fprintf(fp, " %lld\n", r.count);
    }
    fclose(fp);
}
