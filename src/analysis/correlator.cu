#include "correlator.cuh"

void MultipleTauCorrelator::allocate() {
    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    CUDA_CHECK(cudaMalloc(&d_stress_buf, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr,       buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts,     CORR_LEVELS * CORR_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_insert_idx, CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_accum, CORR_LEVELS * STRESS_COMPONENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_count, CORR_LEVELS * sizeof(int)));
    reset();
}

void MultipleTauCorrelator::free() {
    CUDA_CHECK(cudaFree(d_stress_buf));
    CUDA_CHECK(cudaFree(d_corr));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_insert_idx));
    CUDA_CHECK(cudaFree(d_block_accum));
    CUDA_CHECK(cudaFree(d_block_count));
}

void MultipleTauCorrelator::reset() {
    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    CUDA_CHECK(cudaMemset(d_stress_buf, 0, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_corr, 0, buf_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_counts, 0, CORR_LEVELS * CORR_POINTS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_insert_idx, 0, CORR_LEVELS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_block_accum, 0, CORR_LEVELS * STRESS_COMPONENTS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_block_count, 0, CORR_LEVELS * sizeof(int)));
}

__global__ void correlator_push_kernel(
    float* __restrict__ stress_buf,
    float* __restrict__ corr,
    int* __restrict__ counts,
    int* __restrict__ insert_idx,
    float* __restrict__ block_accum,
    int* __restrict__ block_count,
    const float* __restrict__ new_sample)
{
    float sample[STRESS_COMPONENTS];
    for (int c = 0; c < STRESS_COMPONENTS; c++) {
        sample[c] = new_sample[c];
    }

    for (int level = 0; level < CORR_LEVELS; level++) {
        int idx = insert_idx[level];
        int buf_base = level * CORR_POINTS * STRESS_COMPONENTS;

        for (int c = 0; c < STRESS_COMPONENTS; c++) {
            stress_buf[buf_base + idx * STRESS_COMPONENTS + c] = sample[c];
        }

        int n_stored = min(idx + 1, CORR_POINTS);
        int corr_base = level * CORR_POINTS * STRESS_COMPONENTS;

        for (int lag = 0; lag < n_stored; lag++) {
            int older_idx = (idx - lag + CORR_POINTS) % CORR_POINTS;
            for (int c = 0; c < STRESS_COMPONENTS; c++) {
                float older_val = stress_buf[buf_base + older_idx * STRESS_COMPONENTS + c];
                corr[corr_base + lag * STRESS_COMPONENTS + c] += sample[c] * older_val;
            }
            counts[level * CORR_POINTS + lag]++;
        }

        insert_idx[level] = (idx + 1) % CORR_POINTS;

        int ba_base = level * STRESS_COMPONENTS;
        for (int c = 0; c < STRESS_COMPONENTS; c++) {
            block_accum[ba_base + c] += sample[c];
        }
        block_count[level]++;

        if (block_count[level] >= CORR_POINTS) {
            for (int c = 0; c < STRESS_COMPONENTS; c++) {
                sample[c] = block_accum[ba_base + c] / CORR_POINTS;
                block_accum[ba_base + c] = 0.0f;
            }
            block_count[level] = 0;
        } else {
            break;
        }
    }
}

void MultipleTauCorrelator::push_sample(const float* d_stress_6,
                                         cudaStream_t stream) {
    correlator_push_kernel<<<1, 1, 0, stream>>>(
        d_stress_buf, d_corr, d_counts, d_insert_idx,
        d_block_accum, d_block_count, d_stress_6);
}

void MultipleTauCorrelator::get_results(float* h_corr, int* h_counts,
                                         int* h_total_levels) {
    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    CUDA_CHECK(cudaMemcpy(h_corr, d_corr, buf_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts, d_counts,
                           CORR_LEVELS * CORR_POINTS * sizeof(int),
                           cudaMemcpyDeviceToHost));
    *h_total_levels = CORR_LEVELS;
}
