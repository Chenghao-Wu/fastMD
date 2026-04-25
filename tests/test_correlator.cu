#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/correlator.cuh"
#include <cmath>

TEST(Correlator, ConstantSignalAutoCorrelation) {
    MultipleTauCorrelator corr;
    corr.allocate();

    float h_stress[6] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float* d_stress;
    CUDA_CHECK(cudaMalloc(&d_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_stress, h_stress, 6 * sizeof(float),
                           cudaMemcpyHostToDevice));

    for (int i = 0; i < 64; i++) {
        corr.push_sample(d_stress);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    std::vector<float> h_corr(buf_size);
    std::vector<int> h_counts(CORR_LEVELS * CORR_POINTS);
    int total_levels;
    corr.get_results(h_corr.data(), h_counts.data(), &total_levels);

    for (int lag = 0; lag < CORR_POINTS; lag++) {
        int count = h_counts[0 * CORR_POINTS + lag];
        if (count > 0) {
            float c = h_corr[0 * CORR_POINTS * STRESS_COMPONENTS + lag * STRESS_COMPONENTS + 0];
            float normalized = c / count;
            EXPECT_NEAR(normalized, 1.0f, 1e-5f) << "lag " << lag;
        }
    }

    corr.free();
    free_device(d_stress);
}

TEST(Correlator, DecayingSignal) {
    MultipleTauCorrelator corr;
    corr.allocate();

    float* d_stress;
    CUDA_CHECK(cudaMalloc(&d_stress, 6 * sizeof(float)));

    float tau = 10.0f;
    int nsamples = 256;
    for (int i = 0; i < nsamples; i++) {
        float val = expf(-(float)i / tau);
        float h_stress[6] = {val, 0, 0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(d_stress, h_stress, 6 * sizeof(float),
                               cudaMemcpyHostToDevice));
        corr.push_sample(d_stress);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int buf_size = CORR_LEVELS * CORR_POINTS * STRESS_COMPONENTS;
    std::vector<float> h_corr(buf_size);
    std::vector<int> h_counts(CORR_LEVELS * CORR_POINTS);
    int total_levels;
    corr.get_results(h_corr.data(), h_counts.data(), &total_levels);

    float c0 = h_corr[0] / h_counts[0];
    float c1 = h_corr[STRESS_COMPONENTS] / h_counts[1];
    EXPECT_GT(c0, c1) << "Autocorrelation should decay with lag";

    corr.free();
    free_device(d_stress);
}
