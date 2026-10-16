#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/correlator.cuh"
#include <cmath>
#include <array>
#include <random>

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

namespace {
std::vector<std::array<double, 6>> push_series(MultipleTauCorrelator& corr, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<std::array<double, 6>> series(n);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, 6 * sizeof(float)));
    float ar[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        float h[6];
        for (int c = 0; c < 6; c++) {
            ar[c] = 0.8f * ar[c] + g(rng);   // correlated AR(1) signal
            h[c] = ar[c];
            series[i][c] = h[c];
        }
        CUDA_CHECK(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));
        corr.push_sample(d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d));
    return series;
}

double direct_corr(const std::vector<std::array<double, 6>>& s, int lag, int c, long long* count) {
    double sum = 0;
    long long n = 0;
    for (size_t t = lag; t < s.size(); t++, n++) sum += s[t][c] * s[t - lag][c];
    *count = n;
    return n ? sum / n : 0.0;
}
}  // namespace

// After the level-0 ring buffer wraps, every lag must keep accumulating:
// with 64 samples lag 15 has 64 - 15 = 49 pairs (the previous version used
// the wrapped insert index as occupancy and counted only 4).
TEST(Correlator, Level0MatchesDirectCalculationAfterWrap) {
    MultipleTauCorrelator corr;
    corr.allocate();
    auto s = push_series(corr, 64, 7);
    auto rows = corr.results();
    for (int lag = 0; lag < CORR_POINTS; lag++) {
        ASSERT_LT(lag, (int)rows.size());
        const auto& r = rows[lag];
        EXPECT_EQ(r.lag_samples, lag);
        for (int c = 0; c < 6; c++) {
            long long n;
            double ref = direct_corr(s, lag, c, &n);
            EXPECT_EQ(r.count, n) << "lag " << lag;
            EXPECT_NEAR(r.c[c], ref, 1e-5 * (1 + std::fabs(ref))) << "lag " << lag << " comp " << c;
        }
    }
    EXPECT_EQ(rows[15].count, 49);
    corr.free();
}

// Level 1 holds block averages of CORR_POINTS samples; its lag k is
// k * CORR_POINTS sampling intervals and lag 0 is not reported.
TEST(Correlator, Level1LagsAndValues) {
    MultipleTauCorrelator corr;
    corr.allocate();
    const int n = CORR_POINTS * 40;
    auto s = push_series(corr, n, 11);
    std::vector<std::array<double, 6>> blocks(n / CORR_POINTS);
    for (size_t b = 0; b < blocks.size(); b++)
        for (int c = 0; c < 6; c++) {
            double sum = 0;
            for (int k = 0; k < CORR_POINTS; k++) sum += s[b * CORR_POINTS + k][c];
            blocks[b][c] = sum / CORR_POINTS;
        }
    auto rows = corr.results();
    int checked = 0;
    for (const auto& r : rows) {
        if (r.lag_samples < CORR_POINTS) continue;
        if (r.lag_samples >= (long long)CORR_POINTS * CORR_POINTS) break;
        ASSERT_EQ(r.lag_samples % CORR_POINTS, 0);
        int lag = (int)(r.lag_samples / CORR_POINTS);
        EXPECT_GE(lag, 1);
        for (int c = 0; c < 6; c++) {
            long long cnt;
            double ref = direct_corr(blocks, lag, c, &cnt);
            EXPECT_EQ(r.count, cnt);
            EXPECT_NEAR(r.c[c], ref, 1e-4 * (1 + std::fabs(ref)));
        }
        checked++;
    }
    EXPECT_EQ(checked, CORR_POINTS - 1);
    corr.free();
}
