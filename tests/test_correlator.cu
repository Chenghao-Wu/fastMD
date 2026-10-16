#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "analysis/correlator.cuh"
#include <cmath>
#include <random>
#include <vector>

namespace {

// [n][nch] AR(1) signals x_c(t) = phi x_c(t-1) + noise; channel c+1 also
// carries a delayed copy of channel c so cross-correlations are nonzero.
std::vector<double> make_series(int n, int nch, double phi, unsigned seed, int delay = 3) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> s((size_t)n * nch), ar(nch, 0.0);
    for (int t = 0; t < n; t++)
        for (int c = 0; c < nch; c++) {
            ar[c] = phi * ar[c] + g(rng);
            double x = ar[c] + 0.25;   // nonzero mean
            if (c > 0 && t >= delay) x += 0.5 * s[(size_t)(t - delay) * nch + c - 1];
            s[(size_t)t * nch + c] = x;
        }
    return s;
}

void push_all(MultipleTauCorrelator& corr, const std::vector<double>& s, int chunk = 4096) {
    const int nch = corr.nchannels;
    const int n = (int)(s.size() / nch);
    double* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, (size_t)chunk * nch * sizeof(double)));
    for (int t0 = 0; t0 < n; t0 += chunk) {
        int k = std::min(chunk, n - t0);
        CUDA_CHECK(cudaMemcpy(d, s.data() + (size_t)t0 * nch, (size_t)k * nch * sizeof(double),
                              cudaMemcpyHostToDevice));
        corr.push_batch(d, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d));
}

// <x_a(t) x_b(t - lag)> over a series of block averages of size `block`.
double direct(const std::vector<double>& s, int nch, int a, int b, int block, int lag,
              unsigned long long* count) {
    const int n = (int)(s.size() / nch) / block;
    std::vector<double> xa(n), xb(n);
    for (int i = 0; i < n; i++) {
        double sa = 0, sb = 0;
        for (int k = 0; k < block; k++) {
            sa += s[((size_t)i * block + k) * nch + a];
            sb += s[((size_t)i * block + k) * nch + b];
        }
        xa[i] = sa / block;
        xb[i] = sb / block;
    }
    double sum = 0;
    unsigned long long cnt = 0;
    for (int t = lag; t < n; t++, cnt++) sum += xa[t] * xb[t - lag];
    *count = cnt;
    return cnt ? sum / cnt : 0.0;
}

}  // namespace

// Every reported value equals the direct estimate over the block averages
// of its level (lags p/m..p-1 at levels > 0), for auto and both time orders
// of a cross pair, including after the ring buffers wrap.
TEST(Correlator, MatchesBlockAveragedDirectCalculation) {
    for (int m : {2, 4}) {
        const int p = 16, nch = 3;
        MultipleTauCorrelator corr;
        corr.allocate(nch, {make_int2(0, 0), make_int2(1, 0), make_int2(0, 1), make_int2(2, 2)}, p, m);
        auto s = make_series(5000, nch, 0.9, 7 + m);
        push_all(corr, s);
        int checked = 0;
        for (const auto& r : corr.results()) {
            long long block = 1;
            int level = 0;
            while (r.lag_samples / block >= p) { block *= m; level++; }
            ASSERT_EQ(r.lag_samples % block, 0);
            int k = (int)(r.lag_samples / block);
            if (level > 0) EXPECT_GE(k, p / m);
            for (int q = 0; q < corr.npairs; q++) {
                unsigned long long cnt;
                int2 ab = corr.pairs()[q];
                double ref = direct(s, nch, ab.x, ab.y, (int)block, k, &cnt);
                ASSERT_EQ(r.count, cnt) << "lag " << r.lag_samples;
                EXPECT_NEAR(r.c[q], ref, 1e-9 * (1 + std::fabs(ref)))
                    << "m " << m << " lag " << r.lag_samples << " pair " << q;
            }
            checked++;
        }
        EXPECT_GT(checked, p + 3 * (p - p / m));
        corr.free();
    }
}

TEST(Correlator, MeansAndSampleCount) {
    MultipleTauCorrelator corr;
    corr.allocate_auto(2);
    auto s = make_series(1234, 2, 0.5, 3);
    push_all(corr, s, 100);
    EXPECT_EQ(corr.nsamples(), 1234ull);
    auto mu = corr.means();
    for (int c = 0; c < 2; c++) {
        double ref = 0;
        for (int t = 0; t < 1234; t++) ref += s[(size_t)t * 2 + c];
        EXPECT_NEAR(mu[c], ref / 1234, 1e-12);
    }
    corr.free();
}

// Block averaging biases the coarse levels; with p = 16, m = 2 the estimate
// of a smooth correlation function stays within 1% of the exact (direct)
// estimate from the same data wherever C(t) is not small. m = p (the old
// fixed setting) is several times worse.
TEST(Correlator, BlockAveragingBiasIsSmallForM2) {
    const int n = 1 << 18;
    auto s = make_series(n, 1, 0.995, 21, 0);   // tau ~ 200 samples
    auto worst = [&](int p, int m) {
        MultipleTauCorrelator corr;
        corr.allocate_auto(1, p, m);
        push_all(corr, s, 1 << 15);
        unsigned long long c0n;
        const double c0 = direct(s, 1, 0, 0, 1, 0, &c0n);
        double err = 0;
        for (const auto& r : corr.results()) {
            if (r.lag_samples > 4000) break;
            unsigned long long cnt;
            double ref = direct(s, 1, 0, 0, 1, (int)r.lag_samples, &cnt);
            if (ref - 0.0625 < 0.2 * (c0 - 0.0625)) continue;   // mean^2 = 0.0625
            err = std::max(err, std::fabs(r.c[0] - ref) / (ref - 0.0625));
        }
        corr.free();
        return err;
    };
    double e2 = worst(16, 2), e16 = worst(16, 16);
    EXPECT_LT(e2, 0.01) << "p=16 m=2 relative bias";
    EXPECT_GT(e16, 2.0 * e2) << "p=16 m=16 should be clearly worse";
    printf("max relative bias where C > 0.2 C(0): m=2 %.2e, m=16 %.2e\n", e2, e16);
}

TEST(Correlator, SnapshotRestoresState) {
    MultipleTauCorrelator corr;
    corr.allocate_auto(2);
    auto s = make_series(300, 2, 0.8, 5);
    push_all(corr, s);
    corr.save_snapshot();
    auto before = corr.results();
    push_all(corr, make_series(77, 2, 0.8, 6));
    corr.restore_snapshot();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto after = corr.results();
    ASSERT_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); i++) {
        EXPECT_EQ(before[i].count, after[i].count);
        EXPECT_EQ(before[i].c, after[i].c);
    }
    EXPECT_EQ(corr.nsamples(), 300ull);
    corr.free();
}

TEST(Correlator, RejectsBadParameters) {
    MultipleTauCorrelator corr;
    EXPECT_THROW(corr.allocate_auto(2, 16, 3), std::invalid_argument);
    EXPECT_THROW(corr.allocate(2, {make_int2(0, 2)}), std::invalid_argument);
    EXPECT_THROW(corr.allocate(2, {}), std::invalid_argument);
}
