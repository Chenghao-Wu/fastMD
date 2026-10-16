#pragma once
#include "../core/types.cuh"
#include <string>
#include <vector>

// Multiple-tau correlator (Ramirez et al., J. Chem. Phys. 133, 154103 (2010))
// for a vector of `nchannels` signals, run entirely on the device.
//
// Level 0 receives every sample and holds lags 0..p-1. Every m samples of
// level l are averaged into one sample of level l+1, whose lags p/m..p-1
// (in units of m^(l+1) samples) continue where level l stops; shorter lags
// of a coarse level are resolved better by the finer one. Correlations are
// accumulated for a list of ordered channel pairs (a, b):
//     C_ab(k) = < x_a(t) x_b(t - k) >,
// so (a, a) gives autocorrelations and (a, b) / (b, a) the two time orders
// of a cross-correlation. Accuracy improves with p/m; p = 16, m = 2 (the
// LAMMPS fix ave/correlate/long default) keeps the block-averaging bias of
// smooth correlation functions well below a percent. Channel means are kept
// so that fluctuation correlations <dx_a(t) dx_b(0)> = C_ab - mean_a mean_b
// can be formed afterwards.
//
// All state lives in one device allocation, so a rollback snapshot is a
// single copy.
struct MultipleTauCorrelator {
    int nchannels = 0, npairs = 0;
    int p = 16, m = 2, levels = 0;

    struct Row {
        long long lag_samples;            // lag in sampling intervals
        std::vector<double> c;            // one value per pair
        unsigned long long count;         // number of products averaged
    };

    // levels = 0: enough levels for 2^31 samples.
    void allocate(int nchannels, const std::vector<int2>& pairs, int p = 16, int m = 2,
                  int levels = 0);
    // Autocorrelations of every channel.
    void allocate_auto(int nchannels, int p = 16, int m = 2, int levels = 0);
    void free();
    void reset();
    bool allocated() const { return d_state_ != nullptr; }

    void save_snapshot(cudaStream_t stream = 0);
    void restore_snapshot(cudaStream_t stream = 0);

    // One sample: d_sample points to nchannels doubles on the device.
    void push(const double* d_sample, cudaStream_t stream = 0);
    // n consecutive samples ([n][nchannels] on the device) in one launch.
    void push_batch(const double* d_samples, int n, cudaStream_t stream = 0);

    std::vector<Row> results() const;          // one row per distinct lag
    std::vector<double> means() const;          // per channel, over all samples
    unsigned long long nsamples() const;
    const std::vector<int2>& pairs() const { return pairs_; }

    // Text output: header lines (parameters, means), then
    // "lag_steps  <pair columns...>  count". `columns` names the pairs.
    void write(const std::string& path, int sample_interval_steps,
               const std::vector<std::string>& columns,
               const std::string& extra_header = "") const;

private:
    std::vector<int2> pairs_;
    int2* d_pairs_ = nullptr;
    char* d_state_ = nullptr;
    char* d_snapshot_ = nullptr;
    size_t state_bytes_ = 0;
    struct Layout {
        size_t buf, corr, count, idx, fill, accum, acount, sum, nsamp;
    } off_{};
};
