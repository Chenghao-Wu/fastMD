#include "correlator.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {

struct CorrView {
    double* buf;                  // [levels][p][nch] ring buffers
    double* corr;                 // [levels][p][npairs]
    unsigned long long* count;    // [levels][p]
    int* idx;                     // [levels] next ring slot
    int* fill;                    // [levels] samples held (saturates at p)
    double* accum;                // [levels][nch] running block sums
    int* acount;                  // [levels]
    double* sum;                  // [nch] level-0 sums (means)
    unsigned long long* nsamp;    // [1]
};

// One block; samples are processed in order. Per level, every (lag, pair)
// product belongs to exactly one thread, so there are no atomics.
__global__ void correlator_push_kernel(CorrView v, const int2* __restrict__ pairs,
                                       const double* __restrict__ samples, int nsamples_in,
                                       int nch, int npairs, int p, int m, int levels) {
    extern __shared__ double s[];   // [nch] current sample of the level being processed
    __shared__ int s_carry;          // 1: this level completed a block, go on
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int n = 0; n < nsamples_in; n++) {
        for (int c = tid; c < nch; c += nt) {
            double x = samples[(size_t)n * nch + c];
            s[c] = x;
            v.sum[c] += x;
        }
        if (tid == 0) v.nsamp[0]++;
        __syncthreads();
        for (int level = 0; level < levels; level++) {
            const int idx = v.idx[level];
            double* buf = v.buf + (size_t)level * p * nch;
            for (int c = tid; c < nch; c += nt) buf[idx * nch + c] = s[c];
            __syncthreads();
            const int stored = min(v.fill[level] + 1, p);
            const int kmin = level == 0 ? 0 : p / m;
            const int nk = stored - kmin;
            if (nk > 0) {
                double* corr = v.corr + (size_t)level * p * npairs;
                for (int w = tid; w < nk * npairs; w += nt) {
                    const int k = kmin + w / npairs, q = w % npairs;
                    const int older = (idx - k + p) % p;
                    const int2 ab = pairs[q];
                    corr[k * npairs + q] += s[ab.x] * buf[older * nch + ab.y];
                }
                for (int k = kmin + tid; k < stored; k += nt) v.count[level * p + k]++;
            }
            for (int c = tid; c < nch; c += nt) v.accum[level * nch + c] += s[c];
            __syncthreads();
            if (tid == 0) {
                v.fill[level] = stored;
                v.idx[level] = (idx + 1) % p;
                s_carry = (++v.acount[level] == m);
                if (s_carry) v.acount[level] = 0;
            }
            __syncthreads();
            if (!s_carry) break;
            // the block average becomes the next level's sample
            for (int c = tid; c < nch; c += nt) {
                s[c] = v.accum[level * nch + c] / m;
                v.accum[level * nch + c] = 0.0;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

}  // namespace

void MultipleTauCorrelator::allocate(int nch, const std::vector<int2>& pairs, int p_in, int m_in,
                                     int levels_in) {
    if (nch < 1) throw std::invalid_argument("correlator: nchannels must be >= 1");
    if (pairs.empty()) throw std::invalid_argument("correlator: no channel pairs");
    for (const int2& ab : pairs)
        if (ab.x < 0 || ab.x >= nch || ab.y < 0 || ab.y >= nch)
            throw std::invalid_argument("correlator: pair channel out of range");
    if (m_in < 2 || p_in < 2 || p_in % m_in != 0)
        throw std::invalid_argument("correlator: need m >= 2 and p a multiple of m");
    free();
    nchannels = nch;
    pairs_ = pairs;
    npairs = (int)pairs.size();
    p = p_in;
    m = m_in;
    if (levels_in > 0) {
        levels = levels_in;
    } else {   // longest lag p * m^(levels-1) >= 2^31 samples
        levels = 1;
        for (double reach = p; reach < 2147483648.0 && levels < 64; reach *= m) levels++;
    }
    auto align = [](size_t x) { return (x + 15) & ~size_t(15); };
    size_t o = 0;
    off_.buf = o;    o = align(o + sizeof(double) * levels * p * nch);
    off_.corr = o;   o = align(o + sizeof(double) * levels * p * npairs);
    off_.count = o;  o = align(o + sizeof(unsigned long long) * levels * p);
    off_.idx = o;    o = align(o + sizeof(int) * levels);
    off_.fill = o;   o = align(o + sizeof(int) * levels);
    off_.accum = o;  o = align(o + sizeof(double) * levels * nch);
    off_.acount = o; o = align(o + sizeof(int) * levels);
    off_.sum = o;    o = align(o + sizeof(double) * nch);
    off_.nsamp = o;  o = align(o + sizeof(unsigned long long));
    state_bytes_ = o;
    CUDA_CHECK(cudaMalloc(&d_state_, state_bytes_));
    CUDA_CHECK(cudaMalloc(&d_pairs_, npairs * sizeof(int2)));
    CUDA_CHECK(cudaMemcpy(d_pairs_, pairs.data(), npairs * sizeof(int2), cudaMemcpyHostToDevice));
    reset();
}

void MultipleTauCorrelator::allocate_auto(int nch, int p_in, int m_in, int levels_in) {
    std::vector<int2> pairs;
    for (int c = 0; c < nch; c++) pairs.push_back(make_int2(c, c));
    allocate(nch, pairs, p_in, m_in, levels_in);
}

void MultipleTauCorrelator::free() {
    for (void* q : {(void*)d_state_, (void*)d_snapshot_, (void*)d_pairs_})
        if (q) CUDA_CHECK(cudaFree(q));
    d_state_ = d_snapshot_ = nullptr;
    d_pairs_ = nullptr;
    pairs_.clear();
    nchannels = npairs = levels = 0;
    state_bytes_ = 0;
}

void MultipleTauCorrelator::reset() {
    CUDA_CHECK(cudaMemset(d_state_, 0, state_bytes_));
}

void MultipleTauCorrelator::save_snapshot(cudaStream_t stream) {
    if (!d_snapshot_) CUDA_CHECK(cudaMalloc(&d_snapshot_, state_bytes_));
    CUDA_CHECK(cudaMemcpyAsync(d_snapshot_, d_state_, state_bytes_, cudaMemcpyDeviceToDevice, stream));
}

void MultipleTauCorrelator::restore_snapshot(cudaStream_t stream) {
    if (!d_snapshot_) return;
    CUDA_CHECK(cudaMemcpyAsync(d_state_, d_snapshot_, state_bytes_, cudaMemcpyDeviceToDevice, stream));
}

void MultipleTauCorrelator::push(const double* d_sample, cudaStream_t stream) {
    push_batch(d_sample, 1, stream);
}

void MultipleTauCorrelator::push_batch(const double* d_samples, int n, cudaStream_t stream) {
    if (n <= 0) return;
    CorrView v{reinterpret_cast<double*>(d_state_ + off_.buf),
               reinterpret_cast<double*>(d_state_ + off_.corr),
               reinterpret_cast<unsigned long long*>(d_state_ + off_.count),
               reinterpret_cast<int*>(d_state_ + off_.idx),
               reinterpret_cast<int*>(d_state_ + off_.fill),
               reinterpret_cast<double*>(d_state_ + off_.accum),
               reinterpret_cast<int*>(d_state_ + off_.acount),
               reinterpret_cast<double*>(d_state_ + off_.sum),
               reinterpret_cast<unsigned long long*>(d_state_ + off_.nsamp)};
    const int threads = npairs * p >= 256 ? 256 : 64;
    correlator_push_kernel<<<1, threads, nchannels * sizeof(double), stream>>>(
        v, d_pairs_, d_samples, n, nchannels, npairs, p, m, levels);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<MultipleTauCorrelator::Row> MultipleTauCorrelator::results() const {
    std::vector<char> h(state_bytes_);
    CUDA_CHECK(cudaMemcpy(h.data(), d_state_, state_bytes_, cudaMemcpyDeviceToHost));
    const double* corr = reinterpret_cast<const double*>(h.data() + off_.corr);
    const unsigned long long* count = reinterpret_cast<const unsigned long long*>(h.data() + off_.count);
    std::vector<Row> rows;
    long long scale = 1;   // sampling intervals per sample at this level
    for (int level = 0; level < levels; level++) {
        for (int k = (level == 0 ? 0 : p / m); k < p; k++) {
            unsigned long long cnt = count[level * p + k];
            if (cnt == 0) continue;
            Row r;
            r.lag_samples = (long long)k * scale;
            r.c.resize(npairs);
            for (int q = 0; q < npairs; q++) r.c[q] = corr[((size_t)level * p + k) * npairs + q] / cnt;
            r.count = cnt;
            rows.push_back(std::move(r));
        }
        if (scale > (1ll << 50)) break;
        scale *= m;
    }
    return rows;
}

std::vector<double> MultipleTauCorrelator::means() const {
    std::vector<double> sum(nchannels);
    unsigned long long n = nsamples();
    CUDA_CHECK(cudaMemcpy(sum.data(), d_state_ + off_.sum, nchannels * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (double& x : sum) x = n ? x / n : 0.0;
    return sum;
}

unsigned long long MultipleTauCorrelator::nsamples() const {
    unsigned long long n = 0;
    CUDA_CHECK(cudaMemcpy(&n, d_state_ + off_.nsamp, sizeof(n), cudaMemcpyDeviceToHost));
    return n;
}

void MultipleTauCorrelator::write(const std::string& path, int sample_interval_steps,
                                  const std::vector<std::string>& columns,
                                  const std::string& extra_header) const {
    if ((int)columns.size() != npairs) throw std::invalid_argument("correlator: column names");
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot open " + path);
    fprintf(fp, "# multiple-tau correlator: p %d  m %d  levels %d  samples %llu  sample interval %d steps\n",
            p, m, levels, nsamples(), sample_interval_steps);
    if (!extra_header.empty()) fprintf(fp, "%s", extra_header.c_str());
    std::vector<double> mu = means();
    fprintf(fp, "# channel means:");
    for (double x : mu) fprintf(fp, " %.10e", x);
    fprintf(fp, "\n# lag_steps");
    for (const std::string& c : columns) fprintf(fp, "  %s", c.c_str());
    fprintf(fp, "  count\n");
    for (const Row& r : results()) {
        fprintf(fp, "%lld", r.lag_samples * sample_interval_steps);
        for (double x : r.c) fprintf(fp, " %.10e", x);
        fprintf(fp, " %llu\n", r.count);
    }
    fclose(fp);
}
