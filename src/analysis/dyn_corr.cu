#include "dyn_corr.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace {

constexpr int kMaxModes = 8;
constexpr int kMaxPairs = 4;        // origin/lag pairs per bead-kernel launch
constexpr int kMaxBins = 64;
constexpr int kBeadVals = DynCorr::kBeadQ;   // 10 bead quantities

struct Geometry {
    int natoms, nbins, axis, film, nq, nchains, nmodes;
    float width, q;
};

struct Pairs {
    int n;
    int slot[kMaxPairs];
    int lag[kMaxPairs];   // lag index
};

__device__ __forceinline__ float3 unwrapped(float4 r, const int* im, const Box& b) {
    return make_float3(r.x + im[0] * b.L.x + im[1] * b.xy, r.y + im[1] * b.L.y, r.z + im[2] * b.L.z);
}

__device__ __forceinline__ float comp(float3 v, int a) { return a == 0 ? v.x : a == 1 ? v.y : v.z; }

__device__ __forceinline__ int depth_bin(float u_axis, float center, const Geometry& g) {
    if (!g.film) return 0;
    int k = (int)floorf((u_axis - center) / g.width + 0.5f * g.nbins);
    return min(max(k, 0), g.nbins - 1);
}

__global__ void capture_beads(const float4* __restrict__ pos, const int* __restrict__ image,
                              const int* __restrict__ tag, const Box* __restrict__ d_box,
                              const float* __restrict__ d_center, Geometry g, int slot,
                              float4* __restrict__ bpos, int4* __restrict__ bimg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.natoms) return;
    const Box b = *d_box;
    const float4 r = pos[i];
    const int* im = image + 3 * i;
    const float3 u = unwrapped(r, im, b);
    const int bin = depth_bin(comp(u, g.axis), g.film ? *d_center : 0.0f, g);
    const int t = tag ? tag[i] : i;
    bpos[(size_t)slot * g.natoms + t] = make_float4(r.x, r.y, r.z, __int_as_float(bin));
    bimg[(size_t)slot * g.natoms + t] = make_int4(im[0], im[1], im[2], 1);
}

__global__ void eval_beads(const float4* __restrict__ pos, const int* __restrict__ image,
                           const int* __restrict__ tag, const Box* __restrict__ d_box,
                           const float* __restrict__ d_center, Geometry g, Pairs pairs,
                           const float4* __restrict__ bpos, int4* __restrict__ bimg,
                           double* __restrict__ acc) {
    __shared__ float s[kMaxPairs * kMaxBins * kBeadVals];
    const int nv = pairs.n * g.nbins * kBeadVals;
    for (int k = threadIdx.x; k < nv; k += blockDim.x) s[k] = 0.0f;
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < g.natoms) {
        const Box b = *d_box;
        const float4 r = pos[i];
        const int* im = image + 3 * i;
        const float3 u = unwrapped(r, im, b);
        const int cur = depth_bin(comp(u, g.axis), g.film ? *d_center : 0.0f, g);
        const int t = tag ? tag[i] : i;
        const int a1 = (g.axis + 1) % 3, a2 = (g.axis + 2) % 3;
        for (int p = 0; p < pairs.n; p++) {
            const size_t o = (size_t)pairs.slot[p] * g.natoms + t;
            const float4 r0 = bpos[o];
            int4 i0 = bimg[o];
            // displacement from wrapped positions and image differences (exact in float)
            const int dimg[3] = {im[0] - i0.x, im[1] - i0.y, im[2] - i0.z};
            const float3 d = make_float3(r.x - r0.x + dimg[0] * b.L.x + dimg[1] * b.xy,
                                         r.y - r0.y + dimg[1] * b.L.y, r.z - r0.z + dimg[2] * b.L.z);
            const float da1 = comp(d, a1), da2 = comp(d, a2), dn = comp(d, g.axis);
            const float vals[5] = {da1 * da1 + da2 * da2, dn * dn,
                                   0.5f * (__cosf(g.q * da1) + __cosf(g.q * da2)), __cosf(g.q * dn), 1.0f};
            const int bin0 = __float_as_int(r0.w);
            float* sp = s + (p * g.nbins + bin0) * kBeadVals;
            for (int k = 0; k < 5; k++) atomicAdd(&sp[k], vals[k]);
            bool resident = i0.w && cur == bin0;
            if (resident) for (int k = 0; k < 5; k++) atomicAdd(&sp[5 + k], vals[k]);
            if (i0.w && !resident) bimg[o].w = 0;   // left its origin bin: stays out
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < nv; k += blockDim.x) {
        if (s[k] == 0.0f) continue;
        const int p = k / (g.nbins * kBeadVals), rem = k % (g.nbins * kBeadVals);
        const int bin = rem / kBeadVals, q = rem % kBeadVals;
        atomicAdd(&acc[((size_t)pairs.lag[p] * g.nbins + bin) * g.nq + q], (double)s[k]);
    }
}

// One warp per chain: current centre of mass, end-to-end vector and Rouse
// modes; capture (slot >= 0, npairs = 0) or accumulate against origins.
__global__ void chains_kernel(const float4* __restrict__ pos, const int* __restrict__ image,
                              const int* __restrict__ rank, const Box* __restrict__ d_box,
                              const float* __restrict__ d_center, Geometry g,
                              const int* __restrict__ offsets, const int* __restrict__ members,
                              const float* __restrict__ modes, int capture_slot, Pairs pairs,
                              float4* __restrict__ cdata, double* __restrict__ acc) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= g.nchains) return;
    const Box b = *d_box;
    const int start = offsets[warp], len = offsets[warp + 1] - start;
    float3 com = make_float3(0, 0, 0);
    float3 X[kMaxModes];
    for (int m = 0; m < g.nmodes; m++) X[m] = make_float3(0, 0, 0);
    float3 first = make_float3(0, 0, 0), last = make_float3(0, 0, 0);
    for (int k = lane; k < len; k += 32) {
        const int t = members[start + k];
        const int i = rank ? rank[t] : t;
        const float3 u = unwrapped(pos[i], image + 3 * i, b);
        com.x += u.x; com.y += u.y; com.z += u.z;
        for (int m = 0; m < g.nmodes; m++) {
            const float c = cosf(modes[m] * 3.14159265358979f * (k + 0.5f) / len);
            X[m].x += c * u.x; X[m].y += c * u.y; X[m].z += c * u.z;
        }
        if (k == 0) first = u;
        if (k == len - 1) last = u;
    }
    for (int o = 16; o > 0; o >>= 1) {
        com.x += __shfl_down_sync(0xffffffffu, com.x, o);
        com.y += __shfl_down_sync(0xffffffffu, com.y, o);
        com.z += __shfl_down_sync(0xffffffffu, com.z, o);
        for (int m = 0; m < g.nmodes; m++) {
            X[m].x += __shfl_down_sync(0xffffffffu, X[m].x, o);
            X[m].y += __shfl_down_sync(0xffffffffu, X[m].y, o);
            X[m].z += __shfl_down_sync(0xffffffffu, X[m].z, o);
        }
    }
    const int lf = 0, ll = (len - 1) % 32;   // lanes holding the first / last bead
    first.x = __shfl_sync(0xffffffffu, first.x, lf);
    first.y = __shfl_sync(0xffffffffu, first.y, lf);
    first.z = __shfl_sync(0xffffffffu, first.z, lf);
    last.x = __shfl_sync(0xffffffffu, last.x, ll);
    last.y = __shfl_sync(0xffffffffu, last.y, ll);
    last.z = __shfl_sync(0xffffffffu, last.z, ll);
    if (lane != 0) return;
    const float inv = 1.0f / len;
    com.x *= inv; com.y *= inv; com.z *= inv;
    for (int m = 0; m < g.nmodes; m++) { X[m].x *= inv; X[m].y *= inv; X[m].z *= inv; }
    const float3 R = make_float3(last.x - first.x, last.y - first.y, last.z - first.z);
    const int stride = 2 + g.nmodes;
    if (capture_slot >= 0) {
        float4* c = cdata + ((size_t)capture_slot * g.nchains + warp) * stride;
        const int bin = depth_bin(comp(com, g.axis), g.film ? *d_center : 0.0f, g);
        c[0] = make_float4(com.x, com.y, com.z, __int_as_float(bin));
        c[1] = make_float4(R.x, R.y, R.z, 0.0f);
        for (int m = 0; m < g.nmodes; m++) c[2 + m] = make_float4(X[m].x, X[m].y, X[m].z, 0.0f);
        return;
    }
    const int a1 = (g.axis + 1) % 3, a2 = (g.axis + 2) % 3;
    for (int p = 0; p < pairs.n; p++) {
        const float4* c = cdata + ((size_t)pairs.slot[p] * g.nchains + warp) * stride;
        const float4 c0 = c[0], r0 = c[1];
        const float3 d = make_float3(com.x - c0.x, com.y - c0.y, com.z - c0.z);
        const int bin = __float_as_int(c0.w);
        double* a = acc + ((size_t)pairs.lag[p] * g.nbins + bin) * g.nq;
        atomicAdd(&a[DynCorr::COM_PAR], (double)(comp(d, a1) * comp(d, a1) + comp(d, a2) * comp(d, a2)));
        atomicAdd(&a[DynCorr::COM_PERP], (double)(comp(d, g.axis) * comp(d, g.axis)));
        atomicAdd(&a[DynCorr::REE_DOT], (double)(R.x * r0.x + R.y * r0.y + R.z * r0.z));
        atomicAdd(&a[DynCorr::REE0_SQ], (double)(r0.x * r0.x + r0.y * r0.y + r0.z * r0.z));
        atomicAdd(&a[DynCorr::N_CHAIN], 1.0);
        for (int m = 0; m < g.nmodes; m++) {
            const float4 x0 = c[2 + m];
            atomicAdd(&a[DynCorr::kFixedQ + 2 * m], (double)(X[m].x * x0.x + X[m].y * x0.y + X[m].z * x0.z));
            atomicAdd(&a[DynCorr::kFixedQ + 2 * m + 1], (double)(x0.x * x0.x + x0.y * x0.y + x0.z * x0.z));
        }
    }
}

}  // namespace

void DynCorr::allocate(const DynCorrSpec& spec, int natoms, const std::vector<int>& mol_ids) {
    if (spec.origin_every < 1 || spec.norigins < 1)
        throw std::invalid_argument("dyn_corr: origin_every and norigins must be >= 1");
    if (spec.nbins < 1 || spec.nbins > kMaxBins) throw std::invalid_argument("dyn_corr: nbins in [1, 64]");
    if ((int)spec.modes.size() > kMaxModes) throw std::invalid_argument("dyn_corr: at most 8 Rouse modes");
    free();
    spec_ = spec;
    natoms_ = natoms;
    nmodes_ = (int)spec.modes.size();
    nq_ = kFixedQ + 2 * nmodes_;
    const long long max_lag = (long long)spec.norigins * spec.origin_every;
    lags_ = {0};
    for (double x = 1.0; x < max_lag; x *= std::pow(2.0, 0.25)) {
        int l = (int)std::llround(x);
        if (l > lags_.back() && l < max_lag) lags_.push_back(l);
    }
    nlags_ = (int)lags_.size();
    origin_step_.assign(spec.norigins, -1);
    // chains (CSR in tag space, atoms of a molecule in tag order)
    std::vector<int> offsets{0}, order;
    if (!mol_ids.empty()) {
        order.resize(natoms);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return mol_ids[a] < mol_ids[b]; });
        for (int k = 1; k < natoms; k++)
            if (mol_ids[order[k]] != mol_ids[order[k - 1]]) offsets.push_back(k);
        offsets.push_back(natoms);
    }
    nchains_ = mol_ids.empty() ? 0 : (int)offsets.size() - 1;
    if (nchains_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_mol_offsets_, offsets.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_mol_offsets_, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_mol_members_, natoms * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_mol_members_, order.data(), natoms * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (nmodes_ > 0) {
        std::vector<float> m(spec.modes.begin(), spec.modes.end());
        CUDA_CHECK(cudaMalloc(&d_modes_, nmodes_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_modes_, m.data(), nmodes_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    bytes_bpos_ = (size_t)spec.norigins * natoms * sizeof(float4);
    bytes_bimg_ = (size_t)spec.norigins * natoms * sizeof(int4);
    bytes_cdata_ = (size_t)spec.norigins * std::max(nchains_, 1) * (2 + nmodes_) * sizeof(float4);
    bytes_acc_ = (size_t)nlags_ * spec.nbins * nq_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_bpos_, bytes_bpos_));
    CUDA_CHECK(cudaMalloc(&d_bimg_, bytes_bimg_));
    CUDA_CHECK(cudaMalloc(&d_cdata_, bytes_cdata_));
    CUDA_CHECK(cudaMalloc(&d_acc_, bytes_acc_));
    CUDA_CHECK(cudaMemset(d_acc_, 0, bytes_acc_));
    last_step_ = -1;
}

void DynCorr::free() {
    for (void* p : {(void*)d_lags_, (void*)d_mol_offsets_, (void*)d_mol_members_, (void*)d_modes_,
                    (void*)d_bpos_, (void*)d_bimg_, (void*)d_cdata_, (void*)d_acc_, (void*)d_snap_})
        if (p) CUDA_CHECK(cudaFree(p));
    d_lags_ = d_mol_offsets_ = d_mol_members_ = nullptr;
    d_modes_ = nullptr;
    d_bpos_ = nullptr;
    d_bimg_ = nullptr;
    d_cdata_ = nullptr;
    d_acc_ = nullptr;
    d_snap_ = nullptr;
}

void DynCorr::on_step(long long step, const float4* pos, const int* image, const int* tag,
                      const int* rank, const Box* d_box, const float* d_center, cudaStream_t stream) {
    if (!d_acc_ || step == last_step_) return;
    last_step_ = step;
    Geometry g{natoms_, spec_.nbins, spec_.axis, spec_.film ? 1 : 0, nq_, nchains_, nmodes_,
               spec_.bin_width, spec_.q};
    const int K = spec_.norigins;
    std::vector<std::pair<int, int>> due;   // (slot, lag index)
    if (step % spec_.origin_every == 0) {
        const int slot = (int)((step / spec_.origin_every) % K);
        origin_step_[slot] = step;
        capture_beads<<<div_ceil(natoms_, 256), 256, 0, stream>>>(pos, image, tag, d_box, d_center, g,
                                                                  slot, d_bpos_, d_bimg_);
        if (nchains_ > 0) {
            Pairs none{};
            chains_kernel<<<div_ceil(nchains_ * 32, 256), 256, 0, stream>>>(
                pos, image, rank, d_box, d_center, g, d_mol_offsets_, d_mol_members_, d_modes_, slot,
                none, d_cdata_, d_acc_);
        }
    }
    for (int s = 0; s < K; s++) {
        if (origin_step_[s] < 0) continue;
        const long long lag = step - origin_step_[s];
        auto it = std::lower_bound(lags_.begin(), lags_.end(), (int)std::min<long long>(lag, 1ll << 30));
        if (it != lags_.end() && *it == lag) due.push_back({s, (int)(it - lags_.begin())});
    }
    for (size_t k0 = 0; k0 < due.size(); k0 += kMaxPairs) {
        Pairs p{};
        p.n = (int)std::min<size_t>(kMaxPairs, due.size() - k0);
        for (int j = 0; j < p.n; j++) {
            p.slot[j] = due[k0 + j].first;
            p.lag[j] = due[k0 + j].second;
        }
        eval_beads<<<div_ceil(natoms_, 256), 256, 0, stream>>>(pos, image, tag, d_box, d_center, g, p,
                                                               d_bpos_, d_bimg_, d_acc_);
        if (nchains_ > 0)
            chains_kernel<<<div_ceil(nchains_ * 32, 256), 256, 0, stream>>>(
                pos, image, rank, d_box, d_center, g, d_mol_offsets_, d_mol_members_, d_modes_, -1, p,
                d_cdata_, d_acc_);
    }
    CUDA_CHECK(cudaGetLastError());
}

void DynCorr::save_snapshot(cudaStream_t stream) {
    if (!d_acc_) return;
    const size_t total = bytes_bpos_ + bytes_bimg_ + bytes_cdata_ + bytes_acc_;
    if (!d_snap_) CUDA_CHECK(cudaMalloc(&d_snap_, total));
    char* p = d_snap_;
    const std::pair<void*, size_t> parts[4] = {{d_bpos_, bytes_bpos_}, {d_bimg_, bytes_bimg_},
                                               {d_cdata_, bytes_cdata_}, {d_acc_, bytes_acc_}};
    for (const auto& part : parts) {
        CUDA_CHECK(cudaMemcpyAsync(p, part.first, part.second, cudaMemcpyDeviceToDevice, stream));
        p += part.second;
    }
    origin_step_snap_ = origin_step_;
}

void DynCorr::restore_snapshot(cudaStream_t stream) {
    if (!d_acc_ || !d_snap_) return;
    char* p = d_snap_;
    const std::pair<void*, size_t> parts[4] = {{d_bpos_, bytes_bpos_}, {d_bimg_, bytes_bimg_},
                                               {d_cdata_, bytes_cdata_}, {d_acc_, bytes_acc_}};
    for (const auto& part : parts) {
        CUDA_CHECK(cudaMemcpyAsync(part.first, p, part.second, cudaMemcpyDeviceToDevice, stream));
        p += part.second;
    }
    origin_step_ = origin_step_snap_;
    last_step_ = -1;
}

DynCorr::Results DynCorr::results() const {
    Results r;
    r.lags.assign(lags_.begin(), lags_.end());
    r.nbins = spec_.nbins;
    r.nq = nq_;
    r.modes = spec_.modes;
    r.sums.resize(bytes_acc_ / sizeof(double));
    CUDA_CHECK(cudaMemcpy(r.sums.data(), d_acc_, bytes_acc_, cudaMemcpyDeviceToHost));
    return r;
}

void DynCorr::write(const std::string& path, double dt) const {
    Results r = results();
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("cannot open " + path);
    fprintf(fp, "# dynamic correlations: origins every %d steps, %d origins, q %g, %d bins",
            spec_.origin_every, spec_.norigins, spec_.q, spec_.nbins);
    if (spec_.film) fprintf(fp, " of width %g about the film reference plane (axis %d)", spec_.bin_width, spec_.axis);
    fprintf(fp, "\n# per bin: lag_steps time msd_par msd_perp fs_par fs_perp n_bead_samples "
                "rmsd_par rmsd_perp rfs_par rfs_perp n_resident com_par com_perp P_t n_chain_samples");
    for (int m : r.modes) fprintf(fp, " rouse_%d", m);
    fprintf(fp, "\n# (r*: beads found in their origin bin at every evaluated time; P_t and rouse_p "
                "normalised by their own origin values)\n");
    auto at = [&](int l, int b, int q) { return r.sums[((size_t)l * r.nbins + b) * r.nq + q]; };
    for (int b = 0; b < r.nbins; b++) {
        if (spec_.film) {
            double lo = (b - 0.5 * r.nbins) * spec_.bin_width, hi = lo + spec_.bin_width;
            fprintf(fp, "\n# bin %d: depth [%g, %g)%s\n", b, lo, hi,
                    b == 0 || b == r.nbins - 1 ? " (outer bin, unbounded)" : "");
        } else {
            fprintf(fp, "\n# bin 0: all\n");
        }
        for (size_t l = 0; l < r.lags.size(); l++) {
            const double nb = at(l, b, N_BEAD), nr = at(l, b, N_RES), nc = at(l, b, N_CHAIN);
            if (nb == 0 && nc == 0) continue;
            auto mean = [](double s, double n) { return n > 0 ? s / n : NAN; };
            fprintf(fp, "%lld %.6g %.8g %.8g %.8g %.8g %.0f %.8g %.8g %.8g %.8g %.0f %.8g %.8g %.8g %.0f",
                    r.lags[l], r.lags[l] * dt, mean(at(l, b, MSD_PAR), nb), mean(at(l, b, MSD_PERP), nb),
                    mean(at(l, b, FS_PAR), nb), mean(at(l, b, FS_PERP), nb), nb, mean(at(l, b, RMSD_PAR), nr),
                    mean(at(l, b, RMSD_PERP), nr), mean(at(l, b, RFS_PAR), nr), mean(at(l, b, RFS_PERP), nr), nr,
                    mean(at(l, b, COM_PAR), nc), mean(at(l, b, COM_PERP), nc),
                    at(l, b, REE0_SQ) > 0 ? at(l, b, REE_DOT) / at(l, b, REE0_SQ) : NAN, nc);
            for (int m = 0; m < (int)r.modes.size(); m++) {
                double den = at(l, b, kFixedQ + 2 * m + 1);
                fprintf(fp, " %.8g", den > 0 ? at(l, b, kFixedQ + 2 * m) / den : NAN);
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}
