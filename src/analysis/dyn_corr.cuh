#pragma once
#include "../core/types.cuh"
#include <string>
#include <vector>

// On-the-fly multiple-origin correlations, computed on the device.
//
// Time origins every `origin_every` steps; the newest `norigins` are kept,
// and every origin is evaluated at log-spaced lags (factor 2^(1/4), plus 0)
// below norigins * origin_every. Objects are grouped by their depth bin at
// the origin: nbins bins of width `bin_width` centred on the film reference
// plane along the film normal (outer bins unbounded), or one bin without a
// film. Per (lag, bin) the device accumulates
//   beads:  MSD in-plane / along the normal, F_s(q,t) in-plane / normal
//           (mean of cos(q dx) over the in-plane axes, cos(q dz)), and the
//           same restricted to beads found in their origin bin at every
//           evaluated time so far ("residence-conditioned"), with counts;
//   chains: centre-of-mass MSD in-plane / normal, <R(t).R(0)> and <R(0)^2>
//           of the end-to-end vector (P(t) = ratio), and <X_p(t).X_p(0)>,
//           <X_p(0)^2> of Rouse modes X_p = (1/N) sum_n r_n cos(p pi (n-1/2)/N).
// Chains are the molecules (atoms of a chain listed in backbone order).
// Kernels are launched by the host after a step when an origin or a lag
// falls on it (outside the captured step graph).
struct DynCorrSpec {
    int   origin_every = 0;
    int   norigins = 0;
    int   nbins = 1;
    float bin_width = 0.0f;
    int   axis = 2;           // film normal (or z)
    bool  film = false;       // bins relative to the film reference plane
    float q = 7.07f;
    std::vector<int> modes;   // Rouse modes p (chains)
};

class DynCorr {
public:
    // Accumulated quantities per (lag, bin).
    enum BeadQ { MSD_PAR, MSD_PERP, FS_PAR, FS_PERP, N_BEAD,
                 RMSD_PAR, RMSD_PERP, RFS_PAR, RFS_PERP, N_RES, kBeadQ };
    enum ChainQ { COM_PAR = kBeadQ, COM_PERP, REE_DOT, REE0_SQ, N_CHAIN, kFixedQ };

    void allocate(const DynCorrSpec& spec, int natoms, const std::vector<int>& mol_ids);
    void free();
    bool allocated() const { return d_acc_ != nullptr; }

    // After the step `step` (positions of that step): capture an origin
    // and/or evaluate due lags.
    void on_step(long long step, const float4* pos, const int* image, const int* tag,
                 const int* rank, const Box* d_box, const float* d_center, cudaStream_t stream);

    void save_snapshot(cudaStream_t stream);
    void restore_snapshot(cudaStream_t stream);

    struct Results {
        std::vector<long long> lags;   // steps
        int nbins = 0, nq = 0;
        std::vector<double> sums;      // [lag][bin][q]
        std::vector<int> modes;
        int q_index(int mode_slot, bool x0sq) const { return kFixedQ + 2 * mode_slot + (x0sq ? 1 : 0); }
    };
    Results results() const;
    void write(const std::string& path, double dt) const;
    const DynCorrSpec& spec() const { return spec_; }

private:
    DynCorrSpec spec_;
    int natoms_ = 0, nchains_ = 0, nmodes_ = 0, nq_ = 0, nlags_ = 0;
    std::vector<int> lags_;
    std::vector<long long> origin_step_, origin_step_snap_;
    int* d_lags_ = nullptr;
    int* d_mol_offsets_ = nullptr;     // chain CSR in tag space
    int* d_mol_members_ = nullptr;
    float* d_modes_ = nullptr;         // p values
    float4* d_bpos_ = nullptr;         // [K][N] origin wrapped position (w: bin bits)
    int4*   d_bimg_ = nullptr;         // [K][N] origin images (w: residence flag)
    float4* d_cdata_ = nullptr;        // [K][nchains][2 + P] COM (w: bin), Ree, X_p
    double* d_acc_ = nullptr;          // [nlags][nbins][nq]
    char* d_snap_ = nullptr;
    size_t bytes_bpos_ = 0, bytes_bimg_ = 0, bytes_cdata_ = 0, bytes_acc_ = 0;
    long long last_step_ = -1;
};
