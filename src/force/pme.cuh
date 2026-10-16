#pragma once
#include "../core/types.cuh"
#include <cufft.h>
#include <vector>

// Smooth particle-mesh Ewald (Essmann et al., J. Chem. Phys. 103, 8577 (1995))
// for the reciprocal-space Coulomb sum in a cubic box: B-spline charge
// spreading of order `order` on a K^3 grid, cuFFT R2C/C2R, and analytical
// differentiation of the B-splines for forces. Adds the reciprocal force,
// per-atom energy (E = sum_i 0.5 q_i C psi(r_i)) and the full reciprocal
// virial; the real-space erfc term, the excluded/1-4 corrections and the
// self energy are shared with direct Ewald (they depend only on alpha).
struct PME {
    int K = 0;            // grid points per side
    int order = 5;        // B-spline order (interpolation stencil per side)
    int natoms = 0;
    float alpha = 0.0f;
    float coul_const = 1.0f;
    float* d_grid = nullptr;           // [K^3] real charge grid / convolved potential
    cufftComplex* d_cgrid = nullptr;   // [K * K * (K/2+1)]
    float* d_bsp_mod = nullptr;        // [K] |b(m)|^-2 B-spline moduli (same for x, y, z)
    double* d_energy = nullptr;        // [1] total reciprocal energy (diagnostic)
    cufftHandle plan_r2c = 0, plan_c2r = 0;

    void allocate(int natoms, int K, int order, float alpha, float coul_const);
    void free();

    // Accumulate reciprocal forces/energies into force[] and virial[]; the box
    // is read from the device box d_box (cubic: Ewald/PME require a cubic box).
    void compute(const float4* pos, const float* charges, float4* force, double* virial,
                 const Box* d_box, cudaStream_t stream);
    double last_energy(cudaStream_t stream) const;   // synchronous read of d_energy
};

// Smallest K >= n whose prime factors are 2, 3, 5 or 7 (fast cuFFT sizes).
int pme_fft_friendly(int n);

// Parameters for a target relative RMS force error `tol` (as in OpenMM):
//   alpha = sqrt(-ln(2 tol)) / rc
//   PME:   K = ceil(2 alpha L / (3 tol^(1/5)))    (rounded to an FFT size)
//   Ewald: kmax = ceil(alpha L sqrt(-ln tol) / pi)
float ewald_alpha_for_tolerance(float tol, float rc);
int   pme_grid_for_tolerance(float tol, float alpha, float L);
int   ewald_kmax_for_tolerance(float tol, float alpha, float L);
