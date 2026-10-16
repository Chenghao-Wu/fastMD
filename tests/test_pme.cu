#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/pme.cuh"
#include "force/ewald_recip.cuh"
#include <cmath>
#include <random>
#include <cstdio>

namespace {

struct Recip { std::vector<float4> f; double vir[6]; double energy; };

struct System {
    float L = 3.0f;
    std::vector<float4> pos;
    std::vector<float> q;
};

System make_system(int n, unsigned seed) {
    System s;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, s.L);
    std::uniform_real_distribution<float> uq(-0.8f, 0.8f);
    double qsum = 0;
    for (int i = 0; i < n; i++) {
        s.pos.push_back(make_float4(u(rng), u(rng), u(rng), pack_type_id(0)));
        s.q.push_back(uq(rng));
        qsum += s.q.back();
    }
    for (float& v : s.q) v -= (float)(qsum / n);   // neutral
    return s;
}

// Reciprocal part only, via direct Ewald (kmax) or PME (K, order).
Recip run(const System& s, float alpha, float C, bool pme, int kmax_or_K, int order = 5) {
    const int n = (int)s.pos.size();
    const int np = div_ceil(n, 32) * 32;
    std::vector<float4> pos(np, make_float4(0, 0, 0, pack_type_id(-1)));
    std::copy(s.pos.begin(), s.pos.end(), pos.begin());
    std::vector<float> q(np, 0.0f);
    std::copy(s.q.begin(), s.q.end(), q.begin());
    float4* d_pos = to_device(pos);
    float* d_q = to_device(q);
    std::vector<float4> zero(np, make_float4(0, 0, 0, 0));
    float4* d_f = to_device(zero);
    std::vector<double> vz(6, 0.0);
    double* d_vir = to_device(vz);
    Box* d_box = to_device(std::vector<Box>{cubic_box(s.L)});
    Recip r;
    if (pme) {
        PME p;
        p.allocate(n, kmax_or_K, order, alpha, C);
        p.compute(d_pos, d_q, d_f, d_vir, d_box, 0);
        r.energy = p.last_energy(0);
        p.free();
    } else {
        auto kv = ewald_build_kvectors(kmax_or_K, s.L, alpha);
        int3* d_kv = to_device(kv);
        float* d_sfac;
        CUDA_CHECK(cudaMalloc(&d_sfac, 2 * kv.size() * sizeof(float)));
        launch_ewald_sf_kernel(d_pos, d_q, d_kv, d_sfac, n, (int)kv.size(), s.L);
        launch_ewald_recip_kernel(d_pos, d_q, d_kv, d_sfac, d_f, d_vir, n, (int)kv.size(), C, alpha, s.L);
        CUDA_CHECK(cudaDeviceSynchronize());
        free_device(d_kv);
        CUDA_CHECK(cudaFree(d_sfac));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    r.f = to_host(d_f, n);
    CUDA_CHECK(cudaMemcpy(r.vir, d_vir, sizeof(r.vir), cudaMemcpyDeviceToHost));
    if (!pme) { r.energy = 0; for (auto& v : r.f) r.energy += v.w; }
    free_device(d_pos); free_device(d_q); free_device(d_f); free_device(d_vir); free_device(d_box);
    return r;
}

double rel_rms_force_error(const Recip& a, const Recip& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.f.size(); i++) {
        double dx = a.f[i].x - ref.f[i].x, dy = a.f[i].y - ref.f[i].y, dz = a.f[i].z - ref.f[i].z;
        num += dx * dx + dy * dy + dz * dz;
        den += (double)ref.f[i].x * ref.f[i].x + (double)ref.f[i].y * ref.f[i].y + (double)ref.f[i].z * ref.f[i].z;
    }
    return std::sqrt(num / den);
}

}  // namespace

// SPME reproduces converged direct Ewald (energy, per-atom energies sum,
// forces, all six virial components) and converges with the grid.
TEST(PME, MatchesConvergedEwald) {
    const float alpha = 3.0f, C = 138.9354576f;
    System s = make_system(301, 11);
    Recip ref = run(s, alpha, C, false, 18);   // exp(-(pi*18/(alpha*L))^2) ~ 1e-17
    double vmax = 0;
    for (double v : ref.vir) vmax = std::max(vmax, std::fabs(v));
    double prev_err = 1e30;
    for (int K : {20, 32, 48, 64}) {
        Recip r = run(s, alpha, C, true, K, 5);
        double ferr = rel_rms_force_error(r, ref);
        double eatom = 0;
        for (auto& v : r.f) eatom += v.w;
        std::printf("    K=%2d order 5: rel RMS force err %.2e, energy rel err %.2e, virial xy err %.2e\n",
                    K, ferr, std::fabs(r.energy - ref.energy) / std::fabs(ref.energy),
                    std::fabs(r.vir[1] - ref.vir[1]) / vmax);
        EXPECT_LT(ferr, prev_err) << "K=" << K << ": error must decrease with the grid";
        prev_err = ferr;
        if (K >= 48) {
            EXPECT_LT(ferr, 2e-4) << "K=" << K;
            EXPECT_NEAR(r.energy, ref.energy, 2e-5 * std::fabs(ref.energy)) << "K=" << K;
            EXPECT_NEAR(eatom, r.energy, 1e-4 * std::fabs(r.energy)) << "per-atom energies";
            for (int c = 0; c < 6; c++)
                EXPECT_NEAR(r.vir[c], ref.vir[c], 2e-4 * vmax) << "K=" << K << " virial " << c;
        }
    }
    // higher order at fixed grid is more accurate
    double e5 = rel_rms_force_error(run(s, alpha, C, true, 24, 5), ref);
    double e7 = rel_rms_force_error(run(s, alpha, C, true, 24, 7), ref);
    EXPECT_LT(e7, e5);
}

// Automatic parameters meet the requested relative force tolerance.
TEST(PME, ToleranceDrivenParameters) {
    const float rc = 1.0f, C = 138.9354576f;
    System s = make_system(400, 5);
    for (float tol : {5e-4f, 1e-4f}) {
        float alpha = ewald_alpha_for_tolerance(tol, rc);
        int K = pme_grid_for_tolerance(tol, alpha, s.L);
        int kmax = ewald_kmax_for_tolerance(tol, alpha, s.L);
        Recip ref = run(s, alpha, C, false, 30);
        double ep = rel_rms_force_error(run(s, alpha, C, true, K, 5), ref);
        double ee = rel_rms_force_error(run(s, alpha, C, false, kmax), ref);
        std::printf("    tol %.0e: alpha %.3f, PME K=%d err %.2e, Ewald kmax=%d err %.2e\n",
                    tol, alpha, K, ep, kmax, ee);
        EXPECT_LT(ep, tol) << "PME K=" << K << " tol " << tol;
        EXPECT_LT(ee, tol) << "Ewald kmax=" << kmax << " tol " << tol;
    }
    EXPECT_EQ(pme_fft_friendly(61), 63);
    EXPECT_EQ(pme_fft_friendly(64), 64);
    EXPECT_EQ(pme_fft_friendly(97), 98);
}
