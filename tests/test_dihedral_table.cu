#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/dihedral_table.cuh"
#include "force/dihedral_opls.cuh"
#include <cmath>
#include <random>

// Tabulated dihedrals (dihedral_table), plain and combined bending-torsion (cbt):
// E = U(phi) or E = sin^3(theta1) sin^3(theta2) U(phi), phi IUPAC (cis 0, trans 180 degrees)
// from atan2 (Blondel-Karplus), theta1 = angle 1-2-3, theta2 = angle 2-3-4.

namespace {

const double kPi = 3.14159265358979323846;
const int kPoints = 3601;                       // 0.1 degree spacing over [-180, 180]

double U(double p) { return 1.3 * std::cos(p) + 0.5 * std::sin(2 * p) + 0.8 * std::cos(3 * p) + 2.0 * std::sin(p) + 1.0; }
double dU(double p) { return -1.3 * std::sin(p) + 1.0 * std::cos(2 * p) - 2.4 * std::sin(3 * p) + 2.0 * std::cos(p); }

double opls(double p, const double k[4]) {
    return 0.5 * k[0] * (1 + std::cos(p)) + 0.5 * k[1] * (1 - std::cos(2 * p)) + 0.5 * k[2] * (1 + std::cos(3 * p)) +
           0.5 * k[3] * (1 - std::cos(4 * p));
}
double dopls(double p, const double k[4]) {
    return -0.5 * k[0] * std::sin(p) + k[1] * std::sin(2 * p) - 1.5 * k[2] * std::sin(3 * p) + 2 * k[3] * std::sin(4 * p);
}

template <class E, class D>
std::vector<float4> make_table(E e, D de) {
    std::vector<float4> t(kPoints);
    for (int k = 0; k < kPoints; k++) {
        double x = -kPi + 2 * kPi * k / (kPoints - 1);
        t[k] = make_float4((float)x, (float)(-de(x)), (float)e(x), 0.0f);
    }
    t[kPoints - 1].y = t[0].y;                 // one period: identical first / last rows
    t[kPoints - 1].z = t[0].z;
    return t;
}

TableParams table_params(int flags) {
    TableParams tp;
    tp.rmin = (float)-kPi;
    tp.rmax = (float)kPi;
    tp.dr = (float)(2 * kPi / (kPoints - 1));
    tp.inv_dr = 1.0f / tp.dr;
    tp.npoints = kPoints;
    tp.data_offset = 0;
    tp.flags = flags;
    return tp;
}

struct D3 { double x, y, z; };
D3 sub(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3 cross(D3 a, D3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Host reference: phi (IUPAC, Blondel-Karplus) and the CBT prefactor of one dihedral (no PBC needed here).
void host_angles(const std::vector<D3>& r, int i, double& phi, double& w) {
    D3 F = sub(r[i], r[i + 1]), G = sub(r[i + 1], r[i + 2]), H = sub(r[i + 3], r[i + 2]);
    D3 A = cross(F, G), B = cross(H, G);
    double gn = std::sqrt(dot(G, G));
    phi = std::atan2(dot(cross(B, A), G) / gn, dot(A, B));
    double s1 = std::sqrt(dot(A, A) / (dot(F, F) * dot(G, G)));
    double s2 = std::sqrt(dot(B, B) / (dot(H, H) * dot(G, G)));
    w = s1 * s1 * s1 * s2 * s2 * s2;
}

// Total energy of consecutive dihedrals (0,1,2,3), (4,5,6,7), ... with the analytic U.
double host_energy(const std::vector<D3>& r, bool cbt) {
    double e = 0;
    for (size_t i = 0; i + 3 < r.size(); i += 4) {
        double phi, w;
        host_angles(r, (int)i, phi, w);
        e += (cbt ? w : 1.0) * U(phi);
    }
    return e;
}

// Four-atom units with bond length ~1.5, bond angles 95-150 degrees and random torsions, each placed apart.
std::vector<D3> make_units(int n_units, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ang(95.0, 150.0), tor(-180.0, 180.0);
    std::vector<D3> r;
    for (int u = 0; u < n_units; u++) {
        double t1 = ang(rng) * kPi / 180, t2 = ang(rng) * kPi / 180, ph = tor(rng) * kPi / 180;
        D3 o = {3.0 + 6.0 * (u % 3), 3.0 + 6.0 * (u / 3 % 3), 3.0 + 6.0 * (u / 9)};
        D3 p2 = o, p3 = {o.x + 1.5, o.y, o.z};
        D3 p1 = {o.x + 1.5 * std::cos(t1), o.y + 1.5 * std::sin(t1), o.z};   // bond angle t1 at atom 2
        // atom 4 from bond angle t2 at atom 3 and torsion ph about the 2-3 axis (x); ph = 0 is cis
        D3 p4 = {p3.x - 1.5 * std::cos(t2), p3.y + 1.5 * std::sin(t2) * std::cos(ph), p3.z + 1.5 * std::sin(t2) * std::sin(ph)};
        r.insert(r.end(), {p1, p2, p3, p4});
    }
    return r;
}

struct Out { std::vector<float4> f; double vir[6]; double e; };

// Run the scatter kernel on dihedrals (4k, 4k+1, 4k+2, 4k+3); table: plain or cbt.
Out run_table(const std::vector<D3>& r, const std::vector<float4>& table, int flags, float L = 30.0f) {
    const int n = (int)r.size(), nd = n / 4;
    const int np = div_ceil(n, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> pos(np, make_float4(0, 0, 0, pack_type_id(-1)));
    for (int i = 0; i < n; i++) pos[i] = make_float4((float)r[i].x, (float)r[i].y, (float)r[i].z, pack_type_id(0));
    std::vector<int4> dih;
    for (int d = 0; d < nd; d++) dih.push_back(make_int4(4 * d, 4 * d + 1, 4 * d + 2, 4 * d + 3));
    std::vector<int> dt(nd, 0), tidx = {0};
    std::vector<TableParams> tp = {table_params(flags)};
    float4* d_pos = to_device(pos);
    float4* d_force = to_device(std::vector<float4>(np, make_float4(0, 0, 0, 0)));
    double* d_vir = to_device(std::vector<double>(6, 0.0));
    int4* d_dih = to_device(dih);
    int* d_dt = to_device(dt);
    int* d_tidx = to_device(tidx);
    TableParams* d_tp = to_device(tp);
    float4* d_tab = to_device(table);
    launch_dihedral_table_kernel(d_pos, d_force, d_vir, d_dih, d_dt, d_tidx, d_tp, d_tab, nd, cubic_box(L));
    CUDA_CHECK(cudaDeviceSynchronize());
    Out o;
    o.f = to_host(d_force, n);
    CUDA_CHECK(cudaMemcpy(o.vir, d_vir, sizeof(o.vir), cudaMemcpyDeviceToHost));
    o.e = 0;
    for (auto& v : o.f) o.e += v.w;
    for (void* p : {(void*)d_pos, (void*)d_force, (void*)d_vir, (void*)d_dih, (void*)d_dt, (void*)d_tidx,
                    (void*)d_tp, (void*)d_tab})
        CUDA_CHECK(cudaFree(p));
    return o;
}

double fmax_of(const std::vector<float4>& f) {
    double m = 0;
    for (auto& v : f) m = std::max({m, (double)std::fabs(v.x), (double)std::fabs(v.y), (double)std::fabs(v.z)});
    return m;
}

// -dE/dr by central differences of the analytic host energy.
std::vector<D3> host_forces(std::vector<D3> r, bool cbt) {
    const double h = 1e-6;
    std::vector<D3> f(r.size());
    for (size_t i = 0; i < r.size(); i++)
        for (int c = 0; c < 3; c++) {
            double* x = c == 0 ? &r[i].x : c == 1 ? &r[i].y : &r[i].z;
            double x0 = *x;
            *x = x0 + h; double ep = host_energy(r, cbt);
            *x = x0 - h; double em = host_energy(r, cbt);
            *x = x0;
            (c == 0 ? f[i].x : c == 1 ? f[i].y : f[i].z) = -(ep - em) / (2 * h);
        }
    return f;
}

void expect_forces(const Out& o, const std::vector<D3>& ref, double rel, const char* what) {
    double m = 0;
    for (auto& v : ref) m = std::max({m, std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)});
    for (size_t i = 0; i < ref.size(); i++) {
        EXPECT_NEAR(o.f[i].x, ref[i].x, rel * m + 1e-6) << what << " atom " << i;
        EXPECT_NEAR(o.f[i].y, ref[i].y, rel * m + 1e-6) << what << " atom " << i;
        EXPECT_NEAR(o.f[i].z, ref[i].z, rel * m + 1e-6) << what << " atom " << i;
    }
}

}  // namespace

// A table of the OPLS function reproduces the dihedral_opls kernel (energy and forces).
TEST(DihedralTable, ReproducesOPLSKernel) {
    const double k[4] = {3.1, -0.6, 5.9, 0.2};
    auto r = make_units(12, 3);
    auto tab = make_table([&](double p) { return opls(p, k); }, [&](double p) { return dopls(p, k); });
    Out t = run_table(r, tab, 0);

    const int n = (int)r.size(), nd = n / 4, np = div_ceil(n, TILE_SIZE) * TILE_SIZE;
    std::vector<float4> pos(np, make_float4(0, 0, 0, pack_type_id(-1)));
    for (int i = 0; i < n; i++) pos[i] = make_float4((float)r[i].x, (float)r[i].y, (float)r[i].z, pack_type_id(0));
    std::vector<int4> dih;
    for (int d = 0; d < nd; d++) dih.push_back(make_int4(4 * d, 4 * d + 1, 4 * d + 2, 4 * d + 3));
    std::vector<int> dt(nd, 0);
    std::vector<OPLSDihedralParams> prm = {{3.1f, -0.6f, 5.9f, 0.2f}};
    float4* d_pos = to_device(pos);
    float4* d_force = to_device(std::vector<float4>(np, make_float4(0, 0, 0, 0)));
    double* d_vir = to_device(std::vector<double>(6, 0.0));
    int4* d_dih = to_device(dih);
    int* d_dt = to_device(dt);
    OPLSDihedralParams* d_prm = to_device(prm);
    launch_dihedral_opls_kernel(d_pos, d_force, d_vir, d_dih, d_dt, d_prm, nd, cubic_box(30.0f));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto fo = to_host(d_force, n);
    double eo = 0, m = fmax_of(fo);
    for (int i = 0; i < n; i++) {
        eo += fo[i].w;
        EXPECT_NEAR(t.f[i].x, fo[i].x, 2e-3 * m) << "atom " << i;
        EXPECT_NEAR(t.f[i].y, fo[i].y, 2e-3 * m) << "atom " << i;
        EXPECT_NEAR(t.f[i].z, fo[i].z, 2e-3 * m) << "atom " << i;
    }
    EXPECT_NEAR(t.e, eo, 1e-4 * std::fabs(eo) + 1e-4);
    for (void* p : {(void*)d_pos, (void*)d_force, (void*)d_vir, (void*)d_dih, (void*)d_dt, (void*)d_prm})
        CUDA_CHECK(cudaFree(p));
}

// phi follows IUPAC (cis 0, trans 180 degrees) with the Blondel-Karplus sign: an asymmetric table returns
// U(phi) for the constructed angle and not U(-phi).
TEST(DihedralTable, SignConventionIsIUPAC) {
    auto tab = make_table(U, dU);
    for (double deg : {-150.0, -60.0, 0.0, 45.0, 120.0, 180.0}) {
        double p = deg * kPi / 180;
        std::vector<D3> r = {{3.0, 4.0, 3.0}, {3.0, 3.0, 3.0}, {4.5, 3.0, 3.0},
                             {4.5, 3.0 + std::cos(p), 3.0 + std::sin(p)}};
        double phi, w;
        host_angles(r, 0, phi, w);
        EXPECT_NEAR(phi, std::remainder(p, 2 * kPi), 1e-9) << deg;
        Out o = run_table(r, tab, 0);
        EXPECT_NEAR(o.e, U(p), 2e-4) << deg << " degrees";
        if (std::fabs(std::sin(p)) > 0.1) EXPECT_GT(std::fabs(o.e - U(-p)), 0.1) << deg;
    }
}

TEST(DihedralTable, PlainForcesAreEnergyGradient) {
    auto r = make_units(9, 11);
    Out o = run_table(r, make_table(U, dU), 0);
    EXPECT_NEAR(o.e, host_energy(r, false), 1e-4 * std::fabs(host_energy(r, false)) + 1e-4);
    expect_forces(o, host_forces(r, false), 2e-3, "plain");
}

TEST(DihedralTable, CbtEnergyAndForces) {
    auto r = make_units(9, 13);
    Out o = run_table(r, make_table(U, dU), TABLE_DIHEDRAL_CBT);
    double e = host_energy(r, true);
    EXPECT_NEAR(o.e, e, 1e-4 * std::fabs(e) + 1e-4);
    EXPECT_GT(std::fabs(e - host_energy(r, false)), 0.5);                 // the prefactor matters here
    expect_forces(o, host_forces(r, true), 2e-3, "cbt");
}

// A straight bond angle: the plain torsion gradient diverges as 1/sin(theta), CBT stays finite and correct.
TEST(DihedralTable, CbtIsFiniteForStraightAngles) {
    for (double th_deg : {180.0, 179.9, 179.0}) {
        double t1 = th_deg * kPi / 180, ph = 70.0 * kPi / 180, t2 = 115.0 * kPi / 180;
        std::vector<D3> r = {{3.0 + 1.5 * std::cos(t1), 3.0 + 1.5 * std::sin(t1), 3.0}, {3.0, 3.0, 3.0}, {4.5, 3.0, 3.0},
                             {4.5 - 1.5 * std::cos(t2), 3.0 + 1.5 * std::sin(t2) * std::cos(ph),
                              3.0 + 1.5 * std::sin(t2) * std::sin(ph)}};
        Out o = run_table(r, make_table(U, dU), TABLE_DIHEDRAL_CBT);
        for (auto& v : o.f) {
            ASSERT_TRUE(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) << th_deg;
        }
        EXPECT_LT(fmax_of(o.f), 1.0) << th_deg;                          // ~U s^2 / |r|: tiny near 180
        expect_forces(o, host_forces(r, true), 2e-3, "straight");
        if (th_deg < 180.0) {
            Out p = run_table(r, make_table(U, dU), 0);
            EXPECT_GT(fmax_of(p.f), 10.0 * fmax_of(o.f)) << th_deg;      // the singularity CBT removes
        }
    }
}

// Virial: W_ab = sum r_a F_b equals -dE/d(eps) for r_b -> r_b + eps r_a (homogeneous deformation).
TEST(DihedralTable, VirialMatchesDeformationDerivative) {
    for (int flags : {0, TABLE_DIHEDRAL_CBT}) {
        auto r = make_units(9, 17);
        Out o = run_table(r, make_table(U, dU), flags);
        const int comp[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        double wmax = 0;
        for (double v : o.vir) wmax = std::max(wmax, std::fabs(v));
        for (int c = 0; c < 6; c++) {
            const int a = comp[c][0], b = comp[c][1];
            auto deform = [&](double eps) {
                std::vector<D3> s = r;
                for (auto& p : s) {
                    double ra = a == 0 ? p.x : a == 1 ? p.y : p.z;
                    (b == 0 ? p.x : b == 1 ? p.y : p.z) += eps * ra;
                }
                return host_energy(s, flags != 0);
            };
            const double h = 1e-6;
            double w = -(deform(h) - deform(-h)) / (2 * h);
            EXPECT_NEAR(o.vir[c], w, 2e-3 * wmax + 1e-5) << "flags " << flags << " component " << c;
        }
    }
}
