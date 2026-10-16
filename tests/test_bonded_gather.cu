#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "core/topology.cuh"
#include "force/bonded_gather.cuh"
#include "force/fene.cuh"
#include "force/bond_harmonic.cuh"
#include "force/bond_table.cuh"
#include "force/angle.cuh"
#include "force/angle_table.cuh"
#include "force/dihedral_opls.cuh"
#include <cmath>
#include <numeric>
#include <random>

namespace {

struct Chains {
    float L = 9.0f;
    std::vector<float4> pos;
    std::vector<int2> bonds;
    std::vector<int> bond_types;
    std::vector<int4> angles;       // .w = type
    std::vector<int4> dihedrals;
    std::vector<int> dihedral_types;
};

// Random-walk chains (bond length ~0.97, wrapped into the box), 2 types of
// each bonded term. Lengths 1..33 give ends, singletons and non-multiples of 32.
Chains make_chains(unsigned seed) {
    Chains c;
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, c.L);
    const int lengths[] = {33, 1, 7, 20, 2, 31, 3, 16};
    for (int len : lengths) {
        int base = (int)c.pos.size();
        float p[3] = {u(rng), u(rng), u(rng)};
        for (int k = 0; k < len; k++) {
            if (k > 0) {
                float v[3] = {g(rng), g(rng), g(rng)};
                float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                for (int d = 0; d < 3; d++) p[d] += 0.97f * v[d] / n;
            }
            float w[3];
            for (int d = 0; d < 3; d++) w[d] = p[d] - c.L * std::floor(p[d] / c.L);
            c.pos.push_back(make_float4(w[0], w[1], w[2], pack_type_id(0)));
            int i = base + k;
            if (k >= 1) { c.bonds.push_back(make_int2(i - 1, i)); c.bond_types.push_back(k % 2); }
            if (k >= 2) c.angles.push_back(make_int4(i - 2, i - 1, i, k % 2));
            if (k >= 3) { c.dihedrals.push_back(make_int4(i - 3, i - 2, i - 1, i)); c.dihedral_types.push_back(k % 2); }
        }
    }
    return c;
}

std::vector<float4> make_table(double lo, double hi, int n, bool angle) {
    std::vector<float4> t(n);
    for (int k = 0; k < n; k++) {
        double x = lo + (hi - lo) * k / (n - 1);
        double e, f;
        if (angle) { e = 3.0 * (x - 1.9) * (x - 1.9); f = -6.0 * (x - 1.9); }        // per radian
        else { e = 40.0 * (x - 1.0) * (x - 1.0); f = -80.0 * (x - 1.0) / x; }          // F/r
        t[k] = make_float4((float)x, (float)f, (float)e, 0.0f);
    }
    return t;
}

struct Result { std::vector<float4> f; double vir[6]; };

// Run either the scatter kernels or the gather kernel. order[k] = tag of
// the particle stored at index k (a permutation); results are returned in
// tag order.
Result run(const Chains& c, int bond_style, int angle_style, bool gather,
           const std::vector<int>& order, int tpa = 1) {
    const int n = (int)c.pos.size();
    const int np = div_ceil(n, 32) * 32;
    std::vector<int> rank(np), tag(np);
    std::iota(tag.begin(), tag.end(), 0);
    for (int k = 0; k < n; k++) tag[k] = order[k];
    for (int k = 0; k < np; k++) rank[tag[k]] = k;
    std::vector<float4> pos(np, make_float4(0, 0, 0, pack_type_id(-1)));
    for (int k = 0; k < n; k++) pos[k] = c.pos[tag[k]];

    DeviceTopology topo;
    topo.upload(n, c.bonds, c.bond_types, c.angles, c.dihedrals, c.dihedral_types, {},
                {}, {}, {}, {});
    int* d_rank = to_device(rank);
    int* d_tag = to_device(tag);
    topo.remap(d_rank);

    std::vector<FENEParams> fene = {{30.0f, 1.5f, 1.0f, 1.0f}, {25.0f, 1.6f, 0.8f, 0.95f}};
    std::vector<BondHarmonicParams> harm = {{100.0f, 1.0f}, {80.0f, 0.95f}};
    std::vector<AngleParams> ang = {{5.0f, 2.0f}, {8.0f, 1.8f}};
    std::vector<OPLSDihedralParams> opls = {{1.3f, -0.05f, 0.6f, 0.0f}, {0.0f, 0.0f, 0.318f, 0.1f}};
    auto btab = make_table(0.3, 2.5, 801, false);
    auto atab = make_table(0.0, M_PI, 721, true);
    TableParams tpb{0.3f, 2.5f, (float)(2.2 / 800), (float)(800 / 2.2), 801, 0};
    TableParams tpang{0.0f, (float)M_PI, (float)(M_PI / 720), (float)(720 / M_PI), 721, 0};
    std::vector<int> tidx = {0, 0};

    float4* d_pos = to_device(pos);
    std::vector<float4> zero(np, make_float4(0, 0, 0, 0));
    float4* d_force = to_device(zero);
    std::vector<double> vz(6, 0.0);
    double* d_vir = to_device(vz);
    auto* d_fene = to_device(fene);
    auto* d_harm = to_device(harm);
    auto* d_ang = to_device(ang);
    auto* d_opls = to_device(opls);
    auto* d_btab = to_device(btab);
    auto* d_atab = to_device(atab);
    auto* d_tpb = to_device(std::vector<TableParams>{tpb});
    auto* d_tpa = to_device(std::vector<TableParams>{tpang});
    int* d_tidx = to_device(tidx);
    const float L = c.L, iL = 1.0f / c.L;

    if (gather) {
        BondedParams bp;
        bp.bond_style = bond_style;
        bp.bond_type = topo.bond_type;
        bp.fene = d_fene; bp.harmonic = d_harm;
        bp.bond_table_idx = d_tidx; bp.bond_table_params = d_tpb; bp.bond_table_data = d_btab;
        bp.angle_style = angle_style;
        bp.angle = d_ang;
        bp.angle_table_idx = d_tidx; bp.angle_table_params = d_tpa; bp.angle_table_data = d_atab;
        bp.dihedral_style = 1;
        bp.dihedral_type = topo.dihedral_type;
        bp.opls = d_opls;
        launch_bonded_gather_kernel(d_pos, d_force, d_vir, d_tag, topo.term_offsets, topo.terms,
                                    topo.bonds, topo.angles, topo.dihedrals, bp, n, L, iL, 0, tpa);
    } else {
        if (bond_style == 1)
            launch_fene_kernel(d_pos, d_force, d_vir, topo.bonds, topo.bond_type, d_fene, topo.nbonds, 2, L, iL);
        else if (bond_style == 2)
            launch_bond_harmonic_kernel(d_pos, d_force, d_vir, topo.bonds, topo.bond_type, d_harm, topo.nbonds, L, iL);
        else
            launch_bond_table_kernel(d_pos, d_force, d_vir, topo.bonds, topo.bond_type, d_tidx, d_tpb, d_btab,
                                     topo.nbonds, L, iL);
        if (angle_style == 1)
            launch_angle_kernel(d_pos, d_force, d_vir, topo.angles, d_ang, topo.nangles, L, iL);
        else
            launch_angle_table_kernel(d_pos, d_force, d_vir, topo.angles, d_tidx, d_tpa, d_atab,
                                      topo.nangles, L, iL);
        launch_dihedral_opls_kernel(d_pos, d_force, d_vir, topo.dihedrals, topo.dihedral_type,
                                    d_opls, topo.ndihedrals, L, iL);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto f = to_host(d_force, n);
    Result r;
    r.f.resize(n);
    for (int k = 0; k < n; k++) r.f[tag[k]] = f[k];
    CUDA_CHECK(cudaMemcpy(r.vir, d_vir, sizeof(r.vir), cudaMemcpyDeviceToHost));

    topo.free();
    for (void* p : {(void*)d_rank, (void*)d_tag, (void*)d_pos, (void*)d_force, (void*)d_vir,
                    (void*)d_fene, (void*)d_harm, (void*)d_ang, (void*)d_opls, (void*)d_btab,
                    (void*)d_atab, (void*)d_tpb, (void*)d_tpa, (void*)d_tidx})
        CUDA_CHECK(cudaFree(p));
    return r;
}

void expect_same(const Result& a, const Result& b, const char* what) {
    double fmax = 0;
    for (auto& v : a.f) fmax = std::max({fmax, (double)std::fabs(v.x), (double)std::fabs(v.y), (double)std::fabs(v.z)});
    double ea = 0, eb = 0;
    for (size_t i = 0; i < a.f.size(); i++) {
        EXPECT_NEAR(a.f[i].x, b.f[i].x, 2e-5 * fmax) << what << " atom " << i;
        EXPECT_NEAR(a.f[i].y, b.f[i].y, 2e-5 * fmax) << what << " atom " << i;
        EXPECT_NEAR(a.f[i].z, b.f[i].z, 2e-5 * fmax) << what << " atom " << i;
        ea += a.f[i].w; eb += b.f[i].w;
    }
    EXPECT_NEAR(ea, eb, 1e-5 * std::fabs(ea) + 1e-6) << what << " energy";
    double vmax = 0;
    for (double v : a.vir) vmax = std::max(vmax, std::fabs(v));
    for (int k = 0; k < 6; k++) EXPECT_NEAR(a.vir[k], b.vir[k], 1e-5 * vmax) << what << " virial " << k;
}

}  // namespace

TEST(BondedGather, MatchesScatterKernelsForAllStyles) {
    Chains c = make_chains(42);
    std::vector<int> identity(c.pos.size());
    std::iota(identity.begin(), identity.end(), 0);
    for (int bs : {1, 2, 3})
        for (int as : {1, 2}) {
            std::string what = "bond style " + std::to_string(bs) + " angle style " + std::to_string(as);
            Result ref = run(c, bs, as, false, identity);
            for (int tpa : {1, 2, 4, 8})
                expect_same(ref, run(c, bs, as, true, identity, tpa),
                            (what + " tpa " + std::to_string(tpa)).c_str());
        }
}

// Particles stored in a shuffled order (as after spatial sorting): the
// gather kernel reads its rows by tag and the remapped index-space lists.
TEST(BondedGather, PermutationInvariant) {
    Chains c = make_chains(7);
    std::vector<int> identity(c.pos.size()), shuffled(c.pos.size());
    std::iota(identity.begin(), identity.end(), 0);
    shuffled = identity;
    std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(3));
    expect_same(run(c, 1, 1, false, identity), run(c, 1, 1, true, shuffled), "FENE+angle shuffled");
    expect_same(run(c, 2, 2, false, identity), run(c, 2, 2, true, shuffled), "harmonic+table shuffled");
}
