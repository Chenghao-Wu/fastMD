#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "core/system.cuh"
#include "analysis/thermo.cuh"
#include <fstream>
#include <map>
#include <sstream>

TEST(Thermo, KineticEnergyAndTemperature) {
    const int N = 96;
    const float L = 10.0f;
    const int np = div_ceil(N, TILE_SIZE) * TILE_SIZE;

    std::vector<float4> h_vel(np, make_float4(0,0,0,0));
    for (int i = 0; i < N; i++) {
        h_vel[i] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    }

    float4* d_vel = to_device(h_vel);

    std::vector<float4> h_pos(N, make_float4(0,0,0, pack_type_id(0)));
    float4* d_pos = to_device(h_pos);

    float h_masses[16];
    for (int i = 0; i < 16; i++) h_masses[i] = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses, 16 * sizeof(float)));

    float4* d_force;
    CUDA_CHECK(cudaMalloc(&d_force, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(d_force, 0, np * sizeof(float4)));
    double* d_virial;
    CUDA_CHECK(cudaMalloc(&d_virial, 6 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_virial, 0, 6 * sizeof(double)));

    ThermoBuffers bufs;
    bufs.allocate();
    ThermoOutput output;
    compute_thermo(d_vel, d_pos, d_force, d_virial, N, L, &output, bufs, 0, nullptr);

    EXPECT_NEAR(output.kinetic_energy, 0.5f, 0.01f);
    EXPECT_NEAR(output.temperature, 1.0f/3.0f, 0.001f);

    bufs.free();
    free_device(d_vel);
    free_device(d_force);
    free_device(d_virial);
}

// The thermo file header promises Pxx Pyy Pzz Pxy Pxz Pyz; the device virial
// is stored xx xy xz yy yz zz. A synthetic tensor with distinct components
// (trace/3 = 6) must come out in the labelled columns.
TEST(Thermo, StressColumnsMatchHeader) {
    const int N = 32;
    const float L = 2.0f;
    const float V = L * L * L;
    std::vector<float4> zeros(N, make_float4(0, 0, 0, 0));
    std::vector<float4> h_pos(N, make_float4(0, 0, 0, pack_type_id(0)));
    float h_masses[16];
    for (float& m : h_masses) m = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses, sizeof(h_masses)));

    // internal order: xx xy xz yy yz zz
    std::vector<double> vir = {5.0 * V, 1.0 * V, 2.0 * V, 6.0 * V, 3.0 * V, 7.0 * V};
    float4* d_vel = to_device(zeros);
    float4* d_force = to_device(zeros);
    float4* d_pos = to_device(h_pos);
    double* d_virial = to_device(vir);

    const char* path = "thermo_columns_test.dat";
    ThermoBuffers bufs;
    bufs.allocate();
    bufs.open_file(path);
    ThermoOutput out;
    compute_thermo(d_vel, d_pos, d_force, d_virial, N, L, &out, bufs, 7, bufs.fp);
    bufs.close_file();
    bufs.free();

    std::ifstream f(path);
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    std::istringstream hs(header), rs(row);
    std::vector<std::string> names;
    for (std::string w; hs >> w;) if (w != "#") names.push_back(w);
    std::vector<double> vals;
    for (double v; rs >> v;) vals.push_back(v);
    ASSERT_EQ(names.size(), vals.size());
    std::map<std::string, double> col;
    for (size_t k = 0; k < names.size(); k++) col[names[k]] = vals[k];
    EXPECT_NEAR(col["Pxx"], 5.0, 1e-5);
    EXPECT_NEAR(col["Pyy"], 6.0, 1e-5);
    EXPECT_NEAR(col["Pzz"], 7.0, 1e-5);
    EXPECT_NEAR(col["Pxy"], 1.0, 1e-5);
    EXPECT_NEAR(col["Pxz"], 2.0, 1e-5);
    EXPECT_NEAR(col["Pyz"], 3.0, 1e-5);
    EXPECT_NEAR((col["Pxx"] + col["Pyy"] + col["Pzz"]) / 3.0, 6.0, 1e-5);

    free_device(d_vel);
    free_device(d_force);
    free_device(d_pos);
    free_device(d_virial);
}
