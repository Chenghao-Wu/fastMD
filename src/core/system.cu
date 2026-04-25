#include "system.cuh"

void System::allocate(const SimParams& params) {
    natoms  = params.natoms;
    ntypes  = params.ntypes;
    ntiles  = div_ceil(natoms, TILE_SIZE);
    natoms_padded = ntiles * TILE_SIZE;

    size_t n4 = natoms_padded * sizeof(float4);

    CUDA_CHECK(cudaMalloc(&pos,     n4));
    CUDA_CHECK(cudaMalloc(&vel,     n4));
    CUDA_CHECK(cudaMalloc(&force,   n4));
    CUDA_CHECK(cudaMalloc(&pos_ref, n4));

    CUDA_CHECK(cudaMemset(pos,   0, n4));
    CUDA_CHECK(cudaMemset(vel,   0, n4));
    CUDA_CHECK(cudaMemset(force, 0, n4));

    int lj_size = ntypes * ntypes;
    CUDA_CHECK(cudaMalloc(&lj_params, lj_size * sizeof(float2)));

    CUDA_CHECK(cudaMalloc(&virial, 6 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_max_dr2_int, sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&h_rebuild_flag, sizeof(int),
                              cudaHostAllocMapped));
    *h_rebuild_flag = 0;

    bonds = nullptr;
    bond_param_idx = nullptr;
    angles = nullptr;
    nbonds = 0;
    nangles = 0;
}

void System::free() {
    CUDA_CHECK(cudaFree(pos));
    CUDA_CHECK(cudaFree(vel));
    CUDA_CHECK(cudaFree(force));
    CUDA_CHECK(cudaFree(pos_ref));
    CUDA_CHECK(cudaFree(lj_params));
    CUDA_CHECK(cudaFree(virial));
    CUDA_CHECK(cudaFree(d_max_dr2_int));
    CUDA_CHECK(cudaFreeHost(h_rebuild_flag));
    if (bonds) CUDA_CHECK(cudaFree(bonds));
    if (bond_param_idx) CUDA_CHECK(cudaFree(bond_param_idx));
    if (angles) CUDA_CHECK(cudaFree(angles));
}

void System::zero_forces() {
    CUDA_CHECK(cudaMemset(force, 0, natoms_padded * sizeof(float4)));
}

void System::zero_virial() {
    CUDA_CHECK(cudaMemset(virial, 0, 6 * sizeof(float)));
}
