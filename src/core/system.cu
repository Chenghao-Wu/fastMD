#include "system.cuh"

__constant__ float c_masses[MAX_TYPES];

void System::allocate(const SimParams& params, const float* h_masses) {
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

    d_table_idx = nullptr;
    d_table_params = nullptr;
    d_table_data = nullptr;

    CUDA_CHECK(cudaMalloc(&d_max_dr2_int, sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&h_rebuild_flag, sizeof(int),
                              cudaHostAllocMapped));
    *h_rebuild_flag = 0;

    bonds = nullptr;
    bond_param_idx = nullptr;
    angles = nullptr;
    nbonds = 0;
    nangles = 0;

    exclusion_offsets = nullptr;
    exclusion_list = nullptr;
    nexclusions = 0;

    d_mol_id = nullptr;
    d_image = nullptr;

    float h_default_masses[MAX_TYPES];
    if (h_masses != nullptr) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_masses,
                                       ntypes * sizeof(float)));
    } else {
        for (int i = 0; i < MAX_TYPES; i++) h_default_masses[i] = 1.0f;
        CUDA_CHECK(cudaMemcpyToSymbol(c_masses, h_default_masses,
                                       MAX_TYPES * sizeof(float)));
    }
}

void System::free() {
    CUDA_CHECK(cudaFree(pos));
    CUDA_CHECK(cudaFree(vel));
    CUDA_CHECK(cudaFree(force));
    CUDA_CHECK(cudaFree(pos_ref));
    CUDA_CHECK(cudaFree(lj_params));
    CUDA_CHECK(cudaFree(virial));

    if (d_table_idx)     CUDA_CHECK(cudaFree(d_table_idx));
    if (d_table_params)  CUDA_CHECK(cudaFree(d_table_params));
    if (d_table_data)    CUDA_CHECK(cudaFree(d_table_data));

    CUDA_CHECK(cudaFree(d_max_dr2_int));
    CUDA_CHECK(cudaFreeHost(h_rebuild_flag));
    if (bonds) CUDA_CHECK(cudaFree(bonds));
    if (bond_param_idx) CUDA_CHECK(cudaFree(bond_param_idx));
    if (angles) CUDA_CHECK(cudaFree(angles));
    if (exclusion_offsets) CUDA_CHECK(cudaFree(exclusion_offsets));
    if (exclusion_list) CUDA_CHECK(cudaFree(exclusion_list));
    if (d_mol_id) CUDA_CHECK(cudaFree(d_mol_id));
    if (d_image) CUDA_CHECK(cudaFree(d_image));
}

void System::zero_forces() {
    CUDA_CHECK(cudaMemset(force, 0, natoms_padded * sizeof(float4)));
}

void System::zero_virial() {
    CUDA_CHECK(cudaMemset(virial, 0, 6 * sizeof(float)));
}

void System::allocate_rg_buffers(const std::vector<int>& mol_ids,
                                  const std::vector<int>& images,
                                  int natoms_padded) {
    size_t n_int = natoms_padded * sizeof(int);
    size_t n_3int = natoms_padded * 3 * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_mol_id, n_int));
    CUDA_CHECK(cudaMemset(d_mol_id, -1, n_int));
    CUDA_CHECK(cudaMemcpy(d_mol_id, mol_ids.data(),
                          mol_ids.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_image, n_3int));
    CUDA_CHECK(cudaMemset(d_image, 0, n_3int));
    if (!images.empty()) {
        CUDA_CHECK(cudaMemcpy(d_image, images.data(),
                              images.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
}
