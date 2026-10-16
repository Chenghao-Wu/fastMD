#include "topology.cuh"
#include <stdexcept>

namespace {

template <typename T>
T* upload_vec(const std::vector<T>& h) {
    if (h.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <typename T>
T* clone_dev(const T* src, int n) {
    if (n == 0) return nullptr;
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(T), cudaMemcpyDeviceToDevice));
    return d;
}

__global__ void remap_int2_kernel(int2* __restrict__ out, const int2* __restrict__ in,
                                  const int* __restrict__ rank, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int2 v = in[k];
    out[k] = make_int2(rank[v.x], rank[v.y]);
}

// remap the first `nidx` components; component 3 is copied verbatim when nidx == 3
__global__ void remap_int4_kernel(int4* __restrict__ out, const int4* __restrict__ in,
                                  const int* __restrict__ rank, int n, int nidx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int4 v = in[k];
    out[k] = make_int4(rank[v.x], rank[v.y], rank[v.z], nidx == 4 ? rank[v.w] : v.w);
}

}  // namespace

void DeviceTopology::upload(int natoms,
                            const std::vector<int2>& bonds_h, const std::vector<int>& bond_types_h,
                            const std::vector<int4>& angles_h,
                            const std::vector<int4>& dihedrals_h,
                            const std::vector<int>& dihedral_types_h,
                            const std::vector<int2>& pairs14_h,
                            const std::vector<int>& excl_offsets_h,
                            const std::vector<int>& excl_list_h,
                            const std::vector<int>& excl13_offsets_h,
                            const std::vector<int>& excl13_list_h) {
    nbonds = (int)bonds_h.size();
    nangles = (int)angles_h.size();
    ndihedrals = (int)dihedrals_h.size();
    npairs14 = (int)pairs14_h.size();

    bonds_tag = upload_vec(bonds_h);
    bond_type = upload_vec(bond_types_h);
    angles_tag = upload_vec(angles_h);
    dihedrals_tag = upload_vec(dihedrals_h);
    dihedral_type = upload_vec(dihedral_types_h);
    pairs14_tag = upload_vec(pairs14_h);
    // particles start in tag order, so the index-space copies equal the tags
    bonds = clone_dev(bonds_tag, nbonds);
    angles = clone_dev(angles_tag, nangles);
    dihedrals = clone_dev(dihedrals_tag, ndihedrals);
    pairs14 = clone_dev(pairs14_tag, npairs14);

    nexcl = (int)excl_list_h.size();
    if (nexcl > 0) {
        if ((int)excl_offsets_h.size() != natoms + 1)
            throw std::runtime_error("exclusion CSR offsets have wrong length");
        excl_offsets = upload_vec(excl_offsets_h);
        excl_list = upload_vec(excl_list_h);
    }
    // per-atom bonded-term CSR: bonds, then angles, then dihedrals per row
    if (nbonds + nangles + ndihedrals > 0) {
        std::vector<std::vector<int2>> rows(natoms);
        for (int b = 0; b < nbonds; b++) {
            rows[bonds_h[b].x].push_back(make_int2(b, 0 * 4 + 0));
            rows[bonds_h[b].y].push_back(make_int2(b, 0 * 4 + 1));
        }
        for (int a = 0; a < nangles; a++) {
            const int4& v = angles_h[a];
            rows[v.x].push_back(make_int2(a, 1 * 4 + 0));
            rows[v.y].push_back(make_int2(a, 1 * 4 + 1));
            rows[v.z].push_back(make_int2(a, 1 * 4 + 2));
        }
        for (int d = 0; d < ndihedrals; d++) {
            const int4& v = dihedrals_h[d];
            rows[v.x].push_back(make_int2(d, 2 * 4 + 0));
            rows[v.y].push_back(make_int2(d, 2 * 4 + 1));
            rows[v.z].push_back(make_int2(d, 2 * 4 + 2));
            rows[v.w].push_back(make_int2(d, 2 * 4 + 3));
        }
        std::vector<int> off(natoms + 1, 0);
        std::vector<int2> flat;
        for (int t = 0; t < natoms; t++) {
            off[t] = (int)flat.size();
            flat.insert(flat.end(), rows[t].begin(), rows[t].end());
        }
        off[natoms] = (int)flat.size();
        nterms = (int)flat.size();
        term_offsets = upload_vec(off);
        terms = upload_vec(flat);
    }

    nexcl13 = (int)excl13_list_h.size();
    if (nexcl13 > 0) {
        if ((int)excl13_offsets_h.size() != natoms + 1)
            throw std::runtime_error("1-3 exclusion CSR offsets have wrong length");
        excl13_offsets = upload_vec(excl13_offsets_h);
        excl13_list = upload_vec(excl13_list_h);
    }
}

void DeviceTopology::remap(const int* d_rank, cudaStream_t stream) {
    if (nbonds)
        remap_int2_kernel<<<div_ceil(nbonds, 256), 256, 0, stream>>>(bonds, bonds_tag, d_rank, nbonds);
    if (nangles)
        remap_int4_kernel<<<div_ceil(nangles, 256), 256, 0, stream>>>(angles, angles_tag, d_rank, nangles, 3);
    if (ndihedrals)
        remap_int4_kernel<<<div_ceil(ndihedrals, 256), 256, 0, stream>>>(dihedrals, dihedrals_tag, d_rank, ndihedrals, 4);
    if (npairs14)
        remap_int2_kernel<<<div_ceil(npairs14, 256), 256, 0, stream>>>(pairs14, pairs14_tag, d_rank, npairs14);
    CUDA_CHECK(cudaGetLastError());
}

void DeviceTopology::free() {
    for (void* p : {(void*)bonds_tag, (void*)bonds, (void*)bond_type,
                    (void*)angles_tag, (void*)angles,
                    (void*)dihedrals_tag, (void*)dihedrals, (void*)dihedral_type,
                    (void*)pairs14_tag, (void*)pairs14,
                    (void*)excl_offsets, (void*)excl_list,
                    (void*)excl13_offsets, (void*)excl13_list,
                    (void*)term_offsets, (void*)terms})
        if (p) CUDA_CHECK(cudaFree(p));
    *this = DeviceTopology{};
}
