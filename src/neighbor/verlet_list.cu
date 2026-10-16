#include "verlet_list.cuh"
#include "../core/pbc.cuh"
#include <cub/cub.cuh>
#include <cmath>

namespace {

__host__ __device__ inline uint32_t spread3(uint32_t v) {   // 10 bits -> every 3rd bit
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__host__ __device__ inline uint32_t compact3(uint32_t v) {  // inverse of spread3
    v &= 0x09249249;
    v = (v | (v >>  2)) & 0x030C30C3;
    v = (v | (v >>  4)) & 0x0300F00F;
    v = (v | (v >>  8)) & 0x030000FF;
    v = (v | (v >> 16)) & 0x000003FF;
    return v;
}

__host__ __device__ inline int cell_key(int cx, int cy, int cz) {
    return (int)((spread3(cz) << 2) | (spread3(cy) << 1) | spread3(cx));
}

}  // namespace

void VerletList::set_grid(float box_L, float rc_skin) {
    nx = ny = nz = std::max(1, int(box_L / rc_skin));
    if (nx > 1024) throw std::runtime_error("VerletList: more than 1024 cells per dimension");
    ncells = nx * ny * nz;
    dedup_needed = (nx < 3 || ny < 3 || nz < 3);
    cell_size = box_L / nx;
    key_bits = 0;
    while ((1 << key_bits) < nx) key_bits++;
    nkeys = 1 << (3 * key_bits);
    if (nkeys > nkeys_alloc) {
        if (cell_starts) CUDA_CHECK(cudaFree(cell_starts));
        if (cell_ends) CUDA_CHECK(cudaFree(cell_ends));
        CUDA_CHECK(cudaMalloc(&cell_starts, nkeys * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_ends,   nkeys * sizeof(int)));
        nkeys_alloc = nkeys;
    }
}

void VerletList::allocate(int n, float rc_skin, float box_L) {
    natoms = n;
    natoms_padded = div_ceil(n, TILE_SIZE) * TILE_SIZE;
    float density = natoms / (box_L * box_L * box_L);
    float sphere_vol = (4.0f / 3.0f) * 3.14159265358979323846f * rc_skin * rc_skin * rc_skin;
    int estimated = int(density * sphere_vol * 1.5f) + 32;
    max_neighbors = std::max(64, estimated);

    CUDA_CHECK(cudaMalloc(&neighbors,     (size_t)max_neighbors * natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&num_neighbors, natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_ids,      natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_ids_out,  natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atom_ids,      natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sorted_atoms,  natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&order_pad,     natoms_padded * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pos_sorted,    natoms * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_stats, 2 * sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&h_stats, 2 * sizeof(int), cudaHostAllocDefault));
    set_grid(box_L, rc_skin);

    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, natoms);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
}

void VerletList::free() {
    for (void* p : {(void*)neighbors, (void*)num_neighbors, (void*)cell_ids,
                    (void*)cell_ids_out, (void*)atom_ids, (void*)sorted_atoms, (void*)order_pad,
                    (void*)pos_sorted, (void*)cell_starts, (void*)cell_ends, (void*)d_stats,
                    d_cub_temp})
        if (p) CUDA_CHECK(cudaFree(p));
    if (h_stats) CUDA_CHECK(cudaFreeHost(h_stats));
    *this = VerletList{};
}

// ---------------------------------------------------------------------------
// GPU kernels for Verlet list construction
// ---------------------------------------------------------------------------

__global__ void assign_cells(const float4* __restrict__ pos,
                             int* __restrict__ cell_ids,
                             int* __restrict__ atom_ids,
                             int natoms, float inv_cell_size,
                             int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;
    float4 r = pos[i];
    int cx = __float2int_rd(r.x * inv_cell_size);
    int cy = __float2int_rd(r.y * inv_cell_size);
    int cz = __float2int_rd(r.z * inv_cell_size);
    // positions are wrapped into [0, L); the modulo only guards round-off at L
    cx = ((cx % nx) + nx) % nx;
    cy = ((cy % ny) + ny) % ny;
    cz = ((cz % nz) + nz) % nz;
    cell_ids[i] = cell_key(cx, cy, cz);
    atom_ids[i] = i;
}

__global__ void pad_order(const int* __restrict__ sorted_atoms, int* __restrict__ order_pad,
                          int natoms, int natoms_padded) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < natoms) order_pad[k] = sorted_atoms[k];
    else if (k < natoms_padded) order_pad[k] = k;
}

__global__ void find_cell_starts(const int* __restrict__ sorted_cell_ids,
                                 int* __restrict__ cell_starts,
                                 int* __restrict__ cell_ends,
                                 int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    if (i == 0) cell_starts[sorted_cell_ids[0]] = 0;
    if (i == natoms - 1) cell_ends[sorted_cell_ids[natoms - 1]] = natoms;
    if (i > 0) {
        int cell = sorted_cell_ids[i];
        int prev_cell = sorted_cell_ids[i - 1];
        if (cell != prev_cell) {
            cell_starts[cell] = i;
            cell_ends[prev_cell] = i;
        }
    }
}

// Exclusion rows are indexed by tag and list tags.
__device__ __forceinline__ bool is_excluded(int tag_i, int tag_j,
                                            const int* __restrict__ excl_offsets,
                                            const int* __restrict__ excl_list) {
    if (!excl_offsets) return false;
    int e_end = excl_offsets[tag_i + 1];
    for (int e = excl_offsets[tag_i]; e < e_end; e++)
        if (excl_list[e] == tag_j) return true;
    return false;
}

// Record j as a neighbor of i if within the list radius and not excluded.
// count keeps increasing past the capacity so the true maximum is known.
__device__ __forceinline__ void consider_pair(
    int i, int j, int tag_i, float4 pos_i, float4 pos_j,
    const int* __restrict__ tag,
    const int* __restrict__ excl_offsets, const int* __restrict__ excl_list,
    int* __restrict__ neighbors, int natoms, int max_neighbors,
    float rc_skin2, float L, float inv_L, int& count)
{
    float dx = min_image(pos_i.x - pos_j.x, L, inv_L);
    float dy = min_image(pos_i.y - pos_j.y, L, inv_L);
    float dz = min_image(pos_i.z - pos_j.z, L, inv_L);
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc_skin2) return;
    int tag_j = tag ? __ldg(&tag[j]) : j;
    if (is_excluded(tag_i, tag_j, excl_offsets, excl_list)) return;
    if (count < max_neighbors) neighbors[count * natoms + i] = j;
    count++;
}

// Positions gathered into cell order so the candidate loop reads contiguous
// memory (atoms of one cell are adjacent).
__global__ void gather_cell_order(const float4* __restrict__ pos,
                                  const int* __restrict__ sorted_atoms,
                                  float4* __restrict__ pos_sorted, int natoms) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < natoms) pos_sorted[k] = __ldg(&pos[sorted_atoms[k]]);
}

// One thread per particle, taken in cell order: the threads of a warp mostly
// share a home cell and therefore scan the same 27 neighbor cells, which
// keeps the candidate loads coherent and cached. IN_ORDER: particles are
// stored in cell order (index == sorted position), so no indirection.
template <bool IN_ORDER>
__global__ void build_verlet_list(
    const float4* __restrict__ pos_k,       // positions in cell order
    const int* __restrict__ tag,
    const int* __restrict__ sorted_atoms,
    const int* __restrict__ sorted_cell_ids,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ exclusion_list,
    int* __restrict__ neighbors,
    int* __restrict__ num_neighbors,
    int natoms, int nx, int ny, int nz,
    float rc_skin2, float L, float inv_L,
    int max_neighbors, int dedup_needed,
    int* __restrict__ stats)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    if (k < natoms) {
        int i = IN_ORDER ? k : sorted_atoms[k];
        uint32_t key = (uint32_t)sorted_cell_ids[k];
        float4 pos_i = pos_k[k];
        int tag_i = tag ? tag[i] : i;
        int cx = (int)compact3(key), cy = (int)compact3(key >> 1), cz = (int)compact3(key >> 2);

        int visited[27];
        int n_visited = 0;
        for (int dz = -1; dz <= 1; dz++) {
            int ncz = cz + dz;
            if (ncz < 0) ncz += nz; else if (ncz >= nz) ncz -= nz;
            for (int dy = -1; dy <= 1; dy++) {
                int ncy = cy + dy;
                if (ncy < 0) ncy += ny; else if (ncy >= ny) ncy -= ny;
                for (int dx = -1; dx <= 1; dx++) {
                    int ncx = cx + dx;
                    if (ncx < 0) ncx += nx; else if (ncx >= nx) ncx -= nx;
                    int ncell = cell_key(ncx, ncy, ncz);
                    if (dedup_needed) {
                        // < 3 cells per dimension: neighbor cells repeat under PBC
                        bool already = false;
                        for (int v = 0; v < n_visited; v++)
                            if (visited[v] == ncell) { already = true; break; }
                        if (already) continue;
                        visited[n_visited++] = ncell;
                    }
                    int ne = cell_ends[ncell];
                    for (int b = cell_starts[ncell]; b < ne; b++) {
                        if (b == k) continue;
                        int j = IN_ORDER ? b : __ldg(&sorted_atoms[b]);
                        consider_pair(i, j, tag_i, pos_i, pos_k[b], tag,
                                      exclusion_offsets, exclusion_list, neighbors,
                                      natoms, max_neighbors, rc_skin2, L, inv_L, count);
                    }
                }
            }
        }
        num_neighbors[i] = min(count, max_neighbors);
    }

    // block maximum of the true neighbor count: warp shuffle, one atomic per warp
    for (int offset = 16; offset > 0; offset >>= 1)
        count = max(count, __shfl_down_sync(0xffffffffu, count, offset));
    if ((threadIdx.x & 31) == 0) atomicMax(&stats[0], count);
}

__global__ void max_cell_occupancy(const int* __restrict__ cell_starts,
                                   const int* __restrict__ cell_ends,
                                   int ncells, int* __restrict__ stats) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int occ = c < ncells ? cell_ends[c] - cell_starts[c] : 0;
    for (int offset = 16; offset > 0; offset >>= 1)
        occ = max(occ, __shfl_down_sync(0xffffffffu, occ, offset));
    if ((threadIdx.x & 31) == 0) atomicMax(&stats[1], occ);
}

void VerletList::bin(const float4* pos, float box_L, float rc_skin, cudaStream_t stream) {
    if (!(box_L > 0.0f) || !std::isfinite(box_L))
        throw std::runtime_error("VerletList: invalid box length");
    set_grid(box_L, rc_skin);
    int blocks = div_ceil(natoms, 256);
    assign_cells<<<blocks, 256, 0, stream>>>(pos, cell_ids, atom_ids, natoms,
                                             1.0f / cell_size, nx, ny, nz);
    cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes,
        cell_ids, cell_ids_out, atom_ids, sorted_atoms, natoms, 0, std::max(1, 3 * key_bits), stream);
    CUDA_CHECK(cudaMemsetAsync(cell_starts, 0, nkeys * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(cell_ends,   0, nkeys * sizeof(int), stream));
    find_cell_starts<<<blocks, 256, 0, stream>>>(cell_ids_out, cell_starts, cell_ends, natoms);
    pad_order<<<div_ceil(natoms_padded, 256), 256, 0, stream>>>(sorted_atoms, order_pad,
                                                                natoms, natoms_padded);
    CUDA_CHECK(cudaGetLastError());
}

void VerletList::launch_list(const float4* pos, bool in_order, float box_L, float inv_L,
                             const int* tag, const int* exclusion_offsets,
                             const int* exclusion_list, float rc_skin, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    const float4* pos_k = pos;
    if (!in_order) {
        gather_cell_order<<<blocks, 256, 0, stream>>>(pos, sorted_atoms, pos_sorted, natoms);
        pos_k = pos_sorted;
    }
    CUDA_CHECK(cudaMemsetAsync(d_stats, 0, 2 * sizeof(int), stream));
    auto kernel = in_order ? build_verlet_list<true> : build_verlet_list<false>;
    kernel<<<blocks, 256, 0, stream>>>(
        pos_k, tag, sorted_atoms, cell_ids_out, cell_starts, cell_ends,
        exclusion_offsets, exclusion_list, neighbors, num_neighbors,
        natoms, nx, ny, nz, rc_skin * rc_skin, box_L, inv_L,
        max_neighbors, dedup_needed ? 1 : 0, d_stats);
    max_cell_occupancy<<<div_ceil(nkeys, 256), 256, 0, stream>>>(
        cell_starts, cell_ends, nkeys, d_stats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_stats, d_stats, 2 * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}

void VerletList::build_from_bins(const float4* pos, bool in_order, float box_L, float inv_L,
                                 const int* tag, const int* exclusion_offsets,
                                 const int* exclusion_list, float rc_skin,
                                 cudaStream_t stream) {
    launch_list(pos, in_order, box_L, inv_L, tag, exclusion_offsets, exclusion_list,
                rc_skin, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    max_count = h_stats[0];
    max_cell_atoms = h_stats[1];

    if (max_count > max_neighbors) {
        // Grow with headroom and rebuild: the list must never be truncated.
        int new_cap = max_count + max_count / 4 + 16;
        size_t bytes = (size_t)new_cap * natoms * sizeof(int);
        CUDA_CHECK(cudaFree(neighbors));
        neighbors = nullptr;
        if (cudaMalloc(&neighbors, bytes) != cudaSuccess)
            throw std::runtime_error("VerletList: cannot allocate " + std::to_string(bytes) +
                                     " bytes for " + std::to_string(new_cap) +
                                     " neighbors per particle");
        max_neighbors = new_cap;
        n_resizes++;
        launch_list(pos, in_order, box_L, inv_L, tag, exclusion_offsets, exclusion_list,
                    rc_skin, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        max_count = h_stats[0];
        max_cell_atoms = h_stats[1];
        if (max_count > max_neighbors)
            throw std::runtime_error("VerletList: neighbor overflow after resize");
    }
}

void VerletList::build(const float4* pos, int n, float box_L, float inv_L,
                       const int* tag,
                       const int* exclusion_offsets, const int* exclusion_list,
                       float rc_skin, cudaStream_t stream)
{
    if (n != natoms)
        throw std::runtime_error("VerletList::build: particle count changed");
    bin(pos, box_L, rc_skin, stream);
    build_from_bins(pos, false, box_L, inv_L, tag, exclusion_offsets, exclusion_list,
                    rc_skin, stream);
}
