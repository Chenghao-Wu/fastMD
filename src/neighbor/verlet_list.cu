#include "verlet_list.cuh"
#include "../core/pbc.cuh"
#include <cub/cub.cuh>
#include <cmath>

void VerletList::set_grid(float box_L, float rc_skin) {
    nx = ny = nz = std::max(1, int(box_L / rc_skin));
    ncells = nx * ny * nz;
    dedup_needed = (nx < 3 || ny < 3 || nz < 3);
    cell_size = box_L / nx;
    if (ncells > ncells_alloc) {
        if (cell_starts) CUDA_CHECK(cudaFree(cell_starts));
        if (cell_ends) CUDA_CHECK(cudaFree(cell_ends));
        CUDA_CHECK(cudaMalloc(&cell_starts, ncells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_ends,   ncells * sizeof(int)));
        ncells_alloc = ncells;
    }
}

void VerletList::allocate(int n, float rc_skin, float box_L) {
    natoms = n;
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
    CUDA_CHECK(cudaMalloc(&d_stats, 2 * sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&h_stats, 2 * sizeof(int), cudaHostAllocDefault));
    set_grid(box_L, rc_skin);

    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, natoms);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
}

void VerletList::free() {
    for (void* p : {(void*)neighbors, (void*)num_neighbors, (void*)cell_ids,
                    (void*)cell_ids_out, (void*)atom_ids, (void*)sorted_atoms,
                    (void*)cell_starts, (void*)cell_ends, (void*)d_stats, d_cub_temp})
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
    cell_ids[i] = (cz * ny + cy) * nx + cx;
    atom_ids[i] = i;
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

__global__ void build_verlet_list(
    const float4* __restrict__ pos,
    const int* __restrict__ tag,
    const int* __restrict__ sorted_atoms,
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
    extern __shared__ float4 s_cell_pos[];
    int* s_cell_idx = (int*)(s_cell_pos + MAX_SHARED_ATOMS);

    int cell_id = blockIdx.x;
    if (cell_id >= nx * ny * nz) return;

    int start = cell_starts[cell_id];
    int end   = cell_ends[cell_id];
    int nhome = end - start;
    if (nhome == 0) return;

    int cz = cell_id / (nx * ny);
    int rem = cell_id % (nx * ny);
    int cy = rem / nx;
    int cx = rem % nx;

    int block_max = 0;
    // All threads participate in every iteration so __syncthreads is safe
    int niter = (nhome + blockDim.x - 1) / blockDim.x;
    for (int iter = 0; iter < niter; iter++) {
        int a = iter * blockDim.x + threadIdx.x;
        bool active = (a < nhome);
        int i = 0, tag_i = 0, count = 0;
        float4 pos_i = make_float4(0, 0, 0, 0);
        if (active) {
            i = sorted_atoms[start + a];
            pos_i = __ldg(&pos[i]);
            tag_i = tag ? tag[i] : i;
        }

        if (dedup_needed) {
            // < 3 cells per dimension: neighbor cells repeat under PBC, so
            // visit each distinct cell once (no shared-memory staging).
            if (active) {
                int visited[27];
                int n_visited = 0;
                for (int dz = -1; dz <= 1; dz++) {
                    int ncz = (cz + dz + nz) % nz;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ncy = (cy + dy + ny) % ny;
                        for (int dx = -1; dx <= 1; dx++) {
                            int ncx = (cx + dx + nx) % nx;
                            int ncell = (ncz * ny + ncy) * nx + ncx;
                            bool already = false;
                            for (int v = 0; v < n_visited; v++)
                                if (visited[v] == ncell) { already = true; break; }
                            if (already) continue;
                            visited[n_visited++] = ncell;
                            int ne = cell_ends[ncell];
                            for (int b = cell_starts[ncell]; b < ne; b++) {
                                int j = sorted_atoms[b];
                                if (j == i) continue;
                                consider_pair(i, j, tag_i, pos_i, __ldg(&pos[j]), tag,
                                              exclusion_offsets, exclusion_list, neighbors,
                                              natoms, max_neighbors, rc_skin2, L, inv_L, count);
                            }
                        }
                    }
                }
            }
        } else {
            for (int dz = -1; dz <= 1; dz++) {
                int ncz = cz + dz;
                if (ncz < 0) ncz += nz; else if (ncz >= nz) ncz -= nz;
                for (int dy = -1; dy <= 1; dy++) {
                    int ncy = cy + dy;
                    if (ncy < 0) ncy += ny; else if (ncy >= ny) ncy -= ny;
                    for (int dx = -1; dx <= 1; dx++) {
                        int ncx = cx + dx;
                        if (ncx < 0) ncx += nx; else if (ncx >= nx) ncx -= nx;
                        int ncell = (ncz * ny + ncy) * nx + ncx;

                        int ns = cell_starts[ncell];
                        int ne = cell_ends[ncell];
                        int ncount = ne - ns;
                        if (ncount == 0) continue;   // block-uniform

                        if (ncount <= MAX_SHARED_ATOMS) {
                            // All threads load cooperatively — no divergent barrier
                            for (int t = threadIdx.x; t < ncount; t += blockDim.x) {
                                int j = sorted_atoms[ns + t];
                                s_cell_idx[t] = j;
                                s_cell_pos[t] = __ldg(&pos[j]);
                            }
                            __syncthreads();
                            if (active) {
                                for (int t = 0; t < ncount; t++) {
                                    int j = s_cell_idx[t];
                                    if (j == i) continue;
                                    consider_pair(i, j, tag_i, pos_i, s_cell_pos[t], tag,
                                                  exclusion_offsets, exclusion_list, neighbors,
                                                  natoms, max_neighbors, rc_skin2, L, inv_L, count);
                                }
                            }
                            __syncthreads();
                        } else if (active) {
                            for (int b = ns; b < ne; b++) {
                                int j = __ldg(&sorted_atoms[b]);
                                if (j == i) continue;
                                consider_pair(i, j, tag_i, pos_i, __ldg(&pos[j]), tag,
                                              exclusion_offsets, exclusion_list, neighbors,
                                              natoms, max_neighbors, rc_skin2, L, inv_L, count);
                            }
                        }
                    }
                }
            }
        }
        if (active) {
            num_neighbors[i] = min(count, max_neighbors);
            block_max = max(block_max, count);
        }
    }

    // per-block maximum of the true neighbor count, then one atomic per warp
    for (int offset = 16; offset > 0; offset >>= 1)
        block_max = max(block_max, __shfl_down_sync(0xffffffffu, block_max, offset));
    if ((threadIdx.x & 31) == 0) atomicMax(&stats[0], block_max);
    if (threadIdx.x == 0) atomicMax(&stats[1], nhome);
}

void VerletList::launch_build(const float4* pos, float box_L, float inv_L, const int* tag,
                              const int* exclusion_offsets, const int* exclusion_list,
                              float rc_skin, cudaStream_t stream) {
    int blocks = div_ceil(natoms, 256);
    float inv_cell_size = 1.0f / cell_size;

    // Stage 1: assign atoms to cells (grid derived from the current box)
    assign_cells<<<blocks, 256, 0, stream>>>(
        pos, cell_ids, atom_ids, natoms, inv_cell_size, nx, ny, nz);

    // Stage 2: sort by cell ID
    cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes,
        cell_ids, cell_ids_out, atom_ids, sorted_atoms, natoms, 0, 32, stream);

    // Stage 3: clear stale cell data, then find cell starts/ends
    CUDA_CHECK(cudaMemsetAsync(cell_starts, 0, ncells * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(cell_ends,   0, ncells * sizeof(int), stream));
    find_cell_starts<<<blocks, 256, 0, stream>>>(cell_ids_out, cell_starts, cell_ends, natoms);

    // Stage 4: build the list with shared-memory staging of neighbor cells
    int smem = MAX_SHARED_ATOMS * (sizeof(float4) + sizeof(int));
    CUDA_CHECK(cudaMemsetAsync(d_stats, 0, 2 * sizeof(int), stream));
    build_verlet_list<<<ncells, 256, smem, stream>>>(
        pos, tag, sorted_atoms, cell_starts, cell_ends,
        exclusion_offsets, exclusion_list, neighbors, num_neighbors,
        natoms, nx, ny, nz, rc_skin * rc_skin, box_L, inv_L,
        max_neighbors, dedup_needed ? 1 : 0, d_stats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_stats, d_stats, 2 * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}

void VerletList::build(const float4* pos, int n, float box_L, float inv_L,
                       const int* tag,
                       const int* exclusion_offsets, const int* exclusion_list,
                       float rc_skin, cudaStream_t stream)
{
    if (n != natoms)
        throw std::runtime_error("VerletList::build: particle count changed");
    if (!(box_L > 0.0f) || !std::isfinite(box_L))
        throw std::runtime_error("VerletList::build: invalid box length");
    set_grid(box_L, rc_skin);

    launch_build(pos, box_L, inv_L, tag, exclusion_offsets, exclusion_list, rc_skin, stream);
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
        launch_build(pos, box_L, inv_L, tag, exclusion_offsets, exclusion_list, rc_skin, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        max_count = h_stats[0];
        max_cell_atoms = h_stats[1];
        if (max_count > max_neighbors)
            throw std::runtime_error("VerletList: neighbor overflow after resize");
    }
}
