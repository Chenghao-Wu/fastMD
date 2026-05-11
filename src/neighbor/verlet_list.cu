#include "verlet_list.cuh"

void VerletList::allocate(int natoms, float rc_skin, float box_L) {
    float density = natoms / (box_L * box_L * box_L);
    float sphere_vol = (4.0f / 3.0f) * 3.14159265358979323846f * rc_skin * rc_skin * rc_skin;
    int estimated = int(density * sphere_vol * 3.0f);
    max_neighbors = max(256, estimated);
    nx = max(1, int(box_L / rc_skin));
    ny = nx;
    nz = nx;
    ncells = nx * ny * nz;
    dedup_needed = (nx < 3 || ny < 3 || nz < 3);
    cell_size = box_L / nx;

    CUDA_CHECK(cudaMalloc(&neighbors,     max_neighbors * natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&num_neighbors, natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_ids,      natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_ids_out,  natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&atom_ids,      natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sorted_atoms,  natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_starts,   ncells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_ends,     ncells * sizeof(int)));

    CUDA_CHECK(cudaMemset(cell_starts, 0, ncells * sizeof(int)));
    CUDA_CHECK(cudaMemset(cell_ends,   0, ncells * sizeof(int)));

    d_cub_temp = nullptr;
    cub_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
        (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, natoms);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

    // Compute max cell occupancy for shared-memory sizing
    max_cell_atoms = 0;
    CUDA_CHECK(cudaHostAlloc(&h_cell_max, sizeof(int), cudaHostAllocMapped));
    *h_cell_max = 0;
}

void VerletList::free() {
    CUDA_CHECK(cudaFree(neighbors));
    CUDA_CHECK(cudaFree(num_neighbors));
    CUDA_CHECK(cudaFree(cell_ids));
    CUDA_CHECK(cudaFree(cell_ids_out));
    CUDA_CHECK(cudaFree(atom_ids));
    CUDA_CHECK(cudaFree(sorted_atoms));
    CUDA_CHECK(cudaFree(cell_starts));
    CUDA_CHECK(cudaFree(cell_ends));
    CUDA_CHECK(cudaFree(d_cub_temp));
    CUDA_CHECK(cudaFreeHost(h_cell_max));
}

// ---------------------------------------------------------------------------
// GPU kernels for Verlet list construction
// ---------------------------------------------------------------------------

#include "../core/pbc.cuh"

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
    cx = ((cx % nx) + nx) % nx;
    cy = ((cy % ny) + ny) % ny;
    cz = ((cz % nz) + nz) % nz;
    cell_ids[i] = (cz * ny + cy) * nx + cx;
    atom_ids[i] = i;
}

__global__ void init_atom_ids(int* atom_ids, int natoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < natoms) atom_ids[i] = i;
}

__global__ void find_cell_starts(const int* __restrict__ sorted_cell_ids,
                                  int* __restrict__ cell_starts,
                                  int* __restrict__ cell_ends,
                                  int natoms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    if (i == 0) {
        cell_starts[sorted_cell_ids[0]] = 0;
    }
    if (i == natoms - 1) {
        cell_ends[sorted_cell_ids[natoms - 1]] = natoms;
    }
    if (i > 0) {
        int cell = sorted_cell_ids[i];
        int prev_cell = sorted_cell_ids[i - 1];
        if (cell != prev_cell) {
            cell_starts[cell] = i;
            cell_ends[prev_cell] = i;
        }
    }
}

#define MAX_SHARED_ATOMS 96

__global__ void build_verlet_list(
    const float4* __restrict__ pos,
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
    int* max_cell_out)
{
    extern __shared__ float4 s_cell_pos[];
    int* s_cell_idx = (int*)(s_cell_pos + MAX_SHARED_ATOMS);

    int cell_id = blockIdx.x;
    int ncells_total = nx * ny * nz;
    if (cell_id >= ncells_total) return;

    int start = cell_starts[cell_id];
    int end   = cell_ends[cell_id];
    int nhome = end - start;
    if (nhome == 0) return;

    int cz = cell_id / (nx * ny);
    int rem = cell_id % (nx * ny);
    int cy = rem / nx;
    int cx = rem % nx;

    // All threads participate in every iteration so __syncthreads is safe
    int niter = (nhome + blockDim.x - 1) / blockDim.x;

    for (int iter = 0; iter < niter; iter++) {
        int a = iter * blockDim.x + threadIdx.x;
        bool active = (a < nhome);
        int i, count;
        float4 pos_i;

        if (active) {
            i = sorted_atoms[start + a];
            pos_i = __ldg(&pos[i]);
        }
        count = 0;

        if (dedup_needed) {
            if (!active) continue;

            int visited[27];
            int n_visited = 0;

            for (int dz = -1; dz <= 1; dz++) {
                int ncz = cz + dz;
                if (ncz < 0) ncz += nz;
                else if (ncz >= nz) ncz -= nz;

                for (int dy = -1; dy <= 1; dy++) {
                    int ncy = cy + dy;
                    if (ncy < 0) ncy += ny;
                    else if (ncy >= ny) ncy -= ny;

                    for (int dx = -1; dx <= 1; dx++) {
                        int ncx = cx + dx;
                        if (ncx < 0) ncx += nx;
                        else if (ncx >= nx) ncx -= nx;

                        int ncell = (ncz * ny + ncy) * nx + ncx;

                        bool already = false;
                        #pragma unroll
                        for (int v = 0; v < n_visited; v++) {
                            if (visited[v] == ncell) {
                                already = true;
                                break;
                            }
                        }
                        if (already) continue;
                        visited[n_visited++] = ncell;

                        int ns = cell_starts[ncell];
                        int ne = cell_ends[ncell];

                        for (int b = ns; b < ne; b++) {
                            int j = sorted_atoms[b];
                            if (i == j) continue;

                            float4 pos_j = __ldg(&pos[j]);
                            float dx_p = min_image(pos_i.x - pos_j.x, L, inv_L);
                            float dy_p = min_image(pos_i.y - pos_j.y, L, inv_L);
                            float dz_p = min_image(pos_i.z - pos_j.z, L, inv_L);
                            float r2 = dx_p*dx_p + dy_p*dy_p + dz_p*dz_p;

                            if (r2 < rc_skin2) {
                                bool excluded = false;
                                if (exclusion_offsets) {
                                    int e_start = exclusion_offsets[i];
                                    int e_end = exclusion_offsets[i + 1];
                                    for (int e = e_start; e < e_end; e++) {
                                        if (exclusion_list[e] == j) {
                                            excluded = true;
                                            break;
                                        }
                                    }
                                }
                                if (!excluded) {
                                    if (count < max_neighbors) {
                                        neighbors[count * natoms + i] = j;
                                    }
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int dz = -1; dz <= 1; dz++) {
                int ncz = cz + dz;
                if (ncz < 0) ncz += nz;
                else if (ncz >= nz) ncz -= nz;

                for (int dy = -1; dy <= 1; dy++) {
                    int ncy = cy + dy;
                    if (ncy < 0) ncy += ny;
                    else if (ncy >= ny) ncy -= ny;

                    for (int dx = -1; dx <= 1; dx++) {
                        int ncx = cx + dx;
                        if (ncx < 0) ncx += nx;
                        else if (ncx >= nx) ncx -= nx;

                        int ncell = (ncz * ny + ncy) * nx + ncx;

                        int ns = cell_starts[ncell];
                        int ne = cell_ends[ncell];
                        int ncount = ne - ns;

                        if (ncount == 0) continue;

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
                                    if (i == j) continue;

                                    float4 pos_j = s_cell_pos[t];
                                    float dx_p = min_image(pos_i.x - pos_j.x, L, inv_L);
                                    float dy_p = min_image(pos_i.y - pos_j.y, L, inv_L);
                                    float dz_p = min_image(pos_i.z - pos_j.z, L, inv_L);
                                    float r2 = dx_p*dx_p + dy_p*dy_p + dz_p*dz_p;

                                    if (r2 < rc_skin2) {
                                        bool excluded = false;
                                        if (exclusion_offsets) {
                                            int e_start = exclusion_offsets[i];
                                            int e_end = exclusion_offsets[i + 1];
                                            for (int e = e_start; e < e_end; e++) {
                                                if (exclusion_list[e] == j) {
                                                    excluded = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if (!excluded) {
                                            if (count < max_neighbors) {
                                                neighbors[count * natoms + i] = j;
                                            }
                                            count++;
                                        }
                                    }
                                }
                            }
                            __syncthreads();
                        } else {
                            if (!active) continue;

                            for (int b = ns; b < ne; b++) {
                                int j = __ldg(&sorted_atoms[b]);
                                if (i == j) continue;

                                float4 pos_j = __ldg(&pos[j]);
                                float dx_p = min_image(pos_i.x - pos_j.x, L, inv_L);
                                float dy_p = min_image(pos_i.y - pos_j.y, L, inv_L);
                                float dz_p = min_image(pos_i.z - pos_j.z, L, inv_L);
                                float r2 = dx_p*dx_p + dy_p*dy_p + dz_p*dz_p;

                                if (r2 < rc_skin2) {
                                    bool excluded = false;
                                    if (exclusion_offsets) {
                                        int e_start = exclusion_offsets[i];
                                        int e_end = exclusion_offsets[i + 1];
                                        for (int e = e_start; e < e_end; e++) {
                                            if (exclusion_list[e] == j) {
                                                excluded = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (!excluded) {
                                        if (count < max_neighbors) {
                                            neighbors[count * natoms + i] = j;
                                        }
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (active) num_neighbors[i] = min(count, max_neighbors);
    }

    // Track max cell occupancy for diagnostics
    if (threadIdx.x == 0) {
        atomicMax(max_cell_out, nhome);
    }
}

void VerletList::build(const float4* pos, int natoms, float box_L, float inv_L,
                       const int* exclusion_offsets, const int* exclusion_list,
                       float rc_skin, cudaStream_t stream)
{
    float rc_skin2 = rc_skin * rc_skin;
    float inv_cell_size = 1.0f / cell_size;

    int blocks = div_ceil(natoms, 256);

    // Stage 1: assign atoms to cells
    assign_cells<<<blocks, 256, 0, stream>>>(
        pos, cell_ids, atom_ids, natoms, inv_cell_size, nx, ny, nz);

    // Stage 2: sort by cell ID
    cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_temp_bytes,
        cell_ids, cell_ids_out, atom_ids, sorted_atoms,
        natoms, 0, 32, stream);

    // Stage 3: clear stale cell data, then find cell starts/ends
    CUDA_CHECK(cudaMemsetAsync(cell_starts, 0, ncells * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(cell_ends,   0, ncells * sizeof(int), stream));
    find_cell_starts<<<blocks, 256, 0, stream>>>(
        cell_ids_out, cell_starts, cell_ends, natoms);

    // Stage 4: build Verlet list with shared memory tiling
    // Shared memory: MAX_SHARED_ATOMS * (sizeof(float4) + sizeof(int)) = 96 * 20 = 1920 bytes
    int smem = MAX_SHARED_ATOMS * (sizeof(float4) + sizeof(int));
    *h_cell_max = 0;
    CUDA_CHECK(cudaMemsetAsync(num_neighbors, 0, natoms * sizeof(int), stream));
    build_verlet_list<<<ncells, 256, smem, stream>>>(
        pos, sorted_atoms, cell_starts, cell_ends,
        exclusion_offsets, exclusion_list,
        neighbors, num_neighbors,
        natoms, nx, ny, nz,
        rc_skin2, box_L, inv_L,
        max_neighbors, dedup_needed ? 1 : 0,
        h_cell_max);
}
