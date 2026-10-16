#include "verlet_list.cuh"
#include "../core/pbc.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;        // nl_build
constexpr float kHalfCellThreshold = 64.0f;   // particles per (rc+skin)^3 cell
constexpr int kBinBlock = 1024;    // nl_bin: its last block scans/scatters/gathers alone

uint32_t spread3_host(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

__device__ __forceinline__ int wrap_cell(int c, int n) {
    c %= n;
    return c < 0 ? c + n : c;
}

__device__ __forceinline__ int wrap_near(int c, int n) {   // c in [-n, 2n)
    return c < 0 ? c + n : (c >= n ? c - n : c);
}

// Cell coordinates of a position: cells divide the fractional coordinates
// (nx, ny, nz cells along the cell vectors; sheared cells for a tilted box).
__device__ __forceinline__ int3 cell_of(float4 r, const Box& b, int nx, int ny, int nz) {
    const float3 s = fractional(r, b);
    return make_int3(wrap_cell(__float2int_rd(s.x * nx), nx),
                     wrap_cell(__float2int_rd(s.y * ny), ny),
                     wrap_cell(__float2int_rd(s.z * nz), nz));
}

// Rebuild criterion, evaluated identically by every thread from device state:
// a pair outside the list (distance > rc + skin at the last build) can come
// within rc once particles move; with the smallest box scaling factor
// s = min_i L_i / L_build,i and maximum displacement d (relative to the
// affinely scaled reference) the list is valid while s (rc + skin) - 2 d > rc.
__device__ __forceinline__ bool rebuild_needed(const DeviceNeighborState* st, const Box& box,
                                               int max_dr2_bits, float rc, float skin,
                                               int max_age) {
    if (st->force_rebuild || !(st->L_build.x > 0.0f)) return true;
    float s = fminf(box.L.x / st->L_build.x, fminf(box.L.y / st->L_build.y, box.L.z / st->L_build.z));
    float half_margin = 0.5f * (s * (rc + skin) - rc);
    float max_dr2 = __int_as_float(max_dr2_bits);
    if (half_margin <= 0.0f || max_dr2 > half_margin * half_margin) return true;
    return max_age > 0 && st->age + 1 >= max_age;
}

struct BinArgs {
    const float4* pos;
    float4* pos_ref;
    const Box* box;
    int* max_dr2_bits;
    DeviceNeighborState* st;
    int* cell_key;
    int* cell_slot;
    int* cell_count;
    int* cell_starts;
    int* cell_ends;
    const int* cell_rank;
    int* order;
    float4* pos_sorted;
    int natoms, natoms_padded, nx, ny, nz, ncells;
    float rc, skin, rc_min_image;
    int max_age;
    int cell_div;
};

// Exclusive scan of cell counts by one block; returns the max occupancy.
__device__ int block_scan_cells(const int* __restrict__ count, int* __restrict__ starts,
                                int* __restrict__ ends, int ncells) {
    __shared__ int s_warp[32];
    __shared__ int s_carry;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (threadIdx.x == 0) s_carry = 0;
    int occ_max = 0;
    __syncthreads();
    for (int base = 0; base < ncells; base += blockDim.x) {
        int c = base + threadIdx.x;
        int v = c < ncells ? __ldcg(&count[c]) : 0;
        occ_max = max(occ_max, v);
        int x = v;   // inclusive warp scan
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) s_warp[warp] = x;
        __syncthreads();
        if (warp == 0) {
            int w = lane < nwarps ? s_warp[lane] : 0;
            for (int o = 1; o < 32; o <<= 1) {
                int y = __shfl_up_sync(0xffffffffu, w, o);
                if (lane >= o) w += y;
            }
            if (lane < nwarps) s_warp[lane] = w;   // inclusive totals of warps 0..lane
        }
        __syncthreads();
        int excl = s_carry + (warp > 0 ? s_warp[warp - 1] : 0) + x - v;
        if (c < ncells) {
            starts[c] = excl;
            ends[c] = excl + v;
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) s_carry = excl + v;
        __syncthreads();
    }
    for (int o = 16; o > 0; o >>= 1) occ_max = max(occ_max, __shfl_down_sync(0xffffffffu, occ_max, o));
    if (lane == 0) s_warp[warp] = occ_max;
    __syncthreads();
    int m = 0;
    for (int w = 0; w < nwarps; w++) m = max(m, s_warp[w]);
    return m;
}

__global__ void nl_bin_kernel(BinArgs a) {
    const Box box = *a.box;
    if (!rebuild_needed(a.st, box, *a.max_dr2_bits, a.rc, a.skin, a.max_age)) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.natoms) {
        float4 r = a.pos[i];
        int3 c = cell_of(r, box, a.nx, a.ny, a.nz);
        int key = a.cell_rank[(c.z * a.ny + c.y) * a.nx + c.x];
        a.cell_key[i] = key;
        a.cell_slot[i] = atomicAdd(&a.cell_count[key], 1);
        a.pos_ref[i] = r;
    }
    if (i == 0) {
        const float3 w = perpendicular_widths(box);
        if (w.x * a.cell_div / a.nx < a.rc + a.skin || w.y * a.cell_div / a.ny < a.rc + a.skin ||
            w.z * a.cell_div / a.nz < a.rc + a.skin ||
            fminf(w.x, fminf(w.y, w.z)) < 2.0f * a.rc_min_image)
            a.st->grid_error = 1;
    }

    // The last block to finish scans the counts, scatters and gathers.
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = (atomicAdd(&a.st->bin_done, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();

    int occ = block_scan_cells(a.cell_count, a.cell_starts, a.cell_ends, a.ncells);
    __syncthreads();
    for (int j = threadIdx.x; j < a.natoms; j += blockDim.x)
        a.order[a.cell_starts[__ldcg(&a.cell_key[j])] + __ldcg(&a.cell_slot[j])] = j;
    for (int c = threadIdx.x; c < a.ncells; c += blockDim.x) a.cell_count[c] = 0;
    __syncthreads();
    for (int k = threadIdx.x; k < a.natoms_padded; k += blockDim.x) {
        if (k < a.natoms) a.pos_sorted[k] = a.pos[a.order[k]];
        else a.order[k] = k;
    }
    if (threadIdx.x == 0) {
        a.st->rebuild = 1;
        a.st->force_rebuild = 0;
        a.st->L_build = box.L;
        a.st->n_builds++;
        a.st->max_count = 0;
        a.st->max_cell = occ;
        a.st->bin_done = 0u;
        *a.max_dr2_bits = 0;
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

struct BuildArgs {
    const Box* box;
    DeviceNeighborState* st;
    const int* tag;
    const int* excl_offsets;
    const int* excl_list;
    const int* order;
    const float4* pos_sorted;
    const int* cell_starts;
    const int* cell_ends;
    const int* cell_rank;
    int* neighbors;
    int* num_neighbors;
    int natoms, nx, ny, nz, max_neighbors;
    float rc_skin2;
};

// TPA threads per particle, particles taken in cell order (the groups of a
// warp share a home cell and scan the same neighbor cells). Cells are at
// least (rc + skin) / R wide and the stencil reaches R cells each way: R = 2
// (125 half-size cells) scans 15.6 (rc+skin)^3 of candidates instead of 27
// (rc+skin)^3, which pays off for dense systems. The group's
// lanes stride over each cell's candidates; hits are compacted with a
// group-wide ballot so every lane keeps the same running count. Counts
// continue past the capacity so overflow is detected exactly.
template <int TPA, int R, bool TILT>
__global__ void nl_build_kernel(BuildArgs a) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (!a.st->rebuild) {
        if (g == 0) a.st->age++;
        return;
    }
    const int k = g / TPA;
    const int sub = threadIdx.x & (TPA - 1);
    const unsigned group_mask = (TPA == 32 ? 0xffffffffu : ((1u << TPA) - 1u))
                                << ((threadIdx.x & 31) & ~(TPA - 1));
    const Box box = *a.box;
    int count = 0;
    if (k < a.natoms) {
        const int i = a.order[k];
        const float4 pos_i = a.pos_sorted[k];
        const int tag_i = a.tag ? a.tag[i] : i;
        const int3 c = cell_of(pos_i, box, a.nx, a.ny, a.nz);
        const bool dedup = a.nx < 2 * R + 1 || a.ny < 2 * R + 1 || a.nz < 2 * R + 1;
        constexpr int kStencil = (2 * R + 1) * (2 * R + 1) * (2 * R + 1);
        int visited[kStencil];
        int n_visited = 0;
        for (int dz = -R; dz <= R; dz++)
            for (int dy = -R; dy <= R; dy++)
                for (int dx = -R; dx <= R; dx++) {
                    int dense;
                    if (dedup)
                        dense = (wrap_cell(c.z + dz, a.nz) * a.ny + wrap_cell(c.y + dy, a.ny)) * a.nx +
                                wrap_cell(c.x + dx, a.nx);
                    else
                        dense = (wrap_near(c.z + dz, a.nz) * a.ny + wrap_near(c.y + dy, a.ny)) * a.nx +
                                wrap_near(c.x + dx, a.nx);
                    int cell = __ldg(&a.cell_rank[dense]);
                    if (dedup) {   // < 2R+1 cells per side: neighbor cells repeat under PBC
                        bool seen = false;
                        for (int v = 0; v < n_visited; v++) seen |= (visited[v] == cell);
                        if (seen) continue;
                        visited[n_visited++] = cell;
                    }
                    const int start = a.cell_starts[cell], end = a.cell_ends[cell];
                    for (int base = start; base < end; base += TPA) {
                        const int b = base + sub;
                        int j = -1;
                        if (b < end && b != k) {
                            float3 d = min_image_dr_t<TILT>(pos_i, a.pos_sorted[b], box);
                            if (d.x * d.x + d.y * d.y + d.z * d.z < a.rc_skin2) {
                                j = __ldg(&a.order[b]);
                                int tag_j = a.tag ? __ldg(&a.tag[j]) : j;
                                if (is_excluded(tag_i, tag_j, a.excl_offsets, a.excl_list)) j = -1;
                            }
                        }
                        if (TPA == 1) {
                            if (j >= 0) {
                                if (count < a.max_neighbors) a.neighbors[count * a.natoms + i] = j;
                                count++;
                            }
                        } else {
                            unsigned hits = __ballot_sync(group_mask, j >= 0) & group_mask;
                            if (j >= 0) {
                                int slot = count + __popc(hits & ((1u << (threadIdx.x & 31)) - 1u));
                                if (slot < a.max_neighbors) a.neighbors[slot * a.natoms + i] = j;
                            }
                            count += __popc(hits);
                        }
                    }
                }
        if (sub == 0) a.num_neighbors[i] = min(count, a.max_neighbors);
    }
    for (int o = 16; o > 0; o >>= 1) count = max(count, __shfl_down_sync(0xffffffffu, count, o));
    if ((threadIdx.x & 31) == 0) atomicMax(&a.st->max_count, count);

    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = (atomicAdd(&a.st->build_done, 1u) == gridDim.x - 1);
    __syncthreads();
    if (last && threadIdx.x == 0) {
        __threadfence();
        int mc = atomicAdd(&a.st->max_count, 0);
        if (mc > a.max_neighbors) a.st->overflow = max(a.st->overflow, mc);
        a.st->rebuild = 0;
        a.st->age = 0;
        a.st->build_done = 0u;
    }
}

}  // namespace

void VerletList::allocate(int n, float rc_in, float skin_in, const Box& box, float margin, int age) {
    natoms = n;
    natoms_padded = div_ceil(n, TILE_SIZE) * TILE_SIZE;
    rc = rc_in;
    skin = skin_in;
    rc_min_image = rc_in;
    grid_margin = std::max(1.0f, margin);
    max_age = age;
    const float rcs = rc + skin;
    float density = natoms / box.volume();
    int estimated = int(density * (4.0f / 3.0f) * 3.14159265f * rcs * rcs * rcs * 1.5f) + 32;
    max_neighbors = std::max(64, estimated);
    // Half-size cells once a full cell holds enough particles to keep the
    // build's thread groups busy (dense all-atom systems); FASTMD_CELL_DIV=1|2
    // overrides.
    const float per_cell = density * rcs * rcs * rcs;
    cell_div = per_cell > kHalfCellThreshold ? 2 : 1;
    if (const char* env = std::getenv("FASTMD_CELL_DIV")) {
        int v = std::atoi(env);
        if (v == 1 || v == 2) cell_div = v;
    }

    CUDA_CHECK(cudaMalloc(&neighbors,     (size_t)max_neighbors * natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&num_neighbors, natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_key,      natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cell_slot,     natoms * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&order,         natoms_padded * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pos_sorted,    natoms * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(DeviceNeighborState)));
    CUDA_CHECK(cudaHostAlloc(&h_state, sizeof(DeviceNeighborState), cudaHostAllocDefault));
    CUDA_CHECK(cudaMemset(d_state, 0, sizeof(DeviceNeighborState)));
    *h_state = DeviceNeighborState{};
    CUDA_CHECK(cudaMalloc(&d_box_scratch_, sizeof(Box)));
    CUDA_CHECK(cudaMalloc(&d_dr2_scratch_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ref_scratch_, natoms_padded * sizeof(float4)));
    set_grid(box);
}

void VerletList::free() {
    for (void* p : {(void*)neighbors, (void*)num_neighbors, (void*)cell_key, (void*)cell_slot,
                    (void*)cell_count, (void*)cell_starts, (void*)cell_ends, (void*)cell_rank,
                    (void*)order, (void*)pos_sorted, (void*)d_state, (void*)d_box_scratch_,
                    (void*)d_dr2_scratch_, (void*)d_ref_scratch_})
        if (p) CUDA_CHECK(cudaFree(p));
    if (h_state) CUDA_CHECK(cudaFreeHost(h_state));
    *this = VerletList{};
}

bool VerletList::grid_ok_for(const Box& box) const {
    const float need = rc + skin;
    const float3 w = perpendicular_widths(box);
    return nx > 0 && tilted_ == (box.xy != 0.0f) &&   // the build variant follows the tilt
           w.x * cell_div / nx >= need && w.y * cell_div / ny >= need &&
           w.z * cell_div / nz >= need && std::min({w.x, w.y, w.z}) >= 2.0f * rc_min_image;
}

int VerletList::cells_along(float length) const {
    return std::max(1, int(length * cell_div / ((rc + skin) * grid_margin)));
}

bool VerletList::grid_matches(const Box& box) const {
    const float3 w = perpendicular_widths(box);
    return nx == cells_along(w.x) && ny == cells_along(w.y) && nz == cells_along(w.z) &&
           tilted_ == (box.xy != 0.0f);
}

void VerletList::set_grid(const Box& box) {
    for (float l : {box.L.x, box.L.y, box.L.z})
        if (!(l > 0.0f) || !std::isfinite(l)) throw std::runtime_error("VerletList: invalid box length");
    const float3 w = perpendicular_widths(box);
    tilted_ = box.xy != 0.0f;
    nx = cells_along(w.x);
    ny = cells_along(w.y);
    nz = cells_along(w.z);
    if (std::max({nx, ny, nz}) > 1024)
        throw std::runtime_error("VerletList: more than 1024 cells per dimension");
    if ((long long)nx * ny * nz > (1ll << 28)) throw std::runtime_error("VerletList: too many cells");
    ncells = nx * ny * nz;
    cell_size = std::min({w.x / nx, w.y / ny, w.z / nz});
    if (ncells > ncells_alloc) {
        for (int** p : {&cell_count, &cell_starts, &cell_ends, &cell_rank})
            if (*p) CUDA_CHECK(cudaFree(*p));
        CUDA_CHECK(cudaMalloc(&cell_count,  ncells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_starts, ncells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_ends,   ncells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cell_rank,   ncells * sizeof(int)));
        ncells_alloc = ncells;
    }
    // rank of every cell along the Morton curve of its coordinates
    std::vector<std::pair<uint32_t, int>> code(ncells);
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                int dense = (z * ny + y) * nx + x;
                code[dense] = {(spread3_host(z) << 2) | (spread3_host(y) << 1) | spread3_host(x), dense};
            }
    std::sort(code.begin(), code.end());
    std::vector<int> rank(ncells);
    for (int r = 0; r < ncells; r++) rank[code[r].second] = r;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cell_rank, rank.data(), ncells * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(cell_count, 0, ncells * sizeof(int)));
    const int one = 1;
    CUDA_CHECK(cudaMemcpy(&d_state->force_rebuild, &one, sizeof(int), cudaMemcpyHostToDevice));
}

void VerletList::request_rebuild(cudaStream_t stream) {
    static const int one = 1;
    CUDA_CHECK(cudaMemcpyAsync(&d_state->force_rebuild, &one, sizeof(int),
                               cudaMemcpyHostToDevice, stream));
}

void VerletList::read_state(cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(h_state, d_state, sizeof(DeviceNeighborState),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    max_count = h_state->max_count;
    max_cell_atoms = h_state->max_cell;
}

void VerletList::clear_errors(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(&d_state->overflow, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(&d_state->grid_error, 0, sizeof(int), stream));
}

void VerletList::grow(int needed) {
    int new_cap = needed + needed / 4 + 16;
    if (new_cap <= max_neighbors) return;
    size_t bytes = (size_t)new_cap * natoms * sizeof(int);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(neighbors));
    neighbors = nullptr;
    if (cudaMalloc(&neighbors, bytes) != cudaSuccess)
        throw std::runtime_error("VerletList: cannot allocate " + std::to_string(bytes) +
                                 " bytes for " + std::to_string(new_cap) + " neighbors per particle");
    max_neighbors = new_cap;
    n_resizes++;
}

void VerletList::set_capacity(int cap) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(neighbors));
    max_neighbors = std::max(1, cap);
    CUDA_CHECK(cudaMalloc(&neighbors, (size_t)max_neighbors * natoms * sizeof(int)));
}

void VerletList::launch_bin(const float4* pos, float4* pos_ref, const Box* d_box,
                            int* d_max_dr2_bits, cudaStream_t stream) {
    BinArgs a{pos, pos_ref, d_box, d_max_dr2_bits, d_state, cell_key, cell_slot, cell_count,
              cell_starts, cell_ends, cell_rank, order, pos_sorted,
              natoms, natoms_padded, nx, ny, nz, ncells, rc, skin, rc_min_image, max_age,
              cell_div};
    nl_bin_kernel<<<div_ceil(natoms, kBinBlock), kBinBlock, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void VerletList::launch_build(const float4* /*pos*/, const Box* d_box, const int* tag,
                              const int* exclusion_offsets, const int* exclusion_list,
                              cudaStream_t stream) {
    BuildArgs a{d_box, d_state, tag, exclusion_offsets, exclusion_list, order, pos_sorted,
                cell_starts, cell_ends, cell_rank, neighbors, num_neighbors,
                natoms, nx, ny, nz, max_neighbors, (rc + skin) * (rc + skin)};
    const int tpa = build_tpa;
    const int blocks = div_ceil(natoms * tpa, kBlock);
#define FASTMD_BUILD2(T, X) do {                                                       \
        if (cell_div == 2) nl_build_kernel<T, 2, X><<<blocks, kBlock, 0, stream>>>(a); \
        else nl_build_kernel<T, 1, X><<<blocks, kBlock, 0, stream>>>(a);               \
    } while (0)
#define FASTMD_BUILD(T) do { if (tilted_) FASTMD_BUILD2(T, true); else FASTMD_BUILD2(T, false); } while (0)
    switch (tpa) {
        case 1: FASTMD_BUILD(1); break;
        case 2: FASTMD_BUILD(2); break;
        case 4: FASTMD_BUILD(4); break;
        default: FASTMD_BUILD(8); break;
    }
#undef FASTMD_BUILD
#undef FASTMD_BUILD2
    CUDA_CHECK(cudaGetLastError());
}

void VerletList::step(const float4* pos, float4* pos_ref, const Box* d_box,
                      int* d_max_dr2_bits, const int* tag, const int* exclusion_offsets,
                      const int* exclusion_list, cudaStream_t stream) {
    launch_bin(pos, pos_ref, d_box, d_max_dr2_bits, stream);
    launch_build(pos, d_box, tag, exclusion_offsets, exclusion_list, stream);
}

void VerletList::bin(const float4* pos, float4* pos_ref, const Box* d_box,
                     int* d_max_dr2_bits, cudaStream_t stream) {
    request_rebuild(stream);
    launch_bin(pos, pos_ref, d_box, d_max_dr2_bits, stream);
}

void VerletList::build(const float4* pos, int n, const Box& box,
                       const int* tag,
                       const int* exclusion_offsets, const int* exclusion_list,
                       float rc_skin, cudaStream_t stream)
{
    if (n != natoms)
        throw std::runtime_error("VerletList::build: particle count changed");
    if (std::fabs(rc + skin - rc_skin) > 1e-6f * rc_skin) {   // callers pass rc + skin
        rc = rc_skin;
        skin = 0.0f;
        rc_min_image = 0.0f;
    }
    if (!grid_ok_for(box) || !grid_matches(box)) set_grid(box);
    CUDA_CHECK(cudaMemcpyAsync(d_box_scratch_, &box, sizeof(Box), cudaMemcpyHostToDevice, stream));
    for (int attempt = 0; attempt < 4; attempt++) {
        request_rebuild(stream);
        launch_bin(pos, d_ref_scratch_, d_box_scratch_, d_dr2_scratch_, stream);
        launch_build(pos, d_box_scratch_, tag, exclusion_offsets, exclusion_list, stream);
        read_state(stream);
        if (h_state->grid_error) throw std::runtime_error("VerletList: invalid cell grid");
        if (h_state->overflow == 0) return;
        grow(h_state->overflow);
        clear_errors(stream);
    }
    throw std::runtime_error("VerletList: neighbor overflow after resize");
}
