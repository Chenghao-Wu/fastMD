#include "particles.cuh"
#include <algorithm>
#include <stdexcept>

namespace {

template <typename T>
__global__ void gather_kernel(T* __restrict__ dst, const T* __restrict__ src,
                              const int* __restrict__ new_order, int n, int width) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int old = new_order[k];
    for (int c = 0; c < width; c++) dst[k * width + c] = src[old * width + c];
}

__global__ void rank_from_tag_kernel(const int* __restrict__ tag,
                                     int* __restrict__ rank, int natoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < natoms) rank[tag[i]] = i;
}

__global__ void gather_by_tag_kernel(float4* __restrict__ out,
                                     const float4* __restrict__ in,
                                     const int* __restrict__ rank, int natoms) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < natoms) out[t] = in[rank[t]];
}

template <typename T>
void permute_array(T* d, void* scratch, const int* new_order, int n, int width,
                   cudaStream_t stream) {
    if (!d) return;
    T* tmp = static_cast<T*>(scratch);
    gather_kernel<T><<<div_ceil(n, 256), 256, 0, stream>>>(tmp, d, new_order, n, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(d, tmp, (size_t)n * width * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

void ParticleData::allocate(int n) {
    natoms = n;
    natoms_padded = div_ceil(n, TILE_SIZE) * TILE_SIZE;
    size_t np = natoms_padded;
    CUDA_CHECK(cudaMalloc(&pos,     np * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&vel,     np * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&force,   np * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&pos_ref, np * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&image,   np * 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&charge,  np * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mol_id,  np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tag,     np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rank,    np * sizeof(int)));
    // scratch must hold the widest per-particle record (float4 or int[3])
    CUDA_CHECK(cudaMalloc(&scratch_, np * sizeof(float4)));
    CUDA_CHECK(cudaMemset(force, 0, np * sizeof(float4)));
}

void ParticleData::free() {
    for (void* p : {(void*)pos, (void*)vel, (void*)force, (void*)pos_ref,
                    (void*)image, (void*)charge, (void*)mol_id, (void*)tag,
                    (void*)rank, scratch_})
        if (p) CUDA_CHECK(cudaFree(p));
    pos = vel = force = pos_ref = nullptr;
    image = mol_id = tag = rank = nullptr;
    charge = nullptr;
    scratch_ = nullptr;
}

void ParticleData::upload(const std::vector<float4>& positions,
                          const std::vector<float4>& velocities,
                          const std::vector<int>& images,
                          const std::vector<float>& charges,
                          const std::vector<int>& mol_ids) {
    const int np = natoms_padded;
    std::vector<float4> p(np, make_float4(0, 0, 0, pack_type_id(-1)));
    std::copy(positions.begin(), positions.begin() + natoms, p.begin());
    CUDA_CHECK(cudaMemcpy(pos, p.data(), np * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pos_ref, p.data(), np * sizeof(float4), cudaMemcpyHostToDevice));

    std::vector<float4> v(np, make_float4(0, 0, 0, 0));
    std::copy(velocities.begin(), velocities.begin() + std::min<size_t>(velocities.size(), natoms),
              v.begin());
    CUDA_CHECK(cudaMemcpy(vel, v.data(), np * sizeof(float4), cudaMemcpyHostToDevice));

    std::vector<int> im(np * 3, 0);
    std::copy(images.begin(), images.begin() + std::min<size_t>(images.size(), 3 * (size_t)natoms),
              im.begin());
    CUDA_CHECK(cudaMemcpy(image, im.data(), im.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> q(np, 0.0f);
    std::copy(charges.begin(), charges.begin() + std::min<size_t>(charges.size(), natoms), q.begin());
    CUDA_CHECK(cudaMemcpy(charge, q.data(), np * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<int> m(np, -1);
    std::copy(mol_ids.begin(), mol_ids.begin() + std::min<size_t>(mol_ids.size(), natoms), m.begin());
    CUDA_CHECK(cudaMemcpy(mol_id, m.data(), np * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> t(np);
    for (int i = 0; i < np; i++) t[i] = i;
    CUDA_CHECK(cudaMemcpy(tag, t.data(), np * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rank, t.data(), np * sizeof(int), cudaMemcpyHostToDevice));
}

void ParticleData::apply_permutation(const int* d_new_order, cudaStream_t stream) {
    const int np = natoms_padded;
    permute_array(pos,     scratch_, d_new_order, np, 1, stream);
    permute_array(vel,     scratch_, d_new_order, np, 1, stream);
    permute_array(force,   scratch_, d_new_order, np, 1, stream);
    permute_array(pos_ref, scratch_, d_new_order, np, 1, stream);
    permute_array(image,   scratch_, d_new_order, np, 3, stream);
    permute_array(charge,  scratch_, d_new_order, np, 1, stream);
    permute_array(mol_id,  scratch_, d_new_order, np, 1, stream);
    permute_array(tag,     scratch_, d_new_order, np, 1, stream);
    rank_from_tag_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(tag, rank, natoms);
    CUDA_CHECK(cudaGetLastError());
}

void ParticleData::copy_from(const ParticleData& src, cudaStream_t stream) {
    if (src.natoms_padded != natoms_padded) throw std::runtime_error("ParticleData size mismatch");
    const size_t np = natoms_padded;
    auto cp = [&](void* d, const void* s, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(d, s, bytes, cudaMemcpyDeviceToDevice, stream));
    };
    cp(pos, src.pos, np * sizeof(float4));
    cp(vel, src.vel, np * sizeof(float4));
    cp(force, src.force, np * sizeof(float4));
    cp(pos_ref, src.pos_ref, np * sizeof(float4));
    cp(image, src.image, np * 3 * sizeof(int));
    cp(charge, src.charge, np * sizeof(float));
    cp(mol_id, src.mol_id, np * sizeof(int));
    cp(tag, src.tag, np * sizeof(int));
    cp(rank, src.rank, np * sizeof(int));
}

void ParticleData::gather_float4_by_tag(float4* d_out, const float4* d_in,
                                        cudaStream_t stream) const {
    gather_by_tag_kernel<<<div_ceil(natoms, 256), 256, 0, stream>>>(d_out, d_in, rank, natoms);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
std::vector<T> ParticleData::download_by_tag(const T* d, int width) const {
    std::vector<T> raw((size_t)natoms * width);
    std::vector<int> h_tag(natoms);
    CUDA_CHECK(cudaMemcpy(raw.data(), d, raw.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tag.data(), tag, natoms * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<T> out(raw.size());
    for (int i = 0; i < natoms; i++)
        for (int c = 0; c < width; c++)
            out[(size_t)h_tag[i] * width + c] = raw[(size_t)i * width + c];
    return out;
}

std::vector<float4> ParticleData::download_positions() const { return download_by_tag(pos, 1); }
std::vector<float4> ParticleData::download_velocities() const { return download_by_tag(vel, 1); }
std::vector<float4> ParticleData::download_forces() const { return download_by_tag(force, 1); }
std::vector<int>    ParticleData::download_images() const { return download_by_tag(image, 3); }
std::vector<float>  ParticleData::download_charges() const { return download_by_tag(charge, 1); }
std::vector<int>    ParticleData::download_mol_ids() const { return download_by_tag(mol_id, 1); }
