#include "thermo.cuh"
#include "../core/system.cuh"
#include <cub/cub.cuh>
#include <cstring>

void ThermoBuffers::allocate() {
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_vel, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_force, 2 * sizeof(float)));
    allocated = true;
    fp = nullptr;
}

void ThermoBuffers::free() {
    CUDA_CHECK(cudaFree(d_kin_stress));
    CUDA_CHECK(cudaFree(d_pe));
    CUDA_CHECK(cudaFree(d_max_vel));
    CUDA_CHECK(cudaFree(d_max_force));
    allocated = false;
    if (fp) fclose(fp);
}

// Simple diagnostic: find max velocity and force using shared memory + global atomics.
// The index may have a race but is close enough for diagnostics.
__global__ void max_vel_force_kernel(const float4* __restrict__ vel,
                                      const float4* __restrict__ force,
                                      float* __restrict__ max_vel_out,
                                      float* __restrict__ max_force_out,
                                      int natoms) {
    __shared__ float sv[32], sf[32];
    __shared__ int   svi[32], sfi[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float vmag2 = 0.0f, fmag2 = 0.0f;
    int vi = -1, fi = -1;
    if (i < natoms) {
        float4 v = vel[i];
        vmag2 = v.x*v.x + v.y*v.y + v.z*v.z;
        vi = i;
        float4 f = force[i];
        fmag2 = f.x*f.x + f.y*f.y + f.z*f.z;
        fi = i;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(0xFFFFFFFF, vmag2, offset);
        int   vi_other = __shfl_down_sync(0xFFFFFFFF, vi, offset);
        if (vmag2 < v_other) { vmag2 = v_other; vi = vi_other; }
        float f_other = __shfl_down_sync(0xFFFFFFFF, fmag2, offset);
        int   fi_other = __shfl_down_sync(0xFFFFFFFF, fi, offset);
        if (fmag2 < f_other) { fmag2 = f_other; fi = fi_other; }
    }

    if (tid % 32 == 0) {
        sv[tid / 32] = vmag2; svi[tid / 32] = vi;
        sf[tid / 32] = fmag2; sfi[tid / 32] = fi;
    }
    __syncthreads();

    if (tid == 0) {
        int nw = blockDim.x / 32;
        float bv = sv[0], bf = sf[0];
        int bvi = svi[0], bfi = sfi[0];
        for (int w = 1; w < nw; w++) {
            if (sv[w] > bv) { bv = sv[w]; bvi = svi[w]; }
            if (sf[w] > bf) { bf = sf[w]; bfi = sfi[w]; }
        }
        atomicMax((int*)&max_vel_out[0], __float_as_int(bv));
        // Index is best-effort (may race with another block, but diagnostic only)
        max_vel_out[1] = __int_as_float(bvi);
        atomicMax((int*)&max_force_out[0], __float_as_int(bf));
        max_force_out[1] = __int_as_float(bfi);
    }
}

void ThermoBuffers::open_file(const char* path) {
    fp = fopen(path, "w");
    fprintf(fp, "# step  KE  PE  T  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz\n");
    fflush(fp);
}

void ThermoBuffers::close_file() {
    if (fp) { fclose(fp); fp = nullptr; }
}

__global__ void kinetic_stress_kernel(const float4* __restrict__ vel,
                                        const float4* __restrict__ pos,
                                        float* __restrict__ kin_stress,
                                        int natoms) {
    __shared__ float sdata[6 * 32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float vx = 0, vy = 0, vz = 0;
    float m_i = 1.0f;
    if (i < natoms) {
        float4 v = vel[i];
        vx = v.x; vy = v.y; vz = v.z;
        int type_i = unpack_type_id(pos[i].w);
        m_i = c_masses[type_i];
    }

    float s[6] = {m_i * vx*vx, m_i * vx*vy, m_i * vx*vz,
                  m_i * vy*vy, m_i * vy*vz, m_i * vz*vz};

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int c = 0; c < 6; c++)
            s[c] += __shfl_down_sync(0xFFFFFFFF, s[c], offset);
    }

    if (tid % 32 == 0) {
        for (int c = 0; c < 6; c++)
            atomicAdd(&kin_stress[c], s[c]);
    }
}

__global__ void sum_pe_kernel(const float4* __restrict__ force,
                               float* __restrict__ pe_out,
                               int natoms) {
    __shared__ float sdata[32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (i < natoms) {
        val = force[i].w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (tid % 32 == 0) {
        atomicAdd(pe_out, val);
    }
}

void compute_thermo(const float4* vel, const float4* pos, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, ThermoBuffers& bufs,
                     int step, FILE* fp,
                     cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(bufs.d_kin_stress, 0, 6 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(bufs.d_pe, 0, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(bufs.d_max_vel, 0, 2 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(bufs.d_max_force, 0, 2 * sizeof(float), stream));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256, 0, stream>>>(vel, pos, bufs.d_kin_stress, natoms);
    sum_pe_kernel<<<blocks, 256, 0, stream>>>(force, bufs.d_pe, natoms);
    max_vel_force_kernel<<<blocks, 256, 0, stream>>>(vel, force, bufs.d_max_vel, bufs.d_max_force, natoms);

    float h_kin_stress[6];
    float h_pe;
    float h_max_vel[2], h_max_force[2];
    CUDA_CHECK(cudaMemcpyAsync(h_kin_stress, bufs.d_kin_stress, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_pe, bufs.d_pe, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_max_vel, bufs.d_max_vel, 2 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_max_force, bufs.d_max_force, 2 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    float h_virial[6];
    CUDA_CHECK(cudaMemcpyAsync(h_virial, virial, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ke = 0.5f * (h_kin_stress[0] + h_kin_stress[3] + h_kin_stress[5]);

    float inv_n = 1.0f / natoms;
    h_output->kinetic_energy = ke * inv_n;
    h_output->potential_energy = h_pe * inv_n;
    h_output->temperature = 2.0f * h_output->kinetic_energy / 3.0f;

    h_output->max_vel = sqrtf(h_max_vel[0]);
    std::memcpy(&h_output->max_vel_idx, &h_max_vel[1], sizeof(int));
    h_output->max_force = sqrtf(h_max_force[0]);
    std::memcpy(&h_output->max_force_idx, &h_max_force[1], sizeof(int));

    float vol = box_L * box_L * box_L;
    float inv_vol = 1.0f / vol;
    for (int c = 0; c < 6; c++) {
        h_output->stress[c] = (h_kin_stress[c] + h_virial[c]) * inv_vol;
    }

    if (fp) {
        fprintf(fp, "%d  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f\n",
                step,
                h_output->kinetic_energy,
                h_output->potential_energy,
                h_output->temperature,
                h_output->stress[0], h_output->stress[1], h_output->stress[2],
                h_output->stress[3], h_output->stress[4], h_output->stress[5]);
        fflush(fp);
    }
}
