#include "thermo.cuh"
#include <cub/cub.cuh>

void ThermoBuffers::allocate() {
    CUDA_CHECK(cudaMalloc(&d_kin_stress, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pe, sizeof(float)));
    allocated = true;
}

void ThermoBuffers::free() {
    CUDA_CHECK(cudaFree(d_kin_stress));
    CUDA_CHECK(cudaFree(d_pe));
    allocated = false;
}

__global__ void kinetic_stress_kernel(const float4* __restrict__ vel,
                                        float* __restrict__ kin_stress,
                                        int natoms) {
    __shared__ float sdata[6 * 32];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float vx = 0, vy = 0, vz = 0;
    if (i < natoms) {
        float4 v = vel[i];
        vx = v.x; vy = v.y; vz = v.z;
    }

    float s[6] = {vx*vx, vx*vy, vx*vz, vy*vy, vy*vz, vz*vz};

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

void compute_thermo(const float4* vel, const float4* force,
                     const float* virial, int natoms, float box_L,
                     ThermoOutput* h_output, ThermoBuffers& bufs,
                     cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(bufs.d_kin_stress, 0, 6 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(bufs.d_pe, 0, sizeof(float), stream));

    int blocks = div_ceil(natoms, 256);
    kinetic_stress_kernel<<<blocks, 256, 0, stream>>>(vel, bufs.d_kin_stress, natoms);
    sum_pe_kernel<<<blocks, 256, 0, stream>>>(force, bufs.d_pe, natoms);

    float h_kin_stress[6];
    float h_pe;
    CUDA_CHECK(cudaMemcpyAsync(h_kin_stress, bufs.d_kin_stress, 6 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_pe, bufs.d_pe, sizeof(float),
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

    float vol = box_L * box_L * box_L;
    float inv_vol = 1.0f / vol;
    for (int c = 0; c < 6; c++) {
        h_output->stress[c] = (h_kin_stress[c] + h_virial[c]) * inv_vol;
    }
}
