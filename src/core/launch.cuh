#pragma once
#include "types.cuh"
#include <cstdlib>

// Threads cooperating on one particle in the neighbor-list pair kernels.
//
// With one thread per particle a system of N particles provides only
// N / (32 * SMs) warps per SM; below roughly full occupancy the per-thread
// neighbor loops are latency bound. Choose the smallest power of two
// (1..8) giving at least ~1536 threads per SM. FASTMD_TPA=1|2|4|8 overrides.
inline int pair_threads_per_atom(int natoms) {
    static int sms = 0;
    static int forced = -1;
    if (forced < 0) {
        const char* env = std::getenv("FASTMD_TPA");
        forced = env ? std::atoi(env) : 0;
        if (forced != 0 && forced != 1 && forced != 2 && forced != 4 && forced != 8) forced = 0;
    }
    if (forced > 0) return forced;
    if (sms == 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    }
    const long target = 1536L * sms;
    int tpa = 1;
    while (tpa < 8 && (long)natoms * tpa < target) tpa *= 2;
    return tpa;
}
