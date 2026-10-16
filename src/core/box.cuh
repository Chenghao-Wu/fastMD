#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Periodic simulation box: edge lengths L = (Lx, Ly, Lz) with cached
// inverses, and LAMMPS-convention tilt factors (xy, xz, yz) for sheared
// boxes (zero for orthorhombic boxes). Kernels take the box by value, or
// read it from device memory when it changes on the device (NPT).
struct Box {
    float3 L;
    float3 inv_L;
    float xy, xz, yz;
    float pad_ = 0.0f;

    __host__ __device__ float volume() const { return L.x * L.y * L.z; }
    __host__ __device__ float min_length() const { return fminf(L.x, fminf(L.y, L.z)); }
    __host__ __device__ bool tilted() const { return xy != 0.0f || xz != 0.0f || yz != 0.0f; }
    __host__ __device__ bool cubic() const { return L.x == L.y && L.y == L.z && !tilted(); }
};

__host__ __device__ inline Box make_box(float lx, float ly, float lz) {
    Box b;
    b.L = make_float3(lx, ly, lz);
    b.inv_L = make_float3(1.0f / lx, 1.0f / ly, 1.0f / lz);
    b.xy = b.xz = b.yz = 0.0f;
    b.pad_ = 0.0f;
    return b;
}

__host__ __device__ inline Box make_box(float3 L) { return make_box(L.x, L.y, L.z); }
__host__ __device__ inline Box cubic_box(float L) { return make_box(L, L, L); }

// Box scaled by s along every axis (isotropic barostat).
__host__ __device__ inline Box scaled_box(const Box& b, float s) {
    Box r = make_box(b.L.x * s, b.L.y * s, b.L.z * s);
    r.xy = b.xy * s; r.xz = b.xz * s; r.yz = b.yz * s;
    return r;
}

#ifdef __CUDACC__
// Minimum-image separation vector (orthorhombic box).
__device__ __forceinline__ float3 min_image(float3 d, const Box& b) {
    d.x -= b.L.x * rintf(d.x * b.inv_L.x);
    d.y -= b.L.y * rintf(d.y * b.inv_L.y);
    d.z -= b.L.z * rintf(d.z * b.inv_L.z);
    return d;
}

// Minimum-image r1 - r2.
__device__ __forceinline__ float3 min_image_dr(float4 r1, float4 r2, const Box& b) {
    return min_image(make_float3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z), b);
}

// Wrap a position into [0, L) and add the crossed periods to img.
__device__ __forceinline__ void wrap_into_box(float4& r, int* img, const Box& b) {
    int ix = (int)floorf(r.x * b.inv_L.x);
    int iy = (int)floorf(r.y * b.inv_L.y);
    int iz = (int)floorf(r.z * b.inv_L.z);
    r.x -= ix * b.L.x;
    r.y -= iy * b.L.y;
    r.z -= iz * b.L.z;
    if (img) {
        img[0] += ix;
        img[1] += iy;
        img[2] += iz;
    }
}
#endif  // __CUDACC__
