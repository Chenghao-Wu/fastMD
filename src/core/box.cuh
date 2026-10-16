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

// Distances between opposite faces of the (xy-sheared) cell: the cell is
// spanned by a = (Lx, 0, 0), b = (xy, Ly, 0), c = (0, 0, Lz). Neighbor
// cells and minimum-image cutoffs must fit these widths.
__host__ __device__ inline float3 perpendicular_widths(const Box& b) {
    return make_float3(b.L.x * b.L.y / sqrtf(b.L.y * b.L.y + b.xy * b.xy), b.L.y, b.L.z);
}

// Host version of wrap_into_box (data files, set_positions): wraps (x, y, z)
// into the cell, robust to round-off at the upper faces, and adds the
// crossed periods to img[0..2].
inline void wrap_position_host(float& x, float& y, float& z, int* img, const Box& b) {
    int iy = (int)std::floor(y / b.L.y);
    y -= iy * b.L.y;
    x -= iy * b.xy;
    if (y >= b.L.y) { y -= b.L.y; x -= b.xy; iy++; }
    if (y < 0.0f) { y += b.L.y; x += b.xy; iy--; }
    const float sx = (x - b.xy * y / b.L.y) / b.L.x;
    int ix = (int)std::floor(sx);
    x -= ix * b.L.x;
    if ((x - b.xy * y / b.L.y) >= b.L.x) { x -= b.L.x; ix++; }
    if ((x - b.xy * y / b.L.y) < 0.0f) { x += b.L.x; ix--; }
    int iz = (int)std::floor(z / b.L.z);
    z -= iz * b.L.z;
    if (z >= b.L.z) { z -= b.L.z; iz++; }
    if (z < 0.0f) { z += b.L.z; iz--; }
    img[0] += ix;
    img[1] += iy;
    img[2] += iz;
}

#ifdef __CUDACC__
// Minimum-image separation vector for a box with an xy tilt (orthorhombic
// when xy = 0): the y period shifts x by xy, so y is reduced first (LAMMPS
// convention; exact for |xy| <= Lx / 2 and cutoffs below half the
// perpendicular widths).
__device__ __forceinline__ float3 min_image(float3 d, const Box& b) {
    const float ny = rintf(d.y * b.inv_L.y);
    d.y -= ny * b.L.y;
    d.x -= ny * b.xy;
    d.x -= b.L.x * rintf(d.x * b.inv_L.x);
    d.z -= b.L.z * rintf(d.z * b.inv_L.z);
    return d;
}

// Minimum image ignoring tilts (kernels restricted to orthogonal boxes:
// Ewald/PME and the OPLS pair terms).
__device__ __forceinline__ float3 min_image_ortho(float3 d, const Box& b) {
    d.x -= b.L.x * rintf(d.x * b.inv_L.x);
    d.y -= b.L.y * rintf(d.y * b.inv_L.y);
    d.z -= b.L.z * rintf(d.z * b.inv_L.z);
    return d;
}

// Compile-time choice for the hottest kernels (the tilt only changes at host
// events, so launches pick the variant): TILT = false skips the xy term.
template <bool TILT>
__device__ __forceinline__ float3 min_image_dr_t(float4 r1, float4 r2, const Box& b) {
    float3 d = make_float3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    return TILT ? min_image(d, b) : min_image_ortho(d, b);
}

// Minimum-image r1 - r2.
__device__ __forceinline__ float3 min_image_dr(float4 r1, float4 r2, const Box& b) {
    return min_image(make_float3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z), b);
}
__device__ __forceinline__ float3 min_image_dr_ortho(float4 r1, float4 r2, const Box& b) {
    return min_image_ortho(make_float3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z), b);
}

// Fractional coordinates s in the cell (wrapped positions: s in [0, 1)).
__device__ __forceinline__ float3 fractional(float4 r, const Box& b) {
    const float sy = r.y * b.inv_L.y;
    return make_float3((r.x - b.xy * sy) * b.inv_L.x, sy, r.z * b.inv_L.z);
}

// Wrap a position into the cell (fractional coordinates in [0, 1)) and add
// the crossed periods to img; unwrapped x = x + ix Lx + iy xy, y = y + iy Ly.
__device__ __forceinline__ void wrap_into_box(float4& r, int* img, const Box& b) {
    const int iy = (int)floorf(r.y * b.inv_L.y);
    r.y -= iy * b.L.y;
    r.x -= iy * b.xy;
    const int ix = (int)floorf((r.x - b.xy * r.y * b.inv_L.y) * b.inv_L.x);
    r.x -= ix * b.L.x;
    const int iz = (int)floorf(r.z * b.inv_L.z);
    r.z -= iz * b.L.z;
    if (img) {
        img[0] += ix;
        img[1] += iy;
        img[2] += iz;
    }
}
#endif  // __CUDACC__
