#pragma once
#include "../core/types.cuh"

// Extensive stress of layers parallel to a film: layer k (k = 0..n-1)
// holds the particles whose unwrapped coordinate along the film normal,
// relative to the film reference plane, lies in
// [(k - n/2) width, (k + 1 - n/2) width); the outermost layers also take
// everything beyond, so the layers sum to the whole system. Per layer
//     P^(k)_ab = sum_{i in k} (m v_a v_b + W_i,ab),
// with W_i the per-atom virial (pair terms half to each atom, bonded terms
// split equally among their atoms). Divide by k_B T V for moduli.
static constexpr int kMaxLayers = 32;

struct LayerSpec {
    int   n = 0;
    float width = 0.0f;
    int   axis = 2;      // film normal
    int   shear = 1;     // in-plane shear component (VXY for a z normal)
};

// d_out: [n][6] (xx xy xz yy yz zz); d_shear: [n] the in-plane shear
// component (correlator input). d_vscale (optional): pending velocity
// scale (fused Nose-Hoover NVT). d_accum ([n][6] double) and d_ticket are
// zeroed scratch.
void launch_layer_stress(const float4* pos, const float4* vel, const int* image,
                         const float* atom_vir, int natoms, const Box* d_box,
                         const float* d_center, const LayerSpec& spec, const float* d_vscale,
                         double* d_accum, unsigned int* d_ticket, double* d_out, double* d_shear,
                         cudaStream_t stream);
