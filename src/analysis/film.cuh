#pragma once
#include "../core/types.cuh"

// Film reference plane: the mass-weighted mean of the unwrapped coordinate
// along the film normal (axis 0/1/2),
//     c = sum_i m_i (x_i + n_i L) / sum_i m_i,
// computed on the device each step (graph-compatible). Depth-dependent
// friction and layer bins are defined relative to it, so they follow the
// film if it drifts. d_center receives c in unwrapped coordinates;
// d_accum (double) and d_ticket are scratch (zero between launches).
void launch_film_center(const float4* pos, const int* image, int natoms, int axis,
                        const Box* d_box, double inv_total_mass, double* d_accum,
                        unsigned int* d_ticket, float* d_center, cudaStream_t stream);
