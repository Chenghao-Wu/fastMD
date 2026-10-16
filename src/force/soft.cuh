#pragma once
#include "../core/types.cuh"

// Soft repulsion E = A [1 + cos(pi r / rc)] for r < rc (LAMMPS pair_style
// soft), used to push apart overlapping chains of a freshly generated melt.
// The force F = A pi / rc sin(pi r / rc) stays finite at r -> 0. The
// amplitude ramps linearly over the run,
//     A(step) = A_start + (A_stop - A_start) (step - 1) / nsteps,
// with the step read from a device counter (graph-compatible). Overwrites
// force[] (like the LJ kernel).
void launch_soft_kernel(const float4* pos, float4* force, double* virial,
                        const int* neighbors, const int* num_neighbors,
                        int natoms, float rc, float A_start, float A_stop, int nsteps,
                        const int* d_step, float L, float inv_L, cudaStream_t stream,
                        int tpa, const float* d_box, bool ev);
