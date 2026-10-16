#pragma once
#include "../core/types.cuh"

// Planar walls (WallSpec in core/types.cuh) acting on every particle;
// accumulates into force[] (and force.w when ev). Wall planes ramp from
// pos_start to pos_stop over nsteps, read from the device step counter.
// A particle on or behind a wall plane sets *d_error (checked by the host
// at check points). Wall forces are external and not added to the virial.
void launch_wall_kernel(const float4* pos, float4* force, int natoms, const WallSet& walls,
                        const int* d_step, int nsteps, int* d_error, cudaStream_t stream, bool ev);

// Current plane position of wall w at a step (host mirror of the ramp).
float wall_position(const WallSpec& w, int step, int nsteps);
