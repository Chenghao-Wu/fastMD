#include "dihedral_opls.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// Port of LAMMPS DihedralOPLS::compute() (src/MOLECULE/dihedral_opls.cpp).
// Energy and force depend only on c = cos(phi); the gradient w.r.t. c is a
// polynomial in c (sin(n*phi)/sin(phi) = U_{n-1}(c)), so no atan2 is needed.
// LAMMPS stores k_i/2 internally; here ki[] are the full Fourier coeffs and
// the 0.5 factors are applied inline.
__global__ void dihedral_opls_kernel(
    const float4* __restrict__ pos,
    float4* __restrict__ force,
    double* __restrict__ virial,
    const int4* __restrict__ dihedrals,
    const int* __restrict__ dihedral_types,
    const OPLSDihedralParams* __restrict__ params,
    int ndihedrals, float L, float inv_L)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;

    if (d < ndihedrals) {
        int4 dih = dihedrals[d];
        OPLSDihedralParams p = params[dihedral_types[d]];
        // LAMMPS half-factors
        float k1 = 0.5f * p.k1, k2 = 0.5f * p.k2;
        float k3 = 0.5f * p.k3, k4 = 0.5f * p.k4;

        float4 r1 = pos[dih.x];
        float4 r2 = pos[dih.y];
        float4 r3 = pos[dih.z];
        float4 r4 = pos[dih.w];

        // bond vectors (min image)
        float vb1x = min_image(r1.x - r2.x, L, inv_L);
        float vb1y = min_image(r1.y - r2.y, L, inv_L);
        float vb1z = min_image(r1.z - r2.z, L, inv_L);
        float vb2x = min_image(r3.x - r2.x, L, inv_L);
        float vb2y = min_image(r3.y - r2.y, L, inv_L);
        float vb2z = min_image(r3.z - r2.z, L, inv_L);
        float vb3x = min_image(r4.x - r3.x, L, inv_L);
        float vb3y = min_image(r4.y - r3.y, L, inv_L);
        float vb3z = min_image(r4.z - r3.z, L, inv_L);
        float vb2xm = -vb2x, vb2ym = -vb2y, vb2zm = -vb2z;

        float sb1 = 1.0f / (vb1x*vb1x + vb1y*vb1y + vb1z*vb1z);
        float sb2 = 1.0f / (vb2x*vb2x + vb2y*vb2y + vb2z*vb2z);
        float sb3 = 1.0f / (vb3x*vb3x + vb3y*vb3y + vb3z*vb3z);
        float rb1 = sqrtf(sb1);
        float rb3 = sqrtf(sb3);

        float c0 = (vb1x*vb3x + vb1y*vb3y + vb1z*vb3z) * rb1 * rb3;

        float b1mag = rsqrtf(sb1);
        float b2mag = rsqrtf(sb2);
        float b3mag = rsqrtf(sb3);

        float r12c1 = 1.0f / (b1mag * b2mag);
        float c1mag = (vb1x*vb2x + vb1y*vb2y + vb1z*vb2z) * r12c1;

        float r12c2 = 1.0f / (b2mag * b3mag);
        float c2mag = (vb2xm*vb3x + vb2ym*vb3y + vb2zm*vb3z) * r12c2;

        float sc1 = fmaxf(1.0f - c1mag*c1mag, 0.0f);
        sc1 = sqrtf(sc1);
        sc1 = fmaxf(sc1, 1e-6f);
        sc1 = 1.0f / sc1;

        float sc2 = fmaxf(1.0f - c2mag*c2mag, 0.0f);
        sc2 = sqrtf(sc2);
        sc2 = fmaxf(sc2, 1e-6f);
        sc2 = 1.0f / sc2;

        float s1 = sc1 * sc1;
        float s2 = sc2 * sc2;
        float s12 = sc1 * sc2;
        float c = (c0 + c1mag * c2mag) * s12;
        c = fminf(fmaxf(c, -1.0f), 1.0f);

        // cos(n phi) polynomials in c
        float c2 = 2.0f * c * c - 1.0f;
        float c3 = 4.0f * c * c * c - 3.0f * c;
        float c4 = 8.0f * c * c * c * c - 8.0f * c * c + 1.0f;

        // energy: k1(1+c) + k2(1-cos2) + k3(1+cos3) + k4(1-cos4)
        float pe = k1 * (1.0f + c) + k2 * (1.0f - c2) +
                   k3 * (1.0f + c3) + k4 * (1.0f - c4);

        // dp/dc via U_{n-1}(c) = sin(n phi)/sin(phi)
        float pd = k1 - 2.0f * k2 * 2.0f * c
                      + 3.0f * k3 * (4.0f * c * c - 1.0f)
                      - 4.0f * k4 * 4.0f * c * (2.0f * c * c - 1.0f);

        float a = pd;
        c = c * a;
        s12 = s12 * a;
        float a11 = c * sb1 * s1;
        float a22 = -sb2 * (2.0f * c0 * s12 - c * (s1 + s2));
        float a33 = c * sb3 * s2;
        float a12 = -r12c1 * (c1mag * c * s1 + c2mag * s12);
        float a13 = -rb1 * rb3 * s12;
        float a23 = r12c2 * (c2mag * c * s2 + c1mag * s12);

        float sx2 = a12*vb1x + a22*vb2x + a23*vb3x;
        float sy2 = a12*vb1y + a22*vb2y + a23*vb3y;
        float sz2 = a12*vb1z + a22*vb2z + a23*vb3z;

        float f1x = a11*vb1x + a12*vb2x + a13*vb3x;
        float f1y = a11*vb1y + a12*vb2y + a13*vb3y;
        float f1z = a11*vb1z + a12*vb2z + a13*vb3z;

        float f2x = -sx2 - f1x;
        float f2y = -sy2 - f1y;
        float f2z = -sz2 - f1z;

        float f4x = a13*vb1x + a23*vb2x + a33*vb3x;
        float f4y = a13*vb1y + a23*vb2y + a33*vb3y;
        float f4z = a13*vb1z + a23*vb2z + a33*vb3z;

        float f3x = sx2 - f4x;
        float f3y = sy2 - f4y;
        float f3z = sz2 - f4z;

        atomicAdd(&force[dih.x].x, f1x);
        atomicAdd(&force[dih.x].y, f1y);
        atomicAdd(&force[dih.x].z, f1z);
        atomicAdd(&force[dih.y].x, f2x);
        atomicAdd(&force[dih.y].y, f2y);
        atomicAdd(&force[dih.y].z, f2z);
        atomicAdd(&force[dih.z].x, f3x);
        atomicAdd(&force[dih.z].y, f3y);
        atomicAdd(&force[dih.z].z, f3z);
        atomicAdd(&force[dih.w].x, f4x);
        atomicAdd(&force[dih.w].y, f4y);
        atomicAdd(&force[dih.w].z, f4z);

        float pe4 = 0.25f * pe;
        atomicAdd(&force[dih.x].w, pe4);
        atomicAdd(&force[dih.y].w, pe4);
        atomicAdd(&force[dih.z].w, pe4);
        atomicAdd(&force[dih.w].w, pe4);

        // LAMMPS ev_tally convention: v = vb1*f1 + vb2*f3 + (vb3+vb2)*f4
        float g0 = vb1x * f1x + vb2x * f3x + (vb3x + vb2x) * f4x;
        float g1 = vb1y * f1y + vb2y * f3y + (vb3y + vb2y) * f4y;
        float g2 = vb1z * f1z + vb2z * f3z + (vb3z + vb2z) * f4z;
        v0 = g0;
        v3 = g1;
        v5 = g2;
        v1 = vb1x * f1y + vb2x * f3y + (vb3x + vb2x) * f4y;
        v2 = vb1x * f1z + vb2x * f3z + (vb3x + vb2x) * f4z;
        v4 = vb1y * f1z + vb2y * f3z + (vb3y + vb2y) * f4z;
    }

    float vir[6] = {v0, v1, v2, v3, v4, v5};
    block_reduce_virial(vir, virial);
}

void launch_dihedral_opls_kernel(const float4* pos, float4* force, double* virial,
                                  const int4* dihedrals, const int* dihedral_types,
                                  const OPLSDihedralParams* params,
                                  int ndihedrals, float L, float inv_L,
                                  cudaStream_t stream) {
    if (ndihedrals == 0) return;
    int blocks = div_ceil(ndihedrals, 256);
    int smem = 0;
    dihedral_opls_kernel<<<blocks, 256, smem, stream>>>(
        pos, force, virial, dihedrals, dihedral_types, params,
        ndihedrals, L, inv_L);
}
