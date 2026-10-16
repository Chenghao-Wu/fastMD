#include "bonded_gather.cuh"
#include "bonded_terms.cuh"
#include "../core/pbc.cuh"
#include "../core/reduce.cuh"

// EV = false: forces only (no energy added to force.w, no virial).
template <int TPA, bool EV>
__global__ void bonded_gather_kernel(const float4* __restrict__ pos, float4* __restrict__ force,
                                     double* __restrict__ virial,
                                     const int* __restrict__ tag,
                                     const int* __restrict__ term_offsets,
                                     const int2* __restrict__ terms,
                                     const int2* __restrict__ bonds,
                                     const int4* __restrict__ angles,
                                     const int4* __restrict__ dihedrals,
                                     BondedParams bp, int natoms, Box box,
                                     const Box* __restrict__ d_box, float* __restrict__ atom_vir)
{
    if (d_box) box = *d_box;   // box on the device (NPT)

    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = g / TPA;
    const int sub = g & (TPA - 1);
    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    float e = 0.0f;
    float vir[6] = {0, 0, 0, 0, 0, 0};
    auto add_vir = [&](const float (&v)[6]) {
        if (EV) for (int c = 0; c < 6; c++) vir[c] += v[c];
    };
    // per-atom share of every term's virial (1/2, 1/3, 1/4) for layer stresses
    float av[6] = {0, 0, 0, 0, 0, 0};
    const bool per_atom = EV && atom_vir != nullptr;
    auto add_share = [&](const float (&v)[6], float w) {
        if (per_atom) for (int c = 0; c < 6; c++) av[c] += w * v[c];
    };

    if (i < natoms) {
        int t = tag ? tag[i] : i;
        int end = term_offsets[t + 1];
        for (int q = term_offsets[t] + sub; q < end; q += TPA) {
            int2 entry = __ldg(&terms[q]);
            int term = entry.x, kind = entry.y >> 2, role = entry.y & 3;
            if (kind == BONDED_BOND) {
                if (bp.bond_style == 0) continue;
                int2 b = __ldg(&bonds[term]);
                float3 d = min_image_dr(__ldg(&pos[b.x]), __ldg(&pos[b.y]), box);
                int type = __ldg(&bp.bond_type[term]);
                BondTerm bt;
                if (bp.bond_style == 1) {
                    bt = fene_term(d, bp.fene[type]);
                } else if (bp.bond_style == 2) {
                    bt = harmonic_bond_term(d, bp.harmonic[type]);
                } else {
                    int tidx = bp.bond_table_idx[type];
                    if (tidx < 0) continue;
                    TableParams tp = bp.bond_table_params[tidx];
                    bt = table_bond_term(d, tp, bp.bond_table_data + tp.data_offset);
                }
                float s = role == 0 ? 1.0f : -1.0f;
                f.x += s * bt.fi.x; f.y += s * bt.fi.y; f.z += s * bt.fi.z;
                e += 0.5f * bt.e;
                if (role == 0) add_vir(bt.vir);
                add_share(bt.vir, 0.5f);
            } else if (kind == BONDED_ANGLE) {
                if (bp.angle_style == 0) continue;
                int4 a = __ldg(&angles[term]);
                float4 rj = __ldg(&pos[a.y]);
                float3 dij = min_image_dr(__ldg(&pos[a.x]), rj, box);
                float3 dkj = min_image_dr(__ldg(&pos[a.z]), rj, box);
                AngleTerm at;
                if (bp.angle_style == 1) {
                    at = harmonic_angle_term(dij, dkj, bp.angle[a.w]);
                } else if (bp.angle_style == 3) {
                    at = cosine_angle_term(dij, dkj, bp.angle[a.w]);
                } else {
                    int tidx = bp.angle_table_idx[a.w];
                    if (tidx < 0) continue;
                    TableParams tp = bp.angle_table_params[tidx];
                    at = table_angle_term(dij, dkj, tp, bp.angle_table_data + tp.data_offset);
                }
                if (role == 0) { f.x += at.fi.x; f.y += at.fi.y; f.z += at.fi.z; }
                else if (role == 2) { f.x += at.fk.x; f.y += at.fk.y; f.z += at.fk.z; }
                else {
                    f.x -= at.fi.x + at.fk.x; f.y -= at.fi.y + at.fk.y; f.z -= at.fi.z + at.fk.z;
                    add_vir(at.vir);
                }
                e += at.e * (1.0f / 3.0f);
                add_share(at.vir, 1.0f / 3.0f);
            } else {
                if (bp.dihedral_style == 0) continue;
                int4 d4 = __ldg(&dihedrals[term]);
                float4 r2 = __ldg(&pos[d4.y]), r3 = __ldg(&pos[d4.z]);
                DihedralTerm dt = opls_dihedral_term(min_image_dr(__ldg(&pos[d4.x]), r2, box),
                                                     min_image_dr(r3, r2, box),
                                                     min_image_dr(__ldg(&pos[d4.w]), r3, box),
                                                     bp.opls[__ldg(&bp.dihedral_type[term])]);
                float3 fr = role == 0 ? dt.f1 : role == 1 ? dt.f2 : role == 2 ? dt.f3 : dt.f4;
                f.x += fr.x; f.y += fr.y; f.z += fr.z;
                e += 0.25f * dt.e;
                if (role == 0) add_vir(dt.vir);
                add_share(dt.vir, 0.25f);
            }
        }
    }
    if (TPA > 1) {
#pragma unroll
        for (int o = TPA / 2; o > 0; o >>= 1) {
            f.x += __shfl_down_sync(0xffffffffu, f.x, o, TPA);
            f.y += __shfl_down_sync(0xffffffffu, f.y, o, TPA);
            f.z += __shfl_down_sync(0xffffffffu, f.z, o, TPA);
            if (EV) e += __shfl_down_sync(0xffffffffu, e, o, TPA);
        }
    }
    if (i < natoms && sub == 0) {
        float4 fi = force[i];
        force[i] = make_float4(fi.x + f.x, fi.y + f.y, fi.z + f.z, EV ? fi.w + e : fi.w);
    }
    if (per_atom) {   // uniform: atom_vir is a kernel argument
        if (TPA > 1)
            for (int o = TPA / 2; o > 0; o >>= 1)
                for (int c = 0; c < 6; c++) av[c] += __shfl_down_sync(0xffffffffu, av[c], o, TPA);
        if (i < natoms && sub == 0)
            for (int c = 0; c < 6; c++) atom_vir[c * natoms + i] += av[c];
    }
    if (EV) block_reduce_virial(vir, virial);
}

void launch_bonded_gather_kernel(const float4* pos, float4* force, double* virial,
                                 const int* tag, const int* term_offsets, const int2* terms,
                                 const int2* bonds, const int4* angles, const int4* dihedrals,
                                 const BondedParams& params,
                                 int natoms, Box box, cudaStream_t stream,
                                 int tpa, const Box* d_box, bool ev, float* atom_vir) {
    if (!term_offsets) return;
    if (tpa != 1 && tpa != 2 && tpa != 4 && tpa != 8) tpa = 1;
    const int blocks = div_ceil(natoms * tpa, 256);
#define FASTMD_LAUNCH_EV(T, E) bonded_gather_kernel<T, E><<<blocks, 256, 0, stream>>>( \
        pos, force, virial, tag, term_offsets, terms, bonds, angles, dihedrals,    \
        params, natoms, box, d_box, atom_vir)
#define FASTMD_LAUNCH(T) do { if (ev) FASTMD_LAUNCH_EV(T, true); else FASTMD_LAUNCH_EV(T, false); } while (0)
    switch (tpa) {
        case 1: FASTMD_LAUNCH(1); break;
        case 2: FASTMD_LAUNCH(2); break;
        case 4: FASTMD_LAUNCH(4); break;
        default: FASTMD_LAUNCH(8); break;
    }
#undef FASTMD_LAUNCH
#undef FASTMD_LAUNCH_EV
}
