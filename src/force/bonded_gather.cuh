#pragma once
#include "../core/types.cuh"
#include "../io/table_parser.hpp"
#include "fene.cuh"
#include "bond_harmonic.cuh"
#include "angle.cuh"
#include "dihedral_opls.cuh"

// Per-atom ("gather") evaluation of all bonded terms.
//
// Each particle walks the list of bonded terms it belongs to (a CSR in tag
// space: row tag[i], entries {term, kind * 4 + role}), evaluates each term's
// force on itself only and writes force[i] once: no atomics, one thread per
// particle (or tpa threads striding over the particle's terms). Terms are recomputed once per member atom (2x bonds, 3x angles,
// 4x dihedrals), which is cheaper than the scatter kernels' 3-12 global
// atomics per term. Energies are split equally between member atoms; the
// virial of a term is added by one designated member.

enum BondedKind { BONDED_BOND = 0, BONDED_ANGLE = 1, BONDED_DIHEDRAL = 2 };

struct BondedParams {
    int bond_style = 0;        // 0 none, 1 FENE, 2 harmonic, 3 table
    const int* bond_type = nullptr;
    const FENEParams* fene = nullptr;
    const BondHarmonicParams* harmonic = nullptr;
    const int* bond_table_idx = nullptr;
    const TableParams* bond_table_params = nullptr;
    const float4* bond_table_data = nullptr;

    int angle_style = 0;       // 0 none, 1 harmonic, 2 table, 3 cosine
    const AngleParams* angle = nullptr;
    const int* angle_table_idx = nullptr;
    const TableParams* angle_table_params = nullptr;
    const float4* angle_table_data = nullptr;

    int dihedral_style = 0;    // 0 none, 1 OPLS
    const int* dihedral_type = nullptr;
    const OPLSDihedralParams* opls = nullptr;
};

// Accumulates into force[] (call after the overwriting nonbonded kernel).
void launch_bonded_gather_kernel(const float4* pos, float4* force, double* virial,
                                 const int* tag, const int* term_offsets, const int2* terms,
                                 const int2* bonds, const int4* angles, const int4* dihedrals,
                                 const BondedParams& params,
                                 int natoms, Box box,
                                 cudaStream_t stream = 0,
                                 int tpa = 1,    // threads per particle: 1, 2, 4, 8
                                 const Box* d_box = nullptr,
                                 bool ev = true);   // false: forces only (no energy, no virial)
