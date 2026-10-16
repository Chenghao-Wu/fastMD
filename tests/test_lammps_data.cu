#include <gtest/gtest.h>
#include "io/config.hpp"
#include "io/lammps_data.hpp"
#include <set>

TEST(LammpsDataParser, MiniFile) {
    TopologyData topo;
    parse_lammps_data(std::string(FASTMD_TEST_DATA_DIR) + "/mini.data", topo);

    ASSERT_EQ(topo.positions.size(), 4u);
    EXPECT_FLOAT_EQ(topo.positions[0].x, 1.0f);
    EXPECT_FLOAT_EQ(topo.positions[3].z, 6.0f);
    EXPECT_EQ(unpack_type_id(topo.positions[0].w), 0);

    ASSERT_EQ(topo.mol_ids.size(), 4u);
    EXPECT_EQ(topo.mol_ids[0], 1);
    EXPECT_EQ(topo.mol_ids[1], 1);
    EXPECT_EQ(topo.mol_ids[2], 1);
    EXPECT_EQ(topo.mol_ids[3], 1);

    ASSERT_EQ(topo.images.size(), 12u);  // 4 atoms * 3
    for (int v : topo.images) EXPECT_EQ(v, 0);

    ASSERT_EQ(topo.bonds.size(), 2u);
    EXPECT_EQ(topo.bonds[0].x, 0);
    EXPECT_EQ(topo.bonds[0].y, 1);
    EXPECT_EQ(topo.bonds[1].x, 1);
    EXPECT_EQ(topo.bonds[1].y, 2);
    ASSERT_EQ(topo.bond_types.size(), 2u);
    EXPECT_EQ(topo.bond_types[0], 0);
    EXPECT_EQ(topo.bond_types[1], 0);

    ASSERT_EQ(topo.angles.size(), 1u);
    EXPECT_EQ(topo.angles[0].x, 0);
    EXPECT_EQ(topo.angles[0].y, 1);
    EXPECT_EQ(topo.angles[0].z, 2);
    EXPECT_EQ(topo.angles[0].w, 0);
}

#include <cstdio>
#include <fstream>

// Restart files must reproduce the state exactly: coordinates (incl. a
// non-zero box origin), velocities, charges, image flags, molecule ids,
// atom ids, masses and all topology.
TEST(LammpsData, RestartRoundTrip) {
    TopologyData topo;
    topo.box = make_float3(7.5f, 8.0f, 9.25f);   // orthorhombic
    topo.box_lo = make_float3(-3.25f, -3.25f, -3.25f);
    topo.masses = {12.011f, 1.008f};
    topo.atom_ids = {10, 11, 15, 20, 21};
    const int n = 5;
    for (int i = 0; i < n; i++) {
        topo.bond_types.push_back(i % 2);
    }
    topo.bonds = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    topo.bond_types = {0, 1, 0, 1};
    topo.angles = {make_int4(0, 1, 2, 0), make_int4(1, 2, 3, 1), make_int4(2, 3, 4, 0)};
    topo.dihedrals = {make_int4(0, 1, 2, 3), make_int4(1, 2, 3, 4)};
    topo.dihedral_types = {0, 1};

    RestartFrame f;
    for (int i = 0; i < n; i++) {
        f.pos.push_back(make_float4(0.123456789f * (i + 1), 7.4999f - 1.3f * i, 0.000123f + i,
                                    pack_type_id(i % 2)));
        f.vel.push_back(make_float4(-0.5f + 0.1f * i, 1e-7f * i, 3.25f, 0));
        f.charge.push_back(i % 2 ? 0.417f : -0.834f);
        f.mol_id.push_back(1 + i / 3);
        f.image.insert(f.image.end(), {i - 2, 3 * i, -i});
    }
    write_lammps_data("restart_roundtrip.data", f, topo, 2, 2, 2, 2, make_box(topo.box), 1234);

    TopologyData back;
    parse_lammps_data("restart_roundtrip.data", back);
    ASSERT_EQ((int)back.positions.size(), n);
    EXPECT_FLOAT_EQ(back.box.x, topo.box.x);
    EXPECT_FLOAT_EQ(back.box.y, topo.box.y);
    EXPECT_FLOAT_EQ(back.box.z, topo.box.z);
    EXPECT_FLOAT_EQ(back.box_lo.x, topo.box_lo.x);
    EXPECT_EQ(back.atom_ids, topo.atom_ids);
    EXPECT_EQ(back.mol_ids, f.mol_id);
    EXPECT_EQ(back.charges, f.charge);
    EXPECT_EQ(back.images, f.image);
    ASSERT_GE(back.masses.size(), 2u);
    EXPECT_FLOAT_EQ(back.masses[0], 12.011f);
    EXPECT_FLOAT_EQ(back.masses[1], 1.008f);
    for (int i = 0; i < n; i++) {
        // positions are written in the original frame and shifted back on read
        EXPECT_NEAR(back.positions[i].x, f.pos[i].x, 2e-6f) << i;
        EXPECT_NEAR(back.positions[i].y, f.pos[i].y, 2e-6f) << i;
        EXPECT_NEAR(back.positions[i].z, f.pos[i].z, 2e-6f) << i;
        EXPECT_EQ(unpack_type_id(back.positions[i].w), unpack_type_id(f.pos[i].w));
        EXPECT_EQ(back.velocities[i].x, f.vel[i].x);
        EXPECT_EQ(back.velocities[i].y, f.vel[i].y);
        EXPECT_EQ(back.velocities[i].z, f.vel[i].z);
    }
    ASSERT_EQ(back.bonds.size(), topo.bonds.size());
    for (size_t b = 0; b < topo.bonds.size(); b++) {
        EXPECT_EQ(back.bonds[b].x, topo.bonds[b].x);
        EXPECT_EQ(back.bonds[b].y, topo.bonds[b].y);
    }
    EXPECT_EQ(back.bond_types, topo.bond_types);
    ASSERT_EQ(back.angles.size(), topo.angles.size());
    for (size_t a = 0; a < topo.angles.size(); a++) {
        EXPECT_EQ(back.angles[a].x, topo.angles[a].x);
        EXPECT_EQ(back.angles[a].z, topo.angles[a].z);
        EXPECT_EQ(back.angles[a].w, topo.angles[a].w);
    }
    ASSERT_EQ(back.dihedrals.size(), topo.dihedrals.size());
    EXPECT_EQ(back.dihedrals[1].w, topo.dihedrals[1].w);
    EXPECT_EQ(back.dihedral_types, topo.dihedral_types);
}

TEST(LammpsData, RejectsBrokenFiles) {
    auto expect_bad = [](const char* name, const std::string& body) {
        std::ofstream(name) << body;
        TopologyData t;
        EXPECT_THROW(parse_lammps_data(name, t), std::runtime_error) << name;
    };
    const std::string head = "t\n\n2 atoms\n1 bonds\n1 atom types\n1 bond types\n\n"
                             "0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\nAtoms # molecular\n\n"
                             "1 1 1 1 1 1\n2 1 1 2 1 1\n\n";
    expect_bad("bad_bond_ref.data", head + "Bonds\n\n1 1 1 3\n");      // unknown atom id
    expect_bad("bad_count.data", head);                               // 1 bond declared, 0 read
    expect_bad("bad_type.data", head + "Bonds\n\n1 2 1 2\n");         // bond type out of range
    expect_bad("bad_dup.data", "t\n\n2 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n"
                               "Atoms # atomic\n\n1 1 1 1 1\n1 1 2 1 1\n");
    expect_bad("bad_cols.data", "t\n\n1 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 5 zlo zhi\n\n"
                                "Atoms # full\n\n1 1 1 0.5 1 1\n");
}

// LAMMPS' default special_bonds derives 1-4 pairs from bond-graph distance;
// a 4-atom chain with no listed dihedral still has the (0,3) 1-4 pair.
TEST(LammpsData, OplsSpecialPairsFromBondGraph) {
    auto pairs_for = [](std::vector<int2> bonds, std::vector<int4> dihedrals, int n, bool from_dih) {
        TopologyData t;
        t.positions.assign(n, make_float4(0, 0, 0, pack_type_id(0)));
        t.bonds = bonds;
        t.dihedrals = dihedrals;
        build_exclusions_opls(t, t, from_dih);
        std::set<std::pair<int, int>> s;
        for (auto p : t.pairs14) s.insert({p.x, p.y});
        return s;
    };
    using S = std::set<std::pair<int, int>>;
    std::vector<int2> chain = {{0, 1}, {1, 2}, {2, 3}};
    EXPECT_EQ(pairs_for(chain, {}, 4, false), (S{{0, 3}}));
    EXPECT_EQ(pairs_for(chain, {}, 4, true), S{});                       // 'dihedral yes': none listed
    EXPECT_EQ(pairs_for(chain, {make_int4(0, 1, 2, 3)}, 4, true), (S{{0, 3}}));
    // rings: 1-4 pairs that are also 1-2/1-3 keep the closer classification
    EXPECT_EQ(pairs_for({{0, 1}, {1, 2}, {2, 3}, {3, 0}}, {}, 4, false), S{});
    EXPECT_EQ(pairs_for({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}, {}, 5, false), S{});
    EXPECT_EQ(pairs_for({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}}, {}, 6, false),
              (S{{0, 3}, {1, 4}, {2, 5}}));
    // branched: 1-4 across a branch point
    EXPECT_EQ(pairs_for({{0, 1}, {1, 2}, {1, 3}, {3, 4}}, {}, 5, false), (S{{0, 4}, {2, 4}}));
}
