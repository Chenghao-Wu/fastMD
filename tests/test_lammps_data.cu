#include <gtest/gtest.h>
#include "io/config.hpp"
#include "io/lammps_data.hpp"

TEST(LammpsDataParser, MiniFile) {
    TopologyData topo;
    parse_lammps_data("../tests/fixtures/mini.data", topo);

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
