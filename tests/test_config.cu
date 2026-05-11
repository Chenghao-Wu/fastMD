#include <gtest/gtest.h>
#include "io/config.hpp"
#include <fstream>

TEST(Config, ParsesTableLine) {
    std::string cfg = "/tmp/test_config_table.conf";
    {
        std::ofstream f(cfg);
        f << "natoms 2\nntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
          << "nvt_langevin 1.0 1.0 1.0\n"
          << "table 0 0 /tmp/test_table.txt LJ_TABLE\n";
    }
    {
        std::ofstream f("/tmp/test_table.txt");
        f << "LJ_TABLE\nN 2 R 0.5 1.0\n\n1 0.5 0 0\n2 1.0 0 0\n";
    }
    TopologyData topo;
    SimParams params = parse_config(cfg, topo);
    EXPECT_EQ(topo.table_idx.size(), 1u);
    EXPECT_EQ(topo.table_idx[0], 0);
    EXPECT_EQ(topo.table_params.size(), 1u);
    EXPECT_EQ(topo.table_data.size(), 2u);
}

TEST(Config, RejectsLjAndTableConflict) {
    std::string cfg = "/tmp/test_config_conflict.conf";
    {
        std::ofstream f(cfg);
        f << "natoms 2\nntypes 1\nrc 2.5\nskin 0.3\ndt 0.001\nnsteps 1\n"
          << "nvt_langevin 1.0 1.0 1.0\n"
          << "lj 0 0 1.0 1.0\n"
          << "table 0 0 /tmp/test_table.txt LJ_TABLE\n";
    }
    {
        std::ofstream f("/tmp/test_table.txt");
        f << "LJ_TABLE\nN 2 R 0.5 1.0\n\n1 0.5 0 0\n2 1.0 0 0\n";
    }
    TopologyData topo;
    EXPECT_THROW(parse_config(cfg, topo), std::runtime_error);
}
