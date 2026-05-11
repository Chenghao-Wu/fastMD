#include <gtest/gtest.h>
#include "io/table_parser.hpp"
#include <fstream>
#include <sstream>

TEST(TableParser, ParsesValidFile) {
    std::string content = R"(# comment
LJ_TABLE
N 5 R 0.5 1.5

1 0.5000 100.0000 -50.0000
2 0.7500  50.0000 -25.0000
3 1.0000   0.0000   0.0000
4 1.2500 -10.0000   5.0000
5 1.5000  -5.0000   2.5000
)";
    std::string fname = "/tmp/test_table.txt";
    {
        std::ofstream f(fname);
        f << content;
    }
    auto result = parse_table_file(fname, "LJ_TABLE");
    ASSERT_EQ(result.params.size(), 1u);
    ASSERT_EQ(result.data.size(), 5u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
    EXPECT_FLOAT_EQ(result.data[0].y, -50.0f);
    EXPECT_FLOAT_EQ(result.data[4].z, -5.0f);
}

TEST(TableParser, ParsesNOnlyHeader) {
    std::string content = R"(PAIR_0
N 3

1 0.5 10.0 -5.0
2 1.0  5.0 -2.5
3 1.5  0.0  0.0
)";
    std::string fname = "/tmp/test_table_nonly.txt";
    {
        std::ofstream f(fname);
        f << content;
    }
    auto result = parse_table_file(fname, "PAIR_0");
    ASSERT_EQ(result.data.size(), 3u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
}

TEST(TableParser, MissingKeywordThrows) {
    std::string fname = "/tmp/test_table_bad.txt";
    {
        std::ofstream f(fname);
        f << "WRONG\nN 2 R 0 1\n\n1 0 0 0\n2 1 0 0\n";
    }
    EXPECT_THROW(parse_table_file(fname, "RIGHT"), std::runtime_error);
}
