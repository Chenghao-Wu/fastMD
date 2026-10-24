#include <gtest/gtest.h>
#include "io/table_parser.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace {
std::string write_tmp(const std::string& name, const std::string& content) {
    std::string path = "table_parser_" + name + ".txt";
    std::ofstream(path) << content;
    return path;
}
}  // namespace

TEST(TableParser, ParsesValidFileLegacyForceOverR) {
    std::string fname = write_tmp("valid", R"(# comment
LJ_TABLE
N 5 R 0.5 1.5

1 0.5000 100.0000 -50.0000
2 0.7500  50.0000 -25.0000
3 1.0000   0.0000   0.0000
4 1.2500 -10.0000   5.0000
5 1.5000  -5.0000   2.5000
)");
    auto result = parse_table_file(fname, "LJ_TABLE", TableKind::Pair,
                                   TableForceConvention::ForceOverR);
    ASSERT_EQ(result.params.size(), 1u);
    ASSERT_EQ(result.data.size(), 5u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
    EXPECT_FLOAT_EQ(result.data[0].y, -50.0f);   // stored verbatim as F/r
    EXPECT_FLOAT_EQ(result.data[4].z, -5.0f);
}

// LAMMPS tables give the radial force F = -dE/dr in column 4; kernels need
// F/r. A standard table with F = -5 at r = 1.5 must yield F/r = -10/3
// (the previous parser stored -5 and the kernel then applied F = -7.5).
TEST(TableParser, LammpsRadialForceIsConvertedToForceOverR) {
    std::string fname = write_tmp("lammps", "PAIR\nN 3\n\n1 0.5 10.0 -5.0\n2 1.0 5.0 -5.0\n3 1.5 0.0 -5.0\n");
    auto t = parse_table_file(fname, "PAIR", TableKind::Pair, TableForceConvention::Force);
    ASSERT_EQ(t.data.size(), 3u);
    EXPECT_FLOAT_EQ(t.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(t.params[0].rmax, 1.5f);
    EXPECT_FLOAT_EQ(t.data[0].y, -10.0f);
    EXPECT_FLOAT_EQ(t.data[2].y, -5.0f / 1.5f);
    // physical force recovered by the kernel: (F/r) * r
    EXPECT_FLOAT_EQ(t.data[2].y * t.data[2].x, -5.0f);
}

// LAMMPS angle tables: theta in degrees, -dE/dtheta per degree.
TEST(TableParser, LammpsAngleTableConvertedToRadians) {
    // E = k (theta - pi/2)^2 sampled every 45 degrees, force per degree
    const double k = 2.0, d2r = M_PI / 180.0;
    std::ostringstream os;
    os << "ANG\nN 5\n\n";
    for (int i = 0; i < 5; i++) {
        double deg = 45.0 * i, th = deg * d2r;
        double e = k * (th - M_PI / 2) * (th - M_PI / 2);
        double f_per_rad = -2.0 * k * (th - M_PI / 2);
        os << i + 1 << " " << deg << " " << e << " " << f_per_rad * d2r << "\n";
    }
    std::string fname = write_tmp("angle", os.str());
    auto t = parse_table_file(fname, "ANG", TableKind::Angle, TableForceConvention::DegreesPerDegree);
    EXPECT_NEAR(t.params[0].rmin, 0.0, 1e-7);
    EXPECT_NEAR(t.params[0].rmax, M_PI, 1e-6);
    EXPECT_NEAR(t.data[1].x, M_PI / 4, 1e-6);
    EXPECT_NEAR(t.data[1].y, -2.0 * k * (M_PI / 4 - M_PI / 2), 1e-5);   // per radian
}

TEST(TableParser, ParsesNOnlyHeader) {
    std::string fname = write_tmp("nonly", "PAIR_0\nN 3\n\n1 0.5 10.0 -5.0\n2 1.0  5.0 -2.5\n3 1.5  0.0  0.0\n");
    auto result = parse_table_file(fname, "PAIR_0", TableKind::Pair, TableForceConvention::Force);
    ASSERT_EQ(result.data.size(), 3u);
    EXPECT_FLOAT_EQ(result.params[0].rmin, 0.5f);
    EXPECT_FLOAT_EQ(result.params[0].rmax, 1.5f);
}

TEST(TableParser, MissingKeywordThrows) {
    std::string fname = write_tmp("bad", "WRONG\nN 2 R 0 1\n\n1 0 0 0\n2 1 0 0\n");
    EXPECT_THROW(parse_table_file(fname, "RIGHT", TableKind::Pair, TableForceConvention::Force),
                 std::runtime_error);
}

TEST(TableParser, RejectsNonUniformSpacingAndBadInput) {
    auto f1 = write_tmp("nonuniform", "T\nN 3\n\n1 0.5 0 0\n2 0.6 0 0\n3 1.5 0 0\n");
    EXPECT_THROW(parse_table_file(f1, "T", TableKind::Pair, TableForceConvention::Force), std::runtime_error);
    auto f2 = write_tmp("short", "T\nN 4\n\n1 0.5 0 0\n2 1.0 0 0\n");
    EXPECT_THROW(parse_table_file(f2, "T", TableKind::Pair, TableForceConvention::Force), std::runtime_error);
    auto f3 = write_tmp("rsq", "T\nN 2 RSQ 0.5 1.0\n\n1 0.5 0 0\n2 1.0 0 0\n");
    EXPECT_THROW(parse_table_file(f3, "T", TableKind::Pair, TableForceConvention::Force), std::runtime_error);
    auto f4 = write_tmp("rzero", "T\nN 2\n\n1 0.0 0 0\n2 1.0 0 0\n");
    EXPECT_THROW(parse_table_file(f4, "T", TableKind::Pair, TableForceConvention::Force), std::runtime_error);
    EXPECT_THROW(parse_table_convention("degrees", false), std::runtime_error);
}

// Dihedral tables: one full period [-180, 180] degrees, force per degree -> per radian, periodic end rows.
TEST(TableParser, DihedralTableFullPeriodAndPeriodicity) {
    const double d2r = M_PI / 180.0;
    auto table = [&](double lo, double hi, double e_last_shift) {
        std::ostringstream os;
        os << "DIH\nN 5\n\n";
        for (int i = 0; i < 5; i++) {
            double deg = lo + (hi - lo) * i / 4, p = deg * d2r;
            double e = 2.0 * std::cos(p) + (i == 4 ? e_last_shift : 0.0);
            os << i + 1 << " " << deg << " " << e << " " << 2.0 * std::sin(p) * d2r << "\n";   // -dE/dphi per degree
        }
        return os.str();
    };
    auto t = parse_table_file(write_tmp("dih", table(-180, 180, 0.0)), "DIH", TableKind::Dihedral,
                              TableForceConvention::DegreesPerDegree);
    EXPECT_NEAR(t.params[0].rmin, -M_PI, 1e-6);
    EXPECT_NEAR(t.params[0].rmax, M_PI, 1e-6);
    EXPECT_NEAR(t.data[3].y, 2.0 * std::sin(M_PI / 2), 1e-5);      // per radian at +90 degrees
    EXPECT_EQ(t.params[0].flags, 0);
    EXPECT_THROW(parse_table_file(write_tmp("dih_half", table(0, 180, 0.0)), "DIH", TableKind::Dihedral,
                                  TableForceConvention::DegreesPerDegree), std::runtime_error);
    EXPECT_THROW(parse_table_file(write_tmp("dih_nonper", table(-180, 180, 0.5)), "DIH", TableKind::Dihedral,
                                  TableForceConvention::DegreesPerDegree), std::runtime_error);
    EXPECT_THROW(parse_table_file(write_tmp("dih_conv", table(-180, 180, 0.0)), "DIH", TableKind::Dihedral,
                                  TableForceConvention::Force), std::runtime_error);
}
