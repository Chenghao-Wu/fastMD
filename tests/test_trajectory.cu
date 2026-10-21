#include <gtest/gtest.h>
#include "io/trajectory.hpp"
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>

namespace {
struct P { float x, y, z, w; };
}

TEST(Trajectory, LogScheduleOffsets) {
    TrajectorySchedule s;
    s.set_log(100, 2.0);
    EXPECT_EQ(s.offsets, (std::vector<int>{0, 1, 2, 4, 8, 16, 32, 64}));
    for (long long step : {0LL, 1LL, 64LL, 100LL, 101LL, 164LL}) EXPECT_TRUE(s.due(step)) << step;
    for (long long step : {3LL, 65LL, 99LL, 103LL}) EXPECT_FALSE(s.due(step)) << step;
    s.set_log(30, 1.5);   // distinct rounded offsets only
    for (size_t k = 1; k < s.offsets.size(); k++) EXPECT_GT(s.offsets[k], s.offsets[k - 1]);
    EXPECT_EQ(s.offsets.front(), 0);
    EXPECT_LT(s.offsets.back(), 30);
    s.set_linear(7);
    EXPECT_TRUE(s.due(14));
    EXPECT_FALSE(s.due(15));
    EXPECT_THROW(s.set_log(10, 1.0), std::invalid_argument);
    EXPECT_THROW(s.set_linear(0), std::invalid_argument);
}

TEST(Trajectory, FileLayout) {
    const char* path = "test_traj_layout.fmd";
    {
        TrajectoryWriter w;
        w.open(path, {0, 1, 1}, {1, 1, 2});
        const double box[6] = {10, 11, 12, 0.5, 0, 0};
        std::vector<P> pos = {{1, 2, 3, 0}, {4, 5, 6, 0}, {7, 8, 9, 0}};
        w.write_frame(0, box, pos, {0, 0, 0, 1, -1, 0, 0, 0, 2});
        pos[0].x = 1.5f;
        w.write_frame(10, box, pos, {0, 0, 0, 1, -1, 0, 0, 0, 3});
        EXPECT_EQ(w.frames(), 2);
    }
    std::ifstream in(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t head = 24 + 8 * 3, frame = 8 + 48 + 36 + 36;
    ASSERT_EQ(b.size(), head + 2 * frame);
    EXPECT_EQ(std::memcmp(b.data(), "FMDTRJ02", 8), 0);
    int32_t h[4];
    std::memcpy(h, b.data() + 8, 16);
    EXPECT_EQ(h[0], 2);
    EXPECT_EQ(h[1], 3);
    EXPECT_EQ(h[2], 1);
    int64_t step;
    std::memcpy(&step, b.data() + head + frame, 8);
    EXPECT_EQ(step, 10);
    float x0;
    std::memcpy(&x0, b.data() + head + frame + 56, 4);
    EXPECT_EQ(x0, 1.5f);
    int32_t iz2;
    std::memcpy(&iz2, b.data() + head + frame + 56 + 36 + 8 * 4, 4);
    EXPECT_EQ(iz2, 3);
}

#ifdef __linux__
TEST(Trajectory, ReportsFilesystemWriteFailure) {
    // /dev/full accepts fopen but fails when the buffered header is flushed.
    // A silent partial trajectory is worse than a clear sampling error.
    TrajectoryWriter w;
    EXPECT_THROW(w.open("/dev/full", {0, 1}, {1, 1}), std::runtime_error);
    EXPECT_FALSE(w.is_open());
}
#endif
