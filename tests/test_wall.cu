#include <gtest/gtest.h>
#include "test_utils.cuh"
#include "force/wall.cuh"
#include <cmath>

namespace {

WallSpec make_wall(int axis, int side, int style, float eps, float sigma, float cutoff, float p0,
                   float p1 = NAN) {
    WallSpec w{};
    w.axis = axis; w.side = side; w.style = style;
    w.eps = eps; w.sigma = sigma; w.cutoff = cutoff;
    w.pos_start = p0; w.pos_stop = std::isnan(p1) ? p0 : p1;
    if (style == 0) {
        double sr = sigma / cutoff, sr3 = sr * sr * sr;
        w.e_shift = (float)(eps * (2.0 / 15.0 * sr3 * sr3 * sr3 - sr3));
    }
    return w;
}

// Force on and energy of one particle at coordinate x along `axis`.
float eval(const WallSet& ws, int axis, float x, int step, int nsteps, float* e, int* err) {
    std::vector<float4> pos(1, make_float4(5.0f, 5.0f, 5.0f, pack_type_id(0)));
    (&pos[0].x)[axis] = x;
    float4* d_pos = to_device(pos);
    float4* d_f = to_device(std::vector<float4>(1, make_float4(0, 0, 0, 0)));
    int* d_step = to_device(std::vector<int>{step});
    int* d_err = to_device(std::vector<int>{0});
    launch_wall_kernel(d_pos, d_f, 1, ws, d_step, nsteps, d_err, 0, true);
    auto f = to_host(d_f, 1);
    *e = f[0].w;
    if (err) *err = to_host(d_err, 1)[0];
    for (void* p : {(void*)d_pos, (void*)d_f, (void*)d_step, (void*)d_err}) CUDA_CHECK(cudaFree(p));
    return (&f[0].x)[axis];
}

}  // namespace

TEST(Wall, ForcesAreEnergyGradientsBothSidesAndStyles) {
    for (int style : {0, 1})
        for (int side : {0, 1}) {
            WallSet ws;
            ws.n = 1;
            ws.w[0] = make_wall(2, side, style, 1.3f, 0.9f, 2.0f, side == 0 ? 1.0f : 9.0f);
            for (float d : {0.6f, 0.75f, 0.95f, 1.3f, 1.9f}) {
                const float x = side == 0 ? 1.0f + d : 9.0f - d;
                float e, ep, em;
                const float h = 1e-3f;
                float F = eval(ws, 2, x, 1, 0, &e, nullptr);
                eval(ws, 2, x + h, 1, 0, &ep, nullptr);
                eval(ws, 2, x - h, 1, 0, &em, nullptr);
                EXPECT_NEAR(F, -(ep - em) / (2 * h), 2e-3f * std::max(1.0f, std::fabs(F)))
                    << "style " << style << " side " << side << " d " << d;
                // harmonic walls always push away; LJ 9-3 only inside its
                // minimum at (2/5)^(1/6) sigma (attractive further out)
                if (style == 1 || d < 0.858f * 0.9f) EXPECT_GT(side == 0 ? F : -F, 0.0f);
                if (style == 0 && d > 0.9f) EXPECT_LT(side == 0 ? F : -F, 0.0f);
            }
            float e;
            EXPECT_EQ(eval(ws, 2, side == 0 ? 3.5f : 6.5f, 1, 0, &e, nullptr), 0.0f);   // beyond cutoff
            EXPECT_EQ(e, 0.0f);
            if (style == 0) {   // shifted: E -> 0 at the cutoff (E ~ -F(rc) (d - rc))
                eval(ws, 2, side == 0 ? 2.999f : 7.001f, 1, 0, &e, nullptr);
                EXPECT_NEAR(e, 0.0f, 5e-4f);
            }
        }
}

TEST(Wall, RampAndErrorFlag) {
    WallSet ws;
    ws.n = 2;
    ws.w[0] = make_wall(0, 0, 1, 10.0f, 1.0f, 1.0f, 0.0f, 4.0f);   // xlo moves 0 -> 4
    ws.w[1] = make_wall(0, 1, 1, 10.0f, 1.0f, 1.0f, 10.0f);
    float e;
    int err;
    // step 51 of 100: xlo plane at 2.0; particle at 2.5 feels it, at 3.5 not
    float F = eval(ws, 0, 2.5f, 51, 100, &e, &err);
    EXPECT_NEAR(F, 2.0f * 10.0f * 0.5f, 1e-4f);
    EXPECT_EQ(err, 0);
    EXPECT_EQ(eval(ws, 0, 3.5f, 51, 100, &e, &err), 0.0f);
    EXPECT_NEAR(wall_position(ws.w[0], 51, 100), 2.0f, 1e-6f);
    // particle behind the moved plane -> error flag (wall index + 1)
    eval(ws, 0, 1.5f, 101, 100, &e, &err);
    EXPECT_EQ(err, 1);
    eval(ws, 0, 10.2f, 1, 100, &e, &err);
    EXPECT_EQ(err, 2);
}
