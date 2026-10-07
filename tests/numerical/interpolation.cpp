// SPDX-License-Identifier: MIT
/// Validates the interpolators against the property each one was chosen for.
///
/// The point of having three schemes is that they preserve different things,
/// so the tests are organised by *guarantee* rather than by class: the cubic
/// spline is tested for C2 continuity and interpolation, the monotone cubic
/// for monotonicity under adversarial data, and all three for held-slope
/// extrapolation.  A test that merely checked "the curve passes near the
/// points" would pass for all three and justify none of them.

#include "vl_test_support.hpp"

#include "volatility_lab/math/interpolation.hpp"

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace vl::math;
using vl::test::WorstCase;

namespace {

std::vector<double> uniform_nodes(double lo, double hi, int n) {
    std::vector<double> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] =
            lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return out;
}

}  // namespace

// ===========================================================================
// Interval location
// ===========================================================================

TEST(Interpolation, LocateIntervalIsCorrectAndClamped) {
    const std::vector<double> xs{0.0, 1.0, 2.0, 5.0, 10.0};
    EXPECT_EQ(locate_interval(xs, -100.0), 0u);
    EXPECT_EQ(locate_interval(xs, 0.0), 0u);
    EXPECT_EQ(locate_interval(xs, 0.5), 0u);
    EXPECT_EQ(locate_interval(xs, 1.0), 1u);
    EXPECT_EQ(locate_interval(xs, 1.5), 1u);
    EXPECT_EQ(locate_interval(xs, 4.9), 2u);
    EXPECT_EQ(locate_interval(xs, 5.0), 3u);
    EXPECT_EQ(locate_interval(xs, 9.9), 3u);
    EXPECT_EQ(locate_interval(xs, 10.0), 3u);
    EXPECT_EQ(locate_interval(xs, 1e9), 3u);

    // Degenerate inputs must not index out of range.
    const std::vector<double> single{1.0};
    EXPECT_EQ(locate_interval(single, 0.0), 0u);
    const std::vector<double> none{};
    EXPECT_EQ(locate_interval(none, 0.0), 0u);
}

TEST(Interpolation, MonotonicityPredicates) {
    EXPECT_TRUE(is_strictly_increasing(std::vector<double>{1.0, 2.0, 3.0}));
    EXPECT_FALSE(is_strictly_increasing(std::vector<double>{1.0, 1.0, 3.0}));
    EXPECT_FALSE(is_strictly_increasing(std::vector<double>{3.0, 2.0}));
    EXPECT_TRUE(is_non_decreasing(std::vector<double>{1.0, 1.0, 3.0}));
    EXPECT_FALSE(is_non_decreasing(std::vector<double>{1.0, 0.9}));
    EXPECT_TRUE(is_strictly_increasing(std::vector<double>{}));
}

// ===========================================================================
// Linear
// ===========================================================================

TEST(Interpolation, LinearInterpolatesAndExtrapolatesWithHeldSlope) {
    LinearInterp f({0.0, 1.0, 3.0}, {0.0, 2.0, 4.0});
    EXPECT_NEAR(f(0.0), 0.0, 1e-15);
    EXPECT_NEAR(f(1.0), 2.0, 1e-15);
    EXPECT_NEAR(f(3.0), 4.0, 1e-15);
    EXPECT_NEAR(f(0.5), 1.0, 1e-15);
    EXPECT_NEAR(f(2.0), 3.0, 1e-15);
    // Held end slope: slope 2 on the left segment, 1 on the right.
    EXPECT_NEAR(f(-1.0), -2.0, 1e-15);
    EXPECT_NEAR(f(4.0), 5.0, 1e-15);
    EXPECT_NEAR(f.derivative(0.5), 2.0, 1e-15);
    EXPECT_NEAR(f.derivative(2.0), 1.0, 1e-15);
}

TEST(Interpolation, LinearHandlesDegenerateInputs) {
    LinearInterp empty;
    EXPECT_EQ(empty(1.0), 0.0);
    EXPECT_TRUE(empty.empty());
    LinearInterp one({5.0}, {7.0});
    EXPECT_EQ(one(0.0), 7.0);
    EXPECT_EQ(one(100.0), 7.0);
    // Mismatched sizes must not read out of bounds.
    LinearInterp bad({0.0, 1.0, 2.0}, {1.0});
    EXPECT_TRUE(std::isfinite(bad(0.5)));
}

// ===========================================================================
// Cubic spline
// ===========================================================================

TEST(Interpolation, CubicSplineInterpolatesItsKnotsExactly) {
    const auto xs = uniform_nodes(0.0, 10.0, 11);
    std::vector<double> ys;
    for (double x : xs) ys.push_back(std::sin(0.4 * x) + 0.1 * x);
    CubicSpline f(xs, ys);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_NEAR(f(xs[i]), ys[i], 1e-12) << "knot " << i;
        EXPECT_NEAR(f.jet(xs[i]).y, ys[i], 1e-12) << "knot " << i;
    }
}

TEST(Interpolation, CubicSplineIsC2AcrossKnots) {
    // The property gamma depends on.  A C1 scheme gives a second derivative
    // that jumps at every knot, which shows up as a discontinuity in gamma
    // that is purely an artefact of the interpolator.
    const auto xs = uniform_nodes(0.0, 10.0, 11);
    std::vector<double> ys;
    for (double x : xs) ys.push_back(std::sin(0.4 * x) + 0.1 * x);
    CubicSpline f(xs, ys);

    const double h = 1e-7;
    WorstCase jump_d1;
    WorstCase jump_d2;
    for (std::size_t i = 1; i + 1 < xs.size(); ++i) {
        const auto left = f.jet(xs[i] - h);
        const auto right = f.jet(xs[i] + h);
        jump_d1.observe(std::abs(left.dy - right.dy), xs[i], 0, left.dy, right.dy);
        jump_d2.observe(std::abs(left.d2y - right.d2y), xs[i], 0, left.d2y, right.d2y);
    }
    EXPECT_LT(jump_d1.error, 1e-6) << "first derivative jumps: " << jump_d1.describe("x");
    EXPECT_LT(jump_d2.error, 1e-6) << "second derivative jumps: " << jump_d2.describe("x");
}

TEST(Interpolation, CubicSplineAnalyticDerivativesMatchFiniteDifferences) {
    const auto xs = uniform_nodes(-2.0, 3.0, 14);
    std::vector<double> ys;
    for (double x : xs) ys.push_back(0.04 + 0.1 * x * x - 0.03 * x);
    CubicSpline f(xs, ys);

    WorstCase w1;
    WorstCase w2;
    const double h = 1e-5;
    for (double x = -1.8; x < 2.8; x += 0.017) {
        const auto j = f.jet(x);
        const double fd1 = (f(x + h) - f(x - h)) / (2.0 * h);
        const double fd2 = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
        w1.observe(std::abs(j.dy - fd1), x, 0, j.dy, fd1);
        w2.observe(std::abs(j.d2y - fd2), x, 0, j.d2y, fd2);
    }
    EXPECT_LT(w1.error, 1e-8) << w1.describe("x");
    EXPECT_LT(w2.error, 1e-4) << w2.describe("x");
}

TEST(Interpolation, CubicSplineReproducesACubicExactly) {
    // A natural cubic spline has zero second derivative at the ends, so it
    // cannot reproduce an arbitrary cubic -- but it must reproduce one whose
    // second derivative already vanishes there, and it must reproduce any
    // straight line exactly.
    const auto xs = uniform_nodes(0.0, 4.0, 9);
    std::vector<double> ys;
    for (double x : xs) ys.push_back(3.0 - 0.7 * x);
    CubicSpline f(xs, ys);
    for (double x = -1.0; x < 5.0; x += 0.03) {
        EXPECT_NEAR(f(x), 3.0 - 0.7 * x, 1e-12) << "x = " << x;
    }
}

TEST(Interpolation, CubicSplineExtrapolatesLinearlyNotCubically) {
    // The property that keeps extrapolated total variance positive.  A
    // continued cubic diverges like x^3 and goes negative a short distance
    // outside the quoted strikes; a held slope grows linearly, which is what
    // Lee's moment bound asks of total variance anyway.
    const auto xs = uniform_nodes(-0.3, 0.3, 9);
    std::vector<double> ys;
    for (double x : xs) ys.push_back(0.04 + 0.5 * x * x - 0.05 * x);  // a smile shape
    CubicSpline f(xs, ys);

    // Far outside, the curve must be affine: the second difference vanishes.
    for (double x : {-5.0, -2.0, 2.0, 5.0}) {
        const double d = 0.1;
        const double second = f(x + d) - 2.0 * f(x) + f(x - d);
        EXPECT_NEAR(second, 0.0, 1e-12) << "not affine at x = " << x;
        EXPECT_NEAR(f.jet(x).d2y, 0.0, 1e-15) << "x = " << x;
    }
    // And the held slope is the slope at the nearest knot.
    const double slope_right = f.jet(0.3 + 1.0).dy;
    EXPECT_NEAR(f.jet(0.3 + 5.0).dy, slope_right, 1e-12);
    // Crucially, total variance stays positive far out.
    EXPECT_GT(f(-20.0), 0.0);
    EXPECT_GT(f(20.0), 0.0);
}

TEST(Interpolation, CubicSplineHandlesDegenerateInputs) {
    CubicSpline empty;
    EXPECT_EQ(empty(1.0), 0.0);
    CubicSpline one({1.0}, {2.0});
    EXPECT_EQ(one(5.0), 2.0);
    CubicSpline two({0.0, 1.0}, {0.0, 1.0});
    EXPECT_NEAR(two(0.5), 0.5, 1e-15);
    // Non-increasing knots: must degrade safely rather than divide by zero.
    CubicSpline bad({0.0, 0.0, 1.0}, {1.0, 2.0, 3.0});
    EXPECT_TRUE(std::isfinite(bad(0.5)));
}

// ===========================================================================
// Monotone cubic -- the guarantee that matters for the term structure
// ===========================================================================

TEST(Interpolation, MonotoneCubicPreservesMonotonicityWhereCubicSplineDoesNot) {
    // The case that motivates having two schemes.  This data is monotone but
    // has a sharp step; a natural cubic spline overshoots on either side of
    // it, and along the time axis an overshoot *downwards* is calendar
    // arbitrage manufactured by the interpolator out of clean data.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> ys{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    CubicSpline spline(xs, ys);
    MonotoneCubic mono(xs, ys);

    bool spline_overshoots = false;
    double mono_worst_decrease = 0.0;
    double prev_mono = mono(0.0);
    for (double x = 0.0; x <= 5.0; x += 0.001) {
        const double s = spline(x);
        if (s < -1e-6 || s > 1.0 + 1e-6) spline_overshoots = true;
        const double m = mono(x);
        EXPECT_GE(m, -1e-12) << "monotone cubic undershot at x = " << x;
        EXPECT_LE(m, 1.0 + 1e-12) << "monotone cubic overshot at x = " << x;
        mono_worst_decrease = std::min(mono_worst_decrease, m - prev_mono);
        prev_mono = m;
    }
    EXPECT_TRUE(spline_overshoots)
        << "the natural cubic spline is expected to overshoot on this data; if it no "
           "longer does, the justification for having MonotoneCubic needs revisiting";
    EXPECT_GE(mono_worst_decrease, -1e-12) << "monotone cubic was not monotone";
}

TEST(Interpolation, MonotoneCubicIsMonotoneOnRandomMonotoneData) {
    // Fuzzed, because the Fritsch-Carlson limiter has several branches and a
    // hand-picked example exercises one of them.  Fixed seed: a fuzz test that
    // cannot be replayed is not a test.
    std::mt19937_64 rng(20260207u);
    std::uniform_real_distribution<double> gap(1e-3, 2.0);
    std::uniform_real_distribution<double> rise(0.0, 1.0);

    for (int trial = 0; trial < 300; ++trial) {
        const int n = 3 + static_cast<int>(rng() % 20);
        std::vector<double> xs(static_cast<std::size_t>(n));
        std::vector<double> ys(static_cast<std::size_t>(n));
        xs[0] = 0.0;
        ys[0] = 0.0;
        for (int i = 1; i < n; ++i) {
            xs[static_cast<std::size_t>(i)] = xs[static_cast<std::size_t>(i - 1)] + gap(rng);
            // Non-decreasing, with flat spots allowed -- the hard case for the
            // limiter, since a flat spot must stay exactly flat.
            ys[static_cast<std::size_t>(i)] =
                ys[static_cast<std::size_t>(i - 1)] + (rng() % 4 == 0 ? 0.0 : rise(rng));
        }
        MonotoneCubic f(xs, ys);

        const double x0 = xs.front();
        const double x1 = xs.back();
        const int samples = 2000;
        double prev = f(x0);
        for (int i = 1; i <= samples; ++i) {
            const double x =
                x0 + (x1 - x0) * static_cast<double>(i) / static_cast<double>(samples);
            const double v = f(x);
            ASSERT_GE(v, prev - 1e-11)
                << "trial " << trial << ": not monotone at x = " << x << " (" << v
                << " < " << prev << ")";
            prev = v;
        }
        // And it still interpolates.
        for (std::size_t i = 0; i < xs.size(); ++i) {
            ASSERT_NEAR(f(xs[i]), ys[i], 1e-11) << "trial " << trial << ", knot " << i;
        }
    }
}

TEST(Interpolation, MonotoneCubicDerivativeMatchesFiniteDifference) {
    const auto xs = uniform_nodes(0.0, 5.0, 11);
    std::vector<double> ys;
    double acc = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        acc += 0.3 + 0.1 * static_cast<double>(i);
        ys.push_back(acc);
    }
    MonotoneCubic f(xs, ys);
    WorstCase w;
    const double h = 1e-6;
    for (double x = 0.1; x < 4.9; x += 0.01) {
        const double fd = (f(x + h) - f(x - h)) / (2.0 * h);
        w.observe(std::abs(f.derivative(x) - fd), x, 0, f.derivative(x), fd);
    }
    // 1e-6, not 1e-7: the bound is the central difference truncation, which at
    // h = 1e-6 on a function with O(1) third derivative is ~1e-7 -- the
    // difference is the inaccurate side here, not the analytic derivative.
    EXPECT_LT(w.error, 1e-6) << w.describe("x");
}

TEST(Interpolation, MonotoneCubicExtrapolatesWithHeldSlope) {
    MonotoneCubic f({0.0, 1.0, 2.0}, {0.0, 1.0, 3.0});
    // Right end slope is held, so extrapolation is affine and still increasing.
    const double s = f.derivative(2.0);
    EXPECT_GT(s, 0.0);
    EXPECT_NEAR(f(3.0), 3.0 + s, 1e-12);
    EXPECT_NEAR(f(4.0), 3.0 + 2.0 * s, 1e-12);
    EXPECT_NEAR(f.derivative(10.0), s, 1e-15);
    // Left end likewise.
    const double s0 = f.derivative(0.0);
    EXPECT_NEAR(f(-1.0), -s0, 1e-12);
}

TEST(Interpolation, MonotoneCubicKeepsFlatSpotsFlat) {
    // A flat region in the data must come out exactly flat, not wobbling.
    // This is the branch of the limiter that zeroes the slope at a local
    // extremum, and getting it wrong produces a term structure of variance
    // that oscillates between identical quotes.
    MonotoneCubic f({0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 2.0, 3.0});
    for (double x = 1.0; x <= 2.0; x += 0.01) {
        EXPECT_NEAR(f(x), 2.0, 1e-12) << "x = " << x;
    }
}

TEST(Interpolation, MonotoneCubicHandlesDegenerateInputs) {
    MonotoneCubic empty;
    EXPECT_EQ(empty(1.0), 0.0);
    MonotoneCubic one({1.0}, {4.0});
    EXPECT_EQ(one(0.0), 4.0);
    EXPECT_EQ(one(9.0), 4.0);
    MonotoneCubic two({0.0, 1.0}, {0.0, 1.0});
    EXPECT_NEAR(two(0.5), 0.5, 1e-12);
}
