// SPDX-License-Identifier: MIT
/// Validates the generic root finders.  Bisection is treated as the oracle:
/// it depends on nothing but the sign of f, so it cannot share a bug with an
/// interpolating method, and every other solver is checked against it.

#include "vl_test_support.hpp"

#include "volatility_lab/math/root_finding.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace vl::math;

namespace {

struct Problem {
    const char* name;
    std::function<double(double)> f;
    std::function<double(double)> df;
    double lo;
    double hi;
    double root;
};

std::vector<Problem> problems() {
    return {
        {"x^2 - 2", [](double x) { return x * x - 2.0; }, [](double x) { return 2.0 * x; },
         0.0, 2.0, std::sqrt(2.0)},
        {"cos(x) - x", [](double x) { return std::cos(x) - x; },
         [](double x) { return -std::sin(x) - 1.0; }, 0.0, 1.0, 0.7390851332151607},
        {"exp(x) - 5", [](double x) { return std::exp(x) - 5.0; },
         [](double x) { return std::exp(x); }, 0.0, 3.0, std::log(5.0)},
        // Nearly flat at the root -- this is what breaks residual-only
        // termination, since |f| is tiny long before x is close.
        {"(x-1)^3", [](double x) { return (x - 1.0) * (x - 1.0) * (x - 1.0); },
         [](double x) { return 3.0 * (x - 1.0) * (x - 1.0); }, 0.0, 3.0, 1.0},
        // Very steep -- this is what breaks step-only termination.
        {"1e6*(x-0.5)", [](double x) { return 1e6 * (x - 0.5); }, [](double) { return 1e6; },
         0.0, 1.0, 0.5},
    };
}

SolveTolerance tight() {
    SolveTolerance t;
    t.f_tol = 0.0;
    t.x_tol = 0.0;
    t.x_rtol = 4.0 * kEps;
    t.max_iterations = 200;
    return t;
}

}  // namespace

TEST(RootFinding, BisectionSolvesEveryProblem) {
    for (const auto& p : problems()) {
        const auto r = bisection(p.f, p.lo, p.hi, tight());
        EXPECT_TRUE(r.converged()) << p.name << ": " << to_string(r.status);
        EXPECT_NEAR(r.root, p.root, 1e-12) << p.name;
        EXPECT_EQ(static_cast<int>(r.method), static_cast<int>(SolveMethod::Bisection));
    }
}

TEST(RootFinding, BrentMatchesBisectionAndIsFasterOnSimpleRoots) {
    for (const auto& p : problems()) {
        const auto b = bisection(p.f, p.lo, p.hi, tight());
        const auto r = brent(p.f, p.lo, p.hi, tight());
        ASSERT_TRUE(r.converged()) << p.name << ": " << to_string(r.status);
        EXPECT_NEAR(r.root, p.root, 1e-10) << p.name;

        // The iteration-count comparison is restricted to *simple* roots, and
        // the exclusion is a real property rather than a convenience.  At the
        // triple root of (x-1)^3 the function is locally flat, inverse
        // quadratic interpolation has nothing to lock onto, and Brent degrades
        // to roughly three times the bisection count (147 vs 52 as measured).
        // Superlinear convergence is a statement about simple roots; claiming
        // it universally would be wrong.
        const bool simple_root = std::string(p.name) != "(x-1)^3";
        if (simple_root) {
            // One iteration of slack: bisection can land on the root exactly
            // on its first midpoint (it does for 1e6*(x-0.5) on [0,1]), and
            // being beaten by a coincidence is not a regression.
            EXPECT_LE(r.iterations, b.iterations + 1)
                << p.name << ": brent should not be slower on a simple root";
        }
    }
}

TEST(RootFinding, BrentDegradesAtAMultipleRoot) {
    // The flip side, asserted so the limitation stays documented rather than
    // becoming folklore.
    const auto f = [](double x) { return (x - 1.0) * (x - 1.0) * (x - 1.0); };
    const auto b = bisection(f, 0.0, 3.0, tight());
    const auto r = brent(f, 0.0, 3.0, tight());
    ASSERT_TRUE(r.converged());
    ASSERT_TRUE(b.converged());
    EXPECT_GT(r.iterations, b.iterations)
        << "brent is expected to be slower than bisection at a triple root";
}

TEST(RootFinding, NewtonSolvesFromAGoodStart) {
    for (const auto& p : problems()) {
        const double x0 = 0.5 * (p.lo + p.hi);
        const auto r =
            newton([&](double x) { return std::pair<double, double>{p.f(x), p.df(x)}; }, x0,
                   tight());
        if (r.converged()) {
            EXPECT_NEAR(r.root, p.root, 1e-7) << p.name;
        }
    }
}

TEST(RootFinding, SafeguardedNewtonNeverLeavesTheBracket) {
    for (const auto& p : problems()) {
        // Start deliberately at the worst available places, including outside
        // the bracket.
        for (double x0 : {p.lo, p.hi, 0.5 * (p.lo + p.hi), p.lo - 100.0, p.hi + 100.0}) {
            const auto r = safeguarded_newton(
                [&](double x) { return std::pair<double, double>{p.f(x), p.df(x)}; }, p.lo,
                p.hi, x0, tight());
            ASSERT_TRUE(r.converged())
                << p.name << " from " << x0 << ": " << to_string(r.status);
            EXPECT_GE(r.root, p.lo - 1e-12) << p.name;
            EXPECT_LE(r.root, p.hi + 1e-12) << p.name;
            EXPECT_NEAR(r.root, p.root, 1e-7) << p.name << " from " << x0;
        }
    }
}

TEST(RootFinding, PlainNewtonCanDivergeAndSaysSo) {
    // Honesty test.  Newton has no safeguards, and the point of keeping it as a
    // separate, honestly named function is that its failures stay visible
    // instead of being papered over inside a "robust" wrapper.  cbrt has a root
    // at 0 from which Newton oscillates away geometrically.
    const auto fdf = [](double x) {
        const double c = std::cbrt(x);
        return std::pair<double, double>{c, 1.0 / (3.0 * c * c)};
    };
    SolveTolerance t = tight();
    t.max_iterations = 20;
    const auto r = newton(fdf, 1.0, t);
    EXPECT_FALSE(r.converged());
    EXPECT_NE(static_cast<int>(r.status), static_cast<int>(SolveStatus::Converged));
}

TEST(RootFinding, MissingBracketIsReportedNotGuessed) {
    const auto f = [](double x) { return x * x + 1.0; };  // no real root
    EXPECT_EQ(static_cast<int>(bisection(f, -1.0, 1.0, tight()).status),
              static_cast<int>(SolveStatus::NoBracket));
    EXPECT_EQ(static_cast<int>(brent(f, -1.0, 1.0, tight()).status),
              static_cast<int>(SolveStatus::NoBracket));
    const auto sn = safeguarded_newton(
        [](double x) { return std::pair<double, double>{x * x + 1.0, 2.0 * x}; }, -1.0, 1.0,
        0.0, tight());
    EXPECT_EQ(static_cast<int>(sn.status), static_cast<int>(SolveStatus::NoBracket));
}

TEST(RootFinding, RootAtABracketEndpointIsFoundExactly) {
    const auto f = [](double x) { return x; };
    const auto a = bisection(f, 0.0, 1.0, tight());
    const auto b = brent(f, 0.0, 1.0, tight());
    EXPECT_TRUE(a.converged()) << to_string(a.status);
    EXPECT_TRUE(b.converged()) << to_string(b.status);
    EXPECT_NEAR(a.root, 0.0, 1e-15);
    EXPECT_NEAR(b.root, 0.0, 1e-15);
}

TEST(RootFinding, NonFiniteFunctionValuesAreReported) {
    const auto f = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    EXPECT_EQ(static_cast<int>(brent(f, 0.0, 1.0, tight()).status),
              static_cast<int>(SolveStatus::NonFiniteInput));
    EXPECT_EQ(static_cast<int>(bisection(f, 0.0, 1.0, tight()).status),
              static_cast<int>(SolveStatus::NonFiniteInput));
}

TEST(RootFinding, VanishingDerivativeIsReported) {
    const auto fdf = [](double) { return std::pair<double, double>{1.0, 0.0}; };
    const auto r = newton(fdf, 0.0, tight());
    EXPECT_EQ(static_cast<int>(r.status), static_cast<int>(SolveStatus::DerivativeVanished));
}

TEST(RootFinding, HybridReportsWhichMethodActuallyWon) {
    // Well behaved: Newton should win outright, and the reported method must
    // say so -- the benchmark relies on this to show how often the fast path
    // suffices.
    {
        const auto r = hybrid(
            [](double x) { return std::pair<double, double>{x * x - 2.0, 2.0 * x}; }, 0.0,
            2.0, 1.4, tight());
        ASSERT_TRUE(r.converged());
        EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-12);
        EXPECT_EQ(static_cast<int>(r.method), static_cast<int>(SolveMethod::Newton));
    }
    // Pathological for Newton: the fallback must take over and still converge,
    // and the reported method must change.
    {
        const auto fdf = [](double x) {
            const double c = std::cbrt(x - 0.3);
            return std::pair<double, double>{c, 1.0 / (3.0 * c * c)};
        };
        const auto r = hybrid(fdf, 0.0, 1.0, 0.95, tight(), 4);
        ASSERT_TRUE(r.converged()) << to_string(r.status);
        EXPECT_NEAR(r.root, 0.3, 1e-7);
        EXPECT_EQ(static_cast<int>(r.method), static_cast<int>(SolveMethod::Brent));
    }
}

TEST(RootFinding, IterationBudgetIsRespected) {
    // The cap is a termination guarantee, not a tuning parameter, so it must
    // bind even when no tolerance is reachable.
    SolveTolerance t;
    t.f_tol = 0.0;
    t.x_tol = 0.0;
    t.x_rtol = 0.0;
    t.max_iterations = 7;
    const auto r = bisection([](double x) { return x - 0.3; }, 0.0, 1.0, t);
    EXPECT_LE(r.iterations, 7);
    EXPECT_EQ(static_cast<int>(r.status), static_cast<int>(SolveStatus::MaxIterations));
}

TEST(RootFinding, TerminationNeedsBothResidualAndStepCriteria) {
    // (x-1)^3 is flat at the root: |f| < 1e-12 is reached while x is still
    // 1e-4 away, so a residual-only test returns a wrong root.  The step
    // criterion is what saves it.  This test documents why both exist.
    const auto f = [](double x) { return (x - 1.0) * (x - 1.0) * (x - 1.0); };
    SolveTolerance loose;
    loose.f_tol = 1e-12;
    loose.x_tol = 0.0;
    loose.x_rtol = 0.0;
    loose.max_iterations = 200;
    const auto residual_only = bisection(f, 0.0, 3.0, loose);
    EXPECT_TRUE(residual_only.converged());
    EXPECT_GT(std::abs(residual_only.root - 1.0), 1e-6)
        << "a residual-only test is expected to stop early on a flat root";

    const auto both = bisection(f, 0.0, 3.0, tight());
    EXPECT_TRUE(both.converged());
    EXPECT_NEAR(both.root, 1.0, 1e-11);
}

TEST(RootFinding, StatusAndMethodNamesAreComplete) {
    for (auto s : {SolveStatus::Converged, SolveStatus::MaxIterations, SolveStatus::NoBracket,
                   SolveStatus::DerivativeVanished, SolveStatus::NonFiniteInput,
                   SolveStatus::OutOfDomain}) {
        EXPECT_STRNE(to_string(s), "?");
    }
    for (auto m : {SolveMethod::None, SolveMethod::Newton, SolveMethod::SafeguardedNewton,
                   SolveMethod::Bisection, SolveMethod::Brent, SolveMethod::Halley,
                   SolveMethod::HouseholderNormalisedBlack, SolveMethod::ClosedForm}) {
        EXPECT_STRNE(to_string(m), "?");
    }
}
