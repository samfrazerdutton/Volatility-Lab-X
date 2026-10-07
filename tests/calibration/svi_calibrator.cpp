// SPDX-License-Identifier: MIT
/// Validates the calibration layer: the optimiser, the weighting model, and
/// the quasi-explicit SVI calibrator.
///
/// The tests that matter most here are the ones that keep the *claims* in
/// svi_calibrator.hpp honest.  That header asserts specific measured numbers
/// about speed, accuracy and local minima; if those drift, these tests fail
/// and the documentation gets corrected rather than quietly becoming false.

#include "vl_test_support.hpp"

#include "volatility_lab/calibration/optimizer.hpp"
#include "volatility_lab/calibration/svi_calibrator.hpp"
#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/math/linalg.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <chrono>
#include <cmath>
#include <map>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

/// Normalise and weight a regime, returning its fittable slices.
std::vector<std::vector<OptionQuote>> fitted_slices(MarketRegime regime,
                                                    std::uint64_t seed = 20260207u) {
    const auto m = generate_market(regime, seed);
    auto n = normalize(m.snapshot);
    (void)assign_weights_by_slice(n.quotes);
    std::map<double, std::vector<OptionQuote>> groups;
    for (const auto& q : n.quotes) groups[q.years].push_back(q);
    std::vector<std::vector<OptionQuote>> out;
    for (auto& [years, v] : groups) {
        if (v.size() >= 5) out.push_back(std::move(v));
    }
    return out;
}

}  // namespace

// ===========================================================================
// Linear algebra
// ===========================================================================

TEST(Linalg, CholeskySolvesAKnownSystem) {
    // A = [[4,2,1],[2,5,3],[1,3,6]], b chosen so x = (1,2,3).
    math::SmallMatrix a(3);
    const double vals[3][3] = {{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j <= i; ++j) a(i, j) = vals[i][j];
    }
    std::array<double, 3> b{};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            b[i] += vals[i][j] * static_cast<double>(j + 1);
        }
    }
    const auto r = math::cholesky_solve(a, std::span<double>(b.data(), 3));
    ASSERT_TRUE(r.ok()) << math::to_string(r.status);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(b[i], static_cast<double>(i + 1), 1e-12) << "i = " << i;
    }
    EXPECT_GT(r.condition_estimate, 1.0);
}

TEST(Linalg, CholeskyLocalisesTheFailingPivot) {
    // The index is the point: in a Levenberg-Marquardt step, pivot j failing
    // means parameter j is not identified by the residuals, and the calibrator
    // reports which one rather than a generic singularity.
    math::SmallMatrix a(3);
    a(0, 0) = 1.0;
    a(1, 0) = 0.0;
    a(1, 1) = 1.0;
    a(2, 0) = 0.0;
    a(2, 1) = 0.0;
    a(2, 2) = -1.0;  // indefinite at pivot 2
    const auto r = math::cholesky_factor(a);
    EXPECT_EQ(static_cast<int>(r.status),
              static_cast<int>(math::LinalgStatus::NotPositiveDefinite));
    EXPECT_EQ(r.failed_pivot, 2u);
}

TEST(Linalg, NormalEquationsAccumulateCorrectlyAndReproducibly) {
    // A straight line through three points, by hand: y = 2x + 1.
    math::NormalEquations eq(2);
    for (int i = 0; i < 3; ++i) {
        const double x = static_cast<double>(i);
        const double row[2] = {1.0, x};
        eq.add(std::span<const double>(row, 2), 2.0 * x + 1.0, 1.0);
    }
    EXPECT_EQ(eq.count(), 3u);
    EXPECT_EQ(eq.total_weight(), 3.0);
    math::SmallMatrix a = eq.matrix();
    std::array<double, 2> rhs{eq.rhs()[0], eq.rhs()[1]};
    ASSERT_TRUE(math::cholesky_solve(a, std::span<double>(rhs.data(), 2)).ok());
    EXPECT_NEAR(rhs[0], 1.0, 1e-12);
    EXPECT_NEAR(rhs[1], 2.0, 1e-12);

    // Bitwise reproducible: the accumulation order is fixed, which the
    // determinism contract requires of a calibration.
    math::NormalEquations again(2);
    for (int i = 0; i < 3; ++i) {
        const double x = static_cast<double>(i);
        const double row[2] = {1.0, x};
        again.add(std::span<const double>(row, 2), 2.0 * x + 1.0, 1.0);
    }
    EXPECT_EQ(again.chi_squared(), eq.chi_squared());
    EXPECT_EQ(again.rhs()[0], eq.rhs()[0]);
}

TEST(Linalg, NormalEquationsIgnoreZeroWeightAndNonFiniteResiduals) {
    math::NormalEquations eq(2);
    const double row[2] = {1.0, 1.0};
    eq.add(std::span<const double>(row, 2), 1.0, 0.0);  // zero weight
    eq.add(std::span<const double>(row, 2), std::numeric_limits<double>::quiet_NaN(), 1.0);
    EXPECT_EQ(eq.count(), 0u);
    EXPECT_EQ(eq.chi_squared(), 0.0);
}

TEST(Linalg, BoxedLeastSquaresFindsTheExactConstrainedOptimum) {
    // Unconstrained optimum at x = (1, 2, 3); with the box capping the second
    // variable at 1, the constrained optimum must pin it there and re-solve
    // the other two -- which is what active-set enumeration gives exactly.
    math::NormalEquations eq(3);
    for (int i = 0; i < 12; ++i) {
        const double t = -1.0 + 0.2 * i;
        const double row[3] = {1.0, t, t * t};
        eq.add(std::span<const double>(row, 3), 1.0 + 2.0 * t + 3.0 * t * t, 1.0);
    }
    const std::array<double, 3> lo{-1e9, -1e9, -1e9};
    const std::array<double, 3> hi{1e9, 1e9, 1e9};
    const auto free = math::solve_boxed_least_squares(eq, lo, hi);
    ASSERT_TRUE(free.feasible);
    EXPECT_NEAR(free.solution[0], 1.0, 1e-9);
    EXPECT_NEAR(free.solution[1], 2.0, 1e-9);
    EXPECT_NEAR(free.solution[2], 3.0, 1e-9);
    EXPECT_EQ(free.active_constraints, 0u);

    const std::array<double, 3> hi2{1e9, 1.0, 1e9};
    const auto capped = math::solve_boxed_least_squares(eq, lo, hi2);
    ASSERT_TRUE(capped.feasible);
    EXPECT_NEAR(capped.solution[1], 1.0, 1e-12) << "the active constraint must bind";
    EXPECT_EQ(capped.active_constraints, 1u);
    EXPECT_GT(capped.objective, free.objective) << "constraining cannot help";
    // And it must be the *best* feasible point, not merely a feasible one:
    // perturbing the free variables must not improve the objective.
    for (double d : {-1e-3, 1e-3}) {
        math::SmallVector probe = capped.solution;
        probe[0] += d;
        double obj = eq.chi_squared();
        for (std::size_t i = 0; i < 3; ++i) {
            obj -= 2.0 * probe[i] * eq.rhs()[i];
            for (std::size_t j = 0; j < 3; ++j) {
                const double aij = (i >= j) ? eq.matrix()(i, j) : eq.matrix()(j, i);
                obj += probe[i] * aij * probe[j];
            }
        }
        EXPECT_GE(obj, capped.objective - 1e-12);
    }
}

// ===========================================================================
// The optimiser
// ===========================================================================

TEST(Optimizer, FitsALinearModelExactly) {
    // y = a + b*x through exact data: the optimiser must land on the answer,
    // not near it.
    const std::vector<double> xs{0.0, 1.0, 2.0, 3.0, 4.0};
    LeastSquaresProblem p;
    p.num_params = 2;
    p.num_residuals = xs.size();
    p.typical_scale = {1.0, 1.0};
    p.residuals = [&](std::span<const double> q, std::span<double> r) -> std::size_t {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            r[i] = q[0] + q[1] * xs[i] - (3.0 - 0.5 * xs[i]);
        }
        return xs.size();
    };
    const double start[2] = {0.0, 0.0};
    const auto res = calibrate(p, start);
    ASSERT_TRUE(res.ok()) << to_string(res.status);
    EXPECT_NEAR(res.parameters[0], 3.0, 1e-9);
    EXPECT_NEAR(res.parameters[1], -0.5, 1e-9);
    EXPECT_LT(res.objective, 1e-18);
    EXPECT_LT(res.iterations, 20);
}

TEST(Optimizer, FitsANonlinearModelFromAPoorStart) {
    // y = A exp(-k x): badly scaled parameters (A ~ 100, k ~ 0.1), which is
    // exactly the case Marquardt's diagonal scaling exists for.
    const std::vector<double> xs{0.0, 1.0, 2.0, 4.0, 8.0, 16.0};
    LeastSquaresProblem p;
    p.num_params = 2;
    p.num_residuals = xs.size();
    p.typical_scale = {100.0, 0.1};
    p.lower = {0.0, 0.0};
    p.residuals = [&](std::span<const double> q, std::span<double> r) -> std::size_t {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            r[i] = q[0] * std::exp(-q[1] * xs[i]) - 120.0 * std::exp(-0.17 * xs[i]);
        }
        return xs.size();
    };
    const double start[2] = {10.0, 1.0};
    const auto res = calibrate(p, start);
    ASSERT_TRUE(res.ok()) << to_string(res.status);
    EXPECT_NEAR(res.parameters[0], 120.0, 1e-4);
    EXPECT_NEAR(res.parameters[1], 0.17, 1e-7);
}

TEST(Optimizer, RespectsBoundsByProjection) {
    // The true optimum lies outside the box, so the fit must pin to the bound
    // and say so -- a penalty would instead evaluate the model at an
    // inadmissible point in order to decide how bad it is.
    LeastSquaresProblem p;
    p.num_params = 1;
    p.num_residuals = 1;
    p.lower = {2.0};
    p.upper = {5.0};
    p.typical_scale = {1.0};
    p.residuals = [](std::span<const double> q, std::span<double> r) -> std::size_t {
        r[0] = q[0] - 10.0;  // wants q = 10, bound is 5
        return 1;
    };
    const double start[1] = {3.0};
    const auto res = calibrate(p, start);
    EXPECT_NEAR(res.parameters[0], 5.0, 1e-12);
    ASSERT_EQ(res.parameters_at_bound.size(), 1u);
    EXPECT_EQ(res.parameters_at_bound[0], 1);
    EXPECT_EQ(res.num_at_bound(), 1u);
}

TEST(Optimizer, ReportsTheConditionNumberSoAnUnidentifiedFitIsVisible) {
    // Two parameters that only ever appear as their sum: the objective is flat
    // along a direction, so the fit converges but the individual values are
    // arbitrary.  The objective value alone cannot distinguish that from a
    // good fit; the condition estimate can.
    LeastSquaresProblem p;
    p.num_params = 2;
    p.num_residuals = 4;
    p.typical_scale = {1.0, 1.0};
    p.residuals = [](std::span<const double> q, std::span<double> r) -> std::size_t {
        for (std::size_t i = 0; i < 4; ++i) r[i] = (q[0] + q[1]) - 7.0;
        return 4;
    };
    const double start[2] = {1.0, 1.0};
    const auto res = calibrate(p, start);
    EXPECT_NEAR(res.parameters[0] + res.parameters[1], 7.0, 1e-6);
    // The estimate is a *lower bound* -- (max pivot / min pivot)^2 -- so the
    // threshold is loose on purpose.  The claim being tested is "visibly
    // ill-conditioned", not a particular number.
    EXPECT_GT(res.condition_estimate, 1e4)
        << "an unidentified fit must be reported as ill-conditioned";
}

TEST(Optimizer, MaxIterationsIsNotReportedAsSuccess) {
    // The parameters it returns are often usable, and the temptation is to
    // call it converged -- but then a caller cannot tell "the fit is good"
    // from "the fit ran out of budget", and the quality report loses the
    // ability to flag the second.
    LeastSquaresProblem p;
    p.num_params = 1;
    p.num_residuals = 1;
    p.typical_scale = {1.0};
    p.residuals = [](std::span<const double> q, std::span<double> r) -> std::size_t {
        r[0] = std::sin(50.0 * q[0]) + 0.5 * q[0];  // many local minima
        return 1;
    };
    OptimizerSettings s;
    s.max_iterations = 3;
    s.objective_rtol = 0.0;
    s.param_rtol = 0.0;
    s.gradient_tol = 0.0;
    const double start[1] = {5.0};
    const auto res = calibrate(p, start, s);
    EXPECT_LE(res.iterations, 3);
    EXPECT_FALSE(is_success(res.status));
    EXPECT_FALSE(res.ok());
}

TEST(Optimizer, RejectsMalformedProblems) {
    LeastSquaresProblem p;
    const double start[1] = {0.0};
    EXPECT_EQ(static_cast<int>(calibrate(p, start).status),
              static_cast<int>(OptimizerStatus::InvalidProblem));
    p.num_params = 1;
    p.num_residuals = 1;
    EXPECT_EQ(static_cast<int>(calibrate(p, start).status),
              static_cast<int>(OptimizerStatus::InvalidProblem))
        << "a problem with no residual function must be rejected";
}

TEST(Optimizer, ReportsNonFiniteResidualsAtTheStart) {
    LeastSquaresProblem p;
    p.num_params = 1;
    p.num_residuals = 1;
    p.typical_scale = {1.0};
    p.residuals = [](std::span<const double>, std::span<double> r) -> std::size_t {
        r[0] = std::numeric_limits<double>::quiet_NaN();
        return 1;
    };
    const double start[1] = {1.0};
    EXPECT_EQ(static_cast<int>(calibrate(p, start).status),
              static_cast<int>(OptimizerStatus::ResidualNonFinite));
}

TEST(Optimizer, MultiStartIsDeterministicAndTiesGoToTheEarlierStart) {
    LeastSquaresProblem p;
    p.num_params = 1;
    p.num_residuals = 1;
    p.typical_scale = {1.0};
    p.residuals = [](std::span<const double> q, std::span<double> r) -> std::size_t {
        r[0] = q[0] * q[0] - 4.0;  // roots at +-2
        return 1;
    };
    const std::vector<std::vector<double>> starts{{2.5}, {-2.5}, {0.5}};
    const auto a = calibrate_multi_start(p, starts);
    const auto b = calibrate_multi_start(p, starts);
    EXPECT_EQ(a.best_start, b.best_start);
    EXPECT_EQ(a.best.parameters[0], b.best.parameters[0]);
    EXPECT_EQ(a.starts_tried, 3u);
    EXPECT_EQ(a.objectives.size(), 3u);
    // Ties go to the earlier start, which is what makes the result a function
    // of the inputs alone.
    EXPECT_EQ(a.best_start, 0u);
}

TEST(Optimizer, JacobianCheckCatchesAWrongDerivative) {
    LeastSquaresProblem p;
    p.num_params = 2;
    p.num_residuals = 3;
    p.typical_scale = {1.0, 1.0};
    p.residuals = [](std::span<const double> q, std::span<double> r) -> std::size_t {
        for (std::size_t i = 0; i < 3; ++i) {
            const double x = static_cast<double>(i);
            r[i] = q[0] * std::exp(q[1] * x);
        }
        return 3;
    };
    // Correct Jacobian first.
    p.jacobian = [](std::span<const double> q, std::span<double> j) {
        for (std::size_t i = 0; i < 3; ++i) {
            const double x = static_cast<double>(i);
            j[i * 2 + 0] = std::exp(q[1] * x);
            j[i * 2 + 1] = q[0] * x * std::exp(q[1] * x);
        }
    };
    const double at[2] = {1.5, 0.3};
    const auto good = check_jacobian(p, at);
    EXPECT_TRUE(good.agrees) << "max rel error " << good.max_rel_error;
    EXPECT_LT(good.max_rel_error, 1e-7);

    // Now break it in a way that is easy to write by hand: forget the x.
    p.jacobian = [](std::span<const double> q, std::span<double> j) {
        for (std::size_t i = 0; i < 3; ++i) {
            const double x = static_cast<double>(i);
            j[i * 2 + 0] = std::exp(q[1] * x);
            j[i * 2 + 1] = q[0] * std::exp(q[1] * x);  // missing factor of x
        }
    };
    const auto bad = check_jacobian(p, at);
    EXPECT_FALSE(bad.agrees);
    EXPECT_GT(bad.max_rel_error, 0.1);
    EXPECT_EQ(bad.worst_param, 1u);
}

// ===========================================================================
// Weights
// ===========================================================================

TEST(Weights, NormaliseToAUnitMean) {
    // Keeps the objective and rms_residual comparable between slices with
    // different quote counts, so the LM tolerances need no per-slice tuning.
    auto slices = fitted_slices(MarketRegime::Normal);
    ASSERT_GT(slices.size(), 3u);
    for (auto& s : slices) {
        const auto wb = assign_weights(s, s[s.size() / 2].total_variance);
        ASSERT_EQ(wb.size(), s.size());
        double sum = 0.0;
        std::size_t used = 0;
        for (const auto& q : s) {
            if (q.weight > 0.0) {
                sum += q.weight;
                ++used;
            }
        }
        ASSERT_GT(used, 0u);
        EXPECT_NEAR(sum / static_cast<double>(used), 1.0, 1e-9);
    }
}

TEST(Weights, FavourNearTheMoneyOverTheWings) {
    // The whole point: unweighted least squares lets the wings dominate
    // because there are more of them and they are noisier.
    auto slices = fitted_slices(MarketRegime::Normal);
    auto& s = slices[slices.size() / 2];
    const auto wb = assign_weights(s, s[s.size() / 2].total_variance);

    // Find the quote closest to the money and the furthest from it.
    std::size_t atm = 0, wing = 0;
    double best = 1e9, worst = -1.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double ak = std::abs(s[i].log_moneyness);
        if (ak < best) { best = ak; atm = i; }
        if (ak > worst) { worst = ak; wing = i; }
    }
    EXPECT_GT(s[atm].weight, s[wing].weight)
        << "near-the-money weight " << s[atm].weight << " vs wing " << s[wing].weight;
    // And the wing's half-spread in vol points must be the larger, which is
    // the mechanism rather than the moneyness kernel alone.
    EXPECT_GT(wb[wing].half_spread_vol, wb[atm].half_spread_vol);
}

TEST(Weights, SpreadDominatesAndIsConvertedToVolPoints) {
    // The conversion is the important part: the same price spread is a very
    // different amount of volatility uncertainty on a 2.00 option and a 0.10
    // option, and only the vega-divided version compares them correctly.
    std::vector<OptionQuote> two(2);
    for (auto& q : two) {
        q.forward = 100.0;
        q.years = 1.0;
        q.discount = 1.0;
        q.status = QuoteStatus::Ok;
        q.volume = 1000.0;
        q.open_interest = 5000.0;
        q.age_seconds = 1.0;
        q.implied_vol = 0.2;
        q.total_variance = 0.04;
    }
    // Near the money: large vega, so a 0.05 spread is a few bp of vol.
    two[0].strike = 100.0;
    two[0].log_moneyness = 0.0;
    two[0].vega = 39.8;
    two[0].bid = 7.94;
    two[0].ask = 7.99;
    // Deep wing: small vega, so the same 0.05 spread is many vol points.
    two[1].strike = 160.0;
    two[1].log_moneyness = std::log(1.6);
    two[1].vega = 1.5;
    two[1].bid = 0.10;
    two[1].ask = 0.15;

    WeightConfig cfg;
    cfg.use_moneyness = false;  // isolate the spread effect
    cfg.use_liquidity = false;
    const auto wb = assign_weights(two, 0.0, cfg);
    EXPECT_LT(wb[0].half_spread_vol, wb[1].half_spread_vol);
    EXPECT_GT(two[0].weight, two[1].weight)
        << "the tighter quote (in vol terms) must get more weight";
}

TEST(Weights, CapRelativeToTheMedianStopsOneQuoteDominating) {
    // The characteristic failure of inverse-variance weighting: the estimator
    // trusts whichever observation *claims* the smallest error, so a locked or
    // stale print can outweigh the rest of the slice combined.
    std::vector<OptionQuote> qs(10);
    for (std::size_t i = 0; i < qs.size(); ++i) {
        auto& q = qs[i];
        q.forward = 100.0;
        q.strike = 90.0 + 2.0 * static_cast<double>(i);
        q.years = 1.0;
        q.discount = 1.0;
        q.status = QuoteStatus::Ok;
        q.volume = 1000.0;
        q.open_interest = 5000.0;
        q.age_seconds = 1.0;
        q.vega = 30.0;
        q.log_moneyness = std::log(q.strike / q.forward);
        q.implied_vol = 0.2;
        q.total_variance = 0.04;
        q.bid = 5.0;
        q.ask = 5.3;  // an ordinary spread
    }
    // One suspiciously tight quote.
    qs[3].ask = qs[3].bid + 1e-9;

    WeightConfig cfg;
    cfg.use_moneyness = false;
    cfg.max_weight_ratio = 50.0;
    const auto wb = assign_weights(qs, 0.0, cfg);
    double total = 0.0;
    for (const auto& q : qs) total += q.weight;
    EXPECT_LT(qs[3].weight / total, 0.5)
        << "one quote carries more than half the slice's weight";
    EXPECT_GT(wb[3].combined, 0.0);
}

TEST(Weights, RejectedQuotesGetZeroWeight) {
    std::vector<OptionQuote> qs(3);
    for (auto& q : qs) {
        q.forward = 100.0;
        q.strike = 100.0;
        q.years = 1.0;
        q.discount = 1.0;
        q.vega = 30.0;
        q.bid = 5.0;
        q.ask = 5.2;
        q.status = QuoteStatus::Ok;
    }
    qs[1].status = QuoteStatus::Rejected;
    (void)assign_weights(qs, 0.04);
    EXPECT_GT(qs[0].weight, 0.0);
    EXPECT_EQ(qs[1].weight, 0.0);
    EXPECT_GT(qs[2].weight, 0.0);
}

TEST(Weights, DegradedQuotesAreDownWeightedNotExcluded) {
    std::vector<OptionQuote> qs(2);
    for (auto& q : qs) {
        q.forward = 100.0;
        q.strike = 100.0;
        q.years = 1.0;
        q.discount = 1.0;
        q.vega = 30.0;
        q.bid = 5.0;
        q.ask = 5.2;
        q.status = QuoteStatus::Ok;
        q.volume = 1000.0;
        q.open_interest = 5000.0;
    }
    qs[1].status = QuoteStatus::Degraded;
    (void)assign_weights(qs, 0.04);
    EXPECT_GT(qs[1].weight, 0.0) << "a degraded quote still constrains the fit";
    EXPECT_LT(qs[1].weight, qs[0].weight);
}

TEST(Weights, PerSliceNormalisationStopsABigSliceOutvotingASmallOne) {
    // A global normalisation would let a 60-quote slice dominate a 12-quote
    // one simply by having more quotes, when what is wanted is for each
    // expiry's shape to be fitted on its own terms.
    auto slices = fitted_slices(MarketRegime::Normal);
    std::vector<OptionQuote> all;
    for (auto& s : slices) {
        for (auto& q : s) all.push_back(q);
    }
    (void)assign_weights_by_slice(all);

    std::map<double, std::pair<double, std::size_t>> per_expiry;
    for (const auto& q : all) {
        if (q.weight <= 0.0) continue;
        auto& e = per_expiry[q.years];
        e.first += q.weight;
        ++e.second;
    }
    ASSERT_GT(per_expiry.size(), 3u);
    for (const auto& [years, e] : per_expiry) {
        EXPECT_NEAR(e.first / static_cast<double>(e.second), 1.0, 1e-9)
            << "slice at T = " << years << " is not unit-mean weighted";
    }
}

TEST(Weights, ResidualKindsAgreeToFirstOrder) {
    // Volatility and total-variance residuals must be consistent, since the
    // quasi-explicit inner solve relies on converting between them via
    // dsigma/dw.
    OptionQuote q;
    q.forward = 100.0;
    q.strike = 105.0;
    q.years = 0.5;
    q.discount = 1.0;
    q.implied_vol = 0.2;
    q.total_variance = 0.02;
    q.log_moneyness = std::log(1.05);
    q.vega = 20.0;
    q.mid = 3.0;

    const double w_model = q.total_variance * 1.001;  // a small perturbation
    const double r_w = quote_residual(q, w_model, ResidualKind::TotalVariance);
    const double r_v = quote_residual(q, w_model, ResidualKind::Volatility);
    const double dsigma_dw = 1.0 / (2.0 * std::sqrt(q.total_variance * q.years));
    EXPECT_LT(rel_error(r_v, r_w * dsigma_dw), 1e-3)
        << "vol and variance residuals disagree beyond first order";

    // And the derivatives must match finite differences.
    const double h = q.total_variance * 1e-6;
    for (auto kind : {ResidualKind::TotalVariance, ResidualKind::Volatility,
                      ResidualKind::Price, ResidualKind::VegaScaledPrice}) {
        const double fd = (quote_residual(q, w_model + h, kind) -
                           quote_residual(q, w_model - h, kind)) /
                          (2.0 * h);
        const double an = quote_residual_dw(q, w_model, kind);
        EXPECT_LT(rel_error(an, fd), 1e-5) << to_string(kind);
    }
}

// ===========================================================================
// The quasi-explicit calibrator
// ===========================================================================

TEST(SviCalibrator, InnerSolveIsExactOnPerfectData) {
    // Generate quotes from a known SVI slice with no noise, then check that
    // the inner solve at the true (m, sigma) recovers the other three
    // parameters exactly.  It is a *linear* problem there, so "exactly" is the
    // right standard, not "closely".
    SviParams truth;
    truth.a = 0.012;
    truth.b = 0.085;
    truth.rho = -0.42;
    truth.m = 0.018;
    truth.sigma = 0.14;
    truth.years = 0.5;

    std::vector<OptionQuote> quotes;
    for (int i = -10; i <= 10; ++i) {
        OptionQuote q;
        q.log_moneyness = 0.03 * static_cast<double>(i);
        q.years = truth.years;
        q.forward = 100.0;
        q.strike = 100.0 * std::exp(q.log_moneyness);
        q.discount = 1.0;
        q.total_variance = svi_total_variance(truth, q.log_moneyness);
        q.implied_vol = std::sqrt(q.total_variance / q.years);
        q.weight = 1.0;
        q.status = QuoteStatus::Ok;
        quotes.push_back(q);
    }

    const auto inner = svi_inner_solve(quotes, truth.m, truth.sigma);
    ASSERT_TRUE(inner.feasible);
    // The objective is evaluated as chi2 - 2 x^T b + x^T A x, which cancels
    // when the fit is exact: chi2 here is about 0.03, so a result of 1e-18 is
    // a relative 1e-16 -- the arithmetic floor, not a fit error.  Asserting
    // 1e-24 would be asserting something the formula cannot deliver.
    EXPECT_LT(inner.objective, 1e-16) << "the inner problem is linear and the data exact";
    const SviReduced expected = svi_to_reduced(truth);
    EXPECT_NEAR(inner.reduced.adash, expected.adash, 1e-12);
    EXPECT_NEAR(inner.reduced.d, expected.d, 1e-12);
    EXPECT_NEAR(inner.reduced.c, expected.c, 1e-12);
}

TEST(SviCalibrator, RecoversAKnownSliceFromNoiselessQuotes) {
    SviParams truth;
    truth.a = 0.010;
    truth.b = 0.090;
    truth.rho = -0.40;
    truth.m = 0.020;
    truth.sigma = 0.15;
    truth.years = 1.0;

    std::vector<OptionQuote> quotes;
    for (int i = -12; i <= 12; ++i) {
        OptionQuote q;
        q.log_moneyness = 0.035 * static_cast<double>(i);
        q.years = 1.0;
        q.forward = 100.0;
        q.strike = 100.0 * std::exp(q.log_moneyness);
        q.discount = 1.0;
        q.total_variance = svi_total_variance(truth, q.log_moneyness);
        q.implied_vol = std::sqrt(q.total_variance);
        q.weight = 1.0;
        q.status = QuoteStatus::Ok;
        quotes.push_back(q);
    }

    const auto fit = calibrate_svi_slice(quotes);
    ASSERT_EQ(static_cast<int>(fit.status), static_cast<int>(SviFitStatus::Ok))
        << to_string(fit.status) << "\n" << fit.diagnostics.summary();

    // The *surface*, not the parameters.  SVI parameters are not separately
    // identified -- `a` trades off against `b*sigma` -- so a fit can reproduce
    // the smile to machine precision with visibly different parameters, and
    // asserting on parameters would be testing an artefact of the
    // parameterisation rather than the quality of the fit.
    WorstCase w;
    for (double k = -0.45; k <= 0.45; k += 0.005) {
        const double got = svi_total_variance(fit.params, k);
        const double want = svi_total_variance(truth, k);
        w.observe(rel_error(got, want), k, 0, got, want);
    }
    EXPECT_LT(w.error, 1e-6) << w.describe("k");
    EXPECT_LT(fit.rms_vol_error, 1e-6);
}

TEST(SviCalibrator, FitsEveryRegimeToTheQuoteNoiseLevel) {
    // The fit cannot be better than the noise in the data, and it should not
    // be much worse.  The bound is a multiple of the injected quote noise, so
    // the test scales with the regime rather than using a single magic number.
    struct Expectation {
        MarketRegime regime;
        double noise_multiple;
    };
    const Expectation cases[] = {
        {MarketRegime::Normal, 3.0},   {MarketRegime::HighVol, 3.0},
        {MarketRegime::Crash, 3.0},    {MarketRegime::VolCrush, 6.0},
        {MarketRegime::Earnings, 4.0},
    };
    for (const auto& c : cases) {
        const auto cfg = regime_defaults(c.regime);
        auto slices = fitted_slices(c.regime);
        ASSERT_GE(slices.size(), 3u) << to_string(c.regime);
        double worst_rms = 0.0;
        for (auto& s : slices) {
            const auto fit = calibrate_svi_slice(s);
            ASSERT_TRUE(fit.ok()) << to_string(c.regime) << ": " << to_string(fit.status)
                                  << "\n" << fit.diagnostics.summary();
            worst_rms = std::max(worst_rms, fit.rms_vol_error);
        }
        EXPECT_LT(worst_rms, c.noise_multiple * cfg.vol_noise + 0.004)
            << to_string(c.regime) << ": worst RMS vol error " << worst_rms
            << " against injected noise " << cfg.vol_noise;
    }
}

TEST(SviCalibrator, AlwaysProducesAnAdmissibleSlice) {
    // Whatever the data, the output must be a usable model: positive variance
    // everywhere, |rho| < 1, sigma > 0.  A calibrator that can emit an
    // inadmissible slice has handed a landmine to every downstream consumer.
    for (auto regime : {MarketRegime::Normal, MarketRegime::HighVol, MarketRegime::Crash,
                        MarketRegime::VolCrush, MarketRegime::Earnings,
                        MarketRegime::Illiquid}) {
        for (std::uint64_t seed : {1u, 7u, 99u}) {
            for (auto& s : fitted_slices(regime, seed)) {
                const auto fit = calibrate_svi_slice(s);
                ASSERT_TRUE(svi_parameters_admissible(fit.params))
                    << to_string(regime) << " seed " << seed << ": " << to_string(fit.status);
                for (double k = -3.0; k <= 3.0; k += 0.05) {
                    ASSERT_GE(svi_total_variance(fit.params, k), 0.0)
                        << to_string(regime) << " at k = " << k;
                }
            }
        }
    }
}

TEST(SviCalibrator, RespectsLeesWingBound) {
    // The bound is enforced inside the inner solve, so it holds by
    // construction rather than being checked afterwards.  Verified on the
    // crash regime, where it actually binds.
    for (auto& s : fitted_slices(MarketRegime::Crash)) {
        const auto fit = calibrate_svi_slice(s);
        const SviWings wings = svi_wings(fit.params);
        EXPECT_LE(std::abs(wings.left), kLeeSlopeBound + 1e-9);
        EXPECT_LE(wings.right, kLeeSlopeBound + 1e-9);
    }
}

TEST(SviCalibrator, WingBoundDoesNotBindOnRealisticData) {
    // Worth recording as a negative result.  Lee's bound caps the asymptotic
    // total-variance slope at 2, and *none* of the six synthetic regimes gets
    // anywhere near it: the fitted right wing peaks around 0.4 on the crash
    // regime.  So on plausible data the constraint is inert, and enforcing it
    // costs nothing -- which is a different claim from "it improves the fit",
    // and the honest one.
    SviCalibratorConfig off;
    off.enforce_wing_bound = false;
    for (auto regime : {MarketRegime::Normal, MarketRegime::HighVol,
                        MarketRegime::Crash, MarketRegime::VolCrush}) {
        for (auto& s : fitted_slices(regime)) {
            const auto on = calibrate_svi_slice(s);
            const auto no = calibrate_svi_slice(s, off);
            EXPECT_LT(svi_wings(on.params).right, kLeeSlopeBound)
                << to_string(regime);
            // Inert: the two fits agree because the bound never activates.
            EXPECT_NEAR(on.objective, no.objective,
                        1e-12 * std::max(1.0, on.objective))
                << to_string(regime) << ": the bound bound unexpectedly";
        }
    }
}

TEST(SviCalibrator, WingBoundBindsOnDataThatDemandsASteepWing) {
    // ...but it must still work when it is needed.  Data with a
    // total-variance slope of about 5 -- far beyond Lee's limit of 2, and
    // therefore not producible by any arbitrage-free underlying -- must be
    // fitted with a capped wing rather than reproduced.
    std::vector<OptionQuote> steep;
    for (int i = -10; i <= 10; ++i) {
        OptionQuote q;
        q.log_moneyness = 0.04 * static_cast<double>(i);
        q.years = 1.0;
        q.forward = 100.0;
        q.strike = 100.0 * std::exp(q.log_moneyness);
        q.discount = 1.0;
        // w = 0.04 + 5*|k|: a V shape with slope 5.
        q.total_variance = 0.04 + 5.0 * std::abs(q.log_moneyness);
        q.implied_vol = std::sqrt(q.total_variance);
        q.weight = 1.0;
        q.status = QuoteStatus::Ok;
        steep.push_back(q);
    }

    SviCalibratorConfig off;
    off.enforce_wing_bound = false;
    const auto bounded = calibrate_svi_slice(steep);
    const auto unbounded = calibrate_svi_slice(steep, off);

    EXPECT_LE(svi_wings(bounded.params).right, kLeeSlopeBound + 1e-9)
        << "the bound failed to cap the wing";
    EXPECT_GT(svi_wings(unbounded.params).right, kLeeSlopeBound)
        << "without the bound the fit should follow the inadmissible data";
    // And the bounded fit necessarily has the larger residual -- that is what
    // a binding constraint means.
    EXPECT_GT(bounded.objective, unbounded.objective);
}

TEST(SviCalibrator, DegradesToAFlatSliceRatherThanOverfitting) {
    // Four quotes cannot identify five parameters.  Fitting them anyway
    // produces something that looks excellent and extrapolates to nonsense; a
    // flat slice is a poor model but an honest one.
    std::vector<OptionQuote> few;
    for (int i = 0; i < 3; ++i) {
        OptionQuote q;
        q.log_moneyness = 0.05 * static_cast<double>(i - 1);
        q.years = 1.0;
        q.forward = 100.0;
        q.strike = 100.0 * std::exp(q.log_moneyness);
        q.discount = 1.0;
        q.total_variance = 0.04;
        q.implied_vol = 0.2;
        q.weight = 1.0;
        q.status = QuoteStatus::Ok;
        few.push_back(q);
    }
    const auto fit = calibrate_svi_slice(few);
    EXPECT_EQ(static_cast<int>(fit.status),
              static_cast<int>(SviFitStatus::DegradedToFlat));
    EXPECT_EQ(fit.params.b, 0.0) << "a flat slice has no wings";
    EXPECT_NEAR(fit.params.a, 0.04, 1e-9);
    EXPECT_TRUE(svi_parameters_admissible(fit.params));
    EXPECT_GT(fit.diagnostics.count(DiagCode::SliceTooFewQuotes), 0u);
}

TEST(SviCalibrator, EmptySliceIsReportedNotCrashed) {
    const auto fit = calibrate_svi_slice({});
    EXPECT_EQ(static_cast<int>(fit.status), static_cast<int>(SviFitStatus::TooFewQuotes));
    EXPECT_FALSE(fit.ok());
    EXPECT_GT(fit.diagnostics.size(), 0u);

    // All-zero-weight is the same situation reached differently.
    std::vector<OptionQuote> zeroed(10);
    for (auto& q : zeroed) {
        q.weight = 0.0;
        q.years = 1.0;
    }
    EXPECT_EQ(static_cast<int>(calibrate_svi_slice(zeroed).status),
              static_cast<int>(SviFitStatus::TooFewQuotes));
}

TEST(SviCalibrator, ReportsWhetherTheOuterParametersAreIdentified) {
    // `grid_objective_range` is the spread of the objective across the (m,
    // sigma) grid.  A small spread means the slice barely cares about those
    // two and the fitted values are not identified -- which the objective
    // value alone cannot reveal.
    auto slices = fitted_slices(MarketRegime::Normal);
    const auto fit = calibrate_svi_slice(slices[slices.size() / 2]);
    EXPECT_GT(fit.grid_objective_range, 0.0);
    EXPECT_GT(fit.grid_objective_range / std::max(fit.objective, 1e-300), 10.0)
        << "a real smile should be sensitive to (m, sigma)";

    // A flat slice, by contrast, is insensitive to them.
    std::vector<OptionQuote> flat;
    for (int i = -8; i <= 8; ++i) {
        OptionQuote q;
        q.log_moneyness = 0.03 * static_cast<double>(i);
        q.years = 1.0;
        q.forward = 100.0;
        q.strike = 100.0 * std::exp(q.log_moneyness);
        q.discount = 1.0;
        q.total_variance = 0.04;
        q.implied_vol = 0.2;
        q.weight = 1.0;
        q.status = QuoteStatus::Ok;
        flat.push_back(q);
    }
    const auto flat_fit = calibrate_svi_slice(flat);
    EXPECT_LT(flat_fit.grid_objective_range, 1e-10)
        << "a flat slice is insensitive to (m, sigma) and must be reported as such";
}

TEST(SviCalibrator, RefinementImprovesOnTheGridButNotByMuch) {
    // Both halves matter.  If the refinement never helped it should be
    // removed; if the grid were useless the refinement would be doing all the
    // work and the global-search claim would be empty.
    double worst_gain = 1.0;
    double best_gain = 1.0;
    for (auto& s : fitted_slices(MarketRegime::Normal)) {
        const auto fit = calibrate_svi_slice(s);
        ASSERT_GT(fit.grid_objective, 0.0);
        ASSERT_LE(fit.objective, fit.grid_objective * (1.0 + 1e-9))
            << "the refinement made the fit worse";
        const double gain = fit.grid_objective / std::max(fit.objective, 1e-300);
        worst_gain = std::min(worst_gain, gain);
        best_gain = std::max(best_gain, gain);
    }
    EXPECT_GE(best_gain, 1.01) << "the refinement never improved anything";
    EXPECT_LT(best_gain, 100.0)
        << "the grid is not locating the basin; the refinement is doing all the work";
}

TEST(SviCalibrator, ConvergesInFarFewerInnerSolvesThanTheBudget) {
    // Regression test for a real bug: the refinement originally posed a
    // 1-residual 2-parameter problem, so J^T J had rank one, LM never
    // converged, and every slice burned its full 100-iteration budget --
    // around 530 inner solves instead of 200.
    for (auto& s : fitted_slices(MarketRegime::Normal)) {
        const auto fit = calibrate_svi_slice(s);
        // The budget covers the 121-point grid, the refinement, and the up-to-
        // three extra passes the exact variance-positivity constraint costs.
        // The figure to watch is the order of magnitude: the bug this guards
        // against produced 530+.
        EXPECT_LT(fit.inner_solves, 450)
            << "inner solves regressed to " << fit.inner_solves;
        EXPECT_LT(fit.outer_iterations, 60)
            << "the refinement is not converging: " << fit.outer_iterations
            << " iterations";
    }
}

TEST(SviCalibrator, IsDeterministic) {
    // Part of the library-wide determinism contract: the same quotes must
    // produce bit-identical parameters.
    for (auto regime : {MarketRegime::Normal, MarketRegime::Crash}) {
        for (auto& s : fitted_slices(regime)) {
            const auto a = calibrate_svi_slice(s);
            const auto b = calibrate_svi_slice(s);
            EXPECT_EQ(a.params.a, b.params.a);
            EXPECT_EQ(a.params.b, b.params.b);
            EXPECT_EQ(a.params.rho, b.params.rho);
            EXPECT_EQ(a.params.m, b.params.m);
            EXPECT_EQ(a.params.sigma, b.params.sigma);
            EXPECT_EQ(a.objective, b.objective);
            EXPECT_EQ(a.inner_solves, b.inner_solves);
        }
    }
}

// ---------------------------------------------------------------------------
// The comparison that justifies the reduction
// ---------------------------------------------------------------------------

TEST(SviCalibrator, MatchesTheAccuracyOfAMultiStartDirectFit) {
    // The honest version of the claim.  The quasi-explicit path is not more
    // *accurate* than a well-started 16-point multi-start LM -- it is the same
    // accuracy for less work and with no starting guess.  Asserting
    // superiority in accuracy would be asserting something false.
    for (auto regime : {MarketRegime::Normal, MarketRegime::HighVol,
                        MarketRegime::Earnings}) {
        for (auto& s : fitted_slices(regime)) {
            const auto qe = calibrate_svi_slice(s);
            const auto direct = calibrate_svi_slice_direct(s, svi_default_starts(s));
            ASSERT_TRUE(qe.ok()) << to_string(regime);
            ASSERT_TRUE(direct.ok()) << to_string(regime);
            // Within 25%.  Both paths now impose the same exact
            // variance-positivity condition, so the remaining difference is
            // the quasi-explicit path's wing bound and the finite resolution of
            // its outer search.
            EXPECT_LT(qe.rms_vol_error, direct.rms_vol_error * 1.25 + 1e-6)
                << to_string(regime) << ": qe " << qe.rms_vol_error << " vs direct "
                << direct.rms_vol_error;
        }
    }
}

TEST(SviCalibrator, NeedsNoStartingGuessWhereASingleStartCanFail) {
    // The failure mode the reduction actually protects against.  Run the
    // direct fit from each default start *separately*; on the crash regime the
    // spread of final objectives is enormous, and a single-start fit lands
    // materially above the global optimum a few percent of the time.  The
    // quasi-explicit path has no start to get wrong.
    long bad_single_starts = 0;
    long total_single_starts = 0;
    double worst_spread = 1.0;

    for (auto& s : fitted_slices(MarketRegime::Crash)) {
        const auto qe = calibrate_svi_slice(s);
        ASSERT_TRUE(qe.ok());
        double best = std::numeric_limits<double>::infinity();
        double worst = 0.0;
        for (const auto& start : svi_default_starts(s)) {
            const std::vector<std::vector<double>> one{start};
            const auto d = calibrate_svi_slice_direct(s, one);
            best = std::min(best, d.objective);
            worst = std::max(worst, d.objective);
            ++total_single_starts;
            if (d.objective > qe.objective * 1.10 + 1e-18) ++bad_single_starts;
        }
        if (best > 0.0) worst_spread = std::max(worst_spread, worst / best);
    }

    ASSERT_GT(total_single_starts, 50);
    // The measured figures in svi_calibrator.hpp are a 107x spread and 4/128
    // bad starts.  Asserting a floor keeps the documentation honest: if the
    // problem stopped having local minima, the justification would need
    // rewriting rather than silently becoming false.
    EXPECT_GT(worst_spread, 5.0)
        << "the crash regime no longer exhibits local minima (spread " << worst_spread
        << "x); the justification in svi_calibrator.hpp needs revisiting";
    EXPECT_GT(bad_single_starts, 0)
        << "no single start landed materially wrong; the documented failure mode is gone";
    // And it must still be a minority -- the claim is "rare but real", not
    // "multi-start is useless".
    EXPECT_LT(static_cast<double>(bad_single_starts) /
                  static_cast<double>(total_single_starts),
              0.25);
}

TEST(SviCalibrator, IsFasterThanAMultiStartDirectFit) {
    // Timing in a unit test is inherently noisy, so the bound is loose: the
    // claim being defended is a 1.7-2.9x speedup, and the assertion is merely
    // that the quasi-explicit path is not *slower*.  The precise figures come
    // from benchmarks/calibration, which controls for frequency scaling.
    using Clock = std::chrono::steady_clock;
    auto slices = fitted_slices(MarketRegime::Normal);
    ASSERT_GE(slices.size(), 5u);

    constexpr int kReps = 10;
    double qe_us = 0.0;
    double direct_us = 0.0;
    for (auto& s : slices) {
        const auto starts = svi_default_starts(s);
        const auto t0 = Clock::now();
        for (int r = 0; r < kReps; ++r) {
            const auto fit = calibrate_svi_slice(s);
            (void)fit.objective;
        }
        const auto t1 = Clock::now();
        for (int r = 0; r < kReps; ++r) {
            const auto fit = calibrate_svi_slice_direct(s, starts);
            (void)fit.objective;
        }
        const auto t2 = Clock::now();
        qe_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        direct_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
    }
    EXPECT_LT(qe_us, direct_us * 1.2)
        << "quasi-explicit " << qe_us << " us vs direct " << direct_us << " us";
}

TEST(SviCalibrator, ReportsButterflyViolationsWithoutRejectingTheFit) {
    // A slice that fits the data and violates the density condition is
    // informative -- it says the data itself is close to arbitrageable, and
    // the caller may prefer SSVI.  So it is reported, not suppressed.
    long violations = 0;
    long fits = 0;
    for (auto regime : {MarketRegime::Normal, MarketRegime::Crash, MarketRegime::Illiquid}) {
        for (std::uint64_t seed : {1u, 2u, 3u, 4u, 5u}) {
            for (auto& s : fitted_slices(regime, seed)) {
                const auto fit = calibrate_svi_slice(s);
                ++fits;
                if (fit.status == SviFitStatus::ButterflyViolation) {
                    ++violations;
                    EXPECT_TRUE(fit.ok()) << "a butterfly violation must not fail the fit";
                    EXPECT_LT(fit.butterfly.worst_g, 0.0);
                    EXPECT_GT(fit.diagnostics.count(DiagCode::ButterflyArbitrage), 0u);
                }
            }
        }
    }
    ASSERT_GT(fits, 50);
    // Noisy data on a steep skew does produce them; if it never did, the
    // check would be untested in practice.
    EXPECT_GT(violations, 0) << "no butterfly violation was produced across "
                             << fits << " fits, so the reporting path is untested";
}

TEST(SviCalibrator, ButterflyCheckLooksBeyondTheQuotedStrikes) {
    // The fitted slice will be *used* outside the quoted range -- that is the
    // point of having a parametric model -- so a violation just outside it is
    // exactly the one a grid restricted to the data would miss.
    auto slices = fitted_slices(MarketRegime::Crash);
    const auto fit = calibrate_svi_slice(slices.front());
    double k_min = 1e9, k_max = -1e9;
    for (const auto& q : slices.front()) {
        if (q.weight <= 0.0) continue;
        k_min = std::min(k_min, q.log_moneyness);
        k_max = std::max(k_max, q.log_moneyness);
    }
    // The reported worst point is allowed to lie outside the quoted range.
    EXPECT_TRUE(fit.butterfly.worst_k < k_min || fit.butterfly.worst_k > k_max ||
                (fit.butterfly.worst_k >= k_min && fit.butterfly.worst_k <= k_max))
        << "sanity";
    // Concretely: the search range must extend past the data.
    const auto narrow = svi_butterfly_check(fit.params, k_min, k_max);
    const auto wide = svi_butterfly_check(fit.params, k_min - 1.0, k_max + 1.0);
    EXPECT_LE(wide.worst_g, narrow.worst_g + 1e-15)
        << "a wider search cannot find a better worst case";
}
