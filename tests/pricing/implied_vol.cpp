// SPDX-License-Identifier: MIT
/// Validates the implied-volatility inversion: round-trip accuracy over the
/// whole domain, iteration-count bounds, the convergence-proof assumptions,
/// rejection of out-of-domain input, and agreement with a structurally
/// independent double-double bisection.

#include "vl_test_support.hpp"

#include "volatility_lab/pricing/implied_vol.hpp"
#include "volatility_lab/pricing/reference.hpp"

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

/// Published figures from implied_vol.hpp.  Set just above the measured
/// values so that a regression fails rather than quietly widening the claim.
/// Round-trip error is asserted as a multiple of the problem own conditioning
/// floor (pricer accuracy / elasticity), not as an absolute figure.  A value of
/// 1 would mean "exactly as accurate as the problem permits"; 40 leaves room
/// for the handful of points where the solver stops on its step tolerance
/// slightly before the floor.
constexpr double kConditioningSlack = 40.0;

/// Absolute bound, used only where the problem is known to be well
/// conditioned (elasticity >= 1).
constexpr double kRoundTripRtol = 1.0e-9;
constexpr double kMeanIterations = 4.5;    ///< measured ~3.3
constexpr int kMaxIterations = 40;
constexpr long kMaxFallbacks = 25;  ///< measured a handful over ~7k points

/// Enumerate the admissible (x, beta) inputs: sweep (x, s), price, and keep
/// the points where the price is representable and strictly inside the
/// no-arbitrage bounds.
struct Sample {
    double abs_x;
    double s;
    double beta;

    /// d(log b)/d(log s) = s*vega/b at this point: how much a relative price
    /// error is damped (>1) or amplified (<1) by the inversion.  The round-trip
    /// error cannot be smaller than the pricer relative accuracy divided by
    /// this, and near the upper price bound it falls to 1e-10, so a flat
    /// tolerance over the domain would be asserting accuracy the problem does
    /// not contain.
    double elasticity;

    /// The resulting floor on relative error in the recovered s.
    [[nodiscard]] double error_floor() const {
        constexpr double kPricerRtol = 3.0e-12;
        return kPricerRtol / std::max(elasticity, 1e-300);
    }
};

std::vector<Sample> admissible_samples(int nx = 110, int ns = 110) {
    std::vector<Sample> out;
    const auto dom = vl::test::normalised_domain(nx, ns);
    for (double ax : dom.abs_x) {
        for (double s : dom.s) {
            const auto nb = normalised_black(-ax, s);
            const double beta = nb.value;

            // Representability floor: below this the price is a denormal or
            // zero and carries no information.
            if (!(beta > 1e-290)) continue;

            // **Saturation.**  Once b has reached exp(x/2) to the last bit,
            // every larger s produces the same double, so the inversion is not
            // merely inaccurate -- it is ill-posed, and no solver can recover
            // which s was used.  Including those points would be testing the
            // solver against information that was destroyed before it was
            // called.  The engine reports this region through
            // `BlackBranch::Saturated`, which is exactly the signal needed
            // here.
            if (nb.branch == BlackBranch::Saturated) continue;
            if (beta >= std::exp(-0.5 * ax) * (1.0 - 1e-12)) continue;

            const double elasticity = s * nb.dv_ds / beta;
            out.push_back({ax, s, beta, elasticity});
        }
    }
    return out;
}

}  // namespace

// ===========================================================================
// Round trip
// ===========================================================================

TEST(ImpliedVol, RoundTripRecoversTotalVolatilityOverTheFullDomain) {
    WorstCase w;
    long failures = 0;
    long fallbacks = 0;
    long total_iters = 0;
    long solved = 0;
    int max_iters = 0;

    for (const auto& smp : admissible_samples()) {
        const auto r = implied_total_volatility(smp.beta, -smp.abs_x);
        if (!r.ok()) {
            ++failures;
            continue;
        }
        ++solved;
        total_iters += r.iterations;
        max_iters = std::max(max_iters, r.iterations);
        if (r.used_fallback) ++fallbacks;
        // Scaled by the conditioning floor, so the assertion is "the solver
        // got as close as the problem allows" rather than "the solver hit a
        // fixed number of digits" -- which over this domain would be a claim
        // about the inversion that no implementation could satisfy.
        const double err = rel_error(r.total_volatility, smp.s);
        w.observe(err / std::max(smp.error_floor(), 1e-15), smp.abs_x, smp.s,
                  r.total_volatility, smp.s);
    }

    ASSERT_GT(solved, 5000) << "the sweep must actually cover the domain";
    EXPECT_EQ(failures, 0) << "every admissible input must invert";
    EXPECT_LT(w.error, kConditioningSlack) << w.describe("|x|", "s");
    EXPECT_LT(static_cast<double>(total_iters) / static_cast<double>(solved),
              kMeanIterations)
        << "mean iteration count regressed";
    EXPECT_LE(max_iters, kMaxIterations);
    EXPECT_LE(fallbacks, kMaxFallbacks)
        << "the Brent fallback is being used more than documented (" << fallbacks
        << " times); the fast path has regressed";
}

TEST(ImpliedVol, AgreesWithTheDoubleDoubleBisectionReference) {
    // The reference shares no structure with the production solver: no
    // derivatives, no initial guess, no branch logic, 120 halvings at 106
    // bits.  Agreement is therefore evidence rather than coincidence.
    WorstCase w;
    for (const auto& smp : admissible_samples(60, 60)) {
        const auto r = implied_total_volatility(smp.beta, -smp.abs_x);
        if (!r.ok()) continue;
        const double ref = reference::implied_vol_ref(smp.beta, -smp.abs_x);
        if (!std::isfinite(ref) || ref <= 0.0) {
            w.skip();
            continue;
        }
        w.observe(rel_error(r.total_volatility, ref) /
                      std::max(smp.error_floor(), 1e-15),
                  smp.abs_x, smp.s, r.total_volatility, ref);
    }
    EXPECT_LT(w.error, kConditioningSlack) << w.describe("|x|", "s");
}

TEST(ImpliedVol, IsAccurateInAbsoluteTermsWhereTheProblemIsWellConditioned) {
    // The absolute claim, restricted to where it is meaningful: wherever the
    // inversion damps rather than amplifies price error (elasticity >= 1,
    // which covers every option a market actually quotes), the recovered total
    // volatility is good to 1e-9 relative.
    WorstCase w;
    for (const auto& smp : admissible_samples()) {
        if (smp.elasticity < 1.0) {
            w.skip();
            continue;
        }
        const auto r = implied_total_volatility(smp.beta, -smp.abs_x);
        if (!r.ok()) continue;
        w.observe(rel_error(r.total_volatility, smp.s), smp.abs_x, smp.s,
                  r.total_volatility, smp.s);
    }
    EXPECT_GT(w.samples, 3000);
    EXPECT_LT(w.error, kRoundTripRtol) << w.describe("|x|", "s");
}

TEST(ImpliedVol, PriceSpaceRoundTripRecoversSigma) {
    WorstCase w;
    const auto dom = vl::test::market_domain();
    for (double ratio : dom.strike_ratio) {
        for (double v : dom.vol) {
            for (double T : dom.years) {
                const double F = 100.0;
                const double K = F * ratio;
                for (auto type : {OptionType::Call, OptionType::Put}) {
                    const double px = black_undiscounted(F, K, v, T, type);
                    const auto r = implied_volatility_undiscounted(px, F, K, T, type);
                    if (!r.ok()) {
                        w.skip();
                        continue;
                    }
                    // The achievable accuracy is a property of the input, not
                    // of the solver: an ITM quote has already lost the
                    // information.  The result reports that bound, and this is
                    // the assertion that the bound is honest.
                    const double budget = std::max(r.attainable_rtol * 4.0, 1e-12);
                    const double err = rel_error(r.volatility, v);
                    w.observe(err / budget, K, v, r.volatility, v);
                }
            }
        }
    }
    EXPECT_LT(w.error, 1.0)
        << "a result exceeded its own reported attainable accuracy: "
        << w.describe("strike", "vol");
}

TEST(ImpliedVol, AttainableRtolIsLargeForItmAndTinyForOtm) {
    // The field exists to stop a caller trusting 1e-14 on a quote that only
    // determines three digits.  Check it actually discriminates.
    const double itm_px = black_undiscounted(100.0, 25.0, 0.2, 1.0, OptionType::Call);
    const auto itm = implied_volatility_undiscounted(itm_px, 100.0, 25.0, 1.0,
                                                     OptionType::Call);
    ASSERT_TRUE(itm.ok());
    EXPECT_GT(itm.attainable_rtol, 1e-6) << "deep ITM should report poor conditioning";
    EXPECT_GE(itm.attainable_rtol, rel_error(itm.volatility, 0.2) * 0.5)
        << "the reported bound must not understate the actual error";

    const double otm_px = black_undiscounted(100.0, 200.0, 0.2, 1.0, OptionType::Call);
    const auto otm = implied_volatility_undiscounted(otm_px, 100.0, 200.0, 1.0,
                                                     OptionType::Call);
    ASSERT_TRUE(otm.ok());
    EXPECT_LT(otm.attainable_rtol, 1e-12) << "OTM quotes are well conditioned";
    // The discriminating claim: ITM is orders of magnitude worse than OTM.
    EXPECT_GT(itm.attainable_rtol / otm.attainable_rtol, 1e4);
    EXPECT_GE(otm.attainable_rtol, rel_error(otm.volatility, 0.2) * 0.5);
}

TEST(ImpliedVol, AttainableRtolReflectsTheCollapseOfElasticityNearTheUpperBound) {
    // As the price approaches its sigma -> infinity bound it stops depending on
    // volatility, and no inversion can recover sigma.  The reported bound has
    // to show that, or it is advertising digits that do not exist.
    const double F = 100.0, K = F * std::exp(6.4), T = 1.0;
    const double v = 14.9;  // total volatility ~14.9 at T = 1
    const double px = black_undiscounted(F, K, v, T, OptionType::Call);
    const auto r = implied_volatility_undiscounted(px, F, K, T, OptionType::Call);
    ASSERT_TRUE(r.ok()) << to_string(r.status);
    EXPECT_GT(r.attainable_rtol, 1e-8)
        << "near-saturation conditioning is not being reported";
    EXPECT_GE(r.attainable_rtol, rel_error(r.volatility, v) * 0.5)
        << "the reported bound understates the actual error";
}

// ===========================================================================
// The convergence-proof assumptions
// ===========================================================================

TEST(ImpliedVol, AtmInvertsInClosedFormWithNoIterations) {
    for (double s : vl::test::log_space(1e-5, 20.0, 50)) {
        const double beta = normalised_black_value(0.0, s);
        if (!(beta > 0.0) || beta >= 1.0) continue;
        const auto r = implied_total_volatility(beta, 0.0);
        ASSERT_TRUE(r.ok()) << "s = " << s;
        EXPECT_EQ(r.iterations, 0) << "ATM must not iterate";
        EXPECT_EQ(static_cast<int>(r.method), static_cast<int>(math::SolveMethod::ClosedForm));
        // The limit here is how well `beta` pins down `s`, not the inversion.
        // d(beta)/d(s) = phi(s/2)/2, so near s = 20 a one-ulp change in beta
        // moves s by ~1e-7 -- the closed form is exact, the input is not.
        const double conditioning =
            2.0 * math::kEps / std::max(math::norm_pdf(0.5 * s), 1e-300);
        EXPECT_LT(rel_error(r.total_volatility, s),
                  std::max(4.0 * conditioning / s, 1e-13))
            << "s = " << s << ", beta = " << beta;
    }
}

TEST(ImpliedVol, SolvedRootLiesInTheBranchTheInflectionPredicts) {
    // The whole convergence argument rests on beta < b_c implying the root is
    // below s_c.  Verified directly rather than assumed.
    for (const auto& smp : admissible_samples(50, 50)) {
        const auto infl = normalised_black_inflection(-smp.abs_x);
        if (infl.s_c == 0.0) continue;
        const auto r = implied_total_volatility(smp.beta, -smp.abs_x);
        if (!r.ok()) continue;
        if (smp.beta < infl.b_c) {
            EXPECT_LE(r.total_volatility, infl.s_c * (1.0 + 1e-9))
                << "|x| = " << smp.abs_x << ", beta = " << smp.beta;
        } else {
            EXPECT_GE(r.total_volatility, infl.s_c * (1.0 - 1e-9))
                << "|x| = " << smp.abs_x << ", beta = " << smp.beta;
        }
    }
}

TEST(ImpliedVol, ResidualIsConsistentWithTheReportedRoot) {
    for (const auto& smp : admissible_samples(40, 40)) {
        const auto r = implied_total_volatility(smp.beta, -smp.abs_x);
        if (!r.ok()) continue;
        const double recomputed = normalised_black_value(-smp.abs_x, r.total_volatility) -
                                  smp.beta;
        // The reported residual must be the real one, within one more
        // evaluation's worth of rounding.
        EXPECT_LT(std::abs(r.residual - recomputed),
                  1e-13 * std::max(smp.beta, 1e-300) + 1e-300)
            << "|x| = " << smp.abs_x << ", s = " << smp.s;
    }
}

// ===========================================================================
// Initial guess
// ===========================================================================

TEST(ImpliedVol, InitialGuessIsWithinAFactorOfTwo) {
    // Regression test for the bug that made this 250x wrong: a guess helper
    // substituted s_c when it had no answer, which fooled the |h| test into
    // treating a 7-sigma option as near the money.
    WorstCase w;
    for (const auto& smp : admissible_samples(70, 70)) {
        const double g = implied_vol_initial_guess(smp.beta, -smp.abs_x);
        ASSERT_GT(g, 0.0) << "|x| = " << smp.abs_x << ", s = " << smp.s;
        ASSERT_TRUE(std::isfinite(g));
        w.observe(rel_error(g, smp.s), smp.abs_x, smp.s, g, smp.s);
    }
    EXPECT_LT(w.error, 1.0) << w.describe("|x|", "s");
}

TEST(ImpliedVol, InitialGuessStaysInsideTheKnownBranch) {
    for (const auto& smp : admissible_samples(50, 50)) {
        const auto infl = normalised_black_inflection(-smp.abs_x);
        if (infl.s_c == 0.0) continue;
        const double g = implied_vol_initial_guess(smp.beta, -smp.abs_x);
        if (smp.beta < infl.b_c) {
            EXPECT_LE(g, infl.s_c) << "|x| = " << smp.abs_x;
        } else {
            EXPECT_GE(g, infl.s_c) << "|x| = " << smp.abs_x;
        }
    }
}

// ===========================================================================
// Named edge cases from the brief
// ===========================================================================

TEST(ImpliedVol, NamedEdgeCases) {
    struct Case {
        const char* name;
        double strike;
        double vol;
        double years;
        OptionType type;
        double rtol;
    };
    // Each tolerance is the conditioning of that case, not a tuned number.
    const Case cases[] = {
        {"deep OTM call", 400.0, 0.20, 0.08, OptionType::Call, 1e-13},
        {"deep ITM call", 25.0, 0.20, 1.00, OptionType::Call, 1e-3},
        {"near expiry 1h ATM", 100.0, 0.20, 1.0 / 8760, OptionType::Call, 1e-12},
        {"very low vol", 100.0, 0.001, 1.00, OptionType::Call, 1e-11},
        {"very high vol", 100.0, 5.00, 1.00, OptionType::Call, 1e-13},
        {"tiny T and low vol", 100.0, 0.01, 1e-5, OptionType::Call, 1e-10},
        {"deep OTM put", 10.0, 0.30, 0.50, OptionType::Put, 1e-13},
        {"5y 300 pct vol", 100.0, 3.00, 5.00, OptionType::Call, 1e-13},
        {"strike 1e5", 1e5, 0.40, 2.00, OptionType::Call, 1e-13},
        {"strike 1e-3", 1e-3, 0.40, 2.00, OptionType::Put, 1e-13},
    };
    for (const auto& c : cases) {
        const double px = black_undiscounted(100.0, c.strike, c.vol, c.years, c.type);
        const auto r = implied_volatility_undiscounted(px, 100.0, c.strike, c.years, c.type);
        ASSERT_TRUE(r.ok()) << c.name << ": " << to_string(r.status);
        EXPECT_LT(rel_error(r.volatility, c.vol), c.rtol)
            << c.name << ": got " << r.volatility << ", expected " << c.vol
            << ", attainable " << r.attainable_rtol;
        EXPECT_LE(r.iterations, kMaxIterations) << c.name;
    }
}

TEST(ImpliedVol, DeepOtmNearExpiryIsReportedAsAtIntrinsicNotAsAnError) {
    // A 10%-OTM option one hour from expiry at 20% vol is genuinely worth
    // exp(-995): zero in double.  The volatility is not determined by that
    // input, and the library must say which kind of "cannot answer" this is.
    const double px = black_undiscounted(100.0, 110.0, 0.2, 1.0 / 8760, OptionType::Call);
    EXPECT_EQ(px, 0.0);
    const auto r = implied_volatility_undiscounted(px, 100.0, 110.0, 1.0 / 8760,
                                                   OptionType::Call);
    EXPECT_EQ(static_cast<int>(r.status), static_cast<int>(IvStatus::PriceAtIntrinsic));
    EXPECT_FALSE(r.ok());
}

// ===========================================================================
// Out-of-domain input
// ===========================================================================

TEST(ImpliedVol, OutOfBoundsPricesAreRejectedNotApproximated) {
    const double F = 100.0, K = 110.0, T = 1.0;
    const auto b = forward_price_bounds(F, K, OptionType::Call);

    const auto below = implied_volatility_undiscounted(-1.0, F, K, T, OptionType::Call);
    EXPECT_EQ(static_cast<int>(below.status),
              static_cast<int>(IvStatus::PriceBelowIntrinsic));

    const auto at_intrinsic =
        implied_volatility_undiscounted(b.lower, F, K, T, OptionType::Call);
    EXPECT_EQ(static_cast<int>(at_intrinsic.status),
              static_cast<int>(IvStatus::PriceAtIntrinsic));

    const auto at_bound = implied_volatility_undiscounted(b.upper, F, K, T,
                                                          OptionType::Call);
    EXPECT_EQ(static_cast<int>(at_bound.status),
              static_cast<int>(IvStatus::PriceAboveBound));

    const auto above = implied_volatility_undiscounted(b.upper * 1.5, F, K, T,
                                                       OptionType::Call);
    EXPECT_EQ(static_cast<int>(above.status), static_cast<int>(IvStatus::PriceAboveBound));

    // One ulp inside the upper bound must still solve.
    const auto inside = implied_volatility_undiscounted(std::nextafter(b.upper, 0.0), F, K,
                                                        T, OptionType::Call);
    EXPECT_TRUE(inside.ok()) << to_string(inside.status);
    EXPECT_GT(inside.volatility, 1.0);
}

TEST(ImpliedVol, InvalidInputsAreRejected) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(5.0, 0.0, 100.0, 1.0, OptionType::Call)
                      .status),
              static_cast<int>(IvStatus::InvalidInput));
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(5.0, 100.0, 0.0, 1.0, OptionType::Call)
                      .status),
              static_cast<int>(IvStatus::InvalidInput));
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(5.0, 100.0, 100.0, 0.0, OptionType::Call)
                      .status),
              static_cast<int>(IvStatus::InvalidInput));
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(nan, 100.0, 100.0, 1.0, OptionType::Call)
                      .status),
              static_cast<int>(IvStatus::InvalidInput));
    // Discounted entry point rejects a non-positive discount factor.
    EXPECT_EQ(static_cast<int>(
                  implied_volatility(5.0, 100.0, 100.0, 1.0, 0.0, OptionType::Call).status),
              static_cast<int>(IvStatus::InvalidInput));
}

TEST(ImpliedVol, DiscountedAndUndiscountedEntryPointsAgree) {
    const double F = 100.0, K = 95.0, T = 1.5, v = 0.3, df = 0.94;
    const double und = black_undiscounted(F, K, v, T, OptionType::Call);
    const auto a = implied_volatility_undiscounted(und, F, K, T, OptionType::Call);
    const auto b = implied_volatility(df * und, F, K, T, df, OptionType::Call);
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(b.ok());
    EXPECT_LT(rel_error(b.volatility, a.volatility), 1e-13);
}

TEST(ImpliedVol, VolFloorAndCeilingAreReported) {
    ImpliedVolConfig cfg;
    cfg.vol_floor = 0.05;
    cfg.vol_ceiling = 0.50;
    const double low = black_undiscounted(100.0, 100.0, 0.01, 1.0, OptionType::Call);
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(low, 100.0, 100.0, 1.0, OptionType::Call,
                                                  cfg)
                      .status),
              static_cast<int>(IvStatus::BelowFloor));
    const double high = black_undiscounted(100.0, 100.0, 2.0, 1.0, OptionType::Call);
    EXPECT_EQ(static_cast<int>(
                  implied_volatility_undiscounted(high, 100.0, 100.0, 1.0, OptionType::Call,
                                                  cfg)
                      .status),
              static_cast<int>(IvStatus::AboveCeiling));
}

// ===========================================================================
// Alternative solvers -- the justification for the specialised path
// ===========================================================================

TEST(ImpliedVol, SpecialisedPathBeatsEveryGenericSolver) {
    // This is the test that earns the specialised implementation.  All six
    // solvers run the same sweep with the fallback disabled; the specialised
    // one must be strictly better than each generic alternative on both
    // failure count and mean iterations, or it is not worth its complexity.
    struct Row {
        math::SolveMethod method;
        long failures = 0;
        long solved = 0;
        long iters = 0;
        double worst = 0.0;
    };
    std::vector<Row> rows{{math::SolveMethod::HouseholderNormalisedBlack},
                          {math::SolveMethod::Halley},
                          {math::SolveMethod::Newton},
                          {math::SolveMethod::SafeguardedNewton},
                          {math::SolveMethod::Brent},
                          {math::SolveMethod::Bisection}};

    const auto samples = admissible_samples(70, 70);
    ImpliedVolConfig cfg;
    cfg.allow_fallback = false;

    for (auto& row : rows) {
        for (const auto& smp : samples) {
            const auto r =
                (row.method == math::SolveMethod::HouseholderNormalisedBlack)
                    ? implied_total_volatility(smp.beta, -smp.abs_x, cfg)
                    : implied_total_volatility_with(row.method, smp.beta, -smp.abs_x, cfg);
            if (!r.ok()) {
                ++row.failures;
                continue;
            }
            ++row.solved;
            row.iters += r.iterations;
            row.worst = std::max(row.worst, rel_error(r.total_volatility, smp.s) /
                                            std::max(smp.error_floor(), 1e-15));
        }
    }

    const auto& spec = rows[0];
    const double spec_mean =
        static_cast<double>(spec.iters) / static_cast<double>(std::max(1L, spec.solved));

    // Claim 1: the specialised path is the fastest of the six, strictly.
    for (std::size_t i = 1; i < rows.size(); ++i) {
        const auto& other = rows[i];
        const double other_mean =
            static_cast<double>(other.iters) / static_cast<double>(std::max(1L, other.solved));
        EXPECT_LT(spec_mean, other_mean)
            << "specialised path (" << spec_mean << " mean iters) is not faster than "
            << math::to_string(other.method) << " (" << other_mean << ")";
    }

    // Claim 2: it is also robust -- but note the honest form of the claim.
    // With the fallback disabled, as here, bisection has a *lower* failure
    // count (zero, unconditionally: that is its entire character) at 16x the
    // iterations.  So the right statement is not "better than everything on
    // every axis" -- that would be false -- but "fastest, and with a fast-path
    // failure rate below 0.1%, which the Brent fallback then takes to zero".
    const double spec_failure_rate =
        static_cast<double>(spec.failures) /
        static_cast<double>(std::max(1L, spec.failures + spec.solved));
    EXPECT_LT(spec_failure_rate, 1e-3)
        << spec.failures << " fast-path failures out of "
        << (spec.failures + spec.solved);

    // Claim 3: the accuracy is comparable.  Speed bought with accuracy is not
    // a win, so every solver that converges must land within the same budget --
    // and that budget is the conditioning floor, which they all share because
    // it is a property of the problem rather than of the method.
    for (const auto& row : rows) {
        if (row.solved == 0) continue;
        EXPECT_LT(row.worst, kConditioningSlack) << math::to_string(row.method);
    }
}

TEST(ImpliedVol, BisectionAlwaysConvergesAndActsAsTheOracle) {
    // Bisection depends on nothing but the sign of the residual, so it cannot
    // share a bug with any interpolating method.  It must never fail.
    ImpliedVolConfig cfg;
    cfg.allow_fallback = false;
    cfg.max_iterations = 200;
    long failures = 0;
    for (const auto& smp : admissible_samples(40, 40)) {
        const auto r = implied_total_volatility_with(math::SolveMethod::Bisection, smp.beta,
                                                     -smp.abs_x, cfg);
        if (!r.ok()) ++failures;
    }
    EXPECT_EQ(failures, 0);
}
