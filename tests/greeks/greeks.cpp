// SPDX-License-Identifier: MIT
/// Validates the analytic Greeks engine against the independent dd-reference
/// chain, checks the structural identities a Greeks implementation has to
/// satisfy regardless of the formula (put-call parity relationships, the
/// forward-delta equivalence, aggregation linearity), and exercises the
/// Taylor-vs-exact repricing diagnostic (brief section 9).
///
/// This suite is also where three real issues were found by the act of
/// writing it, two in production and one in how the dd reference has to be
/// used:
///
///  * `charm` had the wrong sign (production computed dDelta/dT; the
///    documented and reference convention is dDelta/dt = -dDelta/dT).
///  * `taylor_vs_exact_reprice` compared its exact-reprice result against
///    `OptionGreeks::price`, a *different* formula (direct d1/d2) from the
///    one `black_scholes_price` uses (the normalised-Black erfcx form) --
///    so even a zero-size bump produced a nonzero "exact PnL" at the ~1e-15
///    level.  Fixed by recomputing the base price from the same formula as
///    the bumped one.
///  * the dd reference's own finite difference loses the tiny genuine
///    second-order Greeks on the *ITM* side of a strike: differencing a
///    price of 123.6 cannot resolve a true signal of 1e-68 sitting three
///    full `double` exponents below the dd working precision of that 123.6.
///    Production does not have this problem -- its closed-form derivation
///    shares vanna/volga/speed algebraically between calls and puts, so it
///    returns the correct, tiny, call-side-consistent value on the ITM leg
///    too -- but comparing *that* against the reference's ITM finite
///    difference was comparing a correct number against reference noise.
///    The fix mirrors `pricing/black.hpp`'s own architecture: validate each
///    strike against the reference only on its natural OTM leg (where both
///    sides can resolve the signal), and rely on the exact put-call parity
///    identities below -- which need no external reference at all -- to
///    cover the ITM leg.
///
/// All three are fixed at the source; the regression tests below pin them.

#include "vl_test_support.hpp"

#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/pricing/black.hpp"
#include "volatility_lab/pricing/reference.hpp"

#include <cmath>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

struct Case {
    double S, K, vol, T, r, q;
    OptionType type;
};

std::vector<Case> domain_sweep() {
    std::vector<Case> out;
    const auto strikes = vl::test::log_space(20.0, 500.0, 9);
    const auto vols = vl::test::log_space(0.02, 2.0, 7);
    const auto years = vl::test::log_space(1.0 / 365.0, 5.0, 7);
    const double rates[] = {-0.02, 0.0, 0.03, 0.08};
    const double carries[] = {0.0, 0.01, 0.04};
    for (double K : strikes) {
        for (double v : vols) {
            for (double T : years) {
                for (double r : rates) {
                    for (double q : carries) {
                        out.push_back({100.0, K, v, T, r, q, OptionType::Call});
                        out.push_back({100.0, K, v, T, r, q, OptionType::Put});
                    }
                }
            }
        }
    }
    return out;
}

/// The same grid as `domain_sweep`, but keeping only the natural OTM leg at
/// each (K, vol, T, r, q) point -- exactly mirroring how `black_undiscounted`
/// itself decides which side to price directly.  This is the set validated
/// against the dd reference; the ITM leg is validated instead by the exact
/// put-call parity identities, which do not depend on the reference at all.
/// See the file comment for why comparing the ITM leg to the reference
/// directly does not work.
std::vector<Case> domain_sweep_otm_only() {
    std::vector<Case> out;
    for (const auto& c : domain_sweep()) {
        const double forward = c.S * std::exp((c.r - c.q) * c.T);
        const bool call_is_otm = c.K >= forward;
        if ((c.type == OptionType::Call) == call_is_otm) out.push_back(c);
    }
    return out;
}

/// Relative error, but with an absolute floor so a Greek that is correctly
/// near zero (vanna and volga both vanish at d2 == 0, which happens whenever
/// ln(F/K) == -s^2/2, not only exactly at the money) does not register as a
/// near-100%-wrong comparison between two numbers that are each individually
/// below any economically meaningful size.
///
/// The floor also has to absorb the reference's *own* finite-difference
/// noise floor, which is not the same for every Greek: an n-th order
/// derivative's finite-difference formula divides a ~1e-31-relative-precision
/// dd price evaluation by h^n, so round-off is amplified by roughly
/// eps_dd/h^n.  With h ~= 1e-9*scale for the first/second derivatives here
/// and h ~= 1e-6*scale for the third (see the widened `hS3` step in
/// `black_scholes_greeks_ref`), that works out to a reference noise ceiling
/// of roughly 1e-22 (first order, negligible), 1e-13 (second order) and
/// 1e-13 (third order, after widening) -- *provided* the base price is O(1).
/// A base price further from 1 (either direction) shifts that ceiling by the
/// same factor, which is why the empirically observed noise ranges from
/// ~1e-14 to ~3e-11 across the domain swept below.  Each call site's floor is
/// set an order of magnitude above the worst noise actually measured for
/// that Greek (see the comments at each `EXPECT_LT` below), not guessed.
double mixed_error(double a, double b, double floor = 1e-10) {
    const double scale = std::max({std::abs(a), std::abs(b), floor});
    return std::abs(a - b) / scale;
}

}  // namespace

// ===========================================================================
// Agreement with the independent double-double reference
// ===========================================================================

TEST(Greeks, MatchesTheDoubleDoubleReferenceAcrossTheDomain) {
    // One independent reference, ten fields, one sweep -- restricted to the
    // OTM leg of each strike, which is the only side the reference's own
    // finite difference can resolve (see the file comment).  Each field gets
    // its own WorstCase so a single bad Greek cannot hide inside an
    // aggregate.
    WorstCase delta, gamma, vega, theta, rho, vanna, volga, charm, speed, price;
    for (const auto& c : domain_sweep_otm_only()) {
        const auto g = black_scholes_greeks(c.S, c.K, c.vol, c.T, c.r, c.q, c.type);
        const auto ref =
            reference::black_scholes_greeks_ref(c.S, c.K, c.vol, c.T, c.r, c.q, c.type);

        // Floors: 1e-10 for the essentially-exact first-order Greeks (where
        // call/ref agree to ~1e-15 everywhere actually measured).  1e-6 for
        // vanna/volga/charm/speed: these are the four Greeks whose reference
        // value legitimately passes through, or close to, zero at isolated
        // points across the domain (vanna and volga at d2 == 0; all four
        // wherever the dd finite-difference's own round-off, amplified by
        // dividing by h^2 or h^3, exceeds the true tiny value) -- measured
        // worst-case reference noise across this sweep was ~2.7e-11, so
        // 1e-6 is roughly five further orders of magnitude above that, and
        // it is also the point below which no real options book would
        // notice the Greek at all (a vanna or volga of 1e-6 per unit vol per
        // unit spot is immaterial on any book size this library targets).
        // gamma keeps the tighter floor: unlike the other three, gamma is
        // never small purely because of a sign-indifferent zero-crossing in
        // *this* parameterisation, so there is no equivalent reason to
        // relax it, and the domain sweep's only other disagreement for
        // gamma (3.6e-8 relative, at 200% annualised vol) is a genuine
        // small relative truncation at an economically real magnitude, which
        // the relative half of this metric already covers correctly.
        price.observe(mixed_error(g.price, ref.price, 1e-10), c.K, c.vol, g.price, ref.price);
        delta.observe(mixed_error(g.delta, ref.delta, 1e-10), c.K, c.vol, g.delta, ref.delta);
        gamma.observe(mixed_error(g.gamma, ref.gamma, 1e-8), c.K, c.vol, g.gamma, ref.gamma);
        vega.observe(mixed_error(g.vega, ref.vega, 1e-10), c.K, c.vol, g.vega, ref.vega);
        theta.observe(mixed_error(g.theta, ref.theta, 1e-10), c.K, c.vol, g.theta, ref.theta);
        rho.observe(mixed_error(g.rho, ref.rho, 1e-10), c.K, c.vol, g.rho, ref.rho);
        vanna.observe(mixed_error(g.vanna, ref.vanna, 1e-6), c.K, c.vol, g.vanna, ref.vanna);
        volga.observe(mixed_error(g.volga, ref.volga, 1e-6), c.K, c.vol, g.volga, ref.volga);
        charm.observe(mixed_error(g.charm, ref.charm, 1e-6), c.K, c.vol, g.charm, ref.charm);
        speed.observe(mixed_error(g.speed, ref.speed, 1e-6), c.K, c.vol, g.speed, ref.speed);
    }

    EXPECT_LT(price.error, 1e-10) << price.describe("K", "vol");
    EXPECT_LT(delta.error, 1e-10) << delta.describe("K", "vol");
    // gamma: a second-derivative finite difference, so its own reference is
    // typically good to ~1e-7 relative even away from any floor region -- not
    // the ~1e-14 the first-order Greeks achieve.  Measured worst case 3.6e-8,
    // at 200% annualised volatility, where price itself varies sharply enough
    // in S that the stencil's own truncation becomes the limit.
    EXPECT_LT(gamma.error, 1e-6) << gamma.describe("K", "vol");
    EXPECT_LT(vega.error, 1e-9) << vega.describe("K", "vol");
    EXPECT_LT(theta.error, 1e-9) << theta.describe("K", "vol");
    EXPECT_LT(rho.error, 1e-9) << rho.describe("K", "vol");
    EXPECT_LT(vanna.error, 1e-4) << vanna.describe("K", "vol");
    EXPECT_LT(volga.error, 5e-4) << volga.describe("K", "vol");  // measured worst 1.56e-4
    // charm: regression bound for the sign bug.  A flipped sign anywhere in
    // the domain would push the worst-case error to roughly 2.0 (equal
    // magnitude, opposite sign), so this bound is nowhere near tight enough
    // to pass by accident of a cancelling pair of sign errors.
    EXPECT_LT(charm.error, 1e-4) << charm.describe("K", "vol");
    // speed: regression bound for the reference-stencil step-size bug. The
    // under-stepped reference disagreed by up to 3.3e-4 before the stencil
    // was widened; this is still well inside that margin.
    EXPECT_LT(speed.error, 1e-4) << speed.describe("K", "vol");
}

TEST(Greeks, ItmLegAgreesWithTheOtmLegThroughExactSymmetryWhereTheReferenceCannotSee) {
    // The flip side of restricting the reference comparison to the OTM leg:
    // this test exists specifically to cover the ITM leg, by exploiting the
    // fact that vanna, volga and speed have **no** w = payoff_sign(type) term
    // in their closed forms (see black_scholes_greeks), so they are bitwise
    // identical between a call and a put at the same strike by construction.
    // That identity does not depend on the dd reference at all, so it is
    // exactly as strong a check on the ITM leg as the reference comparison is
    // on the OTM leg -- it just cannot, by itself, catch a bug that is
    // *symmetric* between calls and puts (which is what PutCallParity and the
    // OTM-side reference comparison are for).
    for (const auto& c : domain_sweep()) {
        const auto call = black_scholes_greeks(c.S, c.K, c.vol, c.T, c.r, c.q, OptionType::Call);
        const auto put = black_scholes_greeks(c.S, c.K, c.vol, c.T, c.r, c.q, OptionType::Put);
        EXPECT_EQ(call.vanna, put.vanna) << "K=" << c.K << " vol=" << c.vol << " T=" << c.T;
        EXPECT_EQ(call.volga, put.volga) << "K=" << c.K << " vol=" << c.vol << " T=" << c.T;
        EXPECT_EQ(call.gamma, put.gamma) << "K=" << c.K << " vol=" << c.vol << " T=" << c.T;
        EXPECT_EQ(call.speed, put.speed) << "K=" << c.K << " vol=" << c.vol << " T=" << c.T;
    }
}

TEST(Greeks, CharmSignMatchesTheCalendarDecayConvention) {
    // A direct, formula-free check of the convention documented in
    // greeks.hpp: charm = dDelta/dt = -dDelta/dT.  Bump T forward by a tiny
    // amount (time passing, T shrinking, so dt = +h corresponds to dT = -h)
    // and confirm Delta moves in the direction charm predicts, independent of
    // the dd reference entirely.
    const double S = 100, K = 95, vol = 0.22, T = 0.75, r = 0.03, q = 0.01;
    const auto base = black_scholes_greeks(S, K, vol, T, r, q, OptionType::Call);
    const double h = 1e-6;
    const auto later =
        black_scholes_greeks(S, K, vol, T - h, r, q, OptionType::Call);  // time has passed
    const double observed_ddelta_dt = (later.delta - base.delta) / h;
    EXPECT_LT(rel_error(base.charm, observed_ddelta_dt), 1e-4)
        << "charm = " << base.charm << " but dDelta/dt measured as "
        << observed_ddelta_dt;
}

// ===========================================================================
// Structural identities (formula-independent)
// ===========================================================================

TEST(Greeks, PutCallParityHoldsForEveryGreek) {
    // Call - Put = exp(-qT)*S - exp(-rT)*K differentiated term by term gives
    // an exact relationship for every Greek, independent of the volatility
    // model.  This is checked across the same domain sweep as the reference
    // comparison, but it does not need any reference at all -- it is true by
    // construction of Black-Scholes, so a violation would mean the call and
    // put formulas disagree with each other, not merely with an external
    // source.
    WorstCase delta_w, gamma_w, vega_w, rho_w, vanna_w, volga_w, charm_w, speed_w;
    for (const auto& c : domain_sweep()) {
        if (c.type != OptionType::Call) continue;
        const auto call = black_scholes_greeks(c.S, c.K, c.vol, c.T, c.r, c.q, OptionType::Call);
        const auto put = black_scholes_greeks(c.S, c.K, c.vol, c.T, c.r, c.q, OptionType::Put);

        const double dq = std::exp(-c.q * c.T);
        const double df = std::exp(-c.r * c.T);

        // Price: C - P = S*exp(-qT) - K*exp(-rT)
        EXPECT_NEAR(call.price - put.price, c.S * dq - c.K * df,
                    1e-9 * std::max(1.0, c.S))
            << "K=" << c.K << " vol=" << c.vol << " T=" << c.T;

        delta_w.observe(mixed_error(call.delta - put.delta, dq), c.K, c.vol, call.delta - put.delta, dq);
        gamma_w.observe(mixed_error(call.gamma, put.gamma), c.K, c.vol, call.gamma, put.gamma);
        vega_w.observe(mixed_error(call.vega, put.vega), c.K, c.vol, call.vega, put.vega);
        rho_w.observe(mixed_error(call.rho - put.rho, c.K * c.T * df), c.K, c.vol,
                     call.rho - put.rho, c.K * c.T * df);
        vanna_w.observe(mixed_error(call.vanna, put.vanna), c.K, c.vol, call.vanna, put.vanna);
        volga_w.observe(mixed_error(call.volga, put.volga), c.K, c.vol, call.volga, put.volga);
        // Delta_call - Delta_put = exp(-qT) = dq, a function of T (time to
        // expiry).  Charm is d/dt (calendar time), and dT/dt = -1, so
        // d/dt[dq] = d/dT[exp(-qT)] * (-1) = (-q*exp(-qT)) * (-1) = +q*dq.
        charm_w.observe(mixed_error(call.charm - put.charm, c.q * dq), c.K, c.vol,
                        call.charm - put.charm, c.q * dq);
        speed_w.observe(mixed_error(call.speed, put.speed), c.K, c.vol, call.speed, put.speed);
    }
    EXPECT_LT(delta_w.error, 1e-9) << "Delta_call - Delta_put != exp(-qT): " << delta_w.describe("K", "vol");
    EXPECT_LT(gamma_w.error, 1e-9) << "Gamma differs between call and put: " << gamma_w.describe("K", "vol");
    EXPECT_LT(vega_w.error, 1e-9) << "Vega differs between call and put: " << vega_w.describe("K", "vol");
    EXPECT_LT(rho_w.error, 1e-7) << "Rho_call - Rho_put != K*T*DF: " << rho_w.describe("K", "vol");
    EXPECT_LT(vanna_w.error, 1e-7) << "Vanna differs between call and put: " << vanna_w.describe("K", "vol");
    EXPECT_LT(volga_w.error, 1e-6) << "Volga differs between call and put: " << volga_w.describe("K", "vol");
    EXPECT_LT(charm_w.error, 1e-6) << "Charm_call - Charm_put != -q*exp(-qT): " << charm_w.describe("K", "vol");
    EXPECT_LT(speed_w.error, 1e-6) << "Speed differs between call and put: " << speed_w.describe("K", "vol");
}

TEST(Greeks, ForwardOverloadAgreesWithTheSpotFormulaAtMatchedParameters) {
    // black_scholes_greeks_forward(F, K, vol, T, r, DF) is documented to equal
    // black_scholes_greeks(spot=F, K, vol, T, r, carry=r) -- i.e. the
    // "forward delta" convention where dF/dS=1 at the chosen spot.  Checked
    // directly rather than merely asserted.
    for (const auto& c : domain_sweep()) {
        const double forward = c.S * std::exp((c.r - c.q) * c.T);
        const double discount = std::exp(-c.r * c.T);
        const auto via_forward =
            black_scholes_greeks_forward(forward, c.K, c.vol, c.T, c.r, discount, c.type);
        const auto via_spot_equiv =
            black_scholes_greeks(forward, c.K, c.vol, c.T, c.r, c.r, c.type);
        EXPECT_EQ(via_forward.delta, via_spot_equiv.delta);
        EXPECT_EQ(via_forward.price, via_spot_equiv.price);
        EXPECT_EQ(via_forward.gamma, via_spot_equiv.gamma);
    }
}

// ===========================================================================
// Degenerate inputs
// ===========================================================================

TEST(Greeks, ZeroTimeCollapsesToIntrinsicAndAStepDelta) {
    const auto itm = black_scholes_greeks(110, 100, 0.2, 0.0, 0.03, 0.0, OptionType::Call);
    EXPECT_NEAR(itm.price, 10.0, 1e-12);
    EXPECT_EQ(itm.delta, 1.0);
    EXPECT_EQ(itm.gamma, 0.0);
    EXPECT_EQ(itm.vega, 0.0);

    const auto otm = black_scholes_greeks(90, 100, 0.2, 0.0, 0.03, 0.0, OptionType::Call);
    EXPECT_EQ(otm.price, 0.0);
    EXPECT_EQ(otm.delta, 0.0);
}

TEST(Greeks, ZeroVolatilityCollapsesToIntrinsic) {
    const auto g = black_scholes_greeks(100, 90, 0.0, 1.0, 0.03, 0.0, OptionType::Call);
    EXPECT_NEAR(g.price, black_scholes_price(100, 90, 0.0, 1.0, 0.03, 0.0, OptionType::Call),
                1e-12);
    EXPECT_EQ(g.vega, 0.0);
}

TEST(Greeks, InvalidInputsReportNaNRatherThanAPlausibleNumber) {
    EXPECT_TRUE(std::isnan(black_scholes_greeks(0.0, 100, 0.2, 1.0, 0.03, 0.0,
                                                OptionType::Call)
                               .price));
    EXPECT_TRUE(std::isnan(black_scholes_greeks(100, -5.0, 0.2, 1.0, 0.03, 0.0,
                                                OptionType::Call)
                               .price));
    EXPECT_TRUE(std::isnan(
        black_scholes_greeks_forward(0.0, 100, 0.2, 1.0, 0.03, 0.9, OptionType::Call).price));
}

// ===========================================================================
// Batch and aggregation
// ===========================================================================

TEST(Greeks, BatchAgreesWithScalarBitwise) {
    const auto cases = domain_sweep();
    std::vector<double> S, K, vol, T, r, q;
    std::vector<std::int8_t> sign;
    for (const auto& c : cases) {
        S.push_back(c.S);
        K.push_back(c.K);
        vol.push_back(c.vol);
        T.push_back(c.T);
        r.push_back(c.r);
        q.push_back(c.q);
        sign.push_back(static_cast<std::int8_t>(c.type));
    }
    std::vector<OptionGreeks> batch(cases.size());
    black_scholes_greeks_batch(S, K, vol, T, r, q, sign, batch);

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto scalar =
            black_scholes_greeks(S[i], K[i], vol[i], T[i], r[i], q[i], cases[i].type);
        ASSERT_EQ(batch[i].price, scalar.price) << "i=" << i;
        ASSERT_EQ(batch[i].delta, scalar.delta) << "i=" << i;
        ASSERT_EQ(batch[i].charm, scalar.charm) << "i=" << i;
        ASSERT_EQ(batch[i].speed, scalar.speed) << "i=" << i;
    }
}

TEST(Greeks, BatchHandlesMismatchedSpanLengthsWithoutReadingOutOfBounds) {
    std::vector<double> S{100, 100, 100}, K{100}, vol{0.2, 0.2}, T{1.0, 1.0, 1.0}, r{0.03},
        q{0.0, 0.0, 0.0};
    std::vector<std::int8_t> sign{1, 1, 1};
    std::vector<OptionGreeks> out(3);
    black_scholes_greeks_batch(S, K, vol, T, r, q, sign, out);  // shortest span is 1
    EXPECT_TRUE(std::isfinite(out[0].price));
}

TEST(Greeks, AggregationIsLinearAndMatchesAHandComputedPortfolio) {
    // Two positions: long 10 contracts of A, short 5 of B, multiplier 100.
    OptionGreeks a{}, b{};
    a.price = 5.0;
    a.delta = 0.6;
    a.gamma = 0.02;
    b.price = 3.0;
    b.delta = -0.3;
    b.gamma = 0.015;
    const OptionGreeks greeks[] = {a, b};
    const double qty[] = {10.0, -5.0};
    const double mult[] = {100.0, 100.0};

    const auto agg = aggregate_greeks(greeks, qty, mult);
    EXPECT_NEAR(agg.value, 10 * 100 * 5.0 + (-5) * 100 * 3.0, 1e-9);
    EXPECT_NEAR(agg.delta, 10 * 100 * 0.6 + (-5) * 100 * (-0.3), 1e-9);
    EXPECT_NEAR(agg.gamma, 10 * 100 * 0.02 + (-5) * 100 * 0.015, 1e-9);

    // Linearity: aggregating twice the quantities doubles every field.
    const double qty2[] = {20.0, -10.0};
    const auto agg2 = aggregate_greeks(greeks, qty2, mult);
    EXPECT_NEAR(agg2.delta, 2.0 * agg.delta, 1e-9);
    EXPECT_NEAR(agg2.value, 2.0 * agg.value, 1e-9);
}

TEST(Greeks, AggregationSkipsZeroWeightPositionsEvenIfTheirGreeksAreNaN) {
    // A closed-out position (quantity 0) must not poison the sum even if its
    // Greeks happen to be garbage -- e.g. a stale snapshot of an expired
    // instrument.
    OptionGreeks live{}, dead{};
    live.delta = 0.5;
    dead.delta = std::numeric_limits<double>::quiet_NaN();
    const OptionGreeks greeks[] = {live, dead};
    const double qty[] = {1.0, 0.0};
    const double mult[] = {1.0, 1.0};
    const auto agg = aggregate_greeks(greeks, qty, mult);
    EXPECT_EQ(agg.delta, 0.5);
}

// ===========================================================================
// Taylor vs exact repricing
// ===========================================================================

TEST(Greeks, ZeroBumpGivesZeroPnlAtEveryOrder) {
    const auto g = black_scholes_greeks(100, 100, 0.2, 1.0, 0.03, 0.0, OptionType::Call);
    const auto cmp = taylor_vs_exact_reprice(g, 100, 100, 0.2, 1.0, 0.03, 0.0, OptionType::Call,
                                             GreekBump{});
    EXPECT_EQ(cmp.exact_pnl, 0.0);
    EXPECT_EQ(cmp.order1_pnl, 0.0);
    EXPECT_EQ(cmp.order2_pnl, 0.0);
}

TEST(Greeks, ExactPnlReconcilesWithDirectRepricing) {
    const double S = 100, K = 105, vol = 0.25, T = 1.0, r = 0.02, q = 0.0;
    const auto g = black_scholes_greeks(S, K, vol, T, r, q, OptionType::Call);
    GreekBump bump{5.0, 0.05, 0.005, -0.1};
    const auto cmp = taylor_vs_exact_reprice(g, S, K, vol, T, r, q, OptionType::Call, bump);

    const double expected_exact =
        black_scholes_price(S + bump.d_spot, K, vol + bump.d_vol, T + bump.d_years,
                            r + bump.d_rate, q, OptionType::Call);
    EXPECT_NEAR(cmp.exact_price, expected_exact, 1e-10);
    EXPECT_NEAR(cmp.exact_pnl, expected_exact - g.price, 1e-10);
}

TEST(Greeks, SecondOrderIsNeverWorseThanFirstOrderForSmallBumps) {
    // The defining property of a Taylor series: for a sufficiently small
    // bump, each additional order should not increase the error (it may not
    // decrease it either, if the function is locally linear, but it must not
    // get materially worse).  Checked over a sweep of small bumps on an
    // ordinary option, which is the regime the approximation is meant for.
    const double S = 100, K = 100, vol = 0.2, T = 1.0, r = 0.03, q = 0.0;
    const auto g = black_scholes_greeks(S, K, vol, T, r, q, OptionType::Call);
    long worse_count = 0, total = 0;
    for (double dS : {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}) {
        for (double dV : {-0.02, -0.01, 0.0, 0.01, 0.02}) {
            GreekBump bump{dS, dV, 0.0, 0.0};
            const auto cmp = taylor_vs_exact_reprice(g, S, K, vol, T, r, q, OptionType::Call, bump);
            ++total;
            if (std::abs(cmp.order2_error()) > std::abs(cmp.order1_error()) + 1e-9) {
                ++worse_count;
            }
        }
    }
    EXPECT_EQ(worse_count, 0) << worse_count << "/" << total
                              << " small bumps got worse at second order";
}

TEST(Greeks, ApproximationErrorGrowsWithBumpSizeForALargeCrashShock) {
    // The property the "risk model error" concept (brief section 9) exists to
    // surface: for a large shock, exact repricing and the Taylor estimate
    // diverge, and that divergence must actually be visible in
    // order2_relative_error, not hidden by a lucky cancellation.
    const double S = 100, K = 100, vol = 0.2, T = 0.5, r = 0.03, q = 0.0;
    const auto g = black_scholes_greeks(S, K, vol, T, r, q, OptionType::Call);

    const auto small = taylor_vs_exact_reprice(g, S, K, vol, T, r, q, OptionType::Call,
                                               GreekBump{-1.0, 0.01, 0.0, 0.0});
    const auto crash = taylor_vs_exact_reprice(g, S, K, vol, T, r, q, OptionType::Call,
                                               GreekBump{-20.0, 0.15, 0.0, 0.0});

    EXPECT_LT(std::abs(small.order2_error()), std::abs(crash.order2_error()))
        << "a -20% spot / +15 vol-point shock must show more Taylor error than a -1% move";
    // And for the crash-sized move, the first-order approximation must be
    // visibly worse than the second-order one -- this is what motivates
    // carrying gamma/vanna/volga/charm at all, rather than delta/vega alone.
    EXPECT_LT(std::abs(crash.order2_error()), std::abs(crash.order1_error()))
        << "second order did not improve on first order for a large, smooth shock";
}

TEST(Greeks, RelativeErrorAccessorsAgreeWithTheRawFields) {
    const double S = 100, K = 100, vol = 0.2, T = 1.0, r = 0.03, q = 0.0;
    const auto g = black_scholes_greeks(S, K, vol, T, r, q, OptionType::Call);
    const auto cmp = taylor_vs_exact_reprice(g, S, K, vol, T, r, q, OptionType::Call,
                                             GreekBump{-5.0, 0.03, 0.0, 0.0});
    EXPECT_NEAR(cmp.order1_relative_error() * std::abs(cmp.exact_pnl), cmp.order1_error(), 1e-9);
    EXPECT_NEAR(cmp.order2_relative_error() * std::abs(cmp.exact_pnl), cmp.order2_error(), 1e-9);
}
