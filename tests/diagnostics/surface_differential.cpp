// SPDX-License-Identifier: MIT
/// Validates the surface differential engine (brief section 2, VOLATILITY
/// LAB X's first flagship feature): the exact reconciliation identity, the
/// forward/shape orthogonality the header promises, sign/dominance checks
/// for skew and curvature, and the event-detection heuristic.
///
/// A recurring theme below: this is a *global* regression over a wide
/// moneyness range, not a local Taylor approximation, so "does the residual
/// shrink as the bump shrinks" is the WRONG test for skew/curvature -- see
/// ResidualFractionIsShapeComplexityNotBumpSize for why, and contrast with
/// the Taylor truncation residual in tests/portfolio/portfolio.cpp, where
/// shrinking the bump *does* shrink the residual because that one really is
/// a local truncation error.

#include "vl_test_support.hpp"

#include "volatility_lab/diagnostics/surface_differential.hpp"

#include <cmath>
#include <vector>

using namespace vl;

namespace {

/// A degenerate (b = 0) SVI slice collapses to w(k) = a, identically, for
/// every k: there is no smile term at all.  That makes vol(k, T) =
/// sqrt(a(T)/T) exactly flat in k, so choosing a(T) gives exact, known
/// control over vol(T) with zero contamination from the sqrt(w/T)
/// nonlinearity that makes a realistic smile mix into every basis term.
/// Tenors deliberately match `default_differential_grid()`'s exactly (not
/// 0.083 ~ 1/12, which differ in the last few bits and leave a tiny but
/// nonzero interpolation residual -- confirmed by probe, not a bug, just a
/// reason to match grids exactly in an *exact*-reconciliation test).
template <typename VolOfT>
VolSurface make_flat_surface(VolOfT vol_of_T) {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.0;
        p.rho = 0.0;
        p.m = 0.0;
        p.sigma = 0.1;
        const double v = vol_of_T(T);
        p.a = v * v * T;
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

/// A realistic smiling surface: SVI with nonzero b/rho/sigma, so skew and
/// curvature bumps have genuine (if imperfectly 4-parameter-describable)
/// content.
VolSurface make_smile_surface(double atm_vol, double rho, double fwd = 100.0) {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = rho;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = atm_vol * atm_vol * T -
              p.b * (p.rho * (0.0 - p.m) + std::sqrt(p.m * p.m + p.sigma * p.sigma));
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(fwd), TermCurve::flat(1.0));
}

/// Recomputes the reconciliation identity from the result's own per-point
/// arrays, independent of however compute_surface_differential derived them.
void expect_reconciles_exactly(const SurfaceDifferential& d) {
    ASSERT_EQ(d.grid_k.size(), d.grid_years.size());
    ASSERT_EQ(d.grid_k.size(), d.grid_delta_w.size());
    ASSERT_EQ(d.grid_k.size(), d.grid_fitted_w.size());
    ASSERT_EQ(d.grid_k.size(), d.grid_residual_w.size());
    for (std::size_t i = 0; i < d.grid_k.size(); ++i) {
        const double k = d.grid_k[i];
        const double T = d.grid_years[i];
        const double fitted =
            d.level_shift + d.skew_shift * k + d.curvature_shift * k * k + d.term_shift * T;
        EXPECT_NEAR(d.grid_fitted_w[i], fitted, 1e-12);
        EXPECT_NEAR(d.grid_delta_w[i], d.grid_fitted_w[i] + d.grid_residual_w[i], 1e-12);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Exact decomposition on surfaces where the move is fully known
// ---------------------------------------------------------------------------

TEST(SurfaceDifferential, SelfComparisonIsExactlyZeroEverywhere) {
    const auto surface = make_smile_surface(0.20, -0.4);
    const auto d = compute_surface_differential(surface, surface);
    EXPECT_EQ(d.level_shift, 0.0);
    EXPECT_EQ(d.skew_shift, 0.0);
    EXPECT_EQ(d.curvature_shift, 0.0);
    EXPECT_EQ(d.term_shift, 0.0);
    EXPECT_EQ(d.forward_shift, 0.0);
    EXPECT_EQ(d.event_shift, 0.0);
    EXPECT_EQ(d.residual_rms, 0.0);
    for (double dw : d.grid_delta_w) EXPECT_EQ(dw, 0.0);
}

TEST(SurfaceDifferential, UniformVolShiftIsPureLevelWithZeroResidual) {
    // The textbook "level" move: annualised vol up by 4 points at every
    // tenor and every strike.  On a flat (no-smile) surface this is exactly
    // representable by the level basis function alone.
    const auto old_surface = make_flat_surface([](double) { return 0.20; });
    const auto new_surface = make_flat_surface([](double) { return 0.24; });
    const auto d = compute_surface_differential(old_surface, new_surface);

    EXPECT_NEAR(d.level_shift, 0.04, 1e-10);
    EXPECT_NEAR(d.skew_shift, 0.0, 1e-10);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-10);
    EXPECT_NEAR(d.term_shift, 0.0, 1e-10);
    EXPECT_NEAR(d.residual_rms, 0.0, 1e-9);
    EXPECT_NEAR(d.explained_fraction(), 1.0, 1e-8);
    expect_reconciles_exactly(d);
}

TEST(SurfaceDifferential, VolShiftProportionalToTenorIsPureTermWithZeroResidual) {
    // Short tenors unchanged, long tenors up a lot: a pure term-structure
    // twist, with no uniform (level) component since the shift is zero at
    // T -> 0.
    const auto old_surface = make_flat_surface([](double) { return 0.20; });
    const auto new_surface = make_flat_surface([](double T) { return 0.20 + 0.05 * T; });
    const auto d = compute_surface_differential(old_surface, new_surface);

    EXPECT_NEAR(d.term_shift, 0.05, 1e-9);
    EXPECT_NEAR(d.level_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.skew_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.residual_rms, 0.0, 1e-7);
    EXPECT_NEAR(d.explained_fraction(), 1.0, 1e-6);
    expect_reconciles_exactly(d);
}

// ---------------------------------------------------------------------------
// Forward / shape orthogonality -- the property the header promises
// ---------------------------------------------------------------------------

TEST(SurfaceDifferential, ForwardOnlyMoveLeavesEveryShapeCoefficientExactlyZero) {
    // Same slices (same SVI params at every tenor), only the forward curve
    // differs. Because the surface is keyed by log-moneyness k = log(K/F),
    // the (k, T) variance/vol grid is queried at identical (k, T) points on
    // both surfaces regardless of where the forward sits -- the whole point
    // of that parameterisation (see volatility/surface.hpp) -- so level,
    // skew, curvature and term must come back *exactly* zero, not just
    // small: there is nothing for the shape regression to see.
    const auto old_surface = make_smile_surface(0.20, -0.4, /*fwd=*/100.0);
    const auto new_surface = make_smile_surface(0.20, -0.4, /*fwd=*/110.0);
    const auto d = compute_surface_differential(old_surface, new_surface);

    EXPECT_EQ(d.level_shift, 0.0);
    EXPECT_EQ(d.skew_shift, 0.0);
    EXPECT_EQ(d.curvature_shift, 0.0);
    EXPECT_EQ(d.term_shift, 0.0);
    EXPECT_EQ(d.residual_rms, 0.0);
    EXPECT_GT(d.forward_shift, 0.0);
    EXPECT_NEAR(d.forward_shift, std::log(110.0 / 100.0), 1e-12);
    expect_reconciles_exactly(d);
}

TEST(SurfaceDifferential, ForwardShiftIsZeroWhenTheForwardCurveIsUnchanged) {
    const auto old_surface = make_smile_surface(0.20, -0.4);
    const auto new_surface = make_smile_surface(0.24, -0.6);  // shape moves, forward doesn't
    const auto d = compute_surface_differential(old_surface, new_surface);
    EXPECT_EQ(d.forward_shift, 0.0);
}

// ---------------------------------------------------------------------------
// Sign and dominance for skew / curvature on a realistic smiling surface
// ---------------------------------------------------------------------------

TEST(SurfaceDifferential, SteepeningNegativeSkewProducesNegativeSkewShift) {
    // rho more negative => a steeper downward (put-skew) smile. The SVI
    // shape term is b*rho*(k-m), linear in k at m=0, so this should be
    // dominated by the skew bucket.
    const auto old_surface = make_smile_surface(0.20, -0.4);
    const auto new_surface = make_smile_surface(0.20, -0.7);
    const auto d = compute_surface_differential(old_surface, new_surface);

    EXPECT_LT(d.skew_shift, 0.0);
    EXPECT_GT(std::abs(d.skew_shift), std::abs(d.level_shift));
    EXPECT_GT(std::abs(d.skew_shift), std::abs(d.curvature_shift));
    expect_reconciles_exactly(d);
}

TEST(SurfaceDifferential, FlatteningSkewProducesPositiveSkewShift) {
    const auto old_surface = make_smile_surface(0.20, -0.7);
    const auto new_surface = make_smile_surface(0.20, -0.4);
    const auto d = compute_surface_differential(old_surface, new_surface);
    EXPECT_GT(d.skew_shift, 0.0);
}

TEST(SurfaceDifferential, ResidualFractionIsShapeComplexityNotBumpSize) {
    // An SVI rho perturbation's effect on vol(k) is not confined to the
    // (1, k, k^2) subspace the regression's basis spans: the sqrt term in
    // SVI's own formula has k-structure beyond quadratic. Because BOTH the
    // part the regression captures and the part it cannot scale linearly
    // with the bump size to leading order, the *fraction* residual/total
    // stays roughly constant as the bump shrinks -- it does not vanish the
    // way a Taylor-truncation residual would (contrast with
    // tests/portfolio/portfolio.cpp's ResidualShrinksAsTheMoveShrinks, where
    // the residual really is a local truncation error).  This is a genuine,
    // bounded characteristic of fitting a wide-moneyness-range smile with a
    // 4-parameter model, not a bug, and is tested directly here so a future
    // change that accidentally "fixes" it (e.g. by only testing a tiny bump)
    // does not mask this with a misleadingly small residual.
    const double base_rho = -0.4;
    const auto base = make_smile_surface(0.20, base_rho);

    std::vector<double> fractions;
    for (double drho : {0.30, 0.10, 0.03, 0.01}) {
        const auto bumped = make_smile_surface(0.20, base_rho - drho);
        const auto d = compute_surface_differential(base, bumped);
        ASSERT_GT(d.total_delta_rms, 0.0);
        fractions.push_back(d.residual_rms / d.total_delta_rms);
    }
    // All of them sit in a similar band -- not literally equal (higher-order
    // terms are still present at the margins), but nowhere near shrinking
    // toward zero across a 30x range of bump sizes.
    for (double f : fractions) {
        EXPECT_GT(f, 0.3);
        EXPECT_LT(f, 0.9);
    }
    EXPECT_NEAR(fractions.front(), fractions.back(), 0.15);
}

TEST(SurfaceDifferential, PureCurvatureChangeIsDominatedByTheCurvatureBucket) {
    // sigma controls the smile's wing curvature in SVI; rho = 0 removes the
    // linear skew term entirely so the change is as close to "pure
    // curvature" as a realistic smile parameterisation gets.
    std::vector<SliceVariant> old_slices, new_slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p_old, p_new;
        p_old.years = p_new.years = T;
        p_old.b = p_new.b = 0.08;
        p_old.rho = p_new.rho = 0.0;
        p_old.m = p_new.m = 0.0;
        p_old.sigma = 0.20;
        p_new.sigma = 0.08;  // narrower -> sharper, more curved smile
        const double atm = 0.20;
        p_old.a = atm * atm * T - p_old.b * p_old.sigma;
        p_new.a = atm * atm * T - p_new.b * p_new.sigma;
        old_slices.emplace_back(svi_project_to_admissible(p_old));
        new_slices.emplace_back(svi_project_to_admissible(p_new));
    }
    const VolSurface old_surface(std::move(old_slices), TermCurve::flat(100.0),
                                 TermCurve::flat(1.0));
    const VolSurface new_surface(std::move(new_slices), TermCurve::flat(100.0),
                                 TermCurve::flat(1.0));
    const auto d = compute_surface_differential(old_surface, new_surface);

    EXPECT_GT(std::abs(d.curvature_shift), std::abs(d.skew_shift));
    EXPECT_GT(std::abs(d.curvature_shift), std::abs(d.term_shift));
    expect_reconciles_exactly(d);
}

// ---------------------------------------------------------------------------
// Event detection
// ---------------------------------------------------------------------------

namespace {

/// Base surface plus a version where only the front slice's ATM level has
/// been bumped hard (20% -> 35%), every other tenor untouched exactly.
std::pair<VolSurface, VolSurface> make_front_tenor_bump_pair() {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = -0.4;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = 0.20 * 0.20 * T -
              p.b * (p.rho * (0.0 - p.m) + std::sqrt(p.m * p.m + p.sigma * p.sigma));
        slices.emplace_back(svi_project_to_admissible(p));
    }
    VolSurface old_surface(slices, TermCurve::flat(100.0), TermCurve::flat(1.0));

    SviParams bumped = std::get<SviParams>(slices[0]);
    bumped.a = 0.35 * 0.35 * bumped.years -
               bumped.b * (bumped.rho * (0.0 - bumped.m) +
                          std::sqrt(bumped.m * bumped.m + bumped.sigma * bumped.sigma));
    std::vector<SliceVariant> new_slices = slices;
    new_slices[0] = svi_project_to_admissible(bumped);
    VolSurface new_surface(std::move(new_slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
    return {std::move(old_surface), std::move(new_surface)};
}

}  // namespace

TEST(SurfaceDifferential, FrontTenorOnlyBumpIsFlaggedAsAnEvent) {
    // Bump only the shortest-tenor slice's level; every other tenor is
    // exactly untouched. Leave-one-tenor-out correctly measures this as "the
    // rest of the surface has no information about the front tenor's move",
    // which is exactly what an isolated event is.
    const auto [old_surface, new_surface] = make_front_tenor_bump_pair();
    const auto d = compute_surface_differential(old_surface, new_surface);
    EXPECT_GT(d.event_shift, 0.0);
    expect_reconciles_exactly(d);
}

TEST(SurfaceDifferential, GenuineIsolatedEventScoresHigherThanAnOrdinaryShapeChange) {
    // This is the case the earlier (shortest-tenor-vs-global-residual)
    // version of the heuristic got backwards: it scored an ordinary skew
    // change -- present at every tenor by construction, not an event in any
    // sense -- *higher* than a deliberate, large, single-tenor-only bump,
    // because the single-tenor bump dragged the global fit enough to spread
    // comparably-sized residual across every other tenor too. The
    // leave-one-tenor-out heuristic does not have that failure mode: each
    // tenor is judged against a fit that never saw its own data, so the
    // isolated bump cannot contaminate its own evidence against itself.
    const auto [front_old, front_new] = make_front_tenor_bump_pair();
    const auto d_front = compute_surface_differential(front_old, front_new);

    const auto skew_old = make_smile_surface(0.20, -0.4);
    const auto skew_new = make_smile_surface(0.20, -0.7);
    const auto d_skew = compute_surface_differential(skew_old, skew_new);

    EXPECT_GT(d_front.event_shift, 0.0);
    EXPECT_GT(d_skew.event_shift, 0.0);  // skew change isn't perfectly explained either
    EXPECT_GT(d_front.event_shift, d_skew.event_shift)
        << "an isolated single-tenor bump must score higher than a change present at "
           "every tenor, even though the skew change's raw residual is larger overall";
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST(SurfaceDifferential, EmptyGridReturnsAllZerosRatherThanCrashing) {
    const auto old_surface = make_smile_surface(0.20, -0.4);
    const auto new_surface = make_smile_surface(0.24, -0.6);
    const auto d = compute_surface_differential(old_surface, new_surface,
                                                 std::span<const DifferentialGridPoint>{});
    // No explicit grid means the default grid is used (see header), not an
    // empty one -- this just confirms that path is well-defined. A truly
    // empty-after-default-is-non-empty case isn't reachable through the
    // public API, so what's tested here is that passing {} falls through to
    // the default grid rather than degenerating.
    EXPECT_FALSE(d.grid_k.empty());
}

TEST(SurfaceDifferential, SingularGridLeavesCoefficientsAtZeroAndEverythingAsResidual) {
    // Three points, all at the same single tenor and same k: far fewer
    // distinct directions than the four basis functions need, so the normal
    // matrix is singular and the honest answer is "could not decompose
    // this", not a silently wrong decomposition.
    std::vector<DifferentialGridPoint> grid = {{0.0, 0.5}, {0.0, 0.5}, {0.0, 0.5}};
    const auto old_surface = make_smile_surface(0.20, -0.4);
    const auto new_surface = make_smile_surface(0.24, -0.6);
    const auto d = compute_surface_differential(old_surface, new_surface, grid);

    EXPECT_EQ(d.level_shift, 0.0);
    EXPECT_EQ(d.skew_shift, 0.0);
    EXPECT_EQ(d.curvature_shift, 0.0);
    EXPECT_EQ(d.term_shift, 0.0);
    for (std::size_t i = 0; i < d.grid_delta_w.size(); ++i) {
        EXPECT_NEAR(d.grid_residual_w[i], d.grid_delta_w[i], 1e-12);
    }
}

TEST(SurfaceDifferential, DefaultGridHasTheDocumentedShape) {
    const auto grid = default_differential_grid();
    EXPECT_EQ(grid.size(), 25u);
    for (const auto& pt : grid) {
        EXPECT_GE(pt.years, 1.0 / 12.0 - 1e-12);
        EXPECT_LE(pt.years, 2.0 + 1e-12);
        EXPECT_GE(pt.k, -0.4 - 1e-12);
        EXPECT_LE(pt.k, 0.4 + 1e-12);
    }
}
