// SPDX-License-Identifier: MIT
#pragma once
/// \file svi.hpp
/// \brief Raw SVI: the five-parameter slice parameterisation.
///
/// ## The model
///
///     w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )
///
/// with k = log(K/F) and w = total implied variance.  Five parameters per
/// expiry, each with a direct reading:
///
///     a      vertical level        (minimum total variance, roughly)
///     b      overall wing slope    (b >= 0; b*(1+-rho) are the wing slopes)
///     rho    skew                  (|rho| < 1; negative tilts the put wing up)
///     m      horizontal shift      (where the minimum sits in k)
///     sigma  smoothness at the bottom (sigma > 0; sigma -> 0 gives a kink)
///
/// The two wings are asymptotically linear with slopes b(1 - rho) on the left
/// and b(1 + rho) on the right, which is the right shape: Lee's moment formula
/// says total variance must grow at most linearly in |k|, and SVI is the
/// simplest smooth form that is linear at both ends and convex in between.
///
/// ## Why this and not a polynomial
///
/// A quadratic or cubic in k fitted to implied variance will, with probability
/// one, turn over in the wings and produce negative variance a few strikes
/// outside the quoted range -- and then the extrapolated price is not merely
/// inaccurate but arbitrageable.  SVI cannot do that: w is bounded below by
/// a + b*sigma*sqrt(1 - rho^2) and grows linearly outward, by construction.
/// That is the entire reason a five-parameter nonlinear fit is worth the
/// trouble over a three-parameter linear one.
///
/// ## Derivatives
///
/// With y = k - m and r = sqrt(y^2 + sigma^2):
///
///     w   = a + b (rho*y + r)
///     w'  = b (rho + y/r)
///     w'' = b sigma^2 / r^3
///
/// All closed form, all cheap, and w'' > 0 whenever b > 0 and sigma > 0 -- so
/// a raw SVI slice is **always convex in k**.  Convexity is necessary for
/// absence of butterfly arbitrage but not sufficient, which is why
/// `svi_butterfly_free` below checks the actual Durrleman condition rather
/// than stopping at convexity.
///
/// ## Parameter constraints
///
/// Three are structural (violating them makes the formula meaningless) and one
/// is economic:
///
///     b >= 0, sigma > 0, |rho| < 1          structural
///     a + b*sigma*sqrt(1 - rho^2) >= 0      w >= 0 everywhere
///
/// The last is exact, not conservative: the minimum of w over k is attained at
/// y = -rho*sigma/sqrt(1-rho^2) and equals a + b*sigma*sqrt(1-rho^2).  The
/// calibrator enforces it as a hard bound rather than penalising it, because a
/// negative variance is not a bad fit, it is not a fit at all.

#include <cmath>
#include <limits>
#include <span>

#include "volatility_lab/core/config.hpp"
#include "volatility_lab/volatility/slice.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Raw SVI parameters for one expiry.  Trivially copyable and 48 bytes, so a
/// whole surface is cheap to clone -- see the note in slice.hpp about why that
/// matters for the scenario engine.
struct SviParams {
    double a = 0.0;      ///< level
    double b = 0.0;      ///< wing slope scale, >= 0
    double rho = 0.0;    ///< skew, |rho| < 1
    double m = 0.0;      ///< horizontal shift
    double sigma = 0.1;  ///< curvature scale at the bottom, > 0
    double years = 0.0;  ///< the expiry this slice belongs to

    static constexpr int kNumParams = 5;
};

static_assert(std::is_trivially_copyable_v<SviParams>);

/// A flat slice: constant total variance.  The degenerate baseline, used as
/// the starting point for a cold calibration and as the fallback when a slice
/// has too few quotes to identify five parameters.
struct FlatParams {
    double w = 0.0;
    double years = 0.0;
};

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

[[nodiscard]] VL_FORCE_INLINE double svi_total_variance(const SviParams& p,
                                                        double k) noexcept {
    const double y = k - p.m;
    const double r = std::sqrt(std::fma(y, y, p.sigma * p.sigma));
    return std::fma(p.b, std::fma(p.rho, y, r), p.a);
}

[[nodiscard]] VL_FORCE_INLINE SliceJet svi_jet(const SviParams& p, double k) noexcept {
    const double y = k - p.m;
    const double s2 = p.sigma * p.sigma;
    const double r = std::sqrt(std::fma(y, y, s2));
    SliceJet j;
    j.w = std::fma(p.b, std::fma(p.rho, y, r), p.a);
    if (r > 0.0) {
        const double inv_r = 1.0 / r;
        j.dw = p.b * (p.rho + y * inv_r);
        j.d2w = p.b * s2 * inv_r * inv_r * inv_r;
    } else {
        // sigma == 0 and k == m: the vertex of a degenerate (kinked) slice.
        // The one-sided derivatives differ; report the symmetric value and let
        // the admissibility check flag sigma == 0 separately.
        j.dw = p.b * p.rho;
        j.d2w = std::numeric_limits<double>::infinity();
    }
    return j;
}

/// Batch evaluation.  One call per slice per batch, so the loop body is a
/// direct inlinable expression with no dispatch -- see slice.hpp.
void svi_total_variance_batch(const SviParams& p, std::span<const double> k,
                              std::span<double> w) noexcept;

/// The minimum of w over all k, attained at k = m - rho*sigma/sqrt(1-rho^2).
/// Exact, so it can be used as a hard constraint rather than a penalty.
[[nodiscard]] VL_FORCE_INLINE double svi_min_variance(const SviParams& p) noexcept {
    return p.a + p.b * p.sigma * std::sqrt(std::max(0.0, 1.0 - p.rho * p.rho));
}

/// Where that minimum sits.
[[nodiscard]] VL_FORCE_INLINE double svi_argmin(const SviParams& p) noexcept {
    const double d = std::sqrt(std::max(1e-300, 1.0 - p.rho * p.rho));
    return p.m - p.rho * p.sigma / d;
}

/// Asymptotic wing slopes, dw/dk as k -> -inf and k -> +inf.  Lee's bound
/// requires both to be at most 2 in magnitude.
struct SviWings {
    double left;   ///< -b(1 - rho)
    double right;  ///< +b(1 + rho)
};

[[nodiscard]] VL_FORCE_INLINE SviWings svi_wings(const SviParams& p) noexcept {
    return {-p.b * (1.0 - p.rho), p.b * (1.0 + p.rho)};
}

// ---------------------------------------------------------------------------
// Admissibility
// ---------------------------------------------------------------------------

/// Structural and positivity constraints.  Cheap; checked before any
/// evaluation and on every calibration step.
[[nodiscard]] bool svi_parameters_admissible(const SviParams& p) noexcept;

/// Clamp a parameter vector into the admissible set.
///
/// Used by the calibrator's projection step.  `a` is raised rather than `b`
/// lowered when the positivity constraint binds, because `a` is the parameter
/// the data identifies least well (it trades off against b*sigma) and moving
/// it does least damage to the fit.
[[nodiscard]] SviParams svi_project_to_admissible(SviParams p) noexcept;

/// Whether the slice is free of butterfly arbitrage, by evaluating the
/// Durrleman function on a grid and reporting the worst point.
///
/// A grid search rather than an analytic condition, deliberately.  Sufficient
/// conditions in SVI parameter space exist but are conservative -- they reject
/// slices that are in fact fine -- and, more importantly, they cannot say
/// *where* a bad slice is bad.  Section 11 of the brief requires locating the
/// violation, and a caller fixing a surface needs the strike.
///
/// The grid is dense near the money and spreads out in the wings, where g is
/// smooth and monotone; `kDefaultGridPoints` was chosen by checking against a
/// 100x denser grid over the fuzz corpus (tests/calibration).
struct ButterflyCheck {
    bool arbitrage_free = true;
    double worst_g = 0.0;      ///< min over the grid of g(k); negative is bad
    double worst_k = 0.0;      ///< where it occurred
    double worst_density = 0.0;  ///< the implied density there
};

inline constexpr int kDefaultGridPoints = 257;

[[nodiscard]] ButterflyCheck svi_butterfly_check(const SviParams& p, double k_lo,
                                                 double k_hi,
                                                 int grid_points = kDefaultGridPoints) noexcept;

// ---------------------------------------------------------------------------
// The quasi-explicit reduction (Zeliade)
// ---------------------------------------------------------------------------
//
// This is the piece that makes SVI calibration tractable, and it is worth
// stating separately from the calibrator that uses it.
//
// Fix (m, sigma).  Substitute
//
//     z = (k - m) / sigma,   so   w = a + b*sigma*(rho*z + sqrt(z^2 + 1))
//
// and change variables to
//
//     adash = a,   d = rho*b*sigma,   c = b*sigma
//
// Then
//
//     w(z) = adash + d*z + c*sqrt(z^2 + 1)
//
// which is **linear in (adash, d, c)**.  So for any fixed (m, sigma) the
// optimal (adash, d, c) is the solution of a three-variable linear least
// squares problem -- solvable exactly, in closed form, with no iteration and
// no initial guess.  The five-parameter nonlinear fit collapses to a
// two-parameter outer search over (m, sigma) with an exact inner solve.
//
// The payoff is not merely speed.  It is that the outer problem is
// two-dimensional, so it can be searched globally (a coarse grid plus a local
// refinement) instead of hoped at from a single starting point.  Five-
// parameter SVI fits are notorious for landing in local minima; a 2-D outer
// problem removes that failure mode almost entirely.  Measured against
// multi-start Levenberg-Marquardt on the raw five parameters in
// benchmarks/calibration.
//
// The inner problem is a constrained least squares, because (adash, d, c) must
// satisfy the admissibility constraints translated into the new variables:
//
//     c >= 0,  |d| <= c,  |d| <= b*sigma*... (see svi_calibrator.hpp),
//     adash >= 0,  adash <= max(w_i)
//
// which is a convex region, so the constrained minimum is either interior (the
// unconstrained normal-equation solution) or on the boundary -- a small finite
// set of cases that `svi_inner_solve` enumerates.

/// The reduced linear coordinates for a fixed (m, sigma).
struct SviReduced {
    double adash = 0.0;  ///< = a
    double d = 0.0;      ///< = rho * b * sigma
    double c = 0.0;      ///< = b * sigma
};

/// Convert reduced coordinates back to raw SVI parameters.
[[nodiscard]] SviParams svi_from_reduced(const SviReduced& r, double m, double sigma,
                                         double years) noexcept;

/// Convert raw parameters into reduced coordinates.
[[nodiscard]] SviReduced svi_to_reduced(const SviParams& p) noexcept;

/// Evaluate the reduced form directly: w = adash + d*z + c*sqrt(z^2+1).
[[nodiscard]] VL_FORCE_INLINE double svi_reduced_variance(const SviReduced& r,
                                                          double z) noexcept {
    return std::fma(r.c, std::sqrt(std::fma(z, z, 1.0)), std::fma(r.d, z, r.adash));
}

}  // namespace vl
