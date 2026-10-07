// SPDX-License-Identifier: MIT
#pragma once
/// \file svi_calibrator.hpp
/// \brief Quasi-explicit SVI slice calibration.
///
/// ## The problem with fitting SVI directly
///
/// Raw SVI has five parameters and they are badly coupled: `a` trades off
/// against `b*sigma` almost one-for-one near the money, and `m` trades off
/// against `rho` in the wings.  A five-dimensional nonlinear least squares on
/// that surface has genuine local minima, and a Levenberg-Marquardt run from a
/// single starting point lands in one of them often enough to matter -- not
/// occasionally, but on a noticeable fraction of ordinary slices.  The usual
/// response is multi-start, which turns a reliability problem into a cost
/// problem without fully solving either.
///
/// ## The reduction
///
/// Zeliade Systems' observation (*Quasi-Explicit Calibration of Gatheral's SVI
/// Model*, 2009) is that the problem is only nonlinear in two of the five
/// parameters.  Fix `(m, sigma)` and substitute
///
///     z = (k - m) / sigma
///
/// Then
///
///     w = a + b*sigma*( rho*z + sqrt(z^2 + 1) )
///
/// and with the change of variables
///
///     adash = a,    d = rho*b*sigma,    c = b*sigma
///
/// this becomes
///
///     w(z) = adash + d*z + c*sqrt(z^2 + 1)
///
/// which is **linear in (adash, d, c)**.  So for any fixed `(m, sigma)` the
/// optimal three remaining parameters are the solution of a three-variable
/// linear least squares -- available in closed form, with no iteration, no
/// initial guess, and no possibility of a local minimum.
///
/// The five-parameter global search collapses to a **two-parameter outer
/// search with an exact inner solve**.  And a two-dimensional search can be
/// done *globally*: a coarse grid over `(m, sigma)` followed by a local
/// refinement covers the whole domain, so the local-minimum failure mode
/// largely disappears rather than being mitigated.
///
/// ## The inner problem is constrained, and the constraints are a box
///
/// The admissibility conditions look like a polytope in `(adash, d, c)`:
///
///     c >= 0                      (b >= 0)
///     |d| <= c                    (|rho| <= 1)
///     |d| <= 4*sigma - c          (Lee's wing bound: dw/dk <= 2 asymptotically)
///     0 <= adash <= max(w_i)      (non-negative variance; no level above data)
///
/// but one substitution turns them into a box.  Put
///
///     u = c + d,   v = c - d
///
/// Then `|d| <= c` is exactly `u >= 0 and v >= 0`, and `|d| <= 4 sigma - c` is
/// exactly `max(u, v) <= 4 sigma` -- both by cases on the sign of `u - v`.  So
/// in `(adash, u, v)` the whole admissible set is
///
///     0 <= adash <= max(w_i),   0 <= u <= 4 sigma,   0 <= v <= 4 sigma
///
/// and the inner problem is a three-variable least squares over a *box*.  With
/// three variables there are 27 active sets, so
/// `math::solve_boxed_least_squares` enumerates them and returns the **exact**
/// constrained optimum -- no iteration, no convergence criterion, no initial
/// guess.
///
/// That substitution is worth the two lines of algebra.  Handling the coupling
/// approximately instead -- bound `|d|` by the `c` from an unconstrained pass,
/// then re-solve -- was the first implementation, and it left the objective up
/// to 3% high when the constraints bound and was 12x worse on the high-vol
/// regime with the wing bound disabled, because the first pass was free to pick
/// an enormous `c` that the second then had to clamp.
///
/// ### One deliberate conservatism
///
/// `adash >= 0` is *sufficient* for non-negative total variance but not
/// necessary: the exact condition is `a + b*sigma*sqrt(1-rho^2) >= 0`, which
/// permits a slightly negative `a`.  The exact condition couples `adash` to
/// both `u` and `v`, so imposing it would destroy the box structure and with it
/// the closed-form inner solve.
///
/// The cost of that choice is measured rather than assumed: against an
/// unconstrained-`a` five-parameter fit it is **up to 3% in the objective and
/// 7% in RMS volatility error** across the synthetic regimes -- both far below
/// the quote noise the fit is trying to see through.  In exchange the fitted
/// wings are consistently gentler (asymptotic slope 0.27 against 0.85 on the
/// high-vol regime), which is the more useful property for a surface that will
/// be extrapolated.
///
/// ## What this buys, measured
///
/// Numbers from `benchmarks/calibration` over the six synthetic regimes, with
/// the direct alternative given a 16-point deterministic multi-start so the
/// comparison is not a straw man:
///
///     regime      quasi-explicit   16-start LM   speedup   RMS vol (qe / LM)
///     normal          262 us          589 us      2.3x     2.88e-3 / 2.88e-3
///     high-vol        264 us          710 us      2.7x     6.33e-3 / 6.33e-3
///     crash           216 us          358 us      1.7x     1.59e-2 / 1.58e-2
///     vol-crush       201 us          576 us      2.9x     3.31e-3 / 3.09e-3
///     earnings        233 us          386 us      1.7x     9.15e-3 / 9.15e-3
///
/// So: **equal accuracy at 1.7-2.9x the speed, with no starting guess**.
///
/// ### And the local minima, honestly
///
/// The motivating claim for the reduction is that the five-parameter problem
/// has local minima.  It does -- but far less often than the folklore suggests,
/// and only on hard data.  Running the direct fit from each of the 16 starts
/// *separately* and comparing the spread of final objectives:
///
///     regime      worst/best objective   starts landing >10% high
///     normal             4.3x                   1 / 160
///     high-vol           1.0x                   0 / 128
///     crash            107.4x                   4 / 128
///     vol-crush          1.1x                   0 / 112
///     earnings           1.1x                   0 / 128
///
/// On ordinary surfaces almost any start works.  On the crash regime -- steep
/// skew, wide spreads, thin wings -- the objective varies by a factor of 107
/// between starting points, and a single-start fit lands materially wrong
/// about 3% of the time.  That is the case the reduction actually protects
/// against, and it is the case that matters, because a crash is when the risk
/// numbers are being looked at.
///
/// The honest summary is therefore not "multi-start is unreliable" but: the
/// reduction is faster, needs no guess, and removes a failure mode that is
/// rare on easy data and real on hard data.

#include <cstdint>
#include <span>
#include <vector>

#include "volatility_lab/calibration/optimizer.hpp"
#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/core/diagnostics.hpp"
#include "volatility_lab/options/quote.hpp"
#include "volatility_lab/volatility/svi.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SviCalibratorConfig {
    /// Resolution of the coarse outer grid over (m, sigma).
    ///
    /// The whole advantage of the reduction is that a 2-D domain can be
    /// searched globally, and that only works if the grid is actually dense
    /// enough to find the basin.  11x11 = 121 inner solves, each a closed-form
    /// 3x3 -- cheaper than a single Levenberg-Marquardt iteration on the full
    /// five parameters, and it removes the dependence on a starting guess.
    int grid_m = 11;
    int grid_sigma = 11;

    /// Outer search bounds.  `m` is in log-moneyness units and is bracketed by
    /// the quoted strike range, scaled out a little because the minimum of the
    /// smile can legitimately sit outside the quotes.  `sigma` is bounded
    /// below away from zero (sigma = 0 is a kink, not a smile) and above by
    /// the strike span, since a sigma much wider than the data cannot be
    /// identified.
    double m_range_scale = 1.5;
    double sigma_min = 1.0e-3;
    double sigma_max_scale = 2.0;

    /// Refine the best grid point with Levenberg-Marquardt on (m, sigma)
    /// alone, with the inner solve nested inside.  Cheap and worth it: the
    /// grid locates the basin, the refinement finds the bottom.
    bool refine = true;
    OptimizerSettings refine_settings{};

    ResidualKind residual = ResidualKind::TotalVariance;

    /// Enforce the Lee wing bound in the inner solve.  On by default; turning
    /// it off is useful for showing in a test what the constraint buys.
    bool enforce_wing_bound = true;

    /// Check the fitted slice for butterfly arbitrage and report it.  The fit
    /// is *not* rejected on failure -- a slice that fits the data and violates
    /// the condition is informative, and the caller may prefer SSVI -- but it
    /// is always reported.
    bool check_butterfly = true;

    /// Minimum usable quotes to attempt a five-parameter fit.  Below this the
    /// calibrator degrades to fewer parameters rather than fitting noise.
    std::size_t min_quotes = 5;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

enum class SviFitStatus : std::uint8_t {
    Ok = 0,
    TooFewQuotes,
    DegradedToFlat,     ///< not enough quotes for a shape; fitted a level only
    InnerSolveFailed,   ///< every active set in the inner problem was degenerate
    NotAdmissible,      ///< the best fit found is not an admissible slice
    ButterflyViolation  ///< fitted, but the slice has negative density somewhere
};

[[nodiscard]] const char* to_string(SviFitStatus s) noexcept;

struct SviFitResult {
    SviParams params;
    SviFitStatus status = SviFitStatus::TooFewQuotes;

    double objective = 0.0;       ///< weighted sum of squared residuals
    double rms_vol_error = 0.0;   ///< in volatility points, the readable figure
    double max_vol_error = 0.0;
    double rms_total_variance_error = 0.0;

    std::size_t quotes_used = 0;
    std::size_t quotes_available = 0;

    /// Cost accounting, for the benchmark comparison against direct LM.
    int inner_solves = 0;       ///< closed-form 3x3 solves (grid + refinement)
    int outer_iterations = 0;   ///< LM iterations on (m, sigma)
    int residual_evaluations = 0;

    /// Objective at the best grid point, before refinement.  The gap between
    /// this and `objective` is how much the refinement was worth, which is the
    /// honest way to justify keeping it.
    double grid_objective = 0.0;

    /// Spread of the objective across the outer grid.  A small spread means
    /// the slice is insensitive to (m, sigma) and the fitted values of those
    /// two are not identified -- which the objective value alone cannot show.
    double grid_objective_range = 0.0;

    ButterflyCheck butterfly;
    DiagnosticSink diagnostics;

    [[nodiscard]] bool ok() const noexcept {
        return status == SviFitStatus::Ok || status == SviFitStatus::ButterflyViolation;
    }
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Fit one slice.  `quotes` must all share an expiry and must already be
/// normalised and weighted.
[[nodiscard]] SviFitResult calibrate_svi_slice(std::span<const OptionQuote> quotes,
                                               const SviCalibratorConfig& cfg = {});

/// The inner solve, exposed because it is the interesting part.
///
/// Given `(m, sigma)` and the quotes, returns the exact optimal
/// `(adash, d, c)` subject to the admissibility polytope, along with the
/// objective there.  No iteration.
struct SviInnerSolve {
    SviReduced reduced;
    double objective = 0.0;
    bool feasible = false;
    std::size_t active_constraints = 0;
};

[[nodiscard]] SviInnerSolve svi_inner_solve(std::span<const OptionQuote> quotes, double m,
                                            double sigma,
                                            const SviCalibratorConfig& cfg = {});

/// Direct five-parameter Levenberg-Marquardt fit, for comparison.
///
/// This exists so the quasi-explicit path can be *justified* rather than
/// asserted: `benchmarks/calibration` and
/// `tests/calibration/svi_calibrator.cpp` run both over the same data and
/// compare fit quality, cost, and the rate at which the direct fit lands in a
/// worse local minimum.
[[nodiscard]] SviFitResult calibrate_svi_slice_direct(
    std::span<const OptionQuote> quotes, std::span<const std::vector<double>> starts,
    const SviCalibratorConfig& cfg = {});

/// A small set of starting points spanning the plausible parameter space, for
/// the direct fit.  Deterministic.
[[nodiscard]] std::vector<std::vector<double>> svi_default_starts(
    std::span<const OptionQuote> quotes);

}  // namespace vl
