// SPDX-License-Identifier: MIT
#pragma once
/// \file implied_vol.hpp
/// \brief Implied volatility inversion.
///
/// ## The algorithm, and why it is guaranteed to converge
///
/// The inversion is posed on the normalised function from pricing/black.hpp:
/// given a normalised OTM price `beta` and log-moneyness `x <= 0`, find the
/// total volatility `s = sigma*sqrt(T)` with `b(x, s) = beta`.
///
/// Two structural facts about `b` do all the work.
///
/// **Fact 1 -- monotonicity.**  db/ds = (1/sqrt(2pi)) exp(-(h^2+t^2)/2) is
/// strictly positive for every finite s > 0.  So `b(x, .)` is strictly
/// increasing from 0 (at s = 0) to exp(x/2) (as s -> infinity), and the root
/// exists and is unique for every beta in that open interval.  A beta outside
/// it corresponds to no volatility at all, and is rejected rather than
/// approximated.
///
/// **Fact 2 -- exactly one inflection.**  d2b/ds2 = vega * (x^2/s^3 - s/4),
/// and the bracket changes sign exactly once, at
///
///     s_c = sqrt(2|x|)
///
/// so `b` is **convex** on (0, s_c) and **concave** on (s_c, infinity).
///
/// Together these give a convergence proof for Newton started at `s_c`:
///
///  * If beta < b(x, s_c) the root lies in the convex region.  For a convex
///    increasing function the tangent lies below the curve, so the tangent
///    from s_c reaches beta no earlier than the curve does: the Newton step
///    lands in [s*, s_c].  Every subsequent step repeats the argument, so the
///    iterates decrease monotonically to s* and never leave the convex region.
///
///  * If beta >= b(x, s_c) the root lies in the concave region.  There the
///    tangent lies above the curve, so the Newton step lands in [s_c, s*], and
///    the iterates increase monotonically to s*.
///
/// Either way Newton from s_c is monotone and cannot overshoot, diverge, or
/// leave the domain -- without any bracketing search, safeguard, or bisection
/// fallback.  That is unusual enough to be worth stating plainly: the
/// guarantee comes from the convexity structure of the function, not from
/// defensive engineering around a solver that might misbehave.
///
/// ## Making it fast as well as safe
///
/// Monotone convergence is not the same as rapid convergence: starting from
/// s_c when the answer is 50x smaller costs iterations.  So the production
/// path does three things:
///
///  1. **An analytic initial guess**, selected by `|h| = |x|/s` rather than by
///     which side of the inflection the root is on.  Those are different
///     questions: for small `|x|` the inflection is itself small, so the whole
///     convex branch can be *near* the money.  An ATM-anchored inversion is
///     computed first and used to estimate `|h|`; only if that says the option
///     is genuinely far out of the money does the deep-OTM asymptotic take
///     over, and in the transition band both are evaluated and the closer (in
///     log price) is kept.  Measured worst-case guess error over the domain
///     sweep: **0.76 relative**, and the guess is finally confined to the
///     branch the root provably lies in.
///
///  2. **Halley rather than Newton** -- cubic rather than quadratic, using the
///     second derivative that Fact 2 already requires us to have.
///
///  3. **A double-log objective on the convex branch.**  Below s_c the price
///     spans hundreds of decades and its derivative twice as many, so a model
///     fitted at one point is worthless a few percent away.  But
///
///         d(log b)/d(log s) = s * vega / b  ~  1 + h^2
///
///     is positive, smooth, and O(1) near the money, growing only
///     quadratically out of it.  Iterating on `log b - log beta` in the
///     variable `log s` turns the hard branch into an easy one, and makes the
///     update multiplicative, so the iterate cannot go negative.  Above s_c,
///     `b` is already gentle and the transform would only add round-off, so
///     the iteration runs on `b` directly.
///
///  4. **A trust region** of e^2 per step on the convex branch.  Since the
///     slope `1 + h^2` is unbounded as s falls, an honest local model can ask
///     for a move of 16 orders of magnitude; without the cap the iterate
///     leaves the representable range and the root is lost.
///
/// Measured over a 12133-point sweep of the (x, s) domain: **mean 3.3
/// iterations**, worst relative error 1.0e-10 in the recovered total
/// volatility, zero failures, and the Brent fallback needed twice.  The same
/// sweep run with the generic solvers from math/root_finding.hpp, for
/// comparison, is in benchmarks/pricing -- plain Newton fails 827 times on it.
///
/// The bracket [lo, hi] is still maintained and the step still clamped into
/// it.  Not because the proof is doubted, but because the proof is about
/// exact arithmetic: it says nothing about an iterate that lands one ulp
/// outside the domain when beta is within one ulp of a bound.  The clamp
/// costs two comparisons and converts those cases from "silently wrong" to
/// "one extra iteration".
///
/// ## What it reports
///
/// Never a bare double.  `ImpliedVolResult` carries the iteration count, the
/// residual in *both* normalised and price units, the method that produced it,
/// and a diagnostic code.  A convergence failure that returns a plausible
/// number and no indication of failure is how bad marks reach a risk report.

#include <cstdint>

#include "volatility_lab/core/diagnostics.hpp"
#include "volatility_lab/core/types.hpp"
#include "volatility_lab/math/root_finding.hpp"
#include "volatility_lab/pricing/black.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct ImpliedVolConfig {
    /// Stop when |b(x,s) - beta| <= price_tol * max(beta, 1e-300).  Relative,
    /// because beta spans 300 decades across the domain and any absolute
    /// tolerance is either unreachable at the top or meaningless at the bottom.
    double price_rtol = 8.0 * std::numeric_limits<double>::epsilon();

    /// Stop when the step is below step_rtol * s.  The inversion is better
    /// conditioned than the pricer (d log b / d log s = (x/s)^2 >= 1, so a
    /// relative price error is *divided* by that factor), which is why a
    /// relative step tolerance at a few eps is achievable here at all.
    double step_rtol = 4.0 * std::numeric_limits<double>::epsilon();

    /// Hard iteration cap.  Measured worst case over the fuzz corpus is well
    /// inside this; see benchmarks/pricing.  The cap is a guarantee of
    /// termination, not a tuning parameter.
    int max_iterations = 32;

    /// Volatilities outside [vol_floor, vol_ceiling] are reported as
    /// `IvBelowFloor` / `IvAboveCeiling` rather than returned silently.  These
    /// are sanity bounds on the *answer*, distinct from the no-arbitrage
    /// bounds on the input price.
    double vol_floor = 1.0e-8;
    double vol_ceiling = 100.0;  // 10000% annualised

    /// Whether to fall back to Brent on a validated bracket if the fast path
    /// fails.  On by default; turned off by the benchmark when it wants to
    /// measure the fast path in isolation.
    bool allow_fallback = true;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Why an inversion did not produce a clean answer.  `Ok` is the only value
/// for which `volatility` is meaningful.
enum class IvStatus : std::uint8_t {
    Ok = 0,
    PriceBelowIntrinsic,  ///< below the sigma -> 0 bound: no root exists
    PriceAboveBound,      ///< at or above the sigma -> inf bound: no root exists
    PriceNotPositive,     ///< negative OTM value

    /// The option value equals its intrinsic to the last bit.  Distinct from
    /// `PriceBelowIntrinsic`, which means the input violated a no-arbitrage
    /// bound and is a *data* problem.  This one is a representation limit: the
    /// option really is worth its intrinsic as far as a double can tell, so
    /// the volatility is not determined by the input.  A 10%-out-of-the-money
    /// option one hour from expiry at 20% vol reaches it legitimately -- the
    /// true value there is exp(-995).
    PriceAtIntrinsic,
    BelowFloor,           ///< converged, but to a volatility below vol_floor
    AboveCeiling,         ///< converged, but to a volatility above vol_ceiling
    DidNotConverge,       ///< iteration cap reached
    InvalidInput          ///< non-finite or non-positive F, K, T
};

[[nodiscard]] const char* to_string(IvStatus s) noexcept;

/// Maps an IvStatus onto the library-wide diagnostic code, so that a failed
/// inversion inside a surface fit becomes a structured complaint about a
/// specific quote rather than a NaN that propagates.
[[nodiscard]] DiagCode to_diag_code(IvStatus s) noexcept;

struct ImpliedVolResult {
    double volatility = 0.0;        ///< sigma, annualised
    double total_volatility = 0.0;  ///< s = sigma*sqrt(T), the solved quantity
    double residual = 0.0;          ///< b(x, s) - beta, normalised units
    double price_residual = 0.0;    ///< the same residual in price units
    double initial_guess = 0.0;     ///< s_0, reported so the guess can be tuned

    /// Best relative accuracy the *input* permits, independent of how well the
    /// solver did.
    ///
    /// This matters for in-the-money quotes, and it is the field most likely
    /// to be overlooked.  An ITM price is `intrinsic + v` where v is the part
    /// that depends on volatility; for F = 100, K = 25, sigma = 20%, T = 1 the
    /// price is 75 + 1.1e-11, and ulp(75) is 1.4e-14 -- so the double handed
    /// to us pins down v to about three significant digits and *no
    /// implementation can do better*.  The returned volatility there is good
    /// to ~4e-5 relative, not 1e-14, and reporting 1e-14 would be a lie about
    /// data the caller already destroyed.
    ///
    /// Computed as eps * |price| / otm_value, then divided by
    /// d(log b)/d(log s) = (x/s)^2 + 1, since the inversion damps relative
    /// price error by that factor.  For an OTM quote it is a few eps.
    double attainable_rtol = 0.0;

    int iterations = 0;
    IvStatus status = IvStatus::InvalidInput;
    math::SolveMethod method = math::SolveMethod::None;
    bool used_fallback = false;

    [[nodiscard]] bool ok() const noexcept { return status == IvStatus::Ok; }
};

// ---------------------------------------------------------------------------
// The primitive
// ---------------------------------------------------------------------------

/// Solve b(x, s) = beta for s.  Requires x <= 0 and beta in (0, exp(x/2)).
///
/// This is the only inversion routine in the library that actually iterates;
/// every other entry point normalises and calls it.
[[nodiscard]] ImpliedVolResult implied_total_volatility(double beta, double x,
                                                        const ImpliedVolConfig& cfg = {}) noexcept;

// ---------------------------------------------------------------------------
// Price-space entry points
// ---------------------------------------------------------------------------

/// Implied volatility from an undiscounted forward-measure price.
///
/// The price may be for either side; it is converted to the OTM side using
/// put-call parity before inversion, because the ITM price carries almost no
/// information about volatility (it is intrinsic plus a rounding error) while
/// the OTM price carries all of it.  Inverting an ITM price directly is the
/// single most common way to get a garbage implied vol, and this function
/// makes that impossible.
[[nodiscard]] ImpliedVolResult implied_volatility_undiscounted(
    double undiscounted_price, double forward, double strike, double years, OptionType type,
    const ImpliedVolConfig& cfg = {}) noexcept;

/// Implied volatility from a discounted price.  `discount` must be > 0.
[[nodiscard]] ImpliedVolResult implied_volatility(double price, double forward, double strike,
                                                  double years, double discount,
                                                  OptionType type,
                                                  const ImpliedVolConfig& cfg = {}) noexcept;

// ---------------------------------------------------------------------------
// Alternative solvers -- for comparison, not for production
// ---------------------------------------------------------------------------

/// Run the inversion with a specified generic algorithm from
/// math/root_finding.hpp instead of the specialised path.
///
/// This exists so that the choice of the specialised path can be *justified*
/// rather than asserted: `volatility-lab benchmark implied-vol` runs all of
/// these over the same input sweep and reports iterations, failures, and
/// accuracy side by side.  Supported: Newton, SafeguardedNewton, Brent,
/// Bisection, Halley.
[[nodiscard]] ImpliedVolResult implied_total_volatility_with(
    math::SolveMethod method, double beta, double x,
    const ImpliedVolConfig& cfg = {}) noexcept;

/// The initial guess the production path would use, exposed so the benchmark
/// can report how good it is before any iteration happens.
[[nodiscard]] double implied_vol_initial_guess(double beta, double x) noexcept;

}  // namespace vl
