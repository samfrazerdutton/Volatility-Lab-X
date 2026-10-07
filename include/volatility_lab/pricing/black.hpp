// SPDX-License-Identifier: MIT
#pragma once
/// \file black.hpp
/// \brief The Black-76 / Black-Scholes analytic engine.
///
/// ## Everything is built on one normalised function
///
/// The library does not implement "the Black-Scholes formula".  It implements
/// the *normalised* undiscounted price of an out-of-the-money option,
///
///     b(x, s) = C_und(F, K, s) / sqrt(F*K),   x = log(F/K),  s = sigma*sqrt(T)
///
/// and derives every price, every Greek, and the implied-volatility inversion
/// from it.  Three reasons, each of which costs real accuracy if ignored:
///
/// **1. Two arguments instead of five.**  (F, K, T, sigma, r) -> (x, s) is an
/// exact reduction, not an approximation: the undiscounted forward price
/// depends on the five inputs only through those two combinations.  So the
/// test sweep, the error analysis, and the SIMD kernel all work over a
/// 2-D domain rather than a 5-D one, and "validated over the whole domain"
/// becomes a statement one can actually make.
///
/// **2. The OTM side is the well-conditioned one.**  A deep-ITM call is
/// (F - K) + epsilon where epsilon is 1e-18 of the intrinsic.  Computing it as
/// `F*Phi(d1) - K*Phi(d2)` subtracts two numbers that agree to 18 digits and
/// returns noise for epsilon.  Computing the OTM put -- which *is* epsilon --
/// and adding the exactly-representable intrinsic gives full relative
/// accuracy on both.  Put-call parity is then satisfied by construction
/// rather than to within a tolerance.
///
/// **3. The erfcx form removes the small-variance catastrophe.**  See
/// `normalised_black` below.
///
/// ## Conventions
///
/// Forward measure throughout: all core functions take a *forward* and return
/// *undiscounted* values.  Discounting is a single multiplication applied at
/// the boundary (`black_price`), which keeps the discount factor out of every
/// inner expression and makes the rate-shock path in the scenario engine a
/// rescale rather than a reprice.  `T` is a year fraction; `sigma` is
/// annualised lognormal volatility; `r` and `q` are continuously compounded.

#include <cmath>
#include <limits>

#include "volatility_lab/core/config.hpp"
#include "volatility_lab/core/types.hpp"
#include "volatility_lab/math/special.hpp"

namespace vl {

// ===========================================================================
// Normalised Black
// ===========================================================================

/// Which branch `normalised_black` took.  Exposed for the branch-coverage
/// test and the benchmark histogram, not for callers to act on.
enum class BlackBranch : std::uint8_t {
    Atm = 0,           ///< x == 0: closed form, erf(s/(2 sqrt2))
    Series = 1,        ///< form (2) via the erfcx midpoint series (preferred)
    ErfcxDirect = 2,   ///< form (2) as a direct erfcx difference
    HighVariance = 5,  ///< form (3), the reflected, overflow-free form
    ZeroVariance = 3,  ///< s == 0: the option is worth its intrinsic
    Saturated = 4      ///< s so large the OTM value has reached its bound exactly
};

struct NormalisedBlack {
    double value;  ///< b(x, s)
    double dv_ds;  ///< db/ds -- the normalised vega
    BlackBranch branch;
};

/// b(x, s): normalised undiscounted price of the out-of-the-money option.
///
/// **Requires x <= 0.**  That is not a limitation: b(x, s) for x > 0 is
/// b(-x, s) by put-call symmetry (substituting h -> -h in the expression
/// below maps the normalised call onto the normalised put), so the caller
/// reduces with `x = -|log(F/K)|` and the function always prices the option
/// that is actually out of the money.
///
/// ### The two branches, and why each exists
///
/// Write h = x/s and t = s/2, so that d1 = h + t and d2 = h - t.  The
/// textbook form is
///
///     b = exp(x/2) Phi(h+t) - exp(-x/2) Phi(h-t)                       (1)
///
/// Using Phi(z) = erfc(-z/sqrt2)/2 and erfc(y) = erfcx(y) exp(-y^2), both
/// terms acquire the *same* Gaussian factor exp(-(h^2+t^2)/2) -- the cross
/// terms +-h*t cancel against exp(+-x/2) = exp(+-h*t) exactly -- giving, with
/// x <= 0 so that h = -|h|,
///
///     b = (1/2) G [ erfcx((|h|-t)/sqrt2) - erfcx((|h|+t)/sqrt2) ],
///         G = exp(-(h^2+t^2)/2)                                        (2)
///
/// Form (2) is not a refinement of (1); it is the difference between a usable
/// function and an unusable one.  In (1) both terms are O(1) while their
/// difference is the option value, which for a short-dated OTM option is
/// 1e-20 of either -- every significant digit is lost.  In (2) the tiny
/// Gaussian is an explicit factor and the bracket is a difference of two
/// O(1/|h|) quantities.  Measured at x = -0.2, s = 0.02, where the true value
/// is 1.49e-26: form (1) in double returns a relative error of 2.6e-12 after
/// the exponent is accounted for and is meaningless in absolute terms, while
/// form (2) is accurate to 1e-15.
///
/// **Branch A (t <= |h|, the normal case):** evaluate (2) directly.
///
/// **Branch B (t > |h|, very high total variance):** (2) would evaluate
/// erfcx at a negative argument, where erfcx(y) ~ 2 exp(y^2) overflows for
/// y < -26.6 (i.e. s > 75).  The Gaussian factor G underflows in exactly the
/// same proportion, so the product is finite but computes as 0 * inf.
/// Applying the reflection erfcx(-y) = 2 exp(y^2) - erfcx(y) to the first term
/// and simplifying the resulting exponent -- it collapses to exp(-|h|t) =
/// exp(x/2) exactly -- gives an equivalent form with both arguments positive:
///
///     b = exp(x/2) - (1/2) G [ erfcx((t-|h|)/sqrt2) + erfcx((t+|h|)/sqrt2) ]  (3)
///
/// which is overflow-free and exhibits the s -> infinity limit b -> exp(x/2)
/// as an explicit leading term rather than as a cancellation.
///
/// ### Accuracy, and why there is no series branch
///
/// The bracket in (2) loses about log10(|x|/s^2) decimal digits of relative
/// accuracy: the two erfcx arguments differ by t*sqrt2, and erfcx ~ 1/(z
/// sqrt(pi)) there, so the relative size of the difference is ~2t/|h| = s^2/|x|.
///
/// An earlier version of this file carried a Taylor-series branch for small
/// s on the assumption that this loss was unbounded.  It is not.  Because
/// b ~ exp(-x^2/(2 s^2)), the regions of large |x|/s^2 are precisely the
/// regions where b has already underflowed the double range, and the two
/// constraints are not independent.  Tabulating the loss *subject to b being
/// representable* bounds it at about 6 digits (worst case: near-ATM with
/// s ~ 1e-4), leaving >= 10 significant digits everywhere the answer exists.
/// The conditioning table is in docs/numerical-validation.md and is
/// regenerated by `volatility-lab validate --black`.
///
/// Ten digits on the price is also better than it sounds for the inversion
/// that consumes it: d(ln b)/d(ln s) = (x/s)^2, so a relative price error is
/// *divided* by (x/s)^2 ~ 100-1400 in the implied volatility.  The
/// ill-conditioned corner of the pricer is the well-conditioned corner of the
/// solver.
///
/// The series branch was therefore deleted rather than fixed.  See
/// docs/design-decisions.md (D-04) for the full account, including the
/// derivation error that made the first attempt wrong.
///
/// ### ATM
///
/// x == 0 gives h = 0 and (1) collapses to b = 2 Phi(t) - 1 = erf(s/(2 sqrt2)),
/// a single libm call with no cancellation anywhere.  Worth a branch because
/// ATM is the most-queried point on any surface.
[[nodiscard]] NormalisedBlack normalised_black(double x, double s) noexcept;

/// `normalised_black` without the vega, for the pricing-only hot path.
[[nodiscard]] double normalised_black_value(double x, double s) noexcept;

/// db/ds: the normalised vega, (1/sqrt(2pi)) exp(-(h^2+t^2)/2).
///
/// Note what this expression is *not*: it is not phi(d1) and it is not
/// phi(d2), though it equals both up to the exp(+-x/2) factors that the
/// normalisation removed.  Deriving it is two lines (see the header comment
/// for `normalised_black`) and the result is manifestly positive, which is
/// why the implied-volatility iteration never has to check for a sign flip.
[[nodiscard]] double normalised_black_vega(double x, double s) noexcept;

/// d2b/ds2.  Equal to vega * (x^2/s^3 - s/4), hence **exactly one** inflection
/// point, at s = sqrt(2|x|), convex below and concave above.  The implied
/// volatility solver's convergence proof rests entirely on this fact.
[[nodiscard]] double normalised_black_d2(double x, double s) noexcept;

/// The inflection point s_c = sqrt(2|x|) of b(x, .), and the value there.
/// Returned together because the implied-vol solver needs both to pick its
/// branch and they share the work.
struct BlackInflection {
    double s_c;
    double b_c;
};
[[nodiscard]] BlackInflection normalised_black_inflection(double x) noexcept;

// ===========================================================================
// Prices
// ===========================================================================

/// Undiscounted Black-76 price in forward measure.
///
/// Computed as `sqrt(F*K) * b(-|x|, s) + intrinsic`, so the OTM value carries
/// full relative accuracy and the ITM value carries full relative accuracy on
/// its (exactly representable) intrinsic.  Put-call parity holds to the last
/// bit by construction: both sides share the same b.
[[nodiscard]] double black_undiscounted(double forward, double strike, double vol,
                                        double years, OptionType type) noexcept;

/// Discounted price.  `discount` is the DF to the payment date.
[[nodiscard]] VL_FORCE_INLINE double black_price(double forward, double strike, double vol,
                                                 double years, double discount,
                                                 OptionType type) noexcept {
    return discount * black_undiscounted(forward, strike, vol, years, type);
}

/// Spot-parameterised Black-Scholes, for callers that think in (S, r, q).
/// Forward and discount are formed once and handed to the forward-measure
/// core: F = S exp((r-q)T), DF = exp(-rT).
[[nodiscard]] double black_scholes_price(double spot, double strike, double vol, double years,
                                         double rate, double carry, OptionType type) noexcept;

/// Intrinsic value in forward measure, undiscounted: max(omega*(F-K), 0).
[[nodiscard]] VL_FORCE_INLINE double forward_intrinsic(double forward, double strike,
                                                       OptionType type) noexcept {
    const double d = payoff_sign(type) * (forward - strike);
    return d > 0.0 ? d : 0.0;
}

// ===========================================================================
// No-arbitrage bounds
// ===========================================================================

/// The admissible range of an undiscounted forward-measure option price.
///
/// A price outside this interval is not "a bad quote to be fitted with a large
/// residual" -- it corresponds to no volatility whatsoever, and feeding it to
/// an inversion is asking for the root of a function that has none.  The
/// normalisation layer rejects such quotes with `PriceBelowIntrinsic` /
/// `PriceAboveForwardBound` before the solver ever sees them.
struct PriceBounds {
    double lower;  ///< intrinsic (attained as sigma -> 0)
    double upper;  ///< F for a call, K for a put (attained as sigma -> inf)
};

[[nodiscard]] VL_FORCE_INLINE PriceBounds forward_price_bounds(double forward, double strike,
                                                               OptionType type) noexcept {
    return {forward_intrinsic(forward, strike, type),
            type == OptionType::Call ? forward : strike};
}

// ===========================================================================
// Reductions used across the library
// ===========================================================================

/// Log-moneyness log(K/F) -- the surface's natural abscissa.
///
/// Note the orientation: surfaces are parameterised in k = log(K/F) (so that
/// k increases with strike, as a smile is conventionally drawn), whereas the
/// Black normalisation uses x = log(F/K) = -k.  The two differ by a sign and
/// confusing them inverts the skew, so they have different names everywhere
/// and never appear in the same expression without one of them negated.
[[nodiscard]] VL_FORCE_INLINE double log_moneyness(double forward, double strike) noexcept {
    return std::log(strike / forward);
}

/// Total variance w = sigma^2 * T.  The surface stores and interpolates this
/// rather than sigma, because calendar-arbitrage and the no-arbitrage
/// conditions are statements about w, and because w is the quantity that
/// interpolates linearly in the absence of information.
[[nodiscard]] VL_FORCE_INLINE double total_variance(double vol, double years) noexcept {
    return vol * vol * years;
}

/// sigma from total variance, guarding T -> 0.
[[nodiscard]] VL_FORCE_INLINE double vol_from_total_variance(double w, double years) noexcept {
    return (years > 0.0 && w > 0.0) ? std::sqrt(w / years) : 0.0;
}

}  // namespace vl
