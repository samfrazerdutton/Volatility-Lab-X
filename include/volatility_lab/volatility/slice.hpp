// SPDX-License-Identifier: MIT
#pragma once
/// \file slice.hpp
/// \brief Vocabulary for a single-expiry volatility smile, and the
///        no-arbitrage conditions every slice model is checked against.
///
/// ## Coordinates: total variance against log-moneyness
///
/// Every slice model in this library represents
///
///     w(k) = sigma_BS(k)^2 * T,      k = log(K / F)
///
/// not `sigma(K)`.  Four reasons, and they compound:
///
///  1. **The no-arbitrage conditions are statements about w.**  Butterfly
///     arbitrage is a condition on w, w' and w'' (the Durrleman function
///     below); calendar arbitrage is monotonicity of w in T.  Writing the
///     model in sigma means converting back and forth every time a constraint
///     is checked, and every conversion is a chance to drop a factor of 2T.
///
///  2. **w interpolates linearly in T in the absence of information.**  A
///     flat-forward-variance assumption between two expiries is linear in w,
///     not in sigma.  Interpolating sigma linearly in T introduces calendar
///     arbitrage for free.
///
///  3. **k is the natural abscissa.**  Strike is not: the same smile shape
///     sits at different strikes as the forward moves, and a scenario engine
///     that shocks spot would have to re-fit.  In k the shape is invariant and
///     a spot shock is a translation.
///
///  4. **w is what the pricer wants.**  `normalised_black` takes
///     s = sqrt(w) directly, so the slice hands the pricer its argument with
///     one sqrt and no multiplication by T.
///
/// Note the sign convention, which is a standing trap: the surface uses
/// k = log(K/F), increasing with strike, while the Black normalisation uses
/// x = log(F/K) = -k.  They never appear in the same expression without one
/// being negated, and they have different names everywhere.
///
/// ## No virtual functions
///
/// Slice models are plain parameter structs with free evaluation functions,
/// and the surface holds them in a `std::variant`.  There is no slice base
/// class and no vtable.
///
/// That is a deliberate choice with two distinct payoffs:
///
///  * **The hot loop stays inlinable.**  Dispatch happens once per slice per
///    batch (a `std::visit` outside the loop), not once per option, so the
///    innermost code is a direct call the compiler can inline and vectorise.
///    A virtual `total_variance(k)` called per option would defeat both.
///
///  * **A surface is cheap to copy.**  An SVI slice is five doubles; a whole
///    surface is a few hundred bytes.  The scenario engine perturbs and
///    evaluates surfaces millions of times, and that is only affordable if
///    cloning one is a memcpy rather than a graph of heap allocations.  This
///    is the single most important reason the scenario engine is built on the
///    parametric models rather than the interpolated grid.

#include <cmath>
#include <cstdint>

#include "volatility_lab/core/config.hpp"
#include "volatility_lab/core/types.hpp"

namespace vl {

/// Which slice model a surface is carrying.  Reported in diagnostics and
/// quality reports so that "the fit is poor" can be attributed.
enum class SliceKind : std::uint8_t {
    Flat = 0,  ///< constant total variance; the degenerate baseline
    Svi = 1,   ///< raw SVI, 5 parameters per expiry
    Ssvi = 2,  ///< SSVI, 2 global parameters + an ATM variance term structure
    Grid = 3   ///< interpolated total variance, cubic in k
};

[[nodiscard]] constexpr const char* to_string(SliceKind k) noexcept {
    switch (k) {
        case SliceKind::Flat: return "flat";
        case SliceKind::Svi:  return "svi";
        case SliceKind::Ssvi: return "ssvi";
        case SliceKind::Grid: return "grid";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// SliceJet
// ---------------------------------------------------------------------------

/// Total variance and its first two derivatives in log-moneyness at one point.
///
/// Returned as a bundle rather than through three separate calls because every
/// consumer needs all three together: the Durrleman condition, the local
/// volatility transform, the risk-neutral density, and the strike derivatives
/// of price (which need dw/dk to convert a vol move into a price move).
/// Computing them separately would repeat the shared square root three times.
///
/// "Jet" is the standard name for a truncated Taylor coefficient bundle; it is
/// used here rather than something like `SliceDerivatives` because the type
/// shows up in a lot of expressions and brevity at the call site matters.
struct SliceJet {
    double w = 0.0;    ///< total variance, sigma^2 * T
    double dw = 0.0;   ///< dw/dk
    double d2w = 0.0;  ///< d2w/dk2

    /// Implied volatility at this point, given the expiry.
    [[nodiscard]] double vol(double years) const noexcept {
        return (years > 0.0 && w > 0.0) ? std::sqrt(w / years) : 0.0;
    }

    /// Total volatility s = sigma*sqrt(T) = sqrt(w), the pricer's argument.
    [[nodiscard]] double total_vol() const noexcept {
        return (w > 0.0) ? std::sqrt(w) : 0.0;
    }
};

// ---------------------------------------------------------------------------
// The Durrleman condition
// ---------------------------------------------------------------------------

/// The Durrleman function g(k): the risk-neutral density is proportional to
/// g(k) times a positive factor, so **g(k) >= 0 for all k is exactly the
/// absence of butterfly arbitrage** in a slice.
///
///     g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2
///
/// This is not a heuristic smoothness check; it is the condition, necessary
/// and sufficient, for the implied density to be non-negative (Durrleman 2003;
/// Gatheral-Jacquier 2014, eq. 2.2).  The library checks it directly on the
/// fitted slice rather than relying on a parameter-space sufficient condition,
/// because a parameter-space condition tells you a slice is safe but not
/// *where* it is unsafe when it is not -- and section 11 of the brief requires
/// locating the violation, not merely detecting it.
///
/// Derivation sketch, since the formula is otherwise opaque: write the
/// undiscounted call in normalised Black form with s = sqrt(w(k)), differentiate
/// twice in strike, and collect.  The three terms are, in order, the
/// contribution of the level and slope, the "vega convexity" penalty that
/// punishes steep smiles, and the curvature that can rescue them.
///
/// Requires w > 0.  At w <= 0 the slice is already invalid and the caller
/// should have rejected it; this returns -inf there so that a violation is
/// unmistakable rather than a NaN that propagates.
[[nodiscard]] VL_FORCE_INLINE double durrleman_g(double k, const SliceJet& j) noexcept {
    if (!(j.w > 0.0)) return -std::numeric_limits<double>::infinity();
    const double term1 = 1.0 - k * j.dw / (2.0 * j.w);
    const double term2 = (j.dw * j.dw * 0.25) * (1.0 / j.w + 0.25);
    return term1 * term1 - term2 + 0.5 * j.d2w;
}

/// The risk-neutral probability density of log-moneyness implied by a slice.
///
///     p(k) = g(k) / sqrt(2 pi w) * exp(-d2(k)^2 / 2),   d2 = -k/sqrt(w) - sqrt(w)/2
///
/// Positive exactly when g(k) is, which is the point: the arbitrage check and
/// the density are the same statement, and having both in one place makes that
/// visible.  Used by the arbitrage diagnostics to report the magnitude of a
/// violation in units a reader can interpret (a negative probability) rather
/// than as an abstract g-value.
[[nodiscard]] double implied_density(double k, const SliceJet& j) noexcept;

// ---------------------------------------------------------------------------
// Admissibility of a slice at a point
// ---------------------------------------------------------------------------

/// What is wrong with a slice at a given log-moneyness, if anything.
/// Deliberately a bitmask: a badly fitted wing typically violates several
/// conditions at once, and reporting only the first found loses information.
enum class SliceDefect : std::uint8_t {
    None = 0,
    NonPositiveVariance = 1 << 0,  ///< w <= 0: not a volatility at all
    NonFinite = 1 << 1,            ///< NaN or inf in w, w' or w''
    ButterflyArbitrage = 1 << 2,   ///< g(k) < 0: negative implied density
    SlopeOutOfBounds = 1 << 3,     ///< |dw/dk| violates the Lee moment bounds
};

[[nodiscard]] constexpr SliceDefect operator|(SliceDefect a, SliceDefect b) noexcept {
    return static_cast<SliceDefect>(static_cast<std::uint8_t>(a) |
                                    static_cast<std::uint8_t>(b));
}
[[nodiscard]] constexpr bool has(SliceDefect set, SliceDefect flag) noexcept {
    return (static_cast<std::uint8_t>(set) & static_cast<std::uint8_t>(flag)) != 0;
}

/// Lee's moment formula bounds the asymptotic slope of total variance: a slice
/// with |dw/dk| > 2 in the wings implies a non-existent moment of the
/// underlying, and the corresponding call price violates monotonicity.  The
/// bound is asymptotic, so it is applied only where |k| is large enough for it
/// to bite; `kLeeSlopeBound` is the limit and `kLeeSlopeTestFrom` the
/// moneyness beyond which it is enforced.
inline constexpr double kLeeSlopeBound = 2.0;
inline constexpr double kLeeSlopeTestFrom = 0.5;

/// Check one point of a slice.  Cheap enough to call on a dense grid, which is
/// how the arbitrage engine locates violations.
[[nodiscard]] SliceDefect inspect_slice_point(double k, const SliceJet& j) noexcept;

}  // namespace vl
