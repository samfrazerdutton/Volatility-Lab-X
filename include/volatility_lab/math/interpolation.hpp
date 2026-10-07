// SPDX-License-Identifier: MIT
#pragma once
/// \file interpolation.hpp
/// \brief One-dimensional interpolators, with the choice between them driven
///        by what each one *preserves* rather than by smoothness alone.
///
/// ## The choice that matters
///
/// For a volatility surface, interpolation is not a cosmetic question. Three
/// properties are at stake and no scheme has all of them:
///
///  * **C2 continuity.**  Needed because gamma is a second derivative of price
///    in strike, and the second derivative of total variance enters it.  A
///    C1 scheme gives a gamma that jumps at every knot, which looks like a
///    risk discontinuity and is purely an artefact of the interpolator.
///
///  * **Monotonicity preservation.**  Needed along the *time* axis, where
///    total variance must not decrease (calendar arbitrage), and in the
///    discount/forward curves.  A natural cubic spline through monotone data
///    is not monotone -- it overshoots between knots -- so using one for the
///    term structure manufactures arbitrage out of clean data.
///
///  * **Locality.**  A natural cubic spline is global: moving one knot changes
///    the curve everywhere, which makes incremental surface updates (section
///    31) impossible to localise.
///
/// So the library provides three and uses each where its guarantee is the one
/// that matters:
///
///  * `CubicSpline`      -- C2, global, not monotone.  Used *across strike*
///    within a slice, where smoothness buys a clean gamma and monotonicity is
///    not wanted anyway (total variance is U-shaped in k).
///
///  * `MonotoneCubic`    -- C1, local, monotone (Fritsch-Carlson limiter).
///    Used *across time* for the total-variance term structure, where
///    preserving monotonicity is a no-arbitrage requirement and C1 is
///    sufficient because nothing differentiates the surface twice in T.
///
///  * `LinearInterp`     -- C0, local, monotone.  The baseline, and the
///    correct choice when there are only two knots or when the caller wants no
///    extrapolation structure at all.
///
/// The reasoning is recorded in docs/design-decisions.md (D-08).
///
/// ## Extrapolation
///
/// Every interpolator here extrapolates by *holding the end slope*, never by
/// continuing the end polynomial.  A cubic continued past its last knot
/// diverges cubically and will produce negative total variance a short
/// distance outside the quoted strikes; a held slope grows linearly, which is
/// what Lee's moment bound requires of total variance anyway.  This is the
/// single most common source of absurd extrapolated volatilities, and it is a
/// one-line choice.

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "volatility_lab/core/config.hpp"

namespace vl::math {

// ---------------------------------------------------------------------------
// Knot location
// ---------------------------------------------------------------------------

/// Index of the interval containing `x`, clamped to [0, n-2].
///
/// Binary search rather than a linear scan: slices routinely have 60+ strikes
/// and the scenario engine queries them millions of times.  For the *sorted*
/// batch case the surface uses a marching index instead, which is O(1)
/// amortised -- see `GridSlice::total_variance_batch`.
[[nodiscard]] std::size_t locate_interval(std::span<const double> xs, double x) noexcept;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

class LinearInterp {
  public:
    LinearInterp() = default;
    LinearInterp(std::vector<double> xs, std::vector<double> ys);

    [[nodiscard]] double operator()(double x) const noexcept;
    /// Value and first derivative.  The second derivative is zero inside an
    /// interval and undefined at knots; it is reported as zero.
    [[nodiscard]] double derivative(double x) const noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return xs_.size(); }
    [[nodiscard]] std::span<const double> nodes() const noexcept { return xs_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return ys_; }
    [[nodiscard]] bool empty() const noexcept { return xs_.empty(); }

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
};

// ---------------------------------------------------------------------------
// Natural cubic spline
// ---------------------------------------------------------------------------

/// C2 cubic spline with natural end conditions (zero second derivative at the
/// ends), evaluated from precomputed second derivatives.
///
/// The tridiagonal system is solved once at construction by the Thomas
/// algorithm -- O(n), stable here without pivoting because the natural-spline
/// matrix is symmetric positive definite and strictly diagonally dominant.
/// That is worth stating because Thomas *is* unstable in general; it is the
/// diagonal dominance of this particular matrix that makes it safe.
class CubicSpline {
  public:
    CubicSpline() = default;
    CubicSpline(std::vector<double> xs, std::vector<double> ys);

    [[nodiscard]] double operator()(double x) const noexcept;

    /// Value and the first two derivatives, in one pass.  The slice models
    /// need all three (see SliceJet), and evaluating the cubic three times
    /// would repeat the interval search and the Horner chain.
    struct Jet {
        double y = 0.0;
        double dy = 0.0;
        double d2y = 0.0;
    };
    [[nodiscard]] Jet jet(double x) const noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return xs_.size(); }
    [[nodiscard]] std::span<const double> nodes() const noexcept { return xs_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return ys_; }
    [[nodiscard]] bool empty() const noexcept { return xs_.empty(); }

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<double> y2_;  ///< second derivatives at the knots
};

// ---------------------------------------------------------------------------
// Monotone cubic (Fritsch-Carlson)
// ---------------------------------------------------------------------------

/// C1 Hermite cubic whose knot slopes are limited so that the result is
/// monotone wherever the data is.
///
/// The limiter is Fritsch-Carlson (1980): compute the secant slopes, take an
/// initial estimate of the knot derivatives, then shrink any derivative that
/// would allow an overshoot into the region
///
///     alpha^2 + beta^2 <= 9,   alpha = d_i/S_i,  beta = d_{i+1}/S_i
///
/// which is the sufficient condition for a Hermite cubic to be monotone on
/// that interval.  The standard implementation projects onto a circle of
/// radius 3, which is what `monotone_limit` below does.
///
/// Used for the total-variance term structure, where the guarantee is a
/// no-arbitrage requirement rather than a nicety: a natural cubic spline
/// through monotone total variances overshoots between expiries, and an
/// overshoot downward *is* calendar arbitrage.
class MonotoneCubic {
  public:
    MonotoneCubic() = default;
    MonotoneCubic(std::vector<double> xs, std::vector<double> ys);

    [[nodiscard]] double operator()(double x) const noexcept;
    [[nodiscard]] double derivative(double x) const noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return xs_.size(); }
    [[nodiscard]] std::span<const double> nodes() const noexcept { return xs_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return ys_; }
    [[nodiscard]] std::span<const double> slopes() const noexcept { return m_; }
    [[nodiscard]] bool empty() const noexcept { return xs_.empty(); }

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<double> m_;  ///< limited knot slopes
};

/// Whether a sequence is strictly increasing -- used to validate knot vectors
/// before building an interpolator, since every scheme here assumes it.
[[nodiscard]] bool is_strictly_increasing(std::span<const double> xs) noexcept;

/// Whether a sequence is non-decreasing (the condition total variance must
/// satisfy along the time axis).
[[nodiscard]] bool is_non_decreasing(std::span<const double> xs) noexcept;

}  // namespace vl::math
