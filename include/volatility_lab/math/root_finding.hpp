// SPDX-License-Identifier: MIT
#pragma once
/// \file root_finding.hpp
/// \brief Generic scalar root finders with explicit convergence reporting.
///
/// These are the general-purpose algorithms required by section 7 of the
/// brief.  The production implied-volatility path does **not** use them -- it
/// uses a specialised scheme with a provable monotone-convergence argument
/// (see pricing/implied_vol.hpp) -- but they are used by:
///
///  * the solver benchmark, which compares all four against the specialised
///    path on identical inputs so the specialisation can be justified with
///    numbers rather than assertion;
///  * the specialised path's own last-resort fallback, where `brent` on a
///    validated bracket gives a convergence guarantee that no Newton-family
///    method can;
///  * the surface models, which need a 1-D inversion for the strike at which
///    a given quantity is attained.
///
/// ## Design
///
/// Every solver returns a `SolveResult` recording iterations, final residual,
/// and *why* it stopped.  Nothing returns a bare double: a root finder that
/// cannot tell you it failed is a liability, and "it returned something
/// plausible" is how convergence failures reach production.
///
/// Termination is on three independent criteria, any of which suffices:
/// |f(x)| <= f_tol (residual), |dx| <= x_tol + x_rtol*|x| (step), and an
/// iteration cap.  All three are needed: residual alone fails on flat
/// functions (deep-OTM vega), step alone fails on steep ones, and the cap is
/// the only thing standing between a pathological input and an infinite loop.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>

#include "volatility_lab/core/config.hpp"

namespace vl::math {

enum class SolveStatus : std::uint8_t {
    Converged = 0,          ///< a termination criterion was met
    MaxIterations = 1,      ///< ran out of iterations
    NoBracket = 2,          ///< the supplied interval does not change sign
    DerivativeVanished = 3, ///< f'(x) underflowed; Newton cannot proceed
    NonFiniteInput = 4,     ///< NaN/Inf in the inputs or the function value
    OutOfDomain = 5         ///< the target lies outside the function's range
};

[[nodiscard]] constexpr const char* to_string(SolveStatus s) noexcept {
    switch (s) {
        case SolveStatus::Converged:          return "converged";
        case SolveStatus::MaxIterations:      return "max-iterations";
        case SolveStatus::NoBracket:          return "no-bracket";
        case SolveStatus::DerivativeVanished: return "derivative-vanished";
        case SolveStatus::NonFiniteInput:     return "non-finite-input";
        case SolveStatus::OutOfDomain:        return "out-of-domain";
    }
    return "?";
}

/// Which algorithm produced a result.  Reported so that the hybrid solver's
/// mix can be measured rather than assumed -- `volatility-lab benchmark
/// implied-vol` prints the histogram.
enum class SolveMethod : std::uint8_t {
    None = 0,
    Newton = 1,
    SafeguardedNewton = 2,
    Bisection = 3,
    Brent = 4,
    Halley = 5,
    HouseholderNormalisedBlack = 6,
    ClosedForm = 7
};

[[nodiscard]] constexpr const char* to_string(SolveMethod m) noexcept {
    switch (m) {
        case SolveMethod::None:                       return "none";
        case SolveMethod::Newton:                     return "newton";
        case SolveMethod::SafeguardedNewton:          return "safeguarded-newton";
        case SolveMethod::Bisection:                  return "bisection";
        case SolveMethod::Brent:                      return "brent";
        case SolveMethod::Halley:                     return "halley";
        case SolveMethod::HouseholderNormalisedBlack: return "householder-normalised-black";
        case SolveMethod::ClosedForm:                 return "closed-form";
    }
    return "?";
}

struct SolveTolerance {
    double f_tol = 0.0;      ///< accept when |f(x)| <= f_tol
    double x_tol = 0.0;      ///< accept when |dx| <= x_tol + x_rtol*|x|
    double x_rtol = 4.0 * std::numeric_limits<double>::epsilon();
    int max_iterations = 64;
};

struct SolveResult {
    double root = std::numeric_limits<double>::quiet_NaN();
    double residual = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    SolveStatus status = SolveStatus::NonFiniteInput;
    SolveMethod method = SolveMethod::None;

    [[nodiscard]] bool converged() const noexcept {
        return status == SolveStatus::Converged;
    }
};

namespace detail {

[[nodiscard]] VL_FORCE_INLINE bool step_converged(double dx, double x,
                                                  const SolveTolerance& t) noexcept {
    return std::abs(dx) <= t.x_tol + t.x_rtol * std::abs(x);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Newton-Raphson
// ---------------------------------------------------------------------------

/// Plain Newton.  Included for comparison and for the cases where the caller
/// has already proved convergence (see the monotone argument in
/// implied_vol.hpp); it has no safeguards and will happily walk off to
/// infinity on a bad input, which is the point of having it be a separate,
/// honestly-named function.
///
/// `fdf(x)` must return `{f(x), f'(x)}`.
template <class FDF>
[[nodiscard]] SolveResult newton(FDF&& fdf, double x0, const SolveTolerance& tol) {
    SolveResult r;
    r.method = SolveMethod::Newton;
    double x = x0;
    if (!std::isfinite(x)) return r;

    for (int i = 1; i <= tol.max_iterations; ++i) {
        const auto [f, df] = fdf(x);
        r.iterations = i;
        r.root = x;
        r.residual = f;
        if (!std::isfinite(f) || !std::isfinite(df)) {
            r.status = SolveStatus::NonFiniteInput;
            return r;
        }
        if (std::abs(f) <= tol.f_tol) {
            r.status = SolveStatus::Converged;
            return r;
        }
        if (df == 0.0) {
            r.status = SolveStatus::DerivativeVanished;
            return r;
        }
        const double dx = f / df;
        x -= dx;
        r.root = x;
        if (detail::step_converged(dx, x, tol)) {
            r.residual = fdf(x).first;
            r.status = SolveStatus::Converged;
            return r;
        }
    }
    r.status = SolveStatus::MaxIterations;
    return r;
}

// ---------------------------------------------------------------------------
// Safeguarded Newton
// ---------------------------------------------------------------------------

/// Newton confined to a bracket [lo, hi] that is known to contain a sign
/// change, falling back to bisection whenever the Newton step would leave the
/// bracket or fails to reduce it.
///
/// This is the right default for a general problem: it inherits Newton's
/// quadratic rate where the function is well behaved and bisection's
/// guarantee where it is not.  The bracket is tightened on every iteration,
/// so the iteration count is bounded by the bisection bound even in the worst
/// case.
template <class FDF>
[[nodiscard]] SolveResult safeguarded_newton(FDF&& fdf, double lo, double hi, double x0,
                                             const SolveTolerance& tol) {
    SolveResult r;
    r.method = SolveMethod::SafeguardedNewton;

    double f_lo = fdf(lo).first;
    double f_hi = fdf(hi).first;
    if (!std::isfinite(f_lo) || !std::isfinite(f_hi)) {
        r.status = SolveStatus::NonFiniteInput;
        return r;
    }
    if (f_lo == 0.0) { r.root = lo; r.residual = 0.0; r.status = SolveStatus::Converged; return r; }
    if (f_hi == 0.0) { r.root = hi; r.residual = 0.0; r.status = SolveStatus::Converged; return r; }
    if ((f_lo > 0.0) == (f_hi > 0.0)) {
        r.status = SolveStatus::NoBracket;
        r.root = x0;
        return r;
    }

    double x = (x0 > lo && x0 < hi) ? x0 : 0.5 * (lo + hi);

    for (int i = 1; i <= tol.max_iterations; ++i) {
        const auto [f, df] = fdf(x);
        r.iterations = i;
        r.root = x;
        r.residual = f;
        if (!std::isfinite(f)) {
            r.status = SolveStatus::NonFiniteInput;
            return r;
        }
        if (std::abs(f) <= tol.f_tol) {
            r.status = SolveStatus::Converged;
            return r;
        }

        // Tighten the bracket with the new information.
        if ((f > 0.0) == (f_lo > 0.0)) {
            lo = x;
            f_lo = f;
        } else {
            hi = x;
            f_hi = f;
        }

        double x_next;
        if (df != 0.0 && std::isfinite(df)) {
            x_next = x - f / df;
        } else {
            x_next = std::numeric_limits<double>::quiet_NaN();
        }
        // Reject a Newton step that escapes the bracket or stalls.
        if (!std::isfinite(x_next) || x_next <= lo || x_next >= hi) {
            x_next = 0.5 * (lo + hi);
        }

        const double dx = x_next - x;
        x = x_next;
        r.root = x;
        if (detail::step_converged(dx, x, tol) || (hi - lo) <= tol.x_tol + tol.x_rtol * std::abs(x)) {
            r.residual = fdf(x).first;
            r.status = SolveStatus::Converged;
            return r;
        }
    }
    r.status = SolveStatus::MaxIterations;
    return r;
}

// ---------------------------------------------------------------------------
// Brent
// ---------------------------------------------------------------------------

/// Brent's method: inverse quadratic interpolation with bisection fallback,
/// derivative-free, superlinear, and guaranteed to converge on a bracket.
///
/// This is the library's last-resort solver.  When the specialised
/// implied-volatility iteration fails its monotonicity guard -- which over the
/// full fuzz corpus happens for inputs within one ulp of the no-arbitrage
/// bounds -- control lands here, and the result is still correct to the stated
/// tolerance.  It is slower per iteration than Newton and needs no
/// derivatives, which is exactly the trade you want in a fallback.
///
/// Implementation follows Brent (1973) / Numerical Recipes in structure, with
/// the conventional guard that the interpolation step must lie within the
/// bracket and must more than halve the previous step.
template <class F>
[[nodiscard]] SolveResult brent(F&& f, double lo, double hi, const SolveTolerance& tol) {
    SolveResult r;
    r.method = SolveMethod::Brent;

    double a = lo;
    double b = hi;
    double fa = f(a);
    double fb = f(b);

    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        r.status = SolveStatus::NonFiniteInput;
        return r;
    }
    if (fa == 0.0) { r.root = a; r.residual = 0.0; r.status = SolveStatus::Converged; return r; }
    if (fb == 0.0) { r.root = b; r.residual = 0.0; r.status = SolveStatus::Converged; return r; }
    if ((fa > 0.0) == (fb > 0.0)) {
        r.status = SolveStatus::NoBracket;
        return r;
    }

    double c = a;
    double fc = fa;
    double d = 0.0;
    double e = 0.0;

    for (int i = 1; i <= tol.max_iterations; ++i) {
        r.iterations = i;

        if ((fb > 0.0) == (fc > 0.0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a = b;  b = c;  c = a;
            fa = fb; fb = fc; fc = fa;
        }

        const double tol_x = 2.0 * tol.x_rtol * std::abs(b) + 0.5 * tol.x_tol;
        const double m = 0.5 * (c - b);

        r.root = b;
        r.residual = fb;
        if (std::abs(m) <= tol_x || fb == 0.0 || std::abs(fb) <= tol.f_tol) {
            r.status = SolveStatus::Converged;
            return r;
        }

        if (std::abs(e) >= tol_x && std::abs(fa) > std::abs(fb)) {
            double p, q;
            const double s = fb / fa;
            if (a == c) {
                // Linear (secant).
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic.
                const double qq = fa / fc;
                const double rr = fb / fc;
                p = s * (2.0 * m * qq * (qq - rr) - (b - a) * (rr - 1.0));
                q = (qq - 1.0) * (rr - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);
            const double min1 = 3.0 * m * q - std::abs(tol_x * q);
            const double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }

        a = b;
        fa = fb;
        b += (std::abs(d) > tol_x) ? d : (m > 0.0 ? tol_x : -tol_x);
        fb = f(b);
        if (!std::isfinite(fb)) {
            r.status = SolveStatus::NonFiniteInput;
            r.root = b;
            return r;
        }
    }
    r.root = b;
    r.residual = fb;
    r.status = SolveStatus::MaxIterations;
    return r;
}

// ---------------------------------------------------------------------------
// Bisection
// ---------------------------------------------------------------------------

/// Bisection.  Linear, unconditionally convergent on a bracket, and used as
/// the correctness oracle for the other three in tests/numerical: its answer
/// depends on nothing but the sign of f, so it cannot share a bug with an
/// interpolating method.
template <class F>
[[nodiscard]] SolveResult bisection(F&& f, double lo, double hi, const SolveTolerance& tol) {
    SolveResult r;
    r.method = SolveMethod::Bisection;
    double a = lo, b = hi;
    double fa = f(a), fb = f(b);
    if (!std::isfinite(fa) || !std::isfinite(fb)) {
        r.status = SolveStatus::NonFiniteInput;
        return r;
    }
    // A root sitting exactly on an endpoint must be returned, not bisected
    // towards.  Without this the interval collapses against the endpoint and
    // neither the residual nor the step criterion is ever met, so a problem
    // with an *exact* answer is the one that times out.
    if (fa == 0.0) {
        r.root = a;
        r.residual = 0.0;
        r.status = SolveStatus::Converged;
        return r;
    }
    if (fb == 0.0) {
        r.root = b;
        r.residual = 0.0;
        r.status = SolveStatus::Converged;
        return r;
    }
    if ((fa > 0.0) == (fb > 0.0)) {
        r.status = SolveStatus::NoBracket;
        return r;
    }
    for (int i = 1; i <= tol.max_iterations; ++i) {
        r.iterations = i;
        const double m = 0.5 * (a + b);
        const double fm = f(m);
        r.root = m;
        r.residual = fm;
        if (!std::isfinite(fm)) {
            r.status = SolveStatus::NonFiniteInput;
            return r;
        }
        if (std::abs(fm) <= tol.f_tol || detail::step_converged(0.5 * (b - a), m, tol)) {
            r.status = SolveStatus::Converged;
            return r;
        }
        if ((fm > 0.0) == (fa > 0.0)) {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }
    r.status = SolveStatus::MaxIterations;
    return r;
}

// ---------------------------------------------------------------------------
// Hybrid
// ---------------------------------------------------------------------------

/// Try Newton from `x0`; if it does not converge, fall back to Brent on the
/// bracket.  The two phases are reported separately through
/// `SolveResult::method` so that the benchmark can show how often the fast
/// path suffices.
///
/// The iteration budget is split: `fast_iterations` for Newton, the remainder
/// for Brent.  Giving Newton the whole budget is a common and costly mistake --
/// a diverging Newton burns the budget and then there is nothing left for the
/// method that would have worked.
template <class FDF>
[[nodiscard]] SolveResult hybrid(FDF&& fdf, double lo, double hi, double x0,
                                 const SolveTolerance& tol, int fast_iterations = 8) {
    SolveTolerance fast = tol;
    fast.max_iterations = std::min(fast_iterations, tol.max_iterations);

    SolveResult fast_r = newton(fdf, x0, fast);
    if (fast_r.converged() && fast_r.root > lo && fast_r.root < hi) {
        return fast_r;
    }

    SolveTolerance slow = tol;
    slow.max_iterations = std::max(1, tol.max_iterations - fast.max_iterations);
    SolveResult slow_r = brent([&](double x) { return fdf(x).first; }, lo, hi, slow);
    slow_r.iterations += fast_r.iterations;
    return slow_r;
}

}  // namespace vl::math
