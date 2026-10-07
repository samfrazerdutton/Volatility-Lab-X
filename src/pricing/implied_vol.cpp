// SPDX-License-Identifier: MIT
#include "volatility_lab/pricing/implied_vol.hpp"

#include <algorithm>
#include <cmath>

namespace vl {

using math::SolveMethod;

/// Worst-case relative accuracy of `normalised_black` over its whole domain,
/// as measured against the double-double reference in tests/pricing/black.cpp.
/// Used to report honest error bounds on the inversion: a bound that ignores
/// the pricer own error is not a bound.
constexpr double kNormalisedBlackWorstRtol = 3.0e-12;

const char* to_string(IvStatus s) noexcept {
    switch (s) {
        case IvStatus::Ok:                  return "ok";
        case IvStatus::PriceBelowIntrinsic: return "price-below-intrinsic";
        case IvStatus::PriceAboveBound:     return "price-above-bound";
        case IvStatus::PriceNotPositive:    return "price-not-positive";
        case IvStatus::PriceAtIntrinsic:    return "price-at-intrinsic";
        case IvStatus::BelowFloor:          return "below-floor";
        case IvStatus::AboveCeiling:        return "above-ceiling";
        case IvStatus::DidNotConverge:      return "did-not-converge";
        case IvStatus::InvalidInput:        return "invalid-input";
    }
    return "?";
}

DiagCode to_diag_code(IvStatus s) noexcept {
    switch (s) {
        case IvStatus::Ok:                  return DiagCode::Ok;
        case IvStatus::PriceBelowIntrinsic: return DiagCode::PriceBelowIntrinsic;
        case IvStatus::PriceAboveBound:     return DiagCode::PriceAboveForwardBound;
        case IvStatus::PriceNotPositive:    return DiagCode::NonPositivePrice;
        case IvStatus::PriceAtIntrinsic:    return DiagCode::IvVegaTooSmall;
        case IvStatus::BelowFloor:          return DiagCode::IvBelowFloor;
        case IvStatus::AboveCeiling:        return DiagCode::IvAboveCeiling;
        case IvStatus::DidNotConverge:      return DiagCode::IvSolverMaxIterations;
        case IvStatus::InvalidInput:        return DiagCode::NonFiniteValue;
    }
    return DiagCode::NonFiniteValue;
}

// ===========================================================================
// Initial guess
// ===========================================================================

namespace {

/// Exact ATM inversion: solve 2 Phi(s/2) - 1 = beta for s.
///
/// The obvious spelling, `2 * norm_inv((1 + beta)/2)`, silently loses accuracy
/// as beta approaches 1 -- which is exactly where it approaches its upper
/// bound, i.e. at high total volatility.  `1 + beta` discards the low bits of
/// beta, and for beta near 1 the recovered s came out 3e-5 wrong in relative
/// terms: not a rounding artefact but a visible error in a branch the header
/// advertises as closed form and exact.
///
/// The fix is to work in whichever tail still carries the information:
///
///   beta <  0.5 : p = (1 + beta)/2 lies in [0.5, 0.75) and is well conditioned.
///   beta >= 0.5 : q = (1 - beta)/2.  Here `1 - beta` is **exact** by
///                 Sterbenz's lemma, and the upper-tail entry point consumes q
///                 directly, so nothing is discarded at all.
double atm_total_volatility(double beta) noexcept {
    if (beta < 0.5) {
        return 2.0 * math::norm_inv(0.5 * (1.0 + beta));
    }
    return 2.0 * math::norm_inv_upper(0.5 * (1.0 - beta));
}

}  // namespace



namespace {

/// Asymptotic inversion for the convex (low-variance) branch.
///
/// As s -> 0 at fixed x < 0 the leading behaviour of b is
///
///     b ~ (2 s / |x|) * (1/sqrt(2pi)) * exp(-x^2/(2 s^2) - s^2/8)
///
/// obtained by Laplace's method on the integral representation
/// b = 2 exp(ht) (1/sqrt(2pi)) exp(-h^2/2) * integral_0^t exp(-hu - u^2/2) du
/// with the dominant contribution at the upper endpoint.  Dropping everything
/// but the Gaussian and taking logs,
///
///     log beta ~ -x^2/(2 s^2)   =>   s ~ |x| / sqrt(-2 log beta)
///
/// which is the classic deep-OTM estimate.  One fixed-point refinement then
/// puts the neglected algebraic prefactor back:
///
///     s^2 = x^2 / (2 [ log(2 s / (|x| sqrt(2pi))) - log beta - s^2/8 ])
///
/// Two passes are enough to land within a few percent, which is all Halley
/// needs.
double guess_low_branch(double beta, double ax) noexcept {
    const double log_beta = std::log(beta);
    double s = ax / std::sqrt(-2.0 * log_beta);
    if (!(s > 0.0) || !std::isfinite(s)) return ax;  // beta ~ 1: fall back

    for (int i = 0; i < 2; ++i) {
        const double pref = std::log(2.0 * s / (ax * math::kSqrt2Pi));
        const double denom = 2.0 * (pref - log_beta - 0.125 * s * s);
        if (!(denom > 0.0) || !std::isfinite(denom)) break;
        const double s_new = ax / std::sqrt(denom);
        if (!(s_new > 0.0) || !std::isfinite(s_new)) break;
        s = s_new;
    }
    return s;
}

/// Inversion for the concave (high-variance) branch.
///
/// Above s_c the total variance dominates the moneyness, so the smile is
/// locally ATM-like.  At x == 0 the inversion is exact and closed form:
/// b(0, s) = erf(s / (2 sqrt2)) = 2 Phi(s/2) - 1, hence
/// s = 2 Phi^{-1}((1 + b)/2).  For x < 0 the same formula is applied to the
/// price rescaled by its own upper bound, u = beta / exp(x/2) in (0, 1), which
/// reduces to the exact result at x = 0 and degrades gracefully away from it.
/// Returns 0 when the formula carries no information, which happens whenever
/// `u` is small enough that `1 + u` rounds to 1 and `norm_inv(0.5)` returns
/// exactly 0.  **It must not substitute a plausible-looking value here.**  An
/// earlier version returned sqrt(2|x|) = s_c as a fallback, and because the
/// caller then computed |h| = |x|/s_atm from it, the estimate came out at 0.019
/// -- "near the money" -- for an option 7 standard deviations out.  The caller
/// trusted that, returned s_c as the guess, and the subsequent log-space Newton
/// step from a point 380x too high collapsed the iterate to 1e-18 in two
/// iterations.  All 593 fast-path failures in the sweep had this single cause.
/// A guess function that cannot answer must say so.
double guess_high_branch(double beta, double ax) noexcept {
    const double upper = std::exp(-0.5 * ax);
    const double u = std::min(beta / upper, 1.0 - 1e-16);
    const double s = atm_total_volatility(u);
    return (s > 0.0 && std::isfinite(s)) ? s : 0.0;
}

}  // namespace

double implied_vol_initial_guess(double beta, double x) noexcept {
    const double ax = std::abs(x);
    if (ax == 0.0) {
        return atm_total_volatility(std::min(beta, 1.0 - 1e-16));
    }

    // Which approximation applies is decided by |h| = |x|/s, not by which side
    // of the inflection the root lies on.  Those are different questions, and
    // conflating them was the first version mistake: for small |x| the
    // inflection s_c = sqrt(2|x|) is itself small, so the whole convex branch
    // has |h| > sqrt(|x|/2) -- which is *tiny*, not large.  Feeding a deep-OTM
    // asymptotic into that region produced guesses wrong by a factor of 400
    // and iteration counts in the thirties.
    //
    // So: compute the ATM-anchored guess first (it costs one norm_inv and is
    // valid whenever |x| << s), use it to estimate |h|, and only switch to the
    // deep-OTM asymptotic when that estimate says the option really is far out
    // of the money in standard-deviation terms.
    double s;
    const double s_atm = guess_high_branch(beta, ax);
    if (!(s_atm > 0.0) || !std::isfinite(s_atm)) {
        s = guess_low_branch(beta, ax);
    } else if (ax / s_atm <= 1.0) {
        s = s_atm;
    } else {
        const double s_low = guess_low_branch(beta, ax);
        if (!(s_low > 0.0) || !std::isfinite(s_low)) {
            s = s_atm;
        } else {
            // In the transition band both approximations are mediocre.  Rather
            // than guess which, evaluate the objective at each and keep the
            // closer one -- measured in log price, because beta spans 300
            // decades and a linear comparison would always pick the larger
            // candidate.
            const double log_beta = std::log(beta);
            const double e_atm = std::abs(
                std::log(std::max(normalised_black_value(-ax, s_atm), 1e-320)) - log_beta);
            const double e_low = std::abs(
                std::log(std::max(normalised_black_value(-ax, s_low), 1e-320)) - log_beta);
            s = (e_low <= e_atm) ? s_low : s_atm;
        }
    }

    if (!(s > 0.0) || !std::isfinite(s)) s = std::sqrt(2.0 * ax);

    // Confine the guess to the branch the root is known to lie in.  The
    // inflection s_c is an exact bound on the root from one side -- that is
    // what `beta < b_c` means -- so a guess outside it is known to be wrong
    // before any work is done.
    //
    // This clamp is deliberately the *single* exit point of the function.  An
    // earlier version returned early from two of the cases above and so
    // skipped it, which let a concave-branch guess land below s_c and cost the
    // solver iterations it had no need to spend.
    const auto infl = normalised_black_inflection(-ax);
    if (infl.s_c > 0.0) {
        if (beta < infl.b_c) {
            s = std::min(s, infl.s_c * (1.0 - 1e-12));
        } else {
            s = std::max(s, infl.s_c * (1.0 + 1e-12));
        }
    }
    return s;
}

// ===========================================================================
// The production inversion
// ===========================================================================

ImpliedVolResult implied_total_volatility(double beta, double x,
                                          const ImpliedVolConfig& cfg) noexcept {
    ImpliedVolResult r;
    r.method = SolveMethod::HouseholderNormalisedBlack;

    if (!std::isfinite(beta) || !std::isfinite(x)) {
        r.status = IvStatus::InvalidInput;
        return r;
    }
    const double ax = std::abs(x);
    const double xx = -ax;

    // ---- domain ---------------------------------------------------------
    // b is strictly increasing from 0 to exp(x/2).  Outside that open
    // interval there is no root, and nothing useful can be returned.
    if (!(beta > 0.0)) {
        r.status = (beta == 0.0) ? IvStatus::PriceBelowIntrinsic : IvStatus::PriceNotPositive;
        return r;
    }
    const double upper = std::exp(-0.5 * ax);
    if (beta >= upper) {
        r.status = IvStatus::PriceAboveBound;
        r.residual = beta - upper;
        return r;
    }

    // ---- ATM is closed form ---------------------------------------------
    // b(0, s) = 2 Phi(s/2) - 1 inverts exactly, so no iteration is needed and
    // none is reported.  This is not a shortcut for speed; it is the correct
    // answer, and iterating towards it would only add round-off.
    if (ax == 0.0) {
        const double s = atm_total_volatility(beta);
        r.total_volatility = s;
        r.initial_guess = s;
        r.iterations = 0;
        r.residual = normalised_black_value(0.0, s) - beta;
        r.method = SolveMethod::ClosedForm;
        r.status = IvStatus::Ok;
        return r;
    }

    // ---- branch selection -----------------------------------------------
    const auto infl = normalised_black_inflection(xx);
    const bool convex_branch = beta < infl.b_c;

    // Bracket.  Exact, from the convexity structure: the inflection separates
    // the two branches, so it is a hard bound on the root from one side, and
    // the domain endpoint bounds it from the other.
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    if (convex_branch) {
        hi = infl.s_c;
    } else {
        lo = infl.s_c;
    }

    double s = implied_vol_initial_guess(beta, xx);
    if (!(s > 0.0) || !std::isfinite(s)) s = infl.s_c;
    // Keep the start inside the branch; the monotone-convergence argument is
    // stated for the branch interior.
    s = std::min(std::max(s, lo * (1.0 + 1e-12)), std::isfinite(hi) ? hi : s);
    if (!(s > 0.0)) s = infl.s_c;
    r.initial_guess = s;

    const double log_beta = convex_branch ? std::log(beta) : 0.0;

    for (int i = 1; i <= cfg.max_iterations; ++i) {
        r.iterations = i;

        const NormalisedBlack nb = normalised_black(xx, s);
        const double b = nb.value;
        const double b1 = nb.dv_ds;
        const double x_over_s = xx / s;
        if (!std::isfinite(b) || !std::isfinite(b1)) {
            r.status = IvStatus::DidNotConverge;
            break;
        }

        const double resid = b - beta;
        r.residual = resid;

        // Residual test, relative to the price scale.
        if (std::abs(resid) <= cfg.price_rtol * beta) {
            r.total_volatility = s;
            r.status = IvStatus::Ok;
            break;
        }

        // Tighten the bracket.  b is increasing, so the sign of the residual
        // says directly which side of the root we are on.
        if (resid > 0.0) {
            hi = std::min(hi, s);
        } else {
            lo = std::max(lo, s);
        }

        if (b1 <= 0.0 || b <= 0.0) {
            // Vega has underflowed: no derivative information remains.  This
            // is the flat-vega case the brief calls out, and it is reported
            // rather than papered over.
            r.status = IvStatus::DidNotConverge;
            break;
        }

        const double b2 = normalised_black_d2(xx, s);

        // Halley step on the branch-appropriate objective.
        //
        // **Concave branch:** iterate on g = b - beta directly in s.  There
        // b is gentle -- bounded, with bounded curvature -- and any transform
        // would only add round-off.
        //
        // **Convex branch:** iterate on g = log b - log beta in the variable
        // u = log s, i.e. a *double* log transform.  The reason is that
        //
        //     d(log b)/d(log s) = s * vega / b  ~  1 + h^2
        //
        // which is positive, smooth, and O(1) near the money while growing
        // only quadratically deep out of it.  In raw (b, s) coordinates the
        // same function varies over 300 decades and its derivative over 600,
        // so a Newton model fitted at one point is worthless a few percent
        // away.  In (log b, log s) coordinates it is nearly a straight line
        // and the model is excellent everywhere.  This single change removed
        // the entire slow tail of the iteration-count distribution.
        //
        // Halley in the stable form  dz = -nu / (1 - 0.5*nu*g''/g'),
        // nu = g/g', where z is s or log s according to the branch.
        double g, gp, gpp;
        if (convex_branch) {
            const double q = s * b1 / b;  // dg/du, exactly
            g = std::log(b) - log_beta;
            gp = q;
            // d2g/du2 = q + s^2 (b''/b) - q^2.  Since b'' = b'(x^2/s^3 - s/4),
            // the middle term is exactly q*(h^2 - t^2), which collapses the
            // whole expression to
            //      d2g/du2 = q * (1 + h^2 - t^2 - q)
            // avoiding the formation of two O(h^4) quantities that cancel to
            // O(h^2).  The residual cancellation is between (1 + h^2 - t^2)
            // and q, both O(h^2) with an O(1) difference, so ~log10(h^2)
            // digits are still lost -- which is why the `denom` guard below
            // falls back to the plain log-space Newton step rather than
            // trusting a curvature estimate that may have no digits left.
            const double h2 = (x_over_s) * (x_over_s);
            const double t2 = 0.25 * s * s;
            gpp = q * (1.0 + h2 - t2 - q);
            (void)b2;
        } else {
            g = resid;
            gp = b1;
            gpp = b2;
        }

        double step;
        if (gp == 0.0 || !std::isfinite(gp)) {
            step = 0.0;
        } else {
            const double nu = g / gp;
            const double denom = 1.0 - 0.5 * nu * (gpp / gp);
            // A denominator that has drifted far from 1 means the quadratic
            // model is not trustworthy here; fall back to the plain Newton
            // step, which the convexity argument still protects.
            step = (denom > 0.25 && std::isfinite(denom)) ? (nu / denom) : nu;
        }

        // Trust region.  On the convex branch d(log b)/d(log s) = 1 + h^2 grows
        // without bound as s falls, so a model fitted at the current point can
        // ask for an enormous move: from s_c with beta 18 decades below b_c the
        // raw step is -36 in log s, i.e. a factor of 2e-16, which lands below
        // the representable range and loses the root entirely.  Capping the
        // log-space step at a factor of e^2 per iteration costs a handful of
        // iterations in that corner and makes the collapse impossible.
        constexpr double kMaxLogStep = 2.0;
        if (convex_branch) step = std::clamp(step, -kMaxLogStep, kMaxLogStep);

        // Map the step back into s.  On the convex branch the step is in
        // log s, so the update is multiplicative -- which also guarantees the
        // iterate stays positive without a clamp.
        double s_next = convex_branch ? (s * std::exp(-step)) : (s - step);

        // Clamp into the bracket.  Not because the convergence proof is
        // doubted, but because the proof is about exact arithmetic: it says
        // nothing about an iterate that lands outside the domain when beta is
        // within an ulp of a bound.
        if (!std::isfinite(s_next) || s_next <= lo || s_next >= hi) {
            s_next = std::isfinite(hi) ? 0.5 * (lo + hi) : (s * 2.0);
        }

        // Relative size of the move that was *actually* taken.  It has to be
        // measured from s_next, not from `step`: when the clamp below replaces
        // the Halley step with a bisection, `step` no longer describes what
        // happened, and reading it instead reports convergence on a move that
        // was never made.  (That bug cost 309 spurious failures and a worst
        // relative error of 166 before it was caught by the round-trip sweep.)
        const double rel_step = std::abs(s_next - s) / s_next;
        s = s_next;

        if (rel_step <= cfg.step_rtol) {
            r.total_volatility = s;
            r.residual = normalised_black_value(xx, s) - beta;
            r.status = IvStatus::Ok;
            break;
        }
    }

    if (r.status != IvStatus::Ok) {
        if (r.iterations >= cfg.max_iterations) r.status = IvStatus::DidNotConverge;
        if (cfg.allow_fallback && std::isfinite(hi) && hi > lo) {
            // Last resort: Brent on the validated bracket.  It shares no
            // structure with the iteration above -- no derivatives, no initial
            // guess, no branch logic -- so it cannot fail the same way.
            math::SolveTolerance tol;
            tol.f_tol = cfg.price_rtol * beta;
            tol.x_rtol = cfg.step_rtol;
            tol.max_iterations = 128;
            const auto br = math::brent(
                [&](double ss) { return normalised_black_value(xx, ss) - beta; }, lo, hi, tol);
            if (br.converged()) {
                r.total_volatility = br.root;
                r.residual = br.residual;
                r.iterations += br.iterations;
                r.method = SolveMethod::Brent;
                r.used_fallback = true;
                r.status = IvStatus::Ok;
            }
        }
    }

    if (r.status == IvStatus::Ok) {
        r.volatility = r.total_volatility;  // caller divides by sqrt(T)
    }
    return r;
}

// ===========================================================================
// Price-space entry points
// ===========================================================================

ImpliedVolResult implied_volatility_undiscounted(double undiscounted_price, double forward,
                                                 double strike, double years, OptionType type,
                                                 const ImpliedVolConfig& cfg) noexcept {
    ImpliedVolResult r;
    if (!(forward > 0.0) || !(strike > 0.0) || !(years > 0.0) ||
        !std::isfinite(undiscounted_price)) {
        r.status = IvStatus::InvalidInput;
        return r;
    }

    // Convert to the OTM side via put-call parity.  The ITM price is
    // intrinsic + epsilon, and epsilon is the only part that depends on
    // volatility; subtracting the (exactly representable) intrinsic recovers
    // it without the catastrophic cancellation that inverting the ITM price
    // directly would suffer.
    const PriceBounds bounds = forward_price_bounds(forward, strike, type);
    const double otm_value = undiscounted_price - bounds.lower;

    if (otm_value <= 0.0) {
        // Zero and negative are different failures and are reported as such:
        // zero means the value is indistinguishable from intrinsic at double
        // precision (a representation limit), negative means the quote
        // violates the sigma -> 0 no-arbitrage bound (a data error).
        r.status = (otm_value == 0.0) ? IvStatus::PriceAtIntrinsic
                                      : IvStatus::PriceBelowIntrinsic;
        r.price_residual = otm_value;
        return r;
    }
    if (undiscounted_price >= bounds.upper) {
        r.status = IvStatus::PriceAboveBound;
        r.price_residual = undiscounted_price - bounds.upper;
        return r;
    }

    const double sqrt_fk = std::sqrt(forward) * std::sqrt(strike);
    const double beta = otm_value / sqrt_fk;
    const double x = -std::abs(std::log(forward / strike));

    r = implied_total_volatility(beta, x, cfg);
    r.price_residual = r.residual * sqrt_fk;

    if (r.status == IvStatus::Ok) {
        const double sqrt_t = std::sqrt(years);
        r.volatility = r.total_volatility / sqrt_t;

        // How much accuracy the input itself allows.  For an OTM quote
        // |price| == otm_value and this is a few eps; for a deep ITM quote the
        // ratio is large and the honest answer is far worse than the solver's
        // own residual suggests.  See the field comment in the header.
        // Two independent sources of relative error in the volatility, both
        // obtained by dividing a relative *price* error by the elasticity
        //
        //     q = d(log b)/d(log s) = s * vega / b
        //
        // which is exactly how much a relative price error is damped (q > 1) or
        // amplified (q < 1) on its way into the volatility.  The sources are:
        //
        //  * the *input* representation limit, eps*|price|/otm_value, which is
        //    what an in-the-money quote has already thrown away before the
        //    library is called; and
        //  * the pricer own worst-case relative accuracy, which no inversion
        //    can see past.
        //
        // `q` is computed exactly rather than from the small-s approximation
        // 1 + h^2.  The approximation is fine near the money but wildly wrong
        // in the opposite corner: at |x| = 6.4, s = 14.9 the true elasticity is
        // 1.1e-10, so the inversion *amplifies* relative price error by ten
        // orders of magnitude and the recovered volatility is good to only
        // ~1e-6.  That is a genuine property of the problem -- the price has
        // nearly stopped depending on volatility as it approaches its upper
        // bound -- and a conditioning field that reported 1 + h^2 = 1.2 there
        // would be advertising twelve digits the answer does not have.
        const double eps = std::numeric_limits<double>::epsilon();
        const double s_hat = r.total_volatility;
        const double b_hat = normalised_black_value(x, s_hat);
        const double vega_hat = normalised_black_vega(x, s_hat);
        const double elasticity =
            (b_hat > 0.0) ? (s_hat * vega_hat / b_hat) : 1.0;
        const double input_limit = eps * std::abs(undiscounted_price) / otm_value;
        r.attainable_rtol = (input_limit + kNormalisedBlackWorstRtol) /
                            std::max(elasticity, 1e-300);
        if (r.volatility < cfg.vol_floor) {
            r.status = IvStatus::BelowFloor;
        } else if (r.volatility > cfg.vol_ceiling) {
            r.status = IvStatus::AboveCeiling;
        }
    }
    return r;
}

ImpliedVolResult implied_volatility(double price, double forward, double strike, double years,
                                    double discount, OptionType type,
                                    const ImpliedVolConfig& cfg) noexcept {
    ImpliedVolResult r;
    if (!(discount > 0.0) || !std::isfinite(discount)) {
        r.status = IvStatus::InvalidInput;
        return r;
    }
    return implied_volatility_undiscounted(price / discount, forward, strike, years, type, cfg);
}

// ===========================================================================
// Alternative solvers, for the comparison benchmark
// ===========================================================================

ImpliedVolResult implied_total_volatility_with(SolveMethod method, double beta, double x,
                                               const ImpliedVolConfig& cfg) noexcept {
    ImpliedVolResult r;
    r.method = method;

    const double ax = std::abs(x);
    const double xx = -ax;
    if (!(beta > 0.0) || !std::isfinite(beta)) {
        r.status = IvStatus::PriceNotPositive;
        return r;
    }
    const double upper = std::exp(-0.5 * ax);
    if (beta >= upper) {
        r.status = IvStatus::PriceAboveBound;
        return r;
    }

    // A bracket wide enough for any admissible input.  The lower end cannot be
    // 0 for the derivative-based methods (vega underflows there), so it is the
    // smallest total volatility the library admits.
    const double lo = 1.0e-10;
    double hi = std::max(1.0, 4.0 * std::sqrt(2.0 * ax + 1.0));
    for (int i = 0; i < 60 && normalised_black_value(xx, hi) < beta; ++i) hi *= 2.0;

    const double s0 = implied_vol_initial_guess(beta, xx);
    r.initial_guess = s0;

    math::SolveTolerance tol;
    tol.f_tol = cfg.price_rtol * beta;
    tol.x_rtol = cfg.step_rtol;
    tol.max_iterations = cfg.max_iterations;

    const auto f = [&](double s) { return normalised_black_value(xx, s) - beta; };
    const auto fdf = [&](double s) {
        const NormalisedBlack nb = normalised_black(xx, s);
        return std::pair<double, double>{nb.value - beta, nb.dv_ds};
    };

    math::SolveResult sr;
    switch (method) {
        case SolveMethod::Newton:
            sr = math::newton(fdf, s0, tol);
            break;
        case SolveMethod::SafeguardedNewton:
            sr = math::safeguarded_newton(fdf, lo, hi, s0, tol);
            break;
        case SolveMethod::Brent:
            sr = math::brent(f, lo, hi, tol);
            break;
        case SolveMethod::Bisection:
            tol.max_iterations = std::max(tol.max_iterations, 200);
            sr = math::bisection(f, lo, hi, tol);
            break;
        case SolveMethod::Halley: {
            // Plain Halley on b - beta with no transform and no bracket, so
            // that the benchmark can show what the branch-appropriate
            // objective and the clamp actually buy.
            double s = s0;
            sr.method = SolveMethod::Halley;
            for (int i = 1; i <= tol.max_iterations; ++i) {
                sr.iterations = i;
                const NormalisedBlack nb = normalised_black(xx, s);
                const double g = nb.value - beta;
                sr.root = s;
                sr.residual = g;
                if (std::abs(g) <= tol.f_tol) {
                    sr.status = math::SolveStatus::Converged;
                    break;
                }
                const double gp = nb.dv_ds;
                if (gp == 0.0) {
                    sr.status = math::SolveStatus::DerivativeVanished;
                    break;
                }
                const double gpp = normalised_black_d2(xx, s);
                const double nu = g / gp;
                const double denom = 1.0 - 0.5 * nu * (gpp / gp);
                const double step = (std::isfinite(denom) && denom != 0.0) ? nu / denom : nu;
                s -= step;
                if (!std::isfinite(s) || s <= 0.0) {
                    sr.status = math::SolveStatus::OutOfDomain;
                    break;
                }
                sr.root = s;
                if (std::abs(step) <= tol.x_rtol * s) {
                    sr.residual = normalised_black_value(xx, s) - beta;
                    sr.status = math::SolveStatus::Converged;
                    break;
                }
            }
            if (sr.status != math::SolveStatus::Converged &&
                sr.iterations >= tol.max_iterations) {
                sr.status = math::SolveStatus::MaxIterations;
            }
            break;
        }
        default:
            return implied_total_volatility(beta, x, cfg);
    }

    r.total_volatility = sr.root;
    r.volatility = sr.root;
    r.residual = sr.residual;
    r.iterations = sr.iterations;
    r.status = sr.converged() ? IvStatus::Ok : IvStatus::DidNotConverge;
    return r;
}

}  // namespace vl
