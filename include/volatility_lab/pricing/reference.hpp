// SPDX-License-Identifier: MIT
#pragma once
/// \file reference.hpp
/// \brief Deliberately slow, deliberately independent reference pricer.
///
/// This is the top of the validation chain (brief section 27).  Its only job
/// is to be right; it is roughly 50x slower than `black_undiscounted` and
/// nothing outside tests, `volatility-lab validate`, and the accuracy columns
/// of the benchmarks may call it.
///
/// ## Independence
///
/// A reference that shares the implementation's structure validates nothing
/// but its arithmetic.  So the reference is built differently on purpose:
///
///  * **Different formula.**  The production path evaluates the erfcx-difference
///    form (equation (2) in pricing/black.hpp).  The reference evaluates the
///    *textbook* form (1), `exp(x/2) Phi(h+t) - exp(-x/2) Phi(h-t)`, directly.
///    That form is numerically hopeless in double -- which is the entire
///    reason the production path does not use it -- but in double-double it
///    has 106 bits to lose, so it stays accurate to well past double
///    resolution across the region where the production code uses form (2).
///    If form (2) had been derived or coded wrongly, this would catch it.
///
///  * **Different precision.**  Double-double (~31 digits), so a disagreement
///    at the 16th digit is unambiguously the implementation's.
///
///  * **Different fallback.**  Where even 106 bits are insufficient -- the
///    deep small-variance corner, w/|x| < 1e-10, where form (1) loses more
///    than 20 digits -- the reference uses the Taylor series *in double-double*.
///    The production code reaches its series far earlier (w/|x| < 1e-2), so
///    over the whole band 1e-10 < w/|x| < 1e-2 the double series is checked
///    against the double-double form (1), and below 1e-10 the two series are
///    compared against each other at different precisions.  No region is
///    validated only against itself.
///
/// ## Greeks
///
/// The reference Greeks are central differences *in double-double* with a step
/// chosen for the dd epsilon.  With ~1e-31 working precision, a step of 1e-10
/// gives truncation O(h^2) = 1e-20 and round-off O(eps_dd/h) = 1e-21, so the
/// reference Greek is good to ~1e-20 -- four orders of magnitude better than
/// the double analytic value it validates.  Doing this in plain double is
/// impossible: the optimal step there leaves only ~1e-11, which is *worse*
/// than the quantity under test.  This is the main reason the double-double
/// layer exists.

#include "volatility_lab/core/types.hpp"
#include "volatility_lab/math/dd_real.hpp"

namespace vl::reference {

using math::DDouble;

/// Normalised Black in double-double.  Requires x <= 0 (folded if not).
[[nodiscard]] DDouble normalised_black_dd(DDouble x, DDouble s) noexcept;

/// Normalised Black, double in / double out, computed at 106 bits internally.
[[nodiscard]] double normalised_black_ref(double x, double s) noexcept;

/// Undiscounted forward-measure price at 106 bits internally.
[[nodiscard]] double black_undiscounted_ref(double forward, double strike, double vol,
                                            double years, OptionType type) noexcept;

/// Reference normalised vega, db/ds, by a double-double central difference.
///
/// A double-precision central difference cannot validate this.  The optimal
/// step there leaves ~1e-11 of error, and worse, the step must be large enough
/// to survive the subtraction of two nearly equal doubles -- at |x| = 3,
/// s = 0.089 the log-derivative of b is 4e5, so any step big enough to be
/// representable is a large excursion in the exponent and the truncation term
/// dominates at 3e-7.  Carrying the difference in double-double removes both
/// constraints at once: with h = 1e-12*s the truncation term is ~1e-24 and the
/// round-off term ~1e-19, so the reference is seven orders of magnitude better
/// than the analytic value it is checking.
[[nodiscard]] double normalised_black_vega_ref(double x, double s) noexcept;

/// Reference Greeks by double-double central differences.  See the file
/// comment for why the step sizes are what they are.
struct ReferenceGreeks {
    double price;
    double delta;   ///< d/dF   (forward delta, undiscounted)
    double gamma;   ///< d2/dF2
    double vega;    ///< d/dsigma
    double volga;   ///< d2/dsigma2
    double vanna;   ///< d2/(dF dsigma)
    double theta;   ///< d/dT  (undiscounted, forward held fixed)
};

[[nodiscard]] ReferenceGreeks black_greeks_ref(double forward, double strike, double vol,
                                               double years, OptionType type) noexcept;

/// Reference implied volatility: bisection in double-double on the normalised
/// price, run to the full dd resolution.
///
/// Bisection is chosen on purpose.  It shares no structure at all with the
/// production Householder iteration -- no derivatives, no initial guess, no
/// branch logic -- so the two can only agree by both being right.  It needs
/// ~110 iterations at 106 bits, which is why this is a reference and not a
/// solver.
[[nodiscard]] double implied_vol_ref(double normalised_price, double x) noexcept;

}  // namespace vl::reference
