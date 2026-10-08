// SPDX-License-Identifier: MIT
#include "scalar/erfcx_poly.hpp"

#include <cmath>
#include <limits>

#include "volatility_lab/math/special.hpp"

namespace vl::kernels::scalar {

namespace {

/// `t * exp(poly(t))`, `t = 1/(1+x/2)`, for `x >= 0` -- see the header for
/// why the `exp(x^2)`/`exp(-x^2)` cancellation makes this valid for erfcx
/// (not merely for erfc) with no domain switch.
double erfcx_poly_nonneg(double x) noexcept {
    const double t = 1.0 / (1.0 + 0.5 * x);
    const double poly =
        -1.26551223 +
        t * (1.00002368 +
             t * (0.37409196 +
                  t * (0.09678418 +
                       t * (-0.18628806 +
                            t * (0.27886807 +
                                 t * (-1.13520398 +
                                      t * (1.48851587 +
                                           t * (-0.82215223 + t * 0.17087277))))))));
    return t * std::exp(poly);
}

}  // namespace

double erfcx_poly(double x) noexcept {
    if (std::isnan(x)) return x;
    if (x >= 0.0) return erfcx_poly_nonneg(x);

    const double y = -x;
    if (y > math::kExpSqMax) return std::numeric_limits<double>::infinity();
    return 2.0 * math::exp_sq(y) - erfcx_poly_nonneg(y);
}

}  // namespace vl::kernels::scalar
