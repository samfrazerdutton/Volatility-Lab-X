// SPDX-License-Identifier: MIT
#pragma once
/// \file reference_special.hpp
/// \brief Double-double reference implementations of the special functions.
///
/// These are the top of the validation chain required by section 27:
///
///     reference (double-double, here)
///         -> scalar optimised (math/special.hpp)
///             -> SIMD (kernels/simd)
///                 -> parallel (kernels/parallel)
///                     -> GPU (kernels/cuda)
///
/// Each stage is tested against the one above it, with the tolerance named in
/// `math/compare.hpp`.  Nothing in a hot path may call anything declared here:
/// these routines are 20-100x slower than their double counterparts and exist
/// purely so that the fast paths have something independent to be wrong
/// against.
///
/// The `_ref` overloads take and return `double` for convenience in tests; the
/// `_dd` overloads expose the full precision for chained reference
/// computations (the reference Black pricer needs Phi to 106 bits internally,
/// not to 53).

#include "volatility_lab/math/dd_real.hpp"

namespace vl::math::reference {

// -- full-precision interfaces ----------------------------------------------
[[nodiscard]] DDouble norm_cdf_dd(DDouble x) noexcept;
[[nodiscard]] DDouble norm_pdf_dd(DDouble x) noexcept;
[[nodiscard]] DDouble erf_dd(DDouble z) noexcept;
[[nodiscard]] DDouble erfc_dd(DDouble z) noexcept;
[[nodiscard]] DDouble erfcx_dd(DDouble x) noexcept;

// -- double-returning convenience wrappers ----------------------------------
[[nodiscard]] double norm_cdf_ref(double x) noexcept;
[[nodiscard]] double norm_pdf_ref(double x) noexcept;
[[nodiscard]] double erfcx_ref(double x) noexcept;
[[nodiscard]] double norm_inv_ref(double p) noexcept;

}  // namespace vl::math::reference
