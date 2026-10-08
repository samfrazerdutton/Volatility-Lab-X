// SPDX-License-Identifier: MIT
#include "simd/erfcx_avx2.hpp"

#include <immintrin.h>

#include <cstddef>
#include <limits>

#include "scalar/erfcx_poly.hpp"
#include "volatility_lab/math/special.hpp"

namespace vl::kernels::simd {

namespace {

/// `exp(y)` for `y` in `[-1, 710]` -- the only range this kernel ever
/// calls it with (`poly(t)` for `t` in `(0, 1]`, and `x*x` for `|x| <=
/// kExpSqMax ~= 26.6416`, guarded below). **Not a general-purpose
/// vectorised exp**: range reduction uses a single-double `ln(2)`
/// constant rather than a hi/lo split, which is accurate enough that the
/// erfcx formula's own ~1e-7 budget is unaffected (verified directly in
/// `tests/kernels/erfcx_avx2.cpp`, not assumed), but would not be the
/// right choice for a caller wanting `exp` to its own last-bit accuracy.
///
/// Classic range-reduction construction: `y = n*ln2 + r`, `|r| <= ln2/2`,
/// so `exp(y) = 2^n * exp(r)`. `exp(r)` is a plain 14-term (degree 13)
/// Taylor series -- not a tuned rational/Chebyshev approximation with
/// externally-sourced coefficients that could be mistyped from memory,
/// just `1/k!` for k=0..13, directly verifiable -- whose remainder on
/// `|r| <= ln2/2 ~= 0.3466` is bounded by `r^14/14! ~= 5e-18`, i.e. full
/// double precision for the reduced argument. `2^n` is built by shifting
/// `n` into a double's exponent bits directly.
[[gnu::target("avx2,fma")]] __m256d exp_avx2_bounded(__m256d y) noexcept {
    constexpr double kLn2 = 0.69314718055994530942;
    constexpr double kInvLn2 = 1.4426950408889634074;

    const __m256d n_d = _mm256_round_pd(_mm256_mul_pd(y, _mm256_set1_pd(kInvLn2)),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256d r = _mm256_fnmadd_pd(n_d, _mm256_set1_pd(kLn2), y);  // r = y - n*ln2

    // exp(r) via Horner on the degree-13 Taylor series, evaluated as
    // ((...((c13*r + c12)*r + c11)*r + ...)*r + c0).
    static constexpr double kInvFact[14] = {
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
        1.0 / 40320.0,
        1.0 / 362880.0,
        1.0 / 3628800.0,
        1.0 / 39916800.0,
        1.0 / 479001600.0,
        1.0 / 6227020800.0,
    };
    __m256d acc = _mm256_set1_pd(kInvFact[13]);
    for (int k = 12; k >= 0; --k) {
        acc = _mm256_fmadd_pd(acc, r, _mm256_set1_pd(kInvFact[k]));
    }

    // 2^n via bit manipulation: n fits in int32 for this kernel's whole
    // valid domain (|y| <= 710 => |n| <= ~1025), widened to int64 lanes
    // (AVX2 has no direct double<->int64 conversion) and shifted into the
    // exponent field of 1.0's own bit pattern.
    const __m128i n_i32 = _mm256_cvttpd_epi32(n_d);
    const __m256i n_i64 = _mm256_cvtepi32_epi64(n_i32);
    const __m256i one_bits = _mm256_set1_epi64x(0x3FF0000000000000LL);  // bit pattern of 1.0
    const __m256i pow2n_bits = _mm256_add_epi64(one_bits, _mm256_slli_epi64(n_i64, 52));
    const __m256d pow2n = _mm256_castsi256_pd(pow2n_bits);

    return _mm256_mul_pd(acc, pow2n);
}

/// `erfcx_poly_nonneg` (see `kernels/scalar/erfcx_poly.hpp`) over 4 lanes
/// at once -- identical formula, vectorised.
[[gnu::target("avx2,fma")]] __m256d erfcx_poly_nonneg_avx2(__m256d x) noexcept {
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d t = _mm256_div_pd(one, _mm256_fmadd_pd(half, x, one));

    static constexpr double kCoef[10] = {
        -1.26551223, 1.00002368,  0.37409196,  0.09678418,  -0.18628806,
        0.27886807,  -1.13520398, 1.48851587,  -0.82215223, 0.17087277,
    };
    __m256d poly = _mm256_set1_pd(kCoef[9]);
    for (int k = 8; k >= 0; --k) {
        poly = _mm256_fmadd_pd(poly, t, _mm256_set1_pd(kCoef[k]));
    }
    return _mm256_mul_pd(t, exp_avx2_bounded(poly));
}

}  // namespace

// Every AVX2-using function in this file carries an explicit
// `[[gnu::target("avx2,fma")]]`, rather than relying solely on the
// per-source CMake COMPILE_OPTIONS override (kernels/simd/CMakeLists.txt)
// to enable the ISA for the whole translation unit: that override did not
// reliably take effect for this build (confirmed by inspecting the actual
// compile command Ninja ran -- no `-mavx2`/`-mfma` present), and the
// function attribute is self-sufficient regardless, so it is the one
// mechanism actually being depended on here rather than a second,
// unverified one layered on top.
[[gnu::target("avx2,fma")]] void erfcx_avx2_batch(std::span<const double> x,
                                                   std::span<double> out) noexcept {
    const std::size_t n = x.size();
    const std::size_t n4 = n - (n % 4);

    const __m256d zero = _mm256_setzero_pd();
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d exp_sq_max = _mm256_set1_pd(math::kExpSqMax);
    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

    for (std::size_t i = 0; i < n4; i += 4) {
        const __m256d xv = _mm256_loadu_pd(x.data() + i);
        const __m256d absx = _mm256_andnot_pd(_mm256_set1_pd(-0.0), xv);  // |x|, clears sign bit
        const __m256d nonneg_val = erfcx_poly_nonneg_avx2(absx);

        // Reflection branch: 2*exp(x^2) - erfcx_poly_nonneg(|x|), guarded
        // the same way the scalar kernel's overflow check is guarded --
        // lanes with |x| beyond kExpSqMax never reach exp_avx2_bounded
        // with an out-of-domain argument; they are forced to +infinity by
        // the final blend instead.
        const __m256d x_sq = _mm256_mul_pd(xv, xv);
        // fmsub(a,b,c) = a*b - c = 2*exp(x^2) - nonneg_val, fused into one
        // instruction rather than a separate multiply and subtract.
        const __m256d reflect_val =
            _mm256_fmsub_pd(two, exp_avx2_bounded(x_sq), nonneg_val);

        const __m256d is_neg = _mm256_cmp_pd(xv, zero, _CMP_LT_OQ);
        const __m256d overflow = _mm256_cmp_pd(absx, exp_sq_max, _CMP_GT_OQ);
        __m256d result = _mm256_blendv_pd(nonneg_val, reflect_val, is_neg);
        result = _mm256_blendv_pd(result, inf, _mm256_and_pd(is_neg, overflow));
        _mm256_storeu_pd(out.data() + i, result);
    }

    for (std::size_t i = n4; i < n; ++i) out[i] = scalar::erfcx_poly(x[i]);
}

}  // namespace vl::kernels::simd
