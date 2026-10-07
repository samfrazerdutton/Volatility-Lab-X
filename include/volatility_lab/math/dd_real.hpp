// SPDX-License-Identifier: MIT
#pragma once
/// \file dd_real.hpp
/// \brief Double-double arithmetic: ~31 decimal digits from two doubles.
///
/// ## Why this exists
///
/// Section 27 of the project brief requires a slow, high-precision reference
/// implementation that the optimised kernels can be validated against.  The
/// obvious vehicle is `long double`, and on Linux/x86-64 that would give 64
/// bits of mantissa (about 3 extra decimal digits).  But on every MSVC-ABI
/// target -- including the clang++ configuration this project is developed on
/// -- `long double` **is** `double`.  A reference built on it would be
/// comparing the implementation against itself and would silently validate
/// nothing.
///
/// Double-double (Dekker 1971, Knuth 1969) represents a number as an
/// unevaluated sum `hi + lo` of two doubles with non-overlapping mantissas.
/// That yields 106 bits of significand (~31 digits) on *any* IEEE-754
/// platform, with no compiler or hardware dependency.  It is roughly 20x
/// slower than double, which is irrelevant for a reference and is precisely
/// why it is confined to `pricing/reference.cpp`.
///
/// ## Correctness requirements
///
/// The error-free transformations below are only valid under round-to-nearest
/// IEEE-754 arithmetic with no extra precision and no reassociation.  The
/// build therefore compiles with `-fno-fast-math` (see cmake/VLCompilerFlags)
/// and this header additionally refuses to compile if the compiler advertises
/// fast math.  `two_product` requires an FMA to be exact in one instruction;
/// where FMA is unavailable it falls back to Dekker splitting, which is exact
/// but slower.
///
/// ## What is NOT here
///
/// No square root of a negative, no transcendental beyond `exp`/`log`/`sqrt`
/// (that is all the reference needs), and no attempt at a general-purpose
/// multiprecision type.  Scope is kept to what the reference pricer uses so
/// that every line is exercised by a test.

#if defined(__FAST_MATH__)
#  error "dd_real.hpp requires strict IEEE-754 semantics; -ffast-math is enabled"
#endif

#include <cmath>
#include <cstdint>
#include <limits>

#include "volatility_lab/core/config.hpp"

namespace vl::math {

// ---------------------------------------------------------------------------
// Error-free transformations
// ---------------------------------------------------------------------------

/// Knuth two-sum: returns s = fl(a+b) and the exact rounding error e, such
/// that a + b == s + e exactly.  Valid for any finite a, b (no magnitude
/// ordering assumed), at the cost of six flops.
VL_FORCE_INLINE void two_sum(double a, double b, double& s, double& e) noexcept {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

/// Dekker fast two-sum.  Requires |a| >= |b|; three flops instead of six.
VL_FORCE_INLINE void quick_two_sum(double a, double b, double& s, double& e) noexcept {
    s = a + b;
    e = b - (s - a);
}

/// Exact product: p = fl(a*b), e = a*b - p.
///
/// With FMA this is two instructions and exact by construction.  Without, we
/// use Dekker splitting into 26+26 bit halves; the magic constant is
/// 2^27 + 1.
VL_FORCE_INLINE void two_product(double a, double b, double& p, double& e) noexcept {
#if defined(FP_FAST_FMA) || defined(__FMA__) || defined(__AVX2__)
    p = a * b;
    e = std::fma(a, b, -p);
#else
    constexpr double kSplit = 134217729.0;  // 2^27 + 1
    p = a * b;
    const double a_c = kSplit * a;
    const double a_hi = a_c - (a_c - a);
    const double a_lo = a - a_hi;
    const double b_c = kSplit * b;
    const double b_hi = b_c - (b_c - b);
    const double b_lo = b - b_hi;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#endif
}

// ---------------------------------------------------------------------------
// DDouble
// ---------------------------------------------------------------------------

/// Unevaluated sum of two non-overlapping doubles.  Invariant: |lo| <= ulp(hi)/2.
class DDouble {
  public:
    DDouble() = default;
    constexpr DDouble(double h) noexcept : hi_(h), lo_(0.0) {}  // NOLINT
    constexpr DDouble(double h, double l) noexcept : hi_(h), lo_(l) {}

    [[nodiscard]] constexpr double hi() const noexcept { return hi_; }
    [[nodiscard]] constexpr double lo() const noexcept { return lo_; }

    /// Collapse to the nearest double.  This is the only place precision is
    /// intentionally discarded, and it is always the last step of a reference
    /// computation.
    [[nodiscard]] constexpr double to_double() const noexcept { return hi_ + lo_; }

    // -- addition ----------------------------------------------------------
    friend DDouble operator+(DDouble a, DDouble b) noexcept {
        double s1, e1, s2, e2;
        two_sum(a.hi_, b.hi_, s1, e1);
        two_sum(a.lo_, b.lo_, s2, e2);
        e1 += s2;
        quick_two_sum(s1, e1, s1, e1);
        e1 += e2;
        quick_two_sum(s1, e1, s1, e1);
        return {s1, e1};
    }

    friend DDouble operator+(DDouble a, double b) noexcept {
        double s, e;
        two_sum(a.hi_, b, s, e);
        e += a.lo_;
        quick_two_sum(s, e, s, e);
        return {s, e};
    }
    friend DDouble operator+(double a, DDouble b) noexcept { return b + a; }

    friend DDouble operator-(DDouble a) noexcept { return {-a.hi_, -a.lo_}; }
    friend DDouble operator-(DDouble a, DDouble b) noexcept { return a + (-b); }
    friend DDouble operator-(DDouble a, double b) noexcept { return a + (-b); }
    friend DDouble operator-(double a, DDouble b) noexcept { return (-b) + a; }

    // -- multiplication ----------------------------------------------------
    friend DDouble operator*(DDouble a, DDouble b) noexcept {
        double p, e;
        two_product(a.hi_, b.hi_, p, e);
        e += a.hi_ * b.lo_ + a.lo_ * b.hi_;
        quick_two_sum(p, e, p, e);
        return {p, e};
    }

    friend DDouble operator*(DDouble a, double b) noexcept {
        double p, e;
        two_product(a.hi_, b, p, e);
        e += a.lo_ * b;
        quick_two_sum(p, e, p, e);
        return {p, e};
    }
    friend DDouble operator*(double a, DDouble b) noexcept { return b * a; }

    // -- division ----------------------------------------------------------
    //
    // Long division: one double-precision quotient digit, then one correction
    // digit computed from the exact remainder.  Two digits suffice because
    // each carries ~53 bits.
    friend DDouble operator/(DDouble a, DDouble b) noexcept {
        const double q1 = a.hi_ / b.hi_;
        DDouble r = a - b * q1;
        const double q2 = r.hi_ / b.hi_;
        r = r - b * q2;
        const double q3 = r.hi_ / b.hi_;
        double s, e;
        quick_two_sum(q1, q2, s, e);
        DDouble out{s, e};
        return out + q3;
    }
    friend DDouble operator/(DDouble a, double b) noexcept { return a / DDouble(b); }
    friend DDouble operator/(double a, DDouble b) noexcept { return DDouble(a) / b; }

    DDouble& operator+=(DDouble b) noexcept { return *this = *this + b; }
    DDouble& operator-=(DDouble b) noexcept { return *this = *this - b; }
    DDouble& operator*=(DDouble b) noexcept { return *this = *this * b; }
    DDouble& operator/=(DDouble b) noexcept { return *this = *this / b; }

    friend bool operator<(DDouble a, DDouble b) noexcept {
        return (a.hi_ < b.hi_) || (a.hi_ == b.hi_ && a.lo_ < b.lo_);
    }
    friend bool operator>(DDouble a, DDouble b) noexcept { return b < a; }
    friend bool operator==(DDouble a, DDouble b) noexcept {
        return a.hi_ == b.hi_ && a.lo_ == b.lo_;
    }
    friend bool operator<=(DDouble a, DDouble b) noexcept { return !(b < a); }
    friend bool operator>=(DDouble a, DDouble b) noexcept { return !(a < b); }

  private:
    double hi_ = 0.0;
    double lo_ = 0.0;
};

[[nodiscard]] inline DDouble abs(DDouble a) noexcept { return a.hi() < 0.0 ? -a : a; }

/// Newton-refined square root.  One Newton step on the double-precision
/// estimate doubles the correct digits from ~53 to ~106, which is exactly the
/// available precision, so no second step is needed.
[[nodiscard]] inline DDouble sqrt_dd(DDouble a) noexcept {
    if (a.hi() == 0.0) return DDouble(0.0);
    const double x = 1.0 / std::sqrt(a.hi());
    const double ax = a.hi() * x;
    // a - ax^2 computed in double-double, then the correction.
    const DDouble corr = (a - DDouble(ax) * DDouble(ax)) * (x * 0.5);
    return DDouble(ax) + corr;
}

/// exp for double-double.
///
/// Argument reduction a = k*ln2 + r with |r| <= ln2/2, then the Taylor series
/// for exp(r) -- which converges in ~18 terms at that argument range for 106
/// bits -- then scaling by 2^k.  ln2 is carried to double-double precision
/// because a single-double ln2 would cap the reduction's accuracy at 53 bits
/// and so cap the whole result.
[[nodiscard]] inline DDouble exp_dd(DDouble a) noexcept {
    // ln(2) to 106 bits.
    constexpr DDouble kLn2{6.93147180559945286e-01, 2.31904681384629956e-17};
    constexpr double kLn2d = 6.93147180559945286e-01;

    if (a.hi() <= -745.2) return DDouble(0.0);   // underflows double
    if (a.hi() >= 709.9) return DDouble(std::numeric_limits<double>::infinity());

    const double kf = std::nearbyint(a.hi() / kLn2d);
    const DDouble r = a - kLn2 * kf;

    // sum_{n>=0} r^n / n!
    DDouble term(1.0);
    DDouble sum(1.0);
    for (int n = 1; n <= 24; ++n) {
        term = term * r / static_cast<double>(n);
        sum += term;
        if (std::abs(term.hi()) < 1e-34 * std::abs(sum.hi())) break;
    }
    const double scale = std::ldexp(1.0, static_cast<int>(kf));
    return sum * scale;
}

/// log for double-double: Newton iteration on exp.
///   x_{n+1} = x_n + a*exp(-x_n) - 1
/// Seeded with the double-precision log, one step reaches full precision; a
/// second is taken because the seed's accuracy degrades for |log| >> 1.
[[nodiscard]] inline DDouble log_dd(DDouble a) noexcept {
    if (a.hi() <= 0.0) return DDouble(-std::numeric_limits<double>::infinity());
    DDouble x(std::log(a.hi()));
    for (int i = 0; i < 2; ++i) {
        x = x + (a * exp_dd(-x) - 1.0);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Constants to 106 bits.  Generated with mpmath and checked in
// tests/numerical/test_dd_real.cpp against independently computed hex values.
// ---------------------------------------------------------------------------
namespace dd_const {
inline constexpr DDouble kPi{3.14159265358979312e+00, 1.22464679914735321e-16};
inline constexpr DDouble kSqrt2{1.41421356237309515e+00, -9.66729331345291345e-17};
inline constexpr DDouble kSqrtHalf{7.07106781186547573e-01, -4.83364665672645673e-17};
inline constexpr DDouble kSqrt2Pi{2.50662827463100069e+00, -1.83285799804591668e-16};
inline constexpr DDouble kInvSqrt2Pi{3.98942280401432703e-01, -2.49232720227773004e-17};
inline constexpr DDouble kTwoOverSqrtPi{1.12837916709551256e+00, 1.53354596131658812e-17};
}  // namespace dd_const

}  // namespace vl::math
