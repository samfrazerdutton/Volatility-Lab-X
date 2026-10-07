// SPDX-License-Identifier: MIT
#pragma once
/// \file types.hpp
/// \brief Strong scalar types and the fundamental enumerations.
///
/// ## Why strong types, and where they stop
///
/// The two defect classes this file exists to prevent are (a) passing a time
/// in *days* to a function expecting *year fractions*, and (b) passing a
/// variance where a volatility was wanted.  Both are silent, both produce
/// plausible-looking numbers, and both are expensive to find in a surface fit.
///
/// They are therefore used on **public API boundaries only**.  Inside batch
/// kernels the data is raw `double` in SoA columns: a kernel that has already
/// validated its inputs gains nothing from re-wrapping each lane, and the
/// wrapper would obstruct the `-Wconversion`-clean casts the SIMD paths need.
/// The conversion happens once, at the edge.
///
/// `Scalar<Tag>` is a trivially-copyable, standard-layout wrapper over
/// `double` with explicit construction.  It compiles to a bare register on
/// every supported toolchain (verified in tests/unit/test_types.cpp via
/// `std::is_trivially_copyable` + size assertions, and by inspection of the
/// generated code for the pricing entry points).

#include <compare>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace vl {

// ---------------------------------------------------------------------------
// Strong scalar
// ---------------------------------------------------------------------------

template <class Tag>
class Scalar {
  public:
    using value_type = double;

    Scalar() = default;
    constexpr explicit Scalar(double v) noexcept : v_(v) {}

    [[nodiscard]] constexpr double value() const noexcept { return v_; }
    constexpr explicit operator double() const noexcept { return v_; }

    friend constexpr auto operator<=>(Scalar a, Scalar b) noexcept {
        return a.v_ <=> b.v_;
    }
    friend constexpr bool operator==(Scalar a, Scalar b) noexcept {
        return a.v_ == b.v_;
    }

    // Additive structure is meaningful for every tag we define (a shift of a
    // rate is a rate, a shift of a log-moneyness is a log-moneyness).
    friend constexpr Scalar operator+(Scalar a, Scalar b) noexcept { return Scalar{a.v_ + b.v_}; }
    friend constexpr Scalar operator-(Scalar a, Scalar b) noexcept { return Scalar{a.v_ - b.v_}; }
    friend constexpr Scalar operator-(Scalar a) noexcept { return Scalar{-a.v_}; }
    constexpr Scalar& operator+=(Scalar b) noexcept { v_ += b.v_; return *this; }
    constexpr Scalar& operator-=(Scalar b) noexcept { v_ -= b.v_; return *this; }

    // Scaling by a dimensionless factor is meaningful; multiplying two tagged
    // values is not, and is deliberately not provided.
    friend constexpr Scalar operator*(Scalar a, double k) noexcept { return Scalar{a.v_ * k}; }
    friend constexpr Scalar operator*(double k, Scalar a) noexcept { return Scalar{k * a.v_}; }
    friend constexpr Scalar operator/(Scalar a, double k) noexcept { return Scalar{a.v_ / k}; }
    friend constexpr double operator/(Scalar a, Scalar b) noexcept { return a.v_ / b.v_; }

  private:
    double v_{};
};

namespace tags {
struct Strike {};
struct Spot {};
struct Forward {};
struct Years {};        ///< time in year fractions (ACT/365F unless stated)
struct Vol {};          ///< annualised lognormal volatility
struct Variance {};     ///< vol^2
struct TotalVariance {}; ///< vol^2 * T  (the natural surface coordinate)
struct Rate {};         ///< continuously-compounded zero rate
struct LogMoneyness {}; ///< log(K / F)
struct Money {};        ///< a price or PV in currency units
}  // namespace tags

using Strike        = Scalar<tags::Strike>;
using Spot          = Scalar<tags::Spot>;
using Forward       = Scalar<tags::Forward>;
using Years         = Scalar<tags::Years>;
using Vol           = Scalar<tags::Vol>;
using Variance      = Scalar<tags::Variance>;
using TotalVariance = Scalar<tags::TotalVariance>;
using Rate          = Scalar<tags::Rate>;
using LogMoneyness  = Scalar<tags::LogMoneyness>;
using Money         = Scalar<tags::Money>;

static_assert(sizeof(Strike) == sizeof(double));
static_assert(std::is_trivially_copyable_v<Strike>);
static_assert(std::is_standard_layout_v<Strike>);

/// Day-count conversion.  Named so that the call site reads as the assertion
/// it is: `years_from_days(45)` cannot be mistaken for `Years{45}`.
inline constexpr double kDaysPerYearAct365F = 365.0;
[[nodiscard]] constexpr Years years_from_days(double days) noexcept {
    return Years{days / kDaysPerYearAct365F};
}

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

/// Option payoff direction.  The numeric values are the payoff *sign* so that
/// kernels can use `static_cast<double>(static_cast<int>(type))` as the
/// branch-free multiplier `omega` in the generalised Black formula
///     price = omega * DF * (omega*F*N(omega*d1) - omega*K*N(omega*d2))
/// which collapses to the usual call/put forms.  This is the single reason the
/// enum is not the conventional {0,1}: it removes a branch from the innermost
/// loop of every batch pricer.
enum class OptionType : std::int8_t { Call = 1, Put = -1 };

[[nodiscard]] constexpr double payoff_sign(OptionType t) noexcept {
    return static_cast<double>(static_cast<std::int8_t>(t));
}

[[nodiscard]] constexpr OptionType opposite(OptionType t) noexcept {
    return t == OptionType::Call ? OptionType::Put : OptionType::Call;
}

[[nodiscard]] constexpr const char* to_string(OptionType t) noexcept {
    return t == OptionType::Call ? "call" : "put";
}

/// Exercise style.  Only European is priced analytically; the enum exists so
/// that normalisation can *reject* American quotes rather than silently price
/// them with a European formula.
enum class ExerciseStyle : std::int8_t { European = 0, American = 1 };

/// Which side of the market a price refers to.
enum class QuoteSide : std::int8_t { Bid = 0, Ask = 1, Mid = 2, Last = 3 };

[[nodiscard]] constexpr const char* to_string(QuoteSide s) noexcept {
    switch (s) {
        case QuoteSide::Bid:  return "bid";
        case QuoteSide::Ask:  return "ask";
        case QuoteSide::Mid:  return "mid";
        case QuoteSide::Last: return "last";
    }
    return "?";
}

}  // namespace vl

template <class Tag>
struct std::hash<vl::Scalar<Tag>> {
    std::size_t operator()(vl::Scalar<Tag> s) const noexcept {
        return std::hash<double>{}(s.value());
    }
};
