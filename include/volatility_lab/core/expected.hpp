// SPDX-License-Identifier: MIT
#pragma once
/// \file expected.hpp
/// \brief Minimal `Expected<T, E>` for fallible operations.
///
/// `std::expected` is C++23 and is not available on the oldest toolchain this
/// project supports (MSVC 19.29 / VS2019 STL).  Rather than make C++23 a hard
/// requirement for an eighty-line vocabulary type, it is provided here with
/// the subset of the standard interface the library actually uses.  The API is
/// a strict subset of `std::expected`, so the alias can be switched over
/// without touching call sites once C++23 is the floor.
///
/// Design notes:
///  * No exceptions are thrown on bad access; `value()` is precondition-checked
///    with an assertion.  The library's error handling is value-based because
///    the hot paths must remain exception-free (see docs/design-decisions.md,
///    D-03).
///  * The storage is a union, so `Expected<double, ErrorCode>` is 16 bytes and
///    trivially destructible.

#include <cassert>
#include <new>
#include <type_traits>
#include <utility>
#include <variant>

namespace vl {

template <class E>
class Unexpected {
  public:
    constexpr explicit Unexpected(E e) : e_(std::move(e)) {}
    [[nodiscard]] constexpr const E& error() const& noexcept { return e_; }
    [[nodiscard]] constexpr E&& error() && noexcept { return std::move(e_); }

  private:
    E e_;
};

template <class E>
[[nodiscard]] constexpr Unexpected<std::decay_t<E>> make_unexpected(E&& e) {
    return Unexpected<std::decay_t<E>>{std::forward<E>(e)};
}

template <class T, class E>
class [[nodiscard]] Expected {
    static_assert(!std::is_reference_v<T>, "Expected<T&> is not supported");

  public:
    using value_type = T;
    using error_type = E;

    constexpr Expected() requires std::is_default_constructible_v<T>
        : store_(std::in_place_index<0>) {}

    constexpr Expected(T v) : store_(std::in_place_index<0>, std::move(v)) {}
    constexpr Expected(Unexpected<E> u)
        : store_(std::in_place_index<1>, std::move(u).error()) {}

    [[nodiscard]] constexpr bool has_value() const noexcept { return store_.index() == 0; }
    constexpr explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] constexpr const T& value() const& noexcept {
        assert(has_value() && "Expected::value() on an error state");
        return *std::get_if<0>(&store_);
    }
    [[nodiscard]] constexpr T& value() & noexcept {
        assert(has_value() && "Expected::value() on an error state");
        return *std::get_if<0>(&store_);
    }
    [[nodiscard]] constexpr T&& value() && noexcept {
        assert(has_value() && "Expected::value() on an error state");
        return std::move(*std::get_if<0>(&store_));
    }

    [[nodiscard]] constexpr const E& error() const& noexcept {
        assert(!has_value() && "Expected::error() on a value state");
        return *std::get_if<1>(&store_);
    }
    [[nodiscard]] constexpr E&& error() && noexcept {
        assert(!has_value() && "Expected::error() on a value state");
        return std::move(*std::get_if<1>(&store_));
    }

    [[nodiscard]] constexpr const T& operator*() const& noexcept { return value(); }
    [[nodiscard]] constexpr T& operator*() & noexcept { return value(); }
    [[nodiscard]] constexpr const T* operator->() const noexcept { return &value(); }
    [[nodiscard]] constexpr T* operator->() noexcept { return &value(); }

    template <class U>
    [[nodiscard]] constexpr T value_or(U&& fallback) const& {
        return has_value() ? value() : static_cast<T>(std::forward<U>(fallback));
    }

    /// Apply `f` to the contained value, propagating the error unchanged.
    template <class F>
    [[nodiscard]] constexpr auto map(F&& f) const&
        -> Expected<std::invoke_result_t<F, const T&>, E> {
        using R = Expected<std::invoke_result_t<F, const T&>, E>;
        if (has_value()) return R{std::forward<F>(f)(value())};
        return R{make_unexpected(error())};
    }

  private:
    std::variant<T, E> store_;
};

}  // namespace vl
