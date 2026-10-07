// SPDX-License-Identifier: MIT
#include "vl_test_support.hpp"

#include "volatility_lab/core/expected.hpp"

#include <string>
#include <type_traits>

using namespace vl;

namespace {
enum class Err { Bad, Worse };
}

TEST(Expected, HoldsAValue) {
    Expected<int, Err> e{42};
    ASSERT_TRUE(e.has_value());
    EXPECT_TRUE(static_cast<bool>(e));
    EXPECT_EQ(e.value(), 42);
    EXPECT_EQ(*e, 42);
    EXPECT_EQ(e.value_or(7), 42);
}

TEST(Expected, HoldsAnError) {
    Expected<int, Err> e{make_unexpected(Err::Worse)};
    ASSERT_FALSE(e.has_value());
    EXPECT_FALSE(static_cast<bool>(e));
    EXPECT_EQ(e.error(), Err::Worse);
    EXPECT_EQ(e.value_or(7), 7);
}

TEST(Expected, MapTransformsValueAndPropagatesError) {
    Expected<int, Err> ok{10};
    const auto doubled = ok.map([](int v) { return v * 2; });
    ASSERT_TRUE(doubled.has_value());
    EXPECT_EQ(doubled.value(), 20);

    Expected<int, Err> bad{make_unexpected(Err::Bad)};
    const auto still_bad = bad.map([](int v) { return v * 2; });
    ASSERT_FALSE(still_bad.has_value());
    EXPECT_EQ(still_bad.error(), Err::Bad);
}

TEST(Expected, MapCanChangeTheValueType) {
    Expected<int, Err> ok{3};
    const auto as_double = ok.map([](int v) { return v * 0.5; });
    ASSERT_TRUE(as_double.has_value());
    EXPECT_EQ(as_double.value(), 1.5);
}

TEST(Expected, WorksWithNonTrivialTypes) {
    Expected<std::string, Err> e{std::string("hello")};
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e.value(), "hello");
    EXPECT_EQ(e->size(), 5u);
}

TEST(Expected, DoubleErrorPairStaysCompact) {
    // The reason for having this type rather than a pair<bool, T>: the common
    // Expected<double, DiagCode> must stay small and trivially destructible,
    // because it is returned from functions called per quote.
    static_assert(std::is_trivially_destructible_v<Expected<double, Err>>);
    EXPECT_LE(sizeof(Expected<double, Err>), 24u);
}

TEST(Expected, DefaultConstructsToAValue) {
    Expected<int, Err> e;
    EXPECT_TRUE(e.has_value());
    EXPECT_EQ(e.value(), 0);
}
