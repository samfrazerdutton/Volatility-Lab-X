// SPDX-License-Identifier: MIT
/// Pins the properties that make the strong scalar types free: they must be
/// trivially copyable, standard layout, and exactly the size of a double, or
/// the "zero cost at the API boundary" claim in core/types.hpp is false.

#include "vl_test_support.hpp"

#include "volatility_lab/core/types.hpp"

#include <string>
#include <type_traits>
#include <unordered_map>

using namespace vl;

TEST(Types, StrongScalarsAreZeroCost) {
    static_assert(sizeof(Strike) == sizeof(double));
    static_assert(sizeof(Years) == sizeof(double));
    static_assert(alignof(Strike) == alignof(double));
    static_assert(std::is_trivially_copyable_v<Vol>);
    static_assert(std::is_standard_layout_v<Vol>);
    static_assert(std::is_trivially_destructible_v<Vol>);
    SUCCEED();
}

TEST(Types, DistinctTagsDoNotConvert) {
    // The whole point: a Years cannot be passed where a Vol is expected, and a
    // bare double cannot be passed as either.
    static_assert(!std::is_convertible_v<Years, Vol>);
    static_assert(!std::is_convertible_v<double, Years>);
    static_assert(std::is_constructible_v<Years, double>);
    SUCCEED();
}

TEST(Types, ArithmeticIsDimensionallySensible) {
    const Rate a{0.03};
    const Rate b{0.01};
    EXPECT_EQ((a + b).value(), 0.04);
    EXPECT_NEAR((a - b).value(), 0.02, 1e-17);
    EXPECT_EQ((-a).value(), -0.03);
    EXPECT_EQ((a * 2.0).value(), 0.06);
    EXPECT_EQ((2.0 * a).value(), 0.06);
    EXPECT_EQ((a / 2.0).value(), 0.015);
    EXPECT_NEAR(a / b, 3.0, 1e-15);  // ratio of like quantities is dimensionless

    Rate c{0.05};
    c += b;
    EXPECT_NEAR(c.value(), 0.06, 1e-17);
    c -= b;
    EXPECT_NEAR(c.value(), 0.05, 1e-17);
}

TEST(Types, ComparisonAndOrdering) {
    EXPECT_TRUE(Strike{90.0} < Strike{100.0});
    EXPECT_TRUE(Strike{100.0} == Strike{100.0});
    EXPECT_TRUE(Strike{110.0} > Strike{100.0});
    EXPECT_FALSE(Strike{100.0} != Strike{100.0});
    EXPECT_TRUE(Strike{100.0} <= Strike{100.0});
    EXPECT_TRUE(Strike{100.0} >= Strike{100.0});
}

TEST(Types, HashableForUseAsMapKeys) {
    std::unordered_map<Years, int> m;
    m[Years{0.25}] = 1;
    m[Years{0.5}] = 2;
    EXPECT_EQ(m.at(Years{0.25}), 1);
    EXPECT_EQ(m.size(), 2u);
}

TEST(Types, DayCountConversionIsNamedToPreventTheClassicBug) {
    // years_from_days(45) must not be confusable with Years{45}; that
    // confusion is the defect the whole strong-type layer exists to prevent.
    EXPECT_NEAR(years_from_days(365.0).value(), 1.0, 1e-16);
    EXPECT_NEAR(years_from_days(45.0).value(), 45.0 / 365.0, 1e-16);
    EXPECT_TRUE(years_from_days(45.0) != Years{45.0});
}

TEST(Types, PayoffSignEncodesTheBranchFreeMultiplier) {
    // The enum's numeric values are the payoff sign precisely so the batch
    // kernels can avoid a branch.  If someone renumbers it, this fails.
    EXPECT_EQ(static_cast<int>(OptionType::Call), 1);
    EXPECT_EQ(static_cast<int>(OptionType::Put), -1);
    EXPECT_EQ(payoff_sign(OptionType::Call), 1.0);
    EXPECT_EQ(payoff_sign(OptionType::Put), -1.0);
    EXPECT_EQ(opposite(OptionType::Call), OptionType::Put);
    EXPECT_EQ(opposite(OptionType::Put), OptionType::Call);
    EXPECT_STREQ(to_string(OptionType::Call), "call");
    EXPECT_STREQ(to_string(OptionType::Put), "put");
}

// ---------------------------------------------------------------------------
// InstrumentKey
// ---------------------------------------------------------------------------

TEST(InstrumentKey, EqualityComparesEveryField) {
    const InstrumentKey a{Years{0.25}, Strike{100.0}, OptionType::Call};
    const InstrumentKey same{Years{0.25}, Strike{100.0}, OptionType::Call};
    const InstrumentKey diff_years{Years{0.5}, Strike{100.0}, OptionType::Call};
    const InstrumentKey diff_strike{Years{0.25}, Strike{110.0}, OptionType::Call};
    const InstrumentKey diff_type{Years{0.25}, Strike{100.0}, OptionType::Put};

    EXPECT_TRUE(a == same);
    EXPECT_FALSE(a == diff_years);
    EXPECT_FALSE(a == diff_strike);
    EXPECT_FALSE(a == diff_type);
}

TEST(InstrumentKey, IsUsableAsAnUnorderedMapKeyWithNoStringInvolved) {
    // The whole point: this must work with zero std::string construction --
    // the type itself is what proves that (double, double, enum) fields
    // only, no string member to accidentally allocate from.
    static_assert(!std::is_constructible_v<InstrumentKey, std::string, double, OptionType>);
    std::unordered_map<InstrumentKey, int> m;
    m[InstrumentKey{Years{0.25}, Strike{100.0}, OptionType::Call}] = 1;
    m[InstrumentKey{Years{0.25}, Strike{105.0}, OptionType::Call}] = 2;
    m[InstrumentKey{Years{1.0}, Strike{100.0}, OptionType::Put}] = 3;
    EXPECT_EQ(m.size(), 3u);
    EXPECT_EQ(m.at((InstrumentKey{Years{0.25}, Strike{100.0}, OptionType::Call})), 1);
}

TEST(InstrumentKey, DistinctKeysThatDifferOnlyByOptionTypeHashDifferently) {
    // Not a correctness requirement on its own (a hash collision is always
    // legal), but confirms the type field is actually mixed into the hash
    // at all, rather than the implementation accidentally hashing only
    // (years, strike) and relying on equality alone to disambiguate type.
    const InstrumentKey call{Years{0.25}, Strike{100.0}, OptionType::Call};
    const InstrumentKey put{Years{0.25}, Strike{100.0}, OptionType::Put};
    EXPECT_FALSE(call == put);
    EXPECT_NE(std::hash<InstrumentKey>{}(call), std::hash<InstrumentKey>{}(put));
}

TEST(InstrumentKey, IsTriviallyCopyableAndStandardLayout) {
    static_assert(std::is_trivially_copyable_v<InstrumentKey>);
    static_assert(std::is_standard_layout_v<InstrumentKey>);
    SUCCEED();
}

TEST(Types, QuoteSideStrings) {
    EXPECT_STREQ(to_string(QuoteSide::Bid), "bid");
    EXPECT_STREQ(to_string(QuoteSide::Ask), "ask");
    EXPECT_STREQ(to_string(QuoteSide::Mid), "mid");
    EXPECT_STREQ(to_string(QuoteSide::Last), "last");
}
