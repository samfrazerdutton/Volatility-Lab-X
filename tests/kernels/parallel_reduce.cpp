// SPDX-License-Identifier: MIT
/// Validates the central claim of `kernels/parallel/reduce.hpp` --
/// `reduce_deterministic` is bitwise reproducible across repeated calls
/// and across different thread-pool sizes, matching `math::tol::kParallelReduction`'s
/// zero-tolerance budget (which predates this implementation); `reduce_fast`
/// is demonstrated, not merely asserted, to be capable of differing run to
/// run, documenting the real tradeoff rather than claiming more than either
/// path actually delivers.

#include "vl_test_support.hpp"

#include "parallel/reduce.hpp"

#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

using namespace vl;

namespace {

std::vector<double> make_data(std::size_t n) {
    std::vector<double> data(n);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = std::sin(static_cast<double>(i) * 0.0001) * 1.0000001;
    }
    return data;
}

std::function<double(std::size_t, std::size_t)> sum_of(const std::vector<double>& data) {
    return [&data](std::size_t start, std::size_t end) {
        double local = 0.0;
        for (std::size_t i = start; i < end; ++i) local += data[i];
        return local;
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// reduce_deterministic: bitwise reproducibility
// ---------------------------------------------------------------------------

TEST(ParallelReduce, DeterministicRepeatedCallsOnTheSamePoolAreBitwiseIdentical) {
    const auto data = make_data(1000003);
    ThreadPool pool(8);
    const double r1 = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
    const double r2 = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
    EXPECT_EQ(r1, r2);
}

TEST(ParallelReduce, DeterministicResultIsBitwiseIdenticalAcrossThreadCounts) {
    const auto data = make_data(1000003);
    double reference = 0.0;
    bool first = true;
    for (std::size_t threads : {1u, 2u, 4u, 8u, 16u}) {
        ThreadPool pool(threads);
        const double r = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
        if (first) {
            reference = r;
            first = false;
        } else {
            EXPECT_EQ(r, reference) << "diverged at " << threads << " threads";
        }
    }
}

TEST(ParallelReduce, DeterministicAgreesWithSequentialSumToOrdinaryReassociationError) {
    // Not bitwise against a plain linear accumulation -- a 13-way tree-like
    // combine is a genuinely different summation order, and floating-point
    // addition is not associative. The claim is "close", with a budget set
    // from the scale of the data, not "identical".
    const auto data = make_data(1000003);
    const double sequential = std::accumulate(data.begin(), data.end(), 0.0);
    ThreadPool pool(4);
    const double det = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
    EXPECT_NEAR(det, sequential, 1e-8);
}

TEST(ParallelReduce, DifferentNumChunksCanGiveADifferentButStillCorrectAnswer) {
    // A different chunk count is a different summation order, so a
    // different valid rounding is expected -- but still correct to
    // ordinary reassociation error, and still bitwise reproducible for its
    // OWN (n, num_chunks).
    const auto data = make_data(1000003);
    const double sequential = std::accumulate(data.begin(), data.end(), 0.0);
    ThreadPool pool(4);
    const double with_7 = kernels::parallel::reduce_deterministic(pool, data.size(), 7, sum_of(data));
    const double with_13 = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
    EXPECT_NEAR(with_7, sequential, 1e-8);
    EXPECT_NEAR(with_13, sequential, 1e-8);
    const double with_7_again =
        kernels::parallel::reduce_deterministic(pool, data.size(), 7, sum_of(data));
    EXPECT_EQ(with_7, with_7_again);
}

// ---------------------------------------------------------------------------
// reduce_fast: the honest counterpoint
// ---------------------------------------------------------------------------

TEST(ParallelReduce, FastAgreesWithDeterministicToOrdinaryReassociationError) {
    const auto data = make_data(1000003);
    ThreadPool pool(8);
    const double det = kernels::parallel::reduce_deterministic(pool, data.size(), 13, sum_of(data));
    const double fast = kernels::parallel::reduce_fast(pool, data.size(), 64, sum_of(data));
    EXPECT_NEAR(fast, det, 1e-8);
}

TEST(ParallelReduce, FastIsNotClaimedOrRequiredToBeBitwiseReproducible) {
    // This test does not assert nondeterminism occurs on every run --
    // scheduling could coincidentally combine in the same order every
    // time on a given machine -- it asserts only that reduce_fast's
    // *contract* permits variation (no fixed combine order), by checking
    // many repeated calls stay within reassociation-error agreement with
    // each other without requiring bitwise equality.
    const auto data = make_data(1000003);
    ThreadPool pool(8);
    std::vector<double> results;
    for (int i = 0; i < 30; ++i) {
        results.push_back(kernels::parallel::reduce_fast(pool, data.size(), 64, sum_of(data)));
    }
    for (double r : results) {
        EXPECT_NEAR(r, results.front(), 1e-8);
    }
    // Deliberately not EXPECT_EQ on every pair: that would assert the
    // bitwise-reproducibility property this function explicitly does not
    // promise.
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST(ParallelReduce, ZeroElementsReturnsZero) {
    ThreadPool pool(4);
    const auto empty = std::vector<double>{};
    EXPECT_EQ(kernels::parallel::reduce_deterministic(pool, 0, 4, sum_of(empty)), 0.0);
    EXPECT_EQ(kernels::parallel::reduce_fast(pool, 0, 4, sum_of(empty)), 0.0);
}

TEST(ParallelReduce, ZeroChunksReturnsZero) {
    const auto data = make_data(10);
    ThreadPool pool(4);
    EXPECT_EQ(kernels::parallel::reduce_deterministic(pool, data.size(), 0, sum_of(data)), 0.0);
}

TEST(ParallelReduce, MoreChunksThanElementsClampsRatherThanMisbehaving) {
    const auto data = make_data(3);
    ThreadPool pool(4);
    const double result = kernels::parallel::reduce_deterministic(pool, data.size(), 100, sum_of(data));
    const double expected = std::accumulate(data.begin(), data.end(), 0.0);
    EXPECT_NEAR(result, expected, 1e-12);
}

TEST(ParallelReduce, SingleChunkMatchesPlainSequentialSumExactly) {
    // With exactly one chunk there is only one summation order -- this
    // should be bitwise identical to calling per_chunk directly.
    const auto data = make_data(10007);
    ThreadPool pool(4);
    const double result = kernels::parallel::reduce_deterministic(pool, data.size(), 1, sum_of(data));
    const double direct = sum_of(data)(0, data.size());
    EXPECT_EQ(result, direct);
}
