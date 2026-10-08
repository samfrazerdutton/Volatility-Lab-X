// SPDX-License-Identifier: MIT
/// Validates the thread pool: task submission and exception propagation,
/// and parallel_for's correctness properties -- every index covered
/// exactly once, fixed chunk boundaries independent of pool size (the
/// determinism claim this is actually for), and that one chunk's
/// exception surfaces to the caller only after every chunk has finished.

#include "vl_test_support.hpp"

#include "volatility_lab/core/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace vl;

// ---------------------------------------------------------------------------
// submit
// ---------------------------------------------------------------------------

TEST(ThreadPool, NumThreadsReportsWhatWasRequested) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.num_threads(), 4u);
}

TEST(ThreadPool, ZeroRequestedThreadsIsClampedToOne) {
    ThreadPool pool(0);
    EXPECT_EQ(pool.num_threads(), 1u);
}

TEST(ThreadPool, SubmitReturnsTheTaskResult) {
    ThreadPool pool(2);
    auto f = pool.submit([](int a, int b) { return a + b; }, 3, 4);
    EXPECT_EQ(f.get(), 7);
}

TEST(ThreadPool, SubmitPropagatesAnExceptionToTheFuture) {
    ThreadPool pool(2);
    auto f = pool.submit([]() -> int { throw std::runtime_error("boom"); });
    EXPECT_THROW(f.get(), std::runtime_error);
}

TEST(ThreadPool, ManySmallTasksAllCompleteWithTheRightResult) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 200; ++i) {
        futures.push_back(pool.submit([](int x) { return x * x; }, i));
    }
    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(futures[static_cast<std::size_t>(i)].get(), i * i);
    }
}

// ---------------------------------------------------------------------------
// parallel_for: coverage and correctness
// ---------------------------------------------------------------------------

TEST(ThreadPool, ParallelForVisitsEveryIndexExactlyOnce) {
    ThreadPool pool(4);
    constexpr std::size_t n = 10007;  // prime, deliberately not divisible by chunk counts
    std::vector<std::atomic<int>> visits(n);
    for (auto& v : visits) v = 0;

    pool.parallel_for(n, 8, [&](std::size_t start, std::size_t end) {
        for (std::size_t i = start; i < end; ++i) ++visits[i];
    });

    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(visits[i].load(), 1) << "index " << i;
    }
}

TEST(ThreadPool, ParallelForSumMatchesSequentialSum) {
    ThreadPool pool(4);
    constexpr std::size_t n = 100003;
    std::vector<double> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<double>(i);

    std::atomic<double> sum{0.0};
    pool.parallel_for(n, 8, [&](std::size_t start, std::size_t end) {
        double local = 0.0;
        for (std::size_t i = start; i < end; ++i) local += data[i];
        double old = sum.load();
        while (!sum.compare_exchange_weak(old, old + local)) {
        }
    });

    const double expected = std::accumulate(data.begin(), data.end(), 0.0);
    EXPECT_DOUBLE_EQ(sum.load(), expected);
}

TEST(ThreadPool, ChunkBoundariesDependOnlyOnNAndNumChunksNotPoolSize) {
    // The determinism claim this exists for: the split must be identical
    // regardless of how many worker threads actually process it.
    constexpr std::size_t n = 10007;
    constexpr std::size_t num_chunks = 7;

    auto record_boundaries = [&](std::size_t threads) {
        ThreadPool pool(threads);
        std::vector<std::pair<std::size_t, std::size_t>> boundaries(num_chunks);
        std::atomic<std::size_t> next{0};
        pool.parallel_for(n, num_chunks, [&](std::size_t start, std::size_t end) {
            const std::size_t idx = next.fetch_add(1);
            boundaries[idx] = {start, end};
        });
        std::sort(boundaries.begin(), boundaries.end());
        return boundaries;
    };

    const auto with_1 = record_boundaries(1);
    const auto with_4 = record_boundaries(4);
    const auto with_8 = record_boundaries(8);
    EXPECT_EQ(with_1, with_4);
    EXPECT_EQ(with_1, with_8);
}

TEST(ThreadPool, ChunksCoverTheFullRangeContiguouslyWithNoGapOrOverlap) {
    ThreadPool pool(4);
    constexpr std::size_t n = 10007;
    constexpr std::size_t num_chunks = 8;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::mutex m;
    pool.parallel_for(n, num_chunks, [&](std::size_t start, std::size_t end) {
        std::lock_guard<std::mutex> lock(m);
        ranges.emplace_back(start, end);
    });
    std::sort(ranges.begin(), ranges.end());
    ASSERT_EQ(ranges.size(), num_chunks);
    EXPECT_EQ(ranges.front().first, 0u);
    EXPECT_EQ(ranges.back().second, n);
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        EXPECT_EQ(ranges[i - 1].second, ranges[i].first) << "gap/overlap before chunk " << i;
    }
}

// ---------------------------------------------------------------------------
// parallel_for: exceptions and degenerate inputs
// ---------------------------------------------------------------------------

TEST(ThreadPool, ParallelForPropagatesAnExceptionFromAnyChunk) {
    ThreadPool pool(4);
    EXPECT_THROW(
        pool.parallel_for(100, 4,
                          [](std::size_t start, std::size_t) {
                              if (start == 0) throw std::logic_error("chunk 0 failed");
                          }),
        std::logic_error);
}

TEST(ThreadPool, ParallelForWaitsForEveryChunkBeforeRethrowing) {
    // Every chunk's side effect must be visible after parallel_for returns
    // (via exception or otherwise) -- a thrown exception must not skip
    // waiting on chunks that have not thrown.
    ThreadPool pool(4);
    constexpr std::size_t num_chunks = 6;
    std::atomic<int> completed{0};
    EXPECT_THROW(pool.parallel_for(60, num_chunks,
                                  [&](std::size_t start, std::size_t) {
                                      if (start == 0) throw std::runtime_error("boom");
                                      ++completed;
                                  }),
                std::runtime_error);
    EXPECT_EQ(completed.load(), static_cast<int>(num_chunks) - 1);
}

TEST(ThreadPool, ZeroElementsIsANoOp) {
    ThreadPool pool(4);
    bool ran = false;
    pool.parallel_for(0, 4, [&](std::size_t, std::size_t) { ran = true; });
    EXPECT_FALSE(ran);
}

TEST(ThreadPool, ZeroChunksIsANoOp) {
    ThreadPool pool(4);
    bool ran = false;
    pool.parallel_for(10, 0, [&](std::size_t, std::size_t) { ran = true; });
    EXPECT_FALSE(ran);
}

TEST(ThreadPool, RequestingMoreChunksThanElementsClampsRatherThanProducingEmptyChunks) {
    ThreadPool pool(4);
    std::atomic<int> chunk_count{0};
    pool.parallel_for(3, 100, [&](std::size_t start, std::size_t end) {
        ++chunk_count;
        EXPECT_LT(start, end) << "no chunk should be empty";
    });
    EXPECT_EQ(chunk_count.load(), 3);
}
