// SPDX-License-Identifier: MIT
/// Pins the three properties AlignedBuffer exists for: over-alignment, vector
/// tail padding, and reuse of the allocation on shrink.  Each is load-bearing
/// for a claim made elsewhere -- the SIMD kernels assume aligned loads with no
/// prologue, and the determinism contract assumes the chunk decomposition does
/// not depend on the allocation address.

#include "vl_test_support.hpp"

#include "volatility_lab/core/aligned_buffer.hpp"

#include <cstdint>
#include <numeric>
#include <utility>

using namespace vl;

TEST(AlignedBuffer, BaseIsOverAligned) {
    for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{64},
                          std::size_t{1000}, std::size_t{1000003}}) {
        AlignedBuffer<double> b(n);
        const auto addr = reinterpret_cast<std::uintptr_t>(b.data());
        EXPECT_EQ(addr % kSimdAlign, 0u) << "n = " << n;
        EXPECT_EQ(b.size(), n);
    }
}

TEST(AlignedBuffer, CapacityIsRoundedToWholeVectorRegisters) {
    // The SIMD kernels process ceil(n/W) registers with no masked epilogue,
    // which is only defined behaviour if the padding is really allocated and
    // really initialised.
    for (std::size_t lanes : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        for (std::size_t n : {std::size_t{1}, std::size_t{5}, std::size_t{17},
                              std::size_t{100}}) {
            AlignedBuffer<double> b(n, lanes, -1.0);
            EXPECT_EQ(b.capacity() % lanes, 0u) << "n=" << n << " lanes=" << lanes;
            EXPECT_GE(b.capacity(), n);
            EXPECT_LT(b.capacity() - n, lanes);
            for (std::size_t i = n; i < b.capacity(); ++i) {
                EXPECT_EQ(b.data()[i], -1.0) << "pad lane " << i << " not initialised";
            }
        }
    }
}

TEST(AlignedBuffer, EmptyBufferIsNullAndHarmless) {
    AlignedBuffer<double> b;
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_EQ(b.size(), 0u);
    EXPECT_TRUE(b.empty());
    b.fill(1.0);  // must not dereference null
    EXPECT_EQ(b.view().size(), 0u);

    AlignedBuffer<double> z(0);
    EXPECT_EQ(z.data(), nullptr);
    EXPECT_EQ(z.capacity(), 0u);
}

TEST(AlignedBuffer, MoveTransfersOwnership) {
    AlignedBuffer<double> a(100);
    a.fill(3.0);
    const double* p = a.data();
    AlignedBuffer<double> b(std::move(a));
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(b.size(), 100u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);

    AlignedBuffer<double> c(5);
    c = std::move(b);
    EXPECT_EQ(c.data(), p);
    EXPECT_EQ(b.data(), nullptr);
}

TEST(AlignedBuffer, CopyIsDeepAndPreservesAlignment) {
    AlignedBuffer<double> a(37);
    std::iota(a.begin(), a.end(), 1.0);
    AlignedBuffer<double> b(a);
    ASSERT_EQ(b.size(), a.size());
    EXPECT_NE(b.data(), a.data());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % kSimdAlign, 0u);
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(b[i], a[i]);

    AlignedBuffer<double> c(2);
    c = a;
    ASSERT_EQ(c.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(c[i], a[i]);
}

TEST(AlignedBuffer, ResizeReusesTheAllocationWhenItCan) {
    // Matters for the streaming mode, which resizes once per market update;
    // reallocating there would put malloc on the latency path.
    AlignedBuffer<double> b(1000);
    const double* p = b.data();
    const std::size_t cap = b.capacity();
    b.resize(500);
    EXPECT_EQ(b.data(), p) << "shrinking must not reallocate";
    EXPECT_EQ(b.size(), 500u);
    EXPECT_EQ(b.capacity(), cap);
    b.resize(100000);
    EXPECT_EQ(b.size(), 100000u);
    EXPECT_GE(b.capacity(), 100000u);
}

TEST(AlignedBuffer, ResizePadsTheNewTail) {
    AlignedBuffer<double> b(1000, 4, -7.0);
    b.resize(13, 4, -7.0);
    for (std::size_t i = 13; i < b.capacity(); ++i) EXPECT_EQ(b.data()[i], -7.0);
}

TEST(AlignedBuffer, SpanViewsMatchTheBuffer) {
    AlignedBuffer<double> b(10);
    b.fill(2.0);
    EXPECT_EQ(b.view().size(), 10u);
    EXPECT_EQ(b.padded_view().size(), b.capacity());
    EXPECT_EQ(b.view().data(), b.data());
    EXPECT_EQ(static_cast<std::size_t>(b.end() - b.begin()), b.size());
}

TEST(CachePadded, OccupiesWholeCacheLines) {
    // False sharing between per-thread accumulators is the easiest scaling bug
    // to introduce and the hardest to see in a profile.  This is the
    // structural guard.
    EXPECT_EQ(sizeof(CachePadded<double>) % kCacheLine, 0u);
    EXPECT_GE(sizeof(CachePadded<double>), kCacheLine);
    EXPECT_EQ(alignof(CachePadded<double>), kCacheLine);

    struct Big {
        double a[20];
    };
    EXPECT_EQ(sizeof(CachePadded<Big>) % kCacheLine, 0u);

    CachePadded<double> arr[8];
    for (int i = 1; i < 8; ++i) {
        const auto d = reinterpret_cast<std::uintptr_t>(&arr[i]) -
                       reinterpret_cast<std::uintptr_t>(&arr[i - 1]);
        EXPECT_EQ(d % kCacheLine, 0u);
    }
}

TEST(Config, RoundUpIsCorrectForPowersOfTwo) {
    EXPECT_EQ(round_up(0, 8), 0u);
    EXPECT_EQ(round_up(1, 8), 8u);
    EXPECT_EQ(round_up(8, 8), 8u);
    EXPECT_EQ(round_up(9, 8), 16u);
    EXPECT_EQ(round_up(100, 64), 128u);
}
