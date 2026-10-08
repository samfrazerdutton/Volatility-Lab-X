// SPDX-License-Identifier: MIT
#pragma once
/// \file alloc_counter.hpp
/// \brief Benchmark-only heap allocation counting.
///
/// Overrides the global `operator new`/`operator delete` -- but this header
/// and its `.cpp` are linked into exactly one benchmark executable
/// (`bench_incremental_scaling`, see `benchmarks/CMakeLists.txt`), not into
/// `volatility_lab` itself, the test suite, or any other benchmark. Every
/// other binary in this project gets the ordinary, un-instrumented
/// allocator. This is deliberately scoped instrumentation of one
/// executable, not a process-wide hook on the whole codebase.

#include <atomic>
#include <cstdint>

namespace vl::benchutil {

struct AllocCounters {
    static std::atomic<std::uint64_t> alloc_count;
    static std::atomic<std::uint64_t> alloc_bytes;
    static std::atomic<std::uint64_t> free_count;
};

/// RAII snapshot: the allocation count/bytes attributable to whatever ran
/// between construction and `stop()`. Measures real counts via the global
/// override above -- never a guess, never a "should be zero" assertion
/// without having actually counted.
class AllocScope {
  public:
    AllocScope() noexcept
        : start_count_(AllocCounters::alloc_count.load(std::memory_order_relaxed)),
          start_bytes_(AllocCounters::alloc_bytes.load(std::memory_order_relaxed)) {}

    void stop() noexcept {
        if (stopped_) return;
        allocations_ =
            AllocCounters::alloc_count.load(std::memory_order_relaxed) - start_count_;
        bytes_ = AllocCounters::alloc_bytes.load(std::memory_order_relaxed) - start_bytes_;
        stopped_ = true;
    }

    [[nodiscard]] std::uint64_t allocations() const noexcept { return allocations_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

  private:
    std::uint64_t start_count_;
    std::uint64_t start_bytes_;
    std::uint64_t allocations_ = 0;
    std::uint64_t bytes_ = 0;
    bool stopped_ = false;
};

}  // namespace vl::benchutil
