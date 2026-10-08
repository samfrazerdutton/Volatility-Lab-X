// SPDX-License-Identifier: MIT
#include "alloc_counter.hpp"

#include <cstdlib>
#include <new>

namespace vl::benchutil {

std::atomic<std::uint64_t> AllocCounters::alloc_count{0};
std::atomic<std::uint64_t> AllocCounters::alloc_bytes{0};
std::atomic<std::uint64_t> AllocCounters::free_count{0};

}  // namespace vl::benchutil

// Global override, scoped to this one executable -- see alloc_counter.hpp.
// Only the plain and sized/array delete forms are overridden; anything else
// (nothrow, aligned) falls back to the default, which this benchmark's
// workload never exercises.
void* operator new(std::size_t n) {
    vl::benchutil::AllocCounters::alloc_count.fetch_add(1, std::memory_order_relaxed);
    vl::benchutil::AllocCounters::alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}

void* operator new[](std::size_t n) { return ::operator new(n); }

void operator delete(void* p) noexcept {
    vl::benchutil::AllocCounters::free_count.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
}

void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
