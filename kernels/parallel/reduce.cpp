// SPDX-License-Identifier: MIT
#include "parallel/reduce.hpp"

#include <algorithm>
#include <future>
#include <mutex>
#include <utility>
#include <vector>

namespace vl::kernels::parallel {

namespace {

/// The exact chunk partition `ThreadPool::parallel_for` uses internally:
/// `[0, n)` split into `num_chunks` contiguous ranges, the first
/// `n % num_chunks` of them one element larger. Duplicated here (not
/// reused from `ThreadPool::parallel_for` directly) so each chunk's index
/// is known by construction -- via `submit`, not by trying to invert the
/// boundary formula back from a `(start, end)` pair `parallel_for` handed
/// back with no index attached.
std::vector<std::pair<std::size_t, std::size_t>> chunk_ranges(std::size_t n,
                                                               std::size_t num_chunks) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(num_chunks);
    const std::size_t base = n / num_chunks;
    const std::size_t remainder = n % num_chunks;
    std::size_t start = 0;
    for (std::size_t c = 0; c < num_chunks; ++c) {
        const std::size_t size = base + (c < remainder ? 1 : 0);
        ranges.emplace_back(start, start + size);
        start += size;
    }
    return ranges;
}

}  // namespace

double reduce_deterministic(ThreadPool& pool, std::size_t n, std::size_t num_chunks,
                            const std::function<double(std::size_t, std::size_t)>& per_chunk) {
    if (n == 0 || num_chunks == 0) return 0.0;
    num_chunks = std::min(num_chunks, n);
    const auto ranges = chunk_ranges(n, num_chunks);

    std::vector<std::future<double>> futures;
    futures.reserve(num_chunks);
    for (const auto& [start, end] : ranges) {
        futures.push_back(pool.submit(per_chunk, start, end));
    }

    // Combine in chunk-index order -- fixed, regardless of which future
    // happens to be ready first; .get() on an already-finished future
    // returns immediately, so this does not serialise the actual work,
    // only the combine step, which is the one part that must be ordered.
    double total = 0.0;
    for (auto& f : futures) total += f.get();
    return total;
}

double reduce_fast(ThreadPool& pool, std::size_t n, std::size_t num_chunks,
                   const std::function<double(std::size_t, std::size_t)>& per_chunk) {
    if (n == 0 || num_chunks == 0) return 0.0;
    double total = 0.0;
    std::mutex m;
    pool.parallel_for(n, num_chunks, [&](std::size_t start, std::size_t end) {
        const double partial = per_chunk(start, end);
        std::lock_guard<std::mutex> lock(m);
        total += partial;  // combine order = whichever chunk finishes first
    });
    return total;
}

}  // namespace vl::kernels::parallel
