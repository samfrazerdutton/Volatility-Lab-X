// SPDX-License-Identifier: MIT
#pragma once
/// \file reduce.hpp
/// \brief Deterministic vs fast parallel reduction (directive section 11)
///        -- built directly on `core/thread_pool.hpp`'s fixed, dispatch-
///        order-independent chunk boundaries.
///
/// ## The property being demonstrated, not merely claimed
///
/// Floating-point addition is not associative, so a parallel sum's result
/// can depend on the *order* partial sums are combined in -- and that
/// order, if left to "whichever thread finishes first", depends on
/// scheduling, which is not reproducible run to run. `math::tol::kParallelReduction`
/// (`math/compare.hpp`) already states the fix before any implementation
/// existed to check it against: "fixed chunking + in-order combine =>
/// bitwise", with a tolerance of exactly zero -- not "small", zero, so a
/// regression is impossible to paper over with a slightly looser budget.
///
/// `reduce_deterministic` is exactly that: `ThreadPool::parallel_for`'s
/// chunk boundaries are already fixed by `(n, num_chunks)` alone (see its
/// own header), so computing each chunk's partial independently and then
/// combining the partials **in chunk-index order** -- not completion order
/// -- is sufficient for bitwise reproducibility, verified directly in
/// `tests/kernels/parallel_reduce.cpp` across different thread counts and
/// repeated runs, not merely argued for.
///
/// `reduce_fast` is the honest counterpoint: partials are folded into a
/// running total as each chunk *finishes*, under a mutex, with no
/// constraint on order. It is not wrong -- every valid floating-point
/// summation order is a legitimate answer -- but it is not reproducible,
/// and `tests/kernels/parallel_reduce.cpp` demonstrates that directly by
/// showing its result *can* differ from `reduce_deterministic`'s (while
/// agreeing with it to within ordinary floating-point reassociation error),
/// rather than asserting bitwise equality it cannot honestly promise.

#include <cstddef>
#include <functional>

#include "volatility_lab/core/thread_pool.hpp"

namespace vl::kernels::parallel {

/// Reduce `[0, n)` to a single `double` via `per_chunk(start, end) -> double`,
/// combining the `num_chunks` partials in index order. Bitwise reproducible
/// across runs and thread counts for a fixed `(n, num_chunks)` -- see the
/// file comment.
[[nodiscard]] double reduce_deterministic(
    ThreadPool& pool, std::size_t n, std::size_t num_chunks,
    const std::function<double(std::size_t, std::size_t)>& per_chunk);

/// Reduce `[0, n)` the same way, but combine each chunk's partial into the
/// running total as soon as that chunk finishes, under a mutex -- no fixed
/// combine order. Not reproducible bit-for-bit; see the file comment for
/// what *is* guaranteed.
[[nodiscard]] double reduce_fast(
    ThreadPool& pool, std::size_t n, std::size_t num_chunks,
    const std::function<double(std::size_t, std::size_t)>& per_chunk);

}  // namespace vl::kernels::parallel
