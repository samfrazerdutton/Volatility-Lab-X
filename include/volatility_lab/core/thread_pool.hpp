// SPDX-License-Identifier: MIT
#pragma once
/// \file thread_pool.hpp
/// \brief A fixed-worker thread pool (directive Phase 4, section 10) --
///        the project's primary parallel backend, since OpenMP is not
///        available on this toolchain (`docs/BASELINE.md`).
///
/// ## Scope, stated plainly
///
/// A mutex-and-condition-variable task queue, not a lock-free work-stealing
/// deque. `submit`'s queued tasks are type-erased into `std::function<void()>`,
/// which allocates for a capturing lambda -- a genuinely allocation-free
/// pool needs either a fixed-signature task type or a custom work-stealing
/// structure, both real undertakings in their own right. This project's own
/// stated philosophy is "profile first, don't optimise based on dogma"
/// (directive section 27): this is the smallest correct design, and the
/// place to add a lock-free queue is after a benchmark shows contention on
/// this one actually costing something, not before.
///
/// ## Determinism
///
/// `parallel_for` partitions `[0, n)` into exactly `num_chunks` contiguous,
/// fixed-boundary ranges (as even as integer division allows) *before*
/// dispatching any work -- the boundaries depend only on `(n, num_chunks)`,
/// never on which worker happens to pick up which chunk or in what order
/// chunks finish. That is what makes `kernels/parallel/`'s deterministic
/// reduction (`parallel_reduce_deterministic`, built on top of this)
/// bitwise reproducible regardless of thread-scheduling noise -- see
/// `math::tol::kParallelReduction`'s own rationale ("fixed chunking +
/// in-order combine => bitwise"), which predates any implementation of
/// this and is verified directly, not merely asserted, in
/// `tests/kernels/parallel_reduce.cpp`.

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace vl {

class ThreadPool {
  public:
    /// `num_threads == 0` is clamped to 1 -- a pool with zero workers would
    /// deadlock on the first task, which is a programming error worth
    /// making impossible rather than detecting at the first hang.
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] std::size_t num_threads() const noexcept { return workers_.size(); }

    /// Submit one task. The returned future carries the result, or the
    /// exception the task threw -- `std::future`'s own propagation, not a
    /// second mechanism layered on top.
    template <class Fn, class... Args>
    [[nodiscard]] auto submit(Fn&& fn, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>> {
        using R = std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>;
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
        std::future<R> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    /// Partition `[0, n)` into exactly `num_chunks` fixed, contiguous
    /// ranges and run `fn(start, end)` once per chunk across the pool,
    /// blocking until every chunk completes. `num_chunks == 0` or `n == 0`
    /// is a no-op. If one or more chunks throw, the chunk-index-ordered
    /// first exception is rethrown here, after *every* chunk has finished
    /// (never torn down mid-flight leaving other chunks running past the
    /// caller's visibility) -- not silently swallowed, and not racing on
    /// which thread's exception happens to be caught first.
    void parallel_for(std::size_t n, std::size_t num_chunks,
                      const std::function<void(std::size_t, std::size_t)>& fn);

  private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;

    void worker_loop();
};

}  // namespace vl
