// SPDX-License-Identifier: MIT
#include "volatility_lab/core/thread_pool.hpp"

#include <algorithm>
#include <vector>

namespace vl {

ThreadPool::ThreadPool(std::size_t num_threads) {
    const std::size_t n = std::max<std::size_t>(1, num_threads);
    workers_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) {
                if (stop_) return;
                continue;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(std::size_t n, std::size_t num_chunks,
                              const std::function<void(std::size_t, std::size_t)>& fn) {
    if (n == 0 || num_chunks == 0) return;
    num_chunks = std::min(num_chunks, n);

    // Fixed, deterministic chunk boundaries -- depend only on (n,
    // num_chunks), computed once, before any task is dispatched. The
    // first (n % num_chunks) chunks get one extra element so every
    // element is covered exactly once with no remainder chunk.
    const std::size_t base = n / num_chunks;
    const std::size_t remainder = n % num_chunks;

    std::vector<std::future<void>> futures;
    futures.reserve(num_chunks);
    std::vector<std::exception_ptr> errors(num_chunks);

    std::size_t start = 0;
    for (std::size_t c = 0; c < num_chunks; ++c) {
        const std::size_t size = base + (c < remainder ? 1 : 0);
        const std::size_t end = start + size;
        futures.push_back(submit([&fn, &errors, c, start, end] {
            try {
                fn(start, end);
            } catch (...) {
                errors[c] = std::current_exception();
            }
        }));
        start = end;
    }

    // Wait for every chunk, in order, regardless of which finished first --
    // this is what makes "every chunk has finished" true before any
    // exception is rethrown, not merely "the first one we happened to
    // check".
    for (auto& f : futures) f.get();

    for (const auto& e : errors) {
        if (e) std::rethrow_exception(e);
    }
}

}  // namespace vl
