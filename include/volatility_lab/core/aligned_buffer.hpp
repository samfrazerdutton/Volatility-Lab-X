// SPDX-License-Identifier: MIT
#pragma once
/// \file aligned_buffer.hpp
/// \brief Over-aligned, non-initialising contiguous storage for SoA columns.
///
/// ## Why not std::vector<double>
///
/// Three reasons, in order of measured importance:
///
///  1. **Alignment.** std::vector guarantees only alignof(double).  An AVX2
///     kernel on an unaligned column either pays a split-load penalty on every
///     iteration or needs a scalar prologue whose length varies per column,
///     which in turn breaks the bitwise determinism of the chunked reductions
///     (the chunk boundaries would depend on the allocation address).  A
///     64-byte-aligned base makes the vector body aligned and the prologue
///     empty, every time.
///
///  2. **Tail padding.** Columns are allocated with capacity rounded up to a
///     whole number of vector registers, and the pad lanes are filled with a
///     benign value.  The SIMD kernels can then process ceil(n/W) registers
///     with no masked epilogue.  Reading pad lanes is defined behaviour here
///     because the memory is ours and is initialised.
///
///  3. **No value-initialisation.** std::vector<double> v(n) writes n zeros.
///     For a 1M-option batch that is 8 MB of pure store traffic before any
///     useful work is done.  AlignedBuffer leaves the body uninitialised and
///     makes the caller state the fill explicitly.
///
/// ## Ownership and lifetime
///
/// AlignedBuffer is a sole-ownership RAII handle.  It never reallocates
/// implicitly: resize() discards the old contents.  Spans handed out by
/// view()/data() are borrowed and are invalidated by resize() or destruction.
/// Nothing in the library retains such a span beyond the call it was passed to.

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

#include "volatility_lab/core/config.hpp"

namespace vl {
namespace detail {

[[nodiscard]] inline void* aligned_alloc_bytes(std::size_t bytes, std::size_t align) {
    if (bytes == 0) return nullptr;
    const std::size_t padded = round_up(bytes, align);
#if defined(_MSC_VER)
    void* p = ::_aligned_malloc(padded, align);
#else
    void* p = std::aligned_alloc(align, padded);
#endif
    if (p == nullptr) throw std::bad_alloc{};
    return p;
}

inline void aligned_free_bytes(void* p) noexcept {
    if (p == nullptr) return;
#if defined(_MSC_VER)
    ::_aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace detail

template <class T, std::size_t Align = kSimdAlign>
class AlignedBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AlignedBuffer deliberately skips initialisation and so only "
                  "supports trivially copyable element types");
    static_assert(Align >= alignof(T) && (Align & (Align - 1)) == 0,
                  "Align must be a power of two at least alignof(T)");

  public:
    using value_type = T;

    AlignedBuffer() = default;

    /// Allocate storage for n elements, with capacity rounded up so that the
    /// buffer holds a whole number of lanes-wide vector registers.  The body
    /// [0, n) is **uninitialised**; the pad [n, capacity) is set to pad_value
    /// so that SIMD epilogues can run unmasked over real memory.
    explicit AlignedBuffer(std::size_t n, std::size_t lanes = kMaxVectorLanesF64,
                           T pad_value = T{})
        : size_(n), capacity_(n == 0 ? 0 : round_up(n, lanes)) {
        if (capacity_ != 0) {
            data_ = static_cast<T*>(
                detail::aligned_alloc_bytes(capacity_ * sizeof(T), Align));
            std::fill(data_ + size_, data_ + capacity_, pad_value);
        }
    }

    AlignedBuffer(const AlignedBuffer& other)
        : size_(other.size_), capacity_(other.capacity_) {
        if (capacity_ != 0) {
            data_ = static_cast<T*>(
                detail::aligned_alloc_bytes(capacity_ * sizeof(T), Align));
            std::memcpy(data_, other.data_, capacity_ * sizeof(T));
        }
    }

    AlignedBuffer& operator=(const AlignedBuffer& other) {
        if (this != &other) {
            AlignedBuffer tmp(other);
            swap(tmp);
        }
        return *this;
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            detail::aligned_free_bytes(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~AlignedBuffer() { detail::aligned_free_bytes(data_); }

    void swap(AlignedBuffer& o) noexcept {
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
        std::swap(capacity_, o.capacity_);
    }

    /// Discards existing contents.  Reuses the allocation when it is already
    /// large enough, which matters for the streaming mode where buffers are
    /// resized once per market update.
    void resize(std::size_t n, std::size_t lanes = kMaxVectorLanesF64, T pad_value = T{}) {
        const std::size_t want = (n == 0) ? 0 : round_up(n, lanes);
        if (want > capacity_) {
            AlignedBuffer tmp(n, lanes, pad_value);
            swap(tmp);
            return;
        }
        size_ = n;
        if (data_ != nullptr) std::fill(data_ + size_, data_ + capacity_, pad_value);
    }

    void fill(T v) noexcept {
        if (data_ != nullptr) std::fill(data_, data_ + capacity_, v);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    [[nodiscard]] std::span<T> view() noexcept { return {data_, size_}; }
    [[nodiscard]] std::span<const T> view() const noexcept { return {data_, size_}; }

    /// The padded span, including the vector-alignment tail.  Only SIMD
    /// kernels should use this, and only to write values they will discard.
    [[nodiscard]] std::span<T> padded_view() noexcept { return {data_, capacity_}; }

    [[nodiscard]] T* begin() noexcept { return data_; }
    [[nodiscard]] T* end() noexcept { return data_ + size_; }
    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] const T* end() const noexcept { return data_ + size_; }

  private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

/// Cache-line-padded wrapper for per-thread accumulators.  Without it, an
/// array of 16 double partial sums occupies two cache lines and every thread
/// store invalidates its neighbour copies -- the classic false-sharing
/// collapse.  Measured in benchmarks/portfolio.
template <class T>
struct alignas(kCacheLine) CachePadded {
    T value{};

  private:
    static constexpr std::size_t kTail = sizeof(T) % kCacheLine;
    char pad_[kTail == 0 ? kCacheLine : kCacheLine - kTail]{};
};

}  // namespace vl
