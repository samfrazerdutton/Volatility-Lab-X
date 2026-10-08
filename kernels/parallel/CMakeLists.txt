# ---------------------------------------------------------------------------
# Parallel kernel tier: deterministic vs fast reduction, built on
# core/thread_pool.hpp. No special ISA flags needed (unlike kernels/simd/),
# so -- same reasoning as kernels/scalar/ -- the source is added directly
# to the already-existing `volatility_lab` target rather than introduced
# as a separate OBJECT library.
# ---------------------------------------------------------------------------

target_sources(volatility_lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp")
target_include_directories(volatility_lab PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
