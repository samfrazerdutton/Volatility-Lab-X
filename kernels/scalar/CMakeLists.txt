# ---------------------------------------------------------------------------
# Scalar-optimised kernel tier: the REFERENCE -> SCALAR step of the
# directive's validation ladder. Built with the project's ordinary compiler
# flags (no explicit ISA extension) -- the point of this tier is a
# branch-light, SIMD-shaped formula proven correct in scalar form *before*
# it is ever vectorised, not raw speed from a wider register.
#
# No separate OBJECT library here: that pattern (see kernels/CMakeLists.txt)
# exists specifically so the SIMD tier can be compiled with a different ISA
# flag than the rest of the library. This tier uses the exact same flags as
# everything else, so its one source file is simply added to the existing
# `volatility_lab` target directly -- the smallest design that is still
# correct, not a pattern applied just because the SIMD tier next to it needs
# it.
# ---------------------------------------------------------------------------

target_sources(volatility_lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/erfcx_poly.cpp")
# PUBLIC, not PRIVATE: the test binary for this kernel (tests/kernels/)
# needs `kernels/` on its own include path too, and tests link against
# `volatility_lab` rather than redeclaring this directory themselves.
target_include_directories(volatility_lab PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
