# ---------------------------------------------------------------------------
# Compute kernels.
#
# The kernels live outside src/ because they are compiled with *different
# flags*: only kernels/simd gets the AVX2 ISA level, so that the library as a
# whole still loads on a machine without it and the runtime dispatcher can
# choose.  Mixing ISA levels inside one translation unit is the usual way a
# "portable" binary acquires an illegal-instruction crash on older hardware.
#
# Each backend is an OBJECT library linked into volatility_lab, so there is
# one archive to link against and no load-order surprises.
# ---------------------------------------------------------------------------

set(VL_KERNEL_SOURCES "")

# --- scalar: the correctness baseline every other backend is checked against
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scalar/CMakeLists.txt")
    add_subdirectory(scalar)
endif()

# --- simd: compiled with the AVX2 ISA level, dispatched at runtime
if(VL_ENABLE_SIMD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/simd/CMakeLists.txt")
    add_subdirectory(simd)
endif()

# --- parallel: thread pool and/or OpenMP over the scalar and SIMD kernels
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parallel/CMakeLists.txt")
    add_subdirectory(parallel)
endif()

# --- cuda: optional, and only ever for the workloads that profile as wins
if(VL_ENABLE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CMakeLists.txt")
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()
