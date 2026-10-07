# ---------------------------------------------------------------------------
# VLCompilerFlags - one interface target carrying the project's warning and
# optimisation policy, plus a separate target for the SIMD ISA level.
#
# The ISA level is deliberately *not* applied globally.  Only the explicit
# kernels in kernels/simd are compiled with AVX2/FMA so that the library can
# still load on a machine without them; the dispatcher picks at runtime.
# ---------------------------------------------------------------------------

add_library(vl_compiler_flags INTERFACE)
add_library(vl_simd_flags INTERFACE)

set(VL_SIMD_DESCRIPTION "none" CACHE INTERNAL "")

if(MSVC AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
  # ---- MSVC cl.exe / clang-cl -------------------------------------------
  target_compile_options(vl_compiler_flags INTERFACE
    /W4 /permissive- /Zc:preprocessor /Zc:__cplusplus /bigobj
    /wd4267   # size_t -> smaller, audited at the few sites it occurs
    /fp:precise
    $<$<CONFIG:Release,RelWithDebInfo>:/O2>
  )
  if(VL_WARNINGS_AS_ERRORS)
    target_compile_options(vl_compiler_flags INTERFACE /WX)
  endif()
  if(VL_ENABLE_SIMD)
    target_compile_options(vl_simd_flags INTERFACE /arch:AVX2)
    set(VL_SIMD_DESCRIPTION "AVX2+FMA (/arch:AVX2)" CACHE INTERNAL "" FORCE)
  endif()
  target_compile_definitions(vl_compiler_flags INTERFACE
    _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
else()
  # ---- GCC / Clang (incl. clang++ with the MSVC ABI) ---------------------
  target_compile_options(vl_compiler_flags INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion -Wsign-conversion
    -Wcast-align -Wcast-qual -Wdouble-promotion
    -Wold-style-cast -Wnon-virtual-dtor -Woverloaded-virtual
    -Wnull-dereference -Wformat=2 -Wundef
    # Floating point: we require IEEE-754 semantics everywhere.  -ffast-math
    # would let the compiler reassociate reductions and break the bitwise
    # determinism guarantees documented in docs/numerical-validation.md.
    -fno-fast-math
  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vl_compiler_flags INTERFACE
      -Wduplicated-cond -Wduplicated-branches -Wlogical-op
      -Wuseless-cast -fno-unsafe-math-optimizations)
  endif()
  if(VL_WARNINGS_AS_ERRORS)
    target_compile_options(vl_compiler_flags INTERFACE -Werror)
  endif()
  if(MSVC)
    # clang++ driving the MSVC runtime: MSVC's own headers are not -Wall clean.
    target_compile_options(vl_compiler_flags INTERFACE
      -Wno-nonportable-system-include-path -Wno-language-extension-token)
    target_compile_definitions(vl_compiler_flags INTERFACE
      _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
  endif()
  if(VL_ENABLE_SIMD)
    if(VL_ENABLE_NATIVE)
      target_compile_options(vl_simd_flags INTERFACE -march=native)
      set(VL_SIMD_DESCRIPTION "native (-march=native)" CACHE INTERNAL "" FORCE)
    else()
      # AVX2 + FMA is the portable-enough baseline: every x86-64-v3 part.
      # AVX-512 is intentionally not a build-wide target -- see
      # docs/design-decisions.md (D-07).
      target_compile_options(vl_simd_flags INTERFACE -mavx2 -mfma)
      set(VL_SIMD_DESCRIPTION "AVX2+FMA (-mavx2 -mfma)" CACHE INTERNAL "" FORCE)
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP: used only by the parallel kernels.  Absence is not an error; the
# thread-pool backend covers the same functionality.
# ---------------------------------------------------------------------------
set(VL_HAVE_OPENMP OFF CACHE INTERNAL "")
if(VL_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX QUIET)
  if(OpenMP_CXX_FOUND)
    set(VL_HAVE_OPENMP ON CACHE INTERNAL "" FORCE)
  else()
    message(STATUS "OpenMP not found - the thread-pool backend will be used instead")
  endif()
endif()
