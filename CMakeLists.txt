# =============================================================================
#  VOLATILITY LAB - research-grade C++ volatility analytics & scenario engine
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(volatility_lab
        VERSION 0.1.0
        DESCRIPTION "Research-grade C++ volatility analytics and scenario engine"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options.  Every one of these gates a real, measurable capability; none of
# them exist to make the build look sophisticated.
# ---------------------------------------------------------------------------
option(VL_BUILD_TESTS       "Build the test suite"                      ON)
option(VL_BUILD_BENCHMARKS  "Build the benchmark suite"                 ON)
option(VL_BUILD_EXAMPLES    "Build the examples"                        ON)
option(VL_BUILD_CLI         "Build the volatility-lab CLI"              ON)
option(VL_BUILD_PYTHON      "Build the pybind11 Python bindings"        OFF)
option(VL_ENABLE_SIMD       "Build the explicit SIMD kernels"           ON)
option(VL_ENABLE_OPENMP     "Use OpenMP for the parallel backend"       ON)
option(VL_ENABLE_CUDA       "Build the optional CUDA scenario backend"  OFF)
option(VL_ENABLE_NATIVE     "Compile with -march=native (non-portable)" OFF)
option(VL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"        OFF)
option(VL_ENABLE_ASAN       "AddressSanitizer"                          OFF)
option(VL_ENABLE_UBSAN      "UndefinedBehaviorSanitizer"                OFF)
option(VL_ENABLE_TSAN       "ThreadSanitizer"                           OFF)
option(VL_ENABLE_LTO        "Link-time optimization for Release"        OFF)

include(VLCompilerFlags)
include(VLSanitizers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Keep every runtime artefact in one predictable place so the benchmark and
# CLI harnesses can find each other without path archaeology.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(VL_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _vl_ipo OUTPUT _vl_ipo_msg)
  if(_vl_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
  else()
    message(WARNING "LTO requested but unsupported: ${_vl_ipo_msg}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Build-environment capture.  Benchmarks must never hardcode the machine they
# were run on, so the toolchain/CPU description is compiled in from here.
# ---------------------------------------------------------------------------
include(VLBuildInfo)

add_subdirectory(src)
add_subdirectory(kernels)

if(VL_BUILD_CLI)
  add_subdirectory(apps/cli)
endif()

if(VL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VL_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(VL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(VL_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# The interface targets carrying flags must be exported alongside the library,
# or the generated config file references targets a consumer does not have.
install(TARGETS volatility_lab vl_compiler_flags vl_simd_flags vl_sanitizers
        EXPORT volatility_labTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(EXPORT volatility_labTargets
        FILE volatility_labTargets.cmake
        NAMESPACE volatility_lab::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volatility_lab)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/volatility_labConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion)

message(STATUS "")
message(STATUS "=== VOLATILITY LAB ${PROJECT_VERSION} ===")
message(STATUS "  build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  SIMD kernels : ${VL_ENABLE_SIMD}  (${VL_SIMD_DESCRIPTION})")
message(STATUS "  OpenMP       : ${VL_HAVE_OPENMP}")
message(STATUS "  CUDA         : ${VL_ENABLE_CUDA}")
message(STATUS "  tests        : ${VL_BUILD_TESTS}")
message(STATUS "  benchmarks   : ${VL_BUILD_BENCHMARKS}")
message(STATUS "")
