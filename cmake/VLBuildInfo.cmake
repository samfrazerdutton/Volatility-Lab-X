# ---------------------------------------------------------------------------
# VLBuildInfo - compile the real toolchain/host description into the library.
#
# Rationale: every benchmark report must state the machine it ran on.  Reading
# that from a hand-maintained markdown file invites stale claims, so it is
# generated here and exposed through vl::build_info().
# ---------------------------------------------------------------------------

set(VL_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE VL_GIT_SHA
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(NOT VL_GIT_SHA)
    set(VL_GIT_SHA "unknown")
  endif()
endif()

cmake_host_system_information(RESULT VL_HOST_CPU   QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT VL_HOST_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
cmake_host_system_information(RESULT VL_HOST_HW    QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT VL_HOST_RAM   QUERY TOTAL_PHYSICAL_MEMORY)
cmake_host_system_information(RESULT VL_HOST_NAME  QUERY HOSTNAME)

string(STRIP "${VL_HOST_CPU}" VL_HOST_CPU)

if(VL_HAVE_OPENMP)
  set(VL_HAVE_OPENMP_01 1)
else()
  set(VL_HAVE_OPENMP_01 0)
endif()
if(VL_ENABLE_CUDA)
  set(VL_ENABLE_CUDA_01 1)
else()
  set(VL_ENABLE_CUDA_01 0)
endif()

configure_file("${PROJECT_SOURCE_DIR}/cmake/build_info.cpp.in"
               "${PROJECT_BINARY_DIR}/generated/build_info.cpp" @ONLY)
