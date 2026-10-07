# ---------------------------------------------------------------------------
# VLSanitizers - sanitizer wiring.  ASan/UBSan may be combined; TSan must be
# used alone (it is incompatible with ASan's shadow memory layout).
# ---------------------------------------------------------------------------
add_library(vl_sanitizers INTERFACE)

if(VL_ENABLE_TSAN AND (VL_ENABLE_ASAN OR VL_ENABLE_UBSAN))
  message(FATAL_ERROR "ThreadSanitizer cannot be combined with ASan/UBSan")
endif()

set(_vl_san "")
if(VL_ENABLE_ASAN)
  list(APPEND _vl_san address)
endif()
if(VL_ENABLE_UBSAN)
  list(APPEND _vl_san undefined)
endif()
if(VL_ENABLE_TSAN)
  list(APPEND _vl_san thread)
endif()

if(_vl_san)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    if(VL_ENABLE_ASAN)
      target_compile_options(vl_sanitizers INTERFACE /fsanitize=address)
    endif()
    if(VL_ENABLE_UBSAN OR VL_ENABLE_TSAN)
      message(WARNING "MSVC supports only /fsanitize=address; UBSan/TSan ignored")
    endif()
  else()
    list(JOIN _vl_san "," _vl_san_csv)
    target_compile_options(vl_sanitizers INTERFACE
      -fsanitize=${_vl_san_csv} -fno-omit-frame-pointer -g)
    # clang targeting the MSVC ABI links the sanitizer runtime automatically
    # and rejects -fsanitize on the link line.
    if(NOT MSVC)
      target_link_options(vl_sanitizers INTERFACE -fsanitize=${_vl_san_csv})
    endif()
  endif()
  message(STATUS "Sanitizers enabled: ${_vl_san}")
endif()
