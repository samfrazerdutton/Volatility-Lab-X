# ---------------------------------------------------------------------------
# volatility_lab_demo: the end-to-end engineering demonstration. No
# business logic of its own -- every number it prints comes from a call
# into `volatility_lab`, never a value computed locally in this file.
# ---------------------------------------------------------------------------

add_executable(volatility_lab_demo main.cpp)
target_link_libraries(volatility_lab_demo PRIVATE volatility_lab vl_compiler_flags)
