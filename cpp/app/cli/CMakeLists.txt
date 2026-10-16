# eyelive_cli (task T20): headless front-end of the engine, and its end-to-end tests (ctest label cli).
file(GLOB CLI_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(CLI_SRC)
  add_executable(eyelive_cli ${CLI_SRC})
  target_link_libraries(eyelive_cli PRIVATE eyelive_engine eyelive_detect_net eyelive_trigger)
  if(NOT MSVC)
    target_compile_options(eyelive_cli PRIVATE -Wall -Wextra)
  endif()
  file(GLOB CLI_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
  foreach(t ${CLI_TESTS})
    get_filename_component(tn ${t} NAME_WE)
    add_executable(eyelive_cli_${tn} ${t})
    target_link_libraries(eyelive_cli_${tn} PRIVATE eyelive_contracts Threads::Threads)
    add_test(NAME eyelive_cli_${tn} COMMAND eyelive_cli_${tn} $<TARGET_FILE:eyelive_cli> ${CMAKE_SOURCE_DIR})
    set_tests_properties(eyelive_cli_${tn} PROPERTIES LABELS "eyelive;cli" TIMEOUT 1500)   # (ThreadSanitizer builds are 10-20x slower)
  endforeach()
endif()

# simulation of the assisted-annotation workflow on a labelled dataset (ground truth plays the expert); not a test: run by hand, see docs/AUTOLEARN_SIM.md
add_executable(eyelive_autolearn_sim ${CMAKE_CURRENT_SOURCE_DIR}/tools/autolearn_sim.cpp)
target_link_libraries(eyelive_autolearn_sim PRIVATE eyelive_engine eyelive_detect_net eyelive_annot)

# prints what the engine receives from a stimulus script through the frame mailbox (docs/FRAME_MAILBOX.md); used with cpp/app/tools/psychopy_feed_demo.py
add_executable(eyelive_feed_probe ${CMAKE_CURRENT_SOURCE_DIR}/tools/feed_probe.cpp)
target_link_libraries(eyelive_feed_probe PRIVATE eyelive_engine eyelive_detect_net)
