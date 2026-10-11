# eyelive_cli (task T20): headless front-end. Until it exists nothing is built here.
file(GLOB CLI_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(CLI_SRC)
  add_executable(eyelive_cli ${CLI_SRC})
  target_link_libraries(eyelive_cli PRIVATE eyelive_engine)
endif()
