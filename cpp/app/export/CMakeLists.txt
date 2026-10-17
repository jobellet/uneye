# Export (events CSV, BIDS-style, MAT-file), session report, provenance journal. Depends only on the recorder contract.
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} describe --always --dirty --tags OUTPUT_VARIABLE EYELIVE_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _git_rc)
if(NOT _git_rc EQUAL 0 OR "${EYELIVE_GIT_VERSION}" STREQUAL "")
  set(EYELIVE_GIT_VERSION "unknown")
endif()
eyelive_add_module(export eyelive_recorder)
if(TARGET eyelive_export)
  target_compile_definitions(eyelive_export PRIVATE EYELIVE_VERSION="${EYELIVE_GIT_VERSION}")
endif()
