# Phase 0 benchmark harness (not part of the shipped app). Links only the
# frozen-contract modules the app will use: detect_hmm (pre-labeler), annot
# (ExpertLabels), train (trainer). dsp is header-only (its tests are excluded
# by EYELIVE_BUILD_APP=OFF in the parent project... it is always built by the
# cpp/ tree when EYELIVE_BUILD_APP=ON).
add_executable(bench_offline bench_offline.cpp)
target_link_libraries(bench_offline PRIVATE eyelive_detect_hmm eyelive_annot eyelive_train Threads::Threads)
if(NOT MSVC)
  target_compile_options(bench_offline PRIVATE -Wall -Wextra)
endif()
