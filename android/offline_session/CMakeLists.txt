# OfflineSession facade (phase 1). New code in android/ only; links the
# frozen-contract modules. The JNI bridge is compiled only for Android
# (UNEYE_ANDROID defined by the toolchain build).
add_library(offline_session STATIC src/offline_session.cpp)
target_include_directories(offline_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(offline_session PUBLIC
  eyelive_annot eyelive_detect_hmm eyelive_detect_net eyelive_calib eyelive_train uneye_rt Threads::Threads)
if(NOT MSVC)
  target_compile_options(offline_session PRIVATE -Wall -Wextra)
endif()

if(ANDROID OR UNEYE_ANDROID_JNI)
  add_library(offline_session_jni SHARED jni/offline_session_jni.cpp)
  target_link_libraries(offline_session_jni PRIVATE offline_session)
  set_target_properties(offline_session_jni PROPERTIES OUTPUT_NAME offline_session)
endif()

file(GLOB OS_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
foreach(t ${OS_TESTS})
  get_filename_component(tn ${t} NAME_WE)
  add_executable(${tn} ${t})
  target_link_libraries(${tn} PRIVATE offline_session)
  if(NOT MSVC)
    target_compile_options(${tn} PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME ${tn} COMMAND ${tn})
  set_tests_properties(${tn} PROPERTIES TIMEOUT 600)
endforeach()
