cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(clt13 VERSION 1.0.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(GNUInstallDirs)

set(clt13_SOURCES
  src/clt13.c
  src/crt_tree.c
  src/estimates.c
  src/utils.c
  )
set(clt13_HEADERS
  src/clt13.h
  src/crt_tree.h
  src/estimates.h
  src/utils.h
  )

add_library(clt13 SHARED ${clt13_SOURCES})
target_include_directories(clt13 PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(clt13 PUBLIC gmp aesrand)

include(FindOpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Werror -std=gnu11 -march=native ${OpenMP_C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pg -ggdb -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
target_link_libraries(clt13 INTERFACE "${OpenMP_C_FLAGS}")

install(TARGETS clt13 LIBRARY DESTINATION lib)
install(FILES src/clt13.h DESTINATION include)

# Test files

enable_testing()
macro(add_test_ _name)
  add_executable("${_name}" "test/${_name}.c")
  target_include_directories("${_name}" PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries("${_name}" PRIVATE clt13 aesrand)
  add_test(NAME "${_name}" COMMAND "${_name}")
endmacro()

add_test_(test_clt)
add_test_(test_clt_state_new)
