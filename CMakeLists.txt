cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(clt LANGUAGES C)

include(GNUInstallDirs)

set(clt_SOURCES
  src/clt13.c
  src/crt_tree.c
  src/estimates.c
  )
set(clt_HEADERS
  src/clt13.h
  src/crt_tree.h
  src/estimates.h
  )

add_library(clt SHARED ${clt_SOURCES})
target_include_directories(clt PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(clt INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(clt PRIVATE gmp aesrand)
target_compile_options(clt PRIVATE
  -Wall -Wextra -Wpedantic -Werror -std=gnu11 -march=native -fopenmp)

install(TARGETS clt LIBRARY DESTINATION lib)
install(FILES src/clt13.h DESTINATION include)

# Test files

# add_executable(test_aes test/aes.c test/utils.c)
# add_executable(test_circuit test/circuit.c test/utils.c)
# add_executable(test_gates test/gates.c test/utils.c)
# target_include_directories(test_aes PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/src
#   ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(test_aes PRIVATE garble garblec)
# target_include_directories(test_circuit PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/src
#   ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(test_circuit PRIVATE garble garblec)
# target_include_directories(test_gates PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/src
#   ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(test_gates PRIVATE garble garblec)

# enable_testing()
# add_test(NAME test_aes COMMAND test_aes)
# add_test(NAME test_circuit COMMAND test_circuit)
# add_test(NAME test_gates COMMAND test_gates)
