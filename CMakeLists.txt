cmake_minimum_required(VERSION 3.2.2)
project(simple_backpropagation)

file(GLOB SOURCES Backpropagation/*.cpp Backpropagation/*.h)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_executable(simple_backpropagation ${SOURCES})
