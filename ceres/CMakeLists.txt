cmake_minimum_required(VERSION 3.8)
project(ceres)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES)

add_executable(hello ${SOURCE_FILES} hello_ceres_nonlin_lsq.cpp)
add_executable(hello_ceres ${SOURCE_FILES} hello_ceres.cpp)

target_link_libraries(hello ${CERES_LIBRARIES})
target_link_libraries(hello_ceres ${CERES_LIBRARIES})
