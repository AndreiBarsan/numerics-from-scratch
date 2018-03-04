cmake_minimum_required(VERSION 3.8)
project(ceres)

# This automatically adds dependencies on Eigen, SuiteSparse, etc.
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
add_subdirectory(third_party/Sophus)

set(CMAKE_CXX_STANDARD 17)

set(COMMON_SOURCE_FILES BALProblem.h)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -fPIC -msse3")

add_executable(hello ${COMMON_SOURCE_FILES} hello_ceres_nonlin_lsq.cpp)
add_executable(hello_ceres ${COMMON_SOURCE_FILES} hello_ceres.cpp)
add_executable(powell ${COMMON_SOURCE_FILES} powell.cpp)
add_executable(nonlin_lsq ${COMMON_SOURCE_FILES} nonlin_lsq.cpp)
add_executable(ba ${COMMON_SOURCE_FILES} ba.cpp)

target_link_libraries(hello ${CERES_LIBRARIES})
target_link_libraries(hello_ceres ${CERES_LIBRARIES})
target_link_libraries(powell ${CERES_LIBRARIES})
target_link_libraries(nonlin_lsq ${CERES_LIBRARIES})

target_link_libraries(ba ${CERES_LIBRARIES})
target_link_libraries(ba sophus)
