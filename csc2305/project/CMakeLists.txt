cmake_minimum_required(VERSION 3.8)
project(ceres)

# This automatically adds dependencies on Eigen, SuiteSparse, etc.
# Very important: you MUST specify the exact version you want if you have multiple versions of Ceres installed. That
# is, say you have a system libceres-solver-dev that a lot of your stuff depends on, and you want a custom, newer
# version of Ceres for this project. If you build Ceres yourself with EXPORT_BUILD_DIR enabled,
find_package(Ceres 1.14 REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 17)
set(COMMON_SOURCE_FILES BALProblem.h)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -fPIC -msse3")

add_executable(ba ${COMMON_SOURCE_FILES} ba.cpp)

target_link_libraries(ba ${CERES_LIBRARIES})
#target_link_libraries(ba sophus)
