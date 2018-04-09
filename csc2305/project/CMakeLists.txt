cmake_minimum_required(VERSION 3.8)
project(ceres)

# This automatically adds dependencies on Eigen, SuiteSparse, etc.
# Very important: you MUST specify the exact version you want if you have multiple versions of Ceres installed. That
# is, say you have a system libceres-solver-dev that a lot of your stuff depends on, and you want a custom, newer
# version of Ceres for this project. If you build Ceres yourself with EXPORT_BUILD_DIR enabled,
find_package(Ceres 1.14 REQUIRED)

# Build gflags anywhere with -DREGISTER_BUILD_DIR=ON and this will work (no need to install).
find_package(gflags REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

# We want to be bleeding-edge (native path support, stdlib multithreading, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -fPIC -msse3 -fopenmp")

set(COMMON_SOURCE_FILES bal_problem.h csv_writer.cpp bal_problem.cpp utils.cpp)

add_executable(ba_experiment ${COMMON_SOURCE_FILES} bundle_adjustment.cpp experiment.cpp experiment_configs.h)
add_executable(openmp_hello ${COMMON_SOURCE_FILES} openmp.cpp)

target_link_libraries(ba_experiment ${CERES_LIBRARIES} gflags)
target_link_libraries(openmp_hello ${CERES_LIBRARIES})
