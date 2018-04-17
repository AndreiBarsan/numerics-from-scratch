#!/usr/bin/env bash
# Experimental script for building this project and its dependencies from scratch.

set -eu

# Make sure you have g++ 5.4 or higher, even on older Ubuntus.
apt install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y g++-5 gcc-5
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 90 --slave /usr/bin/g++ g++ /usr/bin/g++-5

# Ensure an up-to-date CMake is available
scripts/install_cmake.sh NO_SUDO

# Build Ceres
(
    cd third_party
    mkdir -p ceres-bin
    cd ceres-bin

    cmake ../ceres-solver -DTBB=OFF -DOPENMP=ON -DEXPORT_BUILD_DIR=ON
    make -j$(nproc)

    # No make install, since just making the thing puts it in CMake's registry.
)

# Build the main project
(
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
)


