# Comparing the Performance of Nonlinear Solvers for Bundle Adjustment

## Overview

|  | Info  |
|---|---|
| Course | CSC2305
| Term | Winter 2018
| Instructor | Professor Kenneth R. Jackson
| Languages | C++ and Python

## Instructions

  1. Use the `get_data.py` script to download the datasets used in this project.
     * Bear in mind that there is quite a bit of data. Extracted, it will take up 
       about 6GiB of disk space.
  2. Build the code using CMake.
     1. Ensure you have a modern version of CMake, i.e., `>3.2`.
     2. For a quick-and-dirty way of checking out the code, install the Ceres 
        libraries on your system from your package manager. On Debian-like distributions 
        of Linux, this should be:
        ```bash
        sudo apt install libceres-solver-dev
        ```
        **However, it is STRONGLY RECOMMENDED to build Ceres from source
        if possible in order to have guaranteed support for multithreading and the sparse solvers.**
     1. (Optional, but recommended) To build Ceres from source (no need to make install in the end):
        ```bash
        cd third_party
        mkdir ceres-bin
        cd ceres-bin
        cmake ../ceres-solver -DEXPORT_BUILD_DIR=ON  
        make -j$(nproc)
        ```
        (Please see the Ceres docs for information on any necessary dependencies, such as OpenMP and Eigen.)
     3. Set up `gflags` (no need to make install).
        ```bash
        cd third_party/gflags && mkdir build && cd $_ && cmake .. -DREGISTER_BUILD_DIR=ON && make -j$(nproc)
        ```
     4. Build the main project in the standard CMake fashion, throwing all available
        cores at the compilation task:
        ```bash
        mkdir build && cd $_ && cmake .. && make -j$(nproc)
        ```
  3. The analysis code used to produce the plots and the other results in the 
    report is based on the Python code from the `analysis/` subdirectory. Its 
    dependencies can be resolved very efficiently using a Python virtual 
    environment or Anaconda.
        ```bash
         cd analysis
         virtualenv .venv
         .venv/bin/activate
         pip install -r requirements.txt
        ```
  4. To run the experiments, use the tool `build/ba_experiment`. Run it as
     `build/ba_experiment --help` to see the available flags, and check out
     the function `Experiments` from `experiment.cpp` for selecting which solver
     configurations to use.
  1. By default, the tool dumps experiment data in the subfolder `experiments/00`.
     Make sure it exists before running the tool:
     ```bash
     mkdir -p experiments/00
     ```
  5. After running the experiments, the tool will dump the relevant information in
    the aforementioned experiment output directory. You can analyize this data
    using the 
     `analysis/AnalyzeCeresExperiments.ipynb` Jupyter notebook. To run the notebook,
     simply run `jupyter notebook` in the project root (requires the aforementioned
     Python dependencies), and navigate to the analysis notebook using your browser.


## Miscellaneous

### Hardware used for the experiments

The experiments were run on a workstation with a 24-core Intel Xeon ES-2687W v4 
@ 3.5Ghz with 128GiB of RAM. (Most experiments should run OK on a less beefy 
machine, but you should see some of the solvers start to fail on the larger 
datasets. (And the dense solvers will start to fail even faster.))

