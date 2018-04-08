# Comparing the Performance of Nonlinear Solvers for Bundle Adjustment

## Overview

|  | Info  |
|---|---|
| Course | CSC2305
| Term | Winter 2018
| Instructor | Professor Kenneth R. Jackson
| Languages | C++ (`libceres`) and Python

## Instructions

  1. Use the `get_data.py` script to download the datasets used in this project.
     1. Bear in mind that there is quite a bit of data. Extracted, it will take up 
        about 6GiB of disk space.
     1. TODO(andreib): Do we really need all of it?
     
  2. Build the code using CMake.
     1. Ensure you have a modern version of CMake, i.e., `>3.2`.
     2. For a quick-and-dirty way of checking out the code, install the Ceres 
        libraries on your system from your package manager. On Debian-like distributions 
        of Linux, this should be:
        ```bash
        sudo apt install libceres-solver-dev
        ```
        **However, it is STRONGLY RECOMMENDED to build Ceres from source
        if possible in order to have support multithreading and sparse solvers.**
     3. TODO(andreib): Describe other dependencies.
     4. Build the project in the standard CMake fashion, throwing all available
        cores at the compilation task:
        ```bash
        mkdir build && cd $_ && cmake .. && make -j$(nproc)
        ```
  3. The analysis code used to produce the plots and the other results in the 
    report is based on the Python code from the `analysis/` subdirectory. Its 
    dependencies can be resolved very efficiently using a Python virtual 
    environment or Anaconda.
     0. `cd analysis`
     0. `virtualenv .venv`
     0. `.venv/bin/activate`
     0. `pip install -r requirements.txt`
  4. To run the experiments XXX.
  5. Then, analyize the CSV data dumped by the experiment runs using the 
     `analysis/analyze.py` tool.


## Miscellaneous

### Hardware used for the experiments

The experiments were run on a workstation with a 24-core Intel Xeon ES-2687W v4 
@ 3.5Ghz with 128GiB of RAM. (Most experiments should run OK on a less beefy 
machine, but you should see some of the solvers start to fail on the larger 
datasets. (And the dense solvers will start to fail even faster.))

