# Parallelizing Cooley-Tukey Algorithm

Created By: Utsav Lal

Welcome to the repository for parallelizing Cooley-Tukey Algorithm. In this project my main aim was to build hardware parallelized algorithms for calculating Discrete fourier transform.

## Folder Structure

`Archive/`: Contains experimental code which may or may not run \

`Images/`: Images I used for ppt and reports \

`Recordings/`: Data gathered from experiments \

`Reports/`: Contains the final reports

`Research Material/`: References I used

`Visualize/`: Code used to generate graphs

`fft_bailey.cpp`: The code for MPI based Bailey's algorithm for parallelizing Cooley-Tukey Algorithm

`fft_cuda.cu`: The code for CUDA based parallelized version of Cooley-Tukey Algorithm

`Makefile`: Build instructions

`run.mpi`: Useful for submitting batch jobs on ARC or similar slurm based clusters

## How to run code

There are two main variants:

1. Parallelized version of Cooley-Tukey using Nvidia CUDA
2. Parellized version of Bailey's algorithm using Open MPI

## How to run the code

1. Cuda Parallelization

   - To build the CUDA version first use make to build the executable. `make fft_cuda`
   - To run it use `./fft_cuda` to run the executable
   - The cuda code automatically runs tests for a variety of input sizes

2. MPI Parallelization
   - To build the MPI version use similar command `make fft_bailey`
   - The general structure to run is `prun ./fft_bailey size_of_matrix`
   - For example for input of size 2048 use `prun ./fft_bailey 2048` and for input of size 4096 use `prun ./fft_bailey 4096`
