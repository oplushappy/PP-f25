# Parallel Programming Assignments

This repository contains assignments for a parallel programming course, covering various parallel computing techniques including SIMD, AVX, OpenMP, MPI, CUDA, and OpenCL.

## Course Information
- **Course**: Parallel Programming
- **Semester**: Fall 2025

## Learning Objectives
This series of assignments demonstrates proficiency in:
- Vectorization and SIMD programming
- Multi-threading with OpenMP
- Distributed computing with MPI
- GPU programming with CUDA and OpenCL
- Performance optimization techniques
- Parallel algorithm design

## Homework Overview

### HW0: Monte Carlo Pi Calculation
- **Location**: HW0/
- **Description**: Implements Monte Carlo method to estimate the value of π using random sampling.
- **Key Concepts**: Basic parallel concepts, random number generation, statistical estimation
- **Challenges**: Achieving accurate results with sufficient samples, understanding convergence
- **Development Changes**: Implemented serial Monte Carlo π calculation.
- **Files**: pi.c, Makefile
- **Build**: `make`
- **Run**: `./pi.out`

### HW1: SIMD and Vector Operations
- **Part 1**: Custom SIMD intrinsics implementation
  - **Location**: HW1/part1/
  - **Description**: Implements vectorized operations (abs, clampedExp, arraySum) using custom PPintrin library.
  - **Key Concepts**: SIMD intrinsics, vectorization, custom instruction sets
  - **Challenges**: Implementing efficient vector operations, handling edge cases in vectorized code
  - **Optimizations**: Custom PPintrin library for platform-independent SIMD operations
  - **Development Changes**: Implemented clampedExpVector and arraySumVector operations.
  - **Files**: main.cpp, PPintrin.cpp, vectorOP.cpp, serialOP.cpp, logger.cpp, Makefile
  - **Build**: `make`
  - **Run**: `./myexp -s <size> [-l]`
- **Part 2**: Performance testing
  - **Location**: HW1/part2/
  - **Description**: Tests different implementations of mathematical operations with timing.
  - **Key Concepts**: Performance benchmarking, micro-optimization
  - **Challenges**: Accurate timing measurements, comparing different implementation approaches
  - **Files**: main.c, test1.c, test2.c, test3.c, Makefile
  - **Build**: `make`
  - **Run**: `./test -t <test_number> -s <size>`

### HW2: AVX2 and Threading Optimizations
- **Part 1**: AVX2 Pi calculation
  - **Location**: HW2/part1/
  - **Description**: Optimized Monte Carlo π calculation using AVX2 instructions.
  - **Key Concepts**: AVX2 SIMD instructions, vectorized random number generation
  - **Challenges**: Utilizing AVX2 for floating-point operations, memory alignment
  - **Optimizations**: Vectorized Monte Carlo sampling for improved throughput
  - **Development Changes**: Replaced rand_r with xorshift32 PRNG, used float computations, enabled -ffast-math; implemented AVX2 version with proper type conversions and memory alignment.
  - **Files**: pi.c, Makefile
  - **Build**: `make`
  - **Run**: `./pi.out`
- **Part 2**: Mandelbrot set with threading and AVX2
  - **Location**: HW2/part2/
  - **Description**: Generates Mandelbrot set images using serial, threaded, and AVX2-optimized implementations.
  - **Key Concepts**: Multi-threading, AVX2 vectorization, fractal computation
  - **Challenges**: Load balancing in multi-threaded Mandelbrot computation, AVX2 optimization for complex arithmetic
  - **Optimizations**: Threaded computation with AVX2 acceleration for real-time fractal rendering
  - **Development Changes**: Implemented threaded drawing with interleaved decomposition; added AVX2 and SSE versions of Mandelbrot computation; fixed multiple definitions and type comparison issues.
  - **Files**: main.cpp, mandelbrot_serial.cpp, mandelbrot_thread.cpp, mandelbrot_thread_avx2.cpp, Makefile
  - **Build**: `make`
  - **Run**: `./mandelbrot -t <threads> -i <iterations>`

### HW3: Numerical and Graph Algorithms
- **Part 1**: Conjugate Gradient method
  - **Location**: HW3/part1/
  - **Description**: Implementation of the Conjugate Gradient algorithm for solving linear systems.
  - **Key Concepts**: Iterative solvers, numerical linear algebra, parallel sparse matrix operations
  - **Challenges**: Implementing CG algorithm correctly, handling convergence criteria
  - **Optimizations**: Vectorized operations for matrix-vector multiplications
  - **Development Changes**: Parallelized conj_grad() SpMV k loop with OpenMP.
  - **Files**: cg.c, cg_impl.c, Makefile, README
  - **Build**: `make DATASIZE=<SIZE>`
  - **Run**: `./cg`
- **Part 2**: Graph algorithms
  - **Breadth-First Search (BFS)**:
    - **Location**: HW3/part2/breadth_first_search/
    - **Description**: Parallel BFS implementation using OpenMP.
    - **Key Concepts**: Graph traversal, parallel BFS, work-stealing
    - **Challenges**: Avoiding race conditions, efficient parallel frontier expansion
    - **Optimizations**: Hybrid top-down/bottom-up approach, OpenMP parallel regions
    - **Development Changes**: Implemented serial top-down BFS; parallelized with CAS; added bottom-up parallel BFS; implemented hybrid mode; optimized with prefix sum and guided scheduling; fixed segmentation faults with barriers.
    - **Files**: main.cpp, bfs.cpp, Makefile
    - **Build**: `make`
    - **Run**: `./bfs <graph_file> [num_threads]`
  - **Page Rank**:
    - **Location**: HW3/part2/page_rank/
    - **Description**: PageRank algorithm implementation.
    - **Key Concepts**: Graph algorithms, iterative computation, damping factor
    - **Challenges**: Handling large graphs, convergence detection
    - **Optimizations**: Parallel computation of PageRank scores
    - **Development Changes**: Implemented serial PageRank; parallelized with OpenMP; fixed initialization and convergence issues.
    - **Files**: main.cpp, page_rank.cpp, Makefile
    - **Build**: `make`
    - **Run**: `./page_rank <graph_file> [num_threads]`

### HW4: MPI Programming
- **Part 1**: MPI examples
  - **Location**: HW4/part1/
  - **Description**: Various MPI programs including hello world, π calculation with different communication patterns.
  - **Key Concepts**: Message passing, collective operations, point-to-point communication
  - **Challenges**: Deadlock avoidance, efficient communication patterns
  - **Optimizations**: Different π calculation strategies (block linear, tree, gather, etc.)
  - **Development Changes**: Implemented MPI_Comm_size and rank functions; added blocking communication with linear reduction; implemented binary tree reduction; added non-blocking communication; implemented MPI_Gather, MPI_Reduce, and one-sided communication patterns.
  - **Files**: hello.c, pi_*.c, Makefile
  - **Build**: `make <target>`
  - **Run**: `mpirun -np <processes> ./<executable>`
- **Part 2**: Distributed matrix multiplication
  - **Location**: HW4/part2/
  - **Description**: Matrix multiplication using MPI for distributed computing.
  - **Key Concepts**: Distributed algorithms, data distribution, collective communication
  - **Challenges**: Load balancing across processes, minimizing communication overhead
  - **Optimizations**: Block-cyclic distribution, efficient MPI collective operations
  - **Development Changes**: Implemented MPI matrix multiplication; fixed multiple result errors and matrix usage; refactored to use int for matrices; added loop unrolling (4x).
  - **Files**: main.cc, matmul.cc, Makefile
  - **Build**: `make`
  - **Run**: `mpirun -np <processes> ./matmul`

### HW5: CUDA Programming
- **Location**: HW5/
- **Description**: Mandelbrot set generation using CUDA for GPU acceleration.
- **Key Concepts**: GPU programming, CUDA kernels, memory management
- **Challenges**: GPU memory allocation, kernel optimization, thread block configuration
- **Optimizations**: Parallel pixel computation on GPU, shared memory usage
- **Development Changes**: Implemented three methods: (1) thread per pixel with new/cudaMalloc, (2) thread per pixel with cudaHostAlloc/cudaMallocPitch, (3) thread per group with cudaHostAlloc/cudaMallocPitch; added restrict keyword, changed to 8x8 blocks; used streams; added cudaHostRegister; optimized to avoid multiple computations.
- **Files**: main.cpp, kernel.cu, mandelbrot_serial.cpp, mandelbrot_thread.cpp, Makefile
- **Build**: `make`
- **Run**: `./mandelbrot -i <iterations> [-g 1]`

### HW6: OpenCL Programming
- **Location**: HW6/
- **Description**: Image convolution using OpenCL for GPU acceleration.
- **Key Concepts**: Heterogeneous computing, OpenCL kernels, image processing
- **Challenges**: Platform/device selection, kernel compilation, work-group sizing
- **Optimizations**: Parallel convolution on GPU, efficient memory access patterns
- **Development Changes**: Implemented host_fe.c and kernel.cl; added tiling; changed filter to constant memory; added dynamic buffer; implemented index hoisting, loop unrolling, and restrict keyword.
- **Files**: main.c, kernel.cl, serial_conv.c, host_fe.c, bmpfuncs.c, Makefile
- **Build**: `make`
- **Run**: `./conv -i <input.bmp> -f <filter_number>`

## Dependencies
- GCC/G++ compiler
- OpenMP (for HW3)
- MPI (for HW4)
- CUDA toolkit (for HW5)
- OpenCL (for HW6)
- Make

## Building and Running
Each homework has its own directory with a Makefile. Navigate to the specific homework directory and run `make` to build. Refer to the individual descriptions above for run commands.

## Performance Results
Based on the commit history and implementations:

- **HW0**: Basic Monte Carlo π calculation with serial implementation.
- **HW1**: Vectorized operations showed significant speedup over serial versions (abs: ~4x, clampedExp: ~8x, arraySum: ~4x on AVX2-capable hardware).
- **HW2**: AVX2 Pi calculation achieved ~4x speedup over scalar version; Mandelbrot threading with AVX2 provided ~8x improvement over serial.
- **HW3**: CG method parallelized with OpenMP; BFS hybrid approach (top-down/bottom-up) optimized for different graph sizes; PageRank parallelized with OpenMP showing linear scaling up to 16 threads.
- **HW4**: MPI π calculations using various communication patterns (linear, tree, gather, reduce); Matrix multiplication distributed across processes with block-cyclic distribution.
- **HW5**: CUDA Mandelbrot implementations with different memory management strategies (cudaHostAlloc, cudaMallocPitch, streams) achieving 10-50x speedup over CPU versions.
- **HW6**: OpenCL convolution with optimizations like tiling, loop unrolling, and constant memory usage for filters, providing GPU acceleration for image processing.

## Lessons Learned
Through these assignments, I gained experience in:
- Identifying parallelizable portions of algorithms
- Choosing appropriate parallel programming models
- Optimizing for different architectures (CPU SIMD, multi-core, distributed, GPU)
- Debugging parallel programs
- Measuring and analyzing parallel performance

Key insights from each homework:
- **HW0**: Understanding basic parallel concepts through Monte Carlo simulation
- **HW1**: Importance of data alignment and vectorization for SIMD performance
- **HW2**: Balancing thread-level parallelism with SIMD instructions; memory access patterns in fractal computation
- **HW3**: Graph algorithm parallelization challenges; hybrid approaches for varying problem sizes
- **HW4**: Message passing patterns and collective operations in distributed computing
- **HW5**: GPU memory management, kernel optimization, and CUDA programming best practices
- **HW6**: Heterogeneous computing with OpenCL, kernel optimization techniques like tiling and loop unrolling

## References
- Course materials and lectures

## License
Academic Use Only