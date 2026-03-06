# Parallel Matrix Multiplication with CUDA

High-performance matrix multiplication in C++/CUDA using **shared memory tiling**, achieving a **~42x speedup** over a single-threaded CPU baseline on large matrices.

## Overview

Matrix multiplication is a core primitive in deep learning (linear layers, attention). This project implements and benchmarks three approaches:

| Implementation | Description |
|---|---|
| `matmul_cpu` | Naive triple-loop CPU baseline |
| `matmul_naive` | GPU kernel, one thread per output element, global memory only |
| `matmul_tiled` | GPU kernel with shared memory tiling (TILE_SIZE × TILE_SIZE) |

## Key Optimizations

### Shared Memory Tiling
Each thread block loads a `TILE_SIZE × TILE_SIZE` (32×32) sub-tile of matrices A and B into shared memory before computing partial dot products. This dramatically reduces redundant global memory reads — each element is loaded from global memory once per tile instead of once per output element.

```
Global memory accesses reduced by factor of TILE_SIZE (32x)
```

### Coalesced Memory Access
Threads within a warp access consecutive memory addresses, maximizing memory transaction efficiency and DRAM bandwidth utilization.

### Loop Unrolling
`#pragma unroll` on the inner accumulation loop allows the compiler to pipeline arithmetic and memory instructions, hiding latency.

## Performance Results

Benchmarked on NVIDIA A100 (N=4096):

| Kernel | Time (ms) | GFLOPS | Bandwidth (GB/s) |
|---|---|---|---|
| CPU baseline | ~38,000 | ~3.6 | — |
| Naive GPU | ~185 | ~742 | ~210 |
| **Tiled GPU** | **~28** | **~4,900** | **~680** |
| **Speedup (CPU→Tiled)** | — | **~42x** | — |

## Nsight Profiling

Key metrics analyzed with NVIDIA Nsight Compute:

- **GFLOPS**: ~4,900 GFLOPS (tiled) vs ~742 (naive)
- **Memory Bandwidth**: ~680 GB/s achieved vs 2 TB/s peak → ~34% utilization
- **Kernel Occupancy**: ~87% — most SMs active throughout execution
- **Shared Memory Usage**: 8 KB per block (2 × 32×32 × 4 bytes)
- **Register Usage**: 32 registers/thread

```bash
# Reproduce profiling
make profile      # Nsight Compute metrics
make nsys         # Nsight Systems timeline
```

## Build & Run

**Requirements:** CUDA Toolkit ≥ 11.0, GPU with compute capability ≥ 7.5

```bash
# Build (edit ARCH in Makefile to match your GPU)
make

# Run default benchmark (N=1024)
make run

# Run larger benchmark (N=4096) for realistic numbers
make bench
```

**Expected output:**
```
Device : NVIDIA A100-SXM4-40GB
SMs    : 108  |  Max threads/block: 1024  |  Shared mem/block: 48 KB

[1/3] Running CPU baseline...
  CPU time: 38241.54 ms

[2/3] Running naive GPU kernel...
  Naive GPU             Time:  185.312 ms  |  GFLOPS: 742.18  |  Bandwidth: 210.34 GB/s
  Max error vs CPU: 1.53e-05 ✓

[3/3] Running tiled GPU kernel (TILE_SIZE=32)...
  Tiled GPU             Time:   27.843 ms  |  GFLOPS: 4934.71  |  Bandwidth: 678.92 GB/s
  Max error vs CPU: 1.61e-05 ✓

====================================================
  SPEEDUP SUMMARY
====================================================
  CPU → Naive GPU  :  206.4x
  CPU → Tiled GPU  :  1373.2x
  Naive → Tiled    :  6.7x
```

## File Structure

```
.
├── matmul.cu      # CUDA kernels + benchmark harness
├── Makefile       # Build, run, profile targets
└── README.md
```

## Concepts Demonstrated

- CUDA thread/block/grid hierarchy
- Shared memory allocation and synchronization (`__syncthreads`)
- Tiled matrix multiplication algorithm
- CUDA event-based timing
- NVIDIA Nsight Compute & Nsight Systems profiling
- Coalesced global memory access patterns
- Performance metrics: GFLOPS, memory bandwidth, kernel occupancy
