[English](README.md) | [中文](README_zh.md)

# Little GPU CModel

A lightweight functional / performance model（性能模型）of a GPU compute unit, built around a **RISC-V RV32IMFV ISA** with **RVV 1.0 vector extensions** and custom tensor / special-function / low-precision accelerators. The simulator executes GPU kernels compiled to RISC-V binaries on a **SIMT (Single Instruction, Multiple Threads) engine** with 32-wide warps, divergence tracking, and barrier synchronization.

> ⚠️ **Status:** Work in progress. Several subsystems (TCU, RVV corner cases, Go pool manager) are incomplete or stub-level. See [Current Limitations](#current-limitations) below.

## Architecture Overview

```
RISC-V kernel (.S) ──► customasm.py ──► riscv-gcc ──► .bin (raw RISC-V)
                                                              │
                         ┌────────────────────────────────────┘
                         ▼
              simd_predecode()   ← trie-based opcode matcher
                         │
                         ▼
              ThOp[] dispatch table (pre-decoded instructions)
                         │
                         ▼
              engine_exec()   ← computed-goto threaded interpreter
                         │
                         ▼
              32-lane SIMT execution (SoA register layout)
```

### Key Components

| Component | Description |
|-----------|-------------|
| [core/engine.c](core/engine.c) | Heart of the simulator — computed-goto threaded interpreter with per-lane SIMD execution, SIMT divergence stack, and barrier handling |
| [core/scheduler.c](core/scheduler.c) | Kernel scheduler — grid → block → warp → lane decomposition, pthread-based block parallelism |
| [core/decode_trie.c](core/decode_trie.c) | Binary prefix trie for fast instruction decoding (~7 bit checks average) |
| [core/vram.c](core/vram.c) | Virtual GPU memory with sector cache tracking and shared memory region |
| [core/sfu.h](core/sfu.h) | Special Function Unit — fast math approximations (exp, sigmoid, tanh, rsqrt) with ML-grade precision |
| [lib/lpfp.c](lib/lpfp.c) | Low-precision float conversions: BF16, FP8 (E4M3, E5M2), FP4 (E2M1) |
| [inst/](inst/) | Instruction code generation pipeline — YAML specs → Python generators → C handler headers |
| [gosched/](gosched/) | Go-based alternative scheduler with 4 scheduling policies (CGo bridge to libengine.a) |
| [fuzz/](fuzz/) | Random instruction fuzzer with cross-mask validation for SIMT correctness |
| [kernels/](kernels/) | RISC-V assembly test kernels: vector ops, matmul, softmax, GELU, conv2d, memcpy, etc. |

### Instruction Set Coverage

| Extension | Status | Notes |
|-----------|--------|-------|
| RV32I | ✅ Complete | Base integer: ALU, branches, jumps, loads/stores |
| RV32M | ✅ Complete | Multiply / divide |
| RV32F | ✅ Complete | Single-precision float, FMA, conversions, classify |
| RV32A | ✅ Complete | Atomics (LR/SC, AMO) |
| Zicsr | ✅ Complete | CSR read / write |
| RVV 1.0 | ⚠️ ~85-90% | 33 instruction families, ~507 generated handlers. Some vector operations are stubbed (`/* TODO */` in generator) |
| Custom SFU | ✅ Complete | Fast exp, sigmoid, tanh, rsqrt, sin, cos (scalar + vector) |
| Custom LP | ✅ Complete | BF16 ↔ FP32, FP8 ↔ FP32, FP4 ↔ FP32 conversions |
| Custom MMA/TCU | ⚠️ Stub | Hand-written mma.zero/ld/s/st/relu/bias — basic matmul works, but not yet migrated to spec+generator pipeline |

### SIMT Execution Model

- **Grid → Block → Warp → Thread** hierarchy (CUDA-style)
- 32 threads per warp, **Structure of Arrays** (SoA) register layout for host compiler auto-vectorization
- **Divergence tracking** — per-lane active mask with reconvergence stack (`DIV_BR` macro)
- **Barrier synchronization** — warp yields with resume PC, scheduler coordinates all warps

### Scheduling Policies (Go Scheduler)

The Go scheduler in [gosched/scheduler.go](gosched/scheduler.go) implements four policies:

| Policy | Description |
|--------|-------------|
| Round-Robin | All warps concurrent, semaphore-limited |
| Greedy | Block-by-block, warps within a block concurrent |
| GTO (Greedy-Then-Oldest) | First wave concurrent, then oldest-first FIFO |
| Wavefront | One warp per block per wave, block-synchronized via barriers |

## Quick Start

### Dependencies

```bash
# One-shot install
./setup_env.sh
```

Or manually:

```bash
sudo apt install build-essential liblua5.4-dev          # C toolchain + Lua config
sudo apt install gcc-riscv64-unknown-elf                  # RISC-V cross compiler
pip install pyyaml                                        # Python YAML for code gen
```

Optional: Go 1.25+ (for Go scheduler), clang-format (for formatting).

### Build & Run

```bash
# Full build + run all tests
make test

# Functional correctness tests only
./perf_test.elf func

# Performance benchmarks
./perf_test.elf bench

# Native C comparison (interpreter vs hand-written C)
./perf_test.elf native

# Address sanitizer build
make asan

# Go scheduler
make go-build          # builds gpu_sched binary
make go-test           # runs Go test suite

# Code formatting
make format

# Line count
make count
```

### Running the Go Scheduler

```bash
./gpu_sched --policy=gto kernels/vecmul.bin 2048 1 1 32 1 1
#           gridDimX Y Z  blockDimX Y Z
```

### Configuration

Edit [gpu_config.lua](gpu_config.lua):

```lua
return {
    device = {
        num_cus      = 0,   -- 0 = auto (CPU core count)
        warps_per_cu = 1,
        warp_size    = 32,
        vram_mb      = 64,
    },
    features = {
        vpu    = true,   -- RVV vector accelerator
        tcu    = true,   -- tensor accelerator (WIP)
        sfu    = true,   -- special function unit
        lp     = true,   -- low-precision float
        debug  = false,  -- event ring buffer (WIP)
        trace  = false,  -- instruction-level trace (WIP)
        perf   = false,  -- hot-path performance counters
    },
}
```

### Test Kernels

| Kernel | What It Tests |
|--------|---------------|
| `vector_add` / `fvec_add` | Integer / FP scalar vector add |
| `vecmul` / `vecmul_vpu` | RVV vector multiply (vfmul.vv) |
| `saxpy` | SAXPY (scalar × vector + vector) |
| `scal_mul` | Scalar-vector multiply (vfmul.vf) |
| `dot_product` | FP dot product with fmadd.s |
| `matmul` | TCU matrix multiply (mma.* instructions) |
| `memcpy` / `mem_access` | Bulk copy and sub-word access |
| `rv32m` | Multi-cycle multiply / divide |
| `rv32f` | Full RV32F coverage |
| `fib` | Fibonacci — branch-heavy divergence test |
| `gelu` / `softmax` / `softmax_norm` | ML activation kernels |
| `conv2d` | 2D convolution (no padding, stride=1) |
| `vadd_int` / `vadd_vl` | RVV integer add, VL setting test |

## Project Structure

```
little-gpu-cmodel/
├── main.c                  # CLI entry point
├── config.c / config.h     # Lua-based configuration loader
├── state.h                 # GPGPUState — the full machine state
├── stats.c / stats.h       # Performance counter collection & display
├── test_runner.c / .h      # Test framework & registration
├── gpu_config.lua          # Default config
├── Makefile                # Build system
├── setup_env.sh            # Dependency installer
│
├── core/                   # Simulator engine
│   ├── engine.c / .h       # Threaded interpreter (the core loop)
│   ├── engine_types.h      # SIMTFrame, EngineContext types
│   ├── engine_bridge.c/.h  # CGo-compatible wrappers
│   ├── scheduler.c / .h    # Kernel scheduler (C, pthread-based)
│   ├── gpgpu_core.h        # GPGPUWarp, GPGPULane, register types
│   ├── soa.h               # AoS ↔ SoA marshalling
│   ├── predecode.h         # Trie-based predecoder
│   ├── decode.h / .trie.c  # Instruction decoder & trie builder
│   ├── dispatch.h          # Dispatch table infrastructure
│   ├── inst.h              # Instruction parsing macros
│   ├── vram.c / .h         # GPU memory read/write + sector cache
│   ├── vram_alloc.c / .h   # Best-fit VRAM allocator
│   ├── sfu.h               # Fast math approximations
│   ├── table.c / .h        # ASCII table renderer
│   └── utils.h             # Terminal color macros
│
├── module/                 # Hardware unit handlers (included into engine.c)
│   ├── modules.h           # Aggregates all modules
│   ├── ctrl.h              # Control unit + DIV_BR divergence macro
│   ├── alu.h               # RV32I/M ALU
│   ├── fpu.h               # RV32F FPU + LP conversions
│   ├── lsu.h               # Load/store unit + atomics
│   ├── sfu.h               # Special function unit handlers
│   ├── vpu.h               # RVV vector unit handlers
│   ├── tcu.h               # Tensor compute unit (stub)
│   └── misc.h              # ebreak, done, barrier, tex
│
├── inst/                   # Instruction code generation pipeline
│   ├── Makefile / build_all.py / gen_dispatch.py
│   ├── rv32i/ rv32m/ rv32f/ rv32a/ rv32zicsr/   # Standard RISC-V
│   ├── rvv/                                       # RISC-V Vector 1.0
│   ├── sf/ lp/                                    # Custom SFU / LP float
│   └── mma/                                       # Custom MMA (hand-written)
│
├── kernels/                # RISC-V assembly test programs (*.S → *.bin)
├── tests/                  # C test suite (func, bench, native, register)
├── fuzz/                   # Random instruction fuzzer
├── gosched/                # Go scheduler + pool manager
├── lib/                    # Low-precision float library (lpfp.c)
├── tools/                  # Custom assembler, kernel validator
└── proto/                  # Event/instruction code constants
```

## Current Limitations

### Incomplete / Work-in-Progress

- **TCU (Tensor Compute Unit):** Marked as WIP in config. MMA instruction handlers are hand-written stubs covering basic matmul (zero/load/mma/store/relu/bias). Not yet integrated into the YAML spec → generator pipeline. No support for mixed-precision, block-tiling, or multi-accumulator chaining.
- **RVV 1.0 corner cases:** Some vector instructions fall through to a `/* TODO */` stub in the code generator. Instructions like `vpopc`, `vfirst`, `vid` (full), and certain widening/narrowing patterns may return identity values rather than computed results.
- **Debug / trace events:** The `debug` and `trace` feature flags exist in the config but the runtime plumbing (ring buffer, event emission, trace dump) is incomplete.
- **Fuzzer coverage:** The fuzzer only generates safe ALU/FP/branch ops — it excludes memory, CSR, custom, and vector instructions.
- **Go pool manager:** The `gosched/poolmgr/` daemon and HTTP API are experimental.
- **Multi-CU parallelism:** The C scheduler uses pthreads for block-level parallelism, but true multi-CU device-level scheduling with inter-CU communication is not implemented.
- **No JIT / native compilation:** Pure interpretation. No dynamic binary translation or ahead-of-time lowering to host ISA.

### Known Design Choices

- **No FP64 support** — the FPU is RV32F only (single precision).
- **SEW limited to 8/16/32** — RVV 64-bit element width is not implemented.
- **VRAM is host memory** — `calloc`-backed, not an actual device memory model with realistic bandwidth/latency simulation.
- **Warp size is fixed at 32** — not configurable.

## Performance Notes

- The threaded interpreter uses GCC's **labels-as-values** (`&&label`) extension for zero-overhead computed-goto dispatch — similar to LuaJIT or QEMU interpreters.
- SoA register layout enables **host compiler auto-vectorization (AVX2)** on the per-lane execution loops.
- Fast math functions (SFU) achieve <1% error at 3-5× speed of libm equivalents.
- On a typical x86-64 host, the interpreter achieves ~10-50 MOPS/s for vector workloads, depending on instruction mix.

## License

GPL v2 — see source headers.

## Acknowledgements

This project builds on the RISC-V ISA specifications and uses:
- The RISC-V Vector Extension 1.0 (frozen spec)
- GCC labels-as-values extension for threaded interpretation
- Quake III fast inverse square root algorithm
- Vortex GPU memory allocator (best-fit block allocator)
