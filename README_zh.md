[English](README.md) | [中文](README_zh.md)

# Little GPU CModel

一个轻量级的 GPU 计算核心功能 / 性能模型（performance model），基于 **RISC-V RV32IMFV 指令集**，扩展了 **RVV 1.0 向量指令**以及自定义的矩阵张量、特殊函数、低精度浮点加速单元。模拟器以 **SIMT（单指令多线程）** 方式在 32 路 warp 上执行编译为 RISC-V 二进制的 GPU kernel，支持分支发散追踪和 barrier 同步。

> ⚠️ **状态：** 开发中。部分子系统（TCU、RVV 角落指令、Go pool manager 等）尚未完成或停留在 stub 阶段。详见[已知局限](#已知局限)。

## 架构概览

```
RISC-V kernel (.S) ──► customasm.py ──► riscv-gcc ──► .bin (原始 RISC-V 二进制)
                                                              │
                         ┌────────────────────────────────────┘
                         ▼
              simd_predecode()   ← trie 前缀树操作码匹配
                         │
                         ▼
              ThOp[] 调度表（预解码后的指令序列）
                         │
                         ▼
              engine_exec()   ← computed-goto 线程解释器
                         │
                         ▼
              32-lane SIMT 执行（SoA 寄存器排布）
```

### 核心组件

| 组件 | 说明 |
|-----------|-------------|
| [core/engine.c](core/engine.c) | 模拟器心脏 —— computed-goto 线程解释器，支持逐 lane 的 SIMD 执行、SIMT 发散栈、barrier 处理 |
| [core/scheduler.c](core/scheduler.c) | Kernel 调度器 —— grid → block → warp → lane 分解，基于 pthread 的 block 级并行 |
| [core/decode_trie.c](core/decode_trie.c) | 二进制前缀树（trie）快速指令解码，平均约 7 次位检查即可定位指令 |
| [core/vram.c](core/vram.c) | 虚拟 GPU 显存，带 sector cache 追踪和共享内存区域 |
| [core/sfu.h](core/sfu.h) | 特殊函数单元 —— 快速数学近似（exp、sigmoid、tanh、rsqrt），ML 级精度 |
| [lib/lpfp.c](lib/lpfp.c) | 低精度浮点转换：BF16、FP8（E4M3、E5M2）、FP4（E2M1） |
| [inst/](inst/) | 指令代码生成管线 —— YAML 规格 → Python 生成器 → C handler 头文件 |
| [gosched/](gosched/) | Go 语言实现的替代调度器，支持 4 种调度策略（通过 CGo 桥接 libengine.a） |
| [fuzz/](fuzz/) | 随机指令 fuzzer，通过交叉 active mask 验证 SIMT 正确性 |
| [kernels/](kernels/) | RISC-V 汇编测试 kernel：向量运算、矩阵乘、softmax、GELU、conv2d、memcpy 等 |

### 指令集覆盖度

| 扩展 | 状态 | 备注 |
|-----------|--------|-------|
| RV32I | ✅ 完整 | 基础整数指令：ALU、分支、跳转、访存 |
| RV32M | ✅ 完整 | 乘除法 |
| RV32F | ✅ 完整 | 单精度浮点、FMA、类型转换、分类 |
| RV32A | ✅ 完整 | 原子操作（LR/SC、AMO） |
| Zicsr | ✅ 完整 | CSR 读写 |
| RVV 1.0 | ⚠️ 约 85-90% | 33 个指令族，约 507 个生成的 handler。部分向量操作为 stub（生成器中标注 `/* TODO */`） |
| 自定义 SFU | ✅ 完整 | 快速 exp、sigmoid、tanh、rsqrt、sin、cos（标量 + 向量） |
| 自定义 LP | ✅ 完整 | BF16 ↔ FP32、FP8 ↔ FP32、FP4 ↔ FP32 转换 |
| 自定义 MMA/TCU | ⚠️ Stub | 手写 mma.zero/ld/s/st/relu/bias —— 基础矩阵乘可用，尚未迁移到 spec + 生成器管线 |

### SIMT 执行模型

- **Grid → Block → Warp → Thread** 层级（CUDA 风格）
- 每 warp 32 线程，**SoA（Structure of Arrays）** 寄存器排布，利于宿主编译器自动向量化
- **分支发散追踪** —— 逐 lane 的 active mask 与重收敛栈（`DIV_BR` 宏）
- **Barrier 同步** —— warp 让出执行权并携带恢复 PC，调度器协调所有 warp

### 调度策略（Go 调度器）

[gosched/scheduler.go](gosched/scheduler.go) 中的 Go 调度器实现了四种策略：

| 策略 | 说明 |
|--------|-------------|
| Round-Robin | 所有 warp 并发，信号量限流 |
| Greedy | 逐 block 执行，block 内 warp 并发 |
| GTO（Greedy-Then-Oldest） | 第一波并发执行，之后最旧 warp 优先（FIFO） |
| Wavefront | 每 wave 每 block 一个 warp，所有 block 通过 barrier 同步后推进到下一 warp |

## 快速开始

### 依赖安装

```bash
# 一键安装
./setup_env.sh
```

或手动安装：

```bash
sudo apt install build-essential liblua5.4-dev          # C 工具链 + Lua 配置
sudo apt install gcc-riscv64-unknown-elf                  # RISC-V 交叉编译器
pip install pyyaml                                        # Python YAML（代码生成用）
```

可选：Go 1.25+（Go 调度器）、clang-format（代码格式化）。

### 构建与运行

```bash
# 完整构建 + 运行全部测试
make test

# 仅功能正确性测试
./perf_test.elf func

# 性能基准测试
./perf_test.elf bench

# 与原生 C 对照
./perf_test.elf native

# Address Sanitizer 构建
make asan

# Go 调度器
make go-build          # 构建 gpu_sched 二进制
make go-test           # 运行 Go 测试套件

# 代码格式化
make format

# 代码行数统计
make count
```

### 运行 Go 调度器

```bash
./gpu_sched --policy=gto kernels/vecmul.bin 2048 1 1 32 1 1
#           gridDimX Y Z  blockDimX Y Z
```

### 配置文件

编辑 [gpu_config.lua](gpu_config.lua)：

```lua
return {
    device = {
        num_cus      = 0,   -- 0 = 自动（等于 CPU 核心数）
        warps_per_cu = 1,
        warp_size    = 32,
        vram_mb      = 64,
    },
    features = {
        vpu    = true,   -- RVV 向量加速器
        tcu    = true,   -- 张量加速器（开发中）
        sfu    = true,   -- 特殊函数单元
        lp     = true,   -- 低精度浮点
        debug  = false,  -- 事件环形缓冲区（开发中）
        trace  = false,  -- 指令级 trace（开发中）
        perf   = false,  -- 热路径性能计数器
    },
}
```

### 测试 Kernel 一览

| Kernel | 测试内容 |
|--------|---------------|
| `vector_add` / `fvec_add` | 整型 / 浮点标量向量加法 |
| `vecmul` / `vecmul_vpu` | RVV 向量乘法（vfmul.vv） |
| `saxpy` | SAXPY（标量 × 向量 + 向量） |
| `scal_mul` | 标量-向量乘法（vfmul.vf） |
| `dot_product` | 浮点点积（fmadd.s） |
| `matmul` | TCU 矩阵乘法（mma.* 指令） |
| `memcpy` / `mem_access` | 批量拷贝与不对齐访存 |
| `rv32m` | 多周期乘除法 |
| `rv32f` | 完整 RV32F 覆盖 |
| `fib` | 斐波那契 —— 分支密集型发散测试 |
| `gelu` / `softmax` / `softmax_norm` | ML 激活函数 kernel |
| `conv2d` | 2D 卷积（无 padding，stride=1） |
| `vadd_int` / `vadd_vl` | RVV 整数加、VL 设置测试 |

## 项目结构

```
little-gpu-cmodel/
├── main.c                  # CLI 入口
├── config.c / config.h     # 基于 Lua 的配置加载器
├── state.h                 # GPGPUState —— 完整机器状态
├── stats.c / stats.h       # 性能计数器采集与展示
├── test_runner.c / .h      # 测试框架与注册
├── gpu_config.lua          # 默认配置
├── Makefile                # 构建系统
├── setup_env.sh            # 依赖安装脚本
│
├── core/                   # 模拟器引擎
│   ├── engine.c / .h       # 线程解释器（核心循环）
│   ├── engine_types.h      # SIMTFrame、EngineContext 类型定义
│   ├── engine_bridge.c/.h  # CGo 兼容封装
│   ├── scheduler.c / .h    # Kernel 调度器（C 版本，基于 pthread）
│   ├── gpgpu_core.h        # GPGPUWarp、GPGPULane、寄存器类型
│   ├── soa.h               # AoS ↔ SoA 数据布局转换
│   ├── predecode.h         # 基于 trie 的预解码器
│   ├── decode.h / .trie.c  # 指令解码器与 trie 构建
│   ├── dispatch.h          # 调度表基础设施
│   ├── inst.h              # 指令解析宏
│   ├── vram.c / .h         # GPU 显存读写 + sector cache
│   ├── vram_alloc.c / .h   # 最佳适配显存分配器
│   ├── sfu.h               # 快速数学近似
│   ├── table.c / .h        # ASCII 表格渲染
│   └── utils.h             # 终端颜色/样式宏
│
├── module/                 # 硬件单元 handler（被 engine.c #include 引入）
│   ├── modules.h           # 聚合全部模块
│   ├── ctrl.h              # 控制单元 + DIV_BR 发散宏
│   ├── alu.h               # RV32I/M ALU
│   ├── fpu.h               # RV32F FPU + LP 转换
│   ├── lsu.h               # 访存单元 + 原子操作
│   ├── sfu.h               # 特殊函数单元 handler
│   ├── vpu.h               # RVV 向量单元 handler
│   ├── tcu.h               # 张量计算单元（stub）
│   └── misc.h              # ebreak、done、barrier、tex
│
├── inst/                   # 指令代码生成管线
│   ├── Makefile / build_all.py / gen_dispatch.py
│   ├── rv32i/ rv32m/ rv32f/ rv32a/ rv32zicsr/   # 标准 RISC-V
│   ├── rvv/                                       # RISC-V Vector 1.0
│   ├── sf/ lp/                                    # 自定义 SFU / 低精度浮点
│   └── mma/                                       # 自定义 MMA（手写）
│
├── kernels/                # RISC-V 汇编测试程序（*.S → *.bin）
├── tests/                  # C 测试套件（func、bench、native、register）
├── fuzz/                   # 随机指令 fuzzer
├── gosched/                # Go 调度器 + pool manager
├── lib/                    # 低精度浮点库（lpfp.c）
├── tools/                  # 自定义汇编器、kernel 校验器
└── proto/                  # 事件/指令编码常量
```

## 已知局限

### 未完成 / 开发中

- **TCU（张量计算单元）:** 配置中标记为 WIP。MMA 指令 handler 为手写 stub，覆盖基本矩阵乘（zero/load/mma/store/relu/bias）。尚未接入 YAML spec → 生成器管线。不支持混合精度、分块（tiling）或多累加器流水。
- **RVV 1.0 角落指令:** 部分向量指令在代码生成器中落入 `/* TODO */` 占位分支。`vpopc`、`vfirst`、`vid`（完整实现）等指令以及某些扩宽/缩窄模式返回恒等值而非计算结果。
- **Debug / trace 事件:** 配置中存在 `debug` 和 `trace` 开关，但运行时的环形缓冲区、事件发送、trace 导出等链路尚未完成。
- **Fuzzer 覆盖范围:** Fuzzer 仅生成安全的 ALU/FP/分支指令 —— 排除了访存、CSR、自定义和向量指令。
- **Go pool manager:** `gosched/poolmgr/` 守护进程和 HTTP API 属于实验性质。
- **多 CU 并行:** C 调度器使用 pthread 做 block 级并行，但真正的多 CU 设备级调度、CU 间通信尚未实现。
- **无 JIT / 原生编译:** 纯解释执行。无动态二进制翻译或 ahead-of-time 降级到宿主 ISA。

### 已知设计取舍

- **无 FP64 支持** —— FPU 仅实现 RV32F（单精度）。
- **SEW 限制在 8/16/32** —— RVV 64 位元素宽度未实现。
- **VRAM 即宿主内存** —— 底层为 `calloc`，并非带有真实带宽/延迟建模的设备内存。
- **Warp 大小固定为 32** —— 不可配置。

## 性能说明

- 线程解释器使用 GCC 的 **labels-as-values**（`&&label`）扩展实现零开销 computed-goto 调度 —— 类似于 LuaJIT 或 QEMU 的解释器设计。
- SoA 寄存器排布使宿主编译器能对逐 lane 执行循环进行**自动向量化（AVX2）**。
- 快速数学函数（SFU）在 <1% 误差内达到 libm 等效函数 3-5× 的速度。
- 在典型 x86-64 宿主机上，解释器对向量负载可达约 10-50 MOPS/s，具体取决于指令混合比例。

## 许可证

GPL v2 —— 详见源文件头部。

## 致谢

本项目基于 RISC-V ISA 规范构建，使用了以下技术：
- RISC-V Vector Extension 1.0（冻结规范）
- GCC labels-as-values 扩展（线程解释）
- Quake III 快速平方根倒数算法
- Vortex GPU 内存分配器（最佳适配块分配器）
