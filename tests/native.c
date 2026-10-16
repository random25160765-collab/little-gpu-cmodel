/*
 * tests/native.c — Native C comparison (interpreter vs hand-written C)
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "../state.h"
#include "../core/scheduler.h"
#include "../core/vram_alloc.h"
#include "matmul_fast.h"

#define KERN_ADDR 0x500000
extern void load_kernel(const char *path, GPGPUState *s, uint32_t addr);

static void run_native_vecmul(GPGPUState *s)
{
    uint32_t N = 65536, nw = N / 32;
    struct timespec T0, T1;

    vram_alloc_reset(s);
    uint32_t A_base = vram_alloc(s, N * 4);
    uint32_t B_base = vram_alloc(s, N * 4);
    uint32_t C_base = vram_alloc(s, N * 4);
    vram_ptr_write(s, PTR_SLOT_A, A_base);
    vram_ptr_write(s, PTR_SLOT_B, B_base);
    vram_ptr_write(s, PTR_SLOT_C, C_base);

    for (uint32_t i = 0; i < N; i++) {
        ((float *)(s->vram_ptr + A_base))[i] = (float)(i % 256);
        ((float *)(s->vram_ptr + B_base))[i] = 3.0f;
    }
    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0] = nw;
    s->kernel.grid_dim[1] = 1;
    s->kernel.grid_dim[2] = 1;
    s->kernel.block_dim[0] = 1;
    s->kernel.block_dim[1] = 1;
    s->kernel.block_dim[2] = 1;
    load_kernel("kernels/vecmul.bin", s, KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double ti = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;

    volatile float sum = 0;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    for (uint32_t w = 0; w < nw; w++) {
        for (int l = 0; l < 32; l++) {
            uint32_t t = w * 32 + l;
            float a = *(float *)(s->vram_ptr + A_base + t * 4);
            float b = *(float *)(s->vram_ptr + B_base + t * 4);
            *(float *)(s->vram_ptr + C_base + t * 4) = a * b;
            sum += a * b;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double tn = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;
    printf("  Interp: %5.0fus  Native: %5.0fus  Ratio: %.1fx\n", ti * 1e6, tn * 1e6, ti / tn);
    (void)sum;
}

/* 解释器侧的线程数: scheduler 用 min(核数, block 数), 这里照抄以便对比 */
static int interp_threads(int nb)
{
    int nw = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nw < 1)
        nw = 4;
    if (nw > nb)
        nw = nb;
    return nw;
}

/* 预热 + 计时一次 native SGEMM (缓存/页表/scratch/线程池都在计时之外) */
static double bench_sgemm(int M, int N, int K, float *A, float *B, float *C)
{
    struct timespec T0, T1;
    mf_sgemm(M, N, K, A, K, B, N, C, N);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    mf_sgemm(M, N, K, A, K, B, N, C, N);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    return (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;
}

static void run_native_matmul(GPGPUState *s)
{
    int M = 128, K = 128, N = 128;
    struct timespec T0, T1;

    vram_alloc_reset(s);
    *(uint32_t *)s->vram_ptr = K;
    uint32_t A_base = vram_alloc(s, M * K * 4);
    uint32_t B_base = vram_alloc(s, K * N * 4);
    uint32_t C_base = vram_alloc(s, M * N * 4);
    vram_ptr_write(s, PTR_SLOT_A, A_base);
    vram_ptr_write(s, PTR_SLOT_B, B_base);
    vram_ptr_write(s, PTR_SLOT_C, C_base);

    float *A = (float *)(s->vram_ptr + A_base), *B = (float *)(s->vram_ptr + B_base);
    float *C = (float *)(s->vram_ptr + C_base);
    for (int i = 0; i < M * K; i++)
        A[i] = 1.0f;
    for (int i = 0; i < K * N; i++)
        B[i] = 1.0f;

    s->kernel.kernel_addr = KERN_ADDR;
    s->kernel.grid_dim[0] = M;
    s->kernel.grid_dim[1] = 1;
    s->kernel.grid_dim[2] = 1;
    s->kernel.block_dim[0] = N;
    s->kernel.block_dim[1] = 1;
    s->kernel.block_dim[2] = 1;
    load_kernel("kernels/matmul.bin", s, KERN_ADDR);
    clock_gettime(CLOCK_MONOTONIC, &T0);
    scheduler_run_kernel(s);
    clock_gettime(CLOCK_MONOTONIC, &T1);
    double ti = (T1.tv_sec - T0.tv_sec) + (T1.tv_nsec - T0.tv_nsec) * 1e-9;

    double tn = bench_sgemm(M, N, K, A, B, C);
    /* A/B 全 1 → C 应处处等于 K; 顺便当防死代码消除的屏障 (在计时之外) */
    for (int i = 0; i < M * N; i += 251) {
        if (fabsf(C[i] - (float)K) > 1e-3f) {
            fprintf(stderr, "  native matmul FAIL: C[%d]=%.3f, want %d\n", i, C[i], K);
            break;
        }
    }
    printf("  Interp: %5.0fus (%d 线程)  Native: %5.0fus (%d 线程)  Ratio: %.1fx\n", ti * 1e6,
           interp_threads(M), tn * 1e6, mf_last_nthreads(), ti / tn);
}

/* 1024³: 单独给出 native 侧的绝对性能, 与 bench 里 matmul 1024 的口径对齐 */
static void run_native_matmul_1024(GPGPUState *s)
{
    int M = 1024, K = 1024, N = 1024;

    vram_alloc_reset(s);
    uint32_t A_base = vram_alloc(s, M * K * 4);
    uint32_t B_base = vram_alloc(s, K * N * 4);
    uint32_t C_base = vram_alloc(s, M * N * 4);
    float *A = (float *)(s->vram_ptr + A_base), *B = (float *)(s->vram_ptr + B_base);
    float *C = (float *)(s->vram_ptr + C_base);
    for (int i = 0; i < M * K; i++)
        A[i] = (float)(i & 1023) * 0.001f;
    for (int i = 0; i < K * N; i++)
        B[i] = (float)(i & 511) * 0.001f;

    double tn = bench_sgemm(M, N, K, A, B, C);
    volatile float sum = 0;
    for (int i = 0; i < M * N; i += 4096)
        sum += C[i];
    (void)sum;
    printf("  Native: %6.0fus (%d 线程)  %6.1f GFLOPS\n", tn * 1e6, mf_last_nthreads(),
           2.0 * (double)M * N * K / tn / 1e9);
}

void test_run_native(GPGPUState *s)
{
    printf("=== Native C Comparison ===\n\n");
    printf("--- vecmul ---\n");
    run_native_vecmul(s);
    printf("--- matmul 128 ---\n");
    run_native_matmul(s);
    printf("--- matmul 1024 ---\n");
    run_native_matmul_1024(s);
    printf("\n");
}
