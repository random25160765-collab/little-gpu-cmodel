/*
 * tests/matmul_fast.h — 极限优化 SGEMM (AVX2 + FMA)
 * Copyright (c) 2024-2025, GPL v2
 *
 * native 侧 matmul 的参考实现, 用来衡量"同样算法在真 CPU 上能跑多快"。
 * 结构照搬 GotoBLAS/BLIS 的三层分块:
 *
 *   L0 寄存器层: 4×16 micro-kernel
 *       8 个 YMM 累加器 (4 行 × 2 向量) + 2 个 B 向量 + 4 个 A 广播 = 14/16 个 YMM,
 *       无溢出。每步 k: 8 条 FMA + 6 条 load, 每 4 拍 8 条 FMA = Zen3 的 2 FMA/拍。
 *   L1/L2 缓存层: MC×KC×NC 宏分块 + A/B 面板打包
 *       打包把"跨行取 A"和"按行取 B"变成纯连续流, 消除 TLB 与 L1 行抖动;
 *       B 面板在 L2 中被所有 A 块复用, A 面板 4 行常驻 L1。
 *
 *   多线程: 规模够大时 (MF_PAR_MIN) 用持久线程池按 ic 块切分,
 *           B 面板共享、A 面板每线程私有, 主线程也参与干活。
 *
 * 实测 (Ryzen 7 6800H, Zen3+, 8C/16T; 笔记本功耗墙, 16 线程 AVX2 满载会掉频):
 *   纯 FMA 链上限  1 线程 107 GFLOPS → 16 线程 439 GFLOPS (硬件只给 4.1x, 非 16x)
 *   128³    42µs    ~99 GFLOPS  (1 线程: 太小, 同步开销大于收益)
 *   1024³   24ms    ~93 GFLOPS  (1 线程)
 *   1024³  5.9ms   ~364 GFLOPS  (16 线程, 达纯 FMA 上限的 82%)
 *   噪声 ±20%: 温控使同一份代码在 6~10ms 间跳, 对比时要看多轮最小值
 *
 * 约束: M % 4 == 0 且 N % 16 == 0, 否则退回标量实现。
 * 语义: C = A × B 整体覆写, 调用方无需预先清零。
 */
#ifndef MATMUL_FAST_H
#define MATMUL_FAST_H

#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* --- micro-kernel 形状 --- */
#define MF_MR 4
#define MF_NR 16

/* --- 宏分块 (Ryzen 7 6800H 上 1024³ 实测最优) --- */
/* 可用 -DMF_MC=... 覆盖, 便于按机型/线程数扫参数 */
#ifndef MF_MC
#define MF_MC 32 /* 1024 行 → 32 个 ic 块, 16 线程每线程 2 块, 负载更均衡 */
#endif
#ifndef MF_KC
#define MF_KC 1024 /* k 方向一次走完: 免掉一整趟 C 读-改-写与一半屏障 */
#endif
#ifndef MF_NC
#define MF_NC 512
#endif

/* --- 多线程 --- */
#define MF_MAX_THREADS 32
/* 低于这个工作量 (M×N×K) 不开线程: 一次 (jc,pc) 块的唤醒+汇合约 20µs,
 * 128³ 整个 matmul 才 42µs, 开了反而更慢 */
#ifndef MF_PAR_MIN
#define MF_PAR_MIN (64u * 1024 * 1024)
#endif

/* ============================================================
 * 4×16 micro-kernel
 *
 * ap: 第 k 步的 4 个 A 元素 (连续), 步长 astride
 * bp: 第 k 步的 16 个 B 元素 (连续), 步长 bstride
 * c : 4 行 × 16 列的 C 子块左上角, 行距 ldc
 * first: 首个 k 分块时覆写 C, 否则累加
 * ============================================================ */
static void mf_kernel_4x16(int k, const float *ap, int astride, const float *bp, int bstride,
                           float *c, int ldc, int first)
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31;
    if (first) {
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
    } else {
        c00 = _mm256_loadu_ps(c + 0 * ldc + 0);
        c01 = _mm256_loadu_ps(c + 0 * ldc + 8);
        c10 = _mm256_loadu_ps(c + 1 * ldc + 0);
        c11 = _mm256_loadu_ps(c + 1 * ldc + 8);
        c20 = _mm256_loadu_ps(c + 2 * ldc + 0);
        c21 = _mm256_loadu_ps(c + 2 * ldc + 8);
        c30 = _mm256_loadu_ps(c + 3 * ldc + 0);
        c31 = _mm256_loadu_ps(c + 3 * ldc + 8);
    }
    for (int p = 0; p < k; p++) {
        __m256 b0 = _mm256_loadu_ps(bp + 0);
        __m256 b1 = _mm256_loadu_ps(bp + 8);
        __m256 a0 = _mm256_broadcast_ss(ap + 0);
        __m256 a1 = _mm256_broadcast_ss(ap + 1);
        __m256 a2 = _mm256_broadcast_ss(ap + 2);
        __m256 a3 = _mm256_broadcast_ss(ap + 3);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        ap += astride;
        bp += bstride;
    }
    _mm256_storeu_ps(c + 0 * ldc + 0, c00);
    _mm256_storeu_ps(c + 0 * ldc + 8, c01);
    _mm256_storeu_ps(c + 1 * ldc + 0, c10);
    _mm256_storeu_ps(c + 1 * ldc + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc + 0, c20);
    _mm256_storeu_ps(c + 2 * ldc + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc + 0, c30);
    _mm256_storeu_ps(c + 3 * ldc + 8, c31);
}

/* ============================================================
 * 面板打包: 把两个"跨行"访问变成连续流
 *   A: [i 块][k][4]   每个 k 的 4 个 A 元素挤在 16B 内 → 1 次取指喂 4 次广播
 *   B: [j 块][k][16]  每个 k 的 16 个 B 元素 = 2 条 AVX2 load
 * ============================================================ */
static void mf_pack_a(int mc, int kc, const float *a, int lda, float *ap)
{
    for (int i = 0; i < mc; i += MF_MR) {
        const float *a0 = a + (size_t)(i + 0) * lda;
        const float *a1 = a + (size_t)(i + 1) * lda;
        const float *a2 = a + (size_t)(i + 2) * lda;
        const float *a3 = a + (size_t)(i + 3) * lda;
        for (int p = 0; p < kc; p++) {
            ap[0] = a0[p];
            ap[1] = a1[p];
            ap[2] = a2[p];
            ap[3] = a3[p];
            ap += MF_MR;
        }
    }
}

static void mf_pack_b(int kc, int nc, const float *b, int ldb, float *bp)
{
    for (int j = 0; j < nc; j += MF_NR) {
        for (int p = 0; p < kc; p++) {
            __m256 b0 = _mm256_loadu_ps(b + (size_t)p * ldb + j + 0);
            __m256 b1 = _mm256_loadu_ps(b + (size_t)p * ldb + j + 8);
            _mm256_storeu_ps(bp + 0, b0);
            _mm256_storeu_ps(bp + 8, b1);
            bp += MF_NR;
        }
    }
}

/* --- 非对齐尺寸的标量兜底 (ikj 序, 便于向量化无关紧要) --- */
static void mf_sgemm_ref(int M, int N, int K, const float *A, int lda, const float *B, int ldb,
                         float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < K; k++) {
            float aik = A[(size_t)i * lda + k];
            const float *bk = B + (size_t)k * ldb;
            float *ci = C + (size_t)i * ldc;
            for (int j = 0; j < N; j++)
                ci[j] += aik * bk[j];
        }
    }
}

/* --- grow-only 静态 scratch: 避免每次调用都 malloc/缺页 --- */
static float *mf_scratch(size_t need)
{
    static float *buf = NULL;
    static size_t cap = 0;
    if (need > cap) {
        float *nb = NULL;
        if (posix_memalign((void **)&nb, 64, need))
            return NULL;
        free(buf);
        buf = nb;
        cap = need;
    }
    return buf;
}

/* ============================================================
 * 持久线程池: 规模够大时按 ic 块切分, B 面板共享, A 面板每线程私有
 * ============================================================ */
typedef struct {
    const float *A; /* 原始 A, 与 C 一样未经偏移 */
    float *C;
    float *ap;      /* A 面板池基址, 第 id 份在 ap + id*mc*kc */
    float *bp;      /* 打包好的 B 面板 (共享) */
    int M, lda, ldc;
    int mc, kc, nb, kb, nt;
    int jc, pc, first;
} mf_job;

struct mf_pool;
typedef struct {
    struct mf_pool *p;
    int id;
} mf_wctx;

typedef struct mf_pool {
    pthread_mutex_t mu;
    /* 必须用两个条件变量: 若共用, worker 报完成的 signal 会被"正在等任务"的
     * 另一个 worker 抢走消费掉, 主线程就永远等不到 done (实测直接死锁) */
    pthread_cond_t cv_job;  /* 主线程 → worker: 有新任务 */
    pthread_cond_t cv_done; /* worker → 主线程: 全部做完 */
    int gen;  /* 任务代数, 主线程发布时 ++ */
    int done; /* 已完成的工作线程数 */
    int nw;   /* 工作线程数 (不含主线程) */
    int nt;   /* 总参与数 = nw + 1 */
    mf_job job;
    mf_wctx wctx[MF_MAX_THREADS];
    pthread_t tids[MF_MAX_THREADS];
} mf_pool;

static mf_pool *g_mf_pool = NULL;
static int g_mf_nt_last = 1;

/* 干第 id 份 ic 块: id, id+nt, id+2*nt ... */
static void mf_slice(const mf_job *j, int id)
{
    float *ap = j->ap + (size_t)id * j->mc * j->kc;
    for (int ic = id * j->mc; ic < j->M; ic += j->nt * j->mc) {
        int mb = (j->M - ic < j->mc) ? (j->M - ic) : j->mc;
        mf_pack_a(mb, j->kb, j->A + (size_t)ic * j->lda + j->pc, j->lda, ap);
        for (int jj = 0; jj < j->nb; jj += MF_NR) {
            const float *bpj = j->bp + (size_t)(jj / MF_NR) * j->kb * MF_NR;
            for (int i = 0; i < mb; i += MF_MR)
                mf_kernel_4x16(j->kb, ap + (size_t)(i / MF_MR) * j->kb * MF_MR, MF_MR, bpj, MF_NR,
                               j->C + (size_t)(ic + i) * j->ldc + j->jc + jj, j->ldc, j->first);
        }
    }
}

static void *mf_worker(void *arg)
{
    mf_wctx *w = arg;
    mf_pool *p = w->p;
    int mygen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->gen == mygen)
            pthread_cond_wait(&p->cv_job, &p->mu);
        mygen = p->gen;
        mf_job j = p->job; /* 在锁内快照, 之后只读 */
        pthread_mutex_unlock(&p->mu);

        mf_slice(&j, w->id);

        pthread_mutex_lock(&p->mu);
        if (++p->done >= p->nw)
            pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
    return NULL;
}

static mf_pool *mf_pool_get(void)
{
    if (g_mf_pool)
        return g_mf_pool;
    int nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    const char *e = getenv("MF_THREADS");
    if (e && atoi(e) > 0)
        nt = atoi(e);
    if (nt < 1)
        nt = 1;
    if (nt > MF_MAX_THREADS)
        nt = MF_MAX_THREADS;
    mf_pool *p = calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_job, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->nw = nt - 1;
    for (int i = 0; i < p->nw; i++) {
        p->wctx[i].p = p;
        p->wctx[i].id = i + 1; /* 主线程是 0 号 */
        if (pthread_create(&p->tids[i], NULL, mf_worker, &p->wctx[i])) {
            p->nw = i; /* 起不来就用少几个 */
            break;
        }
    }
    p->nt = p->nw + 1;
    g_mf_pool = p;
    return p;
}

/* 该规模值不值得开线程: 工作量够大 且 任务数够分 */
static mf_pool *mf_pool_pick(int nic, int njc, size_t work)
{
    if (work < MF_PAR_MIN)
        return NULL;
    mf_pool *p = mf_pool_get();
    if (!p || nic * njc < 2 * p->nt)
        return NULL;
    return p;
}

/* 上一次 mf_sgemm 实际用的线程数 */
static int mf_last_nthreads(void)
{
    return g_mf_nt_last;
}

/* ============================================================
 * C(M×N) = A(M×K) × B(K×N), 行主序, C 被整体覆写
 * ============================================================ */
static void mf_sgemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb,
                     float *C, int ldc)
{
    if ((M % MF_MR) || (N % MF_NR) || M <= 0 || N <= 0 || K <= 0) {
        mf_sgemm_ref(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    /* 分块尺寸按实际问题裁剪, 小矩阵不会白占 3MB */
    int mc = (MF_MC < M) ? MF_MC : M;
    int kc = (MF_KC < K) ? MF_KC : K;
    int nc = (MF_NC < N) ? MF_NC : N;
    int nic = (M + mc - 1) / mc, njc = (N + nc - 1) / nc;
    mf_pool *p = mf_pool_pick(nic, njc, (size_t)M * N * K);
    int nt = p ? p->nt : 1;
    g_mf_nt_last = nt;

    float *buf = mf_scratch(((size_t)nt * mc * kc + (size_t)nc * kc) * sizeof(float));
    if (!buf) {
        mf_sgemm_ref(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    mf_job job;
    job.A = A;
    job.C = C;
    job.ap = buf;
    job.bp = buf + (size_t)nt * mc * kc;
    job.M = M;
    job.lda = lda;
    job.ldc = ldc;
    job.mc = mc;
    job.kc = kc;
    job.nt = nt;

    /* jc 外层 → B 面板常驻 L2 被所有 ic 复用
     * pc 中层 → k 方向切块, 首块覆写 C, 其余累加
     * ic 内层 → 多线程切分点; 每块 A 被打进 L1 */
    for (int jc = 0; jc < N; jc += nc) {
        job.nb = (N - jc < nc) ? (N - jc) : nc;
        for (int pc = 0; pc < K; pc += kc) {
            job.kb = (K - pc < kc) ? (K - pc) : kc;
            job.first = (pc == 0);
            job.jc = jc;
            job.pc = pc;
            mf_pack_b(job.kb, job.nb, B + (size_t)pc * ldb + jc, ldb, job.bp);
            if (!p) {
                mf_slice(&job, 0);
                continue;
            }
            pthread_mutex_lock(&p->mu);
            p->job = job;
            p->done = 0;
            p->gen++;
            pthread_cond_broadcast(&p->cv_job);
            pthread_mutex_unlock(&p->mu);

            mf_slice(&job, 0); /* 主线程也干活 */

            pthread_mutex_lock(&p->mu);
            while (p->done < p->nw)
                pthread_cond_wait(&p->cv_done, &p->mu);
            pthread_mutex_unlock(&p->mu);
        }
    }
}

#endif /* MATMUL_FAST_H */
