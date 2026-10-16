# 元编程链条

```
                    编译期                                    加载/运行时
 ───────────────────────────────────────── ──────────────────────────────────

 spec.yaml         handles/*.h        dispatch元信息        kernel .bin
 (人手写)           (机器生成)          (机器生成)            (交叉编译产物)
     │                  │                   │                    │
     │  gen_{isa}.py    │  gen_dispatch.py  │                    │
     │  ──────────────► │  ────────────────►│                    │
     │  模板展开         │  全局聚合          │                    │
     │                  │                   │                    │
     │                  │                   │   module/*.h       │
     │                  │                   │   (人手写归类)       │
     │                  │                   │   ─────────        │
     │                  │                   │   按硬件模块 include │
     │                  │                   │                    │
     ▼                  ▼                   ▼                    ▼
  op_addi:           INSTRUCTION_LIST    engine_exec()       simd_predecode()
  { ... NEXT() }     DISP_ enum          体内 #include        ────────────────
  op_xori:           NUM_OF_INST         全部 handlers        trie 匹配每条 inst
  { ... NEXT() }     RVV_REBIND 宏       展开为巨型函数        → ThOp[].handler
     │                                                                 │
     │                                            SIMD_RESOLVE_HANDLERS│
     │                                            ─────────────────────│
     │                                             handler==NULL→illegal│
     │                                             否则→dispatch[idx]   │
     │                                                                 │
     ▼                                                                 ▼
  computed-goto 线程解释器                                    ThOp[] 数组
  op_barrier → return 0x10000|pc                             goto *ip++->handler
  op_ebreak  → SIMT 栈弹栈                                    vsetvli → rebind

