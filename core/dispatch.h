/*
 * dispatch.h — dispatch 表 + trie 基础设施
 *
 * INSTRUCTION_LIST 由 inst/gen_dispatch.py 从 ISA spec 聚合生成。
 */
#ifndef SIMD_DISPATCH_H
#define SIMD_DISPATCH_H

#include "inst.h"
#include "utils.h"
#include "decode_trie.h"
#include "state.h"

/* === 指令类型 === */
typedef enum {
    TYPE_R, TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_B,
    TYPE_CSR, TYPE_FR, TYPE_FI, TYPE_FS, TYPE_F4
} inst_type_t;

/* === 生成数据: INSTRUCTION_LIST + DISP_ enum + NUM_OF_INST === */
#include "../inst/dispatch_list.h"

/* === 生成: rvv_dispatch_rebind() === */
#include "../inst/dispatch_rebind.h"

static inline int32_t imm0(uint32_t i)
{
    (void)i;
    return 0;
}

/* === Dispatch 表 + trie === */
#define SIMD_NUM_INST NUM_OF_INST

typedef struct {
    void *dispatch[SIMD_NUM_INST];
    int dispatch_ready;
    DecodeTrie trie;
    opcode_entry_t op_table[SIMD_NUM_INST];
    size_t op_count;
} SIMDDecoder;

static inline __attribute__((unused)) void simd_decoder_init(SIMDDecoder *d)
{
    if (d->op_count > 0) return;
    int idx = 0;
#define X(name, pattern, op_type, imm_fn)                                 \
    d->op_table[idx].mask = pattern_to_mask(pattern),                     \
    d->op_table[idx].match = pattern_to_match(pattern),                   \
    d->op_table[idx].exec = NULL, d->op_table[idx].type = op_type, idx++
    { INSTRUCTION_LIST; }
#undef X
    d->op_count = idx;
    decode_trie_build(&d->trie, d->op_table, d->op_count);
}

#endif /* SIMD_DISPATCH_H */
