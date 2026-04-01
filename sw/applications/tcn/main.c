#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "build_config.h"

#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER 1
#endif

#ifndef FLASH_USE_QUAD
#define FLASH_USE_QUAD 0
#endif

#ifndef TCN_INPUT_TILE
#define TCN_INPUT_TILE 64
#endif

#ifndef TCN_WEIGHT_TILE_OUT
#define TCN_WEIGHT_TILE_OUT 32
#endif

#include "x-heep.h"
#include "w25q128jw.h"
#include "core_v_mini_mcu.h"

#if defined(__riscv)
#include "csr.h"
#endif

#include "model.h"
#include "profiler.h"
#include "test_data.h"

#ifndef MODEL_MAIN_USE_STDIO
#define MODEL_MAIN_USE_STDIO 1
#endif

#if MODEL_MAIN_USE_STDIO
#include <stdio.h>
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define PRINTF(...)
#endif

#if MODEL_IN_CH != TEST_DATA_IN_CH
#error "MODEL_IN_CH and TEST_DATA_IN_CH must match"
#endif

#if MODEL_CLASSES != TEST_DATA_NUM_CLASSES
#error "MODEL_CLASSES and TEST_DATA_NUM_CLASSES must match"
#endif

#if TCN_PROF_MAX_SECTIONS < (3 + 4 * MODEL_BLOCKS + 2)
#error "TCN_PROF_MAX_SECTIONS is too small for the current model"
#endif

#if defined(__GNUC__)
#define SCRATCH_ALIGN __attribute__((aligned(16)))
#else
#define SCRATCH_ALIGN
#endif

enum {
    PROF_INPUT_TILE = 0,
    PROF_STEM_CONV,
    PROF_STEM_RELU,
    PROF_BLOCK_BASE,
    PROF_BLOCK_STRIDE = 4,
    PROF_HEAD_CONV = PROF_BLOCK_BASE + MODEL_BLOCKS * PROF_BLOCK_STRIDE,
    PROF_RECORD_TOTAL,
    PROF_SECTION_COUNT
};

static tcn_t net;
static Profiler g_profiler;

static int8_t SCRATCH_ALIGN g_input_tile[TCN_INPUT_TILE][MODEL_IN_CH];
static int8_t SCRATCH_ALIGN g_weight_tile[TCN_WEIGHT_TILE_OUT * MODEL_CH * 5];
static int32_t SCRATCH_ALIGN g_bias_tile[TCN_WEIGHT_TILE_OUT];
static int32_t SCRATCH_ALIGN g_mul_tile[TCN_WEIGHT_TILE_OUT];

static inline int prof_block_conv1_id(int blk)
{
    return PROF_BLOCK_BASE + blk * PROF_BLOCK_STRIDE;
}

static inline int prof_block_relu1_id(int blk)
{
    return prof_block_conv1_id(blk) + 1;
}

static inline int prof_block_conv2_id(int blk)
{
    return prof_block_conv1_id(blk) + 2;
}

static inline int prof_block_add_id(int blk)
{
    return prof_block_conv1_id(blk) + 3;
}

#if ENABLE_PROFILER
#define PROF_BEGIN(id) profiler_begin(&g_profiler, (unsigned)(id))
#define PROF_END(id) profiler_end(&g_profiler, (unsigned)(id))
#else
#define PROF_BEGIN(id) ((void)(id))
#define PROF_END(id) ((void)(id))
#endif

static void prof_hw_init(void)
{
#if ENABLE_PROFILER && defined(__riscv)
    CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);
#endif
}

uint64_t prof_now(void)
{
#if ENABLE_PROFILER && defined(__riscv)
    uint32_t hi0, lo, hi1;

    do {
        CSR_READ(CSR_REG_MCYCLEH, &hi0);
        CSR_READ(CSR_REG_MCYCLE, &lo);
        CSR_READ(CSR_REG_MCYCLEH, &hi1);
    } while (hi0 != hi1);

    return ((uint64_t)hi1 << 32) | (uint64_t)lo;
#else
    return 0;
#endif
}

static int flash_read_bytes(const void *src_sym, void *dst_buf, size_t nbytes)
{
    if (!dst_buf || !src_sym) return -1;
    if (nbytes == 0u) return 0;

#ifdef FLASH_LOAD
    {
        const uint32_t flash_off =
            (uint32_t)(uintptr_t)heep_get_flash_address_offset(
                (uint32_t *)(uintptr_t)src_sym);
#if FLASH_USE_QUAD
        return (int)w25q128jw_read_quad(flash_off, dst_buf, (uint32_t)nbytes);
#else
        return (int)w25q128jw_read_standard(flash_off, dst_buf, (uint32_t)nbytes);
#endif
    }
#else
    memcpy(dst_buf, src_sym, nbytes);
    return 0;
#endif
}

static int app_flash_init(void)
{
#ifdef FLASH_LOAD
    soc_ctrl_t soc_ctrl;

    soc_ctrl.base_addr = mmio_region_from_addr((uintptr_t)SOC_CTRL_START_ADDRESS);

    if (get_spi_flash_mode(&soc_ctrl) == SOC_CTRL_SPI_FLASH_MODE_SPIMEMIO) {
        PRINTF("This application requires FLASH_LOAD, not FLASH_EXEC\n");
        return -1;
    }

    if (w25q128jw_init(spi_flash) != FLASH_OK) {
        PRINTF("Error initializing SPI flash\n");
        return -1;
    }
#endif

    return 0;
}

static void logits_to_pred(const int16_t *logits, uint8_t *pred)
{
    for (int i = 0; i < MODEL_CLASSES; ++i) {
        pred[i] = (logits[i] >= 0) ? 1u : 0u;
    }
}

static int conv1d_causal_step_i8x8_i16_flash_tiled(
    conv1d_ring_t *ring,
    const int8_t *x_now,
    const tcn_conv_t *c,
    int16_t *y)
{
    const int cout = c ? (int)c->out_ch : 0;
    const int cin = c ? (int)c->in_ch : 0;
    const int k = c ? (int)c->k : 0;
    const int dil = c ? (int)c->dil : 0;
    const size_t weights_per_out = (size_t)cin * (size_t)k;

    if (!ring || !x_now || !c || !y || !c->w || !c->m) return -1;
    if (cout <= 0 || cin <= 0 || k <= 0 || dil <= 0) return -1;
    if (!ring->buf || ring->ch != cin || ring->len <= 0) return -1;
    if ((size_t)TCN_WEIGHT_TILE_OUT * weights_per_out > sizeof(g_weight_tile)) {
        return -1;
    }

    conv1d_ring_push(ring, x_now);

    for (int o0 = 0; o0 < cout; o0 += TCN_WEIGHT_TILE_OUT) {
        const int oN = (o0 + TCN_WEIGHT_TILE_OUT <= cout)
                         ? TCN_WEIGHT_TILE_OUT
                         : (cout - o0);
        const size_t tile_bytes = (size_t)oN * weights_per_out;

        if (flash_read_bytes(c->w + (size_t)o0 * weights_per_out,
                             g_weight_tile,
                             tile_bytes) != 0) {
            return -1;
        }
        if (flash_read_bytes(c->m + o0, g_mul_tile, (size_t)oN * sizeof(int32_t)) != 0) {
            return -1;
        }

        if (c->b) {
            if (flash_read_bytes(c->b + o0, g_bias_tile, (size_t)oN * sizeof(int32_t)) != 0) {
                return -1;
            }
        } else {
            for (int o = 0; o < oN; ++o) g_bias_tile[o] = 0;
        }

        for (int o = 0; o < oN; ++o) {
            const int8_t *w_o = g_weight_tile + (size_t)o * weights_per_out;
            int32_t acc = g_bias_tile[o];

            for (int tap = 0; tap < k; ++tap) {
                const int8_t *x_k = conv1d_ring_at(ring, (k - 1 - tap) * dil);
                const int8_t *w_k = w_o + (size_t)tap * (size_t)cin;
                acc += xheep_dot_i8_i8_any(x_k, w_k, cin);
            }

            y[o0 + o] = requant_i16_scalar(acc, g_mul_tile[o], c->out_r);
        }
    }

    return 0;
}

static int tcn_conv_i8_flash_tiled(
    tcn_t *netp,
    uint16_t ring_id,
    const tcn_conv_t *c,
    const int8_t *x,
    int16_t *y)
{
    if (!netp || !c || !x || !y) return -1;
    return conv1d_causal_step_i8x8_i16_flash_tiled(&netp->ring[ring_id], x, c, y);
}

static int tcn_conv_i16_flash_tiled(
    tcn_t *netp,
    uint16_t ring_id,
    const tcn_conv_t *c,
    const int16_t *x,
    int16_t *y)
{
    if (!netp || !c || !x || !y) return -1;
    vec_scale_i16_to_i8(x, c->in_ch, c->in_m, c->in_r, netp->xq);
    return tcn_conv_i8_flash_tiled(netp, ring_id, c, netp->xq, y);
}

static int tcn_step_flash_profiled(
    tcn_t *netp,
    const int8_t *x,
    int16_t *logits)
{
    const tcn_model_t *m;
    int16_t *cur;
    int16_t *tmp;
    uint16_t ring_id = 1;
    int err;

    if (!netp || !netp->m || !x || !logits) return -1;

    m = netp->m;

    PRINTF("TCN step: n_samples=%lu\n", (unsigned long)netp->n_samples);
    PROF_BEGIN(PROF_STEM_CONV);
    err = tcn_conv_i8_flash_tiled(netp, 0, &m->stem, x, netp->a);
    PROF_END(PROF_STEM_CONV);
    if (err != 0) return err;

    PROF_BEGIN(PROF_STEM_RELU);
    tcn_relu(netp->a, m->ch, &m->stem_relu, netp->b);
    PROF_END(PROF_STEM_RELU);

    cur = netp->b;
    tmp = netp->a;

    for (uint16_t i = 0; i < m->n_blocks; ++i) {
        const tcn_block_t *blk = &m->blocks[i];

        memcpy(netp->res, cur, (size_t)m->ch * sizeof(int16_t));

        PROF_BEGIN(prof_block_conv1_id((int)i));
        err = tcn_conv_i16_flash_tiled(netp, ring_id++, &blk->conv1, cur, tmp);
        PROF_END(prof_block_conv1_id((int)i));
        if (err != 0) return err;

        PROF_BEGIN(prof_block_relu1_id((int)i));
        tcn_relu(tmp, m->ch, &blk->relu1, cur);
        PROF_END(prof_block_relu1_id((int)i));

        PROF_BEGIN(prof_block_conv2_id((int)i));
        err = tcn_conv_i16_flash_tiled(netp, ring_id++, &blk->conv2, cur, tmp);
        PROF_END(prof_block_conv2_id((int)i));
        if (err != 0) return err;

        PROF_BEGIN(prof_block_add_id((int)i));
        tcn_add_relu(tmp, netp->res, m->ch, &blk->add_x, &blk->add_res, cur);
        PROF_END(prof_block_add_id((int)i));
    }

    PROF_BEGIN(PROF_HEAD_CONV);
    err = tcn_conv_i16_flash_tiled(netp, ring_id, &m->head, cur, logits);
    PROF_END(PROF_HEAD_CONV);
    if (err != 0) return err;

    netp->n_samples++;
    return 0;
}

static int run_record(int rec, int16_t *logits_out)
{
    if (tcn_init(&net, &model) != 0) return -1;

    PROF_BEGIN(PROF_RECORD_TOTAL);
    for (int t0 = 0; t0 < TEST_DATA_SEQ_LEN; t0 += TCN_INPUT_TILE) {
        const int tile_len =
            (t0 + TCN_INPUT_TILE <= TEST_DATA_SEQ_LEN)
                ? TCN_INPUT_TILE
                : (TEST_DATA_SEQ_LEN - t0);
        int err;

        PROF_BEGIN(PROF_INPUT_TILE);
        err = flash_read_bytes(&test_data_xq[rec][t0][0],
                               g_input_tile,
                               (size_t)tile_len * MODEL_IN_CH * sizeof(int8_t));
        PROF_END(PROF_INPUT_TILE);
        if (err != 0) {
            PROF_END(PROF_RECORD_TOTAL);
            return -2;
        }

        for (int t = 0; t < tile_len; ++t) {
            if (tcn_step_flash_profiled(&net, g_input_tile[t], logits_out) != 0) {
                PROF_END(PROF_RECORD_TOTAL);
                return -3;
            }
        }
    }
    PROF_END(PROF_RECORD_TOTAL);

    return 0;
}

static double class_f1(int tp, int fp, int fn)
{
    const int den_p = tp + fp;
    const int den_r = tp + fn;
    double p = 0.0;
    double r = 0.0;

    if (den_p > 0) p = (double)tp / (double)den_p;
    if (den_r > 0) r = (double)tp / (double)den_r;
    if ((p + r) <= 0.0) return 0.0;
    return 2.0 * p * r / (p + r);
}

static double macro_f1_score(const int *tp, const int *fp, const int *fn)
{
    double macro_f1 = 0.0;

    for (int i = 0; i < MODEL_CLASSES; ++i) {
        macro_f1 += class_f1(tp[i], fp[i], fn[i]);
    }

    return macro_f1 / (double)MODEL_CLASSES;
}

#if MODEL_MAIN_USE_STDIO
static void dump_prof_entry(const char *name, const ProfEntry *entry)
{
    const uint32_t total_lo = (uint32_t)entry->total_ticks;
    const uint32_t avg_lo =
        (entry->calls > 0u) ? (uint32_t)(entry->total_ticks / entry->calls) : 0u;

    PRINTF("profile %-14s calls=%lu total_cycles=%lu avg=%lu\n",
           name,
           (unsigned long)entry->calls,
           (unsigned long)total_lo,
           (unsigned long)avg_lo);
}

static void dump_profiler_stats(void)
{
    dump_prof_entry("input_tile", &g_profiler.sections[PROF_INPUT_TILE]);
    dump_prof_entry("stem.conv", &g_profiler.sections[PROF_STEM_CONV]);
    dump_prof_entry("stem.relu", &g_profiler.sections[PROF_STEM_RELU]);

    for (int blk = 0; blk < MODEL_BLOCKS; ++blk) {
        char name[24];

        snprintf(name, sizeof(name), "blk%d.conv1", blk);
        dump_prof_entry(name, &g_profiler.sections[prof_block_conv1_id(blk)]);

        snprintf(name, sizeof(name), "blk%d.relu1", blk);
        dump_prof_entry(name, &g_profiler.sections[prof_block_relu1_id(blk)]);

        snprintf(name, sizeof(name), "blk%d.conv2", blk);
        dump_prof_entry(name, &g_profiler.sections[prof_block_conv2_id(blk)]);

        snprintf(name, sizeof(name), "blk%d.add", blk);
        dump_prof_entry(name, &g_profiler.sections[prof_block_add_id(blk)]);
    }

    dump_prof_entry("head.conv", &g_profiler.sections[PROF_HEAD_CONV]);
    dump_prof_entry("record.total", &g_profiler.sections[PROF_RECORD_TOTAL]);
}
#endif

int main(void)
{
    int16_t logits[MODEL_CLASSES];
    uint8_t pred[MODEL_CLASSES];
    int short_context = 0;
    int ran = 0;
    int tp[MODEL_CLASSES] = {0};
    int fp[MODEL_CLASSES] = {0};
    int fn[MODEL_CLASSES] = {0};

    if (app_flash_init() != 0) return EXIT_FAILURE;

    prof_hw_init();
    profiler_reset(&g_profiler, PROF_SECTION_COUNT);

    for (int rec = 0; rec < TEST_DATA_NUM_RECORDS; ++rec) {
        if (run_record(rec, logits) != 0) {
            PRINTF("Error running record %u\n", (unsigned)test_data_record_ids[rec]);
            return EXIT_FAILURE;
        }

        if (!tcn_ready(&net)) short_context++;
        logits_to_pred(logits, pred);

        for (int i = 0; i < MODEL_CLASSES; ++i) {
            const int y = test_data_labels[rec][i] ? 1 : 0;
            const int p = pred[i] ? 1 : 0;

            if (p && y) tp[i]++;
            else if (p && !y) fp[i]++;
            else if (!p && y) fn[i]++;
        }

        ran++;
    }

#if MODEL_MAIN_USE_STDIO
    if (ran == 0) {
        PRINTF("No test records\n");
        return EXIT_FAILURE;
    }

    {
        const double macro_f1 = macro_f1_score(tp, fp, fn);

        PRINTF("summary records=%d", ran);
        if (short_context > 0) {
            PRINTF(" short_context=%d", short_context);
        }
        PRINTF("\n");
        PRINTF("summary metric=macro_f1 threshold=logit>=0 value=%.4f\n", macro_f1);
        dump_profiler_stats();
    }
#else
    (void)ran;
#endif

    return EXIT_SUCCESS;
}
