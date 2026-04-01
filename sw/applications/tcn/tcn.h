#pragma once
#ifndef TCN_H_
#define TCN_H_

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "ker_conv1d.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TCN_MAX_IN
#define TCN_MAX_IN 16
#endif

#ifndef TCN_MAX_CH
#define TCN_MAX_CH 128
#endif

#ifndef TCN_MAX_CLASSES
#define TCN_MAX_CLASSES 16
#endif

#ifndef TCN_MAX_BLOCKS
#define TCN_MAX_BLOCKS 12
#endif

#ifndef TCN_MAX_STATE_BYTES
#define TCN_MAX_STATE_BYTES (160 * 1024)
#endif

#if TCN_MAX_IN > TCN_MAX_CH
#define TCN_MAX_VEC8 TCN_MAX_IN
#else
#define TCN_MAX_VEC8 TCN_MAX_CH
#endif

#if TCN_MAX_CLASSES > TCN_MAX_CH
#define TCN_MAX_ACT TCN_MAX_CLASSES
#else
#define TCN_MAX_ACT TCN_MAX_CH
#endif

#define TCN_MAX_CONVS (2 * TCN_MAX_BLOCKS + 2)

typedef struct {
    int32_t m;
    int16_t r;
} tcn_scale_t;

typedef struct {
    uint16_t in_ch;
    uint16_t out_ch;
    uint16_t k;
    uint16_t dil;
    uint16_t ctx;
    int16_t in_r;
    int16_t out_r;
    int32_t in_m;                 // int16 -> int8 input requant for this layer
    const int8_t *w;              // [out_ch][k][in_ch], packed by tap
    const int32_t *b;             // [out_ch]
    const int32_t *m;             // [out_ch], conv requant to int16
} tcn_conv_t;

typedef struct {
    tcn_conv_t conv1;
    tcn_scale_t relu1;
    tcn_conv_t conv2;
    tcn_scale_t add_x;
    tcn_scale_t add_res;
} tcn_block_t;

typedef struct {
    uint16_t in_ch;
    uint16_t ch;
    uint16_t n_blocks;
    uint16_t n_classes;
    uint16_t warmup;
    tcn_conv_t stem;
    tcn_scale_t stem_relu;
    const tcn_block_t *blocks;
    tcn_conv_t head;
} tcn_model_t;

typedef struct {
    const tcn_model_t *m;
    conv1d_ring_t ring[TCN_MAX_CONVS];
    uint32_t hist_used;
    uint16_t n_conv;
    uint32_t n_samples;
    int8_t hist[TCN_MAX_STATE_BYTES];
    int8_t xq[TCN_MAX_VEC8];
    int16_t a[TCN_MAX_ACT];
    int16_t b[TCN_MAX_ACT];
    int16_t res[TCN_MAX_CH];
} tcn_t;

static inline uint16_t tcn_conv_ctx(const tcn_conv_t *c)
{
    if (!c) return 0;
    if (c->ctx) return c->ctx;
    return (uint16_t)((c->k - 1u) * c->dil);
}

static inline int tcn_check_conv(const tcn_conv_t *c)
{
    if (!c || !c->w || !c->m) return -1;
    if (c->in_ch <= 0 || c->out_ch <= 0 || c->k <= 0 || c->dil <= 0) return -1;
    return 0;
}

static inline int tcn_bind_ring(
    tcn_t *net,
    uint16_t idx,
    const tcn_conv_t *c)
{
    const uint16_t ctx = tcn_conv_ctx(c);
    const uint32_t len = (uint32_t)ctx + 1u;
    const uint32_t need = len * (uint32_t)c->in_ch;

    if (idx >= TCN_MAX_CONVS) return -1;
    if (net->hist_used + need > TCN_MAX_STATE_BYTES) return -1;

    net->ring[idx].buf = net->hist + net->hist_used;
    net->ring[idx].ch = c->in_ch;
    net->ring[idx].len = (int)len;
    net->ring[idx].head = (int)len - 1;
    net->hist_used += need;
    return 0;
}

static inline size_t tcn_state_bytes(const tcn_model_t *m)
{
    size_t bytes = 0;
    if (!m) return 0;

    bytes += ((size_t)tcn_conv_ctx(&m->stem) + 1u) * (size_t)m->stem.in_ch;
    for (uint16_t i = 0; i < m->n_blocks; ++i) {
        bytes += ((size_t)tcn_conv_ctx(&m->blocks[i].conv1) + 1u) * (size_t)m->blocks[i].conv1.in_ch;
        bytes += ((size_t)tcn_conv_ctx(&m->blocks[i].conv2) + 1u) * (size_t)m->blocks[i].conv2.in_ch;
    }
    bytes += ((size_t)tcn_conv_ctx(&m->head) + 1u) * (size_t)m->head.in_ch;
    return bytes;
}

static inline void tcn_reset(tcn_t *net)
{
    if (!net) return;
    net->n_samples = 0;
    for (uint16_t i = 0; i < net->n_conv; ++i) conv1d_ring_reset(&net->ring[i]);
}

static inline int tcn_init(tcn_t *net, const tcn_model_t *m)
{
    if (!net || !m) return -1;
    if (m->in_ch <= 0 || m->in_ch > TCN_MAX_IN) return -1;
    if (m->ch <= 0 || m->ch > TCN_MAX_CH) return -1;
    if (m->n_classes <= 0 || m->n_classes > TCN_MAX_CLASSES) return -1;
    if (m->n_blocks > TCN_MAX_BLOCKS) return -1;
    if (m->n_blocks > 0 && !m->blocks) return -1;
    if (tcn_check_conv(&m->stem) != 0 || tcn_check_conv(&m->head) != 0) return -1;

    for (uint16_t i = 0; i < m->n_blocks; ++i) {
        if (tcn_check_conv(&m->blocks[i].conv1) != 0) return -1;
        if (tcn_check_conv(&m->blocks[i].conv2) != 0) return -1;
    }

    memset(net, 0, sizeof(*net));
    net->m = m;
    net->n_conv = (uint16_t)(2u * m->n_blocks + 2u);

    if (tcn_bind_ring(net, 0, &m->stem) != 0) return -1;
    for (uint16_t i = 0; i < m->n_blocks; ++i) {
        if (tcn_bind_ring(net, (uint16_t)(1u + 2u * i), &m->blocks[i].conv1) != 0) return -1;
        if (tcn_bind_ring(net, (uint16_t)(2u + 2u * i), &m->blocks[i].conv2) != 0) return -1;
    }
    if (tcn_bind_ring(net, (uint16_t)(2u * m->n_blocks + 1u), &m->head) != 0) return -1;

    tcn_reset(net);
    return 0;
}

static inline int tcn_ready(const tcn_t *net)
{
    if (!net || !net->m) return 0;
    return net->n_samples > net->m->warmup;
}

static inline void tcn_conv_i8(
    tcn_t *net,
    uint16_t ring_id,
    const tcn_conv_t *c,
    const int8_t *x,
    int16_t *y)
{
    conv1d_causal_step_i8x8_i16(
        &net->ring[ring_id],
        x,
        c->w,
        c->b,
        c->m,
        c->out_ch,
        c->in_ch,
        c->k,
        c->dil,
        c->out_r,
        y);
}

static inline void tcn_conv_i16(
    tcn_t *net,
    uint16_t ring_id,
    const tcn_conv_t *c,
    const int16_t *x,
    int16_t *y)
{
    vec_scale_i16_to_i8(x, c->in_ch, c->in_m, c->in_r, net->xq);
    tcn_conv_i8(net, ring_id, c, net->xq, y);
}

static inline void tcn_relu(
    const int16_t *x,
    int n,
    const tcn_scale_t *q,
    int16_t *y)
{
    vec_scale_i16_to_i16(x, n, q->m, q->r, 1, y);
}

static inline void tcn_add_relu(
    const int16_t *x,
    const int16_t *res,
    int n,
    const tcn_scale_t *qx,
    const tcn_scale_t *qres,
    int16_t *y)
{
    vec_add_i16(x, res, n, qx->m, qx->r, qres->m, qres->r, 1, y);
}

static inline void tcn_step(
    tcn_t *net,
    const int8_t *x,
    int16_t *logits)
{
    const tcn_model_t *m;
    int16_t *cur;
    int16_t *tmp;
    uint16_t ring_id = 1;

    if (!net || !net->m || !x || !logits) return;
    m = net->m;

    tcn_conv_i8(net, 0, &m->stem, x, net->a);
    tcn_relu(net->a, m->ch, &m->stem_relu, net->b);
    cur = net->b;
    tmp = net->a;

    for (uint16_t i = 0; i < m->n_blocks; ++i) {
        const tcn_block_t *blk = &m->blocks[i];

        memcpy(net->res, cur, (size_t)m->ch * sizeof(int16_t));

        tcn_conv_i16(net, ring_id++, &blk->conv1, cur, tmp);
        tcn_relu(tmp, m->ch, &blk->relu1, cur);
        tcn_conv_i16(net, ring_id++, &blk->conv2, cur, tmp);
        tcn_add_relu(tmp, net->res, m->ch, &blk->add_x, &blk->add_res, cur);
    }

    tcn_conv_i16(net, ring_id, &m->head, cur, logits);
    net->n_samples++;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TCN_H_
